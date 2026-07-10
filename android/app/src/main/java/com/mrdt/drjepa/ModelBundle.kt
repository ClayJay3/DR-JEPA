package com.mrdt.drjepa

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.util.zip.ZipInputStream

/** MapPilot fusion / planning constants, exported into the bundle manifest
 *  by export_android.py so the app never hardcodes numbers that could
 *  drift from the Python pilot. */
class PilotConsts(j: JSONObject) {
    val dt = j.getDouble("DT").toFloat()                    // sim control period
    val gpsGain = j.getDouble("GPS_GAIN").toFloat()
    val loddsClamp = j.getDouble("LODDS_CLAMP").toFloat()
    val occThresh = j.getDouble("OCC_THRESH").toFloat()
    val priorLogit = j.getDouble("PRIOR_LOGIT").toFloat()
    val hazPriorLogit = j.getDouble("HAZ_PRIOR_LOGIT").toFloat()
    val sandPrior = j.getDouble("SAND_PRIOR").toFloat()
    val guardEvidence = j.getDouble("GUARD_EVIDENCE").toFloat()
    val replanEvery = j.getInt("REPLAN_EVERY")
    val nArcs = j.getInt("N_ARCS")
    val arcT = j.getDouble("ARC_T").toFloat()
    val arcDt = j.getDouble("ARC_DT").toFloat()
    val paintRangeCells = j.getInt("PAINT_RANGE_CELLS")
    val posEvidenceScale = j.getDouble("POS_EVIDENCE_SCALE").toFloat()
    val decay = j.getDouble("DECAY").toFloat()
    val voWindow = j.getInt("VO_WINDOW")
    val voGain = j.getDouble("VO_GAIN").toFloat()
    val voMargin = j.getDouble("VO_MARGIN").toFloat()
    val voMinL = j.getDouble("VO_MIN_L").toFloat()
}

/** One decoder forward: the five-channel terrain wedge + danger logit.
 *  Arrays are wedge_cells^2, flat-indexed [i * cells + j] with i = x-right,
 *  j = z-forward in the rover frame (same convention as drjepa.model). */
class Wedge(
    val occ: FloatArray, val conf: FloatArray, val elev: FloatArray,
    val sand: FloatArray, val haz: FloatArray, val dangerLogit: Float,
)

/**
 * A `.drjepa` bundle (zip of manifest.json + three ONNX graphs) unpacked
 * from device storage. Holds the ORT sessions and the manifest geometry.
 */
class ModelBundle private constructor(
    val name: String,
    val imgSize: Int,
    val featDim: Int,
    val nTokens: Int,
    val frameOffsets: IntArray,
    val wedgeCells: Int,
    val wedgeRes: Float,
    val wedgeRangeCells: Int,
    val mapCells: Int,
    val compCells: Int,
    val compRes: Float,
    val speedNorm: Float,
    val simFovDeg: Float,
    val roverRadius: Float,
    val pilot: PilotConsts,
    private val env: OrtEnvironment,
    private val backbone: OrtSession,
    private val decoder: OrtSession,
    private val completer: OrtSession,
) {
    companion object {
        /** Copy the picked document into cache, unzip, and load sessions. */
        fun load(context: Context, uri: Uri): ModelBundle {
            val dir = File(context.cacheDir, "bundle")
            dir.deleteRecursively()
            dir.mkdirs()
            context.contentResolver.openInputStream(uri).use { raw ->
                ZipInputStream(raw!!.buffered()).use { zip ->
                    var e = zip.nextEntry
                    while (e != null) {
                        val f = File(dir, File(e.name).name)  // flatten, no traversal
                        f.outputStream().use { zip.copyTo(it) }
                        e = zip.nextEntry
                    }
                }
            }
            val m = JSONObject(File(dir, "manifest.json").readText())
            require(m.getString("format") == "drjepa-android-v1") {
                "not a .drjepa bundle (format=${m.optString("format")})"
            }
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors()
                    .coerceIn(2, 6))
            }

            fun sess(key: String) = env.createSession(
                File(dir, m.getJSONObject("files").getString(key)).absolutePath,
                opts)

            val offs = m.getJSONArray("frame_offsets")
            return ModelBundle(
                name = m.getString("name"),
                imgSize = m.getInt("img_size"),
                featDim = m.getInt("feat_dim"),
                nTokens = m.getInt("n_tokens"),
                frameOffsets = IntArray(offs.length()) { offs.getInt(it) },
                wedgeCells = m.getInt("wedge_cells"),
                wedgeRes = m.getDouble("wedge_res").toFloat(),
                wedgeRangeCells = m.getInt("wedge_range_cells"),
                mapCells = m.getInt("map_cells"),
                compCells = m.getInt("comp_cells"),
                compRes = m.getDouble("comp_res").toFloat(),
                speedNorm = m.getDouble("speed_norm").toFloat(),
                simFovDeg = m.getDouble("sim_fov_deg").toFloat(),
                roverRadius = m.getDouble("rover_radius").toFloat(),
                pilot = PilotConsts(m.getJSONObject("pilot")),
                env = env,
                backbone = sess("backbone"),
                decoder = sess("decoder"),
                completer = sess("completer"),
            )
        }
    }

    private fun tensor(data: FloatArray, vararg shape: Long): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)

    private fun grab(v: Any?): FloatArray {
        val t = v as OnnxTensor
        val buf = t.floatBuffer
        val out = FloatArray(buf.remaining())
        buf.get(out)
        return out
    }

    /** RGB image, CHW float [0,1], imgSize^2 -> (nTokens * featDim) tokens. */
    fun runBackbone(chw: FloatArray): FloatArray =
        tensor(chw, 1, 3, imgSize.toLong(), imgSize.toLong()).use { img ->
            backbone.run(mapOf("image" to img)).use { grab(it.get(0)) }
        }

    /** Stacked tokens (F * nTokens * featDim) + motion (2F) -> wedge. */
    fun runDecoder(tokens: FloatArray, motion: FloatArray): Wedge {
        val f = frameOffsets.size.toLong()
        return tensor(tokens, 1, f, nTokens.toLong(), featDim.toLong()).use { t ->
            tensor(motion, 1, 2 * f).use { mo ->
                decoder.run(mapOf("tokens" to t, "motion" to mo)).use { r ->
                    Wedge(grab(r.get(0)), grab(r.get(1)), grab(r.get(2)),
                        grab(r.get(3)), grab(r.get(4)), grab(r.get(5))[0])
                }
            }
        }
    }

    /** Partial belief (4 * G * G) -> predicted occ/haz/sand probs (3 * G * G).
     *  Temperature calibration and sigmoid are baked into the graph. */
    fun runCompleter(belief: FloatArray): FloatArray {
        val g = compCells.toLong()
        return tensor(belief, 1, 4, g, g).use { b ->
            completer.run(mapOf("belief" to b)).use { grab(it.get(0)) }
        }
    }

    fun close() {
        backbone.close()
        decoder.close()
        completer.close()
    }
}
