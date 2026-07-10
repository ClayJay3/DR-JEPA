package com.mrdt.drjepa

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.google.ar.core.Camera
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Real-world data collection for sim-to-real fine-tuning.
 *
 * Per captured frame (throttled by ArCam to ~5 Hz) this stores everything
 * the offline pipeline (real2dataset.py) needs to build the same wedge
 * training labels the simulator emits:
 *  - frames/NNNNN.jpg   full CPU camera image, SENSOR orientation (the
 *                       recorded intrinsics match this orientation; the
 *                       converter uprights it)
 *  - depth/NNNNN.pgm    ARCore depth, 16-bit P5 PGM (big-endian per spec),
 *                       millimetres, 0 = no data; same FOV as the frame
 *  - meta.jsonl         one JSON object per frame: timestamp, physical
 *                       camera pose (world, quaternion + translation),
 *                       CPU-image intrinsics, dims, VIO speed, heading
 *                       (NaN until north-aligned)
 *  - session.json       device / geometry constants, written on close
 *
 * Files go to the app-scoped external dir (no permissions needed):
 *   Android/data/com.mrdt.drjepa/files/collect/rec_YYYYMMDD_HHMMSS/
 * Pull with: adb pull /sdcard/Android/data/com.mrdt.drjepa/files/collect
 *
 * Writing runs on a single worker thread; frames are DROPPED (not queued
 * unboundedly) if encoding falls behind, so the GL thread never blocks.
 */
class Recorder(
    context: Context,
    private val onCount: (Int) -> Unit,
) {
    val dir: File = File(
        File(context.getExternalFilesDir(null), "collect"),
        "rec_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
            .format(Date()))
    private val framesDir = File(dir, "frames").apply { mkdirs() }
    private val depthDir = File(dir, "depth").apply { mkdirs() }
    private val meta = File(dir, "meta.jsonl").bufferedWriter()

    private val worker = Executors.newSingleThreadExecutor()
    private val pending = AtomicInteger(0)
    private val count = AtomicInteger(0)
    @Volatile private var closed = false
    private var sessionWritten = false

    /** Grabbed on the GL thread (fast copies only), written on the worker. */
    private class Grab(
        val idx: Int, val tNs: Long,
        // camera YUV planes (sensor orientation)
        val w: Int, val h: Int,
        val y: ByteArray, val u: ByteArray, val v: ByteArray,
        val yStride: Int, val uvStride: Int, val uvPixStride: Int,
        // depth (millimetres)
        val dw: Int, val dh: Int, val depth: ShortArray,
        // physical camera pose (ARCore world): tx ty tz qx qy qz qw
        val pose: FloatArray,
        // CPU-image intrinsics fx fy cx cy
        val intr: FloatArray,
        val speed: Float, val headingDeg: Float,
        val localE: Float, val localN: Float,
        // rotation (deg CW) that makes the sensor image upright on screen;
        // the converter applies it so training frames match the pilot feed
        val rot: Int,
    )

    /**
     * Copy what this frame needs and queue the write. Called on the GL
     * thread; both images may be closed as soon as this returns.
     * Returns false if the frame was dropped (writer busy).
     */
    fun submit(camImage: Image, depthImage: Image, cam: Camera,
               speed: Float, headingDeg: Float,
               localE: Float, localN: Float, uprightRot: Int): Boolean {
        if (closed || pending.get() >= 2) return false

        val py = camImage.planes[0]
        val pu = camImage.planes[1]
        val pv = camImage.planes[2]
        fun copy(p: Image.Plane): ByteArray {
            val b = p.buffer.duplicate(); b.rewind()
            return ByteArray(b.remaining()).also { b.get(it) }
        }
        val dPlane = depthImage.planes[0]
        val dBuf = dPlane.buffer.duplicate().order(
            java.nio.ByteOrder.nativeOrder())
        dBuf.rewind()
        val dw = depthImage.width
        val dh = depthImage.height
        val rowShorts = dPlane.rowStride / 2
        val dAll = ShortArray(dBuf.remaining() / 2)
        dBuf.asShortBuffer().get(dAll)
        // strip row padding now so the file is dense dw x dh
        val depth = if (rowShorts == dw) dAll else ShortArray(dw * dh).also {
            for (r in 0 until dh)
                System.arraycopy(dAll, r * rowShorts, it, r * dw, dw)
        }

        val p = cam.pose
        val q = p.rotationQuaternion
        val intr = cam.imageIntrinsics
        val g = Grab(
            count.get(), camImage.timestamp,
            camImage.width, camImage.height,
            copy(py), copy(pu), copy(pv),
            py.rowStride, pu.rowStride, pu.pixelStride,
            dw, dh, depth,
            floatArrayOf(p.tx(), p.ty(), p.tz(), q[0], q[1], q[2], q[3]),
            floatArrayOf(intr.focalLength[0], intr.focalLength[1],
                         intr.principalPoint[0], intr.principalPoint[1]),
            speed, headingDeg, localE, localN, uprightRot)

        pending.incrementAndGet()
        worker.execute {
            try {
                write(g)
            } finally {
                pending.decrementAndGet()
            }
        }
        count.incrementAndGet()
        return true
    }

    private fun write(g: Grab) {
        // no `closed` check: writes queued before close() must complete;
        // the meta writer is flushed after them (same single worker)
        if (!sessionWritten) {
            sessionWritten = true
            File(dir, "session.json").writeText(JSONObject().apply {
                put("device", android.os.Build.MODEL)
                put("cam_height_m", ArCam.CAM_HEIGHT_M)
                put("cpu_w", g.w); put("cpu_h", g.h)
                put("depth_w", g.dw); put("depth_h", g.dh)
                put("fx", g.intr[0]); put("fy", g.intr[1])
                put("cx", g.intr[2]); put("cy", g.intr[3])
                put("note", "intrinsics + pose are SENSOR-oriented; " +
                    "pose = ARCore physical camera (x right, y up, " +
                    "-z look); depth mm along optical axis, 0 = invalid")
            }.toString(2))
        }

        // ---- JPEG (NV21 assembled with stride handling) ----
        val nv21 = ByteArray(g.w * g.h * 3 / 2)
        for (r in 0 until g.h)
            System.arraycopy(g.y, r * g.yStride, nv21, r * g.w, g.w)
        var o = g.w * g.h
        val ch = g.h / 2
        val cw = g.w / 2
        for (r in 0 until ch) {
            val row = r * g.uvStride
            for (c in 0 until cw) {
                nv21[o++] = g.v[row + c * g.uvPixStride]
                nv21[o++] = g.u[row + c * g.uvPixStride]
            }
        }
        val name = "%05d".format(g.idx)
        FileOutputStream(File(framesDir, "$name.jpg")).use {
            YuvImage(nv21, ImageFormat.NV21, g.w, g.h, null)
                .compressToJpeg(Rect(0, 0, g.w, g.h), 90, it)
        }

        // ---- depth: 16-bit binary PGM (big-endian per spec) ----
        BufferedOutputStream(
            FileOutputStream(File(depthDir, "$name.pgm"))).use { s ->
            s.write("P5\n${g.dw} ${g.dh}\n65535\n".toByteArray())
            val b = ByteArray(g.depth.size * 2)
            for (i in g.depth.indices) {
                val v = g.depth[i].toInt() and 0xFFFF
                b[2 * i] = (v shr 8).toByte()
                b[2 * i + 1] = v.toByte()
            }
            s.write(b)
        }

        // ---- per-frame metadata ----
        // org.json rejects NaN (heading/EN stay NaN until north-aligned)
        fun JSONObject.putSafe(k: String, v: Float) =
            put(k, if (v.isNaN()) JSONObject.NULL else v)
        synchronized(meta) {
            meta.write(JSONObject().apply {
                put("i", g.idx); put("t_ns", g.tNs)
                put("pose", org.json.JSONArray(g.pose.map { it.toDouble() }))
                put("fx", g.intr[0]); put("fy", g.intr[1])
                put("cx", g.intr[2]); put("cy", g.intr[3])
                put("speed", g.speed)
                putSafe("heading", g.headingDeg)
                putSafe("local_e", g.localE); putSafe("local_n", g.localN)
                put("rot", g.rot)
            }.toString())
            meta.newLine()
        }
        onCount(g.idx + 1)
    }

    /** Stop accepting frames, flush metadata after in-flight writes. */
    fun close() {
        closed = true
        worker.execute { synchronized(meta) { meta.flush(); meta.close() } }
        worker.shutdown()
    }

    val frames: Int get() = count.get()
}
