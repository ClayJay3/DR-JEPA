package com.mrdt.drjepa

import android.app.Activity
import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import com.google.ar.core.Config
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/** Frozen snapshot of the ARCore-world -> local-EN alignment.
 *  Local frame: x = east, z = north, metres, origin at the session anchor.
 *  ARCore planar frame: e' = arcore x - anchorX, n' = -(arcore z - anchorZ);
 *  a bearing th' in it is true bearing th' + offsetRad. */
class ArAlign(
    val offsetRad: Float,
    val anchorX: Float, val anchorZ: Float,
    /** ARCore world y of the ground at the anchor (camera y - CAM_HEIGHT). */
    val groundY0: Float,
) {
    fun toLocalE(x: Float, z: Float): Float {
        val ep = x - anchorX; val np = -(z - anchorZ)
        return cos(offsetRad) * ep + sin(offsetRad) * np
    }
    fun toLocalN(x: Float, z: Float): Float {
        val ep = x - anchorX; val np = -(z - anchorZ)
        return -sin(offsetRad) * ep + cos(offsetRad) * np
    }
}

/**
 * ARCore camera + VIO. Owns the camera (that is ARCore's requirement),
 * renders the feed as a GL background, and per tracked frame produces:
 *  - metric pose + true-north heading in the local EN frame (compass is
 *    used only during the first seconds to align ARCore's arbitrary
 *    world yaw to north, then the offset is frozen -- a frozen small
 *    error is a constant map rotation, which the pilot never notices,
 *    whereas a drifting one would smear the belief map);
 *  - speed from VIO pose deltas (GPS speed is useless at walking pace);
 *  - view*projection matrices so the AR overlay is pixel-accurate;
 *  - CPU camera frames handed to the pilot worker whenever it is idle.
 */
class ArCam(
    private val activity: Activity,
    private val glView: GLSurfaceView,
    private val sensors: SensorHub,
    /** Called on the single pilot worker thread. */
    private val onPilotFrame: (FloatArray, SensorSample) -> Unit,
) : GLSurfaceView.Renderer {

    companion object {
        const val CAM_HEIGHT_M = 1.4f      // phone above ground while walking
        private const val ALIGN_SAMPLES = 120  // ~4 s of compass agreement
        private const val REC_PERIOD_NS = 200_000_000L  // record at ~5 Hz
    }

    private var lastRecNs = 0L

    var session: Session? = null; private set
    @Volatile var imgSize = 0               // 0 until a bundle is loaded
    @Volatile var recorder: Recorder? = null // non-null = collecting data
    @Volatile var depthSupported = false; private set
    @Volatile var tracking = false; private set
    @Volatile var alignLocked = false; private set
    @Volatile var trackingMsg = "VIO initializing"; private set
    @Volatile var viewProj: FloatArray? = null; private set
    @Volatile var align: ArAlign? = null; private set
    @Volatile var speedMps = 0f; private set
    var onUiUpdate: (() -> Unit)? = null    // invoked on the GL thread

    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var busy = false
    private var converter: YuvConverter? = null

    /** The worker is single-threaded: work queued here runs after any
     *  in-flight pilot step (e.g. closing a superseded model bundle). */
    fun runOnWorker(r: Runnable) {
        worker.execute(r)
    }

    // north alignment (circular mean of compass - VIO yaw)
    private var alignN = 0
    private var alignVx = 0.0
    private var alignVy = 0.0
    private var anchored = false
    private var anchorX = 0f
    private var anchorY = 0f
    private var anchorZ = 0f

    // speed EMA from pose deltas
    private var lastPx = 0f; private var lastPy = 0f; private var lastPz = 0f
    private var lastPoseT = 0L

    private var sensorOrientation = 90
    private var viewportW = 1
    private var viewportH = 1
    private var geomDirty = true
    private var lastRotation = -1

    // GL background
    private var texId = -1
    private var program = 0
    private var aPos = 0
    private var aTex = 0
    private val quadNdc = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
    private val texCoords = FloatArray(8)
    private lateinit var posBuf: FloatBuffer
    private lateinit var texBuf: FloatBuffer

    // ---------------- lifecycle (activity thread) ----------------
    /** Create + configure the session; throws if ARCore can't. */
    fun createSession() {
        val s = Session(activity)
        depthSupported = s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        val config = Config(s).apply {
            planeFindingMode = Config.PlaneFindingMode.DISABLED
            lightEstimationMode = Config.LightEstimationMode.DISABLED
            // depth feeds the data-collection mode (Recorder); the pilot
            // itself never reads it
            depthMode = if (depthSupported) Config.DepthMode.AUTOMATIC
                        else Config.DepthMode.DISABLED
            focusMode = Config.FocusMode.AUTO
            updateMode = Config.UpdateMode.BLOCKING
        }
        s.configure(config)
        // smallest CPU image is plenty (the model input is <= 448px)
        val configs = s.getSupportedCameraConfigs(CameraConfigFilter(s))
        s.cameraConfig = configs.minByOrNull {
            it.imageSize.width * it.imageSize.height
        } ?: configs[0]
        val cm = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        sensorOrientation = cm.getCameraCharacteristics(s.cameraConfig.cameraId)
            .get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        session = s
    }

    fun resume() {
        session?.resume()
        glView.onResume()
    }

    fun pause() {
        glView.onPause()
        session?.pause()
    }

    // ---------------- GL ----------------
    override fun onSurfaceCreated(gl: javax.microedition.khronos.opengles.GL10?,
                                  cfg: javax.microedition.khronos.egl.EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        texId = tex[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        val vs = """
            attribute vec2 aPos; attribute vec2 aTex; varying vec2 vTex;
            void main() { gl_Position = vec4(aPos, 0.0, 1.0); vTex = aTex; }
        """
        val fs = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            uniform samplerExternalOES uTex; varying vec2 vTex;
            void main() { gl_FragColor = texture2D(uTex, vTex); }
        """
        fun shader(type: Int, src: String): Int {
            val id = GLES20.glCreateShader(type)
            GLES20.glShaderSource(id, src)
            GLES20.glCompileShader(id)
            return id
        }
        program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, shader(GLES20.GL_VERTEX_SHADER, vs))
        GLES20.glAttachShader(program, shader(GLES20.GL_FRAGMENT_SHADER, fs))
        GLES20.glLinkProgram(program)
        aPos = GLES20.glGetAttribLocation(program, "aPos")
        aTex = GLES20.glGetAttribLocation(program, "aTex")
        posBuf = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
            .asFloatBuffer().put(quadNdc).also { it.rewind() } as FloatBuffer
        texBuf = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
            .asFloatBuffer()
    }

    override fun onSurfaceChanged(gl: javax.microedition.khronos.opengles.GL10?,
                                  width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewportW = width
        viewportH = height
        geomDirty = true
    }

    override fun onDrawFrame(gl: javax.microedition.khronos.opengles.GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val s = session ?: return
        val rotation = sensors.displayRotation
        if (geomDirty || rotation != lastRotation) {
            s.setDisplayGeometry(rotation, viewportW, viewportH)
            geomDirty = false
            lastRotation = rotation
        }
        s.setCameraTextureName(texId)
        val frame = try {
            s.update()
        } catch (_: CameraNotAvailableException) {
            return
        }
        if (frame.hasDisplayGeometryChanged()) {
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quadNdc,
                Coordinates2d.TEXTURE_NORMALIZED, texCoords)
            texBuf.rewind(); texBuf.put(texCoords); texBuf.rewind()
        }
        drawBackground()

        val cam = frame.camera
        if (cam.trackingState != TrackingState.TRACKING) {
            tracking = false
            trackingMsg = "VIO ${cam.trackingFailureReason.name
                .lowercase().replace('_', ' ')}"
            onUiUpdate?.invoke()
            return
        }
        tracking = true
        trackingMsg = "tracking"

        val pose = cam.displayOrientedPose
        val look = pose.rotateVector(floatArrayOf(0f, 0f, -1f))
        val yawPlanar = atan2(look[0], -look[2])   // bearing in ARCore planar

        // --- north alignment: converge for ~4 s of tracked frames, freeze.
        // Waits for a GPS fix so the declination is known and the offset
        // locks to TRUE north, not magnetic ---
        if (alignN < ALIGN_SAMPLES && sensors.hasOrientation && sensors.hasFix) {
            if (!anchored) {
                anchored = true
                anchorX = pose.tx(); anchorY = pose.ty(); anchorZ = pose.tz()
            }
            val inst = Math.toRadians(sensors.headingDeg.toDouble()) - yawPlanar
            alignVx += cos(inst)
            alignVy += sin(inst)
            alignN++
            align = ArAlign(atan2(alignVy, alignVx).toFloat(),
                anchorX, anchorZ, anchorY - CAM_HEIGHT_M)
            if (alignN >= ALIGN_SAMPLES) alignLocked = true
        }

        // --- data collection: needs tracking only (no model, no north
        // alignment, no GPS) so recording works anywhere immediately ---
        val rec = recorder
        if (rec != null && frame.timestamp - lastRecNs >= REC_PERIOD_NS) {
            var camImg: android.media.Image? = null
            var depImg: android.media.Image? = null
            try {
                camImg = frame.acquireCameraImage()
                depImg = frame.acquireDepthImage16Bits()
                val alNow = align
                val heading = if (alNow != null) (Math.toDegrees(
                    (yawPlanar + alNow.offsetRad).toDouble()).toFloat()
                    + 360f) % 360f else Float.NaN
                val upRot = (sensorOrientation -
                    rotationDegrees(rotation) + 360) % 360
                if (rec.submit(camImg, depImg, cam, speedMps, heading,
                        alNow?.toLocalE(pose.tx(), pose.tz()) ?: Float.NaN,
                        alNow?.toLocalN(pose.tx(), pose.tz()) ?: Float.NaN,
                        upRot))
                    lastRecNs = frame.timestamp
            } catch (_: NotYetAvailableException) {
                // depth needs a few seconds of parallax; retry next frame
            } finally {
                camImg?.close()
                depImg?.close()
            }
        }

        val al = align ?: run { onUiUpdate?.invoke(); return }

        // --- overlay matrices ---
        val vm = FloatArray(16)
        val pm = FloatArray(16)
        val vp = FloatArray(16)
        cam.getViewMatrix(vm, 0)
        cam.getProjectionMatrix(pm, 0, 0.1f, 200f)
        Matrix.multiplyMM(vp, 0, pm, 0, vm, 0)
        viewProj = vp

        // --- speed from VIO pose deltas ---
        val t = frame.timestamp
        if (lastPoseT != 0L) {
            val dt = (t - lastPoseT) / 1e9f
            if (dt > 1e-3f) {
                val v = hypot(pose.tx() - lastPx, pose.tz() - lastPz) / dt
                speedMps += 0.4f * (v - speedMps)
            }
        }
        lastPx = pose.tx(); lastPy = pose.ty(); lastPz = pose.tz()
        lastPoseT = t

        // --- hand a frame to the pilot when it is idle ---
        if (!busy && imgSize > 0 && alignLocked) {
            var conv = converter
            if (conv == null || convSize != imgSize) {
                conv = YuvConverter(imgSize)
                converter = conv
                convSize = imgSize
            }
            val img = try {
                frame.acquireCameraImage()
            } catch (_: NotYetAvailableException) {
                null
            }
            if (img != null) {
                img.use { conv.copyPlanes(it) }
                val heading = (Math.toDegrees(
                    (yawPlanar + al.offsetRad).toDouble()).toFloat() + 360f) % 360f
                val sample = SensorSample(
                    sensors.lat, sensors.lon, speedMps, heading,
                    al.toLocalE(pose.tx(), pose.tz()),
                    al.toLocalN(pose.tx(), pose.tz()), true)
                val rot = (sensorOrientation - rotationDegrees(rotation) + 360) % 360
                busy = true
                worker.execute {
                    try {
                        onPilotFrame(conv.convert(rot), sample)
                    } finally {
                        busy = false
                    }
                }
            }
        }
        onUiUpdate?.invoke()
    }

    private var convSize = 0

    private fun rotationDegrees(r: Int) = when (r) {
        Surface.ROTATION_90 -> 90
        Surface.ROTATION_180 -> 180
        Surface.ROTATION_270 -> 270
        else -> 0
    }

    private fun drawBackground() {
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glUseProgram(program)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texId)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, posBuf)
        GLES20.glVertexAttribPointer(aTex, 2, GLES20.GL_FLOAT, false, 0, texBuf)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glEnableVertexAttribArray(aTex)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aTex)
    }
}
