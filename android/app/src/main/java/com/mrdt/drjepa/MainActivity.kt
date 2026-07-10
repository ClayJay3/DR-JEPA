package com.mrdt.drjepa

import android.Manifest
import android.content.Intent
import android.content.res.Configuration
import android.hardware.camera2.CameraCharacteristics
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.Surface
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.math.atan

/**
 * DR-JEPA real-world test rig: the phone is the rover. Camera frames feed
 * the perception stack, GPS + compass drive the pose filter, and the HUD
 * shows what the pilot would do (throttle/steer), the persistent belief
 * map (top-right, tap to set the goal), and the planned route projected
 * into the camera view.
 *
 * Model bundles (.drjepa, from export_android.py) are picked from device
 * storage so different training runs can be compared side by side.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var mapHud: MapHudView
    private lateinit var status: TextView
    private lateinit var sensors: SensorHub

    @Volatile private var bundle: ModelBundle? = null
    @Volatile private var pilot: Pilot? = null
    @Volatile private var converter: FrameConverter? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var lastStepNs = 0L
    private var lastDt = 0.1f

    private val prefs by lazy { getSharedPreferences("drjepa", MODE_PRIVATE) }

    private val pickModel = registerForActivityResult(
        ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        try {
            contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } catch (_: SecurityException) { /* non-persistable source is fine */ }
        prefs.edit().putString("bundle_uri", uri.toString()).apply()
        loadBundle(uri)
    }

    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        if (grants[Manifest.permission.CAMERA] == true &&
            grants[Manifest.permission.ACCESS_FINE_LOCATION] == true) {
            sensors.start()
            startCamera()
            prefs.getString("bundle_uri", null)?.let { loadBundle(Uri.parse(it)) }
        } else {
            overlay.hint = "camera + location permissions required"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        mapHud = findViewById(R.id.mapHud)
        status = findViewById(R.id.status)
        sensors = SensorHub(this)
        sensors.displayRotation = currentRotation()
        sensors.onOrientation = {
            overlay.deviceToWorld = sensors.deviceToWorld
            overlay.postInvalidateOnAnimation()
        }

        findViewById<Button>(R.id.btnLoad).setOnClickListener {
            pickModel.launch(arrayOf("*/*"))
        }
        findViewById<Button>(R.id.btnGoal).setOnClickListener {
            pilot?.setGoalAhead(30f)
        }
        findViewById<Button>(R.id.btnReset).setOnClickListener {
            pilot?.requestReset()
        }
        mapHud.onGoalTap = { x, z -> pilot?.setGoalWorld(x, z) }

        requestPermissions.launch(arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION))
    }

    private fun loadBundle(uri: Uri) {
        overlay.hint = "loading model ..."
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val b = ModelBundle.load(this@MainActivity, uri)
                val old = bundle
                pilot = Pilot(b)
                converter = FrameConverter(b.imgSize)
                bundle = b
                // the analysis executor is single-threaded: closing there
                // waits out any in-flight step still using the old sessions
                analysisExecutor.execute { old?.close() }
                withContext(Dispatchers.Main) {
                    overlay.hint = if (sensors.hasFix) null
                                   else "waiting for GPS fix ..."
                    Toast.makeText(this@MainActivity,
                        "loaded ${b.name} (${b.imgSize}px)",
                        Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    overlay.hint = "load a model bundle"
                    Toast.makeText(this@MainActivity,
                        "bundle load failed: ${e.message}",
                        Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            val analysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(
                    ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(
                    ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            // a blocked analyzer + KEEP_ONLY_LATEST is the throttle: the
            // pilot steps as fast as the model runs, frames in between drop
            analysis.setAnalyzer(analysisExecutor) { img -> analyze(img) }
            provider.unbindAll()
            val camera = provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            readCameraFov(camera)
        }, ContextCompat.getMainExecutor(this))
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun readCameraFov(camera: Camera) {
        try {
            val info = Camera2CameraInfo.from(camera.cameraInfo)
            val focal = info.getCameraCharacteristic(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                ?.firstOrNull() ?: return
            val size = info.getCameraCharacteristic(
                CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return
            overlay.fovLongDeg = Math.toDegrees(
                2.0 * atan(size.width / (2.0 * focal))).toFloat()
        } catch (_: Exception) { /* keep the default FOV */ }
    }

    private fun analyze(image: ImageProxy) {
        val p = pilot
        val conv = converter
        if (p == null || conv == null || !sensors.hasFix ||
            !sensors.hasOrientation) {
            image.close()
            if (p != null && !sensors.hasFix)
                runOnUiThread { overlay.hint = "waiting for GPS fix ..." }
            return
        }
        val chw = conv.convert(image)
        image.close()

        val now = System.nanoTime()
        val dt = if (lastStepNs == 0L) 0.1f
                 else ((now - lastStepNs) / 1e9f).coerceIn(0.05f, 2f)
        lastStepNs = now
        lastDt = dt

        val r = p.step(chw, sensors.sample(), dt)
        val mv = p.mapView(sizePx = 352)
        runOnUiThread {
            overlay.hint = null
            overlay.result = r
            overlay.invalidate()
            mapHud.update(mv, r.poseX, r.poseZ, r.headingDeg)
            status.text = statusLine(r)
        }
    }

    private fun statusLine(r: PilotResult): String {
        val b = bundle ?: return ""
        val goal = if (r.hasGoal) {
            if (r.pathWorld != null) "goal %.0f m".format(r.goalDist)
            else "goal %.0f m (NO ROUTE)".format(r.goalDist)
        } else "no goal - tap map or press Goal"
        return "%s  |  %d ms  %.1f Hz\nspd %.1f m/s  hdg %03.0f  gps +-%.0f m\n%s"
            .format(b.name, r.stepMs, 1f / lastDt, sensors.speed,
                r.headingDeg, sensors.accuracy, goal)
    }

    private fun currentRotation(): Int =
        if (Build.VERSION.SDK_INT >= 30) display?.rotation ?: Surface.ROTATION_0
        else @Suppress("DEPRECATION") windowManager.defaultDisplay.rotation

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // sensorLandscape can flip 180 without recreating the activity
        sensors.displayRotation = currentRotation()
    }

    override fun onDestroy() {
        super.onDestroy()
        sensors.stop()
        analysisExecutor.shutdown()
        bundle?.close()
    }
}
