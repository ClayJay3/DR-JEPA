package com.mrdt.drjepa

import android.Manifest
import android.content.Intent
import android.content.res.Configuration
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.view.Surface
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.ar.core.ArCoreApk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * DR-JEPA real-world test rig: the phone is the rover. ARCore owns the
 * camera and supplies VIO pose/heading/speed (compass aligns its world to
 * true north once at startup); camera frames feed the perception stack;
 * the HUD shows what the pilot would do (throttle/steer), the persistent
 * belief map (top-right, tap to set the goal), and the planned route
 * projected onto the ground with ARCore's own camera matrices.
 *
 * Model bundles (.drjepa, from export_android.py) are picked from device
 * storage so different training runs can be compared side by side.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var glView: GLSurfaceView
    private lateinit var overlay: OverlayView
    private lateinit var mapHud: MapHudView
    private lateinit var status: TextView
    private lateinit var sensors: SensorHub
    private lateinit var arCam: ArCam

    @Volatile private var bundle: ModelBundle? = null
    @Volatile private var pilot: Pilot? = null
    @Volatile private var lastSample: SensorSample? = null
    private var lastStepNs = 0L
    private var lastDt = 0.1f
    private var permissionsGranted = false
    private var installRequested = false
    private var sensorsRunning = false

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
            permissionsGranted = true
            startEverything()
            prefs.getString("bundle_uri", null)?.let { loadBundle(Uri.parse(it)) }
        } else {
            overlay.hint = "camera + location permissions required"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        glView = findViewById(R.id.glView)
        overlay = findViewById(R.id.overlay)
        mapHud = findViewById(R.id.mapHud)
        status = findViewById(R.id.status)

        sensors = SensorHub(this)
        sensors.displayRotation = currentRotation()

        arCam = ArCam(this, glView, sensors) { chw, sample ->
            onPilotFrame(chw, sample)
        }
        glView.setEGLContextClientVersion(2)
        glView.preserveEGLContextOnPause = true
        glView.setRenderer(arCam)
        glView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        arCam.onUiUpdate = {
            // GL thread; volatile writes + a redraw request are safe
            overlay.viewProj = arCam.viewProj
            overlay.arAlign = arCam.align
            if (pilot != null) {
                if (!arCam.tracking) overlay.hint = arCam.trackingMsg
                else if (!arCam.alignLocked)
                    overlay.hint = "aligning to north (needs GPS fix) ..."
            }
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
        val btnRecord = findViewById<Button>(R.id.btnRecord)
        btnRecord.setOnClickListener {
            val running = arCam.recorder
            if (running != null) {
                arCam.recorder = null
                running.close()
                btnRecord.text = getString(R.string.record)
                Toast.makeText(this,
                    "saved ${running.frames} frames\n${running.dir}",
                    Toast.LENGTH_LONG).show()
            } else {
                if (!arCam.depthSupported) {
                    Toast.makeText(this, "ARCore depth not supported " +
                        "on this device", Toast.LENGTH_LONG).show()
                    return@setOnClickListener
                }
                arCam.recorder = Recorder(this) { n ->
                    runOnUiThread { btnRecord.text = "STOP ($n)" }
                }
                btnRecord.text = "STOP (0)"
            }
        }
        mapHud.onGoalTap = { x, z -> pilot?.setGoalWorld(x, z) }

        requestPermissions.launch(arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION))
    }

    /** Sensors + ARCore session; called on grant and on every onResume. */
    private fun startEverything() {
        if (!permissionsGranted) return
        if (!sensorsRunning) {
            sensors.start()
            sensorsRunning = true
        }
        if (arCam.session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(
                    this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        installRequested = true
                        return          // resumes here after the install flow
                    }
                    ArCoreApk.InstallStatus.INSTALLED -> {}
                    else -> {}
                }
                arCam.createSession()
            } catch (e: Exception) {
                overlay.hint = "ARCore unavailable: ${e.message}"
                return
            }
        }
        try {
            arCam.resume()
        } catch (e: Exception) {
            overlay.hint = "camera unavailable: ${e.message}"
        }
    }

    override fun onResume() {
        super.onResume()
        startEverything()
    }

    override fun onPause() {
        super.onPause()
        if (sensorsRunning) {
            sensors.stop()
            sensorsRunning = false
        }
        arCam.pause()
    }

    private fun loadBundle(uri: Uri) {
        overlay.hint = "loading model ..."
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val b = ModelBundle.load(this@MainActivity, uri)
                val old = bundle
                pilot = Pilot(b)
                bundle = b
                arCam.imgSize = b.imgSize    // starts the frame flow
                // single-threaded worker: runs after any in-flight step
                // still using the old sessions
                arCam.runOnWorker { old?.close() }
                withContext(Dispatchers.Main) {
                    overlay.hint = if (arCam.tracking) null
                                   else "VIO initializing - move the phone slowly"
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

    /** One pilot step; runs on ArCam's single worker thread. */
    private fun onPilotFrame(chw: FloatArray, sample: SensorSample) {
        val p = pilot ?: return
        val now = System.nanoTime()
        val dt = if (lastStepNs == 0L) 0.1f
                 else ((now - lastStepNs) / 1e9f).coerceIn(0.05f, 2f)
        lastStepNs = now
        lastDt = dt
        lastSample = sample

        val r = p.step(chw, sample, dt)
        val mv = p.mapView(sizePx = 352)
        runOnUiThread {
            overlay.hint = null
            overlay.result = r
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
        return "%s  |  %d ms  %.1f Hz  |  %s\nspd %.1f m/s  hdg %03.0f\n%s"
            .format(b.name, r.stepMs, 1f / lastDt, arCam.trackingMsg,
                lastSample?.speed ?: 0f, r.headingDeg, goal)
    }

    private fun currentRotation(): Int =
        if (Build.VERSION.SDK_INT >= 30) display?.rotation ?: Surface.ROTATION_0
        else @Suppress("DEPRECATION") windowManager.defaultDisplay.rotation

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // sensorLandscape can flip 180 without recreating the activity;
        // ArCam re-reads sensors.displayRotation every GL frame
        sensors.displayRotation = currentRotation()
    }

    override fun onDestroy() {
        super.onDestroy()
        arCam.recorder?.let { arCam.recorder = null; it.close() }
        bundle?.close()
        arCam.session?.close()
    }
}
