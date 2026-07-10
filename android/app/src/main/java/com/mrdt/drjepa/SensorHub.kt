package com.mrdt.drjepa

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.GeomagneticField
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Looper
import android.view.Surface
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * Fused GPS + rotation-vector orientation. The phone plays the rover:
 * position/speed from GPS, heading from the camera's look direction
 * (rotation vector, declination-corrected to true north).
 *
 * World frame everywhere: x = east, y = north, z = up (true ENU).
 */
class SensorHub(context: Context) : SensorEventListener {

    private val fused = LocationServices.getFusedLocationProviderClient(context)
    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    @Volatile var lat = 0.0; private set
    @Volatile var lon = 0.0; private set
    @Volatile var speed = 0f; private set          // m/s
    @Volatile var accuracy = 99f; private set      // m
    @Volatile var hasFix = false; private set
    @Volatile private var declinationDeg = 0f

    /** Row-major 3x3, v_world(true ENU) = R * v_screen, where the screen
     *  frame is x = screen right, y = screen up, z = out of the screen
     *  (already remapped for the current display rotation). */
    @Volatile var deviceToWorld = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        private set
    @Volatile var hasOrientation = false; private set
    /** Set by the activity (Surface.ROTATION_*); landscape flips update it. */
    @Volatile var displayRotation = Surface.ROTATION_0

    /** True-north bearing (deg) of the back camera's look direction. */
    val headingDeg: Float
        get() {
            val r = deviceToWorld
            // back camera looks along -Z (same axis in screen coordinates)
            return (Math.toDegrees(
                atan2(-r[2].toDouble(), -r[5].toDouble())).toFloat() + 360f) % 360f
        }

    /** Called on every orientation update (UI-rate AR redraws). */
    var onOrientation: (() -> Unit)? = null

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val loc: Location = result.lastLocation ?: return
            lat = loc.latitude
            lon = loc.longitude
            speed = if (loc.hasSpeed()) loc.speed else 0f
            accuracy = if (loc.hasAccuracy()) loc.accuracy else 99f
            declinationDeg = GeomagneticField(
                loc.latitude.toFloat(), loc.longitude.toFloat(),
                loc.altitude.toFloat(), loc.time).declination
            hasFix = true
        }
    }

    @SuppressLint("MissingPermission")   // caller guarantees the permission
    fun start() {
        val req = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY, 500L).build()
        fused.requestLocationUpdates(req, locationCallback, Looper.getMainLooper())
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    fun stop() {
        fused.removeLocationUpdates(locationCallback)
        sensorManager.unregisterListener(this)
    }

    fun sample() = SensorSample(lat, lon, speed, headingDeg)

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ROTATION_VECTOR) return
        val rDev = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(rDev, event.values)
        // remap device axes -> screen axes for the current display rotation
        // (landscape: screen right is the device's long edge). The screen Z
        // axis stays the device Z, so the camera look direction (-Z) and
        // heading are unaffected by the remap.
        val rMag = FloatArray(9)
        when (displayRotation) {
            Surface.ROTATION_90 -> SensorManager.remapCoordinateSystem(
                rDev, SensorManager.AXIS_Y, SensorManager.AXIS_MINUS_X, rMag)
            Surface.ROTATION_180 -> SensorManager.remapCoordinateSystem(
                rDev, SensorManager.AXIS_MINUS_X, SensorManager.AXIS_MINUS_Y,
                rMag)
            Surface.ROTATION_270 -> SensorManager.remapCoordinateSystem(
                rDev, SensorManager.AXIS_MINUS_Y, SensorManager.AXIS_X, rMag)
            else -> rDev.copyInto(rMag)
        }
        // rMag maps screen -> magnetic ENU; rotate about up by the
        // declination so headings are true-north (the GPS frame)
        val d = Math.toRadians(declinationDeg.toDouble())
        val cd = cos(d).toFloat(); val sd = sin(d).toFloat()
        val r = FloatArray(9)
        for (c in 0 until 3) {
            r[c] = cd * rMag[c] + sd * rMag[3 + c]        // east'
            r[3 + c] = -sd * rMag[c] + cd * rMag[3 + c]   // north'
            r[6 + c] = rMag[6 + c]                        // up
        }
        deviceToWorld = r
        hasOrientation = true
        onOrientation?.invoke()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
