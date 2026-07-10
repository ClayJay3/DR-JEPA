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

    /** Row-major 3x3, v_world(true ENU) = R * v_device. */
    @Volatile var deviceToWorld = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        private set
    @Volatile var hasOrientation = false; private set

    /** True-north bearing (deg) of the back camera's look direction. */
    val headingDeg: Float
        get() {
            val r = deviceToWorld
            // back camera looks along device -Z
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
        val rMag = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(rMag, event.values)
        // rMag maps device -> magnetic ENU; rotate about up by the
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
