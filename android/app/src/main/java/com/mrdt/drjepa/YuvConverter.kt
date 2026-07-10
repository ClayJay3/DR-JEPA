package com.mrdt.drjepa

import android.media.Image
import kotlin.math.min

/**
 * ARCore CPU image (YUV_420_888) -> rotated, center-cropped, size x size
 * CHW float RGB in [0, 1].
 *
 * copyPlanes() runs on the GL thread (a fast memcpy so the Image can be
 * closed immediately -- ARCore only lends a few buffers); convert() does
 * the rotation + bilinear resample + YUV->RGB on the worker thread.
 */
class YuvConverter(private val size: Int) {

    private var yBuf = ByteArray(0)
    private var uBuf = ByteArray(0)
    private var vBuf = ByteArray(0)
    private var w = 0
    private var h = 0
    private var yStride = 0
    private var uvStride = 0
    private var uvPixStride = 0
    val chw = FloatArray(3 * size * size)

    fun copyPlanes(img: Image) {
        w = img.width
        h = img.height
        val py = img.planes[0]
        val pu = img.planes[1]
        val pv = img.planes[2]
        yStride = py.rowStride
        uvStride = pu.rowStride
        uvPixStride = pu.pixelStride

        fun copy(plane: Image.Plane, into: ByteArray): ByteArray {
            val buf = plane.buffer.duplicate()
            buf.rewind()
            val n = buf.remaining()
            val arr = if (into.size >= n) into else ByteArray(n)
            buf.get(arr, 0, n)
            return arr
        }
        yBuf = copy(py, yBuf)
        uBuf = copy(pu, uBuf)
        vBuf = copy(pv, vBuf)
    }

    /**
     * @param rotationDeg 0/90/180/270: rotation that makes the sensor
     *        image upright on the display ((sensorOrientation -
     *        displayDeg + 360) % 360 for a back camera)
     */
    fun convert(rotationDeg: Int): FloatArray {
        // upright dimensions and the centered square crop within them
        val uw = if (rotationDeg % 180 == 0) w else h
        val uh = if (rotationDeg % 180 == 0) h else w
        val crop = min(uw, uh)
        val offX = (uw - crop) / 2f
        val offY = (uh - crop) / 2f
        val scale = crop.toFloat() / size
        val hw = size * size

        for (ty in 0 until size) {
            val fy = offY + (ty + 0.5f) * scale - 0.5f
            for (tx in 0 until size) {
                val fx = offX + (tx + 0.5f) * scale - 0.5f
                // upright -> sensor coordinates
                val sx: Float
                val sy: Float
                when (rotationDeg) {
                    90 -> { sx = fy; sy = h - 1f - fx }
                    180 -> { sx = w - 1f - fx; sy = h - 1f - fy }
                    270 -> { sx = w - 1f - fy; sy = fx }
                    else -> { sx = fx; sy = fy }
                }
                val x0 = sx.toInt().coerceIn(0, w - 1)
                val y0 = sy.toInt().coerceIn(0, h - 1)
                val x1 = min(x0 + 1, w - 1)
                val y1 = min(y0 + 1, h - 1)
                val ax = (sx - x0).coerceIn(0f, 1f)
                val ay = (sy - y0).coerceIn(0f, 1f)
                // bilinear luma
                val y00 = (yBuf[y0 * yStride + x0].toInt() and 0xFF)
                val y01 = (yBuf[y0 * yStride + x1].toInt() and 0xFF)
                val y10 = (yBuf[y1 * yStride + x0].toInt() and 0xFF)
                val y11 = (yBuf[y1 * yStride + x1].toInt() and 0xFF)
                val yv = (y00 * (1 - ax) + y01 * ax) * (1 - ay) +
                    (y10 * (1 - ax) + y11 * ax) * ay
                // nearest chroma (half resolution)
                val ci = (y0 shr 1) * uvStride + (x0 shr 1) * uvPixStride
                val u = (uBuf[ci].toInt() and 0xFF) - 128
                val v = (vBuf[ci].toInt() and 0xFF) - 128
                val r = yv + 1.402f * v
                val g = yv - 0.344136f * u - 0.714136f * v
                val b = yv + 1.772f * u
                val k = ty * size + tx
                chw[k] = (r / 255f).coerceIn(0f, 1f)
                chw[hw + k] = (g / 255f).coerceIn(0f, 1f)
                chw[2 * hw + k] = (b / 255f).coerceIn(0f, 1f)
            }
        }
        return chw
    }
}
