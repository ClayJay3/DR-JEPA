package com.mrdt.drjepa

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import androidx.camera.core.ImageProxy
import kotlin.math.min

/**
 * RGBA ImageProxy -> rotated, center-cropped, size x size CHW float tensor
 * in [0, 1] (ImageNet normalization is baked into the backbone graph).
 * Buffers are reused across frames.
 */
class FrameConverter(private val size: Int) {

    private val dst = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(dst)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()
    private val pixels = IntArray(size * size)
    val chw = FloatArray(3 * size * size)
    private var src: Bitmap? = null
    private var staging: java.nio.ByteBuffer? = null

    fun convert(image: ImageProxy): FloatArray {
        val plane = image.planes[0]
        val stridePx = plane.rowStride / plane.pixelStride
        var bmp = src
        if (bmp == null || bmp.width != stridePx || bmp.height != image.height) {
            bmp = Bitmap.createBitmap(stridePx, image.height,
                Bitmap.Config.ARGB_8888)
            src = bmp
        }
        plane.buffer.rewind()
        val need = stridePx * image.height * 4
        if (plane.buffer.remaining() >= need) {
            bmp.copyPixelsFromBuffer(plane.buffer)
        } else {
            // the last row may be shorter than the stride; pad through a
            // reused staging buffer
            val st: java.nio.ByteBuffer =
                staging?.takeIf { it.capacity() >= need }
                    ?: java.nio.ByteBuffer.allocate(need).also { staging = it }
            st.clear()
            st.put(plane.buffer)
            st.rewind()
            bmp.copyPixelsFromBuffer(st)
        }

        // rotate to upright, center-crop square, scale to model size
        val rot = image.imageInfo.rotationDegrees
        val w = image.width.toFloat()
        val h = image.height.toFloat()
        val rw = if (rot % 180 == 0) w else h
        val rh = if (rot % 180 == 0) h else w
        val s = size / min(rw, rh)
        matrix.reset()
        matrix.postTranslate(-w / 2f, -h / 2f)
        matrix.postRotate(rot.toFloat())
        matrix.postScale(s, s)
        matrix.postTranslate(size / 2f, size / 2f)
        canvas.drawBitmap(bmp, matrix, paint)

        dst.getPixels(pixels, 0, size, 0, 0, size, size)
        val hw = size * size
        for (k in 0 until hw) {
            val p = pixels[k]
            chw[k] = ((p shr 16) and 0xFF) / 255f          // R
            chw[hw + k] = ((p shr 8) and 0xFF) / 255f      // G
            chw[2 * hw + k] = (p and 0xFF) / 255f          // B
        }
        return chw
    }
}
