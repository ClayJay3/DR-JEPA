package com.mrdt.drjepa

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.cos
import kotlin.math.sin

/**
 * Top-right 2D top-down belief map (the same rendering as
 * MapPilot.map_view: grey unknown, JEPA ghost violet/light-grey, white
 * free, red occupied, path/rover/goal markers), plus a heading arrow.
 * Tapping the map sets the navigation goal at the tapped world point.
 */
class MapHudView(context: Context, attrs: AttributeSet?) :
    View(context, attrs) {

    @Volatile private var data: MapViewData? = null
    @Volatile private var poseX = 0f
    @Volatile private var poseZ = 0f
    @Volatile private var headingDeg = 0f
    var onGoalTap: ((Float, Float) -> Unit)? = null

    private val srcRect = Rect()
    private val dstRect = Rect()
    private val borderPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.WHITE
    }
    private val headingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 255, 0)
        strokeWidth = 5f
    }

    fun update(d: MapViewData?, px: Float, pz: Float, heading: Float) {
        data = d
        poseX = px
        poseZ = pz
        headingDeg = heading
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val d = data ?: return
        srcRect.set(0, 0, d.bitmap.width, d.bitmap.height)
        dstRect.set(0, 0, width, height)
        canvas.drawBitmap(d.bitmap, srcRect, dstRect, null)
        canvas.drawRect(dstRect, borderPaint)

        // heading arrow from the rover marker (map is north-up)
        val fx = (poseX - d.x0) / d.spanM
        val fy = 1f - (poseZ - d.z0) / d.spanM
        val cx = fx * width
        val cy = fy * height
        val a = Math.toRadians(headingDeg.toDouble())
        canvas.drawLine(cx, cy, cx + 18f * sin(a).toFloat(),
            cy - 18f * cos(a).toFloat(), headingPaint)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_UP) return true
        val d = data ?: return true
        val fx = (event.x / width).coerceIn(0f, 1f)
        val fy = (event.y / height).coerceIn(0f, 1f)
        onGoalTap?.invoke(d.x0 + fx * d.spanM, d.z0 + (1f - fy) * d.spanM)
        return true
    }
}
