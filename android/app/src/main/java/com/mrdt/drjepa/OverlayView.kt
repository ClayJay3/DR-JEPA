package com.mrdt.drjepa

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Full-screen overlay on the ARCore camera background: the A* planner path
 * projected onto the ground, the goal marker, and the telemetry HUD
 * (danger bar, steering needle, throttle bar -- the same layout as
 * drjepa.pilot.draw_hud).
 *
 * Projection uses ARCore's view*projection matrices plus the frozen
 * EN <-> ARCore-world alignment, so the path is pinned to the ground with
 * VIO accuracy: local EN point -> ARCore world -> clip -> screen.
 */
class OverlayView(context: Context, attrs: AttributeSet?) :
    View(context, attrs) {

    @Volatile var result: PilotResult? = null
    @Volatile var viewProj: FloatArray? = null    // column-major 4x4
    @Volatile var arAlign: ArAlign? = null
    @Volatile var hint: String? = "load a model bundle"

    private val pathPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 255, 140)
        style = Paint.Style.STROKE
        strokeWidth = 10f
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        alpha = 200
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 255, 140)
    }
    private val goalPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 140, 0)
        strokeWidth = 8f
    }
    private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    private val arPath = Path()
    private val proj = FloatArray(3)   // sx, sy, view-depth (scratch)

    /** Local EN point + elevation -> screen; proj[2] <= 0 means behind. */
    private fun project(e: Float, n: Float, elevY: Float, vp: FloatArray,
                        al: ArAlign, w: Float, h: Float): FloatArray {
        // EN -> ARCore world (inverse of ArAlign.toLocal*)
        val co = cos(al.offsetRad); val so = sin(al.offsetRad)
        val ep = co * e - so * n
        val np = so * e + co * n
        val ax = ep + al.anchorX
        val az = al.anchorZ - np
        val ay = al.groundY0 + elevY
        val cx = vp[0] * ax + vp[4] * ay + vp[8] * az + vp[12]
        val cy = vp[1] * ax + vp[5] * ay + vp[9] * az + vp[13]
        val cw = vp[3] * ax + vp[7] * ay + vp[11] * az + vp[15]
        proj[2] = cw
        if (cw > 0.1f) {
            proj[0] = (cx / cw * 0.5f + 0.5f) * w
            proj[1] = (0.5f - cy / cw * 0.5f) * h
        }
        return proj
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val res = result
        val vp = viewProj
        val al = arAlign
        val w = width.toFloat()
        val h = height.toFloat()

        if (res != null && vp != null && al != null) {
            val pth = res.pathWorld
            val elev = res.pathElev
            if (pth != null && elev != null) {
                arPath.reset()
                var pen = false
                for (k in 0 until pth.size / 2) {
                    val p = project(pth[2 * k], pth[2 * k + 1], elev[k],
                        vp, al, w, h)
                    if (p[2] < 0.1f) { pen = false; continue }
                    if (!pen) { arPath.moveTo(p[0], p[1]); pen = true }
                    else arPath.lineTo(p[0], p[1])
                }
                canvas.drawPath(arPath, pathPaint)
                for (k in 0 until pth.size / 2) {
                    val p = project(pth[2 * k], pth[2 * k + 1], elev[k],
                        vp, al, w, h)
                    if (p[2] < 0.1f) continue
                    val rad = (14f / p[2]).coerceIn(3f, 12f)
                    canvas.drawCircle(p[0], p[1], rad, dotPaint)
                }
            }

            if (res.hasGoal) {
                val gElev = res.pathElev?.lastOrNull() ?: 0f
                val base = project(res.goalX, res.goalZ, gElev, vp, al, w, h)
                val bx = base[0]; val by = base[1]; val bw = base[2]
                val top = project(res.goalX, res.goalZ, gElev + 2f, vp, al, w, h)
                if (bw > 0.1f && top[2] > 0.1f) {
                    canvas.drawLine(bx, by, top[0], top[1], goalPaint)
                    canvas.drawCircle(top[0], top[1], 12f, goalPaint)
                    canvas.drawText("%.0f m".format(res.goalDist),
                        top[0] + 16f, top[1], textPaint)
                }
            }

            drawHud(canvas, res, w, h)
        }

        hint?.let {
            textPaint.textSize = 40f
            val tw = textPaint.measureText(it)
            canvas.drawText(it, (w - tw) / 2f, h * 0.45f, textPaint)
            textPaint.textSize = 34f
        }
    }

    /** Port of drjepa.pilot.draw_hud: danger bar, steering needle, throttle. */
    private fun drawHud(canvas: Canvas, res: PilotResult, w: Float, h: Float) {
        // danger bar (top-left)
        val barW = w * 0.3f
        hudPaint.style = Paint.Style.FILL
        hudPaint.color = if (res.danger > 0.5f) Color.rgb(255, 0, 0)
                         else Color.rgb(0, 200, 0)
        canvas.drawRect(24f, 40f, 24f + barW * min(res.danger, 1f), 72f, hudPaint)
        hudPaint.style = Paint.Style.STROKE
        hudPaint.strokeWidth = 2f
        hudPaint.color = Color.WHITE
        canvas.drawRect(24f, 40f, 24f + barW, 72f, hudPaint)
        canvas.drawText("danger %.2f".format(res.danger), 24f + barW + 16f, 68f,
            textPaint)

        // steering needle (bottom center)
        val cx = w / 2f
        val cy = h - 200f
        val rad = 64f
        hudPaint.color = Color.rgb(140, 140, 140)
        canvas.drawCircle(cx, cy, rad, hudPaint)
        hudPaint.color = Color.rgb(0, 255, 0)
        hudPaint.strokeWidth = 6f
        val a = res.steer * 1.2f
        canvas.drawLine(cx, cy, cx + rad * sin(a), cy - rad * cos(a), hudPaint)

        // throttle bar (right of the needle); reverse = orange, down
        val bx = cx + rad + 34f
        hudPaint.color = Color.rgb(140, 140, 140)
        hudPaint.strokeWidth = 2f
        canvas.drawRect(bx, cy - 60f, bx + 24f, cy + 60f, hudPaint)
        hudPaint.style = Paint.Style.FILL
        hudPaint.color = if (res.throttle >= 0) Color.rgb(0, 255, 0)
                         else Color.rgb(255, 120, 0)
        val t = res.throttle.coerceIn(-1f, 1f) * 60f
        canvas.drawRect(bx + 2f, cy - max(t, 0f), bx + 22f, cy - min(t, 0f),
            hudPaint)
        hudPaint.style = Paint.Style.STROKE
    }
}
