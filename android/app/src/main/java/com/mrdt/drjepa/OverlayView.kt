package com.mrdt.drjepa

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.cos
import kotlin.math.tan

/**
 * Full-screen overlay on the camera preview: the A* planner path projected
 * onto the ground (AR), the goal marker, and the telemetry HUD (danger bar,
 * steering needle, throttle bar -- the same layout as drjepa.pilot.draw_hud).
 *
 * Projection: world point -> device frame via the rotation-vector matrix,
 * then pinhole projection with a focal length derived from the camera FOV
 * and the PreviewView FILL_CENTER crop.
 */
class OverlayView(context: Context, attrs: AttributeSet?) :
    View(context, attrs) {

    @Volatile var result: PilotResult? = null
    @Volatile var deviceToWorld: FloatArray? = null
    @Volatile var hint: String? = "load a model bundle"
    /** Horizontal FOV (deg) of the camera sensor's long side. */
    @Volatile var fovLongDeg = 65f
    /** Buffer aspect ratio, long side / short side (4:3 stream). */
    @Volatile var bufferAspect = 4f / 3f
    /** Phone height above the ground while walking (m). */
    var camHeightM = 1.4f

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

    /** World (E, N, U) -> screen (x, y, depth); depth <= 0 means behind. */
    private fun project(e: Float, n: Float, u: Float, r: FloatArray,
                        f: Float, cx: Float, cy: Float): FloatArray {
        val dx = r[0] * e + r[3] * n + r[6] * u
        val dy = r[1] * e + r[4] * n + r[7] * u
        val dz = r[2] * e + r[5] * n + r[8] * u
        val depth = -dz                       // back camera looks along -Z
        if (depth < 0.3f) return floatArrayOf(0f, 0f, depth)
        return floatArrayOf(cx + f * dx / depth, cy + f * (-dy) / depth, depth)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val res = result
        val r = deviceToWorld
        val w = width.toFloat()
        val h = height.toFloat()

        if (res != null && r != null) {
            // focal length of the FILL_CENTER-cropped preview in view px
            val f = max(h, w * bufferAspect) / 2f /
                tan(Math.toRadians(fovLongDeg / 2.0)).toFloat()
            val cx = w / 2f
            val cy = h / 2f
            val camE = res.poseX
            val camN = res.poseZ
            val camU = res.groundY + camHeightM

            val pth = res.pathWorld
            val elev = res.pathElev
            if (pth != null && elev != null) {
                arPath.reset()
                var pen = false
                for (k in 0 until pth.size / 2) {
                    val pt = project(pth[2 * k] - camE, pth[2 * k + 1] - camN,
                        elev[k] - camU, r, f, cx, cy)
                    if (pt[2] < 0.3f) { pen = false; continue }
                    if (!pen) { arPath.moveTo(pt[0], pt[1]); pen = true }
                    else arPath.lineTo(pt[0], pt[1])
                }
                canvas.drawPath(arPath, pathPaint)
                for (k in 0 until pth.size / 2) {
                    val pt = project(pth[2 * k] - camE, pth[2 * k + 1] - camN,
                        elev[k] - camU, r, f, cx, cy)
                    if (pt[2] < 0.3f) continue
                    val rad = (14f / pt[2]).coerceIn(3f, 12f)  // shrink with range
                    canvas.drawCircle(pt[0], pt[1], rad, dotPaint)
                }
            }

            if (res.hasGoal) {
                val gU = 0f  // goal flag base at believed ground
                val base = project(res.goalX - camE, res.goalZ - camN,
                    res.pathElev?.lastOrNull()?.minus(camU) ?: (gU - camU),
                    r, f, cx, cy)
                val top = project(res.goalX - camE, res.goalZ - camN,
                    (res.pathElev?.lastOrNull() ?: gU) + 2f - camU,
                    r, f, cx, cy)
                if (base[2] > 0.3f && top[2] > 0.3f) {
                    canvas.drawLine(base[0], base[1], top[0], top[1], goalPaint)
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
