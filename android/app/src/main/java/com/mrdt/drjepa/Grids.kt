package com.mrdt.drjepa

import java.util.PriorityQueue
import kotlin.math.exp
import kotlin.math.hypot

/** Grid math ported from drjepa.pilot: distance transform + weighted A*.
 *  All grids are flat FloatArray/BooleanArray indexed [i * w + j]. */
object Grids {

    fun sigmoid(x: Float): Float = 1f / (1f + exp(-x.coerceIn(-30f, 30f)))

    /**
     * Two-pass 3x3 chamfer distance transform, the same approximation
     * OpenCV uses for cv2.distanceTransform(src, DIST_L2, maskSize=3):
     * distance (in cells) from every free cell to the nearest blocked one.
     *
     * @param blocked true where the cell is an obstacle (distance 0)
     */
    fun distanceTransform(blocked: BooleanArray, h: Int, w: Int): FloatArray {
        val a = 0.955f      // OpenCV's optimal 3x3 chamfer weights
        val b = 1.3693f
        val big = 1e6f
        val d = FloatArray(h * w) { if (blocked[it]) 0f else big }
        // forward pass
        for (i in 0 until h) {
            val r = i * w
            for (j in 0 until w) {
                var v = d[r + j]
                if (v == 0f) continue
                if (j > 0) v = minOf(v, d[r + j - 1] + a)
                if (i > 0) {
                    v = minOf(v, d[r - w + j] + a)
                    if (j > 0) v = minOf(v, d[r - w + j - 1] + b)
                    if (j < w - 1) v = minOf(v, d[r - w + j + 1] + b)
                }
                d[r + j] = v
            }
        }
        // backward pass
        for (i in h - 1 downTo 0) {
            val r = i * w
            for (j in w - 1 downTo 0) {
                var v = d[r + j]
                if (v == 0f) continue
                if (j < w - 1) v = minOf(v, d[r + j + 1] + a)
                if (i < h - 1) {
                    v = minOf(v, d[r + w + j] + a)
                    if (j < w - 1) v = minOf(v, d[r + w + j + 1] + b)
                    if (j > 0) v = minOf(v, d[r + w + j - 1] + b)
                }
                d[r + j] = v
            }
        }
        return d
    }

    private val STEPS = arrayOf(
        intArrayOf(-1, -1), intArrayOf(-1, 0), intArrayOf(-1, 1),
        intArrayOf(0, -1), intArrayOf(0, 1),
        intArrayOf(1, -1), intArrayOf(1, 0), intArrayOf(1, 1))
    private val STEP_LEN = floatArrayOf(1.414f, 1f, 1.414f, 1f, 1f,
        1.414f, 1f, 1.414f)

    /**
     * Weighted A* (heuristic x1.4) over an 8-connected cost grid; direct
     * port of MapPilot._astar including the lethal-start unblock, the
     * nearest-free goal substitution, and the 60k expansion cap.
     * Returns the path as flat cell indices (start..goal) or null.
     */
    fun astar(cost: FloatArray, lethal: BooleanArray, h: Int, w: Int,
              startI: Int, startJ: Int, goalI: Int, goalJ: Int): IntArray? {
        val start = startI * w + startJ
        var gi = goalI
        var gj = goalJ
        val lethalAt = { idx: Int -> lethal[idx] && idx != start }
        if (lethal[gi * w + gj]) {
            // aim at the nearest non-lethal cell to the goal
            var bestD = Long.MAX_VALUE
            var found = false
            for (i in 0 until h) for (j in 0 until w) {
                if (!lethalAt(i * w + j)) {
                    val dd = (i - goalI).toLong() * (i - goalI) +
                        (j - goalJ).toLong() * (j - goalJ)
                    if (dd < bestD) {
                        bestD = dd; gi = i; gj = j; found = true
                    }
                }
            }
            if (!found) return null
        }
        val goal = gi * w + gj
        val wH = 1.4f
        val inf = Float.MAX_VALUE
        val gsc = FloatArray(h * w) { inf }
        val came = IntArray(h * w) { -1 }
        gsc[start] = 0f
        // queue entries pack (f-score, cell) into a long for allocation-free
        // ordering: upper 32 bits = float bits of f (non-negative floats
        // compare correctly as ints), lower 32 = cell index
        val hq = PriorityQueue<Long>()
        hq.add(start.toLong())
        var expansions = 0
        var found = false
        while (hq.isNotEmpty() && expansions < 60000) {
            val cur = (hq.poll()!! and 0xFFFFFFFFL).toInt()
            expansions++
            if (cur == goal) {
                found = true
                break
            }
            val ci = cur / w
            val cj = cur % w
            val g = gsc[cur]
            for (s in STEPS.indices) {
                val i = ci + STEPS[s][0]
                val j = cj + STEPS[s][1]
                if (i < 0 || i >= h || j < 0 || j >= w) continue
                val idx = i * w + j
                if (lethalAt(idx)) continue
                val ng = g + STEP_LEN[s] * cost[idx]
                if (ng < gsc[idx]) {
                    gsc[idx] = ng
                    came[idx] = cur
                    val f = ng + wH * hypot((gi - i).toFloat(), (gj - j).toFloat())
                    hq.add((f.toRawBits().toLong() shl 32) or idx.toLong())
                }
            }
        }
        if (!found) return null
        val path = ArrayList<Int>()
        var p = goal
        while (p != start) {
            path.add(p)
            p = came[p]
            if (p == -1) return null
        }
        path.add(start)
        path.reverse()
        return path.toIntArray()
    }
}
