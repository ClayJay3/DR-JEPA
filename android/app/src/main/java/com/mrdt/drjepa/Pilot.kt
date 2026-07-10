package com.mrdt.drjepa

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

const val M_PER_DEG = 111139.0

/** One sensor snapshot fed to Pilot.step (the phone *is* the rover). */
class SensorSample(
    val lat: Double, val lon: Double,
    val speed: Float,        // m/s, from GPS
    val headingDeg: Float,   // true-north bearing of the camera look direction
)

/** Immutable per-step output handed to the UI thread. */
class PilotResult(
    val throttle: Float, val steer: Float, val danger: Float,
    val pathWorld: FloatArray?,   // [x0,z0, x1,z1, ...] local metres (x=E, z=N)
    val pathElev: FloatArray?,    // believed ground height per path point
    val poseX: Float, val poseZ: Float, val headingDeg: Float,
    val groundY: Float,           // believed ground height under the rover
    val goalX: Float, val goalZ: Float, val hasGoal: Boolean,
    val goalDist: Float, val stepMs: Long, val stepI: Int,
)

/** Rendered top-down belief map + the world rect it covers (for tap->goal). */
class MapViewData(val bitmap: Bitmap, val x0: Float, val z0: Float,
                  val spanM: Float)

/**
 * Kotlin port of drjepa.pilot.MapPilot: perceive -> remember -> plan -> act.
 *
 * Same math, same constants (read from the bundle manifest); the deliberate
 * phone adaptations are:
 *  - variable step time: pose integration, GPS gain, and log-odds decay are
 *    scaled by the real dt instead of the sim's fixed 10 Hz tick;
 *  - frame_offsets index perception steps (as in the sim's control steps),
 *    so the multi-frame parallax baseline stretches with the actual frame
 *    rate -- the motion input tells the decoder the true speed;
 *  - the speed-based stuck detector is disabled (a standing human is not a
 *    stuck rover); the arc-infeasibility recovery branches are kept and
 *    surface as "back up" advice on the HUD;
 *  - the goal comes from the UI (map tap / N metres ahead) instead of a
 *    goal GPS fix, but lives in the same local metric frame.
 *
 * Everything runs on the single analysis thread except the goal/reset
 * setters, which are latched and applied at the next step.
 */
class Pilot(private val model: ModelBundle) {

    private val p = model.pilot
    private val res = model.wedgeRes
    private val wc = model.wedgeCells
    private val n = model.mapCells
    private val compG = model.compCells

    // persistent belief channels, flat [i * n + j], i = x (east), j = z (north)
    private val L = FloatArray(n * n)     // occupancy log-odds
    private val E = FloatArray(n * n)     // fused elevation (m)
    private val EW = FloatArray(n * n)    // elevation weight
    private val LS = FloatArray(n * n)    // sand log-odds
    private val LH = FloatArray(n * n)    // terrain-hazard log-odds

    // wedge cell centers in the rover frame (x right, z forward)
    private val wxr = FloatArray(wc * wc)
    private val wzr = FloatArray(wc * wc)

    private var originLat = 0.0
    private var originLon = 0.0
    private var hasOrigin = false
    var poseX = 0f; private set
    var poseZ = 0f; private set
    private var hasPose = false
    private var cornerX = 0f
    private var cornerZ = 0f
    private var heading = 0f
    var stepI = 0; private set

    private var goalX = 0f
    private var goalZ = 0f
    private var hasGoal = false
    private var pendingReset = false

    private var path: FloatArray? = null      // [x,z,...] world
    private var prevSteer = 0f
    private var recovery = 0
    private var escape = 0
    private var recoverySteer = 0f
    private var lastCmdSteer = 0f

    // freshest-wedge zero-lag guard
    private var wedgeDist: FloatArray? = null
    private var wedgeYaw = 0f

    // plan-window distance field
    private var distM: FloatArray? = null
    private var winI0 = 0; private var winJ0 = 0
    private var winWi = 0; private var winWj = 0

    // completer (JEPA ghost) cache
    private var pred: FloatArray? = null          // (3, G, G) probabilities
    private var predObserved: FloatArray? = null  // (G, G)
    private var predOriginX = 0f
    private var predOriginZ = 0f
    private var predStep = -99

    // multi-frame token history: (tokens, [speed_norm, cmd_steer]) per step
    private val maxOff = model.frameOffsets.max()
    private val tokHist = ArrayDeque<Pair<FloatArray, FloatArray>>()

    init {
        val half = wc * res / 2f
        for (i in 0 until wc) for (j in 0 until wc) {
            wxr[i * wc + j] = (i + 0.5f) * res - half
            wzr[i * wc + j] = (j + 0.5f) * res
        }
    }

    // ---------------- external control (UI thread; latched) ----------------
    @Synchronized
    fun requestReset() { pendingReset = true }

    @Synchronized
    fun setGoalWorld(x: Float, z: Float) {
        goalX = x; goalZ = z; hasGoal = true; path = null
    }

    @Synchronized
    fun setGoalAhead(metres: Float) {
        if (!hasPose) return
        val yaw = Math.toRadians(heading.toDouble())
        setGoalWorld(poseX + metres * sin(yaw).toFloat(),
                     poseZ + metres * cos(yaw).toFloat())
    }

    private fun doReset() {
        L.fill(0f); E.fill(0f); EW.fill(0f); LS.fill(0f); LH.fill(0f)
        hasOrigin = false; hasPose = false; hasGoal = false
        stepI = 0; path = null; prevSteer = 0f; recovery = 0; escape = 0
        lastCmdSteer = 0f; wedgeDist = null; distM = null
        pred = null; predObserved = null; predStep = -99
        tokHist.clear()
    }

    // ---------------- pose ----------------
    private fun updatePose(s: SensorSample, dt: Float) {
        if (!hasOrigin) {
            originLat = s.lat; originLon = s.lon; hasOrigin = true
        }
        val gx = ((s.lon - originLon) * M_PER_DEG *
            cos(Math.toRadians(originLat))).toFloat()
        val gz = ((s.lat - originLat) * M_PER_DEG).toFloat()
        if (!hasPose) {
            poseX = gx; poseZ = gz; hasPose = true
            val c = n * res / 2f
            cornerX = poseX - c
            cornerZ = poseZ - c
        } else {
            val head = Math.toRadians(s.headingDeg.toDouble())
            poseX += s.speed * dt * sin(head).toFloat()
            poseZ += s.speed * dt * cos(head).toFloat()
            // per-step gain in the sim -> per-dt gain here
            val gain = 1f - Math.pow((1.0 - p.gpsGain), (dt / p.dt).toDouble())
                .toFloat()
            poseX += gain * (gx - poseX)
            poseZ += gain * (gz - poseZ)
        }
        heading = s.headingDeg
        maybeRecenter()
    }

    private fun maybeRecenter() {
        val ii = (poseX - cornerX) / res
        val jj = (poseZ - cornerZ) / res
        val margin = n * 0.15f
        if (ii > margin && ii < n - margin && jj > margin && jj < n - margin)
            return
        val si = floor(ii - n / 2f).toInt()
        val sj = floor(jj - n / 2f).toInt()
        for (arr in arrayOf(L, E, EW, LS, LH)) {
            val src = arr.copyOf()
            arr.fill(0f)
            val iSrc0 = max(0, si); val iSrc1 = min(n, n + si)
            val jSrc0 = max(0, sj); val jSrc1 = min(n, n + sj)
            val iDst0 = max(0, -si); val jDst0 = max(0, -sj)
            for (i in iSrc0 until iSrc1) {
                val di = iDst0 + (i - iSrc0)
                for (j in jSrc0 until jSrc1)
                    arr[di * n + jDst0 + (j - jSrc0)] = src[i * n + j]
            }
        }
        cornerX += si * res
        cornerZ += sj * res
    }

    // ---------------- perception ----------------
    private fun perceive(chw: FloatArray, speed: Float): Wedge {
        val tokens = model.runBackbone(chw)
        tokHist.addLast(Pair(tokens, floatArrayOf(speed / model.speedNorm,
                                                  lastCmdSteer)))
        while (tokHist.size > maxOff + 1) tokHist.removeFirst()
        val ntfd = model.nTokens * model.featDim
        val f = model.frameOffsets.size
        val stacked = FloatArray(f * ntfd)
        val motion = FloatArray(2 * f)
        for (k in 0 until f) {
            val off = model.frameOffsets[k]
            val idx = max(0, tokHist.size - 1 - off)   // clamped in warm-up
            tokHist[idx].first.copyInto(stacked, k * ntfd)
            motion[2 * k] = tokHist[idx].second[0]
            motion[2 * k + 1] = tokHist[idx].second[1]
        }
        return model.runDecoder(stacked, motion)
    }

    // ---------------- fusion ----------------
    private fun paint(w: Wedge, dt: Float) {
        val wc2 = wc * wc
        val conf = FloatArray(wc2)
        for (k in 0 until wc2) {
            var c = Grids.sigmoid(w.conf[k])
            if (k % wc >= p.paintRangeCells) c = 0f    // untrusted far field
            if (c < 0.55f) c = 0f                      // low-vis cells
            conf[k] = c
        }
        // per-step decay in the sim -> per-dt decay here
        val decay = exp(ln(p.decay) * dt / p.dt)
        for (k in L.indices) L[k] *= decay

        val yaw = Math.toRadians(heading.toDouble())
        val cy = cos(yaw).toFloat(); val sy = sin(yaw).toFloat()
        val evidence = FloatArray(wc2)
        val upd = FloatArray(wc2)
        for (k in 0 until wc2) {
            var e = w.occ[k] - p.priorLogit
            if (e > 0) e *= p.posEvidenceScale
            evidence[k] = e
            upd[k] = e.coerceIn(-4f, 4f) * conf[k] * 0.55f
        }

        val ii = IntArray(wc2); val jj = IntArray(wc2)
        val ok = BooleanArray(wc2)
        fun cells() {
            for (k in 0 until wc2) {
                val wx = poseX + wxr[k] * cy + wzr[k] * sy
                val wz = poseZ - wxr[k] * sy + wzr[k] * cy
                val i = floor((wx - cornerX) / res).toInt()
                val j = floor((wz - cornerZ) / res).toInt()
                ii[k] = i; jj[k] = j
                ok[k] = i in 0 until n && j in 0 until n && conf[k] > 0f
            }
        }
        cells()
        if (voAlign(ii, jj, upd, ok)) cells()   // repaint at corrected pose

        for (k in 0 until wc2) if (ok[k]) {
            val idx = ii[k] * n + jj[k]
            // positive cap below negative cap: occupied belief stays revisable
            L[idx] = (L[idx] + upd[k]).coerceIn(-p.loddsClamp, 3.5f)
        }

        // elevation: map stores ABSOLUTE height (local-origin frame); the
        // wedge is relative to the rover's ground, estimated from fused
        // cells near the rover
        val ci = ((poseX - cornerX) / res).toInt()
        val cj = ((poseZ - cornerZ) / res).toInt()
        var wSum = 0f; var eSum = 0f
        for (i in max(0, ci - 2)..min(n - 1, ci + 2))
            for (j in max(0, cj - 2)..min(n - 1, cj + 2)) {
                wSum += EW[i * n + j]
                eSum += E[i * n + j] * EW[i * n + j]
            }
        val h0 = if (wSum > 0.5f) eSum / wSum else 0f
        for (k in 0 until wc2) if (ok[k]) {
            val idx = ii[k] * n + jj[k]
            val wgt = conf[k]
            val old = E[idx] * EW[idx]
            EW[idx] = min(EW[idx] + wgt, 20f)
            E[idx] = (old + (w.elev[k] + h0) * wgt) / max(EW[idx], 1e-6f)
        }

        for (k in 0 until wc2) if (ok[k]) {
            val idx = ii[k] * n + jj[k]
            var sev = (w.sand[k] - p.sandPrior).coerceIn(-4f, 4f)
            if (sev > 0) sev *= 0.5f
            LS[idx] = (LS[idx] + sev * conf[k] * 0.55f).coerceIn(-6f, 4f)

            var hev = (w.haz[k] - p.hazPriorLogit).coerceIn(-4f, 4f)
            if (hev > 0) {
                hev *= p.posEvidenceScale
                // positive hazard evidence only trusted NEAR (range-banded
                // attenuation; far-band FPs flood the map otherwise)
                val j = k % wc
                if (j >= 16) hev *= 0.25f else if (j >= 8) hev *= 0.5f
            }
            LH[idx] = (LH[idx] + hev * conf[k] * 0.55f).coerceIn(-6f, 2f)
        }

        // zero-lag near-field guard in the CURRENT rover frame
        val blocked = BooleanArray(wc2)
        for (k in 0 until wc2)
            blocked[k] = evidence[k] > p.guardEvidence && conf[k] > 0.5f
        val d = Grids.distanceTransform(blocked, wc, wc)
        for (k in d.indices) d[k] *= res
        wedgeDist = d
        wedgeYaw = yaw.toFloat()
    }

    /** Scan-matching pose correction; returns true if the pose moved. */
    private fun voAlign(ii: IntArray, jj: IntArray, upd: FloatArray,
                        ok: BooleanArray): Boolean {
        if (recovery > 0 || escape > 0) return false
        val ks = ArrayList<Int>()
        for (k in upd.indices)
            if (ok[k] && upd[k] > 0.4f) ks.add(k)   // opinionated-positive only
        if (ks.size < 25) return false
        val w = p.voWindow
        var best = -1e18; var bestDi = 0; var bestDj = 0; var zero = 0.0
        for (di in -w..w) for (dj in -w..w) {
            var s = 0.0
            for (k in ks) {
                val i2 = (ii[k] + di).coerceIn(0, n - 1)
                val j2 = (jj[k] + dj).coerceIn(0, n - 1)
                val l = L[i2 * n + j2]
                if (abs(l) > p.voMinL) s += upd[k] * l
            }
            s -= 1.5 * (di * di + dj * dj)
            if (di == 0 && dj == 0) zero = s
            if (s > best) { best = s; bestDi = di; bestDj = dj }
        }
        if ((bestDi != 0 || bestDj != 0) && best - zero > p.voMargin) {
            poseX += p.voGain * bestDi * res
            poseZ += p.voGain * bestDj * res
            return true
        }
        return false
    }

    // ---------------- completer (map-space JEPA ghost) ----------------
    private fun completeMap() {
        if (predStep == stepI) return
        val g = compG
        val halfM = g * model.compRes / 2f
        val x0 = poseX - halfM
        val z0 = poseZ - halfM
        val comp = FloatArray(4 * g * g)
        val gg = g * g
        for (gi in 0 until g) {
            val mi = (((x0 + (gi + 0.5f) * model.compRes) - cornerX) / res)
                .toInt().coerceIn(0, n - 1)
            for (gj in 0 until g) {
                val mj = (((z0 + (gj + 0.5f) * model.compRes) - cornerZ) / res)
                    .toInt().coerceIn(0, n - 1)
                val idx = mi * n + mj
                val occP = Grids.sigmoid(L[idx])
                val obs = if (abs(L[idx]) > 0.4f || EW[idx] > 1f) 1f else 0f
                val o = gi * g + gj
                comp[o] = occP * obs
                comp[gg + o] = if (Grids.sigmoid(LH[idx]) > 0.6f) 1f else 0f
                comp[2 * gg + o] = Grids.sigmoid(LS[idx]) * obs
                comp[3 * gg + o] = obs
            }
        }
        pred = model.runCompleter(comp)
        predObserved = comp.copyOfRange(3 * gg, 4 * gg)
        predOriginX = x0
        predOriginZ = z0
        predStep = stepI
    }

    // ---------------- planning ----------------
    private fun replan() {
        val loX = min(poseX, goalX) - 16f; val hiX = max(poseX, goalX) + 16f
        val loZ = min(poseZ, goalZ) - 16f; val hiZ = max(poseZ, goalZ) + 16f
        val i0 = max(0, ((loX - cornerX) / res).toInt())
        val j0 = max(0, ((loZ - cornerZ) / res).toInt())
        val i1 = min(n, ((hiX - cornerX) / res + 1).toInt())
        val j1 = min(n, ((hiZ - cornerZ) / res + 1).toInt())
        val wi = i1 - i0; val wj = j1 - j0

        val blockedW = BooleanArray(wi * wj)
        val prob = FloatArray(wi * wj)
        for (i in 0 until wi) for (j in 0 until wj) {
            val pr = Grids.sigmoid(L[(i0 + i) * n + j0 + j])
            prob[i * wj + j] = pr
            blockedW[i * wj + j] = pr > p.occThresh
        }
        val dist = Grids.distanceTransform(blockedW, wi, wj)
        for (k in dist.indices) dist[k] *= res
        distM = dist
        winI0 = i0; winJ0 = j0; winWi = wi; winWj = wj

        val ph = wi / 2; val pw = wj / 2
        if (ph < 2 || pw < 2) { path = null; return }

        val cost = FloatArray(ph * pw)
        val lethal = BooleanArray(ph * pw)
        val lethal2 = BooleanArray(ph * pw)     // slimmer fallback
        for (a in 0 until ph) for (b in 0 until pw) {
            var p2 = 0f; var d2 = Float.MAX_VALUE
            var h2 = 0f; var s2 = 0f; var o2 = false
            for (di in 0..1) for (dj in 0..1) {
                val i = 2 * a + di; val j = 2 * b + dj
                val wIdx = i * wj + j
                val mIdx = (i0 + i) * n + j0 + j
                p2 = max(p2, prob[wIdx])
                d2 = min(d2, dist[wIdx])
                h2 = max(h2, Grids.sigmoid(LH[mIdx]))
                s2 = max(s2, Grids.sigmoid(LS[mIdx]))
                if (abs(L[mIdx]) > 0.4f || EW[mIdx] > 1f) o2 = true
            }
            val k = a * pw + b
            lethal[k] = d2 < 0.9f || h2 > 0.7f      // rover radius + margin
            lethal2[k] = d2 < 0.5f || h2 > 0.8f
            var c = 1f + 6f * p2 +
                (if (d2 < 2f) (2f - d2) * 2f else 0f) +
                5f * (h2 / 0.7f).coerceIn(0f, 1f) + 3f * s2
            if (!o2) c += 0.3f    // exploration is not free
            cost[k] = c
        }

        fun cellI(x: Float) =
            ((x - cornerX) / res - i0).coerceIn(0f, ph * 2f - 1).toInt() / 2
        fun cellJ(z: Float) =
            ((z - cornerZ) / res - j0).coerceIn(0f, pw * 2f - 1).toInt() / 2

        var cells = Grids.astar(cost, lethal, ph, pw,
            cellI(poseX), cellJ(poseZ), cellI(goalX), cellJ(goalZ))
        if (cells == null)
            // believed-blocked: relax obstacle inflation once, but never
            // believed-fatal slopes (a tip-over is terminal)
            cells = Grids.astar(cost, lethal2, ph, pw,
                cellI(poseX), cellJ(poseZ), cellI(goalX), cellJ(goalZ))
        if (cells == null) { path = null; return }
        val pts = FloatArray(cells.size * 2)
        for (k in cells.indices) {
            val a = cells[k] / pw; val b = cells[k] % pw
            pts[2 * k] = cornerX + (i0 + (a + 0.5f) * 2) * res
            pts[2 * k + 1] = cornerZ + (j0 + (b + 0.5f) * 2) * res
        }
        path = pts
    }

    // ---------------- clearance lookups ----------------
    private fun mapClearance(x: Float, z: Float): Float {
        val d = distM ?: return 99f
        val i = ((x - cornerX) / res - winI0).coerceIn(0f, winWi - 1f).toInt()
        val j = ((z - cornerZ) / res - winJ0).coerceIn(0f, winWj - 1f).toInt()
        return d[i * winWj + j]
    }

    /** Min clearance along one arc's (dx, dz) offsets via the fresh wedge. */
    private fun wedgeClearance(dx: FloatArray, dz: FloatArray): Float {
        val wd = wedgeDist ?: return 99f
        val cy = cos(wedgeYaw); val sy = sin(wedgeYaw)
        val half = wc * res / 2f
        var m = 99f
        for (t in dx.indices) {
            val rx = dx[t] * cy - dz[t] * sy
            val rz = dx[t] * sy + dz[t] * cy
            val i = floor((rx + half) / res).toInt()
            val j = floor(rz / res).toInt()
            val v = if (i in 0 until wc && j in 0 until wc) wd[i * wc + j]
                    else 99f
            if (v < m) m = v
        }
        return m
    }

    // ---------------- local control ----------------
    private fun control(danger: Float, vMeas: Float): Pair<Float, Float> {
        // waypoint ~7 m ahead on the A* path (or the goal directly)
        var targetX = goalX; var targetZ = goalZ
        val pth = path
        if (pth != null && pth.size >= 4) {
            var kMin = 0; var dMin = Float.MAX_VALUE
            for (k in 0 until pth.size / 2) {
                val d = hypot(pth[2 * k] - poseX, pth[2 * k + 1] - poseZ)
                if (d < dMin) { dMin = d; kMin = k }
            }
            var along = 0f
            var wIdx = pth.size / 2 - 1          // fallback: path end
            for (k in kMin until pth.size / 2 - 1) {
                along += hypot(pth[2 * k + 2] - pth[2 * k],
                               pth[2 * k + 3] - pth[2 * k + 1])
                if (along >= 7f) { wIdx = min(k + 1, pth.size / 2 - 1); break }
            }
            targetX = pth[2 * wIdx]; targetZ = pth[2 * wIdx + 1]
        }

        val goalD = hypot(goalX - poseX, goalZ - poseZ)
        val yaw = Math.toRadians(heading.toDouble()).toFloat()

        if (recovery > 0) {
            recovery--
            prevSteer = recoverySteer
            return Pair(-0.5f, recoverySteer)
        }
        if (escape > 0) {
            escape--
            prevSteer = recoverySteer
            return Pair(0.3f, recoverySteer)
        }

        // arc sampling against the BELIEVED map
        val nA = p.nArcs
        val nT = (p.arcT / p.arcDt).toInt()
        val vPlan = abs(vMeas).coerceIn(1.5f, 6f)
        val steers = FloatArray(nA) { -1f + 2f * it / (nA - 1) }
        val clear = FloatArray(nA)
        val feasible = BooleanArray(nA)
        val endYaw = FloatArray(nA)
        val endX = FloatArray(nA); val endZ = FloatArray(nA)
        val sandNear = FloatArray(nA)
        val arcDx = Array(nA) { FloatArray(nT) }
        val arcDz = Array(nA) { FloatArray(nT) }
        for (a in 0 until nA) {
            val yr = Math.toRadians(steers[a] * 70.0).toFloat() // yaw_rate_max
            var cx = 0f; var cz = 0f
            var mClear = Float.MAX_VALUE
            var hazMax = 0f
            for (t in 1..nT) {
                val tt = t * p.arcDt
                val segYaw = yaw + yr * (tt - p.arcDt / 2f)
                cx += sin(segYaw) * vPlan * p.arcDt
                cz += cos(segYaw) * vPlan * p.arcDt
                arcDx[a][t - 1] = cx
                arcDz[a][t - 1] = cz
                val px = poseX + cx; val pz = poseZ + cz
                mClear = min(mClear, mapClearance(px, pz))
                val mi = ((px - cornerX) / res).toInt().coerceIn(0, n - 1)
                val mj = ((pz - cornerZ) / res).toInt().coerceIn(0, n - 1)
                hazMax = max(hazMax, Grids.sigmoid(LH[mi * n + mj]))
                if (t <= 4)
                    sandNear[a] = max(sandNear[a],
                        Grids.sigmoid(LS[mi * n + mj]))
                if (t == nT) {
                    endYaw[a] = segYaw; endX[a] = px; endZ[a] = pz
                }
            }
            // rover radius + buffer absorbing pose-filter error; the fresh
            // wedge guard has no pose error so it gets a slimmer buffer
            var c = mClear - 0.95f
            c = min(c, wedgeClearance(arcDx[a], arcDz[a]) - 0.85f)
            feasible[a] = c > 0.05f
            if (hazMax > 0.65f) {                 // believed-too-steep arc
                feasible[a] = false
                c = min(c, 0f)
            }
            clear[a] = c
        }
        // NOTE: the sim pilot's speed-based stuck detector is intentionally
        // not ported -- a pedestrian standing still is not a stuck rover.

        if (!feasible.any { it }) {
            var left = -Float.MAX_VALUE; var right = -Float.MAX_VALUE
            for (a in 0 until nA / 2) left = max(left, clear[a])
            for (a in nA / 2 + 1 until nA) right = max(right, clear[a])
            val wd = wedgeDist
            if (wd != null) {
                val ci = wc / 2
                var ahead = Float.MAX_VALUE
                for (i in ci - 2..ci + 2) for (j in 0 until 6)
                    ahead = min(ahead, wd[i * wc + j])
                if (ahead < 0.85f) {              // imminent frontal block
                    recovery = 14
                    recoverySteer = if (left > right) 0.8f else -0.8f
                    prevSteer = recoverySteer
                    return Pair(-0.5f, recoverySteer)
                }
            }
            if (path != null) {
                // tight passage: creep along the plan instead of thrashing
                val bear = atan2(targetX - poseX, targetZ - poseZ)
                var err = bear - yaw
                while (err > Math.PI) err -= 2 * Math.PI.toFloat()
                while (err < -Math.PI) err += 2 * Math.PI.toFloat()
                val steer = (err / Math.toRadians(35.0).toFloat())
                    .coerceIn(-1f, 1f)
                prevSteer = steer
                return Pair(0.25f, steer)
            }
            recovery = 14
            recoverySteer = if (left > right) 0.8f else -0.8f
            prevSteer = recoverySteer
            return Pair(-0.5f, recoverySteer)
        }

        val wd0 = hypot(targetX - poseX, targetZ - poseZ)
        var best = 0; var bestScore = -Float.MAX_VALUE
        for (a in 0 until nA) {
            val prog = wd0 - hypot(targetX - endX[a], targetZ - endZ[a])
            val bear = atan2(targetX - endX[a], targetZ - endZ[a])
            val align = cos(bear - endYaw[a])
            var score = 1.4f * prog + 2.2f * min(clear[a], 2.5f) +
                1.5f * align - 1.2f * abs(steers[a] - prevSteer)
            if (!feasible[a]) score = -1e9f
            if (score > bestScore) { bestScore = score; best = a }
        }
        val steer = steers[best]

        // throttle governors: clearance, turn sharpness, goal proximity, sand
        val vClear = (clear[best] / 3.5f).coerceIn(0.35f, 1f)
        val vTurn = 1f - 0.4f * abs(steer)
        val vGoal = (goalD / 8f).coerceIn(0.3f, 1f)
        val vSand = 1f - 0.45f * sandNear[best]
        var throttle = minOf(vClear, vTurn, vGoal, vSand).coerceIn(0.25f, 1f)
        // thread tight passages slowly
        val wd = wedgeDist
        if (wd != null) {
            val ci = wc / 2
            var ahead = Float.MAX_VALUE
            for (i in ci - 3..ci + 3) for (j in 0 until 8)
                ahead = min(ahead, wd[i * wc + j])
            if (ahead < 2f) throttle = min(throttle, 0.35f)
            else if (ahead < 3.5f) throttle = min(throttle, 0.55f)
        }
        if (danger > 0.65f) throttle = min(throttle, 0.4f)
        prevSteer = steer
        return Pair(throttle, steer)
    }

    // ---------------- main entry (analysis thread) ----------------
    fun step(chw: FloatArray, sample: SensorSample, dt: Float): PilotResult {
        val t0 = System.nanoTime()
        val goalXc: Float; val goalZc: Float; val hasGoalC: Boolean
        synchronized(this) {
            if (pendingReset) { doReset(); pendingReset = false }
            goalXc = goalX; goalZc = goalZ; hasGoalC = hasGoal
        }
        updatePose(sample, dt)
        val wedge = perceive(chw, sample.speed)
        paint(wedge, dt)
        val danger = Grids.sigmoid(wedge.dangerLogit)
        var throttle = 0f; var steer = 0f
        if (hasGoalC) {
            if (stepI % p.replanEvery == 0 || path == null) replan()
            val c = control(danger, sample.speed)
            throttle = c.first; steer = c.second
        }
        stepI++
        lastCmdSteer = steer
        completeMap()   // JEPA ghost for the HUD (cached per step)

        val pth = path
        val elevs = pth?.let { pts ->
            FloatArray(pts.size / 2) { k -> groundHeight(pts[2 * k], pts[2 * k + 1]) }
        }
        return PilotResult(
            throttle, steer, danger, pth, elevs,
            poseX, poseZ, heading, groundHeight(poseX, poseZ),
            goalXc, goalZc, hasGoalC,
            if (hasGoalC) hypot(goalXc - poseX, goalZc - poseZ) else 0f,
            (System.nanoTime() - t0) / 1_000_000, stepI)
    }

    /** Believed ground height (m, local-origin frame) at a world point. */
    fun groundHeight(x: Float, z: Float): Float {
        if (!hasPose) return 0f
        val i = ((x - cornerX) / res).toInt().coerceIn(0, n - 1)
        val j = ((z - cornerZ) / res).toInt().coerceIn(0, n - 1)
        val idx = i * n + j
        return if (EW[idx] > 1f) E[idx] else 0f
    }

    // ---------------- map HUD (port of MapPilot.map_view) ----------------
    private val violetR = 170; private val violetG = 90; private val violetB = 200

    fun mapView(sizePx: Int, spanM: Float = 40f): MapViewData? {
        if (!hasPose) return null
        val c = (spanM / res).toInt()
        val ci = ((poseX - cornerX) / res).toInt()
        val cj = ((poseZ - cornerZ) / res).toInt()
        val i0 = (ci - c / 2).coerceIn(0, n - c)
        val j0 = (cj - c / 2).coerceIn(0, n - c)
        val px = IntArray(c * c)

        val predL = pred; val predObs = predObserved
        val gg = compG * compG
        for (i in 0 until c) for (j in 0 until c) {
            val idx = (i0 + i) * n + (j0 + j)
            val prob = Grids.sigmoid(L[idx])
            val known = abs(L[idx]) > 0.4f
            var r = 128; var g = 128; var b = 128          // unknown = grey

            // JEPA ghost layer: predicted content of UNSEEN cells
            if (!known && predL != null && predObs != null) {
                val gi = floor(((cornerX + (i0 + i + 0.5f) * res) -
                    predOriginX) / model.compRes).toInt()
                val gj = floor(((cornerZ + (j0 + j + 0.5f) * res) -
                    predOriginZ) / model.compRes).toInt()
                if (gi in 0 until compG && gj in 0 until compG &&
                    predObs[gi * compG + gj] < 0.5f) {
                    val blockedP = max(predL[gi * compG + gj],
                                       predL[gg + gi * compG + gj])
                    if (blockedP < 0.25f) {
                        r = 150; g = 150; b = 150          // predicted open
                    } else if (blockedP > 0.45f) {         // predicted wall
                        val a = 0.55f + 0.45f *
                            ((blockedP - 0.45f) / 0.4f).coerceIn(0f, 1f)
                        r = ((1 - a) * 128 + a * violetR).toInt()
                        g = ((1 - a) * 128 + a * violetG).toInt()
                        b = ((1 - a) * 128 + a * violetB).toInt()
                    }
                }
            }
            if (known && prob <= 0.5f) { r = 230; g = 230; b = 230 }  // free
            if (known && prob > 0.5f) {                                // occ
                val v = (prob.coerceIn(0.5f, 1f) - 0.5f) * 2f
                r = (120 + 135 * v).toInt(); g = 40; b = 40
            }
            // north-up: x (east) right, z (north) up
            px[(c - 1 - j) * c + i] = Color.rgb(r, g, b)
        }

        fun mark(x: Float, z: Float, color: Int) {
            val i = ((x - cornerX) / res).toInt() - i0
            val j = ((z - cornerZ) / res).toInt() - j0
            if (i !in 0 until c || j !in 0 until c) return
            for (di in -1..1) for (dj in -1..1) {
                val ei = i + di; val ej = j + dj
                if (ei in 0 until c && ej in 0 until c)
                    px[(c - 1 - ej) * c + ei] = color
            }
        }
        val pth = path
        if (pth != null)
            for (k in 0 until pth.size / 2 step 2)
                mark(pth[2 * k], pth[2 * k + 1], Color.rgb(255, 200, 0))
        mark(poseX, poseZ, Color.rgb(0, 255, 0))
        if (hasGoal) mark(goalX, goalZ, Color.rgb(0, 100, 255))

        val small = Bitmap.createBitmap(px, c, c, Bitmap.Config.ARGB_8888)
        val bmp = Bitmap.createScaledBitmap(small, sizePx, sizePx, false)
        return MapViewData(bmp, cornerX + i0 * res, cornerZ + j0 * res,
                           c * res)
    }
}
