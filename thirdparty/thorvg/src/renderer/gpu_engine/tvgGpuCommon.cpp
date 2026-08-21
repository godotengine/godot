/*
 * Copyright (c) 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "tvgGpuCommon.h"

/************************************************************************/
/* Utility Functions Implementation                                     */
/************************************************************************/

namespace tvg
{

constexpr auto PATH_OPT_PX_TOLERANCE = 0.25f;
constexpr auto DASH_ENDPOINT_TOLERANCE = DASH_PATTERN_THRESHOLD;
// Keep this aligned with the SW rasterizer's AA coverage precision.
// tvgRender.h::MULTIPLY() and the SW coverage paths use an 8-bit coverage model.
constexpr uint32_t SW_AA_COVERAGE_BITS = 8;
constexpr auto AA_COVERAGE_QUANTUM = 1.0f / static_cast<float>(1u << SW_AA_COVERAGE_BITS);
constexpr auto MAX_PIXEL_SPAN = 1.41421356237f;

uint32_t gpuArcSegmentsCnt(float arcAngle, float pixelRadius)
{
    if (pixelRadius < FLOAT_EPSILON) return 2;
    static constexpr auto PX_TOLERANCE = 0.25f;
    // Sagitta-based formula Approximation: 1 - cos(θ/2) ≈ (θ/2)²/2, so θ ≈ 2 * sqrt(2 * s/r)
    auto segmentAngle = 2.0f * sqrtf(2.0f * PX_TOLERANCE / pixelRadius);
    return static_cast<uint32_t>(ceilf(fabsf(arcAngle) / segmentAngle)) + 1;
}

bool gpuPointInTriangle(const Point& p, const Point& a, const Point& b, const Point& c)
{
    auto d1 = tvg::cross(p - a, p - b);
    auto d2 = tvg::cross(p - b, p - c);
    auto d3 = tvg::cross(p - c, p - a);
    auto hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    auto hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(hasNeg && hasPos);
}

RenderRegion gpuTransformBounds(const RenderRegion& bounds, const Matrix& matrix)
{
    if (bounds.invalid()) return bounds;

    auto lt = Point{float(bounds.min.x), float(bounds.min.y)} * matrix;
    auto lb = Point{float(bounds.min.x), float(bounds.max.y)} * matrix;
    auto rt = Point{float(bounds.max.x), float(bounds.min.y)} * matrix;
    auto rb = Point{float(bounds.max.x), float(bounds.max.y)} * matrix;

    auto min = tvg::min(tvg::min(lt, lb), tvg::min(rt, rb));
    auto max = tvg::max(tvg::max(lt, lb), tvg::max(rt, rb));

    return RenderRegion{{int32_t(floor(min.x)), int32_t(floor(min.y))}, {int32_t(ceil(max.x)), int32_t(ceil(max.y))}};
}

struct ThinPathTracker
{
    static constexpr int INLINE_PENDING_CAP = 8;

    Point preAxisPoints[INLINE_PENDING_CAP];
    Array<Point> preAxisOverflow;
    uint32_t preAxisCount = 0;
    Point axisStart{};
    Point axisVec{};
    float axisLen = 0.0f;
    float axisLenInv = 0.0f;
    float axisLenSqInv = 0.0f;
    float minT = 0.0f;
    float maxT = 0.0f;
    float minDist = 0.0f;
    float maxDist = 0.0f;
    bool ready = false;
    bool candidate = true;

    void disable()
    {
        candidate = false;
        ready = false;
        clearPreAxisPoints();
    }

    void remember(const Point& pt)
    {
        if (!candidate || ready) return;
        if (preAxisCount < INLINE_PENDING_CAP) preAxisPoints[preAxisCount++] = pt;
        else preAxisOverflow.push(pt);
    }

    void trackLine(const Point& start, const Point& end, bool closed)
    {
        if (!candidate) return;
        if (!ready) {
            if (closed) remember(start);
            else initAxis(start, end);
            return;
        }
        update(end);
    }

    void trackClosedCubic(const Point& start, const Point& ctrl1, const Point& ctrl2, const Point& end)
    {
        if (!candidate) return;
        if (!ready) {
            remember(start);
            remember(ctrl1);
            remember(ctrl2);
            return;
        }
        update(ctrl1);
        update(ctrl2);
        update(end);
    }

    void trackFlatCubic(const Point& start, const Point& ctrl1, const Point& ctrl2, const Point& end)
    {
        if (!candidate) return;
        if (!ready) initAxis(start, end);
        else update(end);
        update(ctrl1);
        update(ctrl2);
    }

    void trackClose(const Point& start, const Point& end, bool closed)
    {
        if (!candidate) return;
        if (!ready) {
            if (closed) remember(start);
            else initAxis(start, end);
            return;
        }
        update(end);
    }

    // Even a thin candidate should be skipped if its strongest local SW AA coverage stays sub-quantum.
    bool tooThin() const
    {
        auto thinSpan = (maxT - minT) * axisLen;
        auto thinThickness = maxDist - minDist;
        auto localVisibleArea = thinThickness * ((thinSpan < MAX_PIXEL_SPAN) ? thinSpan : MAX_PIXEL_SPAN);
        return localVisibleArea < AA_COVERAGE_QUANTUM;
    }

    void clearPreAxisPoints()
    {
        preAxisCount = 0;
        preAxisOverflow.clear();
    }

    void initAxis(const Point& start, const Point& end)
    {
        auto vec = end - start;
        auto vecLenSq = dot(vec, vec);

        axisStart = start;
        axisVec = vec;
        axisLen = sqrtf(vecLenSq);
        axisLenInv = 1.0f / axisLen;
        axisLenSqInv = 1.0f / vecLenSq;
        minT = 0.0f;
        maxT = 1.0f;
        minDist = 0.0f;
        maxDist = 0.0f;
        ready = true;
        flushPreAxisPoints();
    }

    void flushPreAxisPoints()
    {
        for (uint32_t i = 0; i < preAxisCount; ++i) {
            update(preAxisPoints[i]);
            if (!candidate) {
                clearPreAxisPoints();
                return;
            }
        }
        ARRAY_FOREACH(pt, preAxisOverflow)
        {
            update(*pt);
            if (!candidate) break;
        }
        clearPreAxisPoints();
    }

    void update(const Point& pt)
    {
        auto offset = pt - axisStart;
        auto signedDist = cross(axisVec, offset) * axisLenInv;
        if (fabsf(signedDist) > PATH_OPT_PX_TOLERANCE) {
            disable();
            return;
        }

        auto t = dot(offset, axisVec) * axisLenSqInv;
        if (t < minT) minT = t;
        if (t > maxT) maxT = t;
        if (signedDist < minDist) minDist = signedDist;
        if (signedDist > maxDist) maxDist = signedDist;
    }
};

// Simplify in transformed space and optionally mirror the same decisions into a
// local-coordinate path for stroke tessellation.
void gpuOptimize(const RenderPath& in, GpuOptimizeResult& result, const Matrix& matrix)
{
    result.thin = false;
    result.skipFill = false;
    if (!result.transformed) return;

    auto& out = *result.transformed;
    out.clear();

    auto localOut = result.local;
    if (localOut) localOut->clear();

    if (in.empty()) return;

    out.cmds.reserve(in.cmds.count);
    out.pts.reserve(in.pts.count);

    if (localOut) {
        localOut->cmds.reserve(in.cmds.count);
        localOut->pts.reserve(in.pts.count);
    }

    const auto* pts = in.pts.data;

    Point lastOutT{};
    Point lastInT{};
    Point subpathStartT{};
    auto drawableSubpathCnt = 0u;
    auto subpathOpen = false;
    auto subpathHasSegment = false;
    ThinPathTracker thinTracker;

    auto finalizeSubpath = [&]() {
        if (!subpathHasSegment) return;
        ++drawableSubpathCnt;
        if (drawableSubpathCnt > 1) thinTracker.disable();
        subpathHasSegment = false;
    };

    // vecLen is guaranteed to be non-zero since closed points are already merged
    auto point2Line = [](const Point& point, const Point& start, const Point& vec, float vecLen, float& maxDist, float& minT, float& maxT) {
        Point offset = point - start;
        auto dist = fabsf(tvg::cross(vec, offset)) / vecLen;
        if (dist > maxDist) maxDist = dist;
        auto t = tvg::dot(offset, vec) / (vecLen * vecLen);
        if (t < minT) minT = t;
        if (t > maxT) maxT = t;
    };

    auto validateCubic = [&point2Line](const Point& start, const Point& ctrl1, const Point& ctrl2, const Point& end, float& maxDist, float& minT, float& maxT, float& vecLen) {
        auto vec = end - start;
        vecLen = sqrtf(vec.x * vec.x + vec.y * vec.y);
        maxDist = 0.0f;
        minT = FLT_MAX;
        maxT = FLT_MIN;
        point2Line(ctrl1, start, vec, vecLen, maxDist, minT, maxT);
        point2Line(ctrl2, start, vec, vecLen, maxDist, minT, maxT);
    };

    auto addLineCmd = [&](const Point& local, const Point& transformed) {
        out.lineTo(transformed);
        if (localOut) localOut->lineTo(local);
        lastOutT = transformed;
    };

    auto processCubicTo = [&](const Point* cubicPts, const Point& startOutT, const Point& startInT, Point& endT) {
        auto ctrl1T = cubicPts[0] * matrix;
        auto ctrl2T = cubicPts[1] * matrix;
        endT = cubicPts[2] * matrix;

        auto trackThinCubic = [&](const Point& startT) {
            if (tvg::closed(startT, endT, PATH_OPT_PX_TOLERANCE)) {
                thinTracker.trackClosedCubic(startT, ctrl1T, ctrl2T, endT);
                return;
            }
            float maxDist, minT, maxT, vecLen;
            validateCubic(startT, ctrl1T, ctrl2T, endT, maxDist, minT, maxT, vecLen);
            auto flat = (maxDist <= PATH_OPT_PX_TOLERANCE);
            auto tEps = PATH_OPT_PX_TOLERANCE / vecLen;
            auto inSpan = (minT >= -tEps) && (maxT <= 1.0f + tEps);
            if (flat && inSpan) thinTracker.trackFlatCubic(startT, ctrl1T, ctrl2T, endT);
            else thinTracker.disable();
        };
        trackThinCubic(startInT);

        if (tvg::closed(startOutT, endT, PATH_OPT_PX_TOLERANCE)) return;

        float maxDist, minT, maxT, vecLen;
        validateCubic(startOutT, ctrl1T, ctrl2T, endT, maxDist, minT, maxT, vecLen);
        auto flat = (maxDist <= PATH_OPT_PX_TOLERANCE);
        auto tEps = PATH_OPT_PX_TOLERANCE / vecLen;
        auto inSpan = (minT >= -tEps) && (maxT <= 1.0f + tEps);
        if (flat && inSpan) {
            subpathHasSegment = true;
            addLineCmd(cubicPts[2], endT);
        } else {
            out.cubicTo(ctrl1T, ctrl2T, endT);
            if (localOut) localOut->cubicTo(cubicPts[0], cubicPts[1], cubicPts[2]);
            lastOutT = endT;
            subpathHasSegment = true;
            thinTracker.disable();
        }
    };

    ARRAY_FOREACH(cmd, in.cmds) {
        switch (*cmd) {
            case PathCommand::MoveTo: {
                finalizeSubpath();
                auto pt = *pts;
                auto ptT = pt * matrix;
                out.cmds.push(PathCommand::MoveTo);
                out.pts.push(ptT);
                if (localOut) {
                    localOut->cmds.push(PathCommand::MoveTo);
                    localOut->pts.push(pt);
                }
                lastOutT = ptT;
                lastInT = ptT;
                subpathStartT = ptT;
                subpathOpen = true;
                pts++;
                break;
            }
            case PathCommand::LineTo: {
                auto startInT = lastInT;
                auto pt = *pts;
                auto ptT = pt * matrix;
                auto closedIn = tvg::closed(startInT, ptT, PATH_OPT_PX_TOLERANCE);
                if (!closedIn) subpathHasSegment = true;
                thinTracker.trackLine(startInT, ptT, closedIn);
                lastInT = ptT;
                if (tvg::closed(lastOutT, ptT, PATH_OPT_PX_TOLERANCE)) {
                    pts++;
                    break;
                }
                addLineCmd(pt, ptT);
                pts++;
                break;
            }
            case PathCommand::CubicTo: {
                Point endT{};
                processCubicTo(pts, lastOutT, lastInT, endT);
                lastInT = endT;
                pts += 3;
                break;
            }
            case PathCommand::Close: {
                if (subpathOpen) {
                    auto closedIn = tvg::closed(lastInT, subpathStartT, PATH_OPT_PX_TOLERANCE);
                    if (!closedIn) subpathHasSegment = true;
                    thinTracker.trackClose(lastInT, subpathStartT, closedIn);
                }
                out.cmds.push(PathCommand::Close);
                if (localOut) localOut->cmds.push(PathCommand::Close);
                lastOutT = subpathStartT;
                lastInT = subpathStartT;
                break;
            }
            default: break;
        }
    }
    finalizeSubpath();
    // thin means "use thin fill fallback", not just "the geometry is narrow".
    result.thin = thinTracker.candidate && thinTracker.ready && (drawableSubpathCnt == 1);
    if (result.thin && thinTracker.tooThin()) {
        // Too thin for fallback: keep the path for strokes, but skip the fill.
        result.thin = false;
        result.skipFill = true;
    }
}

bool gpuEdgesCross(const Point& p0, const Point& p1, const Point& p2, const Point& p3)
{
    auto orientSign = [](const Point& a, const Point& b, const Point& c) {
        auto value = cross(b - a, c - a);
        if (zero(value)) return int8_t{0};
        return (value > 0.0f) ? int8_t{1} : int8_t{-1};
    };

    auto straddlesLine = [&](const Point& a, const Point& b, const Point& c, const Point& d) {
        return orientSign(a, b, c) * orientSign(a, b, d) < 0;
    };

    return straddlesLine(p0, p1, p2, p3) && straddlesLine(p2, p3, p0, p1);
}

/************************************************************************/
/* GpuConvexProbe Class Implementation                                  */
/************************************************************************/

void GpuConvexProbe::addEdge(const Point& edge)
{
    if (zero(edge)) return;

    contourHasEdges = true;
    if (!convex) return;

    if (zero(firstEdge)) firstEdge = edge;

    updateDir(edge.x, prevXDir, xDirChanges);
    updateDir(edge.y, prevYDir, yDirChanges);
    if (!convex) return;

    if (zero(prevEdge)) {
        prevEdge = edge;
        return;
    }

    auto turn = cross(prevEdge, edge);
    if (zero(turn)) {
        if (dot(prevEdge, edge) < 0.0f && ++reversals > MaxCollinearReversals) convex = false;
    } else {
        auto sign = (turn > 0.0f) ? int8_t{1} : int8_t{-1};
        if (winding == 0) winding = sign;
        else if (sign != winding) convex = false;
    }

    prevEdge = edge;
}

void GpuConvexProbe::updateDir(float value, int8_t& prevDir, uint8_t& changes)
{
    int8_t dir = 0;
    if (!zero(value)) dir = (value > 0.0f) ? int8_t{1} : int8_t{-1};
    if (dir == 0 || !convex) return;

    if (prevDir != 0 && prevDir != dir && ++changes > MaxAxisDirChanges) {
        convex = false;
        return;
    }
    prevDir = dir;
}

/************************************************************************/
/* StrokeDashPath Class Implementation                                  */
/************************************************************************/

struct DashPatternState
{
    int32_t idx = 0;
    float offset = 0.0f;
    bool gap = false;
};

struct DashSubpathState
{
    bool closed = false;
    bool closeEndsAtStart = false;

    void reset()
    {
        closed = false;
        closeEndsAtStart = false;
    }
};

struct PieceRange
{
    uint32_t cmdBegin;
    uint32_t cmdEnd;
    uint32_t ptBegin;
    uint32_t ptEnd;
};

static inline DashPatternState dashPatternState(const RenderStroke::Dash& dash)
{
    DashPatternState state;
    state.offset = dash.offset;
    if (tvg::zero(dash.offset)) return state;

    auto length = (dash.count % 2) ? dash.length * 2 : dash.length;
    state.offset = fmodf(state.offset, length);
    if (state.offset < 0) state.offset += length;

    for (uint32_t i = 0; i < dash.count * (dash.count % 2 + 1); ++i, ++state.idx) {
        auto curPattern = dash.pattern[i % dash.count];
        if (state.offset < curPattern) break;
        state.offset -= curPattern;
        state.gap = !state.gap;
    }
    state.idx = state.idx % dash.count;
    return state;
}

static inline void appendDashedCommand(RenderPath& dst, const RenderPath& src, PathCommand cmd, uint32_t& ptIdx, bool skipMoveTo)
{
    switch (cmd) {
        case PathCommand::MoveTo: {
            auto pt = src.pts[ptIdx++];
            if (!skipMoveTo) dst.moveTo(pt);
            break;
        }
        case PathCommand::LineTo: {
            dst.lineTo(src.pts[ptIdx++]);
            break;
        }
        case PathCommand::CubicTo: {
            dst.cubicTo(src.pts[ptIdx], src.pts[ptIdx + 1], src.pts[ptIdx + 2]);
            ptIdx += 3;
            break;
        }
        default: break;
    }
}

static inline void appendDashedPiece(RenderPath& dst, const RenderPath& src, const PieceRange& piece, bool skipMoveTo)
{
    auto ptIdx = piece.ptBegin;
    for (uint32_t i = piece.cmdBegin; i < piece.cmdEnd; ++i) {
        appendDashedCommand(dst, src, src.cmds[i], ptIdx, skipMoveTo && (i == piece.cmdBegin));
    }
}

static inline void collectDashedPieces(const RenderPath& subOut, Array<PieceRange>& pieces)
{
    auto ptIdx = 0u;
    auto pieceCmdBegin = UINT32_MAX;
    auto piecePtBegin = 0u;
    auto pieceHasDraw = false;

    for (uint32_t i = 0; i < subOut.cmds.count; ++i) {
        if (subOut.cmds[i] == PathCommand::MoveTo) {
            if (pieceCmdBegin != UINT32_MAX && pieceHasDraw) {
                pieces.push(PieceRange{pieceCmdBegin, i, piecePtBegin, ptIdx});
            }
            pieceCmdBegin = i;
            piecePtBegin = ptIdx;
            pieceHasDraw = false;
        } else {
            pieceHasDraw = true;
        }

        if (subOut.cmds[i] == PathCommand::CubicTo) ptIdx += 3;
        else if (subOut.cmds[i] != PathCommand::Close) ptIdx += 1;
    }

    if (pieceCmdBegin != UINT32_MAX && pieceHasDraw) {
        pieces.push(PieceRange{pieceCmdBegin, subOut.cmds.count, piecePtBegin, ptIdx});
    }
}

static inline void resetDashedSubpath(RenderPath& subOut, DashSubpathState& state)
{
    subOut.clear();
    state.reset();
}

static inline void appendDashedCommands(RenderPath& out, const RenderPath& subOut)
{
    auto ptIdx = 0u;
    ARRAY_FOREACH(cmd, subOut.cmds)
    {
        appendDashedCommand(out, subOut, *cmd, ptIdx, false);
    }
}

static inline bool prepareDashedPieces(RenderPath& subOut, DashSubpathState& state, Array<PieceRange>& pieces)
{
    pieces.clear();
    if (subOut.cmds.empty()) {
        resetDashedSubpath(subOut, state);
        return false;
    }

    pieces.reserve(subOut.cmds.count);
    collectDashedPieces(subOut, pieces);
    if (!pieces.empty()) return true;

    resetDashedSubpath(subOut, state);
    return false;
}

static inline bool closedDashedContour(const RenderPath& subOut, const Array<PieceRange>& pieces, const Point& mappedStart)
{
    auto firstStartsAtClose = tvg::closed(subOut.pts[pieces[0].ptBegin], mappedStart, DASH_ENDPOINT_TOLERANCE);
    auto lastEndsAtClose = tvg::closed(subOut.pts[pieces.last().ptEnd - 1], mappedStart, DASH_ENDPOINT_TOLERANCE);
    return firstStartsAtClose && lastEndsAtClose;
}

static inline bool appendClosedDashedSubpath(RenderPath& out, const RenderPath& subOut, const Array<PieceRange>& pieces, const Point& mappedStart, const DashSubpathState& state)
{
    if (!state.closed) return false;

    auto wrapsStart = closedDashedContour(subOut, pieces, mappedStart);
    if (!wrapsStart) return false;

    if (pieces.count == 1) {
        appendDashedPiece(out, subOut, pieces[0], false);
        out.close();
        return true;
    }

    if (!state.closeEndsAtStart) return false;

    // Closed dash wraps across the original start point. Emit it as one contour so the stroker can join there.
    appendDashedPiece(out, subOut, pieces.last(), false);
    appendDashedPiece(out, subOut, pieces[0], true);
    for (uint32_t i = 1; i + 1 < pieces.count; ++i) {
        appendDashedPiece(out, subOut, pieces[i], false);
    }
    return true;
}

static inline void appendDashedSubpath(RenderPath& out, RenderPath& subOut, const Point& mappedStart, DashSubpathState& state, Array<PieceRange>& pieces)
{
    if (!prepareDashedPieces(subOut, state, pieces)) return;

    if (!appendClosedDashedSubpath(out, subOut, pieces, mappedStart, state)) {
        appendDashedCommands(out, subOut);
    }

    resetDashedSubpath(subOut, state);
}

struct StrokeDashPath
{
    StrokeDashPath(RenderStroke::Dash dash) :
        dash(dash) {}
    bool gen(const RenderPath& in, RenderPath& out, bool drawPoint, const Matrix* transform = nullptr);

private:
    void beginSubpath(const Point& start, const DashPatternState& state);
    void closeSubpath(RenderPath& subOut, const Point& start, bool allowDot, DashSubpathState& state);
    void processCommand(PathCommand cmd, const Point*& pts, const DashPatternState& initialState, Point& start,
                        RenderPath& subOut, RenderPath& out, DashSubpathState& state,
                        Array<PieceRange>& pieces, bool allowDot);
    void lineTo(RenderPath& out, const Point& pt, bool drawPoint);
    void cubicTo(RenderPath& out, const Point& pt1, const Point& pt2, const Point& pt3, bool drawPoint);
    void point(RenderPath& out, const Point& p);
    Point map(const Point& pt) const { return applyTransform ? pt * (*transform) : pt; }
    template<typename Segment, typename LengthFn, typename SplitFn, typename DrawFn, typename PointFn>
    void segment(Segment seg, float len, RenderPath& out, bool allowDot, LengthFn lengthFn, SplitFn splitFn, DrawFn drawFn, PointFn getStartPt, const Point& endPos);

    RenderStroke::Dash dash;
    float curLen = 0.0f;
    int32_t curIdx = 0;
    Point curPos{};
    bool opGap = false;
    bool move = true;
    const Matrix* transform = nullptr;
    bool applyTransform = false;
};

template<typename Segment, typename LengthFn, typename SplitFn, typename DrawFn, typename PointFn>
void StrokeDashPath::segment(Segment seg, float len, RenderPath& out, bool allowDot, LengthFn lengthFn, SplitFn splitFn, DrawFn drawFn, PointFn getStartPt, const Point& end)
{
    static constexpr float MIN_CURR_LEN_THRESHOLD = 0.1f;

    if (tvg::zero(len)) {
        out.moveTo(map(curPos));
    } else if (len <= curLen) {
        curLen -= len;
        if (!opGap) {
            if (move) {
                out.moveTo(map(curPos));
                move = false;
            }
            drawFn(seg);
        }
    } else {
        Segment left, right;
        while (len - curLen > DASH_PATTERN_THRESHOLD) {
            if (curLen > 0.0f) {
                splitFn(seg, curLen, left, right);
                len -= curLen;
                if (!opGap) {
                    if (move || dash.pattern[curIdx] - curLen < FLOAT_EPSILON) {
                        out.moveTo(map(getStartPt(left)));
                        move = false;
                    }
                    drawFn(left);
                }
            } else {
                if (allowDot && !opGap) point(out, getStartPt(seg));
                right = seg;
            }

            curIdx = (curIdx + 1) % dash.count;
            curLen = dash.pattern[curIdx];
            opGap = !opGap;
            seg = right;
            curPos = getStartPt(seg);
            move = true;
        }
        curLen -= len;
        if (!opGap) {
            if (move) {
                out.moveTo(map(getStartPt(seg)));
                move = false;
            }
            drawFn(seg);
        }
        if (curLen < MIN_CURR_LEN_THRESHOLD) {
            curIdx = (curIdx + 1) % dash.count;
            curLen = dash.pattern[curIdx];
            opGap = !opGap;
        }
    }
    curPos = end;
}

// allowDot: zero length segment with non-butt cap still should be rendered as a point - only the caps are visible
bool StrokeDashPath::gen(const RenderPath& in, RenderPath& out, bool allowDot, const Matrix* transform)
{
    this->transform = transform;
    this->applyTransform = (transform && !tvg::identity(transform));

    auto initialState = dashPatternState(dash);

    const auto* pts = in.pts.data;
    Point start{};
    RenderPath subOut;
    DashSubpathState subpathState;
    Array<PieceRange> pieces;

    ARRAY_FOREACH(cmd, in.cmds)
    {
        processCommand(*cmd, pts, initialState, start, subOut, out, subpathState, pieces, allowDot);
    }
    appendDashedSubpath(out, subOut, map(start), subpathState, pieces);
    return true;
}

void StrokeDashPath::beginSubpath(const Point& start, const DashPatternState& state)
{
    curIdx = state.idx;
    curLen = dash.pattern[state.idx] - state.offset;
    opGap = state.gap;
    move = true;
    curPos = start;
}

void StrokeDashPath::closeSubpath(RenderPath& subOut, const Point& start, bool allowDot, DashSubpathState& state)
{
    auto prevPtCount = subOut.pts.count;
    lineTo(subOut, start, allowDot);
    state.closed = true;
    // Rejoin decisions use emitted piece endpoints; this only tracks whether Close reached start.
    state.closeEndsAtStart = (subOut.pts.count > prevPtCount) && tvg::closed(subOut.pts.last(), map(start), DASH_ENDPOINT_TOLERANCE);
}

void StrokeDashPath::processCommand(PathCommand cmd, const Point*& pts, const DashPatternState& initialState, Point& start,
                                    RenderPath& subOut, RenderPath& out, DashSubpathState& state,
                                    Array<PieceRange>& pieces, bool allowDot)
{
    switch (cmd) {
        case PathCommand::Close: {
            closeSubpath(subOut, start, allowDot, state);
            break;
        }
        case PathCommand::MoveTo: {
            appendDashedSubpath(out, subOut, map(start), state, pieces);
            start = *pts++;
            beginSubpath(start, initialState);
            break;
        }
        case PathCommand::LineTo: {
            lineTo(subOut, *pts++, allowDot);
            break;
        }
        case PathCommand::CubicTo: {
            cubicTo(subOut, pts[0], pts[1], pts[2], allowDot);
            pts += 3;
            break;
        }
        default: break;
    }
}

void StrokeDashPath::point(RenderPath& out, const Point& p)
{
    if (move || dash.pattern[curIdx] < FLOAT_EPSILON) {
        out.moveTo(map(p));
        move = false;
    }
    out.lineTo(map(p));
}

void StrokeDashPath::lineTo(RenderPath& out, const Point& to, bool allowDot)
{
    Line line = {curPos, to};
    auto len = length(to - curPos);
    segment<Line>(line, len, out, allowDot, [](const Line& l) { return length(l.pt2 - l.pt1); }, [](const Line& l, float len, Line& left, Line& right) { l.split(len, left, right); }, [&](const Line& l) { out.lineTo(map(l.pt2)); }, [](const Line& l) { return l.pt1; }, to);
}

void StrokeDashPath::cubicTo(RenderPath& out, const Point& cnt1, const Point& cnt2, const Point& end, bool allowDot)
{
    Bezier curve = {curPos, cnt1, cnt2, end};
    auto len = curve.length();
    segment<Bezier>(curve, len, out, allowDot, [](const Bezier& b) { return b.length(); }, [](const Bezier& b, float len, Bezier& left, Bezier& right) { b.split(len, left, right); }, [&](const Bezier& b) { out.cubicTo(map(b.ctrl1), map(b.ctrl2), map(b.end)); }, [](const Bezier& b) { return b.start; }, end);
}

bool gpuStrokeDash(const RenderShape& rs, RenderPath& out, const Matrix* transform)
{
    if (!rs.stroke || rs.stroke->dash.count == 0 || rs.stroke->dash.length < DASH_PATTERN_THRESHOLD) return false;

    out.cmds.reserve(20 * rs.path.cmds.count);
    out.pts.reserve(20 * rs.path.pts.count);

    StrokeDashPath dash(rs.stroke->dash);
    auto allowDot = rs.stroke->cap != StrokeCap::Butt;

    if (rs.trimpath()) {
        RenderPath tpath;
        if (rs.stroke->trim.trim(rs.path, tpath)) return dash.gen(tpath, out, allowDot, transform);
        else return false;
    }
    return dash.gen(rs.path, out, allowDot, transform);
}

}  // namespace tvg
