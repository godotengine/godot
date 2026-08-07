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
// Keep this aligned with the SW rasterizer's AA coverage precision.
// tvgRender.h::MULTIPLY() and the SW coverage paths use an 8-bit coverage model.
constexpr uint32_t SW_AA_COVERAGE_BITS = 8;
constexpr auto AA_COVERAGE_QUANTUM = 1.0f / static_cast<float>(1u << SW_AA_COVERAGE_BITS);
constexpr auto MAX_PIXEL_SPAN = 1.41421356237f;

struct ThinPathTracker
{
    enum
    {
        INLINE_PENDING_CAP = 8
    };

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

// Simplify the transformed path and classify thin-fill fallback / skip-fill cases.
void gpuOptimize(const RenderPath& in, RenderPath& out, const Matrix& matrix, bool& thin, bool& skipFill)
{
    thin = false;
    skipFill = false;
    if (in.empty()) return;

    out.cmds.clear();
    out.pts.clear();
    out.cmds.reserve(in.cmds.count);
    out.pts.reserve(in.pts.count);

    auto cmds = in.cmds.data;
    auto cmdCnt = in.cmds.count;
    auto pts = in.pts.data;

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

    auto addLineCmd = [&](const Point& ptT) {
        out.cmds.push(PathCommand::LineTo);
        out.pts.push(ptT);
        lastOutT = ptT;
    };

    auto processCubicTo = [&](const Point* cubicPts, const Point& startOutT, const Point& startInT, Point& endT) {
        auto ctrl1T = cubicPts[0] * matrix;
        auto ctrl2T = cubicPts[1] * matrix;
        endT = cubicPts[2] * matrix;

        auto trackThinCubic = [&](const Point& startT) {
            auto closed = tvg::closed(startT, endT, PATH_OPT_PX_TOLERANCE);
            if (closed) {
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

        auto closed = tvg::closed(startOutT, endT, PATH_OPT_PX_TOLERANCE);
        if (closed) return;

        float maxDist, minT, maxT, vecLen;
        validateCubic(startOutT, ctrl1T, ctrl2T, endT, maxDist, minT, maxT, vecLen);
        auto flat = (maxDist <= PATH_OPT_PX_TOLERANCE);
        auto tEps = PATH_OPT_PX_TOLERANCE / vecLen;
        auto inSpan = (minT >= -tEps) && (maxT <= 1.0f + tEps);
        if (flat && inSpan) {
            subpathHasSegment = true;
            addLineCmd(endT);
        } else {
            out.cmds.push(PathCommand::CubicTo);
            out.pts.push(ctrl1T);
            out.pts.push(ctrl2T);
            out.pts.push(endT);
            lastOutT = endT;
            subpathHasSegment = true;
            thinTracker.disable();
        }
    };

    for (uint32_t i = 0; i < cmdCnt; i++) {
        switch (cmds[i]) {
            case PathCommand::MoveTo: {
                finalizeSubpath();
                auto ptT = (*pts) * matrix;
                out.cmds.push(PathCommand::MoveTo);
                out.pts.push(ptT);
                lastOutT = ptT;
                lastInT = ptT;
                subpathStartT = ptT;
                subpathOpen = true;
                pts++;
                break;
            }
            case PathCommand::LineTo: {
                auto startInT = lastInT;
                auto ptT = (*pts) * matrix;
                auto closedIn = tvg::closed(startInT, ptT, PATH_OPT_PX_TOLERANCE);
                if (!closedIn) subpathHasSegment = true;
                thinTracker.trackLine(startInT, ptT, closedIn);
                lastInT = ptT;
                if (tvg::closed(lastOutT, ptT, PATH_OPT_PX_TOLERANCE)) {
                    pts++;
                    break;
                }
                addLineCmd(ptT);
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
                lastOutT = subpathStartT;
                lastInT = subpathStartT;
                break;
            }
            default: break;
        }
    }
    finalizeSubpath();
    // thin means "use thin fill fallback", not just "the geometry is narrow".
    thin = thinTracker.candidate && thinTracker.ready && (drawableSubpathCnt == 1);
    if (thin && thinTracker.tooThin()) {
        // Too thin for fallback: keep the path for strokes, but skip the fill.
        thin = false;
        skipFill = true;
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

struct StrokeDashPath
{
    StrokeDashPath(RenderStroke::Dash dash) :
        dash(dash) {}
    bool gen(const RenderPath& in, RenderPath& out, bool drawPoint, const Matrix* transform = nullptr);

private:
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

    int32_t idx = 0;
    auto offset = dash.offset;
    auto gap = false;
    if (!tvg::zero(dash.offset)) {
        auto length = (dash.count % 2) ? dash.length * 2 : dash.length;
        offset = fmodf(offset, length);
        if (offset < 0) offset += length;

        for (uint32_t i = 0; i < dash.count * (dash.count % 2 + 1); ++i, ++idx) {
            auto curPattern = dash.pattern[i % dash.count];
            if (offset < curPattern) break;
            offset -= curPattern;
            gap = !gap;
        }
        idx = idx % dash.count;
    }

    auto pts = in.pts.data;
    Point start{};

    ARRAY_FOREACH(cmd, in.cmds)
    {
        switch (*cmd) {
            case PathCommand::Close: {
                lineTo(out, start, allowDot);
                break;
            }
            case PathCommand::MoveTo: {
                // reset the dash state
                curIdx = idx;
                curLen = dash.pattern[idx] - offset;
                opGap = gap;
                move = true;
                start = curPos = *pts;
                pts++;
                break;
            }
            case PathCommand::LineTo: {
                lineTo(out, *pts, allowDot);
                pts++;
                break;
            }
            case PathCommand::CubicTo: {
                cubicTo(out, pts[0], pts[1], pts[2], allowDot);
                pts += 3;
                break;
            }
            default: break;
        }
    }
    return true;
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
