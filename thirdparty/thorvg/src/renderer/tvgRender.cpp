/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

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

#include "tvgMath.h"
#include "tvgRender.h"

/************************************************************************/
/* RenderMethod Class Implementation                                    */
/************************************************************************/

uint32_t RenderMethod::ref()
{
    ScopedLock lock(key);
    return (++refCnt);
}


uint32_t RenderMethod::unref()
{
    ScopedLock lock(key);
    return (--refCnt);
}


RenderRegion RenderMethod::viewport()
{
    return vport;
}


bool RenderMethod::viewport(const RenderRegion& vp)
{
    vport = vp;
    return true;
}


/************************************************************************/
/* RenderPath Class Implementation                                      */
/************************************************************************/

bool RenderPath::bounds(const Matrix* m, BBox& box)
{
    if (cmds.empty() || cmds.first() == PathCommand::CubicTo) return false;

    auto pt = pts.begin();
    auto cmd = cmds.begin();

    auto assign = [&](const Point& pt, BBox& box) -> void {
        if (pt.x < box.min.x) box.min.x = pt.x;
        if (pt.y < box.min.y) box.min.y = pt.y;
        if (pt.x > box.max.x) box.max.x = pt.x;
        if (pt.y > box.max.y) box.max.y = pt.y;
    };

    while (cmd < cmds.end()) {
        switch (*cmd) {
            case PathCommand::MoveTo: {
                //skip the invalid assignments
                if (cmd + 1 < cmds.end()) {
                    auto next = *(cmd + 1);
                    if (next == PathCommand::LineTo || next == PathCommand::CubicTo) {
                        assign(*pt * m, box);
                    }
                }
                ++pt;
                break;
            }
            case PathCommand::LineTo: {
                assign(*pt * m, box);
                ++pt;
                break;
            }
            case PathCommand::CubicTo: {
                Bezier::bounds(box, pt[-1] * m, pt[0] * m, pt[1] * m, pt[2] * m);
                pt += 3;
                break;
            }
            default: break;
        }
        ++cmd;
    }
    return true;
}


void RenderPath::optimizeGL(RenderPath& out, const Matrix& matrix) const
{
#if defined(THORVG_GL_RASTER_SUPPORT)
    static constexpr auto PX_TOLERANCE = 0.25f;

    if (empty()) return;

    out.cmds.clear();
    out.pts.clear();
    out.cmds.reserve(cmds.count);
    out.pts.reserve(pts.count);

    auto cmds = this->cmds.data;
    auto cmdCnt = this->cmds.count;
    auto pts = this->pts.data;

    Point lastOutT, prevOutT;   // The suffix "T" indicates that the point is transformed.
    uint32_t prevIdx = 0;
    uint32_t prevPrevIdx = 0;
    auto hasPrevPrev = false;

    //vecLen is guaranteed to be non-zero since closed points are already merged
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

    auto point2LineSimple = [](const Point& point, const Point& start, const Point& end, float& dist, float& t, float& vecLen) {
        auto vec = end - start;
        auto vecLenSq = vec.x * vec.x + vec.y * vec.y;
        vecLen = sqrtf(vecLenSq);
        Point offset = point - start;
        dist = fabsf(tvg::cross(vec, offset)) / vecLen;
        t = tvg::dot(offset, vec) / vecLenSq;
    };

    auto addLineCmd = [&](const Point& ptT) {
        out.cmds.push(PathCommand::LineTo);
        out.pts.push(ptT);
        prevOutT = lastOutT;
        lastOutT = ptT;
        prevPrevIdx = prevIdx;
        prevIdx = out.pts.count - 1;
        hasPrevPrev = true;
    };

    auto processLineCollinear = [&](const Point& startT, const Point& ptT) {
        if (!hasPrevPrev || out.pts.count <= 1) {
            addLineCmd(ptT);
            return;
        }

        float dist, t, vecLen;
        point2LineSimple(ptT, prevOutT, startT, dist, t, vecLen);
        if (dist > PX_TOLERANCE) {
            addLineCmd(ptT);
            return;
        }

        auto tEps = PX_TOLERANCE / vecLen;
        if (t <= -tEps) {
            out.pts[prevPrevIdx] = ptT;
            lastOutT = ptT;
        } else if (t >= 1.0f - tEps) {
            out.pts[prevIdx] = ptT;
            lastOutT = ptT;
        }
    };

    auto processCubicTo = [&](const Point* cubicPts, const Point& startT) {
        auto ctrl1T = cubicPts[0] * matrix;
        auto ctrl2T = cubicPts[1] * matrix;
        auto endT = cubicPts[2] * matrix;
        if (tvg::closed(startT, endT, PX_TOLERANCE)) return;
        float maxDist, minT, maxT, vecLen;
        validateCubic(startT, ctrl1T, ctrl2T, endT, maxDist, minT, maxT, vecLen);
        auto flat = (maxDist <= PX_TOLERANCE);
        auto tEps = PX_TOLERANCE / vecLen;
        auto inSpan = (minT >= -tEps) && (maxT <= 1.0f + tEps);
        if (flat && inSpan) {
            processLineCollinear(startT, endT);
        } else {
            out.cmds.push(PathCommand::CubicTo);
            out.pts.push(ctrl1T);
            out.pts.push(ctrl2T);
            out.pts.push(endT);
            prevOutT = lastOutT;
            lastOutT = endT;
            prevPrevIdx = prevIdx;
            prevIdx = out.pts.count - 1;
            hasPrevPrev = true;
        }
    };

    for (uint32_t i = 0; i < cmdCnt; i++) {
        switch (cmds[i]) {
            case PathCommand::MoveTo: {
                auto ptT = (*pts) * matrix;
                out.cmds.push(PathCommand::MoveTo);
                out.pts.push(ptT);
                lastOutT = ptT;
                prevIdx = out.pts.count - 1;
                hasPrevPrev = false;
                pts++;
                break;
            }
            case PathCommand::LineTo: {
                auto startT = lastOutT;
                auto ptT = (*pts) * matrix;
                if (tvg::closed(startT, ptT, PX_TOLERANCE)) {
                    pts++;
                    break;
                }
                processLineCollinear(startT, ptT);
                pts++;
                break;
            }
            case PathCommand::CubicTo: {
                processCubicTo(pts, lastOutT);
                pts += 3;
                break;
            }
            case PathCommand::Close: {
                out.cmds.push(PathCommand::Close);
                hasPrevPrev = false;
                break;
            }
            default: break;
        }
    }
#else
    TVGLOG("RENDERER", "RenderPath transformed optimization is disabled");
#endif
}


void RenderPath::optimizeWG(RenderPath& out, const Matrix& matrix) const
{
#if defined(THORVG_WG_RASTER_SUPPORT)
    static constexpr auto PX_TOLERANCE = 0.25f;

    if (empty()) return;

    out.cmds.clear();
    out.pts.clear();
    out.cmds.reserve(cmds.count);
    out.pts.reserve(pts.count);

    auto cmds = this->cmds.data;
    auto cmdCnt = this->cmds.count;
    auto pts = this->pts.data;

    Point lastOutT, prevOutT;   // The suffix "T" indicates that the point is transformed.
    uint32_t prevIdx = 0;
    uint32_t prevPrevIdx = 0;
    auto hasPrevPrev = false;

    //vecLen is guaranteed to be non-zero since closed points are already merged
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

    auto point2LineSimple = [](const Point& point, const Point& start, const Point& end, float& dist, float& t, float& vecLen) {
        auto vec = end - start;
        auto vecLenSq = vec.x * vec.x + vec.y * vec.y;
        vecLen = sqrtf(vecLenSq);
        Point offset = point - start;
        dist = fabsf(tvg::cross(vec, offset)) / vecLen;
        t = tvg::dot(offset, vec) / vecLenSq;
    };

    auto addLineCmd = [&](const Point& pt, const Point& ptT) {
        out.cmds.push(PathCommand::LineTo);
        out.pts.push(pt);
        prevOutT = lastOutT;
        lastOutT = ptT;
        prevPrevIdx = prevIdx;
        prevIdx = out.pts.count - 1;
        hasPrevPrev = true;
    };

    auto processLineCollinear = [&](const Point& startT, const Point& pt, const Point& ptT) {
        if (!hasPrevPrev || out.pts.count <= 1) {
            addLineCmd(pt, ptT);
            return;
        }

        float dist, t, vecLen;
        point2LineSimple(ptT, prevOutT, startT, dist, t, vecLen);
        if (dist > PX_TOLERANCE) {
            addLineCmd(pt, ptT);
            return;
        }

        auto tEps = PX_TOLERANCE / vecLen;
        if (t <= -tEps) {
            out.pts[prevPrevIdx] = pt;
            lastOutT = ptT;
        } else if (t >= 1.0f - tEps) {
            out.pts[prevIdx] = pt;
            lastOutT = ptT;
        }
    };

    auto processCubicTo = [&](const Point* cubicPts, const Point& startT) {
        auto endT = cubicPts[2] * matrix;
        if (tvg::closed(startT, endT, PX_TOLERANCE)) return;
        float maxDist, minT, maxT, vecLen;
        validateCubic(startT, cubicPts[0] * matrix, cubicPts[1] * matrix, endT, maxDist, minT, maxT, vecLen);
        auto flat = (maxDist <= PX_TOLERANCE);
        auto tEps = PX_TOLERANCE / vecLen;
        auto inSpan = (minT >= -tEps) && (maxT <= 1.0f + tEps);
        if (flat && inSpan) {
            processLineCollinear(startT, cubicPts[2], endT);
        } else {
            out.cmds.push(PathCommand::CubicTo);
            out.pts.push(cubicPts[0]);
            out.pts.push(cubicPts[1]);
            out.pts.push(cubicPts[2]);
            prevOutT = lastOutT;
            lastOutT = endT;
            prevPrevIdx = prevIdx;
            prevIdx = out.pts.count - 1;
            hasPrevPrev = true;
        }
    };

    for (uint32_t i = 0; i < cmdCnt; i++) {
        switch (cmds[i]) {
            case PathCommand::MoveTo: {
                out.cmds.push(PathCommand::MoveTo);
                out.pts.push(*pts);
                lastOutT = *pts * matrix;
                prevIdx = out.pts.count - 1;
                hasPrevPrev = false;
                pts++;
                break;
            }
            case PathCommand::LineTo: {
                auto startT = lastOutT;
                auto ptT = (*pts) * matrix;
                if (tvg::closed(startT, ptT, PX_TOLERANCE)) {
                    pts++;
                    break;
                }
                processLineCollinear(startT, *pts, ptT);
                pts++;
                break;
            }
            case PathCommand::CubicTo: {
                processCubicTo(pts, lastOutT);
                pts += 3;
                break;
            }
            case PathCommand::Close: {
                out.cmds.push(PathCommand::Close);
                hasPrevPrev = false;
                break;
            }
            default: break;
        }
    }
#else
    TVGLOG("RENDERER", "RenderPath Optimization is disabled");
#endif
}


/************************************************************************/
/* RenderRegion Class Implementation                                    */
/************************************************************************/


void RenderRegion::intersect(const RenderRegion& rhs)
{
    if (min.x < rhs.min.x) min.x = rhs.min.x;
    if (min.y < rhs.min.y) min.y = rhs.min.y;
    if (max.x > rhs.max.x) max.x = rhs.max.x;
    if (max.y > rhs.max.y) max.y = rhs.max.y;

    // Not intersected: collapse to zero-area region
    if (max.x < min.x) max.x = min.x;
    if (max.y < min.y) max.y = min.y;
}

#ifdef THORVG_PARTIAL_RENDER_SUPPORT

#include <algorithm>

void RenderDirtyRegion::init(uint32_t w, uint32_t h)
{
    auto cnt = int(sqrt(PARTITIONING));
    auto px = int32_t(w / cnt);
    auto py = int32_t(h / cnt);
    auto lx = int32_t(w % cnt);
    auto ly = int32_t(h % cnt);

    //space partitioning
    for (int y = 0; y < cnt; ++y) {
        for (int x = 0; x < cnt; ++x) {
            auto& partition = partitions[y * cnt + x];
            partition.list[0].reserve(64);
            auto& region = partition.region;
            region.min = {x * px, y * py};
            region.max = {region.min.x + px, region.min.y + py};
            //leftovers
            if (x == cnt -1) region.max.x += lx;
            if (y == cnt -1) region.max.y += ly;
        }
    }
}


bool RenderDirtyRegion::add(const RenderRegion& bbox)
{
    for (int idx = 0; idx < PARTITIONING; ++idx) {
        auto& partition = partitions[idx];
        if (bbox.max.y <= partition.region.min.y) break;
        if (bbox.intersected(partition.region)) {
            ScopedLock lock(key);
            partition.list[partition.current].push(RenderRegion::intersect(bbox, partition.region));
        }
    }
    return true;
}


bool RenderDirtyRegion::add(const RenderRegion& prv, const RenderRegion& cur)
{
    if (prv == cur) return add(prv);

    for (int idx = 0; idx < PARTITIONING; ++idx) {
        auto& partition = partitions[idx];
        if (prv.intersected(partition.region)) {
            ScopedLock lock(key);
            partition.list[partition.current].push(RenderRegion::intersect(prv, partition.region));
        }
        if (cur.intersected(partition.region)) {
            ScopedLock lock(key);
            partition.list[partition.current].push(RenderRegion::intersect(cur, partition.region));
        }
    }
    return true;
}


void RenderDirtyRegion::clear()
{
    for (int idx = 0; idx < PARTITIONING; ++idx) {
        partitions[idx].list[0].clear();
        partitions[idx].list[1].clear();
    }
}


void RenderDirtyRegion::subdivide(Array<RenderRegion>& targets, uint32_t idx, RenderRegion& lhs, RenderRegion& rhs)
{
    RenderRegion temp[3];
    uint32_t cnt = 0;

    //subtract top
    if (rhs.min.y < lhs.min.y) {
        temp[cnt++] = {{rhs.min.x, rhs.min.y}, {rhs.max.x, lhs.min.y}};
        rhs.min.y = lhs.min.y;
    }
    //subtract bottom
    if (rhs.max.y > lhs.max.y) {
        temp[cnt++] = {{rhs.min.x, lhs.max.y}, {rhs.max.x, rhs.max.y}};
        rhs.max.y = lhs.max.y;
    }
    //subtract right
    if (rhs.max.x > lhs.max.x) {
        temp[cnt++] = {{lhs.max.x, rhs.min.y}, {rhs.max.x, rhs.max.y}};
    }

    //Please reserve memory enough with targets.reserve()
    if (targets.count + cnt - 1 >= targets.reserved) {
        TVGERR("RENDERER", "reserved(%d), required(%d)", targets.reserved, targets.count + cnt - 1);
        return;
    }

    /* Shift data. Considered using a list to avoid memory shifting,
       but ultimately, the array outperformed the list due to better cache locality. */
    auto src = &targets[idx + 1];
    auto dst = &targets[idx + cnt];   // <-- shift right by (cnt - 1)
    auto nmove = targets.count - idx - 1;  // number of tail elements
    memmove(dst, src, sizeof(RenderRegion) * nmove);
    memcpy(&targets[idx], temp, sizeof(RenderRegion) * cnt);
    targets.count += (cnt - 1);

    //sorting by x coord again, only for the updated region
    while (dst < targets.end() && dst->min.x < rhs.max.x) ++dst;
    stable_sort(&targets[idx], dst, [](const RenderRegion& a, const RenderRegion& b) -> bool {
        return a.min.x < b.min.x;
    });
}


void RenderDirtyRegion::commit()
{
    if (disabled) return;

    for (int idx = 0; idx < PARTITIONING; ++idx) {
        auto current = partitions[idx].current;
        auto& targets = partitions[idx].list[current];
        if (targets.empty()) continue;

        current = !current; //swapping buffers
        auto& output = partitions[idx].list[current];

        targets.reserve(targets.count * 10);  //one intersection can be divided up to 3
        output.reserve(targets.count);

        partitions[idx].current = current;

        //sorting by x coord. guarantee the stable performance: O(NlogN)
        stable_sort(targets.begin(), targets.end(), [](const RenderRegion& a, const RenderRegion& b) -> bool {
            return a.min.x < b.min.x;
        });

        //Optimized using sweep-line algorithm: O(NlogN)
        for (uint32_t i = 0; i < targets.count; ++i) {
            auto& lhs = targets[i];
            if (lhs.invalid()) continue;
            auto merged = false;

            for (uint32_t j = i + 1; j < targets.count; ++j) {
                auto& rhs = targets[j];
                if (rhs.invalid()) continue;
                if (lhs.max.x < rhs.min.x) break;   //line sweeping

                //fully overlapped. drop lhs
                if (rhs.contained(lhs)) {
                    merged = true;
                    break;
                }
                //fully overlapped. replace the lhs with rhs
                if (lhs.contained(rhs)) {
                    rhs = {};
                    continue;
                }
                //just merge & expand on x axis
                if (lhs.min.y == rhs.min.y && lhs.max.y == rhs.max.y) {
                    if (lhs.max.x >= rhs.min.x) {
                        lhs.max.x = rhs.max.x;
                        rhs = {};
                        j = i;   //lhs dirty region has been damaged, try again.
                        continue;
                    }
                }
                //just merge & expand on y axis
                if (lhs.min.x == rhs.min.x && lhs.max.x == rhs.max.x) {
                    if (lhs.min.y <= rhs.max.y && rhs.min.y <= lhs.max.y) {
                        rhs.min.y = std::min(lhs.min.y, rhs.min.y);
                        rhs.max.y = std::max(lhs.max.y, rhs.max.y);
                        merged = true;
                        break;
                    }
                }
                //subdivide regions
                if (lhs.intersected(rhs)) {
                    subdivide(targets, j, lhs, rhs);
                    --j; //rhs dirty region has been damaged, try again.
                }
            }
            if (!merged) output.push(lhs);  //this region is complete isolated
            lhs = {};
        }
    }
}

#endif

/************************************************************************/
/* RenderTrimPath Class Implementation                                  */
/************************************************************************/

#define EPSILON 1e-4f


static void _trimAt(const PathCommand* cmds, const Point* pts, Point& moveTo, float at1, float at2, bool start, RenderPath& out)
{
    switch (*cmds) {
        case PathCommand::LineTo: {
            Line tmp, left, right;
            Line{*(pts - 1), *pts}.split(at1, left, tmp);
            tmp.split(at2, left, right);
            if (start) {
                out.pts.push(left.pt1);
                moveTo = left.pt1;
                out.cmds.push(PathCommand::MoveTo);
            }
            out.pts.push(left.pt2);
            out.cmds.push(PathCommand::LineTo);
            break;
        }
        case PathCommand::CubicTo: {
            Bezier tmp, left, right;
            Bezier{*(pts - 1), *pts, *(pts + 1), *(pts + 2)}.split(at1, left, tmp);
            tmp.split(at2, left, right);
            if (start) {
                moveTo = left.start;
                out.pts.push(left.start);
                out.cmds.push(PathCommand::MoveTo);
            }
            out.pts.push(left.ctrl1);
            out.pts.push(left.ctrl2);
            out.pts.push(left.end);
            out.cmds.push(PathCommand::CubicTo);
            break;
        }
        case PathCommand::Close: {
            Line tmp, left, right;
            Line{*(pts - 1), moveTo}.split(at1, left, tmp);
            tmp.split(at2, left, right);
            if (start) {
                moveTo = left.pt1;
                out.pts.push(left.pt1);
                out.cmds.push(PathCommand::MoveTo);
            }
            out.pts.push(left.pt2);
            out.cmds.push(PathCommand::LineTo);
            break;
        }
        default: break;
    }
}


static void _add(const PathCommand* cmds, const Point* pts, const Point& moveTo, bool& start, RenderPath& out)
{
    switch (*cmds) {
        case PathCommand::MoveTo: {
            out.cmds.push(PathCommand::MoveTo);
            out.pts.push(*pts);
            start = false;
            break;
        }
        case PathCommand::LineTo: {
            if (start) {
                out.cmds.push(PathCommand::MoveTo);
                out.pts.push(*(pts - 1));
            }
            out.cmds.push(PathCommand::LineTo);
            out.pts.push(*pts);
            start = false;
            break;
        }
        case PathCommand::CubicTo: {
            if (start) {
                out.cmds.push(PathCommand::MoveTo);
                out.pts.push(*(pts - 1));
            }
            out.cmds.push(PathCommand::CubicTo);
            out.pts.push(*pts);
            out.pts.push(*(pts + 1));
            out.pts.push(*(pts + 2));
            start = false;
            break;
        }
        case PathCommand::Close: {
            if (start) {
                out.cmds.push(PathCommand::MoveTo);
                out.pts.push(*(pts - 1));
            }
            out.cmds.push(PathCommand::LineTo);
            out.pts.push(moveTo);
            start = true;
            break;
        }
    }
}


static void _trimPath(const PathCommand* inCmds, uint32_t inCmdsCnt, const Point* inPts, TVG_UNUSED uint32_t inPtsCnt, float trimStart, float trimEnd, RenderPath& out, bool connect = false)
{
    auto cmds = const_cast<PathCommand*>(inCmds);
    auto pts = const_cast<Point*>(inPts);
    auto moveToTrimmed = *pts;
    auto moveTo = *pts;
    auto len = 0.0f;

    auto _length = [&]() -> float {
        switch (*cmds) {
            case PathCommand::MoveTo: return 0.0f;
            case PathCommand::LineTo: return tvg::length(*(pts - 1), *pts);
            case PathCommand::CubicTo: return Bezier{*(pts - 1), *pts, *(pts + 1), *(pts + 2)}.length();
            case PathCommand::Close: return tvg::length(*(pts - 1), moveTo);
        }
        return 0.0f;
    };

    auto _shift = [&]() -> void {
        switch (*cmds) {
            case PathCommand::MoveTo:
                moveTo = *pts;
                moveToTrimmed = *pts;
                ++pts;
                break;
            case PathCommand::LineTo:
                ++pts;
                break;
            case PathCommand::CubicTo:
                pts += 3;
                break;
            case PathCommand::Close:
                break;
        }
        ++cmds;
    };

    auto start = !connect;

    for (uint32_t i = 0; i < inCmdsCnt; ++i) {
        auto dLen = _length();

        //very short segments are skipped since due to the finite precision of Bezier curve subdivision and length calculation (1e-2),
        //trimming may produce very short segments that would effectively have zero length with higher computational accuracy.
        if (len <= trimStart) {
            //cut the segment at the beginning and at the end
            if (len + dLen > trimEnd) {
                _trimAt(cmds, pts, moveToTrimmed, trimStart - len, trimEnd - trimStart, start, out);
                start = false;
                //cut the segment at the beginning
            } else if (len + dLen > trimStart + EPSILON) {
                _trimAt(cmds, pts, moveToTrimmed, trimStart - len, len + dLen - trimStart, start, out);
                start = false;
            }
        } else if (len <= trimEnd - EPSILON) {
            //cut the segment at the end
            if (len + dLen > trimEnd) {
                _trimAt(cmds, pts, moveTo, 0.0f, trimEnd - len, start, out);
                start = true;
            //add the whole segment
            } else if (len + dLen > trimStart + EPSILON) _add(cmds, pts, moveTo, start, out);
        }

        len += dLen;
        _shift();
    }
}


static void _trim(const PathCommand* inCmds, uint32_t inCmdsCnt, const Point* inPts, uint32_t inPtsCnt, float begin, float end, bool connect, RenderPath& out)
{
    auto totalLength = tvg::length(inCmds, inCmdsCnt, inPts, inPtsCnt);
    auto trimStart = begin * totalLength;
    auto trimEnd = end * totalLength;

    if (begin >= end) {
        _trimPath(inCmds, inCmdsCnt, inPts, inPtsCnt, trimStart, totalLength, out);
        _trimPath(inCmds, inCmdsCnt, inPts, inPtsCnt, 0.0f, trimEnd, out, connect);
    } else {
        _trimPath(inCmds, inCmdsCnt, inPts, inPtsCnt, trimStart, trimEnd, out);
    }
}


static void _get(float& begin, float& end)
{
    auto loop = true;

    if (begin > 1.0f && end > 1.0f) loop = false;
    if (begin < 0.0f && end < 0.0f) loop = false;
    if (begin >= 0.0f && begin <= 1.0f && end >= 0.0f  && end <= 1.0f) loop = false;

    if (begin > 1.0f) begin -= 1.0f;
    if (begin < 0.0f) begin += 1.0f;
    if (end > 1.0f) end -= 1.0f;
    if (end < 0.0f) end += 1.0f;

    if ((loop && begin < end) || (!loop && begin > end)) std::swap(begin, end);
}


bool RenderTrimPath::trim(const RenderPath& in, RenderPath& out) const
{
    if (in.pts.count < 2 || tvg::zero(begin - end)) return false;

    float begin = this->begin, end = this->end;
    _get(begin, end);

    out.cmds.reserve(in.cmds.count * 2);
    out.pts.reserve(in.pts.count * 2);

    auto pts = in.pts.data;
    auto cmds = in.cmds.data;

    if (simultaneous) {
        auto startCmds = cmds;
        auto startPts = pts;
        uint32_t i = 0;
        while (i < in.cmds.count) {
            switch (in.cmds[i]) {
                case PathCommand::MoveTo: {
                    if (startCmds != cmds) _trim(startCmds, cmds - startCmds, startPts, pts - startPts, begin, end, *(cmds - 1) == PathCommand::Close, out);
                    startPts = pts;
                    startCmds = cmds;
                    ++pts;
                    ++cmds;
                    break;
                }
                case PathCommand::LineTo: {
                    ++pts;
                    ++cmds;
                    break;
                }
                case PathCommand::CubicTo: {
                    pts += 3;
                    ++cmds;
                    break;
                }
                case PathCommand::Close: {
                    ++cmds;
                    if (startCmds != cmds) _trim(startCmds, cmds - startCmds, startPts, pts - startPts, begin, end, *(cmds - 1) == PathCommand::Close, out);
                    startPts = pts;
                    startCmds = cmds;
                    break;
                }
            }
            i++;
        }
        if (startCmds != cmds) _trim(startCmds, cmds - startCmds, startPts, pts - startPts, begin, end, *(cmds - 1) == PathCommand::Close, out);
    } else {
        _trim(in.cmds.data, in.cmds.count, in.pts.data, in.pts.count, begin, end, false, out);
    }

    return out.pts.count >= 2;
}

/************************************************************************/
/* StrokeDashPath Class Implementation                                  */
/************************************************************************/

//TODO: use this common function from sw engine
#if defined(THORVG_GL_RASTER_SUPPORT) || defined(THORVG_WG_RASTER_SUPPORT)

struct StrokeDashPath
{
public:
    StrokeDashPath(RenderStroke::Dash dash) : dash(dash) {}
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
    #define MIN_CURR_LEN_THRESHOLD 0.1f

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


//allowDot: zero length segment with non-butt cap still should be rendered as a point - only the caps are visible
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

    ARRAY_FOREACH(cmd, in.cmds) {
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
    segment<Line>(line, len, out, allowDot,
        [](const Line& l) { return length(l.pt2 - l.pt1); },
        [](const Line& l, float len, Line& left, Line& right) { l.split(len, left, right); },
        [&](const Line& l) { out.lineTo(map(l.pt2)); },
        [](const Line& l) { return l.pt1; },
        to
    );
}


void StrokeDashPath::cubicTo(RenderPath& out, const Point& cnt1, const Point& cnt2, const Point& end, bool allowDot)
{
    Bezier curve = {curPos, cnt1, cnt2, end};
    auto len = curve.length();
    segment<Bezier>(curve, len, out, allowDot,
        [](const Bezier& b) { return b.length(); },
        [](const Bezier& b, float len, Bezier& left, Bezier& right) { b.split(len, left, right); },
        [&](const Bezier& b) { out.cubicTo(map(b.ctrl1), map(b.ctrl2), map(b.end)); },
        [](const Bezier& b) { return b.start; },
        end
    );
}


bool RenderShape::strokeDash(RenderPath& out, const Matrix* transform) const
{
    if (!stroke || stroke->dash.count == 0 || stroke->dash.length < DASH_PATTERN_THRESHOLD) return false;

    out.cmds.reserve(20 * path.cmds.count);
    out.pts.reserve(20 * path.pts.count);

    StrokeDashPath dash(stroke->dash);
    auto allowDot = stroke->cap != StrokeCap::Butt;

    if (trimpath()) {
        RenderPath tpath;
        if (stroke->trim.trim(path, tpath)) return dash.gen(tpath, out, allowDot, transform);
        else return false;
    }
    return dash.gen(path, out, allowDot, transform);
}
#else
bool RenderShape::strokeDash(RenderPath& out, const Matrix* transform) const
{
    return false;
}
#endif
