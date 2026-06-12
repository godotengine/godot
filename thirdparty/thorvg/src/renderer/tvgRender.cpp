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

// used as a temporary buffer
RenderPath& RenderPath::scratch()
{
    static thread_local RenderPath dbuffers[3];  // tripple-buffering
    static thread_local int idx = 0;
    if (++idx > 2) idx = 0;
    dbuffers[(idx)].clear();
    return dbuffers[(int)idx];
}

void RenderPath::addCircle(float cx, float cy, float rx, float ry, bool cw)
{
    auto rxKappa = rx * PATH_KAPPA;
    auto ryKappa = ry * PATH_KAPPA;

    cmds.grow(6);
    auto cmds = this->cmds.end();

    cmds[0] = PathCommand::MoveTo;
    cmds[1] = PathCommand::CubicTo;
    cmds[2] = PathCommand::CubicTo;
    cmds[3] = PathCommand::CubicTo;
    cmds[4] = PathCommand::CubicTo;
    cmds[5] = PathCommand::Close;

    this->cmds.count += 6;

    int table[2][13] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12}};
    int* idx = cw ? table[0] : table[1];

    pts.grow(13);
    auto pts = this->pts.end();

    pts[idx[0]] = {cx, cy - ry};  // moveTo
    pts[idx[1]] = {cx + rxKappa, cy - ry};
    pts[idx[2]] = {cx + rx, cy - ryKappa};
    pts[idx[3]] = {cx + rx, cy};  // cubicTo
    pts[idx[4]] = {cx + rx, cy + ryKappa};
    pts[idx[5]] = {cx + rxKappa, cy + ry};
    pts[idx[6]] = {cx, cy + ry};  // cubicTo
    pts[idx[7]] = {cx - rxKappa, cy + ry};
    pts[idx[8]] = {cx - rx, cy + ryKappa};
    pts[idx[9]] = {cx - rx, cy};  // cubicTo
    pts[idx[10]] = {cx - rx, cy - ryKappa};
    pts[idx[11]] = {cx - rxKappa, cy - ry};
    pts[idx[12]] = {cx, cy - ry};  // cubicTo

    this->pts.count += 13;
}

void RenderPath::addRect(float x, float y, float w, float h, float rx, float ry, bool cw)
{
    if (tvg::zero(rx) && tvg::zero(ry)) {  // sharp rect
        cmds.grow(5);
        pts.grow(4);

        auto cmds = this->cmds.end();
        auto pts = this->pts.end();

        cmds[0] = PathCommand::MoveTo;
        cmds[1] = cmds[2] = cmds[3] = PathCommand::LineTo;
        cmds[4] = PathCommand::Close;

        pts[0] = {x + w, y};
        pts[2] = {x, y + h};
        if (cw) {
            pts[1] = {x + w, y + h};
            pts[3] = {x, y};
        } else {
            pts[1] = {x, y};
            pts[3] = {x + w, y + h};
        }

        this->cmds.count += 5;
        this->pts.count += 4;
    } else {  // round rect
        auto hsize = Point{w * 0.5f, h * 0.5f};
        rx = (rx > hsize.x) ? hsize.x : rx;
        ry = (ry > hsize.y) ? hsize.y : ry;
        auto hr = Point{rx * PATH_KAPPA, ry * PATH_KAPPA};

        cmds.grow(10);
        pts.grow(17);

        auto cmds = this->cmds.end();
        auto pts = this->pts.end();

        cmds[0] = PathCommand::MoveTo;
        cmds[9] = PathCommand::Close;
        pts[0] = {x + w, y + ry};  // move

        if (cw) {
            cmds[1] = cmds[3] = cmds[5] = cmds[7] = PathCommand::LineTo;
            cmds[2] = cmds[4] = cmds[6] = cmds[8] = PathCommand::CubicTo;

            pts[1] = {x + w, y + h - ry};  // line
            pts[2] = {x + w, y + h - ry + hr.y};
            pts[3] = {x + w - rx + hr.x, y + h};
            pts[4] = {x + w - rx, y + h};  // cubic
            pts[5] = {x + rx, y + h};      // line
            pts[6] = {x + rx - hr.x, y + h};
            pts[7] = {x, y + h - ry + hr.y};
            pts[8] = {x, y + h - ry};  // cubic
            pts[9] = {x, y + ry};      // line
            pts[10] = {x, y + ry - hr.y};
            pts[11] = {x + rx - hr.x, y};
            pts[12] = {x + rx, y};      // cubic
            pts[13] = {x + w - rx, y};  // line
            pts[14] = {x + w - rx + hr.x, y};
            pts[15] = {x + w, y + ry - hr.y};
            pts[16] = {x + w, y + ry};  // cubic
        } else {
            cmds[1] = cmds[3] = cmds[5] = cmds[7] = PathCommand::CubicTo;
            cmds[2] = cmds[4] = cmds[6] = cmds[8] = PathCommand::LineTo;

            pts[1] = {x + w, y + ry - hr.y};
            pts[2] = {x + w - rx + hr.x, y};
            pts[3] = {x + w - rx, y};  // cubic
            pts[4] = {x + rx, y};      // line
            pts[5] = {x + rx - hr.x, y};
            pts[6] = {x, y + ry - hr.y};
            pts[7] = {x, y + ry};      // cubic
            pts[8] = {x, y + h - ry};  // line
            pts[9] = {x, y + h - ry + hr.y};
            pts[10] = {x + rx - hr.x, y + h};
            pts[11] = {x + rx, y + h};      // cubic
            pts[12] = {x + w - rx, y + h};  // line
            pts[13] = {x + w - rx + hr.x, y + h};
            pts[14] = {x + w, y + h - ry + hr.y};
            pts[15] = {x + w, y + h - ry};  // cubic
            pts[16] = {x + w, y + ry};      // line
        }

        this->cmds.count += 10;
        this->pts.count += 17;
    }
}

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
