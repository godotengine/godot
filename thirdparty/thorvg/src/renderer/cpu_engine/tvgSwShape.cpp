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

#include "tvgSwCommon.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static bool _outlineBegin(SwOutline& outline)
{
    // Make a contour if lineTo/curveTo without calling close or moveTo beforehand.
    if (outline.in.empty()) return false;
    outline.cntrs.push(outline.in.count - 1);
    outline.closed.push(false);
    outline.in.push(outline.in[outline.cntrs.last()]);
    outline.types.push(SW_CURVE_TYPE_POINT);
    return false;
}

static bool _outlineEnd(SwOutline& outline)
{
    if (outline.in.empty()) return false;
    outline.cntrs.push(outline.in.count - 1);
    outline.closed.push(false);
    return false;
}

static bool _outlineMoveTo(SwOutline& outline, const Point& to, bool closed = false)
{
    // make it a contour, if the last contour is not closed yet.
    if (!closed) _outlineEnd(outline);
    outline.in.push(to);
    outline.types.push(SW_CURVE_TYPE_POINT);
    return false;
}

static void _outlineLineTo(SwOutline& outline, const Point& to)
{
    outline.in.push(to);
    outline.types.push(SW_CURVE_TYPE_POINT);
}

static void _outlineCubicTo(SwOutline& outline, const Point& ctrl1, const Point& ctrl2, const Point& to)
{
    outline.in.push(ctrl1);
    outline.in.push(ctrl2);
    outline.in.push(to);
    outline.types.push(SW_CURVE_TYPE_CUBIC);
    outline.types.push(SW_CURVE_TYPE_CUBIC);
    outline.types.push(SW_CURVE_TYPE_POINT);
}

static bool _outlineClose(SwOutline& outline)
{
    uint32_t i;
    if (outline.cntrs.count > 0) i = outline.cntrs.last() + 1;
    else i = 0;

    //Make sure there is at least one point in the current path
    if (outline.in.count == i) return false;

    //Close the path
    outline.in.push(outline.in[i]);
    outline.cntrs.push(outline.in.count - 1);
    outline.types.push(SW_CURVE_TYPE_POINT);
    outline.closed.push(true);

    return true;
}

static void _drawPoint(SwDashStroke& dash, const Point& start)
{
    if (dash.move || dash.pattern[dash.curIdx] < FLOAT_EPSILON) {
        _outlineMoveTo(*dash.outline, start);
        dash.move = false;
    }
    _outlineLineTo(*dash.outline, start);
}

static void _dashLineTo(SwDashStroke& dash, const Point& to, bool validPoint)
{
    Line cur = {dash.ptCur, to};
    auto len = cur.length();
    if (tvg::zero(len)) {
        _outlineMoveTo(*dash.outline, dash.ptCur);
    // draw the current line fully
    } else if (len <= dash.curLen) {
        dash.curLen -= len;
        if (!dash.curOpGap) {
            if (dash.move) {
                _outlineMoveTo(*dash.outline, dash.ptCur);
                dash.move = false;
            }
            _outlineLineTo(*dash.outline, to);
        }
    //draw the current line partially
    } else {
        while (len - dash.curLen > DASH_PATTERN_THRESHOLD) {
            Line left, right;
            if (dash.curLen > 0) {
                len -= dash.curLen;
                cur.split(dash.curLen, left, right);
                if (!dash.curOpGap) {
                    if (dash.move || dash.pattern[dash.curIdx] - dash.curLen < FLOAT_EPSILON) {
                        _outlineMoveTo(*dash.outline, left.pt1);
                        dash.move = false;
                    }
                    _outlineLineTo(*dash.outline, left.pt2);
                }
            } else {
                if (validPoint && !dash.curOpGap) _drawPoint(dash, cur.pt1);
                right = cur;
            }
            dash.curIdx = (dash.curIdx + 1) % dash.cnt;
            dash.curLen = dash.pattern[dash.curIdx];
            dash.curOpGap = !dash.curOpGap;
            cur = right;
            dash.ptCur = cur.pt1;
            dash.move = true;
        }
        //leftovers
        dash.curLen -= len;
        if (!dash.curOpGap) {
            if (dash.move) {
                _outlineMoveTo(*dash.outline, cur.pt1);
                dash.move = false;
            }
            _outlineLineTo(*dash.outline, cur.pt2);
        }
        if (dash.curLen < 1.0f && !tvg::zero(len)) {
            //move to next dash
            dash.curIdx = (dash.curIdx + 1) % dash.cnt;
            dash.curLen = dash.pattern[dash.curIdx];
            dash.curOpGap = !dash.curOpGap;
        }
    }
    dash.ptCur = to;
}

static void _dashCubicTo(SwDashStroke& dash, const Point& ctrl1, const Point& ctrl2, const Point& to, bool validPoint)
{
    Bezier cur = {dash.ptCur, ctrl1, ctrl2, to};
    auto len = cur.length();

    //draw the current line fully
    if (tvg::zero(len)) {
        _outlineMoveTo(*dash.outline, dash.ptCur);
    } else if (len <= dash.curLen) {
        dash.curLen -= len;
        if (!dash.curOpGap) {
            if (dash.move) {
                _outlineMoveTo(*dash.outline, dash.ptCur);
                dash.move = false;
            }
            _outlineCubicTo(*dash.outline, ctrl1, ctrl2, to);
        }
    //draw the current line partially
    } else {
        while ((len - dash.curLen) > DASH_PATTERN_THRESHOLD) {
            Bezier left, right;
            if (dash.curLen > 0) {
                len -= dash.curLen;
                cur.split(dash.curLen, left, right);
                if (!dash.curOpGap) {
                    if (dash.move || dash.pattern[dash.curIdx] - dash.curLen < FLOAT_EPSILON) {
                        _outlineMoveTo(*dash.outline, left.start);
                        dash.move = false;
                    }
                    _outlineCubicTo(*dash.outline, left.ctrl1, left.ctrl2, left.end);
                }
            } else {
                if (validPoint && !dash.curOpGap) _drawPoint(dash, cur.start);
                right = cur;
            }
            dash.curIdx = (dash.curIdx + 1) % dash.cnt;
            dash.curLen = dash.pattern[dash.curIdx];
            dash.curOpGap = !dash.curOpGap;
            cur = right;
            dash.ptCur = right.start;
            dash.move = true;
        }
        //leftovers
        dash.curLen -= len;
        if (!dash.curOpGap) {
            if (dash.move) {
                _outlineMoveTo(*dash.outline, cur.start);
                dash.move = false;
            }
            _outlineCubicTo(*dash.outline, cur.ctrl1, cur.ctrl2, cur.end);
        }
        if (dash.curLen < 0.1f && !tvg::zero(len)) {
            //move to next dash
            dash.curIdx = (dash.curIdx + 1) % dash.cnt;
            dash.curLen = dash.pattern[dash.curIdx];
            dash.curOpGap = !dash.curOpGap;
        }
    }
    dash.ptCur = to;
}

static void _dashClose(SwDashStroke& dash, bool validPoint)
{
    _dashLineTo(dash, dash.ptStart, validPoint);
}

static void _dashMoveTo(SwDashStroke& dash, uint32_t offIdx, float offset, const Point& pts)
{
    dash.curIdx = offIdx % dash.cnt;
    dash.curLen = dash.pattern[dash.curIdx] - offset;
    dash.curOpGap = offIdx % 2;
    dash.ptStart = dash.ptCur = pts;
    dash.move = true;
}

static SwOutline* _genDashOutline(const RenderShape* rshape, SwMpool* mpool, unsigned tid, bool trimmed)
{
    PathCommand *cmds, *tcmds = nullptr;
    Point *pts, *tpts = nullptr;
    uint32_t cmdCnt, ptsCnt;

    if (trimmed) {
        RenderPath trimmed;
        if (!rshape->stroke->trim.trim(rshape->path, trimmed)) return nullptr;
        cmds = tcmds = trimmed.cmds.data;
        cmdCnt = trimmed.cmds.count;
        pts = tpts = trimmed.pts.data;
        ptsCnt = trimmed.pts.count;

        trimmed.cmds.data = nullptr;
        trimmed.pts.data = nullptr;
    } else {
        cmds = rshape->path.cmds.data;
        cmdCnt = rshape->path.cmds.count;
        pts = rshape->path.pts.data;
        ptsCnt = rshape->path.pts.count;
    }

    //No actual shape data
    if (cmdCnt == 0 || ptsCnt == 0) return nullptr;

    SwDashStroke dash;
    dash.pattern = rshape->stroke->dash.pattern;
    dash.cnt = rshape->stroke->dash.count;
    auto offset = rshape->stroke->dash.offset;

    //offset
    uint32_t offIdx = 0;
    if (!tvg::zero(offset)) {
        auto length = rshape->stroke->dash.length;
        bool isOdd = dash.cnt % 2;
        if (isOdd) length *= 2;

        offset = fmodf(offset, length);
        if (offset < 0) offset += length;

        for (size_t i = 0; i < dash.cnt * (1 + (size_t)isOdd); ++i, ++offIdx) {
            auto curPattern = dash.pattern[i % dash.cnt];
            if (offset < curPattern) break;
            offset -= curPattern;
        }
    }

    dash.outline = mpool->outline(tid);

    //must begin with moveTo
    if (cmds[0] == PathCommand::MoveTo) {
        _dashMoveTo(dash, offIdx, offset, *pts);
        cmds++;
        pts++;
    }

    //zero length segment with non-butt cap still should be rendered as a point - only the caps are visible
    auto validPoint = rshape->stroke->cap != StrokeCap::Butt;
    while (--cmdCnt > 0) {
        switch (*cmds) {
            case PathCommand::Close: {
                _dashClose(dash, validPoint);
                break;
            }
            case PathCommand::MoveTo: {
                _dashMoveTo(dash, offIdx, offset, *pts);
                ++pts;
                break;
            }
            case PathCommand::LineTo: {
                _dashLineTo(dash, *pts, validPoint);
                ++pts;
                break;
            }
            case PathCommand::CubicTo: {
                _dashCubicTo(dash, *pts, pts[1], pts[2], validPoint);
                pts += 3;
                break;
            }
        }
        ++cmds;
    }

    _outlineEnd(*dash.outline);

    dash.outline->fillRule = rshape->rule;

    tvg::free(tcmds);
    tvg::free(tpts);

    return dash.outline;
}

static bool _axisAlignedRect(const SwOutline* outline)
{
    // TODO: We can return false if the coordinates have a fractional part, for smoother rectangle movement
    if (outline->out.count != 5) return false;
    if (outline->types[2] == SW_CURVE_TYPE_CUBIC) return false;

    auto pt1 = outline->out.data + 0;
    auto pt2 = outline->out.data + 1;
    auto pt3 = outline->out.data + 2;
    auto pt4 = outline->out.data + 3;

    auto a = SwPoint{pt1->x, pt3->y};
    auto b = SwPoint{pt3->x, pt1->y};

    if ((*pt2 == a && *pt4 == b) || (*pt2 == b && *pt4 == a)) return true;

    return false;
}

static SwOutline* _genOutline(const RenderShape* rshape, SwMpool* mpool, unsigned tid, bool trimmed = false)
{
    PathCommand *cmds, *tcmds = nullptr;
    Point *pts, *tpts = nullptr;
    uint32_t cmdCnt, ptsCnt;

    if (trimmed) {
        RenderPath trimmed;
        if (!rshape->stroke->trim.trim(rshape->path, trimmed)) return nullptr;
        cmds = tcmds = trimmed.cmds.data;
        cmdCnt = trimmed.cmds.count;
        pts = tpts = trimmed.pts.data;
        ptsCnt = trimmed.pts.count;

        trimmed.cmds.data = nullptr;
        trimmed.pts.data = nullptr;
    } else {
        cmds = rshape->path.cmds.data;
        cmdCnt = rshape->path.cmds.count;
        pts = rshape->path.pts.data;
        ptsCnt = rshape->path.pts.count;
    }

    // No actual shape data
    if (cmdCnt == 0 || ptsCnt == 0) return nullptr;

    auto outline = mpool->outline(tid);
    auto closed = false;

    // Generate Outlines
    while (cmdCnt-- > 0) {
        switch (*cmds) {
            case PathCommand::Close: {
                if (!closed) closed = _outlineClose(*outline);
                break;
            }
            case PathCommand::MoveTo: {
                closed = _outlineMoveTo(*outline, *pts, closed);
                ++pts;
                break;
            }
            case PathCommand::LineTo: {
                if (closed) closed = _outlineBegin(*outline);
                _outlineLineTo(*outline, *pts);
                ++pts;
                break;
            }
            case PathCommand::CubicTo: {
                if (closed) closed = _outlineBegin(*outline);
                _outlineCubicTo(*outline, *pts, pts[1], pts[2]);
                pts += 3;
                break;
            }
        }
        ++cmds;
    }

    if (!closed) _outlineEnd(*outline);

    outline->fillRule = rshape->rule;

    tvg::free(tcmds);
    tvg::free(tpts);

    return outline;
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

bool shapeGenRle(SwShape& shape, const RenderShape* rshape, const Matrix& transform, const RenderRegion& clipBox, RenderRegion& renderBox, SwMpool* mpool, unsigned tid, bool composite, bool antiAlias)
{
    auto outline = _genOutline(rshape, mpool, tid, rshape->trimpath());
    if (!outline || outline->in.empty()) {
        renderBox.reset();
        return false;
    }

    BBox bbox;
    utilExport(outline, transform, bbox);

    shape.outline = outline;
    shape.fastTrack = !composite && _axisAlignedRect(outline);

    if (!utilBBox(bbox, clipBox, renderBox, shape.fastTrack)) return false;
    shape.bbox = renderBox;

    if (shape.fastTrack) return true;

    shape.rle = rleRender(shape.rle, outline, renderBox, mpool, tid, antiAlias);
    return shape.rle ? true : false;
}

void shapeDelOutline(SwShape& shape)
{
    shape.outline = nullptr;
}

void shapeReset(SwShape& shape)
{
    rleReset(shape.rle);
    shape.outline = nullptr;
    shape.bbox.reset();
    shape.fastTrack = false;
}


void shapeFree(SwShape& shape)
{
    rleFree(shape.rle);
    shape.rle = nullptr;

    shapeDelFill(shape);

    if (shape.stroke) {
        rleFree(shape.strokeRle);
        shape.strokeRle = nullptr;
        strokeFree(shape.stroke);
        shape.stroke = nullptr;
    }
}


void shapeDelStroke(SwShape& shape)
{
    if (!shape.stroke) return;
    rleFree(shape.strokeRle);
    shape.strokeRle = nullptr;
    strokeFree(shape.stroke);
    shape.stroke = nullptr;
}


void shapeResetStroke(SwShape& shape, const RenderShape* rshape, const Matrix& transform, SwMpool* mpool, unsigned tid)
{
    if (!shape.stroke) shape.stroke = tvg::calloc<SwStroke>(1, sizeof(SwStroke));
    strokeReset(shape.stroke, rshape, transform, mpool, tid);
    rleReset(shape.strokeRle);
}


bool shapeGenStrokeRle(SwShape& shape, const RenderShape* rshape, const Matrix& transform, const RenderRegion& clipBox, RenderRegion& renderBox, SwMpool* mpool, unsigned tid, bool antiAlias)
{
    shapeResetStroke(shape, rshape, transform, mpool, tid);

    auto dash = (rshape->stroke->dash.length > DASH_PATTERN_THRESHOLD);
    SwOutline* outline;

    if (dash) outline = _genDashOutline(rshape, mpool, tid, rshape->trimpath());
    // resuse outline generated by shape
    else outline = shape.outline ? shape.outline : _genOutline(rshape, mpool, tid, rshape->trimpath());
    if (!outline) return false;

    if (!strokeParseOutline(shape.stroke, *outline, mpool, tid)) return false;

    BBox bbox;
    outline = strokeExportOutline(shape.stroke, mpool, tid);
    utilExport(outline, transform, bbox);
    if (!utilBBox(bbox, clipBox, renderBox, false)) return false;

    shape.strokeRle = rleRender(shape.strokeRle, outline, renderBox, mpool, tid, antiAlias);
    return shape.strokeRle ? true : false;
}

bool shapeGenFillColors(SwFill*& out, const Fill* fill, const Matrix& transform, SwSurface* surface, uint8_t opacity, bool ctable)
{
    if (!fill) return true;  // a normal case

    if (!out) {
        out = tvg::calloc<SwFill>(1, sizeof(SwFill));
        ctable = true;
    } else if (ctable) {
        fillReset(out);
    }
    return fillGenColorTable(out, fill, transform, surface, opacity, ctable);
}

void shapeResetFill(SwShape& shape)
{
    if (!shape.fill) shape.fill = tvg::calloc<SwFill>(1, sizeof(SwFill));
    fillReset(shape.fill);
}

void shapeDelFill(SwShape& shape)
{
    if (!shape.fill) return;
    fillFree(shape.fill);
    shape.fill = nullptr;
}

bool shapeStrokeBBox(SwShape& shape, const RenderShape* rshape, Point* pt4, const Matrix& m, SwMpool* mpool)
{
    // TODO: We can skip generation here if the stroke hasn't been updated.
    if (rshape->strokeWidth() > 0.0f) {
        auto outline = _genOutline(rshape, mpool, 0, rshape->trimpath());
        if (!outline) return false;

        if (!shape.stroke) shape.stroke = tvg::calloc<SwStroke>(1, sizeof(SwStroke));
        strokeReset(shape.stroke, rshape, m, mpool, 0);
        strokeParseOutline(shape.stroke, *outline, mpool, 0);

        auto func = [](SwStrokeBorder* border, const Matrix& m, Point& min, Point& max) {
            ARRAY_FOREACH(pts, border->pts)
            {
                auto t = *pts * m;
                if (t.x < min.x) min.x = t.x;
                if (t.x > max.x) max.x = t.x;
                if (t.y < min.y) min.y = t.y;
                if (t.y > max.y) max.y = t.y;
            }
        };

        Point min = {FLT_MAX, FLT_MAX};
        Point max = {-FLT_MAX, -FLT_MAX};
        func(shape.stroke->borders[0], m, min, max);
        func(shape.stroke->borders[1], m, min, max);

        pt4[0] = min;
        pt4[1] = {max.x, min.y};
        pt4[2] = max;
        pt4[3] = {min.x, max.y};

        return true;
    }
    return false;
}
