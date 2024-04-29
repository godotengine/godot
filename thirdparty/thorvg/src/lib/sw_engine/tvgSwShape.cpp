/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

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
#include "tvgBezier.h"
#include <float.h>
#include <math.h>

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct Line
{
    Point pt1;
    Point pt2;
};


static float _lineLength(const Point& pt1, const Point& pt2)
{
    /* approximate sqrt(x*x + y*y) using alpha max plus beta min algorithm.
       With alpha = 1, beta = 3/8, giving results with the largest error less
       than 7% compared to the exact value. */
    Point diff = {pt2.x - pt1.x, pt2.y - pt1.y};
    if (diff.x < 0) diff.x = -diff.x;
    if (diff.y < 0) diff.y = -diff.y;
    return (diff.x > diff.y) ? (diff.x + diff.y * 0.375f) : (diff.y + diff.x * 0.375f);
}


static void _lineSplitAt(const Line& cur, float at, Line& left, Line& right)
{
    auto len = _lineLength(cur.pt1, cur.pt2);
    auto dx = ((cur.pt2.x - cur.pt1.x) / len) * at;
    auto dy = ((cur.pt2.y - cur.pt1.y) / len) * at;
    left.pt1 = cur.pt1;
    left.pt2.x = left.pt1.x + dx;
    left.pt2.y = left.pt1.y + dy;
    right.pt1 = left.pt2;
    right.pt2 = cur.pt2;
}


static void _outlineEnd(SwOutline& outline)
{
    if (outline.pts.empty()) return;
    outline.cntrs.push(outline.pts.count - 1);
}


static void _outlineMoveTo(SwOutline& outline, const Point* to, const Matrix* transform)
{
    if (outline.pts.count > 0) outline.cntrs.push(outline.pts.count - 1);

    outline.pts.push(mathTransform(to, transform));
    outline.types.push(SW_CURVE_TYPE_POINT);
}


static void _outlineLineTo(SwOutline& outline, const Point* to, const Matrix* transform)
{
    outline.pts.push(mathTransform(to, transform));
    outline.types.push(SW_CURVE_TYPE_POINT);
}


static void _outlineCubicTo(SwOutline& outline, const Point* ctrl1, const Point* ctrl2, const Point* to, const Matrix* transform)
{
    outline.pts.push(mathTransform(ctrl1, transform));
    outline.types.push(SW_CURVE_TYPE_CUBIC);

    outline.pts.push(mathTransform(ctrl2, transform));
    outline.types.push(SW_CURVE_TYPE_CUBIC);    

    outline.pts.push(mathTransform(to, transform));
    outline.types.push(SW_CURVE_TYPE_POINT);
}


static void _outlineClose(SwOutline& outline)
{
    uint32_t i = 0;

    if (outline.cntrs.count > 0) i = outline.cntrs.last() + 1;
    else i = 0;   //First Path

    //Make sure there is at least one point in the current path
    if (outline.pts.count == i) return;

    //Close the path
    outline.pts.push(outline.pts.data[i]);
    outline.types.push(SW_CURVE_TYPE_POINT);
    outline.closed.push(true);
}


static void _dashLineTo(SwDashStroke& dash, const Point* to, const Matrix* transform)
{
    Line cur = {dash.ptCur, *to};
    auto len = _lineLength(cur.pt1, cur.pt2);

    if (len < dash.curLen) {
        dash.curLen -= len;
        if (!dash.curOpGap) {
            _outlineMoveTo(*dash.outline, &dash.ptCur, transform);
            _outlineLineTo(*dash.outline, to, transform);
        }
    } else {
        while (len > dash.curLen) {
            len -= dash.curLen;
            Line left, right;
            _lineSplitAt(cur, dash.curLen, left, right);;
            dash.curIdx = (dash.curIdx + 1) % dash.cnt;
            if (!dash.curOpGap) {
                _outlineMoveTo(*dash.outline, &left.pt1, transform);
                _outlineLineTo(*dash.outline, &left.pt2, transform);
            }
            dash.curLen = dash.pattern[dash.curIdx];
            dash.curOpGap = !dash.curOpGap;
            cur = right;
            dash.ptCur = cur.pt1;
        }
        //leftovers
        dash.curLen -= len;
        if (!dash.curOpGap) {
            _outlineMoveTo(*dash.outline, &cur.pt1, transform);
            _outlineLineTo(*dash.outline, &cur.pt2, transform);
        }
        if (dash.curLen < 1 && TO_SWCOORD(len) > 1) {
            //move to next dash
            dash.curIdx = (dash.curIdx + 1) % dash.cnt;
            dash.curLen = dash.pattern[dash.curIdx];
            dash.curOpGap = !dash.curOpGap;
        }
    }
    dash.ptCur = *to;
}


static void _dashCubicTo(SwDashStroke& dash, const Point* ctrl1, const Point* ctrl2, const Point* to, const Matrix* transform)
{
    Bezier cur = {dash.ptCur, *ctrl1, *ctrl2, *to};
    auto len = bezLength(cur);

    if (len < dash.curLen) {
        dash.curLen -= len;
        if (!dash.curOpGap) {
            _outlineMoveTo(*dash.outline, &dash.ptCur, transform);
            _outlineCubicTo(*dash.outline, ctrl1, ctrl2, to, transform);
        }
    } else {
        while (len > dash.curLen) {
            Bezier left, right;
            len -= dash.curLen;
            bezSplitAt(cur, dash.curLen, left, right);
            if (!dash.curOpGap) {
                // leftovers from a previous command don't require moveTo
                if (dash.pattern[dash.curIdx] - dash.curLen < FLT_EPSILON) {
                    _outlineMoveTo(*dash.outline, &left.start, transform);
                }
                _outlineCubicTo(*dash.outline, &left.ctrl1, &left.ctrl2, &left.end, transform);
            }
            dash.curIdx = (dash.curIdx + 1) % dash.cnt;
            dash.curLen = dash.pattern[dash.curIdx];
            dash.curOpGap = !dash.curOpGap;
            cur = right;
            dash.ptCur = right.start;
        }
        //leftovers
        dash.curLen -= len;
        if (!dash.curOpGap) {
            _outlineMoveTo(*dash.outline, &cur.start, transform);
            _outlineCubicTo(*dash.outline, &cur.ctrl1, &cur.ctrl2, &cur.end, transform);
        }
        if (dash.curLen < 1 && TO_SWCOORD(len) > 1) {
            //move to next dash
            dash.curIdx = (dash.curIdx + 1) % dash.cnt;
            dash.curLen = dash.pattern[dash.curIdx];
            dash.curOpGap = !dash.curOpGap;
        }
    }
    dash.ptCur = *to;
}


static SwOutline* _genDashOutline(const RenderShape* rshape, const Matrix* transform)
{
    const PathCommand* cmds = rshape->path.cmds.data;
    auto cmdCnt = rshape->path.cmds.count;

    const Point* pts = rshape->path.pts.data;
    auto ptsCnt = rshape->path.pts.count;

    //No actual shape data
    if (cmdCnt == 0 || ptsCnt == 0) return nullptr;

    SwDashStroke dash;
    dash.curIdx = 0;
    dash.curLen = 0;
    dash.ptStart = {0, 0};
    dash.ptCur = {0, 0};
    dash.curOpGap = false;

    const float* pattern;
    dash.cnt = rshape->strokeDash(&pattern);
    if (dash.cnt == 0) return nullptr;

    //OPTMIZE ME: Use mempool???
    dash.pattern = const_cast<float*>(pattern);
    dash.outline = static_cast<SwOutline*>(calloc(1, sizeof(SwOutline)));

    //smart reservation
    auto outlinePtsCnt = 0;
    auto outlineCntrsCnt = 0;

    for (uint32_t i = 0; i < cmdCnt; ++i) {
        switch (*(cmds + i)) {
            case PathCommand::Close: {
                ++outlinePtsCnt;
                break;
            }
            case PathCommand::MoveTo: {
                ++outlineCntrsCnt;
                ++outlinePtsCnt;
                break;
            }
            case PathCommand::LineTo: {
                ++outlinePtsCnt;
                break;
            }
            case PathCommand::CubicTo: {
                outlinePtsCnt += 3;
                break;
            }
        }
    }

    ++outlinePtsCnt;    //for close
    ++outlineCntrsCnt;  //for end

    //No idea exact count.... Reserve Approximitely 20x...
    dash.outline->pts.grow(20 * outlinePtsCnt);
    dash.outline->types.grow(20 * outlinePtsCnt);
    dash.outline->cntrs.grow(20 * outlineCntrsCnt);

    while (cmdCnt-- > 0) {
        switch (*cmds) {
            case PathCommand::Close: {
                _dashLineTo(dash, &dash.ptStart, transform);
                break;
            }
            case PathCommand::MoveTo: {
                //reset the dash
                dash.curIdx = 0;
                dash.curLen = *dash.pattern;
                dash.curOpGap = false;
                dash.ptStart = dash.ptCur = *pts;
                ++pts;
                break;
            }
            case PathCommand::LineTo: {
                _dashLineTo(dash, pts, transform);
                ++pts;
                break;
            }
            case PathCommand::CubicTo: {
                _dashCubicTo(dash, pts, pts + 1, pts + 2, transform);
                pts += 3;
                break;
            }
        }
        ++cmds;
    }

    _outlineEnd(*dash.outline);

    return dash.outline;
}


static bool _axisAlignedRect(const SwOutline* outline)
{
    //Fast Track: axis-aligned rectangle?
    if (outline->pts.count != 5) return false;

    auto pt1 = outline->pts.data + 0;
    auto pt2 = outline->pts.data + 1;
    auto pt3 = outline->pts.data + 2;
    auto pt4 = outline->pts.data + 3;

    auto a = SwPoint{pt1->x, pt3->y};
    auto b = SwPoint{pt3->x, pt1->y};

    if ((*pt2 == a && *pt4 == b) || (*pt2 == b && *pt4 == a)) return true;

    return false;
}


static bool _genOutline(SwShape* shape, const RenderShape* rshape, const Matrix* transform, SwMpool* mpool, unsigned tid, bool hasComposite)
{
    const PathCommand* cmds = rshape->path.cmds.data;
    auto cmdCnt = rshape->path.cmds.count;

    const Point* pts = rshape->path.pts.data;
    auto ptsCnt = rshape->path.pts.count;

    //No actual shape data
    if (cmdCnt == 0 || ptsCnt == 0) return false;

    //smart reservation
    auto outlinePtsCnt = 0;
    auto outlineCntrsCnt = 0;
    auto closeCnt = 0;

    for (uint32_t i = 0; i < cmdCnt; ++i) {
        switch (*(cmds + i)) {
            case PathCommand::Close: {
                ++outlinePtsCnt;
                ++closeCnt;
                break;
            }
            case PathCommand::MoveTo: {
                ++outlineCntrsCnt;
                ++outlinePtsCnt;
                break;
            }
            case PathCommand::LineTo: {
                ++outlinePtsCnt;
                break;
            }
            case PathCommand::CubicTo: {
                outlinePtsCnt += 3;
                break;
            }
        }
    }

    if (static_cast<uint32_t>(outlinePtsCnt - closeCnt) > ptsCnt) {
        TVGERR("SW_ENGINE", "Wrong a pair of the commands & points - required(%d), current(%d)", outlinePtsCnt - closeCnt, ptsCnt);
        return false;
    }

    ++outlinePtsCnt;    //for close
    ++outlineCntrsCnt;  //for end

    shape->outline = mpoolReqOutline(mpool, tid);
    auto outline = shape->outline;

    outline->pts.grow(outlinePtsCnt);
    outline->types.grow(outlinePtsCnt);
    outline->cntrs.grow(outlineCntrsCnt);

    //Dash outlines are always opened.
    //Only normal outlines use this information, it sholud be same to their contour counts.
    outline->closed.reserve(outline->cntrs.reserved);

    memset(outline->closed.data, 0x0, sizeof(bool) * outline->closed.reserved);

    //Generate Outlines
    while (cmdCnt-- > 0) {
        switch (*cmds) {
            case PathCommand::Close: {
                _outlineClose(*outline);
                break;
            }
            case PathCommand::MoveTo: {
                _outlineMoveTo(*outline, pts, transform);
                ++pts;
                break;
            }
            case PathCommand::LineTo: {
                _outlineLineTo(*outline, pts, transform);
                ++pts;
                break;
            }
            case PathCommand::CubicTo: {
                _outlineCubicTo(*outline, pts, pts + 1, pts + 2, transform);
                pts += 3;
                break;
            }
        }
        ++cmds;
    }

    _outlineEnd(*outline);

    outline->fillRule = rshape->rule;
    shape->outline = outline;

    shape->fastTrack = (!hasComposite && _axisAlignedRect(shape->outline));
    return true;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

bool shapePrepare(SwShape* shape, const RenderShape* rshape, const Matrix* transform,  const SwBBox& clipRegion, SwBBox& renderRegion, SwMpool* mpool, unsigned tid, bool hasComposite)
{
    if (!_genOutline(shape, rshape, transform, mpool, tid, hasComposite)) return false;
    if (!mathUpdateOutlineBBox(shape->outline, clipRegion, renderRegion, shape->fastTrack)) return false;

    //Keep it for Rasterization Region
    shape->bbox = renderRegion;

    //Check valid region
    if (renderRegion.max.x - renderRegion.min.x < 1 && renderRegion.max.y - renderRegion.min.y < 1) return false;

    //Check boundary
    if (renderRegion.min.x >= clipRegion.max.x || renderRegion.min.y >= clipRegion.max.y ||
        renderRegion.max.x <= clipRegion.min.x || renderRegion.max.y <= clipRegion.min.y) return false;

    return true;
}


bool shapePrepared(const SwShape* shape)
{
    return shape->rle ? true : false;
}


bool shapeGenRle(SwShape* shape, TVG_UNUSED const RenderShape* rshape, bool antiAlias)
{
    //FIXME: Should we draw it?
    //Case: Stroke Line
    //if (shape.outline->opened) return true;

    //Case A: Fast Track Rectangle Drawing
    if (shape->fastTrack) return true;

    //Case B: Normal Shape RLE Drawing
    if ((shape->rle = rleRender(shape->rle, shape->outline, shape->bbox, antiAlias))) return true;

    return false;
}


void shapeDelOutline(SwShape* shape, SwMpool* mpool, uint32_t tid)
{
    mpoolRetOutline(mpool, tid);
    shape->outline = nullptr;
}


void shapeReset(SwShape* shape)
{
    rleReset(shape->rle);
    rleReset(shape->strokeRle);
    shape->fastTrack = false;
    shape->bbox.reset();
}


void shapeFree(SwShape* shape)
{
    rleFree(shape->rle);
    shapeDelFill(shape);

    if (shape->stroke) {
        rleFree(shape->strokeRle);
        strokeFree(shape->stroke);
    }
}


void shapeDelStroke(SwShape* shape)
{
    if (!shape->stroke) return;
    rleFree(shape->strokeRle);
    shape->strokeRle = nullptr;
    strokeFree(shape->stroke);
    shape->stroke = nullptr;
}


void shapeResetStroke(SwShape* shape, const RenderShape* rshape, const Matrix* transform)
{
    if (!shape->stroke) shape->stroke = static_cast<SwStroke*>(calloc(1, sizeof(SwStroke)));
    auto stroke = shape->stroke;
    if (!stroke) return;

    strokeReset(stroke, rshape, transform);
    rleReset(shape->strokeRle);
}


bool shapeGenStrokeRle(SwShape* shape, const RenderShape* rshape, const Matrix* transform, const SwBBox& clipRegion, SwBBox& renderRegion, SwMpool* mpool, unsigned tid)
{
    SwOutline* shapeOutline = nullptr;
    SwOutline* strokeOutline = nullptr;
    bool freeOutline = false;
    bool ret = true;

    //Dash Style Stroke
    if (rshape->strokeDash(nullptr) > 0) {
        shapeOutline = _genDashOutline(rshape, transform);
        if (!shapeOutline) return false;
        freeOutline = true;
    //Normal Style stroke
    } else {
        if (!shape->outline) {
            if (!_genOutline(shape, rshape, transform, mpool, tid, false)) return false;
        }
        shapeOutline = shape->outline;
    }

    if (!strokeParseOutline(shape->stroke, *shapeOutline)) {
        ret = false;
        goto fail;
    }

    strokeOutline = strokeExportOutline(shape->stroke, mpool, tid);

    if (!mathUpdateOutlineBBox(strokeOutline, clipRegion, renderRegion, false)) {
        ret = false;
        goto fail;
    }

    shape->strokeRle = rleRender(shape->strokeRle, strokeOutline, renderRegion, true);

fail:
    if (freeOutline) {
        free(shapeOutline->cntrs.data);
        free(shapeOutline->pts.data);
        free(shapeOutline->types.data);
        free(shapeOutline->closed.data);
        free(shapeOutline);
    }
    mpoolRetStrokeOutline(mpool, tid);

    return ret;
}


bool shapeGenFillColors(SwShape* shape, const Fill* fill, const Matrix* transform, SwSurface* surface, uint8_t opacity, bool ctable)
{
    return fillGenColorTable(shape->fill, fill, transform, surface, opacity, ctable);
}


bool shapeGenStrokeFillColors(SwShape* shape, const Fill* fill, const Matrix* transform, SwSurface* surface, uint8_t opacity, bool ctable)
{
    return fillGenColorTable(shape->stroke->fill, fill, transform, surface, opacity, ctable);
}


void shapeResetFill(SwShape* shape)
{
    if (!shape->fill) {
        shape->fill = static_cast<SwFill*>(calloc(1, sizeof(SwFill)));
        if (!shape->fill) return;
    }
    fillReset(shape->fill);
}


void shapeResetStrokeFill(SwShape* shape)
{
    if (!shape->stroke->fill) {
        shape->stroke->fill = static_cast<SwFill*>(calloc(1, sizeof(SwFill)));
        if (!shape->stroke->fill) return;
    }
    fillReset(shape->stroke->fill);
}


void shapeDelFill(SwShape* shape)
{
    if (!shape->fill) return;
    fillFree(shape->fill);
    shape->fill = nullptr;
}


void shapeDelStrokeFill(SwShape* shape)
{
    if (!shape->stroke->fill) return;
    fillFree(shape->stroke->fill);
    shape->stroke->fill = nullptr;
}
