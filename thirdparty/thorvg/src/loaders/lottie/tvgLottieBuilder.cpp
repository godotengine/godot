/*
 * Copyright (c) 2023 - 2024 the ThorVG project. All rights reserved.

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

#include <cstring>
#include <algorithm>

#include "tvgCommon.h"
#include "tvgMath.h"
#include "tvgPaint.h"
#include "tvgShape.h"
#include "tvgInlist.h"
#include "tvgTaskScheduler.h"
#include "tvgLottieModel.h"
#include "tvgLottieBuilder.h"
#include "tvgLottieExpressions.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct RenderRepeater
{
    int cnt;
    Matrix transform;
    float offset;
    Point position;
    Point anchor;
    Point scale;
    float rotation;
    uint8_t startOpacity;
    uint8_t endOpacity;
    bool interpOpacity;
    bool inorder;
};


struct RenderContext
{
    INLIST_ITEM(RenderContext);

    Shape* propagator = nullptr;  //for propagating the shape properties excluding paths
    Shape* merging = nullptr;  //merging shapes if possible (if shapes have same properties)
    LottieObject** begin = nullptr; //iteration entry point
    Array<RenderRepeater> repeaters;
    Matrix* transform = nullptr;
    float roundness = 0.0f;
    bool fragmenting = false;  //render context has been fragmented by filling
    bool reqFragment = false;  //requirment to fragment the render context
    bool ownPropagator = true; //this rendering context shares the propergator

    RenderContext()
    {
        propagator = Shape::gen().release();
    }

    ~RenderContext()
    {
        if (ownPropagator) delete(propagator);
        free(transform);
    }

    RenderContext(const RenderContext& rhs, bool mergeable = false)
    {
        if (mergeable) {
            this->ownPropagator = false;
            propagator = rhs.propagator;
            merging = rhs.merging;
        } else {
            propagator = static_cast<Shape*>(rhs.propagator->duplicate());
        }

        for (auto repeater = rhs.repeaters.begin(); repeater < rhs.repeaters.end(); ++repeater) {
            repeaters.push(*repeater);
        }
        roundness = rhs.roundness;
    }
};


static void _updateChildren(LottieGroup* parent, float frameNo, Inlist<RenderContext>& contexts, LottieExpressions* exps);
static void _updateLayer(LottieLayer* root, LottieLayer* layer, float frameNo, LottieExpressions* exps);
static bool _buildComposition(LottieComposition* comp, LottieLayer* parent);
static bool _draw(LottieGroup* parent, RenderContext* ctx);


static void _rotationXYZ(Matrix* m, float degreeX, float degreeY, float degreeZ)
{
    auto radianX = mathDeg2Rad(degreeX);
    auto radianY = mathDeg2Rad(degreeY);
    auto radianZ = mathDeg2Rad(degreeZ);

    auto cx = cosf(radianX), sx = sinf(radianX);
    auto cy = cosf(radianY), sy = sinf(radianY);;
    auto cz = cosf(radianZ), sz = sinf(radianZ);;
    m->e11 = cy * cz;
    m->e12 = -cy * sz;
    m->e21 = sx * sy * cz + cx * sz;
    m->e22 = -sx * sy * sz + cx * cz;
}


static void _rotationZ(Matrix* m, float degree)
{
    if (degree == 0.0f) return;
    auto radian = mathDeg2Rad(degree);
    m->e11 = cosf(radian);
    m->e12 = -sinf(radian);
    m->e21 = sinf(radian);
    m->e22 = cosf(radian);
}


static void _skew(Matrix* m, float angleDeg, float axisDeg)
{
    auto angle = -mathDeg2Rad(angleDeg);
    float tanVal = tanf(angle);

    axisDeg = fmod(axisDeg, 180.0f);
    if (fabsf(axisDeg) < 0.01f || fabsf(axisDeg - 180.0f) < 0.01f || fabsf(axisDeg + 180.0f) < 0.01f) {
        float cosVal = cosf(mathDeg2Rad(axisDeg));
        auto B = cosVal * cosVal * tanVal;
        m->e12 += B * m->e11;
        m->e22 += B * m->e21;
        return;
    } else if (fabsf(axisDeg - 90.0f) < 0.01f || fabsf(axisDeg + 90.0f) < 0.01f) {
        float sinVal = -sinf(mathDeg2Rad(axisDeg));
        auto C = sinVal * sinVal * tanVal;
        m->e11 -= C * m->e12;
        m->e21 -= C * m->e22;
        return;
    }

    auto axis = -mathDeg2Rad(axisDeg);
    float cosVal = cosf(axis);
    float sinVal = sinf(axis);
    auto A = sinVal * cosVal * tanVal;
    auto B = cosVal * cosVal * tanVal;
    auto C = sinVal * sinVal * tanVal;

    auto e11 = m->e11;
    auto e21 = m->e21;
    m->e11 = (1.0f - A) * e11 - C * m->e12;
    m->e12 = B * e11 + (1.0f + A) * m->e12;
    m->e21 = (1.0f - A) * e21 - C * m->e22;
    m->e22 = B * e21 + (1.0f + A) * m->e22;
}


static bool _updateTransform(LottieTransform* transform, float frameNo, bool autoOrient, Matrix& matrix, uint8_t& opacity, LottieExpressions* exps)
{
    mathIdentity(&matrix);

    if (!transform) {
        opacity = 255;
        return false;
    }

    if (transform->coords) {
        mathTranslate(&matrix, transform->coords->x(frameNo), transform->coords->y(frameNo));
    } else {
        auto position = transform->position(frameNo, exps);
        mathTranslate(&matrix, position.x, position.y);
    }

    auto angle = 0.0f;
    if (autoOrient) angle = transform->position.angle(frameNo);
    if (transform->rotationEx) _rotationXYZ(&matrix, transform->rotationEx->x(frameNo, exps), transform->rotationEx->y(frameNo, exps), transform->rotation(frameNo, exps) + angle);
    else _rotationZ(&matrix, transform->rotation(frameNo, exps) + angle);


    auto skewAngle = transform->skewAngle(frameNo, exps);
    if (skewAngle != 0.0f) {
        // For angles where tangent explodes, the shape degenerates into an infinitely thin line.
        // This is handled by zeroing out the matrix due to finite numerical precision.
        skewAngle = fmod(skewAngle, 180.0f);
        if (fabsf(skewAngle - 90.0f) < 0.01f || fabsf(skewAngle + 90.0f) < 0.01f) return false;
        _skew(&matrix, skewAngle, transform->skewAxis(frameNo, exps));
    }

    auto scale = transform->scale(frameNo, exps);
    mathScaleR(&matrix, scale.x * 0.01f, scale.y * 0.01f);

    //Lottie specific anchor transform.
    auto anchor = transform->anchor(frameNo, exps);
    mathTranslateR(&matrix, -anchor.x, -anchor.y);

    //invisible just in case.
    if (scale.x == 0.0f || scale.y == 0.0f) opacity = 0;
    else opacity = transform->opacity(frameNo, exps);

    return true;
}


static void _updateTransform(LottieLayer* layer, float frameNo, LottieExpressions* exps)
{
    if (!layer || mathEqual(layer->cache.frameNo, frameNo)) return;

    auto transform = layer->transform;
    auto parent = layer->parent;

    if (parent) _updateTransform(parent, frameNo, exps);

    auto& matrix = layer->cache.matrix;

    _updateTransform(transform, frameNo, layer->autoOrient, matrix, layer->cache.opacity, exps);

    if (parent) {
        if (!mathIdentity((const Matrix*) &parent->cache.matrix)) {
            if (mathIdentity((const Matrix*) &matrix)) layer->cache.matrix = parent->cache.matrix;
            else layer->cache.matrix = parent->cache.matrix * matrix;
        }
    }
    layer->cache.frameNo = frameNo;
}


static void _updateTransform(LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto transform = static_cast<LottieTransform*>(*child);
    if (!transform) return;

    uint8_t opacity;

    if (parent->mergeable()) {
        if (!ctx->transform) ctx->transform = (Matrix*)malloc(sizeof(Matrix));
        _updateTransform(transform, frameNo, false, *ctx->transform, opacity, exps);
        return;
    }

    ctx->merging = nullptr;

    Matrix matrix;
    if (!_updateTransform(transform, frameNo, false, matrix, opacity, exps)) return;

    auto pmatrix = PP(ctx->propagator)->transform();
    ctx->propagator->transform(pmatrix ? (*pmatrix * matrix) : matrix);
    ctx->propagator->opacity(MULTIPLY(opacity, PP(ctx->propagator)->opacity));

    //FIXME: preserve the stroke width. too workaround, need a better design.
    if (P(ctx->propagator)->rs.strokeWidth() > 0.0f) {
        auto denominator = sqrtf(matrix.e11 * matrix.e11 + matrix.e12 * matrix.e12);
        if (denominator > 1.0f) ctx->propagator->stroke(ctx->propagator->strokeWidth() / denominator);
    }
}


static void _updateGroup(LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& pcontexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto group = static_cast<LottieGroup*>(*child);

    if (group->children.empty()) return;

    //Prepare render data
    group->scene = parent->scene;
    group->reqFragment |= ctx->reqFragment;

    //generate a merging shape to consolidate partial shapes into a single entity
    if (group->mergeable()) _draw(parent, ctx);

    Inlist<RenderContext> contexts;
    contexts.back(new RenderContext(*ctx, group->mergeable()));

    _updateChildren(group, frameNo, contexts, exps);

    contexts.free();
}


static void _updateStroke(LottieStroke* stroke, float frameNo, RenderContext* ctx, LottieExpressions* exps)
{
    ctx->propagator->stroke(stroke->width(frameNo, exps));
    ctx->propagator->stroke(stroke->cap);
    ctx->propagator->stroke(stroke->join);
    ctx->propagator->strokeMiterlimit(stroke->miterLimit);

    if (stroke->dashattr) {
        float dashes[2];
        dashes[0] = stroke->dashSize(frameNo, exps);
        dashes[1] = dashes[0] + stroke->dashGap(frameNo, exps);
        P(ctx->propagator)->strokeDash(dashes, 2, stroke->dashOffset(frameNo, exps));
    } else {
        ctx->propagator->stroke(nullptr, 0);
    }
}


static bool _fragmented(LottieObject** child, Inlist<RenderContext>& contexts, RenderContext* ctx)
{
    if (!ctx->reqFragment) return false;
    if (ctx->fragmenting) return true;

    contexts.back(new RenderContext(*ctx));
    auto fragment = contexts.tail;
    fragment->begin = child - 1;
    ctx->fragmenting = true;

    return false;
}


static void _updateSolidStroke(TVG_UNUSED LottieGroup* parent, LottieObject** child, float frameNo, Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    if (_fragmented(child, contexts, ctx)) return;

    auto stroke = static_cast<LottieSolidStroke*>(*child);

    ctx->merging = nullptr;
    auto color = stroke->color(frameNo, exps);
    ctx->propagator->stroke(color.rgb[0], color.rgb[1], color.rgb[2], stroke->opacity(frameNo, exps));
    _updateStroke(static_cast<LottieStroke*>(stroke), frameNo, ctx, exps);
}


static void _updateGradientStroke(TVG_UNUSED LottieGroup* parent, LottieObject** child, float frameNo, Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    if (_fragmented(child, contexts, ctx)) return;

    auto stroke = static_cast<LottieGradientStroke*>(*child);

    ctx->merging = nullptr;
    ctx->propagator->stroke(unique_ptr<Fill>(stroke->fill(frameNo, exps)));
    _updateStroke(static_cast<LottieStroke*>(stroke), frameNo, ctx, exps);
}


static void _updateSolidFill(TVG_UNUSED LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    if (_fragmented(child, contexts, ctx)) return;

    auto fill = static_cast<LottieSolidFill*>(*child);

    ctx->merging = nullptr;
    auto color = fill->color(frameNo);
    ctx->propagator->fill(color.rgb[0], color.rgb[1], color.rgb[2], fill->opacity(frameNo, exps));
    ctx->propagator->fill(fill->rule);

    if (ctx->propagator->strokeWidth() > 0) ctx->propagator->order(true);
}


static void _updateGradientFill(TVG_UNUSED LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    if (_fragmented(child, contexts, ctx)) return;

    auto fill = static_cast<LottieGradientFill*>(*child);

    ctx->merging = nullptr;
    //TODO: reuse the fill instance?
    ctx->propagator->fill(unique_ptr<Fill>(fill->fill(frameNo, exps)));
    ctx->propagator->fill(fill->rule);
    ctx->propagator->opacity(MULTIPLY(fill->opacity(frameNo), PP(ctx->propagator)->opacity));

    if (ctx->propagator->strokeWidth() > 0) ctx->propagator->order(true);
}


static bool _draw(LottieGroup* parent, RenderContext* ctx)
{
    if (ctx->merging) return false;

    auto shape = cast<Shape>(ctx->propagator->duplicate());
    ctx->merging = shape.get();
    parent->scene->push(std::move(shape));

    return true;
}


//OPTIMIZE: path?
static void _repeat(LottieGroup* parent, unique_ptr<Shape> path, RenderContext* ctx)
{
    Array<Shape*> propagators;
    propagators.push(ctx->propagator);
    Array<Shape*> shapes;

    for (auto repeater = ctx->repeaters.end() - 1; repeater >= ctx->repeaters.begin(); --repeater) {
        shapes.reserve(repeater->cnt);

        for (int i = 0; i < repeater->cnt; ++i) {
            auto multiplier = repeater->offset + static_cast<float>(i);

            for (auto propagator = propagators.begin(); propagator < propagators.end(); ++propagator) {
                auto shape = static_cast<Shape*>((*propagator)->duplicate());
                P(shape)->rs.path = P(path.get())->rs.path;

                auto opacity = repeater->interpOpacity ? mathLerp<uint8_t>(repeater->startOpacity, repeater->endOpacity, static_cast<float>(i + 1) / repeater->cnt) : repeater->startOpacity;
                shape->opacity(opacity);

                Matrix m;
                mathIdentity(&m);
                mathTranslate(&m, repeater->position.x * multiplier + repeater->anchor.x, repeater->position.y * multiplier + repeater->anchor.y);
                mathScale(&m, powf(repeater->scale.x * 0.01f, multiplier), powf(repeater->scale.y * 0.01f, multiplier));
                mathRotate(&m, repeater->rotation * multiplier);
                mathTranslateR(&m, -repeater->anchor.x, -repeater->anchor.y);
                m = repeater->transform * m;

                auto pm = PP(shape)->transform();
                if (pm) {
                    Matrix inverse;
                    mathInverse(&repeater->transform, &inverse);
                    *pm = inverse * *pm;
                }

                shape->transform(pm ? m * *pm : m);
                shapes.push(shape);
            }
        }

        propagators.clear();
        propagators.reserve(shapes.count);

        //push repeat shapes in order.
        if (repeater->inorder) {
            for (auto shape = shapes.begin(); shape < shapes.end(); ++shape) {
                parent->scene->push(cast(*shape));
                propagators.push(*shape);
            }
        } else if (!shapes.empty()) {
            for (auto shape = shapes.end() - 1; shape >= shapes.begin(); --shape) {
                parent->scene->push(cast(*shape));
                propagators.push(*shape);
            }
        }
        shapes.clear();
    }
}


static void _appendRect(Shape* shape, float x, float y, float w, float h, float r, Matrix* transform, bool clockwise)
{
    //sharp rect
    if (mathZero(r)) {
        PathCommand commands[] = {
            PathCommand::MoveTo, PathCommand::LineTo, PathCommand::LineTo,
            PathCommand::LineTo, PathCommand::Close
        };

        Point points[4];
        if (clockwise) {
            points[0] = {x + w, y};
            points[1] = {x + w, y + h};
            points[2] = {x, y + h};
            points[3] = {x, y};
        } else {
            points[0] = {x + w, y};
            points[1] = {x, y};
            points[2] = {x, y + h};
            points[3] = {x + w, y + h};
        }
        if (transform) {
            for (int i = 0; i < 4; i++) {
                points[i] *= *transform;
            }
        }
        shape->appendPath(commands, 5, points, 4);
    //round rect
    } else {
        constexpr int cmdCnt = 10;
        PathCommand commands[cmdCnt];

        auto halfW = w * 0.5f;
        auto halfH = h * 0.5f;
        auto rx = r > halfW ? halfW : r;
        auto ry = r > halfH ? halfH : r;
        auto hrx = rx * PATH_KAPPA;
        auto hry = ry * PATH_KAPPA;

        constexpr int ptsCnt = 17;
        Point points[ptsCnt];
        if (clockwise) {
            commands[0] = PathCommand::MoveTo; commands[1] = PathCommand::LineTo; commands[2] = PathCommand::CubicTo;
            commands[3] = PathCommand::LineTo; commands[4] = PathCommand::CubicTo;commands[5] = PathCommand::LineTo;
            commands[6] = PathCommand::CubicTo; commands[7] = PathCommand::LineTo; commands[8] = PathCommand::CubicTo;
            commands[9] = PathCommand::Close;

            points[0] = {x + w, y + ry}; //moveTo
            points[1] = {x + w, y + h - ry}; //lineTo
            points[2] = {x + w, y + h - ry + hry}; points[3] = {x + w - rx + hrx, y + h}; points[4] = {x + w - rx, y + h}; //cubicTo
            points[5] = {x + rx, y + h}, //lineTo
            points[6] = {x + rx - hrx, y + h}; points[7] = {x, y + h - ry + hry}; points[8] = {x, y + h - ry}; //cubicTo
            points[9] = {x, y + ry}, //lineTo
            points[10] = {x, y + ry - hry}; points[11] = {x + rx - hrx, y}; points[12] = {x + rx, y}; //cubicTo
            points[13] = {x + w - rx, y}; //lineTo
            points[14] = {x + w - rx + hrx, y}; points[15] = {x + w, y + ry - hry}; points[16] = {x + w, y + ry}; //cubicTo
        } else {
            commands[0] = PathCommand::MoveTo; commands[1] = PathCommand::CubicTo; commands[2] = PathCommand::LineTo;
            commands[3] = PathCommand::CubicTo; commands[4] = PathCommand::LineTo; commands[5] = PathCommand::CubicTo;
            commands[6] = PathCommand::LineTo; commands[7] = PathCommand::CubicTo; commands[8] = PathCommand::LineTo;
            commands[9] = PathCommand::Close;

            points[0] = {x + w, y + ry}; //moveTo
            points[1] = {x + w, y + ry - hry}; points[2] = {x + w - rx + hrx, y}; points[3] = {x + w - rx, y}; //cubicTo
            points[4] = {x + rx, y}, //lineTo
            points[5] = {x + rx - hrx, y}; points[6] = {x, y + ry - hry}; points[7] = {x, y + ry}; //cubicTo
            points[8] = {x, y + h - ry}; //lineTo
            points[9] = {x, y + h - ry + hry}; points[10] = {x + rx - hrx, y + h}; points[11] = {x + rx, y + h}; //cubicTo
            points[12] = {x + w - rx, y + h}; //lineTo
            points[13] = {x + w - rx + hrx, y + h}; points[14] = {x + w, y + h - ry + hry}; points[15] = {x + w, y + h - ry}; //cubicTo
            points[16] = {x + w, y + ry}; //lineTo
        }
        if (transform) {
            for (int i = 0; i < ptsCnt; i++) {
                points[i] *= *transform;
            }
        }
        shape->appendPath(commands, cmdCnt, points, ptsCnt);
    }
}

static void _updateRect(LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto rect = static_cast<LottieRect*>(*child);

    auto position = rect->position(frameNo, exps);
    auto size = rect->size(frameNo, exps);
    auto roundness = rect->radius(frameNo, exps);
    if (ctx->roundness > roundness) roundness = ctx->roundness;

    if (roundness > ROUNDNESS_EPSILON) {
        if (roundness > size.x * 0.5f)  roundness = size.x * 0.5f;
        if (roundness > size.y * 0.5f)  roundness = size.y * 0.5f;
    }

    if (!ctx->repeaters.empty()) {
        auto path = Shape::gen();
        _appendRect(path.get(), position.x - size.x * 0.5f, position.y - size.y * 0.5f, size.x, size.y, roundness, ctx->transform, rect->clockwise);
        _repeat(parent, std::move(path), ctx);
    } else {
        _draw(parent, ctx);
        _appendRect(ctx->merging, position.x - size.x * 0.5f, position.y - size.y * 0.5f, size.x, size.y, roundness, ctx->transform, rect->clockwise);
    }
}


static void _appendCircle(Shape* shape, float cx, float cy, float rx, float ry, Matrix* transform)
{
    auto rxKappa = rx * PATH_KAPPA;
    auto ryKappa = ry * PATH_KAPPA;

    constexpr int cmdsCnt = 6;
    PathCommand commands[cmdsCnt] = {
        PathCommand::MoveTo, PathCommand::CubicTo, PathCommand::CubicTo,
        PathCommand::CubicTo, PathCommand::CubicTo, PathCommand::Close
    };

    constexpr int ptsCnt = 13;
    Point points[ptsCnt] = {
        {cx, cy - ry}, //moveTo
        {cx + rxKappa, cy - ry}, {cx + rx, cy - ryKappa}, {cx + rx, cy}, //cubicTo
        {cx + rx, cy + ryKappa}, {cx + rxKappa, cy + ry}, {cx, cy + ry}, //cubicTo
        {cx - rxKappa, cy + ry}, {cx - rx, cy + ryKappa}, {cx - rx, cy}, //cubicTo
        {cx - rx, cy - ryKappa}, {cx - rxKappa, cy - ry}, {cx, cy - ry}  //cubicTo
    };

    if (transform) {
        for (int i = 0; i < ptsCnt; ++i) {
            points[i] *= *transform;
        }
    }
    
    shape->appendPath(commands, cmdsCnt, points, ptsCnt);
}


static void _updateEllipse(LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto ellipse = static_cast<LottieEllipse*>(*child);

    auto position = ellipse->position(frameNo, exps);
    auto size = ellipse->size(frameNo, exps);

    if (!ctx->repeaters.empty()) {
        auto path = Shape::gen();
        _appendCircle(path.get(), position.x, position.y, size.x * 0.5f, size.y * 0.5f, ctx->transform);
        _repeat(parent, std::move(path), ctx);
    } else {
        _draw(parent, ctx);
        _appendCircle(ctx->merging, position.x, position.y, size.x * 0.5f, size.y * 0.5f, ctx->transform);
    }
}


static void _updatePath(LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto path = static_cast<LottiePath*>(*child);

    if (!ctx->repeaters.empty()) {
        auto p = Shape::gen();
        path->pathset(frameNo, P(p)->rs.path.cmds, P(p)->rs.path.pts, ctx->transform, ctx->roundness, exps);
        _repeat(parent, std::move(p), ctx);
    } else {
        _draw(parent, ctx);
        if (path->pathset(frameNo, P(ctx->merging)->rs.path.cmds, P(ctx->merging)->rs.path.pts, ctx->transform, ctx->roundness, exps)) {
            P(ctx->merging)->update(RenderUpdateFlag::Path);
        }
    }
}


static void _applyRoundedCorner(Shape* star, Shape* merging, float outerRoundness, float roundness, bool hasRoundness)
{
    static constexpr auto ROUNDED_POLYSTAR_MAGIC_NUMBER = 0.47829f;

    auto cmdCnt = star->pathCommands(nullptr);
    const Point *pts = nullptr;
    auto ptsCnt = star->pathCoords(&pts);

    auto len = mathLength(pts[1] - pts[2]);
    auto r = len > 0.0f ? ROUNDED_POLYSTAR_MAGIC_NUMBER * mathMin(len * 0.5f, roundness) / len : 0.0f;

    if (hasRoundness) {
        P(merging)->rs.path.cmds.grow((uint32_t)(1.5 * cmdCnt));
        P(merging)->rs.path.pts.grow((uint32_t)(4.5 * cmdCnt));

        int start = 3 * mathZero(outerRoundness);
        merging->moveTo(pts[start].x, pts[start].y);

        for (uint32_t i = 1 + start; i < ptsCnt; i += 6) {
            auto& prev = pts[i];
            auto& curr = pts[i + 2];
            auto& next = (i < ptsCnt - start) ? pts[i + 4] : pts[2];
            auto& nextCtrl = (i < ptsCnt - start) ? pts[i + 5] : pts[3];
            auto dNext = r * (curr - next);
            auto dPrev = r * (curr - prev);

            auto p0 = curr - 2.0f * dPrev;
            auto p1 = curr - dPrev;
            auto p2 = curr - dNext;
            auto p3 = curr - 2.0f * dNext;

            merging->cubicTo(prev.x, prev.y, p0.x, p0.y, p0.x, p0.y);
            merging->cubicTo(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
            merging->cubicTo(p3.x, p3.y, next.x, next.y, nextCtrl.x, nextCtrl.y);
        }
    } else {
        P(merging)->rs.path.cmds.grow(2 * cmdCnt);
        P(merging)->rs.path.pts.grow(4 * cmdCnt);

        auto dPrev = r * (pts[1] - pts[0]);
        auto p = pts[0] + 2.0f * dPrev;
        merging->moveTo(p.x, p.y);

        for (uint32_t i = 1; i < ptsCnt; ++i) {
            auto& curr = pts[i];
            auto& next = (i == ptsCnt - 1) ? pts[1] : pts[i + 1];
            auto dNext = r * (curr - next);

            auto p0 = curr - 2.0f * dPrev;
            auto p1 = curr - dPrev;
            auto p2 = curr - dNext;
            auto p3 = curr - 2.0f * dNext;

            merging->lineTo(p0.x, p0.y);
            merging->cubicTo(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);

            dPrev = -1.0f * dNext;
        }
    }
    merging->close();
}


static void _updateStar(LottieGroup* parent, LottiePolyStar* star, Matrix* transform, float roundness, float frameNo, Shape* merging, LottieExpressions* exps)
{
    static constexpr auto POLYSTAR_MAGIC_NUMBER = 0.47829f / 0.28f;

    auto ptsCnt = star->ptsCnt(frameNo, exps);
    auto innerRadius = star->innerRadius(frameNo, exps);
    auto outerRadius = star->outerRadius(frameNo, exps);
    auto innerRoundness = star->innerRoundness(frameNo, exps) * 0.01f;
    auto outerRoundness = star->outerRoundness(frameNo, exps) * 0.01f;

    auto angle = mathDeg2Rad(-90.0f);
    auto partialPointRadius = 0.0f;
    auto anglePerPoint = (2.0f * MATH_PI / ptsCnt);
    auto halfAnglePerPoint = anglePerPoint * 0.5f;
    auto partialPointAmount = ptsCnt - floorf(ptsCnt);
    auto longSegment = false;
    auto numPoints = size_t(ceilf(ptsCnt) * 2);
    auto direction = star->clockwise ? 1.0f : -1.0f;
    auto hasRoundness = false;
    bool roundedCorner = (roundness > ROUNDNESS_EPSILON) && (mathZero(innerRoundness) || mathZero(outerRoundness));
    //TODO: we can use PathCommand / PathCoord directly.
    auto shape = roundedCorner ? Shape::gen().release() : merging;

    float x, y;

    if (!mathZero(partialPointAmount)) {
        angle += halfAnglePerPoint * (1.0f - partialPointAmount) * direction;
    }

    if (!mathZero(partialPointAmount)) {
        partialPointRadius = innerRadius + partialPointAmount * (outerRadius - innerRadius);
        x = partialPointRadius * cosf(angle);
        y = partialPointRadius * sinf(angle);
        angle += anglePerPoint * partialPointAmount * 0.5f * direction;
    } else {
        x = outerRadius * cosf(angle);
        y = outerRadius * sinf(angle);
        angle += halfAnglePerPoint * direction;
    }

    if (mathZero(innerRoundness) && mathZero(outerRoundness)) {
        P(shape)->rs.path.pts.reserve(numPoints + 2);
        P(shape)->rs.path.cmds.reserve(numPoints + 3);
    } else {
        P(shape)->rs.path.pts.reserve(numPoints * 3 + 2);
        P(shape)->rs.path.cmds.reserve(numPoints + 3);
        hasRoundness = true;
    }

    Point in = {x, y};
    if (transform) in *= *transform;
    shape->moveTo(in.x, in.y);

    for (size_t i = 0; i < numPoints; i++) {
        auto radius = longSegment ? outerRadius : innerRadius;
        auto dTheta = halfAnglePerPoint;
        if (!mathZero(partialPointRadius) && i == numPoints - 2) {
            dTheta = anglePerPoint * partialPointAmount * 0.5f;
        }
        if (!mathZero(partialPointRadius) && i == numPoints - 1) {
            radius = partialPointRadius;
        }
        auto previousX = x;
        auto previousY = y;
        x = radius * cosf(angle);
        y = radius * sinf(angle);

        if (hasRoundness) {
            auto cp1Theta = (mathAtan2(previousY, previousX) - MATH_PI2 * direction);
            auto cp1Dx = cosf(cp1Theta);
            auto cp1Dy = sinf(cp1Theta);
            auto cp2Theta = (mathAtan2(y, x) - MATH_PI2 * direction);
            auto cp2Dx = cosf(cp2Theta);
            auto cp2Dy = sinf(cp2Theta);

            auto cp1Roundness = longSegment ? innerRoundness : outerRoundness;
            auto cp2Roundness = longSegment ? outerRoundness : innerRoundness;
            auto cp1Radius = longSegment ? innerRadius : outerRadius;
            auto cp2Radius = longSegment ? outerRadius : innerRadius;

            auto cp1x = cp1Radius * cp1Roundness * POLYSTAR_MAGIC_NUMBER * cp1Dx / ptsCnt;
            auto cp1y = cp1Radius * cp1Roundness * POLYSTAR_MAGIC_NUMBER * cp1Dy / ptsCnt;
            auto cp2x = cp2Radius * cp2Roundness * POLYSTAR_MAGIC_NUMBER * cp2Dx / ptsCnt;
            auto cp2y = cp2Radius * cp2Roundness * POLYSTAR_MAGIC_NUMBER * cp2Dy / ptsCnt;

            if (!mathZero(partialPointAmount) && ((i == 0) || (i == numPoints - 1))) {
                cp1x *= partialPointAmount;
                cp1y *= partialPointAmount;
                cp2x *= partialPointAmount;
                cp2y *= partialPointAmount;
            }
            Point in2 = {previousX - cp1x, previousY - cp1y};
            Point in3 = {x + cp2x, y + cp2y};
            Point in4 = {x, y};
            if (transform) {
                in2 *= *transform;
                in3 *= *transform;
                in4 *= *transform;
            }
            shape->cubicTo(in2.x, in2.y, in3.x, in3.y, in4.x, in4.y);
        } else {
            Point in = {x, y};
            if (transform) in *= *transform;
            shape->lineTo(in.x, in.y);
        }
        angle += dTheta * direction;
        longSegment = !longSegment;
    }
    shape->close();

    if (roundedCorner) {
        _applyRoundedCorner(shape, merging, outerRoundness, roundness, hasRoundness);
        delete(shape);
    }
}


static void _updatePolygon(LottieGroup* parent, LottiePolyStar* star, Matrix* transform, float frameNo, Shape* merging, LottieExpressions* exps)
{
    static constexpr auto POLYGON_MAGIC_NUMBER = 0.25f;

    auto ptsCnt = size_t(floor(star->ptsCnt(frameNo, exps)));
    auto radius = star->outerRadius(frameNo, exps);
    auto roundness = star->outerRoundness(frameNo, exps) * 0.01f;

    auto angle = mathDeg2Rad(-90.0f);
    auto anglePerPoint = 2.0f * MATH_PI / float(ptsCnt);
    auto direction = star->clockwise ? 1.0f : -1.0f;
    auto hasRoundness = false;
    auto x = radius * cosf(angle);
    auto y = radius * sinf(angle);

    angle += anglePerPoint * direction;

    if (mathZero(roundness)) {
        P(merging)->rs.path.pts.reserve(ptsCnt + 2);
        P(merging)->rs.path.cmds.reserve(ptsCnt + 3);
    } else {
        P(merging)->rs.path.pts.reserve(ptsCnt * 3 + 2);
        P(merging)->rs.path.cmds.reserve(ptsCnt + 3);
        hasRoundness = true;
    }

    Point in = {x, y};
    if (transform) in *= *transform;
    merging->moveTo(in.x, in.y);

    for (size_t i = 0; i < ptsCnt; i++) {
        auto previousX = x;
        auto previousY = y;
        x = (radius * cosf(angle));
        y = (radius * sinf(angle));

        if (hasRoundness) {
            auto cp1Theta = mathAtan2(previousY, previousX) - MATH_PI2 * direction;
            auto cp1Dx = cosf(cp1Theta);
            auto cp1Dy = sinf(cp1Theta);
            auto cp2Theta = mathAtan2(y, x) - MATH_PI2 * direction;
            auto cp2Dx = cosf(cp2Theta);
            auto cp2Dy = sinf(cp2Theta);

            auto cp1x = radius * roundness * POLYGON_MAGIC_NUMBER * cp1Dx;
            auto cp1y = radius * roundness * POLYGON_MAGIC_NUMBER * cp1Dy;
            auto cp2x = radius * roundness * POLYGON_MAGIC_NUMBER * cp2Dx;
            auto cp2y = radius * roundness * POLYGON_MAGIC_NUMBER * cp2Dy;

            Point in2 = {previousX - cp1x, previousY - cp1y};
            Point in3 = {x + cp2x, y + cp2y};
            Point in4 = {x, y};
            if (transform) {
                in2 *= *transform;
                in3 *= *transform;
                in4 *= *transform;
            }
            merging->cubicTo(in2.x, in2.y, in3.x, in3.y, in4.x, in4.y);
        } else {
            Point in = {x, y};
            if (transform) in *= *transform;
            merging->lineTo(in.x, in.y);
        }
        angle += anglePerPoint * direction;
    }
    merging->close();
}


static void _updatePolystar(LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto star= static_cast<LottiePolyStar*>(*child);

    //Optimize: Can we skip the individual coords transform?
    Matrix matrix;
    mathIdentity(&matrix);
    auto position = star->position(frameNo, exps);
    mathTranslate(&matrix, position.x, position.y);
    mathRotate(&matrix, star->rotation(frameNo, exps));

    if (ctx->transform) matrix = *ctx->transform * matrix;

    auto identity = mathIdentity((const Matrix*)&matrix);

    if (!ctx->repeaters.empty()) {
        auto p = Shape::gen();
        if (star->type == LottiePolyStar::Star) _updateStar(parent, star, identity ? nullptr : &matrix, ctx->roundness, frameNo, p.get(), exps);
        else _updatePolygon(parent, star, identity  ? nullptr : &matrix, frameNo, p.get(), exps);
        _repeat(parent, std::move(p), ctx);
    } else {
        _draw(parent, ctx);
        if (star->type == LottiePolyStar::Star) _updateStar(parent, star, identity ? nullptr : &matrix, ctx->roundness, frameNo, ctx->merging, exps);
        else _updatePolygon(parent, star, identity  ? nullptr : &matrix, frameNo, ctx->merging, exps);
        P(ctx->merging)->update(RenderUpdateFlag::Path);
    }
}


static void _updateRoundedCorner(TVG_UNUSED LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto roundedCorner= static_cast<LottieRoundedCorner*>(*child);
    auto roundness = roundedCorner->radius(frameNo, exps);
    if (ctx->roundness < roundness) ctx->roundness = roundness;
}


static void _updateRepeater(TVG_UNUSED LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto repeater= static_cast<LottieRepeater*>(*child);

    RenderRepeater r;
    r.cnt = static_cast<int>(repeater->copies(frameNo, exps));
    if (auto tr = PP(ctx->propagator)->transform()) r.transform = *tr;
    else mathIdentity(&r.transform);
    r.offset = repeater->offset(frameNo, exps);
    r.position = repeater->position(frameNo, exps);
    r.anchor = repeater->anchor(frameNo, exps);
    r.scale = repeater->scale(frameNo, exps);
    r.rotation = repeater->rotation(frameNo, exps);
    r.startOpacity = repeater->startOpacity(frameNo, exps);
    r.endOpacity = repeater->endOpacity(frameNo, exps);
    r.inorder = repeater->inorder;
    r.interpOpacity = (r.startOpacity == r.endOpacity) ? false : true;
    ctx->repeaters.push(r);

    ctx->merging = nullptr;
}


static void _updateTrimpath(TVG_UNUSED LottieGroup* parent, LottieObject** child, float frameNo, TVG_UNUSED Inlist<RenderContext>& contexts, RenderContext* ctx, LottieExpressions* exps)
{
    auto trimpath = static_cast<LottieTrimpath*>(*child);

    float begin, end;
    trimpath->segment(frameNo, begin, end, exps);

    if (P(ctx->propagator)->rs.stroke) {
        auto pbegin = P(ctx->propagator)->rs.stroke->trim.begin;
        auto pend = P(ctx->propagator)->rs.stroke->trim.end;
        auto length = fabsf(pend - pbegin);
        begin = (length * begin) + pbegin;
        end = (length * end) + pbegin;
    }

    P(ctx->propagator)->strokeTrim(begin, end, trimpath->type == LottieTrimpath::Type::Simultaneous);
}


static void _updateChildren(LottieGroup* parent, float frameNo, Inlist<RenderContext>& contexts, LottieExpressions* exps)
{
    contexts.head->begin = parent->children.end() - 1;

    while (!contexts.empty()) {
        auto ctx = contexts.front();
        ctx->reqFragment = parent->reqFragment;
        for (auto child = ctx->begin; child >= parent->children.data; --child) {
            //Here switch-case statements are more performant than virtual methods.
            switch ((*child)->type) {
                case LottieObject::Group: {
                    _updateGroup(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::Transform: {
                    _updateTransform(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::SolidFill: {
                    _updateSolidFill(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::SolidStroke: {
                    _updateSolidStroke(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::GradientFill: {
                    _updateGradientFill(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::GradientStroke: {
                    _updateGradientStroke(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::Rect: {
                    _updateRect(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::Ellipse: {
                    _updateEllipse(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::Path: {
                    _updatePath(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::Polystar: {
                    _updatePolystar(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::Trimpath: {
                    _updateTrimpath(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::Repeater: {
                    _updateRepeater(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                case LottieObject::RoundedCorner: {
                    _updateRoundedCorner(parent, child, frameNo, contexts, ctx, exps);
                    break;
                }
                default: break;
            }
        }
        delete(ctx);
    }
}


static void _updatePrecomp(LottieLayer* precomp, float frameNo, LottieExpressions* exps)
{
    if (precomp->children.empty()) return;

    frameNo = precomp->remap(frameNo, exps);

    for (auto c = precomp->children.end() - 1; c >= precomp->children.begin(); --c) {
        auto child = static_cast<LottieLayer*>(*c);
        if (!child->matteSrc) _updateLayer(precomp, child, frameNo, exps);
    }

    //clip the layer viewport
    if (precomp->w > 0 && precomp->h > 0) {
        auto clipper = Shape::gen().release();
        clipper->appendRect(0, 0, static_cast<float>(precomp->w), static_cast<float>(precomp->h));
        clipper->transform(precomp->cache.matrix);

        //TODO: remove the intermediate scene....
        auto cscene = Scene::gen();
        cscene->composite(cast(clipper), CompositeMethod::ClipPath);
        cscene->push(cast(precomp->scene));
        precomp->scene = cscene.release();
    }
}


static void _updateSolid(LottieLayer* layer)
{
    layer->solidFill->opacity(layer->cache.opacity);
    layer->scene->push(cast(layer->solidFill));
}


static void _updateImage(LottieGroup* layer)
{
    auto image = static_cast<LottieImage*>(layer->children.first());

    if (!image->picture) {
        image->picture = Picture::gen().release();

        //force to load a picture on the same thread
        TaskScheduler::async(false);

        if (image->size > 0) image->picture->load((const char*)image->b64Data, image->size, image->mimeType, false);
        else image->picture->load(image->path);

        TaskScheduler::async(true);

        PP(image->picture)->ref();
        image->picture->size(image->width, image->height);
    }

    if (image->refCnt == 1) layer->scene->push(tvg::cast(image->picture));
    else layer->scene->push(tvg::cast(image->picture->duplicate()));
}


static void _updateText(LottieLayer* layer, float frameNo)
{
    auto text = static_cast<LottieText*>(layer->children.first());
    auto& doc = text->doc(frameNo);
    auto p = doc.text;

    if (!p || !text->font) return;

    auto scale = doc.size * 0.01f;
    Point cursor = {0.0f, 0.0f};
    auto scene = Scene::gen();
    int line = 0;

    //text string
    int idx = 0;
    auto totalChars = strlen(p);
    while (true) {
        //TODO: remove nested scenes.
        //end of text, new line of the cursor position
        if (*p == 13 || *p == 3 || *p == '\0') {
            //text layout position
            auto ascent = text->font->ascent * scale;
            if (ascent > doc.bbox.size.y) ascent = doc.bbox.size.y;
            Point layout = {doc.bbox.pos.x, doc.bbox.pos.y + ascent - doc.shift};

            //adjust the layout
            if (doc.justify == 1) layout.x += doc.bbox.size.x - (cursor.x * scale);  //right aligned
            else if (doc.justify == 2) layout.x += (doc.bbox.size.x * 0.5f) - (cursor.x * 0.5f * scale);  //center aligned

            scene->translate(layout.x, layout.y);
            scene->scale(scale);

            layer->scene->push(std::move(scene));

            if (*p == '\0') break;
            ++p;

            //new text group, single scene for each line
            scene = Scene::gen();
            cursor.x = 0.0f;
            cursor.y = ++line * (doc.height / scale);
        }
        //find the glyph
        bool found = false;
        for (auto g = text->font->chars.begin(); g < text->font->chars.end(); ++g) {
            auto glyph = *g;
            //draw matched glyphs
            if (!strncmp(glyph->code, p, glyph->len)) {
                //TODO: caching?
                auto shape = Shape::gen();
                for (auto g = glyph->children.begin(); g < glyph->children.end(); ++g) {
                    auto group = static_cast<LottieGroup*>(*g);
                    for (auto p = group->children.begin(); p < group->children.end(); ++p) {
                        if (static_cast<LottiePath*>(*p)->pathset(frameNo, P(shape)->rs.path.cmds, P(shape)->rs.path.pts, nullptr, 0.0f)) {
                            P(shape)->update(RenderUpdateFlag::Path);
                        }
                    }
                }
                shape->fill(doc.color.rgb[0], doc.color.rgb[1], doc.color.rgb[2]);
                shape->translate(cursor.x, cursor.y);

                if (doc.stroke.render) {
                    shape->stroke(StrokeJoin::Round);
                    shape->stroke(doc.stroke.width / scale);
                    shape->stroke(doc.stroke.color.rgb[0], doc.stroke.color.rgb[1], doc.stroke.color.rgb[2]);
                }

                //text range process
                for (auto s = text->ranges.begin(); s < text->ranges.end(); ++s) {
                    float divisor = (*s)->rangeUnit == LottieTextRange::Unit::Percent ? (100.0f / totalChars) : 1;
                    auto offset = (*s)->offset(frameNo) / divisor;
                    auto start = round((*s)->start(frameNo) / divisor) + offset;
                    auto end = round((*s)->end(frameNo) / divisor) + offset;

                    if (start > end) std::swap(start, end);

                    if (idx < start || idx >= end) continue;
                    auto matrix = shape->transform();

                    shape->opacity((*s)->style.opacity(frameNo));

                    auto color = (*s)->style.fillColor(frameNo);
                    shape->fill(color.rgb[0], color.rgb[1], color.rgb[2], (*s)->style.fillOpacity(frameNo));

                    mathRotate(&matrix, (*s)->style.rotation(frameNo));

                    auto glyphScale = (*s)->style.scale(frameNo) * 0.01f;
                    mathScale(&matrix, glyphScale.x, glyphScale.y);

                    auto position = (*s)->style.position(frameNo);
                    mathTranslate(&matrix, position.x, position.y);

                    shape->transform(matrix);

                    if (doc.stroke.render) {
                        auto strokeColor = (*s)->style.strokeColor(frameNo);
                        shape->stroke((*s)->style.strokeWidth(frameNo) / scale);
                        shape->stroke(strokeColor.rgb[0], strokeColor.rgb[1], strokeColor.rgb[2], (*s)->style.strokeOpacity(frameNo));
                    }
                    cursor.x += (*s)->style.letterSpacing(frameNo);
                }

                scene->push(std::move(shape));

                p += glyph->len;
                idx += glyph->len;

                //advance the cursor position horizontally
                cursor.x += glyph->width + doc.tracking;

                found = true;
                break;
            }
        }

        if (!found) {
            ++p;
            ++idx;
        }
    }
}


static void _updateMaskings(LottieLayer* layer, float frameNo, LottieExpressions* exps)
{
    if (layer->masks.count == 0) return;

    //Apply the base mask
    auto pMask = static_cast<LottieMask*>(layer->masks[0]);
    auto pMethod = pMask->method;

    auto pShape = Shape::gen().release();
    pShape->fill(255, 255, 255, pMask->opacity(frameNo));
    pShape->transform(layer->cache.matrix);
    if (pMask->pathset(frameNo, P(pShape)->rs.path.cmds, P(pShape)->rs.path.pts, nullptr, 0.0f, exps)) {
        P(pShape)->update(RenderUpdateFlag::Path);
    }

    if (pMethod == CompositeMethod::SubtractMask || pMethod == CompositeMethod::InvAlphaMask) {
        layer->scene->composite(tvg::cast(pShape), CompositeMethod::InvAlphaMask);
    } else {
        layer->scene->composite(tvg::cast(pShape), CompositeMethod::AlphaMask);
    }

    //Apply the subsquent masks
    for (auto m = layer->masks.begin() + 1; m < layer->masks.end(); ++m) {
        auto mask = static_cast<LottieMask*>(*m);
        auto method = mask->method;
        if (method == CompositeMethod::None) continue;

        //Append the mask shape
        if (pMethod == method && (method == CompositeMethod::SubtractMask || method == CompositeMethod::DifferenceMask)) {
            mask->pathset(frameNo, P(pShape)->rs.path.cmds, P(pShape)->rs.path.pts, nullptr, 0.0f, exps);
        //Chain composition
        } else {
            auto shape = Shape::gen().release();
            shape->fill(255, 255, 255, mask->opacity(frameNo));
            shape->transform(layer->cache.matrix);
            if (mask->pathset(frameNo, P(shape)->rs.path.cmds, P(shape)->rs.path.pts, nullptr, 0.0f, exps)) {
                P(shape)->update(RenderUpdateFlag::Path);
            }
            pShape->composite(tvg::cast(shape), method);
            pShape = shape;
            pMethod = method;
        }
    }
}


static bool _updateMatte(LottieLayer* root, LottieLayer* layer, float frameNo, LottieExpressions* exps)
{
    auto target = layer->matteTarget;
    if (!target) return true;

    _updateLayer(root, target, frameNo, exps);

    if (target->scene) {
        layer->scene->composite(cast(target->scene), layer->matteType);
    } else if (layer->matteType == CompositeMethod::AlphaMask || layer->matteType == CompositeMethod::LumaMask) {
        //matte target is not exist. alpha blending definitely bring an invisible result
        delete(layer->scene);
        layer->scene = nullptr;
        return false;
    }
    return true;
}


static void _updateLayer(LottieLayer* root, LottieLayer* layer, float frameNo, LottieExpressions* exps)
{
    layer->scene = nullptr;

    //visibility
    if (frameNo < layer->inFrame || frameNo >= layer->outFrame) return;

    _updateTransform(layer, frameNo, exps);

    //full transparent scene. no need to perform
    if (layer->type != LottieLayer::Null && layer->cache.opacity == 0) return;

    //Prepare render data
    layer->scene = Scene::gen().release();

    //ignore opacity when Null layer?
    if (layer->type != LottieLayer::Null) layer->scene->opacity(layer->cache.opacity);

    layer->scene->transform(layer->cache.matrix);

    if (layer->matteTarget && layer->masks.count > 0) TVGERR("LOTTIE", "FIXME: Matte + Masking??");

    if (!_updateMatte(root, layer, frameNo, exps)) return;

    _updateMaskings(layer, frameNo, exps);

    switch (layer->type) {
        case LottieLayer::Precomp: {
            _updatePrecomp(layer, frameNo, exps);
            break;
        }
        case LottieLayer::Solid: {
            _updateSolid(layer);
            break;
        }
        case LottieLayer::Image: {
            _updateImage(layer);
            break;
        }
        case LottieLayer::Text: {
            _updateText(layer, frameNo);
            break;
        }
        default: {
            if (!layer->children.empty()) {
                Inlist<RenderContext> contexts;
                contexts.back(new RenderContext);
                _updateChildren(layer, frameNo, contexts, exps);
                contexts.free();
            }
            break;
        }
    }

    layer->scene->blend(layer->blendMethod);

    //the given matte source was composited by the target earlier.
    if (!layer->matteSrc) root->scene->push(cast(layer->scene));
}


static void _buildReference(LottieComposition* comp, LottieLayer* layer)
{
    for (auto asset = comp->assets.begin(); asset < comp->assets.end(); ++asset) {
        if (layer->rid != (*asset)->id) continue;
        if (layer->type == LottieLayer::Precomp) {
            auto assetLayer = static_cast<LottieLayer*>(*asset);
            if (_buildComposition(comp, assetLayer)) {
                layer->children = assetLayer->children;
                layer->reqFragment = assetLayer->reqFragment;
            }
        } else if (layer->type == LottieLayer::Image) {
            ++static_cast<LottieImage*>(*asset)->refCnt;
            layer->children.push(*asset);
        }
        break;
    }
}


static void _buildHierarchy(LottieGroup* parent, LottieLayer* child)
{
    if (child->pidx == -1) return;

    if (child->matteTarget && child->pidx == child->matteTarget->idx) {
        child->parent = child->matteTarget;
        return;
    }

    for (auto p = parent->children.begin(); p < parent->children.end(); ++p) {
        auto parent = static_cast<LottieLayer*>(*p);
        if (child == parent) continue;
        if (child->pidx == parent->idx) {
            child->parent = parent;
            break;
        }
        if (parent->matteTarget && parent->matteTarget->idx == child->pidx) {
            child->parent = parent->matteTarget;
            break;
        }
    }
}


static void _attachFont(LottieComposition* comp, LottieLayer* parent)
{
    //TODO: Consider to migrate this attachment to the frame update time.
    for (auto c = parent->children.begin(); c < parent->children.end(); ++c) {
        auto text = static_cast<LottieText*>(*c);
        auto& doc = text->doc(0);
        if (!doc.name) continue;
        auto len = strlen(doc.name);
        for (uint32_t i = 0; i < comp->fonts.count; ++i) {
            auto font = comp->fonts[i];
            auto len2 = strlen(font->name);
            if (len == len2 && !strcmp(font->name, doc.name)) {
                text->font = font;
                break;
            }
        }
    }
}


static bool _buildComposition(LottieComposition* comp, LottieLayer* parent)
{
    if (parent->children.count == 0) return false;
    if (parent->buildDone) return true;
    parent->buildDone = true;

    for (auto c = parent->children.begin(); c < parent->children.end(); ++c) {
        auto child = static_cast<LottieLayer*>(*c);

        //attach the precomp layer.
        if (child->rid) _buildReference(comp, child);

        if (child->matteType != CompositeMethod::None) {
            //no index of the matte layer is provided: the layer above is used as the matte source
            if (child->mid == -1) {
                if (c > parent->children.begin()) {
                    child->matteTarget = static_cast<LottieLayer*>(*(c - 1));
                }
            //matte layer is specified by an index.
            } else child->matteTarget = parent->layerByIdx(child->mid);
        }

        if (child->matteTarget) {
            //parenting
            _buildHierarchy(parent, child->matteTarget);
            //precomp referencing
            if (child->matteTarget->rid) _buildReference(comp, child->matteTarget);
        }
        _buildHierarchy(parent, child);

        //attach the necessary font data
        if (child->type == LottieLayer::Text) _attachFont(comp, child);
    }
    return true;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

bool LottieBuilder::update(LottieComposition* comp, float frameNo)
{
    if (comp->root->children.empty()) return false;

    frameNo += comp->startFrame;
    if (frameNo < comp->startFrame) frameNo = comp->startFrame;
    if (frameNo >= comp->endFrame) frameNo = (comp->endFrame - 1);

    //update children layers
    auto root = comp->root;
    root->scene->clear();

    if (exps && comp->expressions) exps->update(comp->timeAtFrame(frameNo));

    for (auto child = root->children.end() - 1; child >= root->children.begin(); --child) {
        auto layer = static_cast<LottieLayer*>(*child);
        if (!layer->matteSrc) _updateLayer(root, layer, frameNo, exps);
    }

    return true;
}


void LottieBuilder::build(LottieComposition* comp)
{
    if (!comp) return;

    comp->root->scene = Scene::gen().release();
    if (!comp->root->scene) return;

    _buildComposition(comp, comp->root);

    if (!update(comp, 0)) return;

    //viewport clip
    auto clip = Shape::gen();
    clip->appendRect(0, 0, static_cast<float>(comp->w), static_cast<float>(comp->h));
    comp->root->scene->composite(std::move(clip), CompositeMethod::ClipPath);
}
