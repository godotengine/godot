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

#include "tvgMath.h" /* to include math.h before cstring */
#include "tvgShape.h"
#include "tvgCompressor.h"
#include "tvgFill.h"
#include "tvgStr.h"
#include "tvgShape.h"
#include "tvgSvgCommon.h"
#include "tvgSvgBuilder.h"
#include "tvgSvgPath.h"
#include "tvgSvgUtil.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static bool _appendClipShape(SvgParserContext& ctx, SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath, const Matrix* transform);
static bool _applyClip(SvgParserContext& ctx, Paint* paint, Paint* content, const SvgNode* node, const SvgNode* clipNode, const Box& vBox, const string& svgPath, Paint** result);
static Scene* _sceneBuildHelper(SvgParserContext& ctx, const SvgNode* node, const Box& vBox, const string& svgPath, bool mask, int depth);
static Paint* _applyPatternProperty(SvgParserContext& ctx, Shape* vg, SvgNode* node, SvgNode* patternNode, const Box& vBox, const string& svgPath);

static inline bool _isGroupType(SvgNodeType type)
{
    if (type == SvgNodeType::Doc || type == SvgNodeType::G || type == SvgNodeType::Use || type == SvgNodeType::ClipPath || type == SvgNodeType::Symbol || type == SvgNodeType::Filter) return true;
    return false;
}


//According to: https://www.w3.org/TR/SVG11/coords.html#ObjectBoundingBoxUnits (the last paragraph)
//a stroke width should be ignored for bounding box calculations
static Box _bounds(Paint* paint)
{
    float x, y, w, h;
    paint->bounds(&x, &y, &w, &h);
    return {x, y, w, h};
}

static Box _objectBoundingBox(const Box& ratio, const Box& bounds)
{
    return {bounds.x + ratio.x * bounds.w, bounds.y + ratio.y * bounds.h, ratio.w * bounds.w, ratio.h * bounds.h};
}

static inline bool _validBox(const Box& b)
{
    return b.w > 0.0f && b.h > 0.0f;
}

static void _transformMultiply(const Matrix* mBBox, Matrix* gradTransf)
{
    gradTransf->e13 = gradTransf->e13 * mBBox->e11 + mBBox->e13;
    gradTransf->e12 *= mBBox->e11;
    gradTransf->e11 *= mBBox->e11;

    gradTransf->e23 = gradTransf->e23 * mBBox->e22 + mBBox->e23;
    gradTransf->e22 *= mBBox->e22;
    gradTransf->e21 *= mBBox->e22;
}

static LinearGradient* _applyLinearGradientProperty(SvgStyleGradient* g, const Box& vBox, const Box& viewport, int opacity)
{
    Fill::ColorStop* stops;
    auto fillGrad = LinearGradient::gen();
    auto isTransform = (g->transform ? true : false);
    auto& finalTransform = fillGrad->transform();
    if (isTransform) finalTransform = *g->transform;

    if (g->userSpace) {
        fillGrad->linear(g->linear.x1 * (g->linear.isX1Percentage ? vBox.w : viewport.w),
                         g->linear.y1 * (g->linear.isY1Percentage ? vBox.h : viewport.h),
                         g->linear.x2 * (g->linear.isX2Percentage ? vBox.w : viewport.w),
                         g->linear.y2 * (g->linear.isY2Percentage ? vBox.h : viewport.h));
    } else {
        Matrix m = {vBox.w, 0, vBox.x, 0, vBox.h, vBox.y, 0, 0, 1};
        if (isTransform) _transformMultiply(&m, &finalTransform);
        else finalTransform = m;
        fillGrad->linear(g->linear.x1, g->linear.y1, g->linear.x2, g->linear.y2);
    }
    fillGrad->spread(g->spread);

    //Update the stops
    if (g->stops.count == 0) return fillGrad;

    stops = tvg::malloc<Fill::ColorStop>(g->stops.count * sizeof(Fill::ColorStop));
    auto prevOffset = 0.0f;
    for (uint32_t i = 0; i < g->stops.count; ++i) {
        auto colorStop = &g->stops[i];
        //Use premultiplied color
        stops[i].r = colorStop->r;
        stops[i].g = colorStop->g;
        stops[i].b = colorStop->b;
        stops[i].a = static_cast<uint8_t>((colorStop->a * opacity) / 255);
        stops[i].offset = colorStop->offset;
        //check the offset corner cases - refer to: https://svgwg.org/svg2-draft/pservers.html#StopNotes
        if (colorStop->offset < prevOffset) stops[i].offset = prevOffset;
        else if (colorStop->offset > 1) stops[i].offset = 1;
        prevOffset = stops[i].offset;
    }
    fillGrad->colorStops(stops, g->stops.count);
    tvg::free(stops);
    return fillGrad;
}

static RadialGradient* _applyRadialGradientProperty(SvgStyleGradient* g, const Box& vBox, const Box& viewport, int opacity)
{
    Fill::ColorStop *stops;
    auto fillGrad = RadialGradient::gen();
    auto isTransform = (g->transform ? true : false);
    auto& finalTransform = fillGrad->transform();
    if (isTransform) finalTransform = *g->transform;

    if (g->userSpace) {
        //The radius scaling is done according to the Units section:
        //https://www.w3.org/TR/2015/WD-SVG2-20150915/coords.html
        auto diag = sqrtf(powf(vBox.w, 2.0f) + powf(vBox.h, 2.0f)) / sqrtf(2.0f);
        auto viewportDiag = sqrtf(powf(viewport.w, 2.0f) + powf(viewport.h, 2.0f)) / sqrtf(2.0f);
        fillGrad->radial(g->radial.cx * (g->radial.isCxPercentage ? vBox.w : viewport.w),
                         g->radial.cy * (g->radial.isCyPercentage ? vBox.h : viewport.h),
                         g->radial.r * (g->radial.isRPercentage ? diag : viewportDiag),
                         g->radial.fx * (g->radial.isFxPercentage ? vBox.w : viewport.w),
                         g->radial.fy * (g->radial.isFyPercentage ? vBox.h : viewport.h),
                         g->radial.fr * (g->radial.isFrPercentage ? diag : viewportDiag));
    } else {
        Matrix m = {vBox.w, 0, vBox.x, 0, vBox.h, vBox.y, 0, 0, 1};
        if (isTransform) _transformMultiply(&m, &finalTransform);
        else finalTransform = m;
        fillGrad->radial(g->radial.cx, g->radial.cy, g->radial.r, g->radial.fx, g->radial.fy, g->radial.fr);
    }
    fillGrad->spread(g->spread);

    //Update the stops
    if (g->stops.count == 0) return fillGrad;

    stops = tvg::malloc<Fill::ColorStop>(g->stops.count * sizeof(Fill::ColorStop));
    auto prevOffset = 0.0f;
    for (uint32_t i = 0; i < g->stops.count; ++i) {
        auto colorStop = &g->stops[i];
        //Use premultiplied color
        stops[i].r = colorStop->r;
        stops[i].g = colorStop->g;
        stops[i].b = colorStop->b;
        stops[i].a = static_cast<uint8_t>((colorStop->a * opacity) / 255);
        stops[i].offset = colorStop->offset;
        //check the offset corner cases - refer to: https://svgwg.org/svg2-draft/pservers.html#StopNotes
        if (colorStop->offset < prevOffset) stops[i].offset = prevOffset;
        else if (colorStop->offset > 1) stops[i].offset = 1;
        prevOffset = stops[i].offset;
    }
    fillGrad->colorStops(stops, g->stops.count);
    tvg::free(stops);
    return fillGrad;
}


static void _appendRect(Shape* shape, float x, float y, float w, float h, float rx, float ry)
{
    auto halfW = w * 0.5f;
    auto halfH = h * 0.5f;

    //clamping cornerRadius by minimum size
    if (rx > halfW) rx = halfW;
    if (ry > halfH) ry = halfH;

    if (rx == 0 && ry == 0) {
        to<ShapeImpl>(shape)->grow(5, 4);
        shape->moveTo(x, y);
        shape->lineTo(x + w, y);
        shape->lineTo(x + w, y + h);
        shape->lineTo(x, y + h);
        shape->close();
    } else {
        auto hrx = rx * PATH_KAPPA;
        auto hry = ry * PATH_KAPPA;

        to<ShapeImpl>(shape)->grow(10, 17);
        shape->moveTo(x + rx, y);
        shape->lineTo(x + w - rx, y);
        shape->cubicTo(x + w - rx + hrx, y, x + w, y + ry - hry, x + w, y + ry);
        shape->lineTo(x + w, y + h - ry);
        shape->cubicTo(x + w, y + h - ry + hry, x + w - rx + hrx, y + h, x + w - rx, y + h);
        shape->lineTo(x + rx, y + h);
        shape->cubicTo(x + rx - hrx, y + h, x, y + h - ry + hry, x, y + h - ry);
        shape->lineTo(x, y + ry);
        shape->cubicTo(x, y + ry - hry, x + rx - hrx, y, x + rx, y);
        shape->close();
    }
}


static void _appendCircle(Shape* shape, float cx, float cy, float rx, float ry)
{
    auto rxKappa = rx * PATH_KAPPA;
    auto ryKappa = ry * PATH_KAPPA;

    to<ShapeImpl>(shape)->grow(6, 13);
    shape->moveTo(cx + rx, cy);
    shape->cubicTo(cx + rx, cy + ryKappa, cx + rxKappa, cy + ry, cx, cy + ry);
    shape->cubicTo(cx - rxKappa, cy + ry, cx - rx, cy + ryKappa, cx - rx, cy);
    shape->cubicTo(cx - rx, cy - ryKappa, cx - rxKappa, cy - ry, cx, cy - ry);
    shape->cubicTo(cx + rxKappa, cy - ry, cx + rx, cy - ryKappa, cx + rx, cy);
    shape->close();
}

static bool _appendClipChild(SvgParserContext& ctx, SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath, Paint** clipped)
{
    Matrix finalTransform;
    const Matrix* transform = nullptr;

    //The SVG standard allows only for 'use' nodes that point directly to a basic shape.
    if (node->type == SvgNodeType::Use) {
        if (node->child.count != 1) {
            *clipped = nullptr;
            return false;
        }
        auto child = *(node->child.data);
        finalTransform = tvg::identity();
        if (node->transform) finalTransform = *node->transform;
        if (node->node.use.x != 0.0f || node->node.use.y != 0.0f) {
            finalTransform *= {1, 0, node->node.use.x, 0, 1, node->node.use.y, 0, 0, 1};
        }
        if (child->transform) finalTransform *= *child->transform;

        transform = tvg::identity((const Matrix*)(&finalTransform)) ? nullptr : &finalTransform;
        node = child;
    }

    if (!_appendClipShape(ctx, node, shape, vBox, svgPath, transform)) {
        *clipped = nullptr;
        return false;
    }

    // Apply Clip Chaining
    if (auto clipNode = node->style->clipPath.node) {
        if (node->style->clipPath.applying) {
            TVGLOG("SVG", "Multiple composition tried! Check out circular dependency?");
            *clipped = nullptr;
            return false;
        }
        return _applyClip(ctx, shape, shape, node, clipNode, vBox, svgPath, clipped);
    }

    *clipped = shape;
    return true;
}

static Matrix _useTransform(const SvgNode* node)
{
    auto m = tvg::identity();
    if (node->transform) m = *node->transform;
    if (node->node.use.x != 0.0f || node->node.use.y != 0.0f) {
        m *= {1, 0, node->node.use.x, 0, 1, node->node.use.y, 0, 0, 1};
    }
    return m;
}

static Matrix _compositionTransform(Paint* content, const SvgNode* node, const SvgNode* compNode, SvgNodeType type)
{
    auto m = tvg::identity();
    auto userSpace = (type == SvgNodeType::Mask) ? compNode->node.mask.maskContentUserSpace : compNode->node.clip.userSpace;
    //The initial mask transformation ignored according to the SVG standard.
    if (node->type == SvgNodeType::Use) {
        if (type != SvgNodeType::Mask || !userSpace) m = _useTransform(node);
    } else if (node->transform && type != SvgNodeType::Mask) {
        m = *node->transform;
    }
    if (compNode->transform) {
        m *= *compNode->transform;
    }
    if (!userSpace) {
        auto bbox = Box{};
        if (node->type == SvgNodeType::Use) {
            auto useTransform = _useTransform(node);
            Matrix inv;
            auto prevTransform = content->transform();
            if (inverse(&useTransform, &inv)) content->transform(inv * prevTransform);
            bbox = _bounds(content);
            content->transform(prevTransform);
        } else {
            bbox = _bounds(content);
        }
        m *= {bbox.w, 0, bbox.x, 0, bbox.h, bbox.y, 0, 0, 1};
    }
    return m;
}

static bool _clipperUnion(SvgParserContext& ctx, Paint* content, const SvgNode* node, const SvgNode* clipNode, const Box& vBox, const string& svgPath, Paint** region)
{
    node->style->clipPath.applying = true;

    Paint* unionRegion = nullptr;
    Scene* scene = nullptr;
    ARRAY_FOREACH(p, clipNode->child) {
        auto child = Shape::gen();
        Paint* clipped = nullptr;
        if (!_appendClipChild(ctx, *p, child, vBox, svgPath, &clipped)) {
            Paint::rel(child);
            continue;
        }
        child->fill(255, 255, 255, 255);
        if (!unionRegion) unionRegion = clipped;
        else {
            if (!scene) {
                scene = Scene::gen();
                scene->add(unionRegion);
                unionRegion = scene;
            }
            scene->add(clipped);
        }
    }

    node->style->clipPath.applying = false;

    if (!unionRegion) {
        *region = nullptr;
        return false;
    }

    unionRegion->transform(_compositionTransform(content, node, clipNode, SvgNodeType::ClipPath));
    *region = unionRegion;
    return true;
}

static bool _applyClip(SvgParserContext& ctx, Paint* paint, Paint* content, const SvgNode* node, const SvgNode* clipNode, const Box& vBox, const string& svgPath, Paint** result)
{
    Paint* region = nullptr;
    if (!_clipperUnion(ctx, content, node, clipNode, vBox, svgPath, &region)) {
        *result = nullptr;
        return false;
    }

    paint->mask(region, MaskMethod::Alpha);

    *result = paint;
    if (auto innerClipNode = clipNode->style->clipPath.node) {
        if (!innerClipNode->style->clipPath.applying) {
            innerClipNode->style->clipPath.applying = true;
            auto scene = Scene::gen();
            scene->add(paint);
            if (!_applyClip(ctx, scene, content, node, innerClipNode, vBox, svgPath, result)) {
                scene->opacity(0);
                *result = scene;
            }
            innerClipNode->style->clipPath.applying = false;
        }
    }
    return true;
}

static Scene* _applyMask(SvgParserContext& ctx, Paint* content, Scene* target, const SvgNode* node, const SvgNode* maskNode, const Box& vBox, const string& svgPath, bool wrap)
{
    node->style->mask.applying = true;

    Scene* result = target;
    if (auto mask = _sceneBuildHelper(ctx, maskNode, vBox, svgPath, true, 0)) {
        auto& maskData = maskNode->node.mask;
        auto nodeTransform = (node->type == SvgNodeType::Use) ? _useTransform(node) : (node->transform ? *node->transform : tvg::identity());
        if (!maskData.maskContentUserSpace) {
            Matrix finalTransform = _compositionTransform(content, node, maskNode, SvgNodeType::Mask);
            mask->transform(finalTransform);
        } else if (!tvg::identity((const Matrix*)(&nodeTransform))) {
            mask->transform(nodeTransform);
        }

        auto bbox = _bounds(content);
        auto clipper = Shape::gen();
        if (maskData.userSpace) {
            clipper->appendRect(maskData.box.x, maskData.box.y, maskData.box.w, maskData.box.h);
            if (!tvg::identity((const Matrix*)(&nodeTransform))) clipper->transform(nodeTransform);
        } else {
            auto box = _objectBoundingBox(maskData.box, bbox);
            clipper->appendRect(box.x, box.y, box.w, box.h);
        }
        mask->clip(clipper);

        if (wrap) {
            auto scene = Scene::gen();
            scene->add(target);
            result = scene;
        }
        result->mask(mask, maskData.type == SvgMaskType::Luminance ? MaskMethod::Luma : MaskMethod::Alpha);
    }

    node->style->mask.applying = false;
    return result;
}

static Paint* _applyBlend(Paint* paint, const SvgNode* node)
{
    if (paint && (node->style->flags & SvgStyleFlags::BlendMode)) {
        paint->blend(node->style->blendMode);
    }
    return paint;
}


static Paint* _applyComposition(SvgParserContext& ctx, Paint* paint, const SvgNode* node, const Box& vBox, const string& svgPath)
{
    if (node->style->clipPath.applying || node->style->mask.applying) {
        TVGLOG("SVG", "Multiple composition tried! Check out circular dependency?");
        return paint;
    }

    auto clipNode = node->style->clipPath.node;
    auto maskNode = node->style->mask.node;

    if (!clipNode && !maskNode) return paint;
    if ((clipNode && clipNode->child.empty()) || (maskNode && maskNode->child.empty())) {
        Paint::rel(paint);
        return nullptr;
    }

    auto scene = Scene::gen();
    scene->add(paint);

    if (clipNode) {
        Paint* clipped = nullptr;
        if (!_applyClip(ctx, scene, paint, node, clipNode, vBox, svgPath, &clipped)) {
            Paint::rel(scene);
            return nullptr;
        }
        scene = static_cast<Scene*>(clipped);
    }

    if (maskNode) scene = _applyMask(ctx, paint, scene, node, maskNode, vBox, svgPath, clipNode != nullptr);

    return scene;
}

static Paint* _applyFilter(SvgParserContext& ctx, Paint* paint, const SvgNode* node, const Box& vBox, const string& svgPath)
{
    auto filterNode = node->style->filter.node;
    if (!filterNode || filterNode->child.count == 0) return paint;

    auto& filter = filterNode->node.filter;
    auto scene = Scene::gen();
    auto bbox = _bounds(paint);
    auto clipBox = filter.filterUserSpace ? filter.box : _objectBoundingBox(filter.box, bbox);
    auto primitiveUserSpace = filter.primitiveUserSpace;
    auto sx = paint->transform().e11;
    auto sy = paint->transform().e22;

    auto child = filterNode->child.data;
    for (uint32_t i = 0; i < filterNode->child.count; ++i, ++child) {
        if ((*child)->type == SvgNodeType::GaussianBlur) {
            auto& gauss = (*child)->node.gaussianBlur;

            auto direction = gauss.stdDevX > 0.0f ? (gauss.stdDevY > 0.0f ? 0 : 1) : (gauss.stdDevY > 0.0f ? 2 : -1);
            if (direction == -1) continue;

            auto stdDevX = gauss.stdDevX;
            auto stdDevY = gauss.stdDevY;
            if (gauss.hasBox) {
                auto gaussBox = gauss.box;
                auto isPercent = gauss.isPercentage;
                if (primitiveUserSpace) {
                    if (isPercent[0]) gaussBox.x *= ctx.parser->global.w;
                    if (isPercent[1]) gaussBox.y *= ctx.parser->global.h;
                    if (isPercent[2]) gaussBox.w *= ctx.parser->global.w;
                    if (isPercent[3]) gaussBox.h *= ctx.parser->global.h;
                } else {
                    stdDevX *= bbox.w;
                    stdDevY *= bbox.h;
                    if (isPercent[0]) gaussBox.x = bbox.x + gauss.box.x * bbox.w;
                    if (isPercent[1]) gaussBox.y = bbox.y + gauss.box.y * bbox.h;
                    if (isPercent[2]) gaussBox.w *= bbox.w;
                    if (isPercent[3]) gaussBox.h *= bbox.h;
                }
                clipBox.intersect(gaussBox);
            } else if (!primitiveUserSpace) {
                stdDevX *= bbox.w;
                stdDevY *= bbox.h;
            }
            scene->add(SceneEffect::GaussianBlur, (double)(1.25f * (direction == 2 ? stdDevY * sy : stdDevX * sx)), direction, gauss.edgeModeWrap, 55);
        }
    }

    scene->add(paint);

    auto clip = Shape::gen();
    clip->appendRect(clipBox.x, clipBox.y, clipBox.w, clipBox.h);
    scene->clip(clip);

    return scene;
}

static void _applyStroke(SvgStyleProperty* style, Shape* vg, const Box& vBox, const Box& viewport)
{
    vg->strokeWidth(style->stroke.width);
    vg->strokeCap(style->stroke.cap);
    vg->strokeJoin(style->stroke.join);
    vg->strokeMiterlimit(style->stroke.miterlimit);
    vg->strokeDash(style->stroke.dash.array.data, style->stroke.dash.array.count, style->stroke.dash.offset);

    if (style->stroke.paint.none) {
        vg->strokeWidth(0.0f);
    } else if (style->stroke.paint.gradient) {
        auto bBox = style->stroke.paint.gradient->userSpace ? vBox : _bounds(vg);
        if (style->stroke.paint.gradient->type == SvgGradientType::Linear) {
            vg->strokeFill(_applyLinearGradientProperty(style->stroke.paint.gradient, bBox, viewport, style->stroke.opacity));
        } else if (style->stroke.paint.gradient->type == SvgGradientType::Radial) {
            vg->strokeFill(_applyRadialGradientProperty(style->stroke.paint.gradient, bBox, viewport, style->stroke.opacity));
        }
    } else if (style->stroke.paint.url) {
        TVGLOG("SVG", "The stroke's url not supported.");
    } else if (style->stroke.paint.curColor) {
        vg->strokeFill(style->color.r, style->color.g, style->color.b, style->stroke.opacity);
    } else {
        vg->strokeFill(style->stroke.paint.color.r, style->stroke.paint.color.g, style->stroke.paint.color.b, style->stroke.opacity);
    }
}

static Paint* _applyProperty(SvgParserContext& ctx, SvgNode* node, Shape* vg, const Box& vBox, const string& svgPath, bool clip)
{
    SvgStyleProperty* style = node->style;

    //Clip transformation is applied directly to the path in the _appendClipShape function
    if (node->type == SvgNodeType::Doc || !node->style->display) return vg;

    //If fill property is nullptr then do nothing
    if (style->fill.paint.none) {
        //Do nothing
    } else if (style->fill.paint.gradient) {
        auto bBox = style->fill.paint.gradient->userSpace ? vBox : _bounds(vg);
        if (style->fill.paint.gradient->type == SvgGradientType::Linear) {
            vg->fill(_applyLinearGradientProperty(style->fill.paint.gradient, bBox, ctx.parser->global, style->fill.opacity));
        } else if (style->fill.paint.gradient->type == SvgGradientType::Radial) {
            vg->fill(_applyRadialGradientProperty(style->fill.paint.gradient, bBox, ctx.parser->global, style->fill.opacity));
        }
    } else if (style->fill.paint.pattern) {
        if (auto patternPaint = _applyPatternProperty(ctx, vg, node, style->fill.paint.pattern, vBox, svgPath)) {
            vg->fillRule(style->fill.fillRule);
            vg->order(!style->paintOrder);
            _applyStroke(style, vg, vBox, ctx.parser->global);
            auto patternScene = Scene::gen();
            patternScene->add(patternPaint);
            patternScene->add(vg);
            patternScene->opacity(style->opacity);
            if (node->transform && !clip) patternScene->transform(*node->transform);
            auto p = _applyFilter(ctx, patternScene, node, vBox, svgPath);
            p = _applyComposition(ctx, p, node, vBox, svgPath);
            return _applyBlend(p, node);
        }
    } else if (style->fill.paint.url) {
        TVGLOG("SVG", "The fill's url not supported.");
    } else if (style->fill.paint.curColor) {
        //Apply the current style color
        vg->fill(style->color.r, style->color.g, style->color.b, style->fill.opacity);
    } else {
        //Apply the fill color
        vg->fill(style->fill.paint.color.r, style->fill.paint.color.g, style->fill.paint.color.b, style->fill.opacity);
    }

    vg->fillRule(style->fill.fillRule);
    vg->order(!style->paintOrder);
    vg->opacity(style->opacity);

    if (node->type == SvgNodeType::G || node->type == SvgNodeType::Use) {
        if (style->flags & SvgStyleFlags::BlendMode) vg->blend(style->blendMode);
        return vg;
    }

    _applyStroke(style, vg, vBox, ctx.parser->global);

    //apply transform after the local space shape bbox for gradient acquisition
    if (node->transform && !clip) vg->transform(*node->transform);

    auto p = _applyFilter(ctx, vg, node, vBox, svgPath);
    p = _applyComposition(ctx, p, node, vBox, svgPath);
    return _applyBlend(p, node);
}


static bool _recognizeShape(SvgNode* node, Shape* shape)
{
    switch (node->type) {
        case SvgNodeType::Path: {
            if (node->node.path.path) {
                if (!svgPathToShape(node->node.path.path, to<ShapeImpl>(shape)->rs.path)) {
                    TVGERR("SVG", "Invalid path information.");
                    return false;
                }
            }
            break;
        }
        case SvgNodeType::Ellipse: {
            _appendCircle(shape, node->node.ellipse.cx, node->node.ellipse.cy, node->node.ellipse.rx, node->node.ellipse.ry);
            break;
        }
        case SvgNodeType::Polygon: {
            if (node->node.polygon.pts.count < 2) break;
            auto pts = node->node.polygon.pts.begin();
            shape->moveTo(pts[0], pts[1]);
            for (pts += 2; pts < node->node.polygon.pts.end(); pts += 2) {
                shape->lineTo(pts[0], pts[1]);
            }
            shape->close();
            break;
        }
        case SvgNodeType::Polyline: {
            if (node->node.polyline.pts.count < 2) break;
            auto pts = node->node.polyline.pts.begin();
            shape->moveTo(pts[0], pts[1]);
            for (pts += 2; pts < node->node.polyline.pts.end(); pts += 2) {
                shape->lineTo(pts[0], pts[1]);
            }
            break;
        }
        case SvgNodeType::Circle: {
            _appendCircle(shape, node->node.circle.cx, node->node.circle.cy, node->node.circle.r, node->node.circle.r);
            break;
        }
        case SvgNodeType::Rect: {
            _appendRect(shape, node->node.rect.x, node->node.rect.y, node->node.rect.w, node->node.rect.h, node->node.rect.rx, node->node.rect.ry);
            break;
        }
        case SvgNodeType::Line: {
            shape->moveTo(node->node.line.x1, node->node.line.y1);
            shape->lineTo(node->node.line.x2, node->node.line.y2);
            break;
        }
        default: {
            return false;
        }
    }
    return true;
}

static Paint* _shapeBuildHelper(SvgParserContext& ctx, SvgNode* node, const Box& vBox, const string& svgPath)
{
    auto shape = Shape::gen();
    if (!_recognizeShape(node, shape)) return nullptr;
    return _applyProperty(ctx, node, shape, vBox, svgPath, false);
}

static bool _appendClipShape(SvgParserContext& ctx, SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath, const Matrix* transform)
{
    uint32_t currentPtsCnt;
    shape->path(nullptr, nullptr, nullptr, &currentPtsCnt);

    if (!_recognizeShape(node, shape)) return false;

    //The 'transform' matrix has higher priority than the node->transform, since it already contains it
    auto m = transform ? transform : (node->transform ? node->transform : nullptr);

    if (m) {
        const Point *pts;
        uint32_t ptsCnt;
        shape->path(nullptr, nullptr, &pts, &ptsCnt);
        auto p = const_cast<Point*>(pts) + currentPtsCnt;
        while (currentPtsCnt++ < ptsCnt) {
            *p *= *m;
            ++p;
        }
    }

    return true;
}


enum class imageMimeTypeEncoding
{
    base64 = 0x1,
    utf8 = 0x2
};


constexpr imageMimeTypeEncoding operator|(imageMimeTypeEncoding a, imageMimeTypeEncoding b) {
    return static_cast<imageMimeTypeEncoding>(static_cast<int>(a) | static_cast<int>(b));
}


constexpr bool operator&(imageMimeTypeEncoding a, imageMimeTypeEncoding b) {
    return (static_cast<int>(a) & static_cast<int>(b));
}


static constexpr struct
{
    const char* name;
    int sz;
    imageMimeTypeEncoding encoding;
} imageMimeTypes[] = {
    {"jpeg", sizeof("jpeg"), imageMimeTypeEncoding::base64},
    {"png", sizeof("png"), imageMimeTypeEncoding::base64},
    {"webp", sizeof("webp"), imageMimeTypeEncoding::base64},
    {"svg+xml", sizeof("svg+xml"), imageMimeTypeEncoding::base64 | imageMimeTypeEncoding::utf8},
};


static bool _isValidImageMimeTypeAndEncoding(const char** href, const char** mimetype, imageMimeTypeEncoding* encoding) {
    if (strncasecmp(*href, "image/", sizeof("image/") - 1)) return false;  // not allowed mime type
    *href += sizeof("image/") - 1;

    //RFC2397 data:[<mediatype>][;base64],<data>
    //mediatype  := [ type "/" subtype ] *( ";" parameter )
    //parameter  := attribute "=" value
    for (unsigned int i = 0; i < sizeof(imageMimeTypes) / sizeof(imageMimeTypes[0]); i++) {
        if (strncasecmp(*href, imageMimeTypes[i].name, imageMimeTypes[i].sz - 1)) continue;
        *href += imageMimeTypes[i].sz  - 1;
        *mimetype = imageMimeTypes[i].name;

        while (**href && **href != ',') {
            while (**href && **href != ';') ++(*href);
            if (!**href) return false;
            ++(*href);

            if (imageMimeTypes[i].encoding & imageMimeTypeEncoding::base64) {
                if (!strncasecmp(*href, "base64,", sizeof("base64,") - 1)) {
                    *href += sizeof("base64,") - 1;
                    *encoding = imageMimeTypeEncoding::base64;
                    return true; //valid base64
                }
            }
            if (imageMimeTypes[i].encoding & imageMimeTypeEncoding::utf8) {
                if (!strncasecmp(*href, "utf8,", sizeof("utf8,") - 1)) {
                    *href += sizeof("utf8,") - 1;
                    *encoding = imageMimeTypeEncoding::utf8;
                    return true; //valid utf8
                }
            }
        }
        //no encoding defined
        if (**href == ',' && (imageMimeTypes[i].encoding & imageMimeTypeEncoding::utf8)) {
            ++(*href);
            *encoding = imageMimeTypeEncoding::utf8;
            return true; //allow no encoding defined if utf8 expected
        }
        return false;
    }
    return false;
}

#include "tvgTaskScheduler.h"

static Paint* _imageBuildHelper(SvgParserContext& ctx, SvgNode* node, const Box& vBox, const string& svgPath)
{
    if (!node->node.image.href || !strlen(node->node.image.href)) return nullptr;

    auto picture = Picture::gen();

    const char* href = node->node.image.href;
    if (!strncmp(href, "data:", sizeof("data:") - 1)) {
        href += sizeof("data:") - 1;
        const char* mimetype;
        imageMimeTypeEncoding encoding;
        if (!_isValidImageMimeTypeAndEncoding(&href, &mimetype, &encoding)) return nullptr; //not allowed mime type or encoding
        char *decoded = nullptr;
        if (encoding == imageMimeTypeEncoding::base64) {
            auto size = b64Decode(href, strlen(href), &decoded);
            if (picture->load(decoded, size, mimetype) != Result::Success) {
                tvg::free(decoded);
                return nullptr;
            }
        } else {
            auto size = svgUtilURLDecode(href, &decoded);
            if (picture->load(decoded, size, mimetype) != Result::Success) {
                tvg::free(decoded);
                return nullptr;
            }
        }
        ctx.images.push(decoded);
    } else {
        if (!strncmp(href, "file://", sizeof("file://") - 1)) href += sizeof("file://") - 1;
        //TODO: protect against recursive svg image loading
        //Temporarily disable embedded svg:
        const char *dot = strrchr(href, '.');
        if (dot && STR_AS(dot, ".svg")) {
            TVGLOG("SVG", "Embedded svg file is disabled.");
            return nullptr;
        }
        string imagePath = href;
        if (strncmp(href, "/", 1)) {
            auto last = svgPath.find_last_of("/");
            imagePath = svgPath.substr(0, (last == string::npos ? 0 : last + 1)) + imagePath;
        }
        if (picture->load(imagePath.c_str()) != Result::Success) {
            return nullptr;
        }
    }

    float w, h;
    Matrix m;
    if (picture->size(&w, &h) == Result::Success && w > 0 && h > 0) {
        auto sx = node->node.image.w / w;
        auto sy = node->node.image.h / h;
        m = {sx, 0, node->node.image.x, 0, sy, node->node.image.y, 0, 0, 1};
    } else {
        m = tvg::identity();
    }
    if (node->transform) m = *node->transform * m;
    picture->transform(m);

    auto p = _applyFilter(ctx, picture, node, vBox, svgPath);
    p = _applyComposition(ctx, p, node, vBox, svgPath);
    return _applyBlend(p, node);
}


static Matrix _calculateAspectRatioMatrix(AspectRatioAlign align, AspectRatioMeetOrSlice meetOrSlice, float width, float height, const Box& box)
{
    auto sx = width / box.w;
    auto sy = height / box.h;
    auto tvx = box.x * sx;
    auto tvy = box.y * sy;

    if (align == AspectRatioAlign::None) return {sx, 0, -tvx, 0, sy, -tvy, 0, 0, 1};

    //Scale
    if (meetOrSlice == AspectRatioMeetOrSlice::Meet) {
        if (sx < sy) sy = sx;
        else sx = sy;
    } else {
        if (sx < sy) sx = sy;
        else sy = sx;
    }

    //Align
    tvx = box.x * sx;
    tvy = box.y * sy;
    auto tvw = box.w * sx;
    auto tvh = box.h * sy;

    switch (align) {
        case AspectRatioAlign::XMinYMin: {
            break;
        }
        case AspectRatioAlign::XMidYMin: {
            tvx -= (width - tvw) * 0.5f;
            break;
        }
        case AspectRatioAlign::XMaxYMin: {
            tvx -= width - tvw;
            break;
        }
        case AspectRatioAlign::XMinYMid: {
            tvy -= (height - tvh) * 0.5f;
            break;
        }
        case AspectRatioAlign::XMidYMid: {
            tvx -= (width - tvw) * 0.5f;
            tvy -= (height - tvh) * 0.5f;
            break;
        }
        case AspectRatioAlign::XMaxYMid: {
            tvx -= width - tvw;
            tvy -= (height - tvh) * 0.5f;
            break;
        }
        case AspectRatioAlign::XMinYMax: {
            tvy -= height - tvh;
            break;
        }
        case AspectRatioAlign::XMidYMax: {
            tvx -= (width - tvw) * 0.5f;
            tvy -= height - tvh;
            break;
        }
        case AspectRatioAlign::XMaxYMax: {
            tvx -= width - tvw;
            tvy -= height - tvh;
            break;
        }
        default: {
            break;
        }
    }

    return {sx, 0, -tvx, 0, sy, -tvy, 0, 0, 1};
}

static Matrix _symbolTransform(const SvgNode* node, const Box& vBox)
{
    auto& symbol = node->node.use.symbol->node.symbol;
    auto width = (symbol.hasWidth ? symbol.w : vBox.w);
    if (node->node.use.isWidthSet) width = node->node.use.w;
    auto height = (symbol.hasHeight ? symbol.h : vBox.h);
    if (node->node.use.isHeightSet) height = node->node.use.h;
    auto vw = (symbol.hasViewBox ? symbol.vw : width);
    auto vh = (symbol.hasViewBox ? symbol.vh : height);

    auto mViewBox = tvg::identity();
    if ((!tvg::equal(width, vw) || !tvg::equal(height, vh)) && vw > 0 && vh > 0) {
        Box box = {symbol.vx, symbol.vy, vw, vh};
        mViewBox = _calculateAspectRatioMatrix(symbol.align, symbol.meetOrSlice, width, height, box);
    } else if (!tvg::zero(symbol.vx) || !tvg::zero(symbol.vy)) {
        mViewBox = {1, 0, -symbol.vx, 0, 1, -symbol.vy, 0, 0, 1};
    }

    // mSceneTransform = mUseTransform * mSymbolTransform * mViewBox
    Matrix mSceneTransform = mViewBox;
    if (node->node.use.symbol->transform) {
        mSceneTransform = *node->node.use.symbol->transform * mViewBox;
    }
    return _useTransform(node) * mSceneTransform;
}

static Scene* _useBuildHelper(SvgParserContext& ctx, const SvgNode* node, const Box& vBox, const string& svgPath, int depth)
{
    auto scene = _sceneBuildHelper(ctx, node, vBox, svgPath, false, depth + 1);

    if (node->node.use.symbol) {
        if (!node->node.use.symbol->node.symbol.overflowVisible) {
            auto& symbol = node->node.use.symbol->node.symbol;
            auto width = (symbol.hasWidth ? symbol.w : vBox.w);
            if (node->node.use.isWidthSet) width = node->node.use.w;
            auto height = (symbol.hasHeight ? symbol.h : vBox.h);
            if (node->node.use.isHeightSet) height = node->node.use.h;

            auto viewBoxClip = Shape::gen();
            viewBoxClip->appendRect(0, 0, width, height);

            // mClipTransform = mUseTransform * mSymbolTransform
            Matrix mClipTransform = _useTransform(node);
            if (node->node.use.symbol->transform) {
                mClipTransform = mClipTransform * *node->node.use.symbol->transform;
            }
            viewBoxClip->transform(mClipTransform);

            auto clippingLayer = Scene::gen();
            clippingLayer->clip(viewBoxClip);
            clippingLayer->add(scene);
            return clippingLayer;
        }
        return scene;
    }

    return scene;
}

static void _applyTextFill(SvgStyleProperty* style, Text* text, const SvgTextNode& textNode, const Box& vBox, const Box& viewport)
{
    // If fill property is nullptr then do nothing
    if (style->fill.paint.none) {
        //Do nothing
    } else if (style->fill.paint.gradient) {
        auto bBox = style->fill.paint.gradient->userSpace ? vBox : _bounds(text);
        if (style->fill.paint.gradient->type == SvgGradientType::Linear) {
            text->fill(_applyLinearGradientProperty(style->fill.paint.gradient, bBox, viewport, style->fill.opacity));
        } else if (style->fill.paint.gradient->type == SvgGradientType::Radial) {
            text->fill(_applyRadialGradientProperty(style->fill.paint.gradient, bBox, viewport, style->fill.opacity));
        }
    } else if (style->fill.paint.url) {
        //TODO: Apply the color pointed by url
        TVGLOG("SVG", "The fill's url not supported.");
    } else {
        const auto& color = style->fill.paint.curColor ? style->color : style->fill.paint.color;
        text->fill(color.r, color.g, color.b);
        if (style->fontWeight >= SvgFontWeight::Weight600) text->outline(textNode.fontSize * 0.03f, color.r, color.g, color.b);
        text->opacity(style->fill.opacity);
    }
}


static char* _processText(const char* text, SvgXmlSpace space)
{
    if (!text) return nullptr;

    auto len = strlen(text);
    auto processed = (char*)tvg::malloc(len + 1);
    auto dst = processed;
    auto src = text;

    if (space == SvgXmlSpace::Preserve) {
        while (*src) {
            if (*src == '\n' || *src == '\t' || *src == '\r') *dst++ = ' ';
            else *dst++ = *src;
            src++;
        }
        *dst = '\0';
    } else {
        auto spaceFound = false;
        src = svgUtilSkipWhiteSpace(src, nullptr);

        while (*src) {
            if (isspace((unsigned char)*src)) {
                if (!spaceFound) {
                    *dst++ = ' ';
                    spaceFound = true;
                }
            } else {
                *dst++ = *src;
                spaceFound = false;
            }
            src++;
        }
        dst = (char*)svgUtilUnskipWhiteSpace(dst, processed);
        *dst = '\0';
    }
    return processed;
}

static SvgBaseline _effectiveBaseline(const SvgStyleProperty* style)
{
    if (style->alignmentBaseline != SvgBaseline::Auto) return style->alignmentBaseline;
    return style->dominantBaseline;
}

static void _applyTextBaseline(Text* text, SvgBaseline baseline, Matrix& transform)
{
    if (baseline == SvgBaseline::Auto || baseline == SvgBaseline::Alphabetic) return;

    TextMetrics tm;
    if (text->metrics(tm) != Result::Success) return;  // ascent > 0, descent < 0

    auto shift = 0.0f;
    // baseline geometry: https://www.w3.org/TR/css-inline-3/#baseline-types
    // hanging/mathematical are synthesized from the ascent when the font provides no baseline table,
    // see https://www.w3.org/TR/css-inline-3/#baseline-synthesis-fonts
    switch (baseline) {
        case SvgBaseline::BeforeEdge: shift = tm.ascent; break;
        case SvgBaseline::AfterEdge: shift = tm.descent; break;
        case SvgBaseline::Central: shift = 0.5f * (tm.ascent + tm.descent); break;
        case SvgBaseline::Middle: {  // half the x-height (top extent of 'x')
            GlyphMetrics gm;
            if (text->metrics("x", gm) == Result::Success && gm.max.y > 0.0f) shift = 0.5f * gm.max.y;
            else shift = 0.27f * tm.ascent;  // fallback when the 'x' glyph is missing
            break;
        }
        case SvgBaseline::Hanging: shift = 0.8f * tm.ascent; break;
        case SvgBaseline::Mathematical: shift = 0.5f * tm.ascent; break;
        default: return;
    }

    translateR(&transform, {0.0f, shift});
    text->transform(transform);
}

static Text* _buildText(const SvgTextNode* textNode, SvgXmlSpace xmlSpace, const Matrix* transform, SvgBaseline baseline)
{
    if (!textNode->text) return nullptr;

    auto text = Text::gen();

    //TODO: handle def values of font and size as used in a system?
    auto size = textNode->fontSize * 0.75f; //1 pt = 1/72; 1 in = 96 px; -> 72/96 = 0.75
    if (text->font(textNode->fontFamily) != Result::Success) {
        text->font(nullptr);         //fallback to any available font
    }
    text->size(size);

    auto processedText = _processText(textNode->text, xmlSpace);
    text->text(processedText);
    tvg::free(processedText);

    TextMetrics tm;
    text->metrics(tm);
    auto textTransform = transform ? *transform : tvg::identity();
    translateR(&textTransform, {textNode->x + textNode->dx, textNode->y + textNode->dy - tm.ascent});
    text->transform(textTransform);

    _applyTextBaseline(text, baseline, textTransform);

    return text;
}

static float _applySpacing(Text* text, float letterSpacing, float wordSpacing)
{
    if (letterSpacing == 0.0f && wordSpacing == 0.0f) return 1.0f;

    auto utf8 = text->text();
    auto advance = 0.0f;
    uint32_t gaps = 0;
    uint32_t spaces = 0;
    GlyphMetrics gm;
    while (utf8 && *utf8) {
        auto space = *utf8 == ' ';
        if (text->metrics(utf8, gm, &utf8) != Result::Success) return 1.0f;
        if (utf8 && *utf8) {
            advance += gm.advance;
            ++gaps;
            if (space) ++spaces;
        }
    }
    if (advance <= 0.0f) return 1.0f;

    // Text::spacing() scales advances, so match the total offset using measured gaps.
    auto scale = 1.0f + (letterSpacing * gaps + wordSpacing * spaces) / advance;
    if (scale < 0.0f) scale = 0.0f;
    text->spacing(scale, 1.0f);
    return scale;
}

static void _updatePos(Text* text, const SvgTextNode& textNode, float anchor, float spacingScale, Point& textPos)
{
    auto advance = 0.0f;
    if (auto utf8 = text->text()) {
        GlyphMetrics gm;
        while (utf8 && *utf8) {
            if (text->metrics(utf8, gm, &utf8) != Result::Success) break;
            advance += gm.advance;
        }
    }
    // Text::spacing() scales every glyph advance, so scale the measured
    // advance the same way to stay in sync with the rendered output.
    textPos.x = textNode.x + textNode.dx + (1.0f - anchor) * advance * spacingScale;
    textPos.y = textNode.y + textNode.dy;
}

static bool _hasPositionedTspan(const SvgNode* node, int depth)
{
    if (depth > 2192) {
        TVGERR("SVG", "Infinite recursive call - stopped after %d calls! Svg file may be incorrectly formatted.", depth);
        return true;
    }
    ARRAY_FOREACH(p, node->child)
    {
        if ((*p)->type != SvgNodeType::Tspan) continue;
        auto& textNode = (*p)->node.text;
        if (textNode.text) return true;
        if (_hasPositionedTspan(*p, depth + 1)) return true;
    }
    return false;
}

static void _buildTspanScene(SvgParserContext& ctx, const SvgNode* node, Scene* scene, const Box& vBox, const string& svgPath, int depth, Point& textPos)
{
    if (depth > 2192) {
        TVGERR("SVG", "Infinite recursive call - stopped after %d calls! Svg file may be incorrectly formatted.", depth);
        return;
    }
    ARRAY_FOREACH(p, node->child)
    {
        auto child = *p;
        if (child->type != SvgNodeType::Tspan) continue;

        auto textNode = child->node.text;
        if (textNode.text) {
            auto xmlSpace = child->xmlSpace;
            for (auto n = child->parent; n; n = n->parent) {
                if (textNode.fontSize <= 0.0f) textNode.fontSize = n->node.text.fontSize;
                if (!textNode.fontFamily) textNode.fontFamily = n->node.text.fontFamily;
                if (xmlSpace == SvgXmlSpace::None) xmlSpace = n->xmlSpace;
                if (n->type == SvgNodeType::Text) break;
            }
            if (xmlSpace == SvgXmlSpace::None) xmlSpace = SvgXmlSpace::Default;

            if (textNode.x == FLT_MAX) textNode.x = textPos.x;
            if (textNode.y == FLT_MAX) textNode.y = textPos.y;

            auto text = _buildText(&textNode, xmlSpace, nullptr, _effectiveBaseline(child->style));
            if (text) {
                text->align(child->style->textAnchor, 0.0f);
                auto spacingScale = _applySpacing(text, child->style->letterSpacing, child->style->wordSpacing);
                _updatePos(text, textNode, child->style->textAnchor, spacingScale, textPos);
                _applyTextFill(child->style, text, textNode, vBox, ctx.parser->global);
                auto paint = _applyFilter(ctx, text, child, vBox, svgPath);
                paint = _applyComposition(ctx, paint, child, vBox, svgPath);
                paint = _applyBlend(paint, child);
                scene->add(paint);
            }
        }
        _buildTspanScene(ctx, child, scene, vBox, svgPath, depth + 1, textPos);
    }
}

static Paint* _textBuildHelper(SvgParserContext& ctx, const SvgNode* node, const Box& vBox, const string& svgPath)
{
    auto textNode = &node->node.text;

    // Handle xml:space
    auto xmlSpace = node->xmlSpace;
    for (auto n = node->parent; xmlSpace == SvgXmlSpace::None && n; n = n->parent)
        xmlSpace = n->xmlSpace;
    if (xmlSpace == SvgXmlSpace::None) xmlSpace = SvgXmlSpace::Default;

    if (!_hasPositionedTspan(node, 0)) {
        auto text = _buildText(textNode, xmlSpace, node->transform, _effectiveBaseline(node->style));
        if (!text) return nullptr;
        text->align(node->style->textAnchor, 0.0f);
        _applySpacing(text, node->style->letterSpacing, node->style->wordSpacing);
        _applyTextFill(node->style, text, *textNode, vBox, ctx.parser->global);
        auto p = _applyFilter(ctx, text, node, vBox, svgPath);
        p = _applyComposition(ctx, p, node, vBox, svgPath);
        return _applyBlend(p, node);
    }

    auto scene = Scene::gen();
    if (node->transform) scene->transform(*node->transform);

    Point textPos = {textNode->x, textNode->y};

    if (auto text = _buildText(textNode, xmlSpace, nullptr, _effectiveBaseline(node->style))) {
        text->align(node->style->textAnchor, 0.0f);
        auto spacingScale = _applySpacing(text, node->style->letterSpacing, node->style->wordSpacing);
        _updatePos(text, *textNode, node->style->textAnchor, spacingScale, textPos);
        _applyTextFill(node->style, text, *textNode, vBox, ctx.parser->global);
        scene->add(text);
    }

    _buildTspanScene(ctx, node, scene, vBox, svgPath, 0, textPos);

    auto p = _applyFilter(ctx, scene, node, vBox, svgPath);
    p = _applyComposition(ctx, p, node, vBox, svgPath);
    return _applyBlend(p, node);
}

static Scene* _sceneBuildHelper(SvgParserContext& ctx, const SvgNode* node, const Box& vBox, const string& svgPath, bool mask, int depth)
{
    /* Exception handling: Prevent invalid SVG data input.
       The size is the arbitrary value, we need an experimental size. */
    if (depth > 2192) {
        TVGERR("SVG", "Infinite recursive call - stopped after %d calls! Svg file may be incorrectly formatted.", depth);
        return nullptr;
    }

    if (!_isGroupType(node->type) && !mask) return nullptr;

    auto scene = Scene::gen();
    // For a Symbol node, the viewBox transformation has to be applied first - see _symbolTransform()
    if (!mask && node->type != SvgNodeType::Symbol) {
        if (node->type == SvgNodeType::Use) {
            scene->transform(node->node.use.symbol ? _symbolTransform(node, vBox) : _useTransform(node));
        } else if (node->transform) {
            scene->transform(*node->transform);
        }
    }
    if (!node->style->display || node->style->opacity == 0) return scene;

    ARRAY_FOREACH(p, node->child) {
        auto child = *p;
        Paint* paint = nullptr;
        if (child->type == SvgNodeType::ClipPath || child->type == SvgNodeType::Filter || child->type == SvgNodeType::Pattern) continue;
        if (_isGroupType(child->type)) {
            if (child->type == SvgNodeType::Use) paint = _useBuildHelper(ctx, child, vBox, svgPath, depth + 1);
            else if (!(child->type == SvgNodeType::Symbol && node->type != SvgNodeType::Use)) paint = _sceneBuildHelper(ctx, child, vBox, svgPath, false, depth + 1);
        } else {
            if (child->type == SvgNodeType::Image) paint = _imageBuildHelper(ctx, child, vBox, svgPath);
            else if (child->type == SvgNodeType::Text) paint = _textBuildHelper(ctx, child, vBox, svgPath);
            else if (child->type != SvgNodeType::Mask) paint = _shapeBuildHelper(ctx, child, vBox, svgPath);
        }
        if (paint) {
            // TODO: enable this only when accessible is enabled at thorvg v2 (for backward compat)
            if (child->id) {
                paint->id = djb2Encode(child->id);
                if (ctx.accessible) ctx.access.push({paint->id, paint, tvg::duplicate(child->id)});
            }
            scene->add(paint);
        }
    }
    scene->opacity(node->style->opacity);

    return (Scene*)_applyBlend(_applyComposition(ctx, _applyFilter(ctx, scene, node, vBox, svgPath), node, vBox, svgPath), node);
}

static Paint* _buildPatternChild(SvgParserContext& ctx, SvgNode* child, const Box& vBox, const string& svgPath)
{
    if (child->type == SvgNodeType::ClipPath || child->type == SvgNodeType::Filter || child->type == SvgNodeType::Pattern) return nullptr;
    if (_isGroupType(child->type)) {
        if (child->type == SvgNodeType::Use) return _useBuildHelper(ctx, child, vBox, svgPath, 0);
        if (child->type != SvgNodeType::Symbol) return _sceneBuildHelper(ctx, child, vBox, svgPath, false, 0);
        return nullptr;
    }
    if (child->type == SvgNodeType::Image) return _imageBuildHelper(ctx, child, vBox, svgPath);
    if (child->type == SvgNodeType::Text) return _textBuildHelper(ctx, child, vBox, svgPath);
    if (child->type == SvgNodeType::Mask) return nullptr;
    return _shapeBuildHelper(ctx, child, vBox, svgPath);
}

static Paint* _buildBaseTile(SvgParserContext& ctx, SvgNode* patternNode, const Box& vBox, const string& svgPath)
{
    Paint* paint = nullptr;
    Scene* tileScene = nullptr;
    ARRAY_FOREACH(p, patternNode->child) {
        auto child = _buildPatternChild(ctx, *p, vBox, svgPath);
        if (!child) continue;
        if (!paint && !tileScene) {
            paint = child;
            continue;
        }
        if (!tileScene) {
            tileScene = Scene::gen();
            tileScene->add(paint);
            paint = nullptr;
        }
        tileScene->add(child);
    }
    if (tileScene) return tileScene;
    return paint;
}

static bool _patternCellRect(const SvgPatternNode& pat, const Box& bbox, Box& cell)
{
    if (pat.patternUserSpace) {
        cell = pat.box;
    } else {
        cell.x = bbox.x + pat.box.x * bbox.w;
        cell.y = bbox.y + pat.box.y * bbox.h;
        cell.w = pat.box.w * bbox.w;
        cell.h = pat.box.h * bbox.h;
    }
    return _validBox(cell);
}

static Matrix _patternContentTransform(const SvgPatternNode& pat, const Box& bbox, const Box& cell)
{
    if (pat.hasViewBox) {
        auto sx = cell.w / pat.vbox.w;
        auto sy = cell.h / pat.vbox.h;
        return {sx, 0, -pat.vbox.x * sx, 0, sy, -pat.vbox.y * sy, 0, 0, 1};
    }
    if (!pat.contentUserSpace) return {bbox.w, 0, 0, 0, bbox.h, 0, 0, 0, 1};
    return tvg::identity();
}

static Box _transformBounds(const Box& bounds, const Matrix& matrix)
{
    auto lt = Point{bounds.x, bounds.y} * matrix;
    auto lb = Point{bounds.x, bounds.y + bounds.h} * matrix;
    auto rt = Point{bounds.x + bounds.w, bounds.y} * matrix;
    auto rb = Point{bounds.x + bounds.w, bounds.y + bounds.h} * matrix;

    auto min = tvg::min(tvg::min(lt, lb), tvg::min(rt, rb));
    auto max = tvg::max(tvg::max(lt, lb), tvg::max(rt, rb));

    return {min.x, min.y, max.x - min.x, max.y - min.y};
}

static void _patternTileGrid(const Box& cell, const Box& bbox, const Matrix* transform, float& startX, float& startY, int& cols, int& rows)
{
    auto box = bbox;
    Matrix inv;
    if (transform && tvg::inverse(transform, &inv)) box = _transformBounds(bbox, inv);

    startX = cell.x + floorf((box.x - cell.x) / cell.w) * cell.w;
    startY = cell.y + floorf((box.y - cell.y) / cell.h) * cell.h;
    cols = (int)ceilf((box.x + box.w - startX) / cell.w);
    rows = (int)ceilf((box.y + box.h - startY) / cell.h);
}

static Paint* _applyPatternProperty(SvgParserContext& ctx, Shape* vg, SvgNode* node, SvgNode* patternNode, const Box& vBox, const string& svgPath)
{
    if (!patternNode || patternNode->child.empty()) return nullptr;
    auto& pat = patternNode->node.pattern;

    if (pat.applying) {
        TVGLOG("SVG", "Circular pattern reference detected; skipped.");
        return nullptr;
    }

    auto bbox = _bounds(vg);
    if (!_validBox(bbox)) return nullptr;

    Box cell;
    if (!_patternCellRect(pat, bbox, cell)) return nullptr;

    float startX, startY;
    int cols, rows;
    _patternTileGrid(cell, bbox, pat.transform, startX, startY, cols, rows);

    auto contentTransform = _patternContentTransform(pat, bbox, cell);

    pat.applying = true;

    auto base = _buildBaseTile(ctx, patternNode, vBox, svgPath);
    auto tilesScene = Scene::gen();
    if (base) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                auto copy = base->duplicate();
                if (!copy) continue;
                Matrix tileTransform = Matrix{1, 0, startX + c * cell.w, 0, 1, startY + r * cell.h, 0, 0, 1} * contentTransform;
                if (pat.transform) tileTransform = *pat.transform * tileTransform;
                copy->transform(tileTransform * copy->transform());
                tilesScene->add(copy);
            }
        }
        Paint::rel(base);
    }

    auto clipper = Shape::gen();
    if (!_recognizeShape(node, clipper)) {
        pat.applying = false;
        Paint::rel(tilesScene);
        Paint::rel(clipper);
        return nullptr;
    }
    tilesScene->clip(clipper);

    pat.applying = false;
    return tilesScene;
}

static void _updateInvalidViewSize(Scene* scene, Box& vBox, float& w, float& h, SvgViewFlag viewFlag)
{
    auto useW = (viewFlag & SvgViewFlag::Width);
    auto useH = (viewFlag & SvgViewFlag::Height);
    auto bbox = _bounds(scene);

    if (!useW && !useH) {
        vBox = bbox;
    } else {
        vBox.w = useW ? w : bbox.w;
        vBox.h = useH ? h : bbox.h;
    }

    //the size would have 1x1 or percentage values.
    if (!useW) w *= vBox.w;
    if (!useH) h *= vBox.h;
}


static void _loadFonts(Array<FontFace>& fonts)
{
    if (fonts.empty()) return;

    constexpr size_t MAX_SCAN = 40;
    constexpr size_t KEY_LEN = 10;  // "ttf;base64" / "otf;base64"

    ARRAY_FOREACH(p, fonts) {
        if (!p->name) continue;

        size_t shift = 0;
        const char* type = nullptr;
        auto limit = (p->srcLen < MAX_SCAN) ? p->srcLen : MAX_SCAN;

        for (size_t i = 0; i + KEY_LEN <= limit; ++i) {
            if (!memcmp(p->src + i, "ttf;base64", KEY_LEN)) {
                shift = i + KEY_LEN + 1;  // skip ","
                type = "ttf";
                break;
            }
            if (!memcmp(p->src + i, "otf;base64", KEY_LEN)) {
                shift = i + KEY_LEN + 1;
                type = "otf";
                break;
            }
        }
        if (type) {
            auto size = b64Decode(p->src + shift, p->srcLen - shift, &p->decoded);
            Text::load(p->name, p->decoded, size, type);
        }
    }
}

static bool _hasUserSpaceGradients(SvgParserContext& ctx)
{
    ARRAY_FOREACH(p, ctx.gradients) {
        if ((*p)->userSpace) return true;
    }
    if (ctx.def) {
        ARRAY_FOREACH(p, ctx.def->node.defs.gradients) {
            if ((*p)->userSpace) return true;
        }
    }
    return false;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Scene* svgSceneBuild(SvgParserContext& ctx, Box vBox, float w, float h, AspectRatioAlign align, AspectRatioMeetOrSlice meetOrSlice, const string& svgPath, SvgViewFlag viewFlag)
{
    //TODO: aspect ratio is valid only if viewBox was set

    if (!ctx.doc || (ctx.doc->type != SvgNodeType::Doc)) return nullptr;

    _loadFonts(ctx.fonts);

    auto docNode = _sceneBuildHelper(ctx, ctx.doc, vBox, svgPath, false, 0);

    if (!(viewFlag & SvgViewFlag::Viewbox)) {
        auto prevW = vBox.w, prevH = vBox.h;
        _updateInvalidViewSize(docNode, vBox, w, h, viewFlag);
        if ((!tvg::equal(vBox.w, prevW) || !tvg::equal(vBox.h, prevH)) && _hasUserSpaceGradients(ctx)) {
            Paint::rel(docNode);
            ARRAY_FOREACH(p, ctx.images) tvg::free(*p);
            ctx.images.clear();
            docNode = _sceneBuildHelper(ctx, ctx.doc, vBox, svgPath, false, 0);
        }
    }

    if (!tvg::equal(w, vBox.w) || !tvg::equal(h, vBox.h)) {
        Matrix m = _calculateAspectRatioMatrix(align, meetOrSlice, w, h, vBox);
        docNode->transform(m);
    } else if (!tvg::zero(vBox.x) || !tvg::zero(vBox.y)) {
        docNode->translate(-vBox.x, -vBox.y);
    }

    auto viewBoxClip = Shape::gen();
    viewBoxClip->appendRect(0, 0, w, h);

    auto clippingLayer = Scene::gen();
    clippingLayer->clip(viewBoxClip);
    clippingLayer->add(docNode);

    ctx.doc->node.doc.vbox = vBox;
    ctx.doc->node.doc.w = w;
    ctx.doc->node.doc.h = h;

    auto root = Scene::gen();
    root->add(clippingLayer);

    return root;
}
