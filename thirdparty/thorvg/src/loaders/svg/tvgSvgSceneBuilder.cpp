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
#include "tvgSvgLoaderCommon.h"
#include "tvgSvgSceneBuilder.h"
#include "tvgSvgPath.h"
#include "tvgSvgUtil.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static bool _appendClipShape(SvgLoaderData& loaderData, SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath, const Matrix* transform);
static Scene* _sceneBuildHelper(SvgLoaderData& loaderData, const SvgNode* node, const Box& vBox, const string& svgPath, bool mask, int depth);


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


static void _transformMultiply(const Matrix* mBBox, Matrix* gradTransf)
{
    gradTransf->e13 = gradTransf->e13 * mBBox->e11 + mBBox->e13;
    gradTransf->e12 *= mBBox->e11;
    gradTransf->e11 *= mBBox->e11;

    gradTransf->e23 = gradTransf->e23 * mBBox->e22 + mBBox->e23;
    gradTransf->e22 *= mBBox->e22;
    gradTransf->e21 *= mBBox->e22;
}


static LinearGradient* _applyLinearGradientProperty(SvgStyleGradient* g, const Box& vBox, int opacity)
{
    Fill::ColorStop* stops;
    auto fillGrad = LinearGradient::gen();
    auto isTransform = (g->transform ? true : false);
    auto& finalTransform = fillGrad->transform();
    if (isTransform) finalTransform = *g->transform;

    if (g->userSpace) {
        g->linear->x1 = g->linear->x1 * vBox.w;
        g->linear->y1 = g->linear->y1 * vBox.h;
        g->linear->x2 = g->linear->x2 * vBox.w;
        g->linear->y2 = g->linear->y2 * vBox.h;
    } else {
        Matrix m = {vBox.w, 0, vBox.x, 0, vBox.h, vBox.y, 0, 0, 1};
        if (isTransform) _transformMultiply(&m, &finalTransform);
        else finalTransform = m;
    }

    fillGrad->linear(g->linear->x1, g->linear->y1, g->linear->x2, g->linear->y2);
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


static RadialGradient* _applyRadialGradientProperty(SvgStyleGradient* g, const Box& vBox, int opacity)
{
    Fill::ColorStop *stops;
    auto fillGrad = RadialGradient::gen();
    auto isTransform = (g->transform ? true : false);
    auto& finalTransform = fillGrad->transform();
    if (isTransform) finalTransform = *g->transform;

    if (g->userSpace) {
        //The radius scaling is done according to the Units section:
        //https://www.w3.org/TR/2015/WD-SVG2-20150915/coords.html
        g->radial->cx = g->radial->cx * vBox.w;
        g->radial->cy = g->radial->cy * vBox.h;
        g->radial->r = g->radial->r * sqrtf(powf(vBox.w, 2.0f) + powf(vBox.h, 2.0f)) / sqrtf(2.0f);
        g->radial->fx = g->radial->fx * vBox.w;
        g->radial->fy = g->radial->fy * vBox.h;
        g->radial->fr = g->radial->fr * sqrtf(powf(vBox.w, 2.0f) + powf(vBox.h, 2.0f)) / sqrtf(2.0f);
    } else {
        Matrix m = {vBox.w, 0, vBox.x, 0, vBox.h, vBox.y, 0, 0, 1};
        if (isTransform) _transformMultiply(&m, &finalTransform);
        else finalTransform = m;
    }

    fillGrad->radial(g->radial->cx, g->radial->cy, g->radial->r, g->radial->fx, g->radial->fy, g->radial->fr);
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


static bool _appendClipChild(SvgLoaderData& loaderData, SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath)
{
    //The SVG standard allows only for 'use' nodes that point directly to a basic shape.
    if (node->type == SvgNodeType::Use) {
        if (node->child.count != 1) return false;
        auto child = *(node->child.data);
        auto finalTransform = tvg::identity();
        if (node->transform) finalTransform = *node->transform;
        if (node->node.use.x != 0.0f || node->node.use.y != 0.0f) {
            finalTransform *= {1, 0, node->node.use.x, 0, 1, node->node.use.y, 0, 0, 1};
        }
        if (child->transform) finalTransform *= *child->transform;

        return _appendClipShape(loaderData, child, shape, vBox, svgPath, tvg::identity((const Matrix*)(&finalTransform)) ? nullptr : &finalTransform);
    }
    return _appendClipShape(loaderData, node, shape, vBox, svgPath, nullptr);
}


static Matrix _compositionTransform(Paint* paint, const SvgNode* node, const SvgNode* compNode, SvgNodeType type)
{
    auto m = tvg::identity();
    //The initial mask transformation ignored according to the SVG standard.
    if (node->transform && type != SvgNodeType::Mask) {
        m = *node->transform;
    }
    if (compNode->transform) {
        m *= *compNode->transform;
    }
    if (!compNode->node.clip.userSpace) {
        auto bbox = _bounds(paint);
        m *= {bbox.w, 0, bbox.x, 0, bbox.h, bbox.y, 0, 0, 1};
    }
    return m;
}

static bool _applyClip(SvgLoaderData& loaderData, Paint* paint, const SvgNode* node, const SvgNode* clipNode, const Box& vBox, const string& svgPath)
{
    node->style->clipPath.applying = true;

    auto clipper = Shape::gen();
    auto valid = false; //Composite only when valid shapes exist

    ARRAY_FOREACH(p, clipNode->child) {
        if (_appendClipChild(loaderData, *p, clipper, vBox, svgPath)) valid = true;
    }

    if (valid) {
        Matrix finalTransform = _compositionTransform(paint, node, clipNode, SvgNodeType::ClipPath);
        clipper->transform(finalTransform);
        paint->clip(clipper);
    } else {
        Paint::rel(clipper);
    }

    node->style->clipPath.applying = false;
    return valid;
}


static Paint* _applyComposition(SvgLoaderData& loaderData, Paint* paint, const SvgNode* node, const Box& vBox, const string& svgPath)
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
        if (!_applyClip(loaderData, scene, node, clipNode, vBox, svgPath)) {
            Paint::rel(scene);
            return nullptr;
        }
    }

    /* Mask */
    if (maskNode) {
        node->style->mask.applying = true;

        if (auto mask = _sceneBuildHelper(loaderData, maskNode, vBox, svgPath, true, 0)) {
            if (!maskNode->node.mask.userSpace) {
                Matrix finalTransform = _compositionTransform(paint, node, maskNode, SvgNodeType::Mask);
                mask->transform(finalTransform);
            } else if (node->transform) {
                mask->transform(*node->transform);
            }
            scene->mask(mask, maskNode->node.mask.type == SvgMaskType::Luminance ? MaskMethod::Luma: MaskMethod::Alpha);
        }

        node->style->mask.applying = false;
    }

    return scene;
}


static Paint* _applyFilter(SvgLoaderData& loaderData, Paint* paint, const SvgNode* node, const Box& vBox, const string& svgPath)
{
    auto filterNode = node->style->filter.node;
    if (!filterNode || filterNode->child.count == 0) return paint;
    auto& filter = filterNode->node.filter;

    auto scene = Scene::gen();

    auto bbox = _bounds(paint);
    Box clipBox = filter.filterUserSpace ? filter.box : Box{bbox.x + filter.box.x * bbox.w, bbox.y + filter.box.y * bbox.h, filter.box.w * bbox.w, filter.box.h * bbox.h};
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
                    if (isPercent[0]) gaussBox.x *= loaderData.svgParse->global.w;
                    if (isPercent[1]) gaussBox.y *= loaderData.svgParse->global.h;
                    if (isPercent[2]) gaussBox.w *= loaderData.svgParse->global.w;
                    if (isPercent[3]) gaussBox.h *= loaderData.svgParse->global.h;
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

static Paint* _applyProperty(SvgLoaderData& loaderData, SvgNode* node, Shape* vg, const Box& vBox, const string& svgPath, bool clip)
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
            vg->fill(_applyLinearGradientProperty(style->fill.paint.gradient, bBox, style->fill.opacity));
        } else if (style->fill.paint.gradient->type == SvgGradientType::Radial) {
            vg->fill(_applyRadialGradientProperty(style->fill.paint.gradient, bBox, style->fill.opacity));
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

    if (node->type == SvgNodeType::G || node->type == SvgNodeType::Use) return vg;

    //Apply the stroke style property
    vg->strokeWidth(style->stroke.width);
    vg->strokeCap(style->stroke.cap);
    vg->strokeJoin(style->stroke.join);
    vg->strokeMiterlimit(style->stroke.miterlimit);
    vg->strokeDash(style->stroke.dash.array.data, style->stroke.dash.array.count, style->stroke.dash.offset);

    //If stroke property is nullptr then do nothing
    if (style->stroke.paint.none) {
        vg->strokeWidth(0.0f);
    } else if (style->stroke.paint.gradient) {
        auto bBox = style->stroke.paint.gradient->userSpace ? vBox : _bounds(vg);
        if (style->stroke.paint.gradient->type == SvgGradientType::Linear) {
             vg->strokeFill(_applyLinearGradientProperty(style->stroke.paint.gradient, bBox, style->stroke.opacity));
        } else if (style->stroke.paint.gradient->type == SvgGradientType::Radial) {
             vg->strokeFill(_applyRadialGradientProperty(style->stroke.paint.gradient, bBox, style->stroke.opacity));
        }
    } else if (style->stroke.paint.url) {
        //TODO: Apply the color pointed by url
        TVGLOG("SVG", "The stroke's url not supported.");
    } else if (style->stroke.paint.curColor) {
        //Apply the current style color
        vg->strokeFill(style->color.r, style->color.g, style->color.b, style->stroke.opacity);
    } else {
        //Apply the stroke color
        vg->strokeFill(style->stroke.paint.color.r, style->stroke.paint.color.g, style->stroke.paint.color.b, style->stroke.opacity);
    }

    //apply transform after the local space shape bbox for gradient acquisition
    if (node->transform && !clip) vg->transform(*node->transform);

    auto p = _applyFilter(loaderData, vg, node, vBox, svgPath);
    return _applyComposition(loaderData, p, node, vBox, svgPath);
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


static Paint* _shapeBuildHelper(SvgLoaderData& loaderData, SvgNode* node, const Box& vBox, const string& svgPath)
{
    auto shape = Shape::gen();
    if (!_recognizeShape(node, shape)) return nullptr;
    return _applyProperty(loaderData, node, shape, vBox, svgPath, false);
}


static bool _appendClipShape(SvgLoaderData& loaderData, SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath, const Matrix* transform)
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

    //Apply Clip Chaining
    if (auto clipNode = node->style->clipPath.node) {
        if (clipNode->child.count == 0) return false;
        if (node->style->clipPath.applying) {
            TVGLOG("SVG", "Multiple composition tried! Check out circular dependency?");
            return false;
        }
        return _applyClip(loaderData, shape, node, clipNode, vBox, svgPath);
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
    if (strncmp(*href, "image/", sizeof("image/") - 1)) return false; //not allowed mime type
    *href += sizeof("image/") - 1;

    //RFC2397 data:[<mediatype>][;base64],<data>
    //mediatype  := [ type "/" subtype ] *( ";" parameter )
    //parameter  := attribute "=" value
    for (unsigned int i = 0; i < sizeof(imageMimeTypes) / sizeof(imageMimeTypes[0]); i++) {
        if (strncmp(*href, imageMimeTypes[i].name, imageMimeTypes[i].sz - 1)) continue;
        *href += imageMimeTypes[i].sz  - 1;
        *mimetype = imageMimeTypes[i].name;

        while (**href && **href != ',') {
            while (**href && **href != ';') ++(*href);
            if (!**href) return false;
            ++(*href);

            if (imageMimeTypes[i].encoding & imageMimeTypeEncoding::base64) {
                if (!strncmp(*href, "base64,", sizeof("base64,") - 1)) {
                    *href += sizeof("base64,") - 1;
                    *encoding = imageMimeTypeEncoding::base64;
                    return true; //valid base64
                }
            }
            if (imageMimeTypes[i].encoding & imageMimeTypeEncoding::utf8) {
                if (!strncmp(*href, "utf8,", sizeof("utf8,") - 1)) {
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

static Paint* _imageBuildHelper(SvgLoaderData& loaderData, SvgNode* node, const Box& vBox, const string& svgPath)
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
        loaderData.images.push(decoded);
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

    auto p = _applyFilter(loaderData, picture, node, vBox, svgPath);
    return _applyComposition(loaderData, p, node, vBox, svgPath);
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


static Scene* _useBuildHelper(SvgLoaderData& loaderData, const SvgNode* node, const Box& vBox, const string& svgPath, int depth)
{
    auto scene = _sceneBuildHelper(loaderData, node, vBox, svgPath, false, depth + 1);

    // mUseTransform = mUseTransform * mTranslate
    auto mUseTransform = tvg::identity();
    if (node->transform) mUseTransform = *node->transform;
    if (node->node.use.x != 0.0f || node->node.use.y != 0.0f) {
        Matrix mTranslate = {1, 0, node->node.use.x, 0, 1, node->node.use.y, 0, 0, 1};
        mUseTransform *= mTranslate;
    }

    if (node->node.use.symbol) {
        auto symbol = node->node.use.symbol->node.symbol;
        auto width = (symbol.hasWidth ? symbol.w : vBox.w);
        if (node->node.use.isWidthSet) width = node->node.use.w;
        auto height = (symbol.hasHeight ? symbol.h : vBox.h);;
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
        mSceneTransform = mUseTransform * mSceneTransform;
        scene->transform(mSceneTransform);

        if (!node->node.use.symbol->node.symbol.overflowVisible) {
            auto viewBoxClip = Shape::gen();
            viewBoxClip->appendRect(0, 0, width, height);

            // mClipTransform = mUseTransform * mSymbolTransform
            Matrix mClipTransform = mUseTransform;
            if (node->node.use.symbol->transform) {
                mClipTransform = mUseTransform * *node->node.use.symbol->transform;
            }
            viewBoxClip->transform(mClipTransform);

            auto clippingLayer = Scene::gen();
            clippingLayer->clip(viewBoxClip);
            clippingLayer->add(scene);
            return clippingLayer;
        }
        return scene;
    }

    if (auto clipper = PAINT(scene)->clipper) {
        auto& clipTransform = clipper->transform();
        Matrix inv;
        if (node->transform && inverse(node->transform, &inv)) clipTransform = inv * clipTransform;
        clipTransform = mUseTransform * clipTransform ;
    }

    scene->transform(mUseTransform);
    return scene;
}


static void _applyTextFill(SvgStyleProperty* style, Text* text, const Box& vBox)
{
    //If fill property is nullptr then do nothing
    if (style->fill.paint.none) {
        //Do nothing
    } else if (style->fill.paint.gradient) {
        auto bBox = style->fill.paint.gradient->userSpace ? vBox : _bounds(text);
        if (style->fill.paint.gradient->type == SvgGradientType::Linear) {
            text->fill(_applyLinearGradientProperty(style->fill.paint.gradient, bBox, style->fill.opacity));
        } else if (style->fill.paint.gradient->type == SvgGradientType::Radial) {
            text->fill(_applyRadialGradientProperty(style->fill.paint.gradient, bBox, style->fill.opacity));
        }
    } else if (style->fill.paint.url) {
        //TODO: Apply the color pointed by url
        TVGLOG("SVG", "The fill's url not supported.");
    } else if (style->fill.paint.curColor) {
        //Apply the current style color
        text->fill(style->color.r, style->color.g, style->color.b);
        text->opacity(style->fill.opacity);
    } else {
        //Apply the fill color
        text->fill(style->fill.paint.color.r, style->fill.paint.color.g, style->fill.paint.color.b);
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


static Paint* _textBuildHelper(SvgLoaderData& loaderData, const SvgNode* node, const Box& vBox, const string& svgPath)
{
    auto textNode = &node->node.text;
    if (!textNode->text) return nullptr;

    auto text = Text::gen();

    Matrix textTransform;
    if (node->transform) textTransform = *node->transform;
    else textTransform = tvg::identity();

    translateR(&textTransform, {node->node.text.x, node->node.text.y - textNode->fontSize});
    text->transform(textTransform);

    //TODO: handle def values of font and size as used in a system?
    auto size = textNode->fontSize * 0.75f; //1 pt = 1/72; 1 in = 96 px; -> 72/96 = 0.75
    if (text->font(textNode->fontFamily) != Result::Success) {
        text->font(nullptr);         //fallback to any available font
    }
    text->size(size);

    // Handle xml:space
    auto xmlSpace = node->xmlSpace;
    auto parent = node->parent;
    while (xmlSpace == SvgXmlSpace::None && parent) {
        xmlSpace = parent->xmlSpace;
        parent = parent->parent;
    }
    if (xmlSpace == SvgXmlSpace::None) xmlSpace = SvgXmlSpace::Default;
    auto processedText = _processText(textNode->text, xmlSpace);
    text->text(processedText);
    tvg::free(processedText);

    _applyTextFill(node->style, text, vBox);

    auto p = _applyFilter(loaderData, text, node, vBox, svgPath);
    return _applyComposition(loaderData, p, node, vBox, svgPath);
}


static Scene* _sceneBuildHelper(SvgLoaderData& loaderData, const SvgNode* node, const Box& vBox, const string& svgPath, bool mask, int depth)
{
    /* Exception handling: Prevent invalid SVG data input.
       The size is the arbitrary value, we need an experimental size. */
    if (depth > 2192) {
        TVGERR("SVG", "Infinite recursive call - stopped after %d calls! Svg file may be incorrectly formatted.", depth);
        return nullptr;
    }

    if (!_isGroupType(node->type) && !mask) return nullptr;

    auto scene = Scene::gen();
    // For a Symbol node, the viewBox transformation has to be applied first - see _useBuildHelper()
    if (!mask && node->transform && node->type != SvgNodeType::Symbol && node->type != SvgNodeType::Use) {
        scene->transform(*node->transform);
    }

    if (!node->style->display || node->style->opacity == 0) return scene;

    ARRAY_FOREACH(p, node->child) {
        auto child = *p;
        if (_isGroupType(child->type)) {
            if (child->type == SvgNodeType::Use)
                scene->add(_useBuildHelper(loaderData, child, vBox, svgPath, depth + 1));
            else if (!(child->type == SvgNodeType::Symbol && node->type != SvgNodeType::Use))
                scene->add(_sceneBuildHelper(loaderData, child, vBox, svgPath, false, depth + 1));
            if (child->id) scene->id = djb2Encode(child->id);
        } else {
            Paint* paint = nullptr;
            if (child->type == SvgNodeType::Image) paint = _imageBuildHelper(loaderData, child, vBox, svgPath);
            else if (child->type == SvgNodeType::Text) paint = _textBuildHelper(loaderData, child, vBox, svgPath);
            else if (child->type != SvgNodeType::Mask) paint = _shapeBuildHelper(loaderData, child, vBox, svgPath);
            if (paint) {
                if (child->id) paint->id = djb2Encode(child->id);
                scene->add(paint);
            }
        }
    }
    scene->opacity(node->style->opacity);

    auto p = _applyFilter(loaderData, scene, node, vBox, svgPath);
    return static_cast<Scene*>(_applyComposition(loaderData, p, node, vBox, svgPath));
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

    static constexpr struct {
        const char* prefix;
        size_t len;
    } prefixes[] = {
        {"data:font/ttf;base64,", sizeof("data:font/ttf;base64,") - 1},
        {"data:application/font-ttf;base64,", sizeof("data:application/font-ttf;base64,") - 1}
    };


    ARRAY_FOREACH(p, fonts) {
        if (!p->name) continue;

        size_t shift = 0;
        for (const auto& prefix : prefixes) {
            if (p->srcLen > prefix.len && !memcmp(p->src, prefix.prefix, prefix.len)) {
                shift = prefix.len;
                break;
            }
        }
        if (shift == 0) {
            TVGLOG("SVG", "The embedded font \"%s\" data not loaded properly.", p->name);
            continue;
        }

        auto size = b64Decode(p->src + shift, p->srcLen - shift, &p->decoded);

        if (Text::load(p->name, p->decoded, size) != Result::Success) TVGERR("SVG", "Error while loading the ttf font named \"%s\".", p->name);
    }
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Scene* svgSceneBuild(SvgLoaderData& loaderData, Box vBox, float w, float h, AspectRatioAlign align, AspectRatioMeetOrSlice meetOrSlice, const string& svgPath, SvgViewFlag viewFlag)
{
    //TODO: aspect ratio is valid only if viewBox was set

    if (!loaderData.doc || (loaderData.doc->type != SvgNodeType::Doc)) return nullptr;

    _loadFonts(loaderData.fonts);

    auto docNode = _sceneBuildHelper(loaderData, loaderData.doc, vBox, svgPath, false, 0);

    if (!(viewFlag & SvgViewFlag::Viewbox)) _updateInvalidViewSize(docNode, vBox, w, h, viewFlag);

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

    loaderData.doc->node.doc.vbox = vBox;
    loaderData.doc->node.doc.w = w;
    loaderData.doc->node.doc.h = h;

    auto root = Scene::gen();
    root->add(clippingLayer);

    return root;
}
