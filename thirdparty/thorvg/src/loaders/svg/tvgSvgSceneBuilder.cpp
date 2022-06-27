/*
 * Copyright (c) 2020 - 2022 Samsung Electronics Co., Ltd. All rights reserved.

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

/*
 * Copyright notice for the EFL:

 * Copyright (C) EFL developers (see AUTHORS)

 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.

 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <cstring>
#include <string>
#include "tvgMath.h"
#include "tvgSvgLoaderCommon.h"
#include "tvgSvgSceneBuilder.h"
#include "tvgSvgPath.h"
#include "tvgSvgUtil.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct Box
{
    float x, y, w, h;
};


static bool _appendShape(SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath);
static unique_ptr<Scene> _sceneBuildHelper(const SvgNode* node, const Box& vBox, const string& svgPath, bool mask, bool* isMaskWhite = nullptr);


static inline bool _isGroupType(SvgNodeType type)
{
    if (type == SvgNodeType::Doc || type == SvgNodeType::G || type == SvgNodeType::Use || type == SvgNodeType::ClipPath || type == SvgNodeType::Symbol) return true;
    return false;
}


//According to: https://www.w3.org/TR/SVG11/coords.html#ObjectBoundingBoxUnits (the last paragraph)
//a stroke width should be ignored for bounding box calculations
static Box _boundingBox(const Shape* shape)
{
    float x, y, w, h;
    shape->bounds(&x, &y, &w, &h, false);

    if (auto strokeW = shape->strokeWidth()) {
        x += 0.5f * strokeW;
        y += 0.5f * strokeW;
        w -= strokeW;
        h -= strokeW;
    }

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


static unique_ptr<LinearGradient> _applyLinearGradientProperty(SvgStyleGradient* g, const Shape* vg, const Box& vBox, int opacity)
{
    Fill::ColorStop* stops;
    int stopCount = 0;
    auto fillGrad = LinearGradient::gen();

    bool isTransform = (g->transform ? true : false);
    Matrix finalTransform = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (isTransform) finalTransform = *g->transform;

    if (g->userSpace) {
        g->linear->x1 = g->linear->x1 * vBox.w;
        g->linear->y1 = g->linear->y1 * vBox.h;
        g->linear->x2 = g->linear->x2 * vBox.w;
        g->linear->y2 = g->linear->y2 * vBox.h;
    } else {
        Matrix m = {vBox.w, 0, vBox.x, 0, vBox.h, vBox.y, 0, 0, 1};
        if (isTransform) _transformMultiply(&m, &finalTransform);
        else {
            finalTransform = m;
            isTransform = true;
        }
    }

    if (isTransform) fillGrad->transform(finalTransform);

    fillGrad->linear(g->linear->x1, g->linear->y1, g->linear->x2, g->linear->y2);
    fillGrad->spread(g->spread);

    //Update the stops
    stopCount = g->stops.count;
    if (stopCount > 0) {
        stops = (Fill::ColorStop*)calloc(stopCount, sizeof(Fill::ColorStop));
        if (!stops) return fillGrad;
        auto prevOffset = 0.0f;
        for (uint32_t i = 0; i < g->stops.count; ++i) {
            auto colorStop = &g->stops.data[i];
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
        fillGrad->colorStops(stops, stopCount);
        free(stops);
    }
    return fillGrad;
}


static unique_ptr<RadialGradient> _applyRadialGradientProperty(SvgStyleGradient* g, const Shape* vg, const Box& vBox, int opacity)
{
    Fill::ColorStop *stops;
    int stopCount = 0;
    auto fillGrad = RadialGradient::gen();

    bool isTransform = (g->transform ? true : false);
    Matrix finalTransform = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (isTransform) finalTransform = *g->transform;

    if (g->userSpace) {
        //The radius scalling is done according to the Units section:
        //https://www.w3.org/TR/2015/WD-SVG2-20150915/coords.html
        g->radial->cx = g->radial->cx * vBox.w;
        g->radial->cy = g->radial->cy * vBox.h;
        g->radial->r = g->radial->r * sqrtf(powf(vBox.w, 2.0f) + powf(vBox.h, 2.0f)) / sqrtf(2.0f);
        g->radial->fx = g->radial->fx * vBox.w;
        g->radial->fy = g->radial->fy * vBox.h;
    } else {
        Matrix m = {vBox.w, 0, vBox.x, 0, vBox.h, vBox.y, 0, 0, 1};
        if (isTransform) _transformMultiply(&m, &finalTransform);
        else {
            finalTransform = m;
            isTransform = true;
        }
    }

    if (isTransform) fillGrad->transform(finalTransform);

    //TODO: Tvg is not support to focal
    //if (g->radial->fx != 0 && g->radial->fy != 0) {
    //    fillGrad->radial(g->radial->fx, g->radial->fy, g->radial->r);
    //}
    fillGrad->radial(g->radial->cx, g->radial->cy, g->radial->r);
    fillGrad->spread(g->spread);

    //Update the stops
    stopCount = g->stops.count;
    if (stopCount > 0) {
        stops = (Fill::ColorStop*)calloc(stopCount, sizeof(Fill::ColorStop));
        if (!stops) return fillGrad;
        auto prevOffset = 0.0f;
        for (uint32_t i = 0; i < g->stops.count; ++i) {
            auto colorStop = &g->stops.data[i];
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
        fillGrad->colorStops(stops, stopCount);
        free(stops);
    }
    return fillGrad;
}


static bool _appendChildShape(SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath)
{
    auto valid = false;

    if (_appendShape(node, shape, vBox, svgPath)) valid = true;

    if (node->child.count > 0) {
        auto child = node->child.data;
        for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
            if (_appendChildShape(*child, shape, vBox, svgPath)) valid = true;
        }
    }

    return valid;
}


static void _applyComposition(Paint* paint, const SvgNode* node, const Box& vBox, const string& svgPath)
{
    /* ClipPath */
    /* Do not drop in Circular Dependency for ClipPath.
       Composition can be applied recursively if its children nodes have composition target to this one. */
    if (node->style->clipPath.applying) {
        TVGLOG("SVG", "Multiple Composition Tried! Check out Circular dependency?");
    } else {
        auto compNode = node->style->clipPath.node;
        if (compNode && compNode->child.count > 0) {
            node->style->clipPath.applying = true;

            auto comp = Shape::gen();
            comp->fill(255, 255, 255, 255);
            if (node->transform) comp->transform(*node->transform);

            auto child = compNode->child.data;
            auto valid = false; //Composite only when valid shapes are existed

            for (uint32_t i = 0; i < compNode->child.count; ++i, ++child) {
                if (_appendChildShape(*child, comp.get(), vBox, svgPath)) valid = true;
            }

            if (valid) paint->composite(move(comp), CompositeMethod::ClipPath);

            node->style->clipPath.applying = false;
        }
    }

    /* Mask */
    /* Do not drop in Circular Dependency for Mask.
       Composition can be applied recursively if its children nodes have composition target to this one. */
    if (node->style->mask.applying) {
        TVGLOG("SVG", "Multiple Composition Tried! Check out Circular dependency?");
    } else {
        auto compNode = node->style->mask.node;
        if (compNode && compNode->child.count > 0) {
            node->style->mask.applying = true;

            bool isMaskWhite = true;
            auto comp = _sceneBuildHelper(compNode, vBox, svgPath, true, &isMaskWhite);
            if (comp) {
                if (node->transform) comp->transform(*node->transform);

                if (compNode->node.mask.type == SvgMaskType::Luminance && !isMaskWhite) {
                    paint->composite(move(comp), CompositeMethod::LumaMask);
                } else {
                    paint->composite(move(comp), CompositeMethod::AlphaMask);
                }
            }

            node->style->mask.applying = false;
        }
    }
}


static void _applyProperty(SvgNode* node, Shape* vg, const Box& vBox, const string& svgPath)
{
    SvgStyleProperty* style = node->style;

    if (node->transform) vg->transform(*node->transform);
    if (node->type == SvgNodeType::Doc || !node->display) return;

    //If fill property is nullptr then do nothing
    if (style->fill.paint.none) {
        //Do nothing
    } else if (style->fill.paint.gradient) {
        Box bBox = vBox;
        if (!style->fill.paint.gradient->userSpace) bBox = _boundingBox(vg);

        if (style->fill.paint.gradient->type == SvgGradientType::Linear) {
             auto linear = _applyLinearGradientProperty(style->fill.paint.gradient, vg, bBox, style->fill.opacity);
             vg->fill(move(linear));
        } else if (style->fill.paint.gradient->type == SvgGradientType::Radial) {
             auto radial = _applyRadialGradientProperty(style->fill.paint.gradient, vg, bBox, style->fill.opacity);
             vg->fill(move(radial));
        }
    } else if (style->fill.paint.url) {
        //TODO: Apply the color pointed by url
    } else if (style->fill.paint.curColor) {
        //Apply the current style color
        vg->fill(style->color.r, style->color.g, style->color.b, style->fill.opacity);
    } else {
        //Apply the fill color
        vg->fill(style->fill.paint.color.r, style->fill.paint.color.g, style->fill.paint.color.b, style->fill.opacity);
    }

    //Apply the fill rule
    vg->fill((tvg::FillRule)style->fill.fillRule);

    //Apply node opacity
    if (style->opacity < 255) vg->opacity(style->opacity);

    if (node->type == SvgNodeType::G || node->type == SvgNodeType::Use) return;

    //Apply the stroke style property
    vg->stroke(style->stroke.width);
    vg->stroke(style->stroke.cap);
    vg->stroke(style->stroke.join);
    if (style->stroke.dash.array.count > 0) {
        vg->stroke(style->stroke.dash.array.data, style->stroke.dash.array.count);
    }

    //If stroke property is nullptr then do nothing
    if (style->stroke.paint.none) {
        //Do nothing
    } else if (style->stroke.paint.gradient) {
        Box bBox = vBox;
        if (!style->stroke.paint.gradient->userSpace) bBox = _boundingBox(vg);

        if (style->stroke.paint.gradient->type == SvgGradientType::Linear) {
             auto linear = _applyLinearGradientProperty(style->stroke.paint.gradient, vg, bBox, style->stroke.opacity);
             vg->stroke(move(linear));
        } else if (style->stroke.paint.gradient->type == SvgGradientType::Radial) {
             auto radial = _applyRadialGradientProperty(style->stroke.paint.gradient, vg, bBox, style->stroke.opacity);
             vg->stroke(move(radial));
        }
    } else if (style->stroke.paint.url) {
        //TODO: Apply the color pointed by url
    } else if (style->stroke.paint.curColor) {
        //Apply the current style color
        vg->stroke(style->color.r, style->color.g, style->color.b, style->stroke.opacity);
    } else {
        //Apply the stroke color
        vg->stroke(style->stroke.paint.color.r, style->stroke.paint.color.g, style->stroke.paint.color.b, style->stroke.opacity);
    }

    _applyComposition(vg, node, vBox, svgPath);
}


static unique_ptr<Shape> _shapeBuildHelper(SvgNode* node, const Box& vBox, const string& svgPath)
{
    auto shape = Shape::gen();
    if (_appendShape(node, shape.get(), vBox, svgPath)) return shape;
    else return nullptr;
}


static bool _appendShape(SvgNode* node, Shape* shape, const Box& vBox, const string& svgPath)
{
    Array<PathCommand> cmds;
    Array<Point> pts;

    switch (node->type) {
        case SvgNodeType::Path: {
            if (node->node.path.path) {
                if (svgPathToTvgPath(node->node.path.path, cmds, pts)) {
                    shape->appendPath(cmds.data, cmds.count, pts.data, pts.count);
                }
            }
            break;
        }
        case SvgNodeType::Ellipse: {
            shape->appendCircle(node->node.ellipse.cx, node->node.ellipse.cy, node->node.ellipse.rx, node->node.ellipse.ry);
            break;
        }
        case SvgNodeType::Polygon: {
            if (node->node.polygon.pointsCount < 2) break;
            shape->moveTo(node->node.polygon.points[0], node->node.polygon.points[1]);
            for (int i = 2; i < node->node.polygon.pointsCount - 1; i += 2) {
                shape->lineTo(node->node.polygon.points[i], node->node.polygon.points[i + 1]);
            }
            shape->close();
            break;
        }
        case SvgNodeType::Polyline: {
            if (node->node.polygon.pointsCount < 2) break;
            shape->moveTo(node->node.polygon.points[0], node->node.polygon.points[1]);
            for (int i = 2; i < node->node.polygon.pointsCount - 1; i += 2) {
                shape->lineTo(node->node.polygon.points[i], node->node.polygon.points[i + 1]);
            }
            break;
        }
        case SvgNodeType::Circle: {
            shape->appendCircle(node->node.circle.cx, node->node.circle.cy, node->node.circle.r, node->node.circle.r);
            break;
        }
        case SvgNodeType::Rect: {
            shape->appendRect(node->node.rect.x, node->node.rect.y, node->node.rect.w, node->node.rect.h, node->node.rect.rx, node->node.rect.ry);
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

    _applyProperty(node, shape, vBox, svgPath);
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
    {"svg+xml", sizeof("svg+xml"), imageMimeTypeEncoding::base64 | imageMimeTypeEncoding::utf8},
};


static bool _isValidImageMimeTypeAndEncoding(const char** href, const char** mimetype, imageMimeTypeEncoding* encoding) {
    if (strncmp(*href, "image/", sizeof("image/") - 1)) return false; //not allowed mime type
    *href += sizeof("image/") - 1;

    //RFC2397 data:[<mediatype>][;base64],<data>
    //mediatype  := [ type "/" subtype ] *( ";" parameter )
    //parameter  := attribute "=" value
    for (unsigned int i = 0; i < sizeof(imageMimeTypes) / sizeof(imageMimeTypes[0]); i++) {
        if (!strncmp(*href, imageMimeTypes[i].name, imageMimeTypes[i].sz - 1)) {
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
    }
    return false;
}


static unique_ptr<Picture> _imageBuildHelper(SvgNode* node, const Box& vBox, const string& svgPath)
{
    if (!node->node.image.href) return nullptr;
    auto picture = Picture::gen();

    const char* href = node->node.image.href;
    if (!strncmp(href, "data:", sizeof("data:") - 1)) {
        href += sizeof("data:") - 1;
        const char* mimetype;
        imageMimeTypeEncoding encoding;
        if (!_isValidImageMimeTypeAndEncoding(&href, &mimetype, &encoding)) return nullptr; //not allowed mime type or encoding
        if (encoding == imageMimeTypeEncoding::base64) {
            string decoded = svgUtilBase64Decode(href);
            if (picture->load(decoded.c_str(), decoded.size(), mimetype, true) != Result::Success) return nullptr;
        } else {
            string decoded = svgUtilURLDecode(href);
            if (picture->load(decoded.c_str(), decoded.size(), mimetype, true) != Result::Success) return nullptr;
        }
    } else {
        if (!strncmp(href, "file://", sizeof("file://") - 1)) href += sizeof("file://") - 1;
        //TODO: protect against recursive svg image loading
        //Temporarily disable embedded svg:
        const char *dot = strrchr(href, '.');
        if (dot && !strcmp(dot, ".svg")) {
            TVGLOG("SVG", "Embedded svg file is disabled.");
            return nullptr;
        }
        string imagePath = href;
        if (strncmp(href, "/", 1)) {
            auto last = svgPath.find_last_of("/");
            imagePath = svgPath.substr(0, (last == string::npos ? 0 : last + 1)) + imagePath;
        }
        if (picture->load(imagePath) != Result::Success) return nullptr;
    }

    float w, h;
    Matrix m = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (picture->size(&w, &h) == Result::Success && w  > 0 && h > 0) {
        auto sx = node->node.image.w / w;
        auto sy = node->node.image.h / h;
        m = {sx, 0, node->node.image.x, 0, sy, node->node.image.y, 0, 0, 1};
    }
    if (node->transform) m = mathMultiply(node->transform, &m);
    picture->transform(m);

    _applyComposition(picture.get(), node, vBox, svgPath);
    return picture;
}


static unique_ptr<Scene> _useBuildHelper(const SvgNode* node, const Box& vBox, const string& svgPath, bool* isMaskWhite)
{
    unique_ptr<Scene> finalScene;
    auto scene = _sceneBuildHelper(node, vBox, svgPath, false, isMaskWhite);

    // mUseTransform = mUseTransform * mTranslate
    Matrix mUseTransform = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (node->transform) mUseTransform = *node->transform;
    if (node->node.use.x != 0.0f || node->node.use.y != 0.0f) {
        Matrix mTranslate = {1, 0, node->node.use.x, 0, 1, node->node.use.y, 0, 0, 1};
        mUseTransform = mathMultiply(&mUseTransform, &mTranslate);
    }

    if (node->node.use.symbol) {
        auto symbol = node->node.use.symbol->node.symbol;

        auto width = (symbol.hasWidth ? symbol.w : vBox.w);
        if (node->node.use.isWidthSet) width = node->node.use.w;
        auto height = (symbol.hasHeight ? symbol.h : vBox.h);;
        if (node->node.use.isHeightSet) height = node->node.use.h;
        auto vw = (symbol.hasViewBox ? symbol.vw : width);
        auto vh = (symbol.hasViewBox ? symbol.vh : height);

        Matrix mViewBox = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if ((!mathEqual(width, vw) || !mathEqual(height, vh)) && vw > 0 && vh > 0) {
            auto sx = width / vw;
            auto sy = height / vh;
            if (symbol.preserveAspect) {
                if (sx < sy) sy = sx;
                else sx = sy;
            }

            auto tvx = symbol.vx * sx;
            auto tvy = symbol.vy * sy;
            auto tvw = vw * sx;
            auto tvh = vh * sy;
            tvy -= (symbol.h - tvh) * 0.5f;
            tvx -= (symbol.w - tvw) * 0.5f;
            mViewBox = {sx, 0, -tvx, 0, sy, -tvy, 0, 0, 1};
        } else if (!mathZero(symbol.vx) || !mathZero(symbol.vy)) {
            mViewBox = {1, 0, -symbol.vx, 0, 1, -symbol.vy, 0, 0, 1};
        }

        // mSceneTransform = mUseTransform * mSymbolTransform * mViewBox
        Matrix mSceneTransform = mViewBox;
        if (node->node.use.symbol->transform) {
            mSceneTransform = mathMultiply(node->node.use.symbol->transform, &mViewBox);
        }
        mSceneTransform = mathMultiply(&mUseTransform, &mSceneTransform);
        scene->transform(mSceneTransform);

        if (node->node.use.symbol->node.symbol.overflowVisible) {
            finalScene = move(scene);
        } else {
            auto viewBoxClip = Shape::gen();
            viewBoxClip->appendRect(0, 0, width, height, 0, 0);

            // mClipTransform = mUseTransform * mSymbolTransform
            Matrix mClipTransform = mUseTransform;
            if (node->node.use.symbol->transform) {
                mClipTransform = mathMultiply(&mUseTransform, node->node.use.symbol->transform);
            }
            viewBoxClip->transform(mClipTransform);

            auto compositeLayer = Scene::gen();
            compositeLayer->composite(move(viewBoxClip), CompositeMethod::ClipPath);
            compositeLayer->push(move(scene));

            auto root = Scene::gen();
            root->push(move(compositeLayer));

            finalScene = move(root);
        }
    } else {
        if (!mathIdentity((const Matrix*)(&mUseTransform))) scene->transform(mUseTransform);
        finalScene = move(scene);
    }

    return finalScene;
}


static unique_ptr<Scene> _sceneBuildHelper(const SvgNode* node, const Box& vBox, const string& svgPath, bool mask, bool* isMaskWhite)
{
    if (_isGroupType(node->type) || mask) {
        auto scene = Scene::gen();
        // For a Symbol node, the viewBox transformation has to be applied first - see _useBuildHelper()
        if (!mask && node->transform && node->type != SvgNodeType::Symbol) scene->transform(*node->transform);

        if (node->display && node->style->opacity != 0) {
            auto child = node->child.data;
            for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
                if (_isGroupType((*child)->type)) {
                    if ((*child)->type == SvgNodeType::Use)
                        scene->push(_useBuildHelper(*child, vBox, svgPath, isMaskWhite));
                    else
                        scene->push(_sceneBuildHelper(*child, vBox, svgPath, false, isMaskWhite));
                } else if ((*child)->type == SvgNodeType::Image) {
                    auto image = _imageBuildHelper(*child, vBox, svgPath);
                    if (image) scene->push(move(image));
                } else if ((*child)->type != SvgNodeType::Mask) {
                    auto shape = _shapeBuildHelper(*child, vBox, svgPath);
                    if (shape) {
                        if (isMaskWhite) {
                            uint8_t r, g, b;
                            shape->fillColor(&r, &g, &b, nullptr);
                            if (shape->fill() || r < 255 || g < 255 || b < 255 || shape->strokeFill() ||
                                (shape->strokeColor(&r, &g, &b, nullptr) == Result::Success && (r < 255 || g < 255 || b < 255))) {
                                *isMaskWhite = false;
                            }
                        }
                        scene->push(move(shape));
                    }
                }
            }
            _applyComposition(scene.get(), node, vBox, svgPath);
            scene->opacity(node->style->opacity);
        }
        return scene;
    }
    return nullptr;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

unique_ptr<Scene> svgSceneBuild(SvgNode* node, float vx, float vy, float vw, float vh, float w, float h, bool preserveAspect, const string& svgPath)
{
    if (!node || (node->type != SvgNodeType::Doc)) return nullptr;

    Box vBox = {vx, vy, vw, vh};
    auto docNode = _sceneBuildHelper(node, vBox, svgPath, false);

    if (!mathEqual(w, vw) || !mathEqual(h, vh)) {
        auto sx = w / vw;
        auto sy = h / vh;

        if (preserveAspect) {
            //Scale
            auto scale = sx < sy ? sx : sy;
            docNode->scale(scale);
            //Align
            auto tvx = vx * scale;
            auto tvy = vy * scale;
            auto tvw = vw * scale;
            auto tvh = vh * scale;
            tvx -= (w - tvw) * 0.5f;
            tvy -= (h - tvh) * 0.5f;
            docNode->translate(-tvx, -tvy);
        } else {
            //Align
            auto tvx = vx * sx;
            auto tvy = vy * sy;
            Matrix m = {sx, 0, -tvx, 0, sy, -tvy, 0, 0, 1};
            docNode->transform(m);
        }
    } else if (!mathZero(vx) || !mathZero(vy)) {
        docNode->translate(-vx, -vy);
    }

    auto viewBoxClip = Shape::gen();
    viewBoxClip->appendRect(0, 0, w, h, 0, 0);
    viewBoxClip->fill(0, 0, 0, 255);

    auto compositeLayer = Scene::gen();
    compositeLayer->composite(move(viewBoxClip), CompositeMethod::ClipPath);
    compositeLayer->push(move(docNode));

    auto root = Scene::gen();
    root->push(move(compositeLayer));

    return root;
}
