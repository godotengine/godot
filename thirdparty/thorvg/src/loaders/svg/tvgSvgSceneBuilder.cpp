/*
 * Copyright (c) 2020-2021 Samsung Electronics Co., Ltd. All rights reserved.

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
#include <math.h>
#include <string>
#include "tvgSvgLoaderCommon.h"
#include "tvgSvgSceneBuilder.h"
#include "tvgSvgPath.h"

static bool _appendShape(SvgNode* node, Shape* shape, float vx, float vy, float vw, float vh);

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static inline bool _isGroupType(SvgNodeType type)
{
    if (type == SvgNodeType::Doc || type == SvgNodeType::G || type == SvgNodeType::Use || type == SvgNodeType::ClipPath) return true;
    return false;
}


static unique_ptr<LinearGradient> _applyLinearGradientProperty(SvgStyleGradient* g, const Shape* vg, float rx, float ry, float rw, float rh, int opacity)
{
    Fill::ColorStop* stops;
    int stopCount = 0;
    auto fillGrad = LinearGradient::gen();

    if (g->usePercentage) {
        g->linear->x1 = g->linear->x1 * rw + rx;
        g->linear->y1 = g->linear->y1 * rh + ry;
        g->linear->x2 = g->linear->x2 * rw + rx;
        g->linear->y2 = g->linear->y2 * rh + ry;
    }

    if (g->transform) {
        //Calc start point
        auto x = g->linear->x1;
        g->linear->x1 = x * g->transform->e11 + g->linear->y1 * g->transform->e12 + g->transform->e13;
        g->linear->y1 = x * g->transform->e21 + g->linear->y1 * g->transform->e22 + g->transform->e23;

        //Calc end point
        x = g->linear->x2;
        g->linear->x2 = x * g->transform->e11 + g->linear->y2 * g->transform->e12 + g->transform->e13;
        g->linear->y2 = x * g->transform->e21 + g->linear->y2 * g->transform->e22 + g->transform->e23;
    }

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
            stops[i].a = (colorStop->a * opacity) / 255.0f;
            stops[i].offset = colorStop->offset;
            // check the offset corner cases - refer to: https://svgwg.org/svg2-draft/pservers.html#StopNotes
            if (colorStop->offset < prevOffset) stops[i].offset = prevOffset;
            else if (colorStop->offset > 1) stops[i].offset = 1;
            prevOffset = stops[i].offset;
        }
        fillGrad->colorStops(stops, stopCount);
        free(stops);
    }
    return fillGrad;
}


static unique_ptr<RadialGradient> _applyRadialGradientProperty(SvgStyleGradient* g, const Shape* vg, float rx, float ry, float rw, float rh, int opacity)
{
    Fill::ColorStop *stops;
    int stopCount = 0;
    int radius;
    auto fillGrad = RadialGradient::gen();

    radius = sqrt(pow(rw, 2) + pow(rh, 2)) / sqrt(2.0);
    if (!g->userSpace) {
         //That is according to Units in here
         //https://www.w3.org/TR/2015/WD-SVG2-20150915/coords.html
        int min = (rh > rw) ? rw : rh;
        radius = sqrt(pow(min, 2) + pow(min, 2)) / sqrt(2.0);
    }

    if (g->usePercentage) {
        g->radial->cx = g->radial->cx * rw + rx;
        g->radial->cy = g->radial->cy * rh + ry;
        g->radial->r = g->radial->r * radius;
        g->radial->fx = g->radial->fx * rw + rx;
        g->radial->fy = g->radial->fy * rh + ry;
    }

    //TODO: Radial gradient transformation - all tests possible after rx/ry implementation
    if (g->transform) {
        auto cx = g->radial->cx * g->transform->e11 + g->radial->cy * g->transform->e12 + g->transform->e13;
        g->radial->cy = g->radial->cx * g->transform->e21 + g->radial->cy * g->transform->e22 + g->transform->e23;
        g->radial->cx = cx;

        auto sx = sqrt(pow(g->transform->e11, 2) + pow(g->transform->e21, 2));
        g->radial->r *= sx;
    }

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
            stops[i].a = (colorStop->a * opacity) / 255.0f;
            stops[i].offset = colorStop->offset;
            // check the offset corner cases - refer to: https://svgwg.org/svg2-draft/pservers.html#StopNotes
            if (colorStop->offset < prevOffset) stops[i].offset = prevOffset;
            else if (colorStop->offset > 1) stops[i].offset = 1;
            prevOffset = stops[i].offset;
        }
        fillGrad->colorStops(stops, stopCount);
        free(stops);
    }
    return fillGrad;
}


static bool _appendChildShape(SvgNode* node, Shape* shape, float vx, float vy, float vw, float vh)
{
    auto valid = false;

    if (_appendShape(node, shape, vx, vy, vw, vh)) valid = true;

    if (node->child.count > 0) {
        auto child = node->child.data;
        for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
            if (_appendChildShape(*child, shape, vx, vy, vw, vh)) valid = true;
        }
    }

    return valid;
}


static void _applyComposition(Paint* paint, const SvgNode* node, float vx, float vy, float vw, float vh)
{
    if (node->style->comp.method == CompositeMethod::None) return;

    /* Do not drop in Circular Dependency.
       Composition can be applied recursively if its children nodes have composition target to this one. */
    if (node->style->comp.applying) {
#ifdef THORVG_LOG_ENABLED
    printf("SVG: Multiple Composition Tried! Check out Circular dependency?\n");
#endif
        return;
    }

    auto compNode = node->style->comp.node;
    if (!compNode || compNode->child.count == 0) return;

    node->style->comp.applying = true;

    auto comp = Shape::gen();
    comp->fill(255, 255, 255, 255);
    if (node->transform) comp->transform(*node->transform);

    auto child = compNode->child.data;
    auto valid = false;      //Composite only when valid shapes are existed

    for (uint32_t i = 0; i < compNode->child.count; ++i, ++child) {
        if (_appendChildShape(*child, comp.get(), vx, vy, vw, vh)) valid = true;
    }

    if (valid) paint->composite(move(comp), node->style->comp.method);

    node->style->comp.applying = false;
}


static void _applyProperty(SvgNode* node, Shape* vg, float vx, float vy, float vw, float vh)
{
    SvgStyleProperty* style = node->style;

    if (node->transform) vg->transform(*node->transform);
    if (node->type == SvgNodeType::Doc || !node->display) return;

    //If fill property is nullptr then do nothing
    if (style->fill.paint.none) {
        //Do nothing
    } else if (style->fill.paint.gradient) {
        if (!style->fill.paint.gradient->userSpace) vg->bounds(&vx, &vy, &vw, &vh);

        if (style->fill.paint.gradient->type == SvgGradientType::Linear) {
             auto linear = _applyLinearGradientProperty(style->fill.paint.gradient, vg, vx, vy, vw, vh, style->fill.opacity);
             vg->fill(move(linear));
        } else if (style->fill.paint.gradient->type == SvgGradientType::Radial) {
             auto radial = _applyRadialGradientProperty(style->fill.paint.gradient, vg, vx, vy, vw, vh, style->fill.opacity);
             vg->fill(move(radial));
        }
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
        if (!style->stroke.paint.gradient->userSpace) vg->bounds(&vx, &vy, &vw, &vh);

        if (style->stroke.paint.gradient->type == SvgGradientType::Linear) {
             auto linear = _applyLinearGradientProperty(style->stroke.paint.gradient, vg, vx, vy, vw, vh, style->stroke.opacity);
             vg->stroke(move(linear));
        } else if (style->stroke.paint.gradient->type == SvgGradientType::Radial) {
             auto radial = _applyRadialGradientProperty(style->stroke.paint.gradient, vg, vx, vy, vw, vh, style->stroke.opacity);
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

    _applyComposition(vg, node, vx, vy, vw, vh);
}


static unique_ptr<Shape> _shapeBuildHelper(SvgNode* node, float vx, float vy, float vw, float vh)
{
    auto shape = Shape::gen();
    if (_appendShape(node, shape.get(), vx, vy, vw, vh)) return shape;
    else return nullptr;
}


static bool _appendShape(SvgNode* node, Shape* shape, float vx, float vy, float vw, float vh)
{
    Array<PathCommand> cmds;
    Array<Point> pts;

    switch (node->type) {
        case SvgNodeType::Path: {
            if (node->node.path.path) {
                if (svgPathToTvgPath(node->node.path.path->c_str(), cmds, pts)) {
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

    _applyProperty(node, shape, vx, vy, vw, vh);
    return true;
}


static unique_ptr<Scene> _sceneBuildHelper(const SvgNode* node, float vx, float vy, float vw, float vh)
{
    if (_isGroupType(node->type)) {
        auto scene = Scene::gen();
        if (node->transform) scene->transform(*node->transform);

        if (node->display && node->style->opacity != 0) {
            auto child = node->child.data;
            for (uint32_t i = 0; i < node->child.count; ++i, ++child) {
                if (_isGroupType((*child)->type)) {
                    scene->push(_sceneBuildHelper(*child, vx, vy, vw, vh));
                } else {
                    auto shape = _shapeBuildHelper(*child, vx, vy, vw, vh);
                    if (shape) scene->push(move(shape));
                }
            }
            _applyComposition(scene.get(), node, vx, vy, vw, vh);
            scene->opacity(node->style->opacity);
        }
        return scene;
    }
    return nullptr;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

unique_ptr<Scene> svgSceneBuild(SvgNode* node, float vx, float vy, float vw, float vh)
{
    if (!node || (node->type != SvgNodeType::Doc)) return nullptr;

    auto docNode = _sceneBuildHelper(node, vx, vy, vw, vh);

    auto viewBoxClip = Shape::gen();
    viewBoxClip->appendRect(vx, vy ,vw, vh, 0, 0);
    viewBoxClip->fill(0, 0, 0, 255);

    auto compositeLayer = Scene::gen();
    compositeLayer->composite(move(viewBoxClip), CompositeMethod::ClipPath);
    compositeLayer->push(move(docNode));

    auto root = Scene::gen();
    root->push(move(compositeLayer));

    return root;
}
