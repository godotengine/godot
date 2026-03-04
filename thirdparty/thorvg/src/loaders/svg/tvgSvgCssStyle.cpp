/*
 * Copyright (c) 2022 - 2026 ThorVG project. All rights reserved.

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

#include "tvgStr.h"
#include "tvgSvgCssStyle.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static inline bool _isImportanceApplicable(const SvgStyleFlags& toFlagsImportance, SvgStyleFlags fromFlagsImportance, SvgStyleFlags flag)
{
    return !(toFlagsImportance & flag) && (fromFlagsImportance & flag);
}


static void _copyStyle(SvgStyleProperty* to, const SvgStyleProperty* from, bool overwrite)
{
    if (!from) return;

    //Copy the properties of 'from' only if they were explicitly set (not the default ones).
    if ((from->curColorSet && (overwrite || !(to->flags & SvgStyleFlags::Color))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Color)) {
        to->color = from->color;
        to->curColorSet = true;
        to->flags |= SvgStyleFlags::Color;
        if (from->flagsImportance & SvgStyleFlags::Color) to->flagsImportance |= SvgStyleFlags::Color;
    }
    if (((from->flags & SvgStyleFlags::PaintOrder) && (overwrite || !(to->flags & SvgStyleFlags::PaintOrder))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::PaintOrder)) {
        to->paintOrder = from->paintOrder;
        to->flags |= SvgStyleFlags::PaintOrder;
        if (from->flagsImportance & SvgStyleFlags::PaintOrder) to->flagsImportance |= SvgStyleFlags::PaintOrder;
    }
    if (((from->flags & SvgStyleFlags::Display) && (overwrite || !(to->flags & SvgStyleFlags::Display))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Display)) {
        to->display = from->display;
        to->flags |= SvgStyleFlags::Display;
        if (from->flagsImportance & SvgStyleFlags::Display) to->flagsImportance |= SvgStyleFlags::Display;
    }
    //Fill
    if (((from->fill.flags & SvgFillFlags::Paint) && (overwrite || !(to->flags & SvgStyleFlags::Fill))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Fill)) {
        to->fill.paint.color = from->fill.paint.color;
        to->fill.paint.none = from->fill.paint.none;
        to->fill.paint.curColor = from->fill.paint.curColor;
        if (from->fill.paint.url) {
            if (to->fill.paint.url) tvg::free(to->fill.paint.url);
            to->fill.paint.url = duplicate(from->fill.paint.url);
        }
        to->fill.flags |= SvgFillFlags::Paint;
        to->flags |= SvgStyleFlags::Fill;
        if (from->flagsImportance & SvgStyleFlags::Fill) to->flagsImportance |= SvgStyleFlags::Fill;
    }
    if (((from->fill.flags & SvgFillFlags::Opacity) && (overwrite || !(to->flags & SvgStyleFlags::FillOpacity))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::FillOpacity)) {
        to->fill.opacity = from->fill.opacity;
        to->fill.flags |= SvgFillFlags::Opacity;
        to->flags |= SvgStyleFlags::FillOpacity;
        if (from->flagsImportance & SvgStyleFlags::FillOpacity) to->flagsImportance |= SvgStyleFlags::FillOpacity;
    }
    if (((from->fill.flags & SvgFillFlags::FillRule) && (overwrite || !(to->flags & SvgStyleFlags::FillRule))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::FillRule)) {
        to->fill.fillRule = from->fill.fillRule;
        to->fill.flags |= SvgFillFlags::FillRule;
        to->flags |= SvgStyleFlags::FillRule;
        if (from->flagsImportance & SvgStyleFlags::FillRule) to->flagsImportance |= SvgStyleFlags::FillRule;
    }
    //Stroke
    if (((from->stroke.flags & SvgStrokeFlags::Paint) && (overwrite || !(to->flags & SvgStyleFlags::Stroke))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Stroke)) {
        to->stroke.paint.color = from->stroke.paint.color;
        to->stroke.paint.none = from->stroke.paint.none;
        to->stroke.paint.curColor = from->stroke.paint.curColor;
        if (from->stroke.paint.url) {
            if (to->stroke.paint.url) tvg::free(to->stroke.paint.url);
            to->stroke.paint.url = duplicate(from->stroke.paint.url);
        }
        to->stroke.flags |= SvgStrokeFlags::Paint;
        to->flags |= SvgStyleFlags::Stroke;
        if (from->flagsImportance & SvgStyleFlags::Stroke) to->flagsImportance |= SvgStyleFlags::Stroke;
    }
    if (((from->stroke.flags & SvgStrokeFlags::Opacity) && (overwrite || !(to->flags & SvgStyleFlags::StrokeOpacity))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeOpacity)) {
        to->stroke.opacity = from->stroke.opacity;
        to->stroke.flags |= SvgStrokeFlags::Opacity;
        to->flags |= SvgStyleFlags::StrokeOpacity;
        if (from->flagsImportance & SvgStyleFlags::StrokeOpacity) to->flagsImportance |= SvgStyleFlags::StrokeOpacity;
    }
    if (((from->stroke.flags & SvgStrokeFlags::Width) && (overwrite || !(to->flags & SvgStyleFlags::StrokeWidth))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeWidth)) {
        to->stroke.width = from->stroke.width;
        to->stroke.flags |= SvgStrokeFlags::Width;
        to->flags |= SvgStyleFlags::StrokeWidth;
        if (from->flagsImportance & SvgStyleFlags::StrokeWidth) to->flagsImportance |= SvgStyleFlags::StrokeWidth;
    }
    if (((from->stroke.flags & SvgStrokeFlags::Dash) && (overwrite || !(to->flags & SvgStyleFlags::StrokeDashArray))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeDashArray)) {
        if (!from->stroke.dash.array.empty()) {
            to->stroke.dash.array.clear();
            to->stroke.dash.array.reserve(from->stroke.dash.array.count);
            ARRAY_FOREACH(p, from->stroke.dash.array) to->stroke.dash.array.push(*p);
            to->stroke.flags |= SvgStrokeFlags::Dash;
            to->flags |= SvgStyleFlags::StrokeDashArray;
            if (from->flagsImportance & SvgStyleFlags::StrokeDashArray) to->flagsImportance |= SvgStyleFlags::StrokeDashArray;
        }
    }
    if (((from->stroke.flags & SvgStrokeFlags::Cap) && (overwrite || !(to->flags & SvgStyleFlags::StrokeLineCap))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeLineCap)) {
        to->stroke.cap = from->stroke.cap;
        to->stroke.flags |= SvgStrokeFlags::Cap;
        to->flags |= SvgStyleFlags::StrokeLineCap;
        if (from->flagsImportance & SvgStyleFlags::StrokeLineCap) to->flagsImportance |= SvgStyleFlags::StrokeLineCap;
    }
    if (((from->stroke.flags & SvgStrokeFlags::Join) && (overwrite || !(to->flags & SvgStyleFlags::StrokeLineJoin))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeLineJoin)) {
        to->stroke.join = from->stroke.join;
        to->stroke.flags |= SvgStrokeFlags::Join;
        to->flags |= SvgStyleFlags::StrokeLineJoin;
        if (from->flagsImportance & SvgStyleFlags::StrokeLineJoin) to->flagsImportance |= SvgStyleFlags::StrokeLineJoin;
    }
    //Opacity
    if (((from->flags & SvgStyleFlags::Opacity) && (overwrite || !(to->flags & SvgStyleFlags::Opacity))) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Opacity)) {
        to->opacity = from->opacity;
        to->flags |= SvgStyleFlags::Opacity;
        if (from->flagsImportance & SvgStyleFlags::Opacity) to->flagsImportance |= SvgStyleFlags::Opacity;
    }
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/


void cssCopyStyleAttr(SvgNode* to, const SvgNode* from, bool overwrite)
{
    //Copy matrix attribute
    if (from->transform && (overwrite || !(to->style->flags & SvgStyleFlags::Transform))) {
        if (!to->transform) to->transform = tvg::malloc<Matrix>(sizeof(Matrix));
        *to->transform = *from->transform;
        to->style->flags |= SvgStyleFlags::Transform;
    }
    //Copy style attribute
    _copyStyle(to->style, from->style, overwrite);

    if (from->style->clipPath.url) {
        if (to->style->clipPath.url) tvg::free(to->style->clipPath.url);
        to->style->clipPath.url = duplicate(from->style->clipPath.url);
    }
    if (from->style->mask.url) {
        if (to->style->mask.url) tvg::free(to->style->mask.url);
        to->style->mask.url = duplicate(from->style->mask.url);
    }
}


SvgNode* cssFindStyleNode(const SvgNode* style, const char* title, SvgNodeType type)
{
    if (!style) return nullptr;
    ARRAY_FOREACH(p, style->child) {
        if ((*p)->type == type) {
            if ((!title && !(*p)->id) || (title && (*p)->id && STR_AS((*p)->id, title))) return *p;
        }
    }
    return nullptr;
}


SvgNode* cssFindStyleNode(const SvgNode* style, const char* title)
{
    if (!style || !title) return nullptr;
    ARRAY_FOREACH(p, style->child) {
        if ((*p)->type == SvgNodeType::CssStyle) {
            if ((*p)->id && STR_AS((*p)->id, title)) return *p;
        }
    }
    return nullptr;
}


void cssUpdateStyle(SvgNode* doc, SvgNode* style)
{
    ARRAY_FOREACH(p, doc->child) {
        if (auto cssNode = cssFindStyleNode(style, nullptr, (*p)->type)) {
            cssCopyStyleAttr(*p, cssNode);
        }
        cssUpdateStyle(*p, style);
    }
}
