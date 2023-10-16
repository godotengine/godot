/*
 * Copyright (c) 2022 - 2023 the ThorVG project. All rights reserved.

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

#include "tvgSvgCssStyle.h"

#include <cstring>

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static bool _isImportanceApplicable(SvgStyleFlags &toFlagsImportance, SvgStyleFlags fromFlagsImportance, SvgStyleFlags flag)
{
    if (!(toFlagsImportance & flag) && (fromFlagsImportance & flag)) {
        return true;
    }
    return false;
}

static void _copyStyle(SvgStyleProperty* to, const SvgStyleProperty* from)
{
    if (from == nullptr) return;
    //Copy the properties of 'from' only if they were explicitly set (not the default ones).
    if ((from->curColorSet && !(to->flags & SvgStyleFlags::Color)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Color)) {
        to->color = from->color;
        to->curColorSet = true;
        to->flags = (to->flags | SvgStyleFlags::Color);
        if (from->flagsImportance & SvgStyleFlags::Color) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::Color);
        }
    }
    //Fill
    if (((from->fill.flags & SvgFillFlags::Paint) && !(to->flags & SvgStyleFlags::Fill)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Fill)) {
        to->fill.paint.color = from->fill.paint.color;
        to->fill.paint.none = from->fill.paint.none;
        to->fill.paint.curColor = from->fill.paint.curColor;
        if (from->fill.paint.url) {
            if (to->fill.paint.url) free(to->fill.paint.url);
            to->fill.paint.url = strdup(from->fill.paint.url);
        }
        to->fill.flags = (to->fill.flags | SvgFillFlags::Paint);
        to->flags = (to->flags | SvgStyleFlags::Fill);
        if (from->flagsImportance & SvgStyleFlags::Fill) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::Fill);
        }
    }
    if (((from->fill.flags & SvgFillFlags::Opacity) && !(to->flags & SvgStyleFlags::FillOpacity)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::FillOpacity)) {
        to->fill.opacity = from->fill.opacity;
        to->fill.flags = (to->fill.flags | SvgFillFlags::Opacity);
        to->flags = (to->flags | SvgStyleFlags::FillOpacity);
        if (from->flagsImportance & SvgStyleFlags::FillOpacity) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::FillOpacity);
        }
    }
    if (((from->fill.flags & SvgFillFlags::FillRule) && !(to->flags & SvgStyleFlags::FillRule)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::FillRule)) {
        to->fill.fillRule = from->fill.fillRule;
        to->fill.flags = (to->fill.flags | SvgFillFlags::FillRule);
        to->flags = (to->flags | SvgStyleFlags::FillRule);
        if (from->flagsImportance & SvgStyleFlags::FillRule) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::FillRule);
        }
    }
    //Stroke
    if (((from->stroke.flags & SvgStrokeFlags::Paint) && !(to->flags & SvgStyleFlags::Stroke)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Stroke)) {
        to->stroke.paint.color = from->stroke.paint.color;
        to->stroke.paint.none = from->stroke.paint.none;
        to->stroke.paint.curColor = from->stroke.paint.curColor;
        if (from->stroke.paint.url) {
            if (to->stroke.paint.url) free(to->stroke.paint.url);
            to->stroke.paint.url = strdup(from->stroke.paint.url);
        }
        to->stroke.flags = (to->stroke.flags | SvgStrokeFlags::Paint);
        to->flags = (to->flags | SvgStyleFlags::Stroke);
        if (from->flagsImportance & SvgStyleFlags::Stroke) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::Stroke);
        }
    }
    if (((from->stroke.flags & SvgStrokeFlags::Opacity) && !(to->flags & SvgStyleFlags::StrokeOpacity)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeOpacity)) {
        to->stroke.opacity = from->stroke.opacity;
        to->stroke.flags = (to->stroke.flags | SvgStrokeFlags::Opacity);
        to->flags = (to->flags | SvgStyleFlags::StrokeOpacity);
        if (from->flagsImportance & SvgStyleFlags::StrokeOpacity) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::StrokeOpacity);
        }
    }
    if (((from->stroke.flags & SvgStrokeFlags::Width) && !(to->flags & SvgStyleFlags::StrokeWidth)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeWidth)) {
        to->stroke.width = from->stroke.width;
        to->stroke.flags = (to->stroke.flags | SvgStrokeFlags::Width);
        to->flags = (to->flags | SvgStyleFlags::StrokeWidth);
        if (from->flagsImportance & SvgStyleFlags::StrokeWidth) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::StrokeWidth);
        }
    }
    if (((from->stroke.flags & SvgStrokeFlags::Dash) && !(to->flags & SvgStyleFlags::StrokeDashArray)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeDashArray)) {
        if (from->stroke.dash.array.count > 0) {
            to->stroke.dash.array.clear();
            to->stroke.dash.array.reserve(from->stroke.dash.array.count);
            for (uint32_t i = 0; i < from->stroke.dash.array.count; ++i) {
                to->stroke.dash.array.push(from->stroke.dash.array[i]);
            }
            to->stroke.flags = (to->stroke.flags | SvgStrokeFlags::Dash);
            to->flags = (to->flags | SvgStyleFlags::StrokeDashArray);
            if (from->flagsImportance & SvgStyleFlags::StrokeDashArray) {
                to->flagsImportance = (to->flagsImportance | SvgStyleFlags::StrokeDashArray);
            }
        }
    }
    if (((from->stroke.flags & SvgStrokeFlags::Cap) && !(to->flags & SvgStyleFlags::StrokeLineCap)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeLineCap)) {
        to->stroke.cap = from->stroke.cap;
        to->stroke.flags = (to->stroke.flags | SvgStrokeFlags::Cap);
        to->flags = (to->flags | SvgStyleFlags::StrokeLineCap);
        if (from->flagsImportance & SvgStyleFlags::StrokeLineCap) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::StrokeLineCap);
        }
    }
    if (((from->stroke.flags & SvgStrokeFlags::Join) && !(to->flags & SvgStyleFlags::StrokeLineJoin)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::StrokeLineJoin)) {
        to->stroke.join = from->stroke.join;
        to->stroke.flags = (to->stroke.flags | SvgStrokeFlags::Join);
        to->flags = (to->flags | SvgStyleFlags::StrokeLineJoin);
        if (from->flagsImportance & SvgStyleFlags::StrokeLineJoin) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::StrokeLineJoin);
        }
    }
    //Opacity
    //TODO: it can be set to be 255 and shouldn't be changed by attribute 'opacity'
    if ((from->opacity < 255 && !(to->flags & SvgStyleFlags::Opacity)) ||
        _isImportanceApplicable(to->flagsImportance, from->flagsImportance, SvgStyleFlags::Opacity)) {
        to->opacity = from->opacity;
        to->flags = (to->flags | SvgStyleFlags::Opacity);
        if (from->flagsImportance & SvgStyleFlags::Opacity) {
            to->flagsImportance = (to->flagsImportance | SvgStyleFlags::Opacity);
        }
    }
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

void cssCopyStyleAttr(SvgNode* to, const SvgNode* from)
{
    //Copy matrix attribute
    if (from->transform && !(to->style->flags & SvgStyleFlags::Transform)) {
        to->transform = (Matrix*)malloc(sizeof(Matrix));
        if (to->transform) {
            *to->transform = *from->transform;
            to->style->flags = (to->style->flags | SvgStyleFlags::Transform);
        }
    }
    //Copy style attribute
    _copyStyle(to->style, from->style);

    if (from->style->clipPath.url) {
        if (to->style->clipPath.url) free(to->style->clipPath.url);
        to->style->clipPath.url = strdup(from->style->clipPath.url);
    }
    if (from->style->mask.url) {
        if (to->style->mask.url) free(to->style->mask.url);
        to->style->mask.url = strdup(from->style->mask.url);
    }
}


SvgNode* cssFindStyleNode(const SvgNode* style, const char* title, SvgNodeType type)
{
    if (!style) return nullptr;

    auto child = style->child.data;
    for (uint32_t i = 0; i < style->child.count; ++i, ++child) {
        if ((*child)->type == type) {
            if ((!title && !(*child)->id) || (title && (*child)->id && !strcmp((*child)->id, title))) return (*child);
        }
    }
    return nullptr;
}


SvgNode* cssFindStyleNode(const SvgNode* style, const char* title)
{
    if (!style || !title) return nullptr;

    auto child = style->child.data;
    for (uint32_t i = 0; i < style->child.count; ++i, ++child) {
        if ((*child)->type == SvgNodeType::CssStyle) {
            if ((*child)->id && !strcmp((*child)->id, title)) return (*child);
        }
    }
    return nullptr;
}


void cssUpdateStyle(SvgNode* doc, SvgNode* style)
{
    if (doc->child.count > 0) {
        auto child = doc->child.data;
        for (uint32_t i = 0; i < doc->child.count; ++i, ++child) {
            if (auto cssNode = cssFindStyleNode(style, nullptr, (*child)->type)) {
                cssCopyStyleAttr(*child, cssNode);
            }
            cssUpdateStyle(*child, style);
        }
    }
}


void cssApplyStyleToPostponeds(Array<SvgNodeIdPair>& postponeds, SvgNode* style)
{
    for (uint32_t i = 0; i < postponeds.count; ++i) {
        auto nodeIdPair = postponeds[i];

        //css styling: tag.name has higher priority than .name
        if (auto cssNode = cssFindStyleNode(style, nodeIdPair.id, nodeIdPair.node->type)) {
            cssCopyStyleAttr(nodeIdPair.node, cssNode);
        }
        if (auto cssNode = cssFindStyleNode(style, nodeIdPair.id)) {
            cssCopyStyleAttr(nodeIdPair.node, cssNode);
        }
    }
}
