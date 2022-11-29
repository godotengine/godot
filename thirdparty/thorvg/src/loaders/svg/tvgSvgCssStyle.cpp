/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All rights reserved.

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

static void _copyStyle(SvgStyleProperty* to, const SvgStyleProperty* from)
{
    if (from == nullptr) return;
    //Copy the properties of 'from' only if they were explicitly set (not the default ones).
    if (from->curColorSet && !((int)to->flags & (int)SvgStyleFlags::Color)) {
        to->color = from->color;
        to->curColorSet = true;
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::Color);
    }
    //Fill
    if (((int)from->fill.flags & (int)SvgFillFlags::Paint) && !((int)to->flags & (int)SvgStyleFlags::Fill)) {
        to->fill.paint.color = from->fill.paint.color;
        to->fill.paint.none = from->fill.paint.none;
        to->fill.paint.curColor = from->fill.paint.curColor;
        if (from->fill.paint.url) {
            if (to->fill.paint.url) free(to->fill.paint.url);
            to->fill.paint.url = strdup(from->fill.paint.url);
        }
        to->fill.flags = (SvgFillFlags)((int)to->fill.flags | (int)SvgFillFlags::Paint);
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::Fill);
    }
    if (((int)from->fill.flags & (int)SvgFillFlags::Opacity) && !((int)to->flags & (int)SvgStyleFlags::FillOpacity)) {
        to->fill.opacity = from->fill.opacity;
        to->fill.flags = (SvgFillFlags)((int)to->fill.flags | (int)SvgFillFlags::Opacity);
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::FillOpacity);
    }
    if (((int)from->fill.flags & (int)SvgFillFlags::FillRule) && !((int)to->flags & (int)SvgStyleFlags::FillRule)) {
        to->fill.fillRule = from->fill.fillRule;
        to->fill.flags = (SvgFillFlags)((int)to->fill.flags | (int)SvgFillFlags::FillRule);
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::FillRule);
    }
    //Stroke
    if (((int)from->stroke.flags & (int)SvgStrokeFlags::Paint) && !((int)to->flags & (int)SvgStyleFlags::Stroke)) {
        to->stroke.paint.color = from->stroke.paint.color;
        to->stroke.paint.none = from->stroke.paint.none;
        to->stroke.paint.curColor = from->stroke.paint.curColor;
        if (from->stroke.paint.url) {
            if (to->stroke.paint.url) free(to->stroke.paint.url);
            to->stroke.paint.url = strdup(from->stroke.paint.url);
        }
        to->stroke.flags = (SvgStrokeFlags)((int)to->stroke.flags | (int)SvgStrokeFlags::Paint);
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::Stroke);
    }
    if (((int)from->stroke.flags & (int)SvgStrokeFlags::Opacity) && !((int)to->flags & (int)SvgStyleFlags::StrokeOpacity)) {
        to->stroke.opacity = from->stroke.opacity;
        to->stroke.flags = (SvgStrokeFlags)((int)to->stroke.flags | (int)SvgStrokeFlags::Opacity);
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::StrokeOpacity);
    }
    if (((int)from->stroke.flags & (int)SvgStrokeFlags::Width) && !((int)to->flags & (int)SvgStyleFlags::StrokeWidth)) {
        to->stroke.width = from->stroke.width;
        to->stroke.flags = (SvgStrokeFlags)((int)to->stroke.flags | (int)SvgStrokeFlags::Width);
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::StrokeWidth);
    }
    if (((int)from->stroke.flags & (int)SvgStrokeFlags::Dash) && !((int)to->flags & (int)SvgStyleFlags::StrokeDashArray)) {
        if (from->stroke.dash.array.count > 0) {
            to->stroke.dash.array.clear();
            to->stroke.dash.array.reserve(from->stroke.dash.array.count);
            for (uint32_t i = 0; i < from->stroke.dash.array.count; ++i) {
                to->stroke.dash.array.push(from->stroke.dash.array.data[i]);
            }
            to->stroke.flags = (SvgStrokeFlags)((int)to->stroke.flags | (int)SvgStrokeFlags::Dash);
            to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::StrokeDashArray);
        }
    }
    if (((int)from->stroke.flags & (int)SvgStrokeFlags::Cap) && !((int)to->flags & (int)SvgStyleFlags::StrokeLineCap)) {
        to->stroke.cap = from->stroke.cap;
        to->stroke.flags = (SvgStrokeFlags)((int)to->stroke.flags | (int)SvgStrokeFlags::Cap);
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::StrokeLineCap);
    }
    if (((int)from->stroke.flags & (int)SvgStrokeFlags::Join) && !((int)to->flags & (int)SvgStyleFlags::StrokeLineJoin)) {
        to->stroke.join = from->stroke.join;
        to->stroke.flags = (SvgStrokeFlags)((int)to->stroke.flags | (int)SvgStrokeFlags::Join);
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::StrokeLineJoin);
    }
    //Opacity
    //TODO: it can be set to be 255 and shouldn't be changed by attribute 'opacity'
    if (from->opacity < 255 && !((int)to->flags & (int)SvgStyleFlags::Opacity)) {
        to->opacity = from->opacity;
        to->flags = (SvgStyleFlags)((int)to->flags | (int)SvgStyleFlags::Opacity);
    }
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

void cssCopyStyleAttr(SvgNode* to, const SvgNode* from)
{
    //Copy matrix attribute
    if (from->transform && !((int)to->style->flags & (int)SvgStyleFlags::Transform)) {
        to->transform = (Matrix*)malloc(sizeof(Matrix));
        if (to->transform) {
            *to->transform = *from->transform;
            to->style->flags = (SvgStyleFlags)((int)to->style->flags | (int)SvgStyleFlags::Transform);
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
    if (!style) return nullptr;

    auto child = style->child.data;
    for (uint32_t i = 0; i < style->child.count; ++i, ++child) {
        if ((*child)->type == SvgNodeType::CssStyle) {
            if ((title && (*child)->id && !strcmp((*child)->id, title))) return (*child);
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
            if (auto cssNode = cssFindStyleNode(style, nullptr)) {
                cssCopyStyleAttr(*child, cssNode);
            }
            cssUpdateStyle(*child, style);
        }
    }
}


void cssApplyStyleToPostponeds(Array<SvgNodeIdPair>& postponeds, SvgNode* style)
{
    for (uint32_t i = 0; i < postponeds.count; ++i) {
        auto nodeIdPair = postponeds.data[i];

        //css styling: tag.name has higher priority than .name
        if (auto cssNode = cssFindStyleNode(style, nodeIdPair.id, nodeIdPair.node->type)) {
            cssCopyStyleAttr(nodeIdPair.node, cssNode);
        }
        if (auto cssNode = cssFindStyleNode(style, nodeIdPair.id)) {
            cssCopyStyleAttr(nodeIdPair.node, cssNode);
        }
    }
}
