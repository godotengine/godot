/*
 * Copyright (c) 2020 - 2024 the ThorVG project. All rights reserved.

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

#ifndef _TVG_RENDER_H_
#define _TVG_RENDER_H_

#include "tvgCommon.h"
#include "tvgArray.h"
#include "tvgLock.h"

namespace tvg
{

using RenderData = void*;
using pixel_t = uint32_t;

enum RenderUpdateFlag : uint8_t {None = 0, Path = 1, Color = 2, Gradient = 4, Stroke = 8, Transform = 16, Image = 32, GradientStroke = 64, Blend = 128, All = 255};

struct Surface;

enum ColorSpace
{
    ABGR8888 = 0,      //The channels are joined in the order: alpha, blue, green, red. Colors are alpha-premultiplied.
    ARGB8888,          //The channels are joined in the order: alpha, red, green, blue. Colors are alpha-premultiplied.
    ABGR8888S,         //The channels are joined in the order: alpha, blue, green, red. Colors are un-alpha-premultiplied.
    ARGB8888S,         //The channels are joined in the order: alpha, red, green, blue. Colors are un-alpha-premultiplied.
    Grayscale8,        //One single channel data.
    Unsupported        //TODO: Change to the default, At the moment, we put it in the last to align with SwCanvas::Colorspace.
};

struct Surface
{
    union {
        pixel_t* data = nullptr;    //system based data pointer
        uint32_t* buf32;            //for explicit 32bits channels
        uint8_t*  buf8;             //for explicit 8bits grayscale
    };
    Key key;                        //a reserved lock for the thread safety
    uint32_t stride = 0;
    uint32_t w = 0, h = 0;
    ColorSpace cs = ColorSpace::Unsupported;
    uint8_t channelSize = 0;
    bool premultiplied = false;         //Alpha-premultiplied

    Surface()
    {
    }

    Surface(const Surface* rhs)
    {
        data = rhs->data;
        stride = rhs->stride;
        w = rhs->w;
        h = rhs->h;
        cs = rhs->cs;
        channelSize = rhs->channelSize;
        premultiplied = rhs->premultiplied;
    }


};

struct Compositor
{
    CompositeMethod method;
    uint8_t        opacity;
};

struct RenderMesh
{
    Polygon* triangles = nullptr;
    uint32_t triangleCnt = 0;

    ~RenderMesh()
    {
        free(triangles);
    }
};

struct RenderRegion
{
    int32_t x, y, w, h;

    void intersect(const RenderRegion& rhs)
    {
        auto x1 = x + w;
        auto y1 = y + h;
        auto x2 = rhs.x + rhs.w;
        auto y2 = rhs.y + rhs.h;

        x = (x > rhs.x) ? x : rhs.x;
        y = (y > rhs.y) ? y : rhs.y;
        w = ((x1 < x2) ? x1 : x2) - x;
        h = ((y1 < y2) ? y1 : y2) - y;

        if (w < 0) w = 0;
        if (h < 0) h = 0;
    }

    void add(const RenderRegion& rhs)
    {
        if (rhs.x < x) {
            w += (x - rhs.x);
            x = rhs.x;
        }
        if (rhs.y < y) {
            h += (y - rhs.y);
            y = rhs.y;
        }
        if (rhs.x + rhs.w > x + w) w = (rhs.x + rhs.w) - x;
        if (rhs.y + rhs.h > y + h) h = (rhs.y + rhs.h) - y;
    }
};

struct RenderTransform
{
    Matrix m;             //3x3 Matrix Elements
    float x = 0.0f;
    float y = 0.0f;
    float degree = 0.0f;  //rotation degree
    float scale = 1.0f;   //scale factor
    bool overriding = false;  //user transform?

    void update();
    void override(const Matrix& m);

    RenderTransform() {}
    RenderTransform(const RenderTransform* lhs, const RenderTransform* rhs);
};

struct RenderStroke
{
    float width = 0.0f;
    uint8_t color[4] = {0, 0, 0, 0};
    Fill *fill = nullptr;
    float* dashPattern = nullptr;
    uint32_t dashCnt = 0;
    float dashOffset = 0.0f;
    StrokeCap cap = StrokeCap::Square;
    StrokeJoin join = StrokeJoin::Bevel;
    float miterlimit = 4.0f;
    bool strokeFirst = false;

    struct {
        float begin = 0.0f;
        float end = 1.0f;
    } trim;

    ~RenderStroke()
    {
        free(dashPattern);
        delete(fill);
    }
};

struct RenderShape
{
    struct
    {
        Array<PathCommand> cmds;
        Array<Point> pts;
    } path;

    Fill *fill = nullptr;
    RenderStroke *stroke = nullptr;
    uint8_t color[4] = {0, 0, 0, 0};    //r, g, b, a
    FillRule rule = FillRule::Winding;

    ~RenderShape()
    {
        delete(fill);
        delete(stroke);
    }

    void fillColor(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) const
    {
        if (r) *r = color[0];
        if (g) *g = color[1];
        if (b) *b = color[2];
        if (a) *a = color[3];
    }

    float strokeWidth() const
    {
        if (!stroke) return 0;
        return stroke->width;
    }

    bool strokeTrim() const
    {
        if (!stroke) return false;
        if (stroke->trim.begin == 0.0f && stroke->trim.end == 1.0f) return false;
        if (stroke->trim.begin == 1.0f && stroke->trim.end == 0.0f) return false;
        return true;
    }

    bool strokeColor(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) const
    {
        if (!stroke) return false;

        if (r) *r = stroke->color[0];
        if (g) *g = stroke->color[1];
        if (b) *b = stroke->color[2];
        if (a) *a = stroke->color[3];

        return true;
    }

    const Fill* strokeFill() const
    {
        if (!stroke) return nullptr;
        return stroke->fill;
    }

    uint32_t strokeDash(const float** dashPattern, float* offset) const
    {
        if (!stroke) return 0;
        if (dashPattern) *dashPattern = stroke->dashPattern;
        if (offset) *offset = stroke->dashOffset;
        return stroke->dashCnt;
    }

    StrokeCap strokeCap() const
    {
        if (!stroke) return StrokeCap::Square;
        return stroke->cap;
    }

    StrokeJoin strokeJoin() const
    {
        if (!stroke) return StrokeJoin::Bevel;
        return stroke->join;
    }

    float strokeMiterlimit() const
    {
        if (!stroke) return 4.0f;

        return stroke->miterlimit;;
    }
};

class RenderMethod
{
private:
    uint32_t refCnt = 0;        //reference count
    Key key;

public:
    uint32_t ref()
    {
        ScopedLock lock(key);
        return (++refCnt);
    }

    uint32_t unref()
    {
        ScopedLock lock(key);
        return (--refCnt);
    }

    virtual ~RenderMethod() {}
    virtual RenderData prepare(const RenderShape& rshape, RenderData data, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags, bool clipper) = 0;
    virtual RenderData prepare(const Array<RenderData>& scene, RenderData data, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags) = 0;
    virtual RenderData prepare(Surface* surface, const RenderMesh* mesh, RenderData data, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags) = 0;
    virtual bool preRender() = 0;
    virtual bool renderShape(RenderData data) = 0;
    virtual bool renderImage(RenderData data) = 0;
    virtual bool postRender() = 0;
    virtual void dispose(RenderData data) = 0;
    virtual RenderRegion region(RenderData data) = 0;
    virtual RenderRegion viewport() = 0;
    virtual bool viewport(const RenderRegion& vp) = 0;
    virtual bool blend(BlendMethod method) = 0;
    virtual ColorSpace colorSpace() = 0;

    virtual bool clear() = 0;
    virtual bool sync() = 0;

    virtual Compositor* target(const RenderRegion& region, ColorSpace cs) = 0;
    virtual bool beginComposite(Compositor* cmp, CompositeMethod method, uint8_t opacity) = 0;
    virtual bool endComposite(Compositor* cmp) = 0;
};

static inline bool MASK_REGION_MERGING(CompositeMethod method)
{
    switch(method) {
        case CompositeMethod::AlphaMask:
        case CompositeMethod::InvAlphaMask:
        case CompositeMethod::LumaMask:
        case CompositeMethod::InvLumaMask:
        case CompositeMethod::SubtractMask:
        case CompositeMethod::IntersectMask:
            return false;
        //these might expand the rendering region
        case CompositeMethod::AddMask:
        case CompositeMethod::DifferenceMask:
            return true;
        default:
            TVGERR("RENDERER", "Unsupported Composite Method! = %d", (int)method);
            return false;
    }
}

static inline uint8_t CHANNEL_SIZE(ColorSpace cs)
{
    switch(cs) {
        case ColorSpace::ABGR8888:
        case ColorSpace::ABGR8888S:
        case ColorSpace::ARGB8888:
        case ColorSpace::ARGB8888S:
            return sizeof(uint32_t);
        case ColorSpace::Grayscale8:
            return sizeof(uint8_t);
        case ColorSpace::Unsupported:
        default:
            TVGERR("RENDERER", "Unsupported Channel Size! = %d", (int)cs);
            return 0;
    }
}

static inline ColorSpace COMPOSITE_TO_COLORSPACE(RenderMethod* renderer, CompositeMethod method)
{
    switch(method) {
        case CompositeMethod::AlphaMask:
        case CompositeMethod::InvAlphaMask:
        case CompositeMethod::AddMask:
        case CompositeMethod::DifferenceMask:
        case CompositeMethod::SubtractMask:
        case CompositeMethod::IntersectMask:
            return ColorSpace::Grayscale8;
        //TODO: Optimize Luma/InvLuma colorspace to Grayscale8
        case CompositeMethod::LumaMask:
        case CompositeMethod::InvLumaMask:
            return renderer->colorSpace();
        default:
            TVGERR("RENDERER", "Unsupported Composite Size! = %d", (int)method);
            return ColorSpace::Unsupported;
    }
}

static inline uint8_t MULTIPLY(uint8_t c, uint8_t a)
{
    return (((c) * (a) + 0xff) >> 8);
}


}

#endif //_TVG_RENDER_H_
