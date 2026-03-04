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

#ifndef _TVG_RENDER_H_
#define _TVG_RENDER_H_

#include <math.h>
#include <cstdarg>
#include "tvgCommon.h"
#include "tvgArray.h"
#include "tvgLock.h"
#include "tvgColor.h"
#include "tvgMath.h"

namespace tvg
{

using RenderData = void*;
using RenderColor = tvg::RGBA;
using pixel_t = uint32_t;

#define DASH_PATTERN_THRESHOLD 0.001f

//TODO: Separate Color & Opacity for more detailed conditional check
enum RenderUpdateFlag : uint16_t {None = 0, Path = 1, Color = 2, Gradient = 4, Stroke = 8, Transform = 16, Image = 32, GradientStroke = 64, Blend = 128, Clip = 256, All = 0xffff};
enum CompositionFlag : uint8_t {Invalid = 0, Opacity = 1, Blending = 2, Masking = 4, PostProcessing = 8};  //Composition Purpose

static inline void operator|=(RenderUpdateFlag& a, const RenderUpdateFlag b)
{
    a = RenderUpdateFlag(uint16_t(a) | uint16_t(b));
}

static inline RenderUpdateFlag operator|(const RenderUpdateFlag a, const RenderUpdateFlag b)
{
    return RenderUpdateFlag(uint16_t(a) | uint16_t(b));
}

struct RenderSurface
{
    union {
        pixel_t* data = nullptr;    //system based data pointer
        uint32_t* buf32;            //for explicit 32bits channels
        uint8_t*  buf8;             //for explicit 8bits grayscale
    };
    Key key;                        //a reserved lock for the thread safety
    uint32_t stride = 0;
    uint32_t w = 0, h = 0;
    ColorSpace cs = ColorSpace::Unknown;
    uint8_t channelSize = 0;
    bool premultiplied = false;         //Alpha-premultiplied

    RenderSurface()
    {
    }

    RenderSurface(const RenderSurface* rhs)
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

struct RenderCompositor
{
    MaskMethod method;
    uint8_t opacity;
};

struct RenderRegion
{
    struct {
        int32_t x, y;
    } min;

    struct {
        int32_t x, y;
    } max;

    static constexpr RenderRegion intersect(const RenderRegion& lhs, const RenderRegion& rhs)
    {
        RenderRegion ret = {{std::max(lhs.min.x, rhs.min.x), std::max(lhs.min.y, rhs.min.y)}, {std::min(lhs.max.x, rhs.max.x), std::min(lhs.max.y, rhs.max.y)}};
        // Not intersected: collapse to zero-area region
        if (ret.min.x > ret.max.x) ret.max.x = ret.min.x;
        if (ret.min.y > ret.max.y) ret.max.y = ret.min.y;
        return ret;
    }

    static constexpr RenderRegion add(const RenderRegion& lhs, const RenderRegion& rhs)
    {
        return {{std::min(lhs.min.x, rhs.min.x), std::min(lhs.min.y, rhs.min.y)}, {std::max(lhs.max.x, rhs.max.x), std::max(lhs.max.y, rhs.max.y)}};
    }

    void intersect(const RenderRegion& rhs);

    void add(const RenderRegion& rhs)
    {
        if (rhs.min.x < min.x) min.x = rhs.min.x;
        if (rhs.min.y < min.y) min.y = rhs.min.y;
        if (rhs.max.x > max.x) max.x = rhs.max.x;
        if (rhs.max.y > max.y) max.y = rhs.max.y;
    }

    bool contained(const RenderRegion& rhs) const
    {
        return (min.x <= rhs.min.x && max.x >= rhs.max.x && min.y <= rhs.min.y && max.y >= rhs.max.y);
    }

    bool intersected(const RenderRegion& rhs) const
    {
        return (rhs.min.x < max.x && rhs.max.x > min.x && rhs.min.y < max.y && rhs.max.y > min.y);
    }

    bool operator==(const RenderRegion& rhs) const
    {
        return (min.x == rhs.min.x && min.y == rhs.min.y && max.x == rhs.max.x && max.y == rhs.max.y);
    }

    void reset() { min.x = min.y = max.x = max.y = 0; }
    bool valid() const { return (max.x > min.x && max.y > min.y); }
    bool invalid() const { return !valid(); }

    int32_t sx() const { return min.x; }
    int32_t sy() const { return min.y; }
    int32_t sw() const { return max.x - min.x; }
    int32_t sh() const { return max.y - min.y; }

    uint32_t x() const { return (uint32_t) sx(); }
    uint32_t y() const { return (uint32_t) sy(); }
    uint32_t w() const { return (uint32_t) sw(); }
    uint32_t h() const { return (uint32_t) sh(); }
};


#ifdef THORVG_PARTIAL_RENDER_SUPPORT
    struct RenderDirtyRegion
    {
    public:
        static constexpr const int PARTITIONING = 16;   //must be N*N
        bool support = true;

        void init(uint32_t w, uint32_t h);
        void commit();
        bool add(const RenderRegion& bbox);
        bool add(const RenderRegion& prv, const RenderRegion& cur);  //collect the old and new dirty regions together
        void clear();

        bool deactivate(bool on)
        {
            std::swap(on, disabled);
            return on;
        }

        bool deactivated()
        {
            return (!support || disabled);
        }

        const RenderRegion& partition(int idx)
        {
            return partitions[idx].region;
        }

        const Array<RenderRegion>& get(int idx)
        {
            return partitions[idx].list[partitions[idx].current];
        }

    private:
        void subdivide(Array<RenderRegion>& targets, uint32_t idx, RenderRegion& lhs, RenderRegion& rhs);

        struct Partition
        {
            RenderRegion region;
            Array<RenderRegion> list[2];  //double buffer swapping
            uint8_t current = 0;  //double buffer swapping list index. 0 or 1
        };

        Key key;
        Partition partitions[PARTITIONING];
        bool disabled = false;
    };
#else
    struct RenderDirtyRegion
    {
        static constexpr const int PARTITIONING = 16;   //must be N*N
        bool support = true;

        void init(uint32_t w, uint32_t h) {}
        void commit() {}
        bool add(TVG_UNUSED const RenderRegion& bbox) { return true; }
        bool add(TVG_UNUSED const RenderRegion& prv, TVG_UNUSED const RenderRegion& cur) { return true; }
        void clear() {}
        bool deactivate(TVG_UNUSED bool on) { return true; }
        bool deactivated() { return true; }
        const RenderRegion& partition(TVG_UNUSED int idx) { static RenderRegion tmp{}; return tmp; }
        const Array<RenderRegion>& get(TVG_UNUSED int idx) { static Array<RenderRegion> tmp; return tmp; }
    };
#endif


struct RenderPath
{
    Array<PathCommand> cmds;
    Array<Point> pts;

    bool empty() const
    {
        return pts.empty();
    }

    void clear()
    {
        pts.clear();
        cmds.clear();
    }

    void close()
    {
        //Don't close multiple times.
        if (cmds.count > 0 && cmds.last() == PathCommand::Close) return;
        cmds.push(PathCommand::Close);
    }

    void moveTo(const Point& pt)
    {
        pts.push(pt);
        cmds.push(PathCommand::MoveTo);
    }

    void lineTo(const Point& pt)
    {
        pts.push(pt);
        cmds.push(PathCommand::LineTo);
    }

    void cubicTo(const Point& cnt1, const Point& cnt2, const Point& end)
    {
        pts.push(cnt1);
        pts.push(cnt2);
        pts.push(end);
        cmds.push(PathCommand::CubicTo);
    }

    Point point(float progress)
    {
        if (progress <= 0.0f) return pts.first();
        else if (progress >= 1.0f) return pts.last();

        auto pleng = tvg::length(cmds.data, cmds.count, pts.data, pts.count) * progress;
        auto cleng = 0.0f;
        auto p = pts.data;
        auto c = cmds.data;
        Point curr{}, start{}, next{};

        while (c < cmds.data + cmds.count) {
            switch (*c) {
                case PathCommand::MoveTo: {
                    curr = start = *p++;
                    break;
                }
                case PathCommand::LineTo: {
                    next = *p;
                    auto segLen = tvg::length(curr, next);
                    if (cleng + segLen >= pleng) return lerp(curr, next, (pleng - cleng) / segLen);
                    cleng += segLen;
                    curr = *p++;
                    break;
                }
                case PathCommand::CubicTo: {
                    Bezier bz = {curr, *p, *(p + 1), *(p + 2)};
                    auto segLen = bz.length();
                    if (cleng + segLen >= pleng) return bz.at((pleng - cleng) / segLen);
                    cleng += segLen;
                    curr = *(p + 2);
                    p += 3;
                    break;
                }
                case PathCommand::Close: {
                    auto segLen = tvg::length(curr, start);
                    if (cleng + segLen >= pleng) return lerp(curr, start, (pleng - cleng) / segLen);
                    cleng += segLen;
                    curr = start;
                    break;
                }
            }
            ++c;
        }
        return curr;
    }

    /* Optimize path in screen space with merging collinear lines,
       collapsing zero length lines, and removing unnecessary cubic beziers. */
    void optimizeWG(RenderPath& out, const Matrix& matrix) const;
    void optimizeGL(RenderPath& out, const Matrix& matrix) const;
    bool bounds(const Matrix* m, BBox& box);
};

struct RenderTrimPath
{
    float begin = 0.0f;
    float end = 1.0f;
    bool simultaneous = true;

    bool valid()
    {
        if (begin != 0.0f || end != 1.0f) return true;
        return false;
    }

    bool trim(const RenderPath& in, RenderPath& out) const;
};

struct RenderStroke
{
    float width = 0.0f;
    RenderColor color{};
    Fill *fill = nullptr;
    struct Dash {
        float* pattern = nullptr;
        uint32_t count = 0;
        float offset = 0.0f;
        float length = 0.0f;
    } dash;
    float miterlimit = 4.0f;
    RenderTrimPath trim;
    StrokeCap cap = StrokeCap::Square;
    StrokeJoin join = StrokeJoin::Bevel;
    bool first = false;

    void operator=(const RenderStroke& rhs)
    {
        width = rhs.width;
        color = rhs.color;

        delete(fill);
        if (rhs.fill) fill = rhs.fill->duplicate();
        else fill = nullptr;

        tvg::free(dash.pattern);
        dash = rhs.dash;
        if (rhs.dash.count > 0) {
            dash.pattern = tvg::malloc<float>(sizeof(float) * rhs.dash.count);
            memcpy(dash.pattern, rhs.dash.pattern, sizeof(float) * rhs.dash.count);
        }

        miterlimit = rhs.miterlimit;
        trim = rhs.trim;
        cap = rhs.cap;
        join = rhs.join;
        first = rhs.first;
    }

    ~RenderStroke()
    {
        tvg::free(dash.pattern);
        delete(fill);
    }
};

struct RenderShape
{
    RenderPath path;
    Fill *fill = nullptr;
    RenderColor color{};
    RenderStroke *stroke = nullptr;
    FillRule rule = FillRule::NonZero;

    ~RenderShape()
    {
        delete(fill);
        delete(stroke);
    }

    void fillColor(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) const
    {
        if (r) *r = color.r;
        if (g) *g = color.g;
        if (b) *b = color.b;
        if (a) *a = color.a;
    }

    bool trimpath() const
    {
        return stroke ? stroke->trim.valid() : false;
    }

    bool strokeFirst() const
    {
        return (stroke && stroke->first) ? true : false;
    }

    float strokeWidth() const
    {
        return stroke ? stroke->width : 0.0f;
    }

    bool strokeFill(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) const
    {
        if (!stroke) return false;

        if (r) *r = stroke->color.r;
        if (g) *g = stroke->color.g;
        if (b) *b = stroke->color.b;
        if (a) *a = stroke->color.a;

        return true;
    }

    const Fill* strokeFill() const
    {
        return stroke ? stroke->fill : nullptr;
    }

    uint32_t strokeDash(const float** dashPattern, float* offset) const
    {
        if (!stroke) return 0;
        if (dashPattern) *dashPattern = stroke->dash.pattern;
        if (offset) *offset = stroke->dash.offset;
        return stroke->dash.count;
    }

    StrokeCap strokeCap() const
    {
        return stroke ? stroke->cap : StrokeCap::Square;
    }

    StrokeJoin strokeJoin() const
    {
        return stroke ? stroke->join : StrokeJoin::Bevel;
    }

    float strokeMiterlimit() const
    {
        return stroke ? stroke->miterlimit : 4.0f;
    }

    bool strokeDash(RenderPath& out, const Matrix* transform = nullptr) const;
};

struct RenderEffect
{
    RenderData rd = nullptr;
    RenderRegion extend{};
    SceneEffect type;
    bool valid = false;

    virtual ~RenderEffect() {}
};

struct RenderEffectGaussianBlur : RenderEffect
{
    float sigma;
    uint8_t direction; //0: both, 1: horizontal, 2: vertical
    uint8_t border;    //0: duplicate, 1: wrap
    uint8_t quality;   //0 ~ 100  (optional)

    static RenderEffectGaussianBlur* gen(va_list& args)
    {
        auto inst = new RenderEffectGaussianBlur;
        inst->sigma = std::max((float) va_arg(args, double), 0.0f);
        inst->direction = tvg::clamp(va_arg(args, int), 0, 2);
        inst->border = std::min(va_arg(args, int), 1);
        inst->quality = std::min(va_arg(args, int), 100);
        inst->type = SceneEffect::GaussianBlur;
        return inst;
    }
};

struct RenderEffectDropShadow : RenderEffect
{
    uint8_t color[4];  //rgba
    float angle;
    float distance;
    float sigma;
    uint8_t quality;   //0 ~ 100  (optional)

    static RenderEffectDropShadow* gen(va_list& args)
    {
        auto inst = new RenderEffectDropShadow;
        inst->color[0] = va_arg(args, int);
        inst->color[1] = va_arg(args, int);
        inst->color[2] = va_arg(args, int);
        inst->color[3] = va_arg(args, int);
        inst->angle = (float) va_arg(args, double);
        inst->distance = (float) va_arg(args, double);
        inst->sigma = std::max((float) va_arg(args, double), 0.0f);
        inst->quality = std::min(va_arg(args, int), 100);
        inst->type = SceneEffect::DropShadow;
        return inst;
    }
};

struct RenderEffectFill : RenderEffect
{
    uint8_t color[4];  //rgba

    static RenderEffectFill* gen(va_list& args)
    {
        auto inst = new RenderEffectFill;
        inst->color[0] = va_arg(args, int);
        inst->color[1] = va_arg(args, int);
        inst->color[2] = va_arg(args, int);
        inst->color[3] = va_arg(args, int);
        inst->type = SceneEffect::Fill;
        return inst;
    }
};

struct RenderEffectTint : RenderEffect
{
    uint8_t black[3];  //rgb
    uint8_t white[3];  //rgb
    uint8_t intensity; //0 - 255

    static RenderEffectTint* gen(va_list& args)
    {
        auto inst = new RenderEffectTint;
        inst->black[0] = va_arg(args, int);
        inst->black[1] = va_arg(args, int);
        inst->black[2] = va_arg(args, int);
        inst->white[0] = va_arg(args, int);
        inst->white[1] = va_arg(args, int);
        inst->white[2] = va_arg(args, int);
        inst->intensity = (uint8_t)(static_cast<float>(va_arg(args, double)) * 2.55f);
        inst->type = SceneEffect::Tint;
        return inst;
    }
};

struct RenderEffectTritone : RenderEffect
{
    uint8_t shadow[3];       //rgb
    uint8_t midtone[3];      //rgb
    uint8_t highlight[3];    //rgb
    uint8_t blender = 0;     //0 ~ 255

    static RenderEffectTritone* gen(va_list& args)
    {
        auto inst = new RenderEffectTritone;
        inst->shadow[0] = va_arg(args, int);
        inst->shadow[1] = va_arg(args, int);
        inst->shadow[2] = va_arg(args, int);
        inst->midtone[0] = va_arg(args, int);
        inst->midtone[1] = va_arg(args, int);
        inst->midtone[2] = va_arg(args, int);
        inst->highlight[0] = va_arg(args, int);
        inst->highlight[1] = va_arg(args, int);
        inst->highlight[2] = va_arg(args, int);
        inst->blender = va_arg(args, int);
        inst->type = SceneEffect::Tritone;
        return inst;
    }
};

struct RenderMethod
{
private:
    uint32_t refCnt = 0;
    Key key;

protected:
    RenderRegion vport;         //viewport

public:
    //common implementation
    uint32_t ref();
    uint32_t unref();
    RenderRegion viewport();
    bool viewport(const RenderRegion& vp);

    //main features
    virtual ~RenderMethod() {}
    virtual bool preUpdate() = 0;
    virtual RenderData prepare(const RenderShape& rshape, RenderData data, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags, bool clipper) = 0;
    virtual RenderData prepare(RenderSurface* surface, RenderData data, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags) = 0;
    virtual bool postUpdate() = 0;
    virtual bool preRender() = 0;
    virtual bool renderShape(RenderData data) = 0;
    virtual bool renderImage(RenderData data) = 0;
    virtual bool postRender() = 0;
    virtual void dispose(RenderData data) = 0;
    virtual RenderRegion region(RenderData data) = 0;
    virtual bool bounds(RenderData data, Point* pt4, const Matrix& m) = 0;
    virtual bool blend(BlendMethod method) = 0;
    virtual ColorSpace colorSpace() = 0;
    virtual const RenderSurface* mainSurface() = 0;
    virtual bool clear() = 0;
    virtual bool sync() = 0;
    virtual bool intersectsShape(RenderData data, const RenderRegion& region) = 0;
    virtual bool intersectsImage(RenderData data, const RenderRegion& region) = 0;

    //composition
    virtual RenderCompositor* target(const RenderRegion& region, ColorSpace cs, CompositionFlag flags) = 0;
    virtual bool beginComposite(RenderCompositor* cmp, MaskMethod method, uint8_t opacity) = 0;
    virtual bool endComposite(RenderCompositor* cmp) = 0;

    //post effects
    virtual void prepare(RenderEffect* effect, const Matrix& transform) = 0;
    virtual bool region(RenderEffect* effect) = 0;
    virtual bool render(RenderCompositor* cmp, const RenderEffect* effect, bool direct) = 0;
    virtual void dispose(RenderEffect* effect) = 0;

    //partial rendering
    virtual void damage(RenderData rd, const RenderRegion& region) = 0;
    virtual bool partial(bool disable) = 0;
};

static inline bool MASK_REGION_MERGING(MaskMethod method)
{
    switch(method) {
        case MaskMethod::Alpha:
        case MaskMethod::InvAlpha:
        case MaskMethod::Luma:
        case MaskMethod::InvLuma:
        case MaskMethod::Subtract:
        case MaskMethod::Intersect:
            return false;
        //these might expand the rendering region
        case MaskMethod::Add:
        case MaskMethod::Difference:
        case MaskMethod::Lighten:
        case MaskMethod::Darken:
            return true;
        default:
            TVGERR("RENDERER", "Unsupported Masking Method! = %d", (int)method);
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
        case ColorSpace::Unknown:
        default:
            TVGERR("RENDERER", "Unsupported Channel Size! = %d", (int)cs);
            return 0;
    }
}

static inline ColorSpace MASK_TO_COLORSPACE(RenderMethod* renderer, MaskMethod method)
{
    switch(method) {
        case MaskMethod::Alpha:
        case MaskMethod::InvAlpha:
        case MaskMethod::Add:
        case MaskMethod::Difference:
        case MaskMethod::Subtract:
        case MaskMethod::Intersect:
        case MaskMethod::Lighten:
        case MaskMethod::Darken:
            return ColorSpace::Grayscale8;
        //TODO: Optimize Luma/InvLuma colorspace to Grayscale8
        case MaskMethod::Luma:
        case MaskMethod::InvLuma:
            return renderer->colorSpace();
        default:
            TVGERR("RENDERER", "Unsupported Masking Size! = %d", (int)method);
            return ColorSpace::Unknown;
    }
}

static inline uint8_t MULTIPLY(uint8_t c, uint8_t a)
{
    return (((c) * (a) + 0xff) >> 8);
}

}

#endif //_TVG_RENDER_H_
