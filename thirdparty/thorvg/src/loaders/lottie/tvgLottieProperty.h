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

#ifndef _TVG_LOTTIE_PROPERTY_H_
#define _TVG_LOTTIE_PROPERTY_H_

#include "tvgCommon.h"
#include "tvgArray.h"
#include "tvgMath.h"
#include "tvgLines.h"
#include "tvgLottieInterpolator.h"

struct PathSet
{
    Point* pts = nullptr;
    PathCommand* cmds = nullptr;
    uint16_t ptsCnt = 0;
    uint16_t cmdsCnt = 0;
};


struct RGB24
{
    int32_t rgb[3];
};


struct ColorStop
{
    Fill::ColorStop* data = nullptr;
    Array<float>* input = nullptr;
};


struct LottieFont;

struct TextDocument
{
    char* text = nullptr;
    float height;
    float shift;
    RGB24 color;
    struct {
        Point pos;
        Point size;
    } bbox;
    struct {
        RGB24 color;
        float width;
        bool render = false;
    } stroke;
    char* name = nullptr;
    float size;
    uint8_t justify;
    uint8_t tracking;
};


static inline RGB24 operator-(const RGB24& lhs, const RGB24& rhs)
{
    return {lhs.rgb[0] - rhs.rgb[0], lhs.rgb[1] - rhs.rgb[1], lhs.rgb[2] - rhs.rgb[2]};
}


static inline RGB24 operator+(const RGB24& lhs, const RGB24& rhs)
{
    return {lhs.rgb[0] + rhs.rgb[0], lhs.rgb[1] + rhs.rgb[1], lhs.rgb[2] + rhs.rgb[2]};
}


static inline RGB24 operator*(const RGB24& lhs, float rhs)
{
    return {(int32_t)lroundf(lhs.rgb[0] * rhs), (int32_t)lroundf(lhs.rgb[1] * rhs), (int32_t)lroundf(lhs.rgb[2] * rhs)};
}


static void copy(PathSet& pathset, Array<Point>& outPts, Matrix* transform)
{
    Array<Point> inPts;

    if (transform) {
        for (int i = 0; i < pathset.ptsCnt; ++i) {
            Point pt = pathset.pts[i];
            mathMultiply(&pt, transform);
            outPts.push(pt);
        }
    } else {
        inPts.data = pathset.pts;
        inPts.count = pathset.ptsCnt;
        outPts.push(inPts);
        inPts.data = nullptr;
    }
}


static void copy(PathSet& pathset, Array<PathCommand>& outCmds)
{
    Array<PathCommand> inCmds;
    inCmds.data = pathset.cmds;
    inCmds.count = pathset.cmdsCnt;
    outCmds.push(inCmds);
    inCmds.data = nullptr;
}


template<typename T>
struct LottieScalarFrame
{
    T value;                    //keyframe value
    float no;                   //frame number
    LottieInterpolator* interpolator;
    bool hold = false;           //do not interpolate.

    T interpolate(LottieScalarFrame<T>* next, float frameNo)
    {
        auto t = (frameNo - no) / (next->no - no);
        if (interpolator) t = interpolator->progress(t);

        if (hold) {
            if (t < 1.0f) return value;
            else return next->value;
        }
        return mathLerp(value, next->value, t);
    }
};


template<typename T>
struct LottieVectorFrame
{
    T value;                    //keyframe value
    float no;                   //frame number
    LottieInterpolator* interpolator;
    T outTangent, inTangent;
    float length;
    bool hasTangent = false;
    bool hold = false;

    T interpolate(LottieVectorFrame* next, float frameNo)
    {
        auto t = (frameNo - no) / (next->no - no);
        if (interpolator) t = interpolator->progress(t);

        if (hold) {
            if (t < 1.0f) return value;
            else return next->value;
        }

        if (hasTangent) {
            Bezier bz = {value, value + outTangent, next->value + inTangent, next->value};
            t = bezAtApprox(bz, t * length, length);
            return bezPointAt(bz, t);
        } else {
            return mathLerp(value, next->value, t);
        }
    }

    float angle(LottieVectorFrame* next, float frameNo)
    {
        if (!hasTangent) return 0;
        auto t = (frameNo - no) / (next->no - no);
        if (interpolator) t = interpolator->progress(t);
        Bezier bz = {value, value + outTangent, next->value + inTangent, next->value};
        t = bezAtApprox(bz, t * length, length);
        return -bezAngleAt(bz, t);
    }

    void prepare(LottieVectorFrame* next)
    {
        Bezier bz = {value, value + outTangent, next->value + inTangent, next->value};
        length = bezLengthApprox(bz);
    }
};


template<typename T>
uint32_t bsearch(T* frames, float frameNo)
{
    uint32_t low = 0;
    uint32_t high = frames->count - 1;

    while (low <= high) {
        auto mid = low + (high - low) / 2;
        auto frame = frames->data + mid;
        if (mathEqual(frameNo, frame->no)) return mid;
        else if (frameNo < frame->no) high = mid - 1;
        else low = mid + 1;
    }
    if (high < low) low = high;
    return low;
}


struct LottieProperty
{
    enum class Type : uint8_t { Point = 0, Float, Opacity, Color, PathSet, ColorStop, Position, TextDoc, Invalid };
    virtual ~LottieProperty() {}
};


template<typename T>
struct LottieGenericProperty : LottieProperty
{
    //Property has an either keyframes or single value.
    Array<LottieScalarFrame<T>>* frames = nullptr;
    T value;

    LottieGenericProperty(T v) : value(v) {}
    LottieGenericProperty() {}

    ~LottieGenericProperty()
    {
        release();
    }

    void release()
    {
        delete(frames);
    }

    LottieScalarFrame<T>& newFrame()
    {
        if (!frames) frames = new Array<LottieScalarFrame<T>>;
        if (frames->count + 1 >= frames->reserved) {
            auto old = frames->reserved;
            frames->grow(frames->count + 2);
            memset((void*)(frames->data + old), 0x00, sizeof(LottieScalarFrame<T>) * (frames->reserved - old));
        }
        ++frames->count;
        return frames->last();
    }

    LottieScalarFrame<T>& nextFrame()
    {
        return (*frames)[frames->count];
    }

    T operator()(float frameNo)
    {
        if (!frames) return value;
        if (frames->count == 1 || frameNo <= frames->first().no) return frames->first().value;
        if (frameNo >= frames->last().no) return frames->last().value;

        auto frame = frames->data + bsearch(frames, frameNo);
        if (frame->no == frameNo) return frame->value;
        return frame->interpolate(frame + 1, frameNo);
    }

    T& operator=(const T& other)
    {
        //shallow copy, used for slot overriding
        if (other.frames) {
            frames = other.frames;
            const_cast<T&>(other).frames = nullptr;
        } else value = other.value;
        return *this;
    }

    float angle(float frameNo) { return 0; }
    void prepare() {}
};


struct LottiePathSet : LottieProperty
{
    Array<LottieScalarFrame<PathSet>>* frames = nullptr;
    PathSet value;

    ~LottiePathSet()
    {
        release();
    }

    void release()
    {
        free(value.cmds);
        free(value.pts);

        if (!frames) return;

        for (auto p = frames->begin(); p < frames->end(); ++p) {
            free((*p).value.cmds);
            free((*p).value.pts);
        }
        free(frames->data);
        free(frames);
    }

    LottieScalarFrame<PathSet>& newFrame()
    {
        if (!frames) {
            frames = static_cast<Array<LottieScalarFrame<PathSet>>*>(calloc(1, sizeof(Array<LottieScalarFrame<PathSet>>)));
        }
        if (frames->count + 1 >= frames->reserved) {
            auto old = frames->reserved;
            frames->grow(frames->count + 2);
            memset((void*)(frames->data + old), 0x00, sizeof(LottieScalarFrame<PathSet>) * (frames->reserved - old));
        }
        ++frames->count;
        return frames->last();
    }

    LottieScalarFrame<PathSet>& nextFrame()
    {
        return (*frames)[frames->count];
    }

    bool operator()(float frameNo, Array<PathCommand>& cmds, Array<Point>& pts, Matrix* transform = nullptr)
    {
        if (!frames) {
            copy(value, cmds);
            copy(value, pts, transform);
            return true;
        }

        if (frames->count == 1 || frameNo <= frames->first().no) {
            copy(frames->first().value, cmds);
            copy(frames->first().value, pts, transform);
            return true;
        }

        if (frameNo >= frames->last().no) {
            copy(frames->last().value, cmds);
            copy(frames->last().value, pts, transform);
            return true;
        }

        auto frame = frames->data + bsearch(frames, frameNo);

        if (frame->no == frameNo) {
            copy(frame->value, cmds);
            copy(frame->value, pts, transform);
            return true;
        }

        //interpolate
        copy(frame->value, cmds);

        auto t = (frameNo - frame->no) / ((frame + 1)->no - frame->no);
        if (frame->interpolator) t = frame->interpolator->progress(t);

        if (frame->hold) {
            if (t < 1.0f) copy(frame->value, pts, transform);
            else copy((frame + 1)->value, pts, transform);
            return true;
        }

        auto s = frame->value.pts;
        auto e = (frame + 1)->value.pts;

        for (auto i = 0; i < frame->value.ptsCnt; ++i, ++s, ++e) {
            auto pt = mathLerp(*s, *e, t);
            if (transform) mathMultiply(&pt,transform);
            pts.push(pt);
        }
        return true;
    }

    void prepare() {}
};


struct LottieColorStop : LottieProperty
{
    Array<LottieScalarFrame<ColorStop>>* frames = nullptr;
    ColorStop value;
    uint16_t count = 0;     //colorstop count
    bool populated = false;

    ~LottieColorStop()
    {
        release();
    }

    void release()
    {
        if (value.data) {
            free(value.data);
            value.data = nullptr;
        }

        if (!frames) return;

        for (auto p = frames->begin(); p < frames->end(); ++p) {
            free((*p).value.data);
        }
        free(frames->data);
        free(frames);
        frames = nullptr;
    }

    LottieScalarFrame<ColorStop>& newFrame()
    {
        if (!frames) {
            frames = static_cast<Array<LottieScalarFrame<ColorStop>>*>(calloc(1, sizeof(Array<LottieScalarFrame<ColorStop>>)));
        }
        if (frames->count + 1 >= frames->reserved) {
            auto old = frames->reserved;
            frames->grow(frames->count + 2);
            memset((void*)(frames->data + old), 0x00, sizeof(LottieScalarFrame<ColorStop>) * (frames->reserved - old));
        }
        ++frames->count;
        return frames->last();
    }

    LottieScalarFrame<ColorStop>& nextFrame()
    {
        return (*frames)[frames->count];
    }

    void operator()(float frameNo, Fill* fill)
    {
        if (!frames) {
            fill->colorStops(value.data, count);
            return;
        }

        if (frames->count == 1 || frameNo <= frames->first().no) {
            fill->colorStops(frames->first().value.data, count);
            return;
        }

        if (frameNo >= frames->last().no) {
            fill->colorStops(frames->last().value.data, count);
            return;
        }

        auto frame = frames->data + bsearch(frames, frameNo);
        if (frame->no == frameNo) {
            fill->colorStops(frame->value.data, count);
            return;
        }

        //interpolate
        auto t = (frameNo - frame->no) / ((frame + 1)->no - frame->no);
        if (frame->interpolator) t = frame->interpolator->progress(t);

        if (frame->hold) {
            if (t < 1.0f) fill->colorStops(frame->value.data, count);
            else fill->colorStops((frame + 1)->value.data, count);
        }

        auto s = frame->value.data;
        auto e = (frame + 1)->value.data;

        Array<Fill::ColorStop> result;

        for (auto i = 0; i < count; ++i, ++s, ++e) {
            auto offset = mathLerp(s->offset, e->offset, t);
            auto r = mathLerp(s->r, e->r, t);
            auto g = mathLerp(s->g, e->g, t);
            auto b = mathLerp(s->b, e->b, t);
            auto a = mathLerp(s->a, e->a, t);
            result.push({offset, r, g, b, a});
        }
        fill->colorStops(result.data, count);
    }

    LottieColorStop& operator=(const LottieColorStop& other)
    {
        //shallow copy, used for slot overriding
        if (other.frames) {
            frames = other.frames;
            const_cast<LottieColorStop&>(other).frames = nullptr;
        } else {
            value = other.value;
            const_cast<LottieColorStop&>(other).value = {nullptr, nullptr};
        }
        populated = other.populated;
        count = other.count;

        return *this;
    }

    void prepare() {}
};


struct LottiePosition : LottieProperty
{
    Array<LottieVectorFrame<Point>>* frames = nullptr;
    Point value;

    LottiePosition(Point v) : value(v)
    {
    }

    ~LottiePosition()
    {
        release();
    }

    void release()
    {
        delete(frames);
    }

    LottieVectorFrame<Point>& newFrame()
    {
        if (!frames) frames = new Array<LottieVectorFrame<Point>>;
        if (frames->count + 1 >= frames->reserved) {
            auto old = frames->reserved;
            frames->grow(frames->count + 2);
            memset((void*)(frames->data + old), 0x00, sizeof(LottieVectorFrame<Point>) * (frames->reserved - old));
        }
        ++frames->count;
        return frames->last();
    }

    LottieVectorFrame<Point>& nextFrame()
    {
        return (*frames)[frames->count];
    }

    Point operator()(float frameNo)
    {
        if (!frames) return value;
        if (frames->count == 1 || frameNo <= frames->first().no) return frames->first().value;
        if (frameNo >= frames->last().no) return frames->last().value;

        auto frame = frames->data + bsearch(frames, frameNo);
        if (frame->no == frameNo) return frame->value;
        return frame->interpolate(frame + 1, frameNo);
    }

    float angle(float frameNo)
    {
        if (!frames) return 0;
        if (frames->count == 1 || frameNo <= frames->first().no) return 0;
        if (frameNo >= frames->last().no) return 0;

        auto frame = frames->data + bsearch(frames, frameNo);
        return frame->angle(frame + 1, frameNo);
    }

    void prepare()
    {
        if (!frames || frames->count < 2) return;
        for (auto frame = frames->begin() + 1; frame < frames->end(); ++frame) {
            (frame - 1)->prepare(frame);
        }
    }
};


struct LottieTextDoc : LottieProperty
{
    Array<LottieScalarFrame<TextDocument>>* frames = nullptr;
    TextDocument value;

    ~LottieTextDoc()
    {
        release();
    }

    void release()
    {
        if (value.text) {
            free(value.text);
            value.text = nullptr;
        }
        if (value.name) {
            free(value.name);
            value.name = nullptr;
        }

        if (!frames) return;

        for (auto p = frames->begin(); p < frames->end(); ++p) {
            free((*p).value.text);
            free((*p).value.name);
        }
        delete(frames);
        frames = nullptr;
    }

    LottieScalarFrame<TextDocument>& newFrame()
    {
        if (!frames) frames = new Array<LottieScalarFrame<TextDocument>>;
        if (frames->count + 1 >= frames->reserved) {
            auto old = frames->reserved;
            frames->grow(frames->count + 2);
            memset((void*)(frames->data + old), 0x00, sizeof(LottieScalarFrame<TextDocument>) * (frames->reserved - old));
        }
        ++frames->count;
        return frames->last();
    }

    LottieScalarFrame<TextDocument>& nextFrame()
    {
        return (*frames)[frames->count];
    }

    TextDocument& operator()(float frameNo)
    {
        if (!frames) return value;
        if (frames->count == 1 || frameNo <= frames->first().no) return frames->first().value;
        if (frameNo >= frames->last().no) return frames->last().value;

        auto frame = frames->data + bsearch(frames, frameNo);
        return frame->value;
    }

    LottieTextDoc& operator=(const LottieTextDoc& other)
    {
        //shallow copy, used for slot overriding
        if (other.frames) {
            frames = other.frames;
            const_cast<LottieTextDoc&>(other).frames = nullptr;
        } else {
            value = other.value;
            const_cast<LottieTextDoc&>(other).value.text = nullptr;
            const_cast<LottieTextDoc&>(other).value.name = nullptr;
        }
        return *this;
    }

    void prepare() {}
};


using LottiePoint = LottieGenericProperty<Point>;
using LottieFloat = LottieGenericProperty<float>;
using LottieOpacity = LottieGenericProperty<uint8_t>;
using LottieColor = LottieGenericProperty<RGB24>;

#endif //_TVG_LOTTIE_PROPERTY_H_
