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

#include <algorithm>
#include "tvgCommon.h"
#include "tvgArray.h"
#include "tvgMath.h"
#include "tvgLines.h"
#include "tvgLottieInterpolator.h"
#include "tvgLottieExpressions.h"

#define ROUNDNESS_EPSILON 1.0f

struct LottieFont;
struct LottieLayer;
struct LottieObject;


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
    float tracking = 0.0f;
    uint8_t justify;
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
    return {(int32_t)nearbyint(lhs.rgb[0] * rhs), (int32_t)nearbyint(lhs.rgb[1] * rhs), (int32_t)nearbyint(lhs.rgb[2] * rhs)};
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
        if (!hasTangent) {
            Point dp = next->value - value;
            return mathRad2Deg(mathAtan2(dp.y, dp.x));
        }

        auto t = (frameNo - no) / (next->no - no);
        if (interpolator) t = interpolator->progress(t);
        Bezier bz = {value, value + outTangent, next->value + inTangent, next->value};
        t = bezAtApprox(bz, t * length, length);
        return bezAngleAt(bz, t >= 1.0f ? 0.99f : (t <= 0.0f ? 0.01f : t));
    }

    void prepare(LottieVectorFrame* next)
    {
        Bezier bz = {value, value + outTangent, next->value + inTangent, next->value};
        length = bezLengthApprox(bz);
    }
};


//Property would have an either keyframes or single value.
struct LottieProperty
{
    enum class Type : uint8_t { Point = 0, Float, Opacity, Color, PathSet, ColorStop, Position, TextDoc, Invalid };
    virtual ~LottieProperty() {}

    LottieExpression* exp = nullptr;

    //TODO: Apply common bodies?
    virtual uint32_t frameCnt() = 0;
    virtual uint32_t nearest(float time) = 0;
    virtual float frameNo(int32_t key) = 0;
};


struct LottieExpression
{
    enum LoopMode : uint8_t { None = 0, InCycle = 1, InPingPong, InOffset, InContinue, OutCycle, OutPingPong, OutOffset, OutContinue };

    char* code;
    LottieComposition* comp;
    LottieLayer* layer;
    LottieObject* object;
    LottieProperty* property;
    LottieProperty::Type type;

    bool enabled;

    struct {
        uint32_t key = 0;      //the keyframe number repeating to
        float in = FLT_MAX;    //looping duration in frame number
        LoopMode mode = None;
    } loop;
;
    ~LottieExpression()
    {
        free(code);
    }
};


static void _copy(PathSet* pathset, Array<Point>& outPts, Matrix* transform)
{
    Array<Point> inPts;

    if (transform) {
        for (int i = 0; i < pathset->ptsCnt; ++i) {
            Point pt = pathset->pts[i];
            pt *= *transform;
            outPts.push(pt);
        }
    } else {
        inPts.data = pathset->pts;
        inPts.count = pathset->ptsCnt;
        outPts.push(inPts);
        inPts.data = nullptr;
    }
}


static void _copy(PathSet* pathset, Array<PathCommand>& outCmds)
{
    Array<PathCommand> inCmds;
    inCmds.data = pathset->cmds;
    inCmds.count = pathset->cmdsCnt;
    outCmds.push(inCmds);
    inCmds.data = nullptr;
}


static void _roundCorner(Array<PathCommand>& cmds, Array<Point>& pts, const Point& prev, const Point& curr, const Point& next, float roundness)
{
    auto lenPrev = mathLength(prev - curr);
    auto rPrev = lenPrev > 0.0f ? 0.5f * mathMin(lenPrev * 0.5f, roundness) / lenPrev : 0.0f;
    auto lenNext = mathLength(next - curr);
    auto rNext = lenNext > 0.0f ? 0.5f * mathMin(lenNext * 0.5f, roundness) / lenNext : 0.0f;

    auto dPrev = rPrev * (curr - prev);
    auto dNext = rNext * (curr - next);

    pts.push(curr - 2.0f * dPrev);
    pts.push(curr - dPrev);
    pts.push(curr - dNext);
    pts.push(curr - 2.0f * dNext);
    cmds.push(PathCommand::LineTo);
    cmds.push(PathCommand::CubicTo);
}


static bool _modifier(Point* inPts, uint32_t inPtsCnt, PathCommand* inCmds, uint32_t inCmdsCnt, Array<PathCommand>& cmds, Array<Point>& pts, Matrix* transform, float roundness)
{
    cmds.reserve(inCmdsCnt * 2);
    pts.reserve((uint16_t)(inPtsCnt * 1.5));
    auto ptsCnt = pts.count;

    auto startIndex = 0;
    for (uint32_t iCmds = 0, iPts = 0; iCmds < inCmdsCnt; ++iCmds) {
        switch (inCmds[iCmds]) {
            case PathCommand::MoveTo: {
                startIndex = pts.count;
                cmds.push(PathCommand::MoveTo);
                pts.push(inPts[iPts++]);
                break;
            }
            case PathCommand::CubicTo: {
                auto& prev = inPts[iPts - 1];
                auto& curr = inPts[iPts + 2];
                if (iCmds < inCmdsCnt - 1 &&
                    mathZero(inPts[iPts - 1] - inPts[iPts]) &&
                    mathZero(inPts[iPts + 1] - inPts[iPts + 2])) {
                    if (inCmds[iCmds + 1] == PathCommand::CubicTo &&
                        mathZero(inPts[iPts + 2] - inPts[iPts + 3]) &&
                        mathZero(inPts[iPts + 4] - inPts[iPts + 5])) {
                        _roundCorner(cmds, pts, prev, curr, inPts[iPts + 5], roundness);
                        iPts += 3;
                        break;
                    } else if (inCmds[iCmds + 1] == PathCommand::Close) {
                        _roundCorner(cmds, pts, prev, curr, inPts[2], roundness);
                        pts[startIndex] = pts.last();
                        iPts += 3;
                        break;
                    }
                }
                cmds.push(PathCommand::CubicTo);
                pts.push(inPts[iPts++]);
                pts.push(inPts[iPts++]);
                pts.push(inPts[iPts++]);
                break;
            }
            case PathCommand::Close: {
                cmds.push(PathCommand::Close);
                break;
            }
            default: break;
        }
    }
    if (transform) {
        for (auto i = ptsCnt; i < pts.count; ++i) {
            pts[i] *= *transform;
        }
    }
    return true;
}


template<typename T>
uint32_t _bsearch(T* frames, float frameNo)
{
    int32_t low = 0;
    int32_t high = int32_t(frames->count) - 1;

    while (low <= high) {
        auto mid = low + (high - low) / 2;
        auto frame = frames->data + mid;
        if (frameNo < frame->no) high = mid - 1;
        else low = mid + 1;
    }
    if (high < low) low = high;
    if (low < 0) low = 0;
    return low;
}


template<typename T>
uint32_t _nearest(T* frames, float frameNo)
{
    if (frames) {
        auto key = _bsearch(frames, frameNo);
        if (key == frames->count - 1) return key;
        return (fabsf(frames->data[key].no - frameNo) < fabsf(frames->data[key + 1].no - frameNo)) ? key : (key + 1);
    }
    return 0;
}


template<typename T>
float _frameNo(T* frames, int32_t key)
{
    if (!frames) return 0.0f;
    if (key < 0) key = 0;
    if (key >= (int32_t) frames->count) key = (int32_t)(frames->count - 1);
    return (*frames)[key].no;
}


template<typename T>
float _loop(T* frames, float frameNo, LottieExpression* exp)
{
    if (frameNo >= exp->loop.in || frameNo < frames->first().no ||frameNo < frames->last().no) return frameNo;

    switch (exp->loop.mode) {
        case LottieExpression::LoopMode::InCycle: {
            frameNo -= frames->first().no;
            return fmodf(frameNo, frames->last().no - frames->first().no) + (*frames)[exp->loop.key].no;
        }
        case LottieExpression::LoopMode::OutCycle: {
            frameNo -= frames->first().no;
            return fmodf(frameNo, (*frames)[frames->count - 1 - exp->loop.key].no - frames->first().no) + frames->first().no;
        }
        default: break;
    }
    return frameNo;
}


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
        frames = nullptr;
        if (exp) {
            delete(exp);
            exp = nullptr;
        }
    }

    uint32_t nearest(float frameNo) override
    {
        return _nearest(frames, frameNo);
    }

    uint32_t frameCnt() override
    {
        return frames ? frames->count : 1;
    }

    float frameNo(int32_t key) override
    {
        return _frameNo(frames, key);
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

        auto frame = frames->data + _bsearch(frames, frameNo);
        if (frame->no == frameNo) return frame->value;
        return frame->interpolate(frame + 1, frameNo);
    }

    T operator()(float frameNo, LottieExpressions* exps)
    {
        if (exps && (exp && exp->enabled)) {
            T out{};
            if (exp->loop.mode != LottieExpression::LoopMode::None) frameNo = _loop(frames, frameNo, exp);
            if (exps->result<LottieGenericProperty<T>>(frameNo, out, exp)) return out;
        }
        return operator()(frameNo);
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
        if (exp) {
            delete(exp);
            exp = nullptr;
        }

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

    uint32_t nearest(float frameNo) override
    {
        return _nearest(frames, frameNo);
    }

    uint32_t frameCnt() override
    {
        return frames ? frames->count : 1;
    }

    float frameNo(int32_t key) override
    {
        return _frameNo(frames, key);
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

    bool operator()(float frameNo, Array<PathCommand>& cmds, Array<Point>& pts, Matrix* transform, float roundness)
    {
        PathSet* path = nullptr;
        LottieScalarFrame<PathSet>* frame = nullptr;
        float t;
        bool interpolate = false;

        if (!frames) path = &value;
        else if (frames->count == 1 || frameNo <= frames->first().no) path = &frames->first().value;
        else if (frameNo >= frames->last().no) path = &frames->last().value;
        else {
            frame = frames->data + _bsearch(frames, frameNo);
            if (mathEqual(frame->no, frameNo)) path = &frame->value;
            else if (frame->value.ptsCnt != (frame + 1)->value.ptsCnt) {
                path = &frame->value;
                TVGLOG("LOTTIE", "Different numbers of points in consecutive frames - interpolation omitted.");
            } else {
                t = (frameNo - frame->no) / ((frame + 1)->no - frame->no);
                if (frame->interpolator) t = frame->interpolator->progress(t);
                if (frame->hold) path = &(frame + ((t < 1.0f) ? 0 : 1))->value;
                else interpolate = true;
            }
        }

        if (!interpolate) {
            if (roundness > ROUNDNESS_EPSILON) return _modifier(path->pts, path->ptsCnt, path->cmds, path->cmdsCnt, cmds, pts, transform, roundness);
            _copy(path, cmds);
            _copy(path, pts, transform);
            return true;
        }

        auto s = frame->value.pts;
        auto e = (frame + 1)->value.pts;

        if (roundness > ROUNDNESS_EPSILON) {
            auto interpPts = (Point*)malloc(frame->value.ptsCnt * sizeof(Point));
            auto p = interpPts;
            for (auto i = 0; i < frame->value.ptsCnt; ++i, ++s, ++e, ++p) {
                *p = mathLerp(*s, *e, t);
                if (transform) *p *= *transform;
            }
            _modifier(interpPts, frame->value.ptsCnt, frame->value.cmds, frame->value.cmdsCnt, cmds, pts, nullptr, roundness);
            free(interpPts);
            return true;
        } else {
            for (auto i = 0; i < frame->value.ptsCnt; ++i, ++s, ++e) {
                auto pt = mathLerp(*s, *e, t);
                if (transform) pt *= *transform;
                pts.push(pt);
            }
            _copy(&frame->value, cmds);
        }
        return true;
    }


    bool operator()(float frameNo, Array<PathCommand>& cmds, Array<Point>& pts, Matrix* transform, float roundness, LottieExpressions* exps)
    {
        if (exps && (exp && exp->enabled)) {
            if (exp->loop.mode != LottieExpression::LoopMode::None) frameNo = _loop(frames, frameNo, exp);
            if (exps->result<LottiePathSet>(frameNo, cmds, pts, transform, roundness, exp)) return true;
        }
        return operator()(frameNo, cmds, pts, transform, roundness);
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
        if (exp) {
            delete(exp);
            exp = nullptr;
        }

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

    uint32_t nearest(float frameNo) override
    {
        return _nearest(frames, frameNo);
    }

    uint32_t frameCnt() override
    {
        return frames ? frames->count : 1;
    }

    float frameNo(int32_t key) override
    {
        return _frameNo(frames, key);
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

    Result operator()(float frameNo, Fill* fill, LottieExpressions* exps)
    {
        if (exps && (exp && exp->enabled)) {
            if (exp->loop.mode != LottieExpression::LoopMode::None) frameNo = _loop(frames, frameNo, exp);
            if (exps->result<LottieColorStop>(frameNo, fill, exp)) return Result::Success;
        }

        if (!frames) return fill->colorStops(value.data, count);

        if (frames->count == 1 || frameNo <= frames->first().no) {
            return fill->colorStops(frames->first().value.data, count);
        }

        if (frameNo >= frames->last().no) {
            return fill->colorStops(frames->last().value.data, count);
        }

        auto frame = frames->data + _bsearch(frames, frameNo);
        if (frame->no == frameNo) {
            return fill->colorStops(frame->value.data, count);
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
        return fill->colorStops(result.data, count);
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
        frames = nullptr;

        if (exp) {
            delete(exp);
            exp = nullptr;
        }
    }

    uint32_t nearest(float frameNo) override
    {
        return _nearest(frames, frameNo);
    }

    uint32_t frameCnt() override
    {
        return frames ? frames->count : 1;
    }

    float frameNo(int32_t key) override
    {
        return _frameNo(frames, key);
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

        auto frame = frames->data + _bsearch(frames, frameNo);
        if (frame->no == frameNo) return frame->value;
        return frame->interpolate(frame + 1, frameNo);
    }

    Point operator()(float frameNo, LottieExpressions* exps)
    {
        Point out{};
        if (exps && (exp && exp->enabled)) {
            if (exp->loop.mode != LottieExpression::LoopMode::None) frameNo = _loop(frames, frameNo, exp);
            if (exps->result<LottiePosition>(frameNo, out, exp)) return out;
        }
        return operator()(frameNo);
    }

    float angle(float frameNo)
    {
        if (!frames || frames->count == 1) return 0;

        if (frameNo <= frames->first().no) return frames->first().angle(frames->data + 1, frames->first().no);
        if (frameNo >= frames->last().no) {
            auto frame = frames->data + frames->count - 2;
            return frame->angle(frame + 1, frames->last().no);
        }

        auto frame = frames->data + _bsearch(frames, frameNo);
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
        if (exp) {
            delete(exp);
            exp = nullptr;
        }

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

    uint32_t nearest(float frameNo) override
    {
        return _nearest(frames, frameNo);
    }

    uint32_t frameCnt() override
    {
        return frames ? frames->count : 1;
    }

    float frameNo(int32_t key) override
    {
        return _frameNo(frames, key);
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

        auto frame = frames->data + _bsearch(frames, frameNo);
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
