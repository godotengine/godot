/*
 * Copyright (c) 2024 the ThorVG project. All rights reserved.

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


#include "tvgMath.h"
#include "tvgCompressor.h"
#include "tvgLottieModel.h"
#include "tvgLottieExpressions.h"

#ifdef THORVG_LOTTIE_EXPRESSIONS_SUPPORT

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct ExpContent
{
    LottieObject* obj;
    float frameNo;
};


//reserved expressions speicifiers
static const char* EXP_NAME = "name";
static const char* EXP_CONTENT = "content";
static const char* EXP_WIDTH = "width";
static const char* EXP_HEIGHT = "height";
static const char* EXP_CYCLE = "cycle";
static const char* EXP_PINGPONG = "pingpong";
static const char* EXP_OFFSET = "offset";
static const char* EXP_CONTINUE = "continue";
static const char* EXP_TIME = "time";
static const char* EXP_VALUE = "value";
static const char* EXP_INDEX = "index";
static const char* EXP_EFFECT= "effect";

static LottieExpressions* exps = nullptr;   //singleton instance engine


static void contentFree(void *native_p, struct jerry_object_native_info_t *info_p)
{
    free(native_p);
}

static jerry_object_native_info_t freeCb {contentFree, 0, 0};
static uint32_t engineRefCnt = 0;  //Expressions Engine reference count


static char* _name(jerry_value_t args)
{
    auto arg0 = jerry_value_to_string(args);
    auto len = jerry_string_length(arg0);
    auto name = (jerry_char_t*)malloc(len * sizeof(jerry_char_t) + 1);
    jerry_string_to_buffer(arg0, JERRY_ENCODING_UTF8, name, len);
    name[len] = '\0';
    jerry_value_free(arg0);
    return (char*) name;
}


static unsigned long _idByName(jerry_value_t args)
{
    auto name = _name(args);
    auto id = djb2Encode(name);
    free(name);
    return id;
}


static jerry_value_t _toComp(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    TVGERR("LOTTIE", "toComp is not supported in expressions!");

    return jerry_undefined();
}


static void _buildTransform(jerry_value_t context, LottieTransform* transform)
{
    if (!transform) return;

    auto obj = jerry_object();
    jerry_object_set_sz(context, "transform", obj);

    auto anchorPoint = jerry_object();
    jerry_object_set_native_ptr(anchorPoint, nullptr, &transform->anchor);
    jerry_object_set_sz(obj, "anchorPoint", anchorPoint);
    jerry_value_free(anchorPoint);

    auto position = jerry_object();
    jerry_object_set_native_ptr(position, nullptr, &transform->position);
    jerry_object_set_sz(obj, "position", position);
    jerry_value_free(position);

    auto scale = jerry_object();
    jerry_object_set_native_ptr(scale, nullptr, &transform->scale);
    jerry_object_set_sz(obj, "scale", scale);
    jerry_value_free(scale);

    auto rotation = jerry_object();
    jerry_object_set_native_ptr(rotation, nullptr, &transform->rotation);
    jerry_object_set_sz(obj, "rotation", rotation);
    jerry_value_free(rotation);

    auto opacity = jerry_object();
    jerry_object_set_native_ptr(opacity, nullptr, &transform->opacity);
    jerry_object_set_sz(obj, "opacity", opacity);
    jerry_value_free(opacity);

    jerry_value_free(obj);
}


static void _buildLayer(jerry_value_t context, LottieLayer* layer, LottieComposition* comp)
{
    auto width = jerry_number(layer->w);
    jerry_object_set_sz(context, EXP_WIDTH, width);
    jerry_value_free(width);

    auto height = jerry_number(layer->h);
    jerry_object_set_sz(context, EXP_HEIGHT, height);
    jerry_value_free(height);

    auto index = jerry_number(layer->idx);
    jerry_object_set_sz(context, EXP_INDEX, index);
    jerry_value_free(index);

    auto parent = jerry_object();
    jerry_object_set_native_ptr(parent, nullptr, layer->parent);
    jerry_object_set_sz(context, "parent", parent);
    jerry_value_free(parent);

    auto hasParent = jerry_boolean(layer->parent ? true : false);
    jerry_object_set_sz(context, "hasParent", hasParent);
    jerry_value_free(hasParent);

    auto inPoint = jerry_number(layer->inFrame);
    jerry_object_set_sz(context, "inPoint", inPoint);
    jerry_value_free(inPoint);

    auto outPoint = jerry_number(layer->outFrame);
    jerry_object_set_sz(context, "outPoint", outPoint);
    jerry_value_free(outPoint);

    auto startTime = jerry_number(comp->timeAtFrame(layer->startFrame));
    jerry_object_set_sz(context, "startTime", startTime);
    jerry_value_free(startTime);

    auto hasVideo = jerry_boolean(false);
    jerry_object_set_sz(context, "hasVideo", hasVideo);
    jerry_value_free(hasVideo);

    auto hasAudio = jerry_boolean(false);
    jerry_object_set_sz(context, "hasAudio", hasAudio);
    jerry_value_free(hasAudio);

    //active, #current in the animation range?

    auto enabled = jerry_boolean(!layer->hidden);
    jerry_object_set_sz(context, "enabled", enabled);
    jerry_value_free(enabled);

    auto audioActive = jerry_boolean(false);
    jerry_object_set_sz(context, "audioActive", audioActive);
    jerry_value_free(audioActive);

    //sampleImage(point, radius = [.5, .5], postEffect=true, t=time)

    _buildTransform(context, layer->transform);

    //audioLevels, #the value of the Audio Levels property of the layer in decibels

    auto timeRemap = jerry_object();
    jerry_object_set_native_ptr(timeRemap, nullptr, &layer->timeRemap);
    jerry_object_set_sz(context, "timeRemap", timeRemap);
    jerry_value_free(timeRemap);

    //marker.key(index)
    //marker.key(name)
    //marker.nearestKey(t)
    //marker.numKeys

    auto name = jerry_string_sz(layer->name);
    jerry_object_set_sz(context, EXP_NAME, name);
    jerry_value_free(name);

    auto toComp = jerry_function_external(_toComp);
    jerry_object_set_sz(context, "toComp", toComp);
    jerry_object_set_native_ptr(toComp, nullptr, comp);
    jerry_value_free(toComp);
}


static jerry_value_t _value(float frameNo, LottieExpression* exp)
{
    switch (exp->type) {
        case LottieProperty::Type::Point: {
            auto value = jerry_object();
            auto pos = (*static_cast<LottiePoint*>(exp->property))(frameNo);
            auto val1 = jerry_number(pos.x);
            auto val2 = jerry_number(pos.y);
            jerry_object_set_index(value, 0, val1);
            jerry_object_set_index(value, 1, val2);
            jerry_value_free(val1);
            jerry_value_free(val2);
            return value;
        }
        case LottieProperty::Type::Float: {
            return jerry_number((*static_cast<LottieFloat*>(exp->property))(frameNo));
        }
        case LottieProperty::Type::Opacity: {
            return jerry_number((*static_cast<LottieOpacity*>(exp->property))(frameNo));
        }
        case LottieProperty::Type::PathSet: {
            auto value = jerry_object();
            jerry_object_set_native_ptr(value, nullptr, exp->property);
            return value;
        }
        case LottieProperty::Type::Position: {
            auto value = jerry_object();
            auto pos = (*static_cast<LottiePosition*>(exp->property))(frameNo);
            auto val1 = jerry_number(pos.x);
            auto val2 = jerry_number(pos.y);
            jerry_object_set_index(value, 0, val1);
            jerry_object_set_index(value, 1, val2);
            jerry_value_free(val1);
            jerry_value_free(val2);
            return value;
        }
        default: {
            TVGERR("LOTTIE", "Non supported type for value? = %d", (int) exp->type);
        }
    }
    return jerry_undefined();
}


static jerry_value_t _addsub(const jerry_value_t args[], float addsub)
{
    //1d
    if (jerry_value_is_number(args[0])) return jerry_number(jerry_value_as_number(args[0]) + addsub * jerry_value_as_number(args[1]));

    //2d
    auto val1 = jerry_object_get_index(args[0], 0);
    auto val2 = jerry_object_get_index(args[0], 1);
    auto val3 = jerry_object_get_index(args[1], 0);
    auto val4 = jerry_object_get_index(args[1], 1);
    auto x = jerry_value_as_number(val1) + addsub * jerry_value_as_number(val3);
    auto y = jerry_value_as_number(val2) + addsub * jerry_value_as_number(val4);

    jerry_value_free(val1);
    jerry_value_free(val2);
    jerry_value_free(val3);
    jerry_value_free(val4);

    auto obj = jerry_object();
    val1 = jerry_number(x);
    val2 = jerry_number(y);
    jerry_object_set_index(obj, 0, val1);
    jerry_object_set_index(obj, 1, val2);
    jerry_value_free(val1);
    jerry_value_free(val2);

    return obj;
}


static jerry_value_t _muldiv(const jerry_value_t arg1, float arg2)
{
    //1d
    if (jerry_value_is_number(arg1)) return jerry_number(jerry_value_as_number(arg1) * arg2);

    //2d
    auto val1 = jerry_object_get_index(arg1, 0);
    auto val2 = jerry_object_get_index(arg1, 1);
    auto x = jerry_value_as_number(val1) * arg2;
    auto y = jerry_value_as_number(val2) * arg2;

    jerry_value_free(val1);
    jerry_value_free(val2);

    auto obj = jerry_object();
    val1 = jerry_number(x);
    val2 = jerry_number(y);
    jerry_object_set_index(obj, 0, val1);
    jerry_object_set_index(obj, 1, val2);
    jerry_value_free(val1);
    jerry_value_free(val2);

    return obj;
}


static jerry_value_t _add(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    return _addsub(args, 1.0f);
}


static jerry_value_t _sub(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    return _addsub(args, -1.0f);
}


static jerry_value_t _mul(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    return _muldiv(args[0], jerry_value_as_number(args[1]));
}


static jerry_value_t _div(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    return _muldiv(args[0], 1.0f / jerry_value_as_number(args[1]));
}


static jerry_value_t _interp(float t, const jerry_value_t args[], int argsCnt)
{
    auto tMin = 0.0f;
    auto tMax = 1.0f;
    int idx = 0;

    if (argsCnt > 3) {
        tMin = jerry_value_as_number(args[1]);
        tMax = jerry_value_as_number(args[2]);
        idx += 2;
    }

    //2d
    if (jerry_value_is_object(args[idx + 1]) && jerry_value_is_object(args[idx + 2])) {
        auto val1 = jerry_object_get_index(args[0], 0);
        auto val2 = jerry_object_get_index(args[0], 1);
        auto val3 = jerry_object_get_index(args[1], 0);
        auto val4 = jerry_object_get_index(args[1], 1);

        Point pt1 = {(float)jerry_value_as_number(val1),  (float)jerry_value_as_number(val2)};
        Point pt2 = {(float)jerry_value_as_number(val3),  (float)jerry_value_as_number(val4)};
        Point ret;
        if (t <= tMin) ret = pt1;
        else if (t >= tMax) ret = pt2;
        else ret = mathLerp(pt1, pt2, t);

        jerry_value_free(val1);
        jerry_value_free(val2);
        jerry_value_free(val3);
        jerry_value_free(val4);

        auto obj = jerry_object();
        val1 = jerry_number(ret.x);
        val2 = jerry_number(ret.y);
        jerry_object_set_index(obj, 0, val1);
        jerry_object_set_index(obj, 1, val2);
        jerry_value_free(val1);
        jerry_value_free(val2);

        return obj;
    }

    //1d
    auto val1 = (float) jerry_value_as_number(args[idx + 1]);
    if (t <= tMin) jerry_number(val1);
    auto val2 = (float) jerry_value_as_number(args[idx + 2]);
    if (t >= tMax) jerry_number(val2);
    return jerry_number(mathLerp(val1, val2, t));
}


static jerry_value_t _linear(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto t = (float) jerry_value_as_number(args[0]);
    return _interp(t, args, jerry_value_as_uint32(argsCnt));
}


static jerry_value_t _ease(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto t = (float) jerry_value_as_number(args[0]);
    t = (t < 0.5) ? (4 * t * t * t) : (1.0f - pow(-2.0f * t + 2.0f, 3) * 0.5f);
    return _interp(t, args, jerry_value_as_uint32(argsCnt));
}



static jerry_value_t _easeIn(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto t = (float) jerry_value_as_number(args[0]);
    t = t * t * t;
    return _interp(t, args, jerry_value_as_uint32(argsCnt));
}


static jerry_value_t _easeOut(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto t = (float) jerry_value_as_number(args[0]);
    t = 1.0f - pow(1.0f - t, 3);
    return _interp(t, args, jerry_value_as_uint32(argsCnt));
}


static jerry_value_t _clamp(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto num = jerry_value_as_number(args[0]);
    auto limit1 = jerry_value_as_number(args[1]);
    auto limit2 = jerry_value_as_number(args[2]);

    //clamping
    if (num < limit1) num = limit1;
    if (num > limit2) num = limit2;

    return jerry_number(num);
}


static jerry_value_t _dot(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto val1 = jerry_object_get_index(args[0], 0);
    auto val2 = jerry_object_get_index(args[0], 1);
    auto val3 = jerry_object_get_index(args[1], 0);
    auto val4 = jerry_object_get_index(args[1], 1);

    auto x = jerry_value_as_number(val1) * jerry_value_as_number(val3);
    auto y = jerry_value_as_number(val2) * jerry_value_as_number(val4);

    jerry_value_free(val1);
    jerry_value_free(val2);
    jerry_value_free(val3);
    jerry_value_free(val4);

    return jerry_number(x + y);
}


static jerry_value_t _cross(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto val1 = jerry_object_get_index(args[0], 0);
    auto val2 = jerry_object_get_index(args[0], 1);
    auto val3 = jerry_object_get_index(args[1], 0);
    auto val4 = jerry_object_get_index(args[1], 1);

    auto x = jerry_value_as_number(val1) * jerry_value_as_number(val4);
    auto y = jerry_value_as_number(val2) * jerry_value_as_number(val3);

    jerry_value_free(val1);
    jerry_value_free(val2);
    jerry_value_free(val3);
    jerry_value_free(val4);

    return jerry_number(x - y);
}


static jerry_value_t _normalize(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto val1 = jerry_object_get_index(args[0], 0);
    auto val2 = jerry_object_get_index(args[0], 1);
    auto x = jerry_value_as_number(val1);
    auto y = jerry_value_as_number(val2);

    jerry_value_free(val1);
    jerry_value_free(val2);

    auto length = sqrtf(x * x + y * y);

    x /= length;
    y /= length;

    auto obj = jerry_object();
    val1 = jerry_number(x);
    val2 = jerry_number(y);
    jerry_object_set_index(obj, 0, val1);
    jerry_object_set_index(obj, 0, val2);
    jerry_value_free(val1);
    jerry_value_free(val2);

    return obj;
}


static jerry_value_t _length(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto val1 = jerry_object_get_index(args[0], 0);
    auto val2 = jerry_object_get_index(args[0], 1);
    auto x = jerry_value_as_number(val1);
    auto y = jerry_value_as_number(val2);

    jerry_value_free(val1);
    jerry_value_free(val2);

    return jerry_number(sqrtf(x * x + y * y));
}


static jerry_value_t _random(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto val = (float)(rand() % 10000001);
    return jerry_number(val * 0.0000001f);
}


static jerry_value_t _deg2rad(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    return jerry_number(mathDeg2Rad((float)jerry_value_as_number(args[0])));
}


static jerry_value_t _rad2deg(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    return jerry_number(mathRad2Deg((float)jerry_value_as_number(args[0])));
}


static jerry_value_t _effect(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    TVGERR("LOTTIE", "effect is not supported in expressions!");

    return jerry_undefined();
}


static jerry_value_t _fromCompToSurface(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    TVGERR("LOTTIE", "fromCompToSurface is not supported in expressions!");

    return jerry_undefined();
}


static jerry_value_t _content(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto data = static_cast<ExpContent*>(jerry_object_get_native_ptr(info->function, &freeCb));
    auto group = static_cast<LottieGroup*>(data->obj);
    auto target = group->content(_idByName(args[0]));
    if (!target) return jerry_undefined();

    //find the a path property(sh) in the group layer?
    switch (target->type) {
        case LottieObject::Group: {
            auto group = static_cast<LottieGroup*>(target);
            auto obj = jerry_function_external(_content);

            //attach a transform
            for (auto c = group->children.begin(); c < group->children.end(); ++c) {
                if ((*c)->type == LottieObject::Type::Transform) {
                    _buildTransform(obj, static_cast<LottieTransform*>(*c));
                    break;
                }
            }
            auto data2 = (ExpContent*)malloc(sizeof(ExpContent));
            data2->obj = group;
            data2->frameNo = data->frameNo;
            jerry_object_set_native_ptr(obj, &freeCb, data2);
            jerry_object_set_sz(obj, EXP_CONTENT, obj);
            return obj;
        }
        case LottieObject::Path: {
            jerry_value_t obj = jerry_object();
            jerry_object_set_native_ptr(obj, nullptr, &static_cast<LottiePath*>(target)->pathset);
            jerry_object_set_sz(obj, "path", obj);
            return obj;
        }
        case LottieObject::Trimpath: {
            auto trimpath = static_cast<LottieTrimpath*>(target);
            jerry_value_t obj = jerry_object();
            auto start = jerry_number(trimpath->start(data->frameNo));
            jerry_object_set_sz(obj, "start", start);
            jerry_value_free(start);
            auto end = jerry_number(trimpath->end(data->frameNo));
            jerry_object_set_sz(obj, "end", end);
            jerry_value_free(end);
            auto offset = jerry_number(trimpath->offset(data->frameNo));
            jerry_object_set_sz(obj, "offset", end);
            jerry_value_free(offset);
            return obj;
        }
        default: break;
    }
    return jerry_undefined();
}


static jerry_value_t _layer(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto comp = static_cast<LottieComposition*>(jerry_object_get_native_ptr(info->function, nullptr));
    LottieLayer* layer;

    //layer index
    if (jerry_value_is_number(args[0])) {
        auto idx = (uint16_t)jerry_value_as_int32(args[0]);
        layer = comp->layerByIdx(idx);
        jerry_value_free(idx);
    //layer name
    } else {
        layer = comp->layerById(_idByName(args[0]));
    }

    if (!layer) return jerry_undefined();

    auto obj = jerry_object();
    jerry_object_set_native_ptr(obj, nullptr, layer);
    _buildLayer(obj, layer, comp);

    return obj;
}


static jerry_value_t _nearestKey(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));
    auto time = jerry_value_as_number(args[0]);
    auto frameNo = exp->comp->frameAtTime(time);
    auto index = jerry_number(exp->property->nearest(frameNo));

    auto obj = jerry_object();
    jerry_object_set_sz(obj, EXP_INDEX, index);
    jerry_value_free(index);

    return obj;
}


static jerry_value_t _valueAtTime(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));
    auto time = jerry_value_as_number(args[0]);
    auto frameNo = exp->comp->frameAtTime(time);
    return _value(frameNo, exp);
}


static jerry_value_t _velocityAtTime(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));
    auto time = jerry_value_as_number(args[0]);
    auto frameNo = exp->comp->frameAtTime(time);
    auto key = exp->property->nearest(frameNo);
    auto pframe = exp->property->frameNo(key - 1);
    auto cframe = exp->property->frameNo(key);
    auto elapsed = (cframe - pframe) / (exp->comp->frameRate);

    Point cur, prv;

    //compute the velocity
    switch (exp->type) {
        case LottieProperty::Type::Point: {
            prv = (*static_cast<LottiePoint*>(exp->property))(pframe);
            cur = (*static_cast<LottiePoint*>(exp->property))(cframe);
            break;
        }
        case LottieProperty::Type::Position: {
            prv = (*static_cast<LottiePosition*>(exp->property))(pframe);
            cur = (*static_cast<LottiePosition*>(exp->property))(cframe);
            break;
        }
        default: {
            TVGERR("LOTTIE", "Non supported type for velocityAtTime?");
            return jerry_undefined();
        }
    }

    float velocity[] = {(cur.x - prv.x) / elapsed, (cur.y - prv.y) / elapsed};

    auto obj = jerry_object();
    auto val1 = jerry_number(velocity[0]);
    auto val2 = jerry_number(velocity[1]);
    jerry_object_set_index(obj, 0, val1);
    jerry_object_set_index(obj, 1, val2);
    jerry_value_free(val1);
    jerry_value_free(val2);

    return obj;
}


static jerry_value_t _speedAtTime(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));
    auto time = jerry_value_as_number(args[0]);
    auto frameNo = exp->comp->frameAtTime(time);
    auto key = exp->property->nearest(frameNo);
    auto pframe = exp->property->frameNo(key - 1);
    auto cframe = exp->property->frameNo(key);
    auto elapsed = (cframe - pframe) / (exp->comp->frameRate);

    Point cur, prv;

    //compute the velocity
    switch (exp->type) {
        case LottieProperty::Type::Point: {
            prv = (*static_cast<LottiePoint*>(exp->property))(pframe);
            cur = (*static_cast<LottiePoint*>(exp->property))(cframe);
            break;
        }
        case LottieProperty::Type::Position: {
            prv = (*static_cast<LottiePosition*>(exp->property))(pframe);
            cur = (*static_cast<LottiePosition*>(exp->property))(cframe);
            break;
        }
        default: {
            TVGERR("LOTTIE", "Non supported type for speedAtTime?");
            return jerry_undefined();
        }
    }

    auto speed = sqrtf(pow(cur.x - prv.x, 2) + pow(cur.y - prv.y, 2)) / elapsed;
    auto obj = jerry_number(speed);
    return obj;
}


static bool _loopOutCommon(LottieExpression* exp, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    exp->loop.mode = LottieExpression::LoopMode::OutCycle;

    if (argsCnt > 0) {
        auto name = _name(args[0]);
        if (!strcmp(name, EXP_CYCLE)) exp->loop.mode = LottieExpression::LoopMode::OutCycle;
        else if (!strcmp(name, EXP_PINGPONG)) exp->loop.mode = LottieExpression::LoopMode::OutPingPong;
        else if (!strcmp(name, EXP_OFFSET)) exp->loop.mode = LottieExpression::LoopMode::OutOffset;
        else if (!strcmp(name, EXP_CONTINUE)) exp->loop.mode = LottieExpression::LoopMode::OutContinue;
        free(name);
    }

    if (exp->loop.mode != LottieExpression::LoopMode::OutCycle) {
        TVGERR("hermet", "Not supported loopOut type = %d", exp->loop.mode);
        return false;
    }

    return true;
}


static jerry_value_t _loopOut(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));

    if (!_loopOutCommon(exp, args, argsCnt)) return jerry_undefined();

    if (argsCnt > 1) exp->loop.key = jerry_value_as_int32(args[1]);

    auto obj = jerry_object();
    jerry_object_set_native_ptr(obj, nullptr, exp->property);
    return obj;
}


static jerry_value_t _loopOutDuration(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));

    if (!_loopOutCommon(exp, args, argsCnt)) return jerry_undefined();

    if (argsCnt > 1) {
        exp->loop.in = exp->comp->frameAtTime((float)jerry_value_as_int32(args[1]));
    }

    auto obj = jerry_object();
    jerry_object_set_native_ptr(obj, nullptr, exp->property);
    return obj;
}


static bool _loopInCommon(LottieExpression* exp, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    exp->loop.mode = LottieExpression::LoopMode::InCycle;

    if (argsCnt > 0) {
        auto name = _name(args[0]);
        if (!strcmp(name, EXP_CYCLE)) exp->loop.mode = LottieExpression::LoopMode::InCycle;
        else if (!strcmp(name, EXP_PINGPONG)) exp->loop.mode = LottieExpression::LoopMode::InPingPong;
        else if (!strcmp(name, EXP_OFFSET)) exp->loop.mode = LottieExpression::LoopMode::InOffset;
        else if (!strcmp(name, EXP_CONTINUE)) exp->loop.mode = LottieExpression::LoopMode::InContinue;
        free(name);
    }

    if (exp->loop.mode != LottieExpression::LoopMode::InCycle) {
        TVGERR("hermet", "Not supported loopOut type = %d", exp->loop.mode);
        return false;
    }

    return true;
}

static jerry_value_t _loopIn(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));

    if (!_loopInCommon(exp, args, argsCnt)) return jerry_undefined();

    if (argsCnt > 1) {
        exp->loop.in = exp->comp->frameAtTime((float)jerry_value_as_int32(args[1]));
    }

    auto obj = jerry_object();
    jerry_object_set_native_ptr(obj, nullptr, exp->property);
    return obj;
}


static jerry_value_t _loopInDuration(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));

    if (argsCnt > 1) {
        exp->loop.in = exp->comp->frameAtTime((float)jerry_value_as_int32(args[1]));
    }

    if (!_loopInCommon(exp, args, argsCnt)) return jerry_undefined();

    auto obj = jerry_object();
    jerry_object_set_native_ptr(obj, nullptr, exp->property);
    return obj;
}


static jerry_value_t _key(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto exp = static_cast<LottieExpression*>(jerry_object_get_native_ptr(info->function, nullptr));
    auto key = jerry_value_as_int32(args[0]);
    auto frameNo = exp->property->frameNo(key);
    auto time = jerry_number(exp->comp->timeAtFrame(frameNo));
    auto value = _value(frameNo, exp);

    auto obj = jerry_object();
    jerry_object_set_sz(obj, EXP_TIME, time);
    jerry_object_set_sz(obj, EXP_INDEX, args[0]);
    jerry_object_set_sz(obj, EXP_VALUE, value);

    jerry_value_free(time);
    jerry_value_free(value);

    return obj;
}



static jerry_value_t _createPath(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    //TODO: arg1: points, arg2: inTagents, arg3: outTangents, arg4: isClosed
    auto arg1 = jerry_value_to_object(args[0]);
    auto pathset = jerry_object_get_native_ptr(arg1, nullptr);
    if (!pathset) {
        TVGERR("LOTTIE", "failed createPath()");
        return jerry_undefined();
    }

    jerry_value_free(arg1);

    auto obj = jerry_object();
    jerry_object_set_native_ptr(obj, nullptr, pathset);
    return obj;
}


static jerry_value_t _uniformPath(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto pathset = static_cast<LottiePathSet*>(jerry_object_get_native_ptr(info->function, nullptr));

    /* TODO: ThorVG prebuilds the path data for performance.
       It acutally need to constructs the Array<Point> for points, inTangents, outTangents and then return here... */
    auto obj = jerry_object();
    jerry_object_set_native_ptr(obj, nullptr, pathset);
    return obj;
}


static jerry_value_t _isClosed(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    //TODO: Not used
    return jerry_boolean(true);
}


static void _buildPath(jerry_value_t context, LottieExpression* exp)
{
    //Trick for fast buliding path.
    auto points = jerry_function_external(_uniformPath);
    jerry_object_set_native_ptr(points, nullptr, exp->property);
    jerry_object_set_sz(context, "points", points);
    jerry_value_free(points);

    auto inTangents = jerry_function_external(_uniformPath);
    jerry_object_set_native_ptr(inTangents, nullptr, exp->property);
    jerry_object_set_sz(context, "inTangents", inTangents);
    jerry_value_free(inTangents);

    auto outTangents = jerry_function_external(_uniformPath);
    jerry_object_set_native_ptr(outTangents, nullptr, exp->property);
    jerry_object_set_sz(context, "outTangents", outTangents);
    jerry_value_free(outTangents);

    auto isClosed = jerry_function_external(_isClosed);
    jerry_object_set_native_ptr(isClosed, nullptr, exp->property);
    jerry_object_set_sz(context, "isClosed", isClosed);
    jerry_value_free(isClosed);

}


static void _buildProperty(float frameNo, jerry_value_t context, LottieExpression* exp)
{
    auto value = _value(frameNo, exp);
    jerry_object_set_sz(context, EXP_VALUE, value);
    jerry_value_free(value);

    auto valueAtTime = jerry_function_external(_valueAtTime);
    jerry_object_set_sz(context, "valueAtTime", valueAtTime);
    jerry_object_set_native_ptr(valueAtTime, nullptr, exp);
    jerry_value_free(valueAtTime);

    auto velocity = jerry_number(0.0f);
    jerry_object_set_sz(context, "velocity", velocity);
    jerry_value_free(velocity);

    auto velocityAtTime = jerry_function_external(_velocityAtTime);
    jerry_object_set_sz(context, "velocityAtTime", velocityAtTime);
    jerry_object_set_native_ptr(velocityAtTime, nullptr, exp);
    jerry_value_free(velocityAtTime);

    auto speed = jerry_number(0.0f);
    jerry_object_set_sz(context, "speed", speed);
    jerry_value_free(speed);

    auto speedAtTime = jerry_function_external(_speedAtTime);
    jerry_object_set_sz(context, "speedAtTime", speedAtTime);
    jerry_object_set_native_ptr(speedAtTime, nullptr, exp);
    jerry_value_free(speedAtTime);

    //wiggle(freq, amp, octaves=1, amp_mult=.5, t=time)
    //temporalWiggle(freq, amp, octaves=1, amp_mult=.5, t=time)
    //smooth(width=.2, samples=5, t=time)

    auto loopIn = jerry_function_external(_loopIn);
    jerry_object_set_sz(context, "loopIn", loopIn);
    jerry_object_set_native_ptr(loopIn, nullptr, exp);
    jerry_value_free(loopIn);

    auto loopOut = jerry_function_external(_loopOut);
    jerry_object_set_sz(context, "loopOut", loopOut);
    jerry_object_set_native_ptr(loopOut, nullptr, exp);
    jerry_value_free(loopOut);

    auto loopInDuration = jerry_function_external(_loopInDuration);
    jerry_object_set_sz(context, "loopInDuration", loopInDuration);
    jerry_object_set_native_ptr(loopInDuration, nullptr, exp);
    jerry_value_free(loopInDuration);

    auto loopOutDuration = jerry_function_external(_loopOutDuration);
    jerry_object_set_sz(context, "loopOutDuration", loopOutDuration);
    jerry_object_set_native_ptr(loopOutDuration, nullptr, exp);
    jerry_value_free(loopOutDuration);

    auto key = jerry_function_external(_key);
    jerry_object_set_sz(context, "key", key);
    jerry_object_set_native_ptr(key, nullptr, exp);
    jerry_value_free(key);

    //key(markerName)

    auto nearestKey = jerry_function_external(_nearestKey);
    jerry_object_set_native_ptr(nearestKey, nullptr, exp);
    jerry_object_set_sz(context, "nearestKey", nearestKey);
    jerry_value_free(nearestKey);

    auto numKeys = jerry_number(exp->property->frameCnt());
    jerry_object_set_sz(context, "numKeys", numKeys);
    jerry_value_free(numKeys);

    //propertyGroup(countUp = 1)
    //propertyIndex
    //name

    //content("name"), #look for the named property from a layer
    auto data = (ExpContent*)malloc(sizeof(ExpContent));
    data->obj = exp->layer;
    data->frameNo = frameNo;

    auto content = jerry_function_external(_content);
    jerry_object_set_sz(context, EXP_CONTENT, content);
    jerry_object_set_native_ptr(content, &freeCb, data);
    jerry_value_free(content);
}


static jerry_value_t _comp(const jerry_call_info_t* info, const jerry_value_t args[], const jerry_length_t argsCnt)
{
    auto comp = static_cast<LottieComposition*>(jerry_object_get_native_ptr(info->function, nullptr));
    auto layer = comp->asset(_idByName(args[0]));

    if (!layer) return jerry_undefined();

    auto obj = jerry_object();
    jerry_object_set_native_ptr(obj, nullptr, layer);
    _buildLayer(obj, layer, comp);

    return obj;
}


static void _buildMath(jerry_value_t context)
{
    auto bm_mul = jerry_function_external(_mul);
    jerry_object_set_sz(context, "$bm_mul", bm_mul);
    jerry_value_free(bm_mul);

    auto bm_sum = jerry_function_external(_add);
    jerry_object_set_sz(context, "$bm_sum", bm_sum);
    jerry_value_free(bm_sum);

    auto bm_add = jerry_function_external(_add);
    jerry_object_set_sz(context, "$bm_add", bm_add);
    jerry_value_free(bm_add);

    auto bm_sub = jerry_function_external(_sub);
    jerry_object_set_sz(context, "$bm_sub", bm_sub);
    jerry_value_free(bm_sub);

    auto bm_div = jerry_function_external(_div);
    jerry_object_set_sz(context, "$bm_div", bm_div);
    jerry_value_free(bm_div);

    auto mul = jerry_function_external(_mul);
    jerry_object_set_sz(context, "mul", mul);
    jerry_value_free(mul);

    auto sum = jerry_function_external(_add);
    jerry_object_set_sz(context, "sum", sum);
    jerry_value_free(sum);

    auto add = jerry_function_external(_add);
    jerry_object_set_sz(context, "add", add);
    jerry_value_free(add);

    auto sub = jerry_function_external(_sub);
    jerry_object_set_sz(context, "sub", sub);
    jerry_value_free(sub);

    auto div = jerry_function_external(_div);
    jerry_object_set_sz(context, "div", div);
    jerry_value_free(div);

    auto clamp = jerry_function_external(_clamp);
    jerry_object_set_sz(context, "clamp", clamp);
    jerry_value_free(clamp);

    auto dot = jerry_function_external(_dot);
    jerry_object_set_sz(context, "dot", dot);
    jerry_value_free(dot);

    auto cross = jerry_function_external(_cross);
    jerry_object_set_sz(context, "cross", cross);
    jerry_value_free(cross);

    auto normalize = jerry_function_external(_normalize);
    jerry_object_set_sz(context, "normalize", normalize);
    jerry_value_free(normalize);

    auto length = jerry_function_external(_length);
    jerry_object_set_sz(context, "length", length);
    jerry_value_free(length);

    auto random = jerry_function_external(_random);
    jerry_object_set_sz(context, "random", random);
    jerry_value_free(random);

    auto deg2rad = jerry_function_external(_deg2rad);
    jerry_object_set_sz(context, "degreesToRadians", deg2rad);
    jerry_value_free(deg2rad);

    auto rad2deg = jerry_function_external(_rad2deg);
    jerry_object_set_sz(context, "radiansToDegrees", rad2deg);
    jerry_value_free(rad2deg);

    auto linear = jerry_function_external(_linear);
    jerry_object_set_sz(context, "linear", linear);
    jerry_value_free(linear);

    auto ease = jerry_function_external(_ease);
    jerry_object_set_sz(context, "ease", ease);
    jerry_value_free(ease);

    auto easeIn = jerry_function_external(_easeIn);
    jerry_object_set_sz(context, "easeIn", easeIn);
    jerry_value_free(easeIn);

    auto easeOut = jerry_function_external(_easeOut);
    jerry_object_set_sz(context, "easeOut", easeOut);
    jerry_value_free(easeOut);

    //lookAt
}


void LottieExpressions::buildComp(LottieComposition* comp)
{
    jerry_object_set_native_ptr(this->comp, nullptr, comp);
    jerry_object_set_native_ptr(thisComp, nullptr, comp);
    jerry_object_set_native_ptr(layer, nullptr, comp);

    //marker
    //marker.key(index)
    //marker.key(name)
    //marker.nearestKey(t)
    //marker.numKeys

    auto numLayers = jerry_number(comp->root->children.count);
    jerry_object_set_sz(thisComp, "numLayers", numLayers);
    jerry_value_free(numLayers);

    //activeCamera

    auto width = jerry_number(comp->w);
    jerry_object_set_sz(thisComp, EXP_WIDTH, width);
    jerry_value_free(width);

    auto height = jerry_number(comp->h);
    jerry_object_set_sz(thisComp, EXP_HEIGHT, height);
    jerry_value_free(height);

    auto duration = jerry_number(comp->duration());
    jerry_object_set_sz(thisComp, "duration", duration);
    jerry_value_free(duration);

    //ntscDropFrame
    //displayStartTime

    auto frameDuration = jerry_number(1.0f / comp->frameRate);
    jerry_object_set_sz(thisComp, "frameDuration", frameDuration);
    jerry_value_free(frameDuration);

    //shutterAngle
    //shutterPhase
    //bgColor
    //pixelAspect

    auto name = jerry_string((jerry_char_t*)comp->name, strlen(comp->name), JERRY_ENCODING_UTF8);
    jerry_object_set_sz(thisComp, EXP_NAME, name);
    jerry_value_free(name);
}


jerry_value_t LottieExpressions::buildGlobal()
{
    global = jerry_current_realm();

    //comp(name)
    comp = jerry_function_external(_comp);
    jerry_object_set_sz(global, "comp", comp);

    //footage(name)

    thisComp = jerry_object();
    jerry_object_set_sz(global, "thisComp", thisComp);

    //layer(index) / layer(name) / layer(otherLayer, reIndex)
    layer = jerry_function_external(_layer);
    jerry_object_set_sz(thisComp, "layer", layer);

    thisLayer = jerry_object();
    jerry_object_set_sz(global, "thisLayer", thisLayer);

    thisProperty = jerry_object();
    jerry_object_set_sz(global, "thisProperty", thisProperty);

    auto effect = jerry_function_external(_effect);
    jerry_object_set_sz(global, EXP_EFFECT, effect);
    jerry_value_free(effect);

    auto fromCompToSurface = jerry_function_external(_fromCompToSurface);
    jerry_object_set_sz(global, "fromCompToSurface", fromCompToSurface);
    jerry_value_free(fromCompToSurface);

    auto createPath = jerry_function_external(_createPath);
    jerry_object_set_sz(global, "createPath", createPath);
    jerry_value_free(createPath);

    //posterizeTime(framesPerSecond)
    //value

    return global;
}


jerry_value_t LottieExpressions::evaluate(float frameNo, LottieExpression* exp)
{
    buildComp(exp->comp);

    //update global context values
    jerry_object_set_native_ptr(thisLayer, nullptr, exp->layer);
    _buildLayer(thisLayer, exp->layer, exp->comp);

    jerry_object_set_native_ptr(thisProperty, nullptr, exp->property);
    _buildProperty(frameNo, global, exp);

    if (exp->type == LottieProperty::Type::PathSet) _buildPath(thisProperty, exp);
    if (exp->object->type == LottieObject::Transform) _buildTransform(global, static_cast<LottieTransform*>(exp->object));

    //evaluate the code
    auto eval = jerry_eval((jerry_char_t *) exp->code, strlen(exp->code), JERRY_PARSE_NO_OPTS);

    if (jerry_value_is_exception(eval) || jerry_value_is_undefined(eval)) {
        exp->enabled = false;  // The feature is experimental, it will be forcely turned off if it's incompatible.
        return jerry_undefined();
    }

    jerry_value_free(eval);

    return jerry_object_get_sz(global, "$bm_rt");
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

LottieExpressions::~LottieExpressions()
{
    jerry_value_free(thisProperty);
    jerry_value_free(thisLayer);
    jerry_value_free(layer);
    jerry_value_free(thisComp);
    jerry_value_free(comp);
    jerry_value_free(global);
    jerry_cleanup();
}


LottieExpressions::LottieExpressions()
{
    jerry_init(JERRY_INIT_EMPTY);
    _buildMath(buildGlobal());
}


void LottieExpressions::update(float curTime)
{
    //time, #current time in seconds
    auto time = jerry_number(curTime);
    jerry_object_set_sz(global, EXP_TIME, time);
    jerry_value_free(time);
}


//FIXME: Threads support
#include "tvgTaskScheduler.h"

LottieExpressions* LottieExpressions::instance()
{
    //FIXME: Threads support
    if (TaskScheduler::threads() > 1) {
        TVGLOG("LOTTIE", "Lottie Expressions are not supported with tvg threads");
        return nullptr;
    }

    if (!exps) exps = new LottieExpressions;
    ++engineRefCnt;
    return exps;
}


void LottieExpressions::retrieve(LottieExpressions* instance)
{
    if (--engineRefCnt == 0) {
        delete(instance);
        exps = nullptr;
    }
}


#endif //THORVG_LOTTIE_EXPRESSIONS_SUPPORT
