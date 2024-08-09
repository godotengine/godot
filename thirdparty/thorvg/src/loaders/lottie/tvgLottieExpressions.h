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

#ifndef _TVG_LOTTIE_EXPRESSIONS_H_
#define _TVG_LOTTIE_EXPRESSIONS_H_

#include "tvgCommon.h"

struct LottieExpression;
struct LottieComposition;
struct RGB24;

#ifdef THORVG_LOTTIE_EXPRESSIONS_SUPPORT

#include "jerryscript.h"


struct LottieExpressions
{
public:
    template<typename Property, typename NumType>
    bool result(float frameNo, NumType& out, LottieExpression* exp)
    {
        auto bm_rt = evaluate(frameNo, exp);

        if (auto prop = static_cast<Property*>(jerry_object_get_native_ptr(bm_rt, nullptr))) {
            out = (*prop)(frameNo);
        } else if (jerry_value_is_number(bm_rt)) {
            out = (NumType) jerry_value_as_number(bm_rt);
        } else {
            TVGERR("LOTTIE", "Failed dispatching a Value!");
            return false;
        }
        jerry_value_free(bm_rt);
        return true;
    }

    template<typename Property>
    bool result(float frameNo, Point& out, LottieExpression* exp)
    {
        auto bm_rt = evaluate(frameNo, exp);

        if (jerry_value_is_object(bm_rt)) {
            if (auto prop = static_cast<Property*>(jerry_object_get_native_ptr(bm_rt, nullptr))) {
                out = (*prop)(frameNo);
            } else {
                auto x = jerry_object_get_index(bm_rt, 0);
                auto y = jerry_object_get_index(bm_rt, 1);
                out.x = jerry_value_as_number(x);
                out.y = jerry_value_as_number(y);
                jerry_value_free(x);
                jerry_value_free(y);
            }
        } else {
            TVGERR("LOTTIE", "Failed dispatching Point!");
            return false;
        }
        jerry_value_free(bm_rt);
        return true;
    }

    template<typename Property>
    bool result(float frameNo, RGB24& out, LottieExpression* exp)
    {
        auto bm_rt = evaluate(frameNo, exp);

        if (auto color = static_cast<Property*>(jerry_object_get_native_ptr(bm_rt, nullptr))) {
            out = (*color)(frameNo);
        } else {
            TVGERR("LOTTIE", "Failed dispatching Color!");
            return false;
        }
        jerry_value_free(bm_rt);
        return true;
    }

    template<typename Property>
    bool result(float frameNo, Fill* fill, LottieExpression* exp)
    {
        auto bm_rt = evaluate(frameNo, exp);

        if (auto colorStop = static_cast<Property*>(jerry_object_get_native_ptr(bm_rt, nullptr))) {
            (*colorStop)(frameNo, fill, this);
        } else {
            TVGERR("LOTTIE", "Failed dispatching ColorStop!");
            return false;
        }
        jerry_value_free(bm_rt);
        return true;
    }

    template<typename Property>
    bool result(float frameNo, Array<PathCommand>& cmds, Array<Point>& pts, Matrix* transform, float roundness, LottieExpression* exp)
    {
        auto bm_rt = evaluate(frameNo, exp);

        if (auto pathset = static_cast<Property*>(jerry_object_get_native_ptr(bm_rt, nullptr))) {
            (*pathset)(frameNo, cmds, pts, transform, roundness);
         } else {
            TVGERR("LOTTIE", "Failed dispatching PathSet!");
            return false;
         }
        jerry_value_free(bm_rt);
        return true;
    }

    void update(float curTime);

    //singleton (no thread safety)
    static LottieExpressions* instance();
    static void retrieve(LottieExpressions* instance);

private:
    LottieExpressions();
    ~LottieExpressions();

    jerry_value_t evaluate(float frameNo, LottieExpression* exp);
    jerry_value_t buildGlobal();
    void buildComp(LottieComposition* comp);

    //global object, attributes, methods
    jerry_value_t global;
    jerry_value_t comp;
    jerry_value_t layer;
    jerry_value_t thisComp;
    jerry_value_t thisLayer;
    jerry_value_t thisProperty;
};

#else

struct LottieExpressions
{
    template<typename Property, typename NumType> bool result(TVG_UNUSED float, TVG_UNUSED NumType&, TVG_UNUSED LottieExpression*) { return false; }
    template<typename Property> bool result(TVG_UNUSED float, TVG_UNUSED Point&, LottieExpression*) { return false; }
    template<typename Property> bool result(TVG_UNUSED float, TVG_UNUSED RGB24&, TVG_UNUSED LottieExpression*) { return false; }
    template<typename Property> bool result(TVG_UNUSED float, TVG_UNUSED Fill*, TVG_UNUSED LottieExpression*) { return false; }
    template<typename Property> bool result(TVG_UNUSED float, TVG_UNUSED Array<PathCommand>&, TVG_UNUSED Array<Point>&, TVG_UNUSED Matrix* transform, TVG_UNUSED float, TVG_UNUSED LottieExpression*) { return false; }
    void update(TVG_UNUSED float) {}
    static LottieExpressions* instance() { return nullptr; }
    static void retrieve(TVG_UNUSED LottieExpressions* instance) {}
};

#endif //THORVG_LOTTIE_EXPRESSIONS_SUPPORT

#endif //_TVG_LOTTIE_EXPRESSIONS_H_
