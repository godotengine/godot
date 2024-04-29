/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

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

#ifndef _TVG_PAINT_H_
#define _TVG_PAINT_H_

#include "tvgRender.h"


namespace tvg
{
    enum ContextFlag : uint8_t {Invalid = 0, FastTrack = 1};

    struct Iterator
    {
        virtual ~Iterator() {}
        virtual const Paint* next() = 0;
        virtual uint32_t count() = 0;
        virtual void begin() = 0;
    };

    struct StrategyMethod
    {
        virtual ~StrategyMethod() {}

        virtual bool dispose(RenderMethod& renderer) = 0;
        virtual void* update(RenderMethod& renderer, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag pFlag, bool clipper) = 0;   //Return engine data if it has.
        virtual bool render(RenderMethod& renderer) = 0;
        virtual bool bounds(float* x, float* y, float* w, float* h) = 0;
        virtual RenderRegion bounds(RenderMethod& renderer) const = 0;
        virtual Paint* duplicate() = 0;
        virtual Iterator* iterator() = 0;
    };

    struct Composite
    {
        Paint* target;
        Paint* source;
        CompositeMethod method;
    };

    struct Paint::Impl
    {
        StrategyMethod* smethod = nullptr;
        RenderTransform* rTransform = nullptr;
        Composite* compData = nullptr;
        BlendMethod blendMethod = BlendMethod::Normal;              //uint8_t
        uint8_t renderFlag = RenderUpdateFlag::None;
        uint8_t ctxFlag = ContextFlag::Invalid;
        uint8_t id;
        uint8_t opacity = 255;

        ~Impl()
        {
            if (compData) {
                delete(compData->target);
                free(compData);
            }
            delete(smethod);
            delete(rTransform);
        }

        void method(StrategyMethod* method)
        {
            smethod = method;
        }

        bool transform(const Matrix& m)
        {
            if (!rTransform) {
                rTransform = new RenderTransform();
                if (!rTransform) return false;
            }
            rTransform->override(m);
            renderFlag |= RenderUpdateFlag::Transform;

            return true;
        }

        Matrix* transform()
        {
            if (rTransform) {
                rTransform->update();
                return &rTransform->m;
            }
            return nullptr;
        }

        RenderRegion bounds(RenderMethod& renderer) const
        {
            return smethod->bounds(renderer);
        }

        bool dispose(RenderMethod& renderer)
        {
            if (compData) compData->target->pImpl->dispose(renderer);
            return smethod->dispose(renderer);
        }

        Iterator* iterator()
        {
            return smethod->iterator();
        }

        bool composite(Paint* source, Paint* target, CompositeMethod method)
        {
            //Invalid case
            if ((!target && method != CompositeMethod::None) || (target && method == CompositeMethod::None)) return false;

            if (compData) {
                delete(compData->target);
                //Reset scenario
                if (!target && method == CompositeMethod::None) {
                    free(compData);
                    compData = nullptr;
                    return true;
                }
            } else {
                if (!target && method == CompositeMethod::None) return true;
                compData = static_cast<Composite*>(calloc(1, sizeof(Composite)));
            }
            compData->target = target;
            compData->source = source;
            compData->method = method;
            return true;
        }

        bool rotate(float degree);
        bool scale(float factor);
        bool translate(float x, float y);
        bool bounds(float* x, float* y, float* w, float* h, bool transformed);
        RenderData update(RenderMethod& renderer, const RenderTransform* pTransform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag pFlag, bool clipper = false);
        bool render(RenderMethod& renderer);
        Paint* duplicate();
    };


    template<class T>
    struct PaintMethod : StrategyMethod
    {
        T* inst = nullptr;

        PaintMethod(T* _inst) : inst(_inst) {}
        ~PaintMethod() {}

        bool bounds(float* x, float* y, float* w, float* h) override
        {
            return inst->bounds(x, y, w, h);
        }

        RenderRegion bounds(RenderMethod& renderer) const override
        {
            return inst->bounds(renderer);
        }

        bool dispose(RenderMethod& renderer) override
        {
            return inst->dispose(renderer);
        }

        RenderData update(RenderMethod& renderer, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag renderFlag, bool clipper) override
        {
            return inst->update(renderer, transform, clips, opacity, renderFlag, clipper);
        }

        bool render(RenderMethod& renderer) override
        {
            return inst->render(renderer);
        }

        Paint* duplicate() override
        {
            return inst->duplicate();
        }

        Iterator* iterator() override
        {
            return inst->iterator();
        }
    };
}

#endif //_TVG_PAINT_H_
