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

#ifndef _TVG_SCENE_H_
#define _TVG_SCENE_H_

#include <algorithm>
#include "tvgMath.h"
#include "tvgPaint.h"


struct SceneIterator : Iterator
{
    list<Paint*>* paints;
    list<Paint*>::iterator itr;

    SceneIterator(list<Paint*>* p) : paints(p)
    {
        begin();
    }

    const Paint* next() override
    {
        if (itr == paints->end()) return nullptr;
        auto paint = *itr;
        ++itr;
        return paint;
    }

    uint32_t count() override
    {
       return paints->size();
    }

    void begin() override
    {
        itr = paints->begin();
    }
};

struct SceneImpl : Scene
{
    Paint::Impl impl;
    list<Paint*> paints;     //children list
    RenderRegion vport = {};
    Array<RenderEffect*>* effects = nullptr;
    Point fsize;          //fixed scene size
    bool fixed = false;   //true: fixed scene size, false: dynamic size
    bool vdirty = false;
    uint8_t opacity;      //for composition

    SceneImpl() : impl(Paint::Impl(this))
    {
    }

    ~SceneImpl()
    {
        clearPaints();
        resetEffects(false);
    }

    void size(const Point& size)
    {
        this->fsize = size;
        fixed = (size.x > 0 && size.y > 0) ? true : false;
    }

    uint8_t needComposition(uint8_t opacity)
    {
        if (opacity == 0 || paints.empty()) return 0;

        //post effects, masking, blending may require composition
        if (effects) impl.mark(CompositionFlag::PostProcessing);
        if (PAINT(this)->mask(nullptr) != MaskMethod::None) impl.mark(CompositionFlag::Masking);
        if (impl.blendMethod != BlendMethod::Normal) impl.mark(CompositionFlag::Blending);

        //Half translucent requires intermediate composition.
        if (opacity == 255) return impl.cmpFlag;

        //Only shape or picture may not require composition.
        if (paints.size() == 1) {
            auto type = paints.front()->type();
            if (type == Type::Shape || type == Type::Picture) return impl.cmpFlag;
        }

        impl.mark(CompositionFlag::Opacity);

        return 1;
    }

    bool skip(RenderUpdateFlag flag)
    {
        return false;
    }

    bool update(RenderMethod* renderer, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flag, TVG_UNUSED bool clipper)
    {
        if (paints.empty()) return true;

        if (needComposition(opacity)) {
            /* Overriding opacity value. If this scene is half-translucent,
               It must do intermediate composition with that opacity value. */
            this->opacity = opacity;
            opacity = 255;
        }

        //allow partial rendering?
        auto recover = fixed ? renderer->partial(true) : false;

        for (auto paint : paints) {
            PAINT(paint)->update(renderer, transform, clips, opacity, flag, false);
        }

        //recover the condition
        if (fixed) renderer->partial(recover);

        if (effects) {
            ARRAY_FOREACH(p, *effects) {
                renderer->prepare(*p, transform);
            }
        }

        //this viewport update is more performant than in bounds(). No idea.
        vport = renderer->viewport();

        if (fixed) {
            auto pt = fsize * transform;
            vport.intersect({{int32_t(round(transform.e13)), int32_t(round(transform.e23))}, {int32_t(round(pt.x)), int32_t(round(pt.y))}});
        } else {
            vdirty = true;
        }

        //bounds(renderer) here hinders parallelization
        //TODO: we can bring the precise effects region here
        if (fixed || effects) impl.damage(vport);

        return true;
    }

    bool render(RenderMethod* renderer)
    {
        if (paints.empty()) return true;

        RenderCompositor* cmp = nullptr;
        auto ret = true;

        renderer->blend(impl.blendMethod);

        if (impl.cmpFlag) {
            cmp = renderer->target(bounds(), renderer->colorSpace(), impl.cmpFlag);
            renderer->beginComposite(cmp, MaskMethod::None, opacity);
        }

        for (auto paint : paints) {
            ret &= paint->pImpl->render(renderer);
        }

        if (cmp) {
            //Apply post effects if any.
            if (effects) {
                //Notify the possibility of the direct composition of the effect result to the origin surface.
                auto direct = (effects->count == 1) & (impl.marked(CompositionFlag::PostProcessing));
                ARRAY_FOREACH(p, *effects) {
                    if ((*p)->valid) renderer->render(cmp, *p, direct);
                }
            }
            renderer->endComposite(cmp);
        }

        return ret;
    }

    RenderRegion bounds()
    {
        if (paints.empty()) return {};
        if (!vdirty) return vport;
        vdirty = false;

        //Merge regions
        RenderRegion pRegion = {{INT32_MAX, INT32_MAX}, {0, 0}};
        for (auto paint : paints) {
            auto region = paint->pImpl->bounds();
            if (region.min.x < pRegion.min.x) pRegion.min.x = region.min.x;
            if (pRegion.max.x < region.max.x) pRegion.max.x = region.max.x;
            if (region.min.y < pRegion.min.y) pRegion.min.y = region.min.y;
            if (pRegion.max.y < region.max.y) pRegion.max.y = region.max.y;
        }

        //Extends the render region if post effects require
        RenderRegion eRegion{};
        if (effects) {
            ARRAY_FOREACH(p, *effects) {
                auto effect = *p;
                if (effect->valid && impl.renderer->region(effect)) eRegion.add(effect->extend);
            }
        }

        pRegion.min.x += eRegion.min.x;
        pRegion.min.y += eRegion.min.y;
        pRegion.max.x += eRegion.max.x;
        pRegion.max.y += eRegion.max.y;

        vport = RenderRegion::intersect(vport, pRegion);
        return vport;
    }

    bool bounds(Point* pt4, const Matrix& m, bool obb)
    {
        if (paints.empty()) return false;

        Point min = {FLT_MAX, FLT_MAX};
        Point max = {-FLT_MAX, -FLT_MAX};
        auto ret = false;

        for (auto paint : paints) {
            Point tmp[4];
            if (!PAINT(paint)->bounds(tmp, obb ? nullptr : &m, false)) continue;
            //Merge regions
            for (int i = 0; i < 4; ++i) {
                if (tmp[i].x < min.x) min.x = tmp[i].x;
                if (tmp[i].x > max.x) max.x = tmp[i].x;
                if (tmp[i].y < min.y) min.y = tmp[i].y;
                if (tmp[i].y > max.y) max.y = tmp[i].y;
            }
            ret = true;
        }
        pt4[0] = min;
        pt4[1] = Point{max.x, min.y};
        pt4[2] = max;
        pt4[3] = Point{min.x, max.y};

        if (obb) {
            pt4[0] *= m;
            pt4[1] *= m;
            pt4[2] *= m;
            pt4[3] *= m;
        }

        return ret;
    }


    bool intersects(const RenderRegion& region)
    {
        if (!impl.renderer) return false;

        if (this->bounds().intersected(region)) {
            for (auto paint : paints) {
                if (PAINT(paint)->intersects(region)) return true;
            }
        }

        return false;
    }

    Paint* duplicate(Paint* ret)
    {
        if (ret) TVGERR("RENDERER", "TODO: duplicate()");

        auto scene = Scene::gen();
        auto dup = to<SceneImpl>(scene);

        for (auto paint : paints) {
            auto cdup = paint->duplicate();
            PAINT(cdup)->parent = scene;
            cdup->ref();
            dup->paints.push_back(cdup);
        }

        if (effects) {
            dup->effects = new Array<RenderEffect*>;
            ARRAY_FOREACH(p, *effects) {
                RenderEffect* ret = nullptr;
                switch ((*p)->type) {
                    case SceneEffect::GaussianBlur: {
                        ret = new RenderEffectGaussianBlur(*(RenderEffectGaussianBlur*)(*p));
                        break;
                    }
                    case SceneEffect::DropShadow: {
                        ret = new RenderEffectDropShadow(*(RenderEffectDropShadow*)(*p));
                        break;
                    }
                    case SceneEffect::Fill: {
                        ret = new RenderEffectFill(*(RenderEffectFill*)(*p));
                        break;
                    }
                    case SceneEffect::Tint: {
                        ret = new RenderEffectTint(*(RenderEffectTint*)(*p));
                        break;
                    }
                    case SceneEffect::Tritone: {
                        ret = new RenderEffectTritone(*(RenderEffectTritone*)(*p));
                        break;
                    }
                    default: break;
                }
                if (ret) {
                    ret->rd = nullptr;
                    ret->valid = false;
                    dup->effects->push(ret);
                }
            }
        }

        if (fixed) dup->size(fsize);

        return scene;
    }

    Result clearPaints()
    {
        if (paints.empty()) return Result::Success;

        //Don't need to damage for children
        auto recover = (fixed && impl.renderer) ? impl.renderer->partial(true) : false;
        auto partialDmg = !(effects || fixed || recover);

        auto itr = paints.begin();
        while (itr != paints.end()) {
            auto paint = PAINT((*itr));
            //when the paint is destroyed damage will be triggered
            if (paint->refCnt > 1 && partialDmg) paint->damage();
            paint->unref();
            paints.erase(itr++);
        }
        if (fixed && impl.renderer) impl.renderer->partial(recover);
        if (effects || fixed) impl.damage(vport);  //redraw scene full region

        return Result::Success;
    }

    Result remove(Paint* paint)
    {
        if (PAINT(paint)->parent != this) return Result::InsufficientCondition;
        //when the paint is destroyed damage will be triggered
        if (PAINT(paint)->refCnt > 1) PAINT(paint)->damage();
        PAINT(paint)->unref();
        paints.remove(paint);
        return Result::Success;
    }

    Result insert(Paint* target, Paint* at)
    {
        if (!target) return Result::InvalidArguments;
        auto timpl = PAINT(target);

        if (timpl->parent) {
            TVGERR("RENDERER", "Target paint(%p) is already owned by a parent(%p)", target, timpl->parent);
            return Result::InsufficientCondition;
        }

        target->ref();

        //Relocated the paint to the current scene space
        timpl->mark(RenderUpdateFlag::Transform);

        if (!at) {
            paints.push_back(target);
        } else {
            //OPTIMIZE: Remove searching?
            auto itr = find_if(paints.begin(), paints.end(),[&at](const Paint* paint){ return at == paint; });
            if (itr == paints.end()) return Result::InvalidArguments;
            paints.insert(itr, target);
        }
        timpl->parent = this;
        if (timpl->clipper) PAINT(timpl->clipper)->parent = this;
        if (timpl->maskData) PAINT(timpl->maskData->target)->parent = this;
        return Result::Success;
    }

    Iterator* iterator()
    {
        return new SceneIterator(&paints);
    }

    Result resetEffects(bool damage = true)
    {
        if (effects) {
            ARRAY_FOREACH(p, *effects) {
                if (impl.renderer) impl.renderer->dispose(*p);
                delete(*p);
            }
            delete(effects);
            effects = nullptr;
            if (damage) impl.damage(vport);
        }
        return Result::Success;
    }

    Result add(SceneEffect effect, va_list& args)
    {
        if (effect == SceneEffect::Clear) return resetEffects();

        if (!this->effects) this->effects = new Array<RenderEffect*>;

        RenderEffect* re = nullptr;

        switch (effect) {
            case SceneEffect::GaussianBlur: {
                re = RenderEffectGaussianBlur::gen(args);
                break;
            }
            case SceneEffect::DropShadow: {
                re = RenderEffectDropShadow::gen(args);
                break;
            }
            case SceneEffect::Fill: {
                re = RenderEffectFill::gen(args);
                break;
            }
            case SceneEffect::Tint: {
                re = RenderEffectTint::gen(args);
                break;
            }
            case SceneEffect::Tritone: {
                re = RenderEffectTritone::gen(args);
                break;
            }
            default: break;
        }

        if (!re) return Result::InvalidArguments;

        this->effects->push(re);

        return Result::Success;
    }
};

#endif //_TVG_SCENE_H_
