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

#ifndef _TVG_CANVAS_H_
#define _TVG_CANVAS_H_

#include "tvgPaint.h"

enum Status : uint8_t {Synced = 0, Painting, Updating, Drawing, Damaged};

struct Canvas::Impl
{
    Scene* scene;
    RenderMethod* renderer;
    RenderRegion vport = {{0, 0}, {INT32_MAX, INT32_MAX}};
    Status status = Status::Synced;

    Impl() : scene(Scene::gen())
    {
        scene->ref();
    }

    ~Impl()
    {
        //make it sure any deferred jobs
        renderer->sync();
        scene->unref();
        if (renderer->unref() == 0) delete(renderer);
    }

    Result add(Paint* target, Paint* at)
    {
        if (PAINT(target)->renderer && PAINT(target)->renderer != renderer) {
            TVGERR("RENDERER", "Target paint(%p) is already owned by a different renderer.", target);
            return Result::InsufficientCondition;
        }

        //You cannot add paints during rendering.
        if (status == Status::Drawing) {
            TVGLOG("RENDERER", "add() was called during drawing.");
            return Result::InsufficientCondition;
        }
        status = Status::Painting;
        return scene->add(target, at);
    }

    Result remove(Paint* paint)
    {
        if (status == Status::Drawing) {
            TVGLOG("RENDERER", "remove() was called during drawing.");
            return Result::InsufficientCondition;
        }
        status = Status::Painting;
        return scene->remove(paint);
    }

    Result update()
    {
        if (status == Status::Updating) return Result::Success;

        if (status == Status::Drawing) {
            TVGLOG("RENDERER", "update() was called during drawing.");
            return Result::InsufficientCondition;
        }

        Array<RenderData> clips;
        auto flag = RenderUpdateFlag::None;

        //TODO: All is too harsh, can be optimized.
        if (status == Status::Damaged) flag = RenderUpdateFlag::All;

        if (!renderer->preUpdate()) return Result::InsufficientCondition;

        auto m = tvg::identity();
        PAINT(scene)->update(renderer, m, clips, 255, flag);

        if (!renderer->postUpdate()) return Result::InsufficientCondition;

        status = Status::Updating;
        return Result::Success;
    }

    Result draw(bool clear)
    {
        if (status == Status::Drawing) {
            TVGLOG("RENDERER", "draw() was called multiple times.");
            return Result::InsufficientCondition;
        }
        if (status == Status::Painting || status == Status::Damaged) update();
        if (status != Status::Updating) return Result::InsufficientCondition;
        if (clear && !renderer->clear()) return Result::InsufficientCondition;
        if (!renderer->preRender()) return Result::InsufficientCondition;
        if (!PAINT(scene)->render(renderer) || !renderer->postRender()) return Result::InsufficientCondition;

        status = Status::Drawing;

        return Result::Success;
    }

    Result sync()
    {
        if (status == Status::Synced) return Result::Success;
        if (renderer->sync()) {
            status = Status::Synced;
            return Result::Success;
        }
        return Result::Unknown;
    }

    Result viewport(int32_t x, int32_t y, int32_t w, int32_t h)
    {
        if (status == Status::Synced || status == Status::Damaged) {
            RenderRegion val = {{x, y}, {x + w, y + h}};
            //intersect if the target buffer is already set.
            auto surface = renderer->mainSurface();
            if (surface && surface->w > 0 && surface->h > 0) {
                val.intersect({{0, 0}, {(int32_t)surface->w, (int32_t)surface->h}});
            }
            if (vport == val) return Result::Success;
            renderer->viewport(val);
            vport = val;
            status = Status::Damaged;
            return Result::Success;
        }
        TVGLOG("RENDERER", "viewport() is only allowed after sync.");
        return Result::InsufficientCondition;
    }
};

#endif /* _TVG_CANVAS_H_ */
