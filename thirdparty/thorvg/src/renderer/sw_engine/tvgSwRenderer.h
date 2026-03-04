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

#ifndef _TVG_SW_RENDERER_H_
#define _TVG_SW_RENDERER_H_

#include "tvgRender.h"

struct SwSurface;
struct SwTask;
struct SwCompositor;
struct SwMpool;

namespace tvg
{

struct SwRenderer : RenderMethod
{
    //main features
    bool preUpdate() override;
    RenderData prepare(const RenderShape& rshape, RenderData data, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags, bool clipper) override;
    RenderData prepare(RenderSurface* surface, RenderData data, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags) override;
    bool postUpdate() override;
    bool preRender() override;
    bool renderShape(RenderData data) override;
    bool renderImage(RenderData data) override;
    bool postRender() override;
    void dispose(RenderData data) override;
    RenderRegion region(RenderData data) override;
    bool bounds(RenderData data, Point* pt4, const Matrix& m) override;
    bool blend(BlendMethod method) override;
    ColorSpace colorSpace() override;
    const RenderSurface* mainSurface() override;
    bool clear() override;
    bool sync() override;
    bool intersectsShape(RenderData data, const RenderRegion& region) override;
    bool intersectsImage(RenderData data, const RenderRegion& region) override;
    bool target(pixel_t* data, uint32_t stride, uint32_t w, uint32_t h, ColorSpace cs);

    //composition
    SwSurface* request(int channelSize, bool square);
    RenderCompositor* target(const RenderRegion& region, ColorSpace cs, CompositionFlag flags) override;
    bool beginComposite(RenderCompositor* cmp, MaskMethod method, uint8_t opacity) override;
    bool endComposite(RenderCompositor* cmp) override;
    void clearCompositors();

    //post effects
    void prepare(RenderEffect* effect, const Matrix& transform) override;
    bool region(RenderEffect* effect) override;
    bool render(RenderCompositor* cmp, const RenderEffect* effect, bool direct) override;
    void dispose(RenderEffect* effect) override;

    //partial rendering
    void damage(RenderData rd, const RenderRegion& region) override;
    bool partial(bool disable) override;

    SwRenderer(uint32_t threads, EngineOption op);
    static bool term();

private:
    SwSurface*           surface = nullptr;           //active surface
    Array<SwTask*>       tasks;                       //async task list
    Array<SwSurface*>    compositors;                 //render targets cache list
    RenderDirtyRegion    dirtyRegion;                 //partial rendering support
    SwMpool*             mpool;                       //private memory pool
    bool                 sharedMpool;                 //memory-pool behavior policy
    bool                 fulldraw = true;             //buffer is cleared (need to redraw full screen)

    ~SwRenderer();

    RenderData prepareCommon(SwTask* task, const Matrix& transform, const Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags);
};

}

#endif /* _TVG_SW_RENDERER_H_ */
