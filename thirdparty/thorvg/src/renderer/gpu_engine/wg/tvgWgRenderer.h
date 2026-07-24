/*
 * Copyright (c) 2023 - 2026 ThorVG project. All rights reserved.

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

#ifndef _TVG_WG_RENDERER_H_
#define _TVG_WG_RENDERER_H_

#include "tvgWgRenderTask.h"

struct WgRenderer : RenderMethod
{
    //main features
    bool preUpdate() override;
    RenderData prepare(const RenderShape& rshape, RenderData data, const Matrix& transform, const Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags, bool clipper) override;
    RenderData prepare(RenderSurface* surface, RenderData data, const Matrix& transform, const Array<RenderData>& clips, uint8_t opacity, FilterMethod filter, RenderUpdateFlag flags) override;
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
    bool intersectsImage(RenderData data, const RenderRegion& region) override;
    bool intersectsShape(RenderData data, const RenderRegion& region) override;
    bool target(WGPUDevice device, WGPUInstance instance, void* target, uint32_t w, uint32_t h, ColorSpace cs, int type = 0);

    //composition
    RenderCompositor* target(const RenderRegion& region, ColorSpace cs, CompositionFlag flags) override;
    bool beginComposite(RenderCompositor* cmp, MaskMethod method, uint8_t opacity) override;
    bool endComposite(RenderCompositor* cmp) override;

    //post effects
    void prepare(RenderEffect* effect, const Matrix& transform) override;
    bool region(RenderEffect* effect) override;
    bool render(RenderCompositor* cmp, const RenderEffect* effect, bool direct) override;
    void dispose(RenderEffect* effect) override;

    //partial rendering
    void damage(RenderData rd, const RenderRegion& region) override;
    bool partial(bool disable) override;

    WgRenderer(uint32_t threads, EngineOption op);
    static bool term();

private:
    ~WgRenderer();
    void release();
    void disposeObjects();
    void releaseSurfaceTexture();

    void clearTargets();
    bool surfaceConfigure(WGPUSurface surface, WgContext& context, uint32_t width, uint32_t height);

    // render tree stacks
    WgRenderTarget mRenderTargetRoot;
    Array<WgCompose*> mCompositorList;
    Array<WgRenderTarget*> mRenderTargetStack;
    Array<WgSceneTask*> mSceneTaskStack;
    Array<WgRenderTask*> mRenderTaskList;

    // render target pool
    WgRenderTargetPool mRenderTargetPool;

    // render data paint pools
    WgRenderDataShapePool mRenderDataShapePool;
    WgRenderDataPicturePool mRenderDataPicturePool;
    WgRenderDataEffectParamsPool mRenderDataEffectParamsPool;

    // rendering context
    WgContext mContext;
    WgCompositor mCompositor;

    // rendering states
    RenderSurface mTargetSurface;
    BlendMethod mBlendMethod{};

    // disposable data list
    Array<RenderData> mDisposeRenderDatas{};
    Key mDisposeKey{};

    // gpu handles
    WGPUTexture targetTexture{}; // external handle
    WGPUSurfaceTexture surfaceTexture{};
    WGPUSurface surface{};  // external handle
};

#endif /* _TVG_WG_RENDERER_H_ */
