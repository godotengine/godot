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

#include "tvgTaskScheduler.h"
#include "tvgWgRenderer.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static int32_t _rendererCnt = -1;
static mutex _rendererMtx;


void WgRenderer::release()
{
    if (!mContext.queue) return;

    disposeObjects();

    // clear render data paint pools
    mRenderDataShapePool.release(mContext);
    mRenderDataPicturePool.release(mContext);
    mRenderDataEffectParamsPool.release(mContext);

    // clear render  pool
    mRenderTargetPool.release(mContext);

    // clear rendering tree stacks
    mCompositorList.clear();
    mRenderTargetStack.clear();
    mRenderTargetRoot.release(mContext);

    // release context handles
    mCompositor.release(mContext);
    mContext.release();

    // release gpu handles
    clearTargets();
}


void WgRenderer::disposeObjects()
{
    ARRAY_FOREACH(p, mDisposeRenderDatas) {
        auto renderData = (WgRenderDataPaint*)(*p);
        if (renderData->type() == Type::Shape) {
            mRenderDataShapePool.free(mContext, (WgRenderDataShape*)renderData);
        } else {
            mRenderDataPicturePool.free(mContext, (WgRenderDataPicture*)renderData);
        }
    }
    mDisposeRenderDatas.clear();
}


void WgRenderer::releaseSurfaceTexture()
{
    if (surfaceTexture.texture) {
        wgpuTextureRelease(surfaceTexture.texture);
        surfaceTexture.texture = nullptr;
    }
}


void WgRenderer::clearTargets()
{
    releaseSurfaceTexture();
    if (surface) wgpuSurfaceUnconfigure(surface);
    targetTexture = nullptr;
    surface = nullptr;
    mTargetSurface.stride = 0;
    mTargetSurface.w = 0;
    mTargetSurface.h = 0;

}


bool WgRenderer::surfaceConfigure(WGPUSurface surface, WgContext& context, uint32_t width, uint32_t height)
{
    this->surface = surface;
    if (width == 0 || height == 0 || !surface) return false;

    // setup surface configuration
    WGPUSurfaceConfiguration surfaceConfiguration {
        .device = context.device,
        .format = context.format,
        .usage = WGPUTextureUsage_RenderAttachment,
        .width = width,
        .height = height,
    #ifdef __EMSCRIPTEN__
        .alphaMode = WGPUCompositeAlphaMode_Premultiplied,
        .presentMode = WGPUPresentMode_Fifo
    #elif __linux__
    #else
        .presentMode = WGPUPresentMode_Immediate
    #endif
    };
    wgpuSurfaceConfigure(surface, &surfaceConfiguration);
    return true;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

RenderData WgRenderer::prepare(const RenderShape& rshape, RenderData data, const Matrix& transform, const Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags, bool clipper)
{
    auto renderDataShape = data ? (WgRenderDataShape*)data : mRenderDataShapePool.allocate(mContext);

    // update geometry
    if (!data || (flags & (RenderUpdateFlag::Transform | RenderUpdateFlag::Path | RenderUpdateFlag::Stroke))) {
        renderDataShape->updateMeshes(rshape, flags, transform);
    }

    // update transform
    if ((!data) || (flags & RenderUpdateFlag::Transform)) {
        renderDataShape->transform = transform;
        renderDataShape->updateAABB();
    }

    // update paint settings
    if ((!data) || (flags & (RenderUpdateFlag::Transform | RenderUpdateFlag::Blend | RenderUpdateFlag::Color))) {
        renderDataShape->renderSettingsShape.update(mContext, mTargetSurface.cs, opacity);
        renderDataShape->renderSettingsStroke.update(mContext, mTargetSurface.cs, opacity);
        renderDataShape->fillRule = rshape.rule;
    }

    // setup fill settings
    renderDataShape->viewport = vport;
    renderDataShape->updateVisibility(rshape, opacity);
    // update shape render settings
    if (!renderDataShape->renderSettingsShape.skip) {
        if (rshape.fill && (!data || (flags & (RenderUpdateFlag::Gradient | RenderUpdateFlag::Transform)))) {
            bool updateColorRamp = !data || ((flags & RenderUpdateFlag::Gradient) != RenderUpdateFlag::None);
            renderDataShape->renderSettingsShape.update(mContext, rshape.fill, &transform, updateColorRamp);
        } else if (!data || (flags & RenderUpdateFlag::Color)) {
            renderDataShape->renderSettingsShape.update(mContext, rshape.color);
        }
    }
    // update strokes render settings
    if (rshape.stroke && !renderDataShape->renderSettingsStroke.skip) {
        if (rshape.stroke->fill && (!data || (flags & (RenderUpdateFlag::GradientStroke | RenderUpdateFlag::Transform)))) {
            bool updateColorRamp = !data || ((flags & RenderUpdateFlag::GradientStroke) != RenderUpdateFlag::None);
            renderDataShape->renderSettingsStroke.update(mContext, rshape.stroke->fill, &transform, updateColorRamp);
        } else if (!data || (flags & RenderUpdateFlag::Stroke)) {
            renderDataShape->renderSettingsStroke.update(mContext, rshape.stroke->color);
        }
    }

    if (flags & RenderUpdateFlag::Clip) renderDataShape->updateClips(clips);

    return renderDataShape;
}

RenderData WgRenderer::prepare(RenderSurface* surface, RenderData data, const Matrix& transform, const Array<RenderData>& clips, uint8_t opacity, FilterMethod filter, RenderUpdateFlag flags)
{
    auto renderDataPicture = data ? (WgRenderDataPicture*)data : mRenderDataPicturePool.allocate(mContext);

    // update paint settings
    renderDataPicture->viewport = vport;
    renderDataPicture->transform = transform;
    if (!data || (flags & (RenderUpdateFlag::Blend | RenderUpdateFlag::Color))) {
        renderDataPicture->renderSettings.update(mContext, surface->cs, opacity);
    }

    // update image data
    if (!data || (flags & (RenderUpdateFlag::Transform | RenderUpdateFlag::Path | RenderUpdateFlag::Image))) {
        auto updateTexture = !data || ((flags & (RenderUpdateFlag::Path | RenderUpdateFlag::Image)) != RenderUpdateFlag::None);
        renderDataPicture->updateSurface(mContext, surface, transform, filter, updateTexture);
    }

    if (flags & RenderUpdateFlag::Clip) renderDataPicture->updateClips(clips);

    return renderDataPicture;
}


bool WgRenderer::preRender()
{
    if (mContext.invalid()) return false;

    mCompositor.reset(mContext);

    assert(mRenderTargetStack.count == 0);
    mRenderTargetStack.push(&mRenderTargetRoot);

    // create root compose settings
    WgCompose* compose = new WgCompose();
    compose->aabb = { { 0, 0 }, { (int32_t)mTargetSurface.w, (int32_t)mTargetSurface.h } };
    compose->blend = BlendMethod::Normal;
    compose->method = MaskMethod::None;
    compose->opacity = 255;
    mCompositorList.push(compose);

    // create root scene
    WgSceneTask* sceneTask = new WgSceneTask(&mRenderTargetRoot, compose, nullptr);
    mRenderTaskList.push(sceneTask);
    mSceneTaskStack.push(sceneTask);
    return true;
}


bool WgRenderer::renderShape(RenderData data)
{
    WgPaintTask* paintTask = new WgPaintTask((WgRenderDataPaint*)data, mBlendMethod);
    WgSceneTask* sceneTask = mSceneTaskStack.last();
    sceneTask->children.push(paintTask);
    mRenderTaskList.push(paintTask);
    mCompositor.requestShape((WgRenderDataShape*)data);
    return true;
}


bool WgRenderer::renderImage(RenderData data)
{
    WgPaintTask* paintTask = new WgPaintTask((WgRenderDataPaint*)data, mBlendMethod);
    WgSceneTask* sceneTask = mSceneTaskStack.last();
    sceneTask->children.push(paintTask);
    mRenderTaskList.push(paintTask);
    mCompositor.requestImage((WgRenderDataPicture*)data);
    return true;
}


bool WgRenderer::postRender()
{
    // flush stage data to gpu
    mCompositor.flush(mContext);

    // create command encoder for drawing
    WGPUCommandEncoder commandEncoder = mContext.createCommandEncoder();

    // run rendering (all the fun is here)
    WgSceneTask* sceneTaskRoot = mSceneTaskStack.last();
    sceneTaskRoot->run(mContext, mCompositor, commandEncoder);

    // execute and release command encoder
    mContext.submitCommandEncoder(commandEncoder);
    mContext.releaseCommandEncoder(commandEncoder);

    // clear the render tasks tree
    mSceneTaskStack.pop();
    assert(mSceneTaskStack.count == 0);
    mRenderTargetStack.pop();
    assert(mRenderTargetStack.count == 0);
    ARRAY_FOREACH(p, mRenderTaskList) { delete (*p); };
    mRenderTaskList.clear();
    ARRAY_FOREACH(p, mCompositorList) { delete (*p); };
    mCompositorList.clear();
    return true;
}


void WgRenderer::dispose(RenderData data) {
    if (!mContext.queue) return;
    ScopedLock lock(mDisposeKey);
    mDisposeRenderDatas.push(data);
}


bool WgRenderer::bounds(RenderData data, Point* pt4, const Matrix& m)
{
    if (data) {
        auto renderDataPaint = (WgRenderDataPaint*)data;
        if (renderDataPaint->type() == Type::Shape) {
            auto renderData = (WgRenderDataShape*)data;
            if (!renderData->renderSettingsStroke.skip) {
                tvg::BBox bbox;
                bbox.init();
                auto& vertexes = renderData->meshStrokes.vbuffer;

                if (m == renderData->transform) {
                    for (uint32_t i = 0; i < vertexes.count; i++) {
                        Point vert = vertexes[i];
                        bbox.min = min(bbox.min, vert);
                        bbox.max = max(bbox.max, vert);
                    }
                } else {
                    Matrix inverseModel;
                    inverse(&renderData->transform, &inverseModel);
                    inverseModel *= m;
                    for (uint32_t i = 0; i < vertexes.count; i++) {
                        Point vert = vertexes[i];
                        vert *= inverseModel;
                        bbox.min = min(bbox.min, vert);
                        bbox.max = max(bbox.max, vert);
                    }
                }

                pt4[0] = bbox.min;
                pt4[1] = {bbox.max.x, bbox.min.y};
                pt4[2] = bbox.max;
                pt4[3] = {bbox.min.x, bbox.max.y};
                return true;
            }
        }
    }
    return false;
}

RenderRegion WgRenderer::region(RenderData data)
{
    if (!data) return {};
    auto renderData = (WgRenderDataPaint*)data;
    if (renderData->type() == Type::Shape) {
        auto& v1 = renderData->aabb.min;
        auto& v2 = renderData->aabb.max;
        return {{int32_t(nearbyint(v1.x)), int32_t(nearbyint(v1.y))}, {int32_t(nearbyint(v2.x)), int32_t(nearbyint(v2.y))}};
    }
    return {{0, 0}, {(int32_t)mTargetSurface.w, (int32_t)mTargetSurface.h}};
}


bool WgRenderer::blend(BlendMethod method)
{
    mBlendMethod = (method == BlendMethod::Composition ? BlendMethod::Normal : method);

    return true;
}


ColorSpace WgRenderer::colorSpace()
{
    return mTargetSurface.cs;
}


const RenderSurface* WgRenderer::mainSurface()
{
    return &mTargetSurface;
}


bool WgRenderer::clear()
{
    if (mContext.invalid()) return false;

    //TODO: clear the current target buffer only if clear() is called
    return true;
}


bool WgRenderer::sync()
{
    if (mContext.invalid()) return false;

    disposeObjects();

    // if texture buffer used
    WGPUTexture dstTexture = targetTexture;
    if (surface) {
        releaseSurfaceTexture();
        wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
        dstTexture = surfaceTexture.texture;
    }

    if (!dstTexture) return false;

    // insure that surface and offscreen target have the same size
    if ((wgpuTextureGetWidth(dstTexture) == mRenderTargetRoot.width) && 
        (wgpuTextureGetHeight(dstTexture) == mRenderTargetRoot.height)) {
        WGPUTextureView dstTextureView = mContext.createTextureView(dstTexture);
        WGPUCommandEncoder commandEncoder = mContext.createCommandEncoder();
        // show root offscreen buffer
        mCompositor.blit(mContext, commandEncoder, &mRenderTargetRoot, dstTextureView);
        mContext.submitCommandEncoder(commandEncoder);
        mContext.releaseCommandEncoder(commandEncoder);
        mContext.releaseTextureView(dstTextureView);
    }

    return true;
}


bool WgRenderer::target(WGPUDevice device, WGPUInstance instance, void* target, uint32_t w, uint32_t h, ColorSpace cs, int type)
{
    if (!instance || !device || !target) {
        release();
        return true;
    }

    if (w == 0 || h == 0) return false;

    // device or instance was changed, need to recreate all instances
    if ((mContext.device != device) || (mContext.instance != instance)) {
        release();
        mContext.initialize(instance, device);
        mRenderTargetPool.initialize(mContext, w, h);
        mRenderTargetRoot.initialize(mContext, w, h);
        mCompositor.initialize(mContext, w, h);

    // update render targets dimensions
    } else if ((mTargetSurface.w != w) || (mTargetSurface.h != h) || (type == 0 ? (surface != (WGPUSurface)target) : (targetTexture != (WGPUTexture)target))) {
        mRenderTargetPool.release(mContext);
        mRenderTargetRoot.release(mContext);
        clearTargets();
        mRenderTargetPool.initialize(mContext, w, h);
        mRenderTargetRoot.initialize(mContext, w, h);
        mCompositor.resize(mContext, w, h);
    }

    // configure surface (must be called after context creation)
    if (type == 0) {
        surface = (WGPUSurface)target;
        surfaceConfigure(surface, mContext, w, h);
    } else {
        targetTexture = (WGPUTexture)target;
    }

    mTargetSurface.stride = w;
    mTargetSurface.w = w;
    mTargetSurface.h = h;
    mTargetSurface.cs = cs;

    return true;
}

WgRenderer::~WgRenderer()
{
    release();

    _rendererMtx.lock();
    --_rendererCnt;
    _rendererMtx.unlock();
}


RenderCompositor* WgRenderer::target(const RenderRegion& region, TVG_UNUSED ColorSpace cs, TVG_UNUSED CompositionFlag flags)
{
    // create and setup compose data
    WgCompose* compose = new WgCompose();
    compose->aabb = region;
    compose->flags = flags;
    mCompositorList.push(compose);
    return compose;
}


// please, keep beginComposite/endComposite aligned by logic
bool WgRenderer::beginComposite(RenderCompositor* cmp, MaskMethod method, uint8_t opacity)
{
    // current composition
    WgCompose* compose = (WgCompose*)cmp;
    // allocate new render target and push to the render tree stack
    WgRenderTarget* newRenderTarget = mRenderTargetPool.allocate(mContext);
    mRenderTargetStack.push(newRenderTarget);
    // current and new scenes
    WgSceneTask* curSceneTask = mSceneTaskStack.last();
    WgSceneTask* newSceneTask = new WgSceneTask(newRenderTarget, compose, curSceneTask);
    // setup masking and blending scenes configuration
    if ((compose->flags & (tvg::Blending | tvg::Masking)) == (tvg::Blending | tvg::Masking)) {
        // new scene task must be as masked but not blended
        newSceneTask->compose->method = curSceneTask->compose->method;
        newSceneTask->compose->opacity = 255;
        newSceneTask->compose->blend = BlendMethod::Normal;
        newSceneTask->renderTargetMsk = curSceneTask->renderTargetMsk;
        newSceneTask->renderTargetDst = curSceneTask->renderTarget;
        // cur scene task must be blended but not masked
        curSceneTask->compose->method = MaskMethod::None;
        curSceneTask->compose->opacity = opacity;
        curSceneTask->compose->blend = mBlendMethod;
        curSceneTask->renderTargetMsk = nullptr;
    } else // setup masking scenes configuration
    if (method != MaskMethod::None) {
        // new scene task must be masked
        newSceneTask->compose->masked = true;
        newSceneTask->compose->method = method;
        newSceneTask->compose->opacity = opacity;
        newSceneTask->compose->blend = BlendMethod::Normal;
        newSceneTask->renderTargetMsk = curSceneTask->renderTarget;
        newSceneTask->renderTargetDst = curSceneTask->renderTargetDst;
        // cur scene task must not be blended
        curSceneTask->renderTargetDst = nullptr;
    } else // setup blending scenes configuration
    if (method == MaskMethod::None) {
        // new scene task must be blended
        newSceneTask->compose->method = MaskMethod::None;
        newSceneTask->compose->opacity = opacity;
        newSceneTask->compose->blend = mBlendMethod;
        newSceneTask->renderTargetMsk = nullptr;
        newSceneTask->renderTargetDst = curSceneTask->renderTarget;
    }
    curSceneTask->children.push(newSceneTask);
    mRenderTaskList.push(newSceneTask);
    mSceneTaskStack.push(newSceneTask);

    return true;
}


// please, keep beginComposite/endComposite aligned by logic
bool WgRenderer::endComposite(RenderCompositor* cmp)
{
    // pop targets and scenes from render tree
    mRenderTargetPool.free(mContext, mRenderTargetStack.last());
    mSceneTaskStack.pop();
    mRenderTargetStack.pop();
    // in a case of masked target we must pop mask targets and scenes also
    WgCompose* compose = (WgCompose*)cmp;
    if (compose->masked) {
        mRenderTargetPool.free(mContext, mRenderTargetStack.last());
        mSceneTaskStack.pop();
        mRenderTargetStack.pop();
    }
    return true;
}


void WgRenderer::prepare(RenderEffect* effect, const Matrix& transform)
{
    if (!effect->rd) effect->rd = mRenderDataEffectParamsPool.allocate(mContext);
    auto renderData = (WgRenderDataEffectParams*)effect->rd;

    if (effect->type == SceneEffect::GaussianBlur) {
        renderData->update(mContext, (RenderEffectGaussianBlur*)effect, transform);
    } else if (effect->type == SceneEffect::DropShadow) {
        renderData->update(mContext, (RenderEffectDropShadow*)effect, transform);
    } else if (effect->type == SceneEffect::Fill) {
        renderData->update(mContext, (RenderEffectFill*)effect);
    } else if (effect->type == SceneEffect::Tint) {
        renderData->update(mContext, (RenderEffectTint*)effect);
    } else if (effect->type == SceneEffect::Tritone) {
        renderData->update(mContext, (RenderEffectTritone*)effect);
    } else {
        TVGERR("WG_ENGINE", "Missing effect type? = %d", (int) effect->type);
        return;
    }
}


bool WgRenderer::region(RenderEffect* effect)
{
    if (effect->type == SceneEffect::GaussianBlur) {
        auto gaussian = (RenderEffectGaussianBlur*)effect;
        auto renderData = (WgRenderDataEffectParams*)gaussian->rd;
        if (gaussian->direction != 2) {
            gaussian->extend.min.x = -renderData->extend;
            gaussian->extend.max.x = +renderData->extend;
        }
        if (gaussian->direction != 1) {
            gaussian->extend.min.y = -renderData->extend;
            gaussian->extend.max.y = +renderData->extend;
        }
        return true;
    } else if (effect->type == SceneEffect::DropShadow) {
        auto dropShadow = (RenderEffectDropShadow*)effect;
        auto renderData = (WgRenderDataEffectParams*)dropShadow->rd;
        dropShadow->extend.min.x = -std::ceil(renderData->extend + std::abs(renderData->offset.x));
        dropShadow->extend.min.y = -std::ceil(renderData->extend + std::abs(renderData->offset.y));
        dropShadow->extend.max.x = +std::floor(renderData->extend + std::abs(renderData->offset.x));
        dropShadow->extend.max.y = +std::floor(renderData->extend + std::abs(renderData->offset.y));
        return true;
    }
    return false;
}


bool WgRenderer::render(RenderCompositor* cmp, const RenderEffect* effect, TVG_UNUSED bool direct)
{
    mSceneTaskStack.last()->effect = effect;
    return true;
}


void WgRenderer::dispose(RenderEffect* effect)
{
    auto renderData = (WgRenderDataEffectParams*)effect->rd;
    mRenderDataEffectParamsPool.free(mContext, renderData);
    effect->rd = nullptr;
};


bool WgRenderer::preUpdate()
{
    if (mContext.invalid()) return false;

    return true;
}


bool WgRenderer::postUpdate()
{
    return true;
}


void WgRenderer::damage(TVG_UNUSED RenderData rd, TVG_UNUSED const RenderRegion& region)
{
    //TODO
}


bool WgRenderer::partial(bool disable)
{
    //TODO
    return false;
}


bool WgRenderer::intersectsShape(RenderData data, TVG_UNUSED const RenderRegion& region)
{
    if (!data) return false;
    auto shape = (WgRenderDataShape*)data;
    RenderRegion bbox = {
        {(int32_t)shape->aabb.min.x, (int32_t)shape->aabb.min.y},
        {(int32_t)shape->aabb.max.x, (int32_t)shape->aabb.max.y}
    };
    if (region.intersected(bbox)) {
        if (region.contained(bbox)) return true;
        WgIntersector intersector;
        return intersector.intersectShape(RenderRegion::intersect(region, bbox), shape);
    }
    return false;
}


bool WgRenderer::intersectsImage(RenderData data, TVG_UNUSED const RenderRegion& region)
{
    if (!data) return false;
    auto picture = (WgRenderDataPicture*)data;
    WgIntersector intersector;
    if (intersector.intersectImage(region, picture)) return true;
    return false;
}


bool WgRenderer::term()
{
    _rendererMtx.lock();
    if (_rendererCnt > 0) {
        _rendererMtx.unlock();
        return false;
    }
    _rendererCnt = -1;
    _rendererMtx.unlock();

    //TODO: clean up global resources

    return true;
}

WgRenderer::WgRenderer(TVG_UNUSED uint32_t threads, TVG_UNUSED EngineOption op)
{
    _rendererMtx.lock();
    if (_rendererCnt == -1) {
        //TODO: initialize the global engine
        _rendererCnt = 0;
    }
    ++_rendererCnt;
    _rendererMtx.unlock();
}
