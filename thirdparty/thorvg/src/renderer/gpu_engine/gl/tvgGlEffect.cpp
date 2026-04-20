/*
 * Copyright (c) 2026 ThorVG project. All rights reserved.

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
#include "tvgGlRenderTask.h"
#include "tvgGlGpuBuffer.h"
#include "tvgGlRenderPass.h"
#include "tvgGlShaderSrc.h"
#include "tvgGlEffect.h"


/************************************************************************/
/* Gaussian Blur                                                        */
/************************************************************************/

struct GlGaussianBlur {
    float sigma{};
    float scale{};
    float extend{};
    float dummy0{};
};


bool GlEffect::region(RenderEffectGaussianBlur* effect)
{
    auto gaussianBlur = (GlGaussianBlur*)effect->rd;
    if (effect->direction != 2) {
        effect->extend.min.x = -gaussianBlur->extend;
        effect->extend.max.x = +gaussianBlur->extend;
    }
    if (effect->direction != 1) {
        effect->extend.min.y = -gaussianBlur->extend;
        effect->extend.max.y = +gaussianBlur->extend;
    }
    return true;
};


void GlEffect::update(RenderEffectGaussianBlur* effect, const Matrix& transform)
{
    GlGaussianBlur* blur = (GlGaussianBlur*)effect->rd;
    if (!blur) blur = tvg::malloc<GlGaussianBlur>(sizeof(GlGaussianBlur));
    blur->sigma = effect->sigma;
    blur->scale = std::sqrt(transform.e11 * transform.e11 + transform.e12 * transform.e12);
    blur->extend = 2 * blur->sigma * blur->scale;
    effect->rd = blur;
    effect->valid = (blur->extend > 0);
}


GlRenderTask* GlEffect::render(RenderEffectGaussianBlur* effect, GlRenderTarget* dstFbo, Array<GlRenderTargetPool*>& blendPool, const RenderRegion& vp, uint32_t voffset, uint32_t ioffset)
{
    if (!pBlurV) pBlurV = new GlProgram(EFFECT_VERTEX, GAUSSIAN_VERTICAL);
    if (!pBlurH) pBlurH = new GlProgram(EFFECT_VERTEX, GAUSSIAN_HORIZONTAL);

    // get current and intermediate framebuffers
    auto dstCopyFbo0 = blendPool[0]->getRenderTarget(vp);
    auto dstCopyFbo1 = blendPool[1]->getRenderTarget(vp);

    // add uniform data
    float viewport[4] {(float)vp.min.x, (float)vp.min.y, (float)vp.max.x, (float)vp.max.y};
    auto blurOffset = gpuBuffer->push((GlGaussianBlur*)(effect->rd), sizeof(GlGaussianBlur), true);
    auto viewportOffset = gpuBuffer->push(viewport, sizeof(viewport), true);

    // create gaussian blur tasks
    auto task = new GlGaussianBlurTask(dstFbo, dstCopyFbo0, dstCopyFbo1);
    task->effect = effect;
    task->setViewport({{0, 0}, {vp.sw(), vp.sh()}});
    // horizontal blur task and geometry
    task->horzTask = new GlRenderTask(pBlurH);
    task->horzTask->addBindResource(GlBindingResource{0, pBlurH->getUniformBlockIndex("Gaussian"), gpuBuffer->getBufferId(), blurOffset, sizeof(GlGaussianBlur)});
    task->horzTask->addBindResource(GlBindingResource{1, pBlurH->getUniformBlockIndex("Viewport"), gpuBuffer->getBufferId(), viewportOffset, sizeof(viewport)});
    task->horzTask->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), voffset});
    task->horzTask->setDrawRange(ioffset, 6);
    // vertical blur task and geometry
    task->vertTask = new GlRenderTask(pBlurV);
    task->vertTask->addBindResource(GlBindingResource{0, pBlurV->getUniformBlockIndex("Gaussian"), gpuBuffer->getBufferId(), blurOffset, sizeof(GlGaussianBlur)});
    task->vertTask->addBindResource(GlBindingResource{1, pBlurV->getUniformBlockIndex("Viewport"), gpuBuffer->getBufferId(), viewportOffset, sizeof(viewport)});
    task->vertTask->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), voffset});
    task->vertTask->setDrawRange(ioffset, 6);

    return task;
}


/************************************************************************/
/* DropShadow                                                           */
/************************************************************************/

struct GlDropShadow: GlGaussianBlur {
    float color[4];
    float offset[2];
};


bool GlEffect::region(RenderEffectDropShadow* effect)
{
    auto gaussianBlur = (GlDropShadow*)effect->rd;
    effect->extend.min.x = -gaussianBlur->extend;
    effect->extend.max.x = +gaussianBlur->extend;
    effect->extend.min.y = -gaussianBlur->extend;
    effect->extend.max.y = +gaussianBlur->extend;
    return true;
};


void GlEffect::update(RenderEffectDropShadow* effect, const Matrix& transform)
{
    GlDropShadow* dropShadow = (GlDropShadow*)effect->rd;
    if (!dropShadow) dropShadow = tvg::malloc<GlDropShadow>(sizeof(GlDropShadow));
    const auto scale = std::sqrt(transform.e11 * transform.e11 + transform.e12 * transform.e12);
    const auto radian = tvg::deg2rad(90.0f - effect->angle) - tvg::radian(transform);
    const Point offset = {-effect->distance * cosf(radian) * scale, -effect->distance * sinf(radian) * scale};

    dropShadow->sigma = effect->sigma;
    dropShadow->scale = scale;
    dropShadow->color[3] = effect->color[3] / 255.0f;
    //Drop shadow effect applies blending in the shader (GL_BLEND disabled), so the color should be premultiplied:
    dropShadow->color[0] = effect->color[0] / 255.0f * dropShadow->color[3];
    dropShadow->color[1] = effect->color[1] / 255.0f * dropShadow->color[3];
    dropShadow->color[2] = effect->color[2] / 255.0f * dropShadow->color[3];
    dropShadow->offset[0] = offset.x;
    dropShadow->offset[1] = offset.y;
    dropShadow->extend = 2 * std::max(effect->sigma * scale + std::abs(offset.x), effect->sigma * scale + std::abs(offset.y));
    effect->rd = dropShadow;
    effect->valid = (dropShadow->extend >= 0);
}


GlRenderTask* GlEffect::render(RenderEffectDropShadow* effect, GlRenderTarget* dstFbo, Array<GlRenderTargetPool*>& blendPool, const RenderRegion& vp, uint32_t voffset, uint32_t ioffset)
{
    if (!pBlurV) pBlurV = new GlProgram(EFFECT_VERTEX, GAUSSIAN_VERTICAL);
    if (!pBlurH) pBlurH = new GlProgram(EFFECT_VERTEX, GAUSSIAN_HORIZONTAL);
    if (!pDropShadow) pDropShadow = new GlProgram(EFFECT_VERTEX, EFFECT_DROPSHADOW);

    // get current and intermediate framebuffers
    auto dstCopyFbo0 = blendPool[0]->getRenderTarget(vp);
    auto dstCopyFbo1 = blendPool[1]->getRenderTarget(vp);

    // add uniform data
    float viewport[4] {(float)vp.min.x, (float)vp.min.y, (float)vp.max.x, (float)vp.max.y};
    GlDropShadow* params = (GlDropShadow*)(effect->rd);
    auto paramsOffset = gpuBuffer->push(params, sizeof(GlDropShadow), true);
    auto viewportOffset = gpuBuffer->push(viewport, sizeof(viewport), true);

    // create gaussian blur tasks
    auto task = new GlEffectDropShadowTask(pDropShadow, dstFbo, dstCopyFbo0, dstCopyFbo1);
    task->effect = (RenderEffectDropShadow*)effect;
    task->setViewport({{0, 0}, {vp.sw(), vp.sh()}});
    task->addBindResource(GlBindingResource{0, pDropShadow->getUniformBlockIndex("DropShadow"), gpuBuffer->getBufferId(), paramsOffset, sizeof(GlDropShadow)});
    task->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), voffset});
    task->setDrawRange(ioffset, 6);

    // horizontal blur task and geometry
    task->horzTask = new GlRenderTask(pBlurH);
    task->horzTask->addBindResource(GlBindingResource{0, pBlurH->getUniformBlockIndex("Gaussian"), gpuBuffer->getBufferId(), paramsOffset, sizeof(GlGaussianBlur)});
    task->horzTask->addBindResource(GlBindingResource{1, pBlurH->getUniformBlockIndex("Viewport"), gpuBuffer->getBufferId(), viewportOffset, sizeof(viewport)});
    task->horzTask->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), voffset});
    task->horzTask->setDrawRange(ioffset, 6);

    // vertical blur task and geometry
    task->vertTask = new GlRenderTask(pBlurV);
    task->vertTask->addBindResource(GlBindingResource{0, pBlurV->getUniformBlockIndex("Gaussian"), gpuBuffer->getBufferId(), paramsOffset, sizeof(GlGaussianBlur)});
    task->vertTask->addBindResource(GlBindingResource{1, pBlurV->getUniformBlockIndex("Viewport"), gpuBuffer->getBufferId(), viewportOffset, sizeof(viewport)});
    task->vertTask->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), voffset});
    task->vertTask->setDrawRange(ioffset, 6);

    return task;
}


/************************************************************************/
/* ColorReplacement                                                     */
/************************************************************************/

struct GlEffectParams {
    // fill:          [0..3]: color
    // tint:          [0..2]: black,  [4..6]: white,   [8]: intensity
    // tritone:       [0..2]: shadow, [4..6]: midtone, [8..10]: highlight [11]: blender
    float params[4+4+4];
};


void GlEffect::update(RenderEffectFill* effect, const Matrix& transform)
{
    auto params = (GlEffectParams*)effect->rd;
    if (!params) params = tvg::malloc<GlEffectParams>(sizeof(GlEffectParams));
    params->params[0] = effect->color[0] / 255.0f;
    params->params[1] = effect->color[1] / 255.0f;
    params->params[2] = effect->color[2] / 255.0f;
    params->params[3] = effect->color[3] / 255.0f;
    effect->rd = params;
    effect->valid = true;
}


void GlEffect::update(RenderEffectTint* effect, const Matrix& transform)
{
    effect->valid = (effect->intensity > 0);
    if (!effect->valid) return;

    auto params = (GlEffectParams*)effect->rd;
    if (!params) params = tvg::malloc<GlEffectParams>(sizeof(GlEffectParams));
    params->params[0] = effect->black[0] / 255.0f;
    params->params[1] = effect->black[1] / 255.0f;
    params->params[2] = effect->black[2] / 255.0f;
    params->params[3] = 0.0f;
    params->params[4] = effect->white[0] / 255.0f;
    params->params[5] = effect->white[1] / 255.0f;
    params->params[6] = effect->white[2] / 255.0f;
    params->params[7] = 0.0f;
    params->params[8] = effect->intensity / 255.0f;
    effect->rd = params;
}


void GlEffect::update(RenderEffectTritone* effect, const Matrix& transform)
{
    effect->valid = (effect->blender < 255);
    if (!effect->valid) return;

    auto params = (GlEffectParams*)effect->rd;
    if (!params) params = tvg::malloc<GlEffectParams>(sizeof(GlEffectParams));
    params->params[0] = effect->shadow[0] / 255.0f;
    params->params[1] = effect->shadow[1] / 255.0f;
    params->params[2] = effect->shadow[2] / 255.0f;
    params->params[3] = 0.0f;
    params->params[4] = effect->midtone[0] / 255.0f;
    params->params[5] = effect->midtone[1] / 255.0f;
    params->params[6] = effect->midtone[2] / 255.0f;
    params->params[7] = 0.0f;
    params->params[8] = effect->highlight[0] / 255.0f;
    params->params[9] = effect->highlight[1] / 255.0f;
    params->params[10] = effect->highlight[2] / 255.0f;
    params->params[11] = effect->blender / 255.0f;
    effect->rd = params;
}


GlRenderTask* GlEffect::render(RenderEffect* effect, GlRenderTarget* dstFbo, Array<GlRenderTargetPool*>& blendPool, const RenderRegion& vp, uint32_t voffset, uint32_t ioffset)
{
    //common color replacement effects
    GlProgram* program = nullptr;
    if (effect->type == SceneEffect::Fill) {
        if (!pFill) pFill = new GlProgram(EFFECT_VERTEX, EFFECT_FILL);
        program = pFill;
    } else if (effect->type == SceneEffect::Tint) {
        if (!pTint) pTint = new GlProgram(EFFECT_VERTEX, EFFECT_TINT);
        program = pTint;
    } else if (effect->type == SceneEffect::Tritone) {
        if (!pTritone) pTritone = new GlProgram(EFFECT_VERTEX, EFFECT_TRITONE);
        program = pTritone;
    } else return nullptr;

    // get current and intermediate framebuffers
    auto dstCopyFbo = blendPool[0]->getRenderTarget(vp);

    // add uniform data
    auto params = (GlEffectParams*)(effect->rd);
    auto paramsOffset = gpuBuffer->push(params, sizeof(GlEffectParams), true);

    // create and setup task
    auto task = new GlEffectColorTransformTask(program, dstFbo, dstCopyFbo);
    task->setViewport({{0, 0}, {vp.sw(), vp.sh()}});
    task->addBindResource(GlBindingResource{0, program->getUniformBlockIndex("Params"), gpuBuffer->getBufferId(), paramsOffset, sizeof(GlEffectParams)});
    task->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), voffset});
    task->setDrawRange(ioffset, 6);

    return task;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

void GlEffect::update(RenderEffect* effect, const Matrix& transform)
{
    switch (effect->type) {
        case SceneEffect::GaussianBlur: update(static_cast<RenderEffectGaussianBlur*>(effect), transform); break;
        case SceneEffect::DropShadow : update(static_cast<RenderEffectDropShadow*>(effect), transform); break;
        case SceneEffect::Fill: update(static_cast<RenderEffectFill*>(effect), transform); break;
        case SceneEffect::Tint: update(static_cast<RenderEffectTint*>(effect), transform); break;
        case SceneEffect::Tritone: update(static_cast<RenderEffectTritone*>(effect), transform); break;
        default: break;
    }
}


bool GlEffect::region(RenderEffect* effect)
{
    switch (effect->type) {
        case SceneEffect::GaussianBlur: return region(static_cast<RenderEffectGaussianBlur*>(effect));
        case SceneEffect::DropShadow : return region(static_cast<RenderEffectDropShadow*>(effect));
        default: return false;
    }
}


bool GlEffect::render(RenderEffect* effect, GlRenderPass* pass, Array<GlRenderTargetPool*>& blendPool)
{
    if (pass->isEmpty()) return false;
    auto vp = pass->getViewport();

    // add render geometry
    const float vdata[] = {-1.0f, +1.0f, +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, -1.0f};
    const uint32_t idata[] = { 0, 1, 2, 0, 2, 3 };
    auto voffset = gpuBuffer->push((void*)vdata, sizeof(vdata));
    auto ioffset = gpuBuffer->pushIndex((void*)idata, sizeof(idata));
    GlRenderTask* output = nullptr;

    if (effect->type == SceneEffect::GaussianBlur) {
        output = render(static_cast<RenderEffectGaussianBlur*>(effect), pass->getFbo(), blendPool, vp, voffset, ioffset);
    } else if (effect->type == SceneEffect::DropShadow) {
        output = render(static_cast<RenderEffectDropShadow*>(effect), pass->getFbo(), blendPool, vp, voffset, ioffset);
    } else {
        output = render(effect, pass->getFbo(), blendPool, vp, voffset, ioffset);
    }

    if (!output) return false;

    pass->addRenderTask(output);
    return true;
}


GlEffect::GlEffect(GlStageBuffer* buffer) : gpuBuffer(buffer)
{
}


GlEffect::~GlEffect()
{
    delete(pBlurV);
    delete(pBlurH);
    delete(pDropShadow);
    delete(pFill);
    delete(pTint);
    delete(pTritone);
}
