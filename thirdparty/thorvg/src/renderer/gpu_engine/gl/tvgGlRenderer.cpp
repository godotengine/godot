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

#include "tvgFill.h"
#include "tvgGlCommon.h"
#include "tvgGlRenderer.h"
#include "tvgGlGpuBuffer.h"
#include "tvgGlRenderTask.h"
#include "tvgGlProgram.h"
#include "tvgGlShaderSrc.h"
#include "tvgRender.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

#define NOISE_LEVEL 0.5f

static int32_t _rendererCnt = -1;
static mutex _rendererMtx;

void GlRenderer::disposeTexture(GLuint texId)
{
    if (!texId) return;
    ScopedLock lock(mDisposed.key);
    mDisposed.textures.push(texId);
}

void GlRenderer::clearDisposes()
{
    if (mDisposed.textures.count > 0) {
        GL_CHECK(glDeleteTextures(mDisposed.textures.count, mDisposed.textures.data));
        mDisposed.textures.clear();
    }

    ARRAY_FOREACH(p, mRenderPassStack)
    delete (*p);
    mRenderPassStack.clear();
    mSolidBatch.clear();
}


void GlRenderer::flush()
{
    clearDisposes();

    mRootTarget.reset();

    ARRAY_FOREACH(p, mComposePool) delete(*p);
    mComposePool.clear();

    ARRAY_FOREACH(p, mBlendPool) delete(*p);
    mBlendPool.clear();

    ARRAY_FOREACH(p, mComposeStack) delete(*p);
    mComposeStack.clear();
}


bool GlRenderer::currentContext()
{
#if defined(__EMSCRIPTEN__)
    const auto targetContext = reinterpret_cast<EMSCRIPTEN_WEBGL_CONTEXT_HANDLE>(mContext);
    if (emscripten_webgl_get_current_context() == targetContext) return true;
    return emscripten_webgl_make_context_current(targetContext) == 0;
#elif defined(_WIN32) && !defined(__CYGWIN__) && defined(THORVG_GL_TARGET_GL)
    if (tvgWglGetCurrentContext() == static_cast<HGLRC>(mContext)) return true;
    return (bool) tvgWglMakeCurrent((HDC)mSurface, static_cast<HGLRC>(mContext));
#elif defined(THORVG_GL_TARGET_GLES)
    if (tvgEglGetCurrentContext() == static_cast<EGLContext>(mContext)) return true;
    if (mDisplay && mSurface) return (bool) tvgEglMakeCurrent((EGLDisplay)mDisplay, (EGLSurface)mSurface, (EGLSurface)mSurface, (EGLContext)mContext);
#endif
    TVGLOG("GL_ENGINE", "Maybe missing currentContext()?");
    return true;
}


GlRenderer::GlRenderer() : mEffect(GlEffect(&mGpuBuffer))
{
}


GlRenderer::~GlRenderer()
{
    if (mContext) currentContext();
    flush();
    mTextures.clear();

    ARRAY_FOREACH(p, mPrograms) delete(*p);

    _rendererMtx.lock();
    --_rendererCnt;
    _rendererMtx.unlock();
}


void GlRenderer::initShaders()
{
    mPrograms.reserve((int)RT_None);

#if 1  //for optimization
    #define LINEAR_TOTAL_LENGTH 2831
    #define RADIAL_TOTAL_LENGTH 5315
    #define BLEND_TOTAL_LENGTH 5096
#else
    #define COMMON_TOTAL_LENGTH strlen(STR_GRADIENT_FRAG_COMMON_VARIABLES) + strlen(STR_GRADIENT_FRAG_COMMON_FUNCTIONS) + 1
    #define LINEAR_TOTAL_LENGTH strlen(STR_LINEAR_GRADIENT_VARIABLES) + strlen(STR_LINEAR_GRADIENT_FUNCTIONS) + strlen(STR_LINEAR_GRADIENT_MAIN) + COMMON_TOTAL_LENGTH
    #define RADIAL_TOTAL_LENGTH strlen(STR_RADIAL_GRADIENT_VARIABLES) + strlen(STR_RADIAL_GRADIENT_FUNCTIONS) + strlen(STR_RADIAL_GRADIENT_MAIN) + COMMON_TOTAL_LENGTH
    #define BLEND_TOTAL_LENGTH strlen(BLEND_SCENE_FRAG_HEADER) + strlen(BLEND_FRAG_LUM_HELPER) + strlen(BLEND_FRAG_SAT_HELPER) + strlen(COLOR_BURN_BLEND_FRAG) + 1
#endif

    char linearGradientFragShader[LINEAR_TOTAL_LENGTH];
    snprintf(linearGradientFragShader, LINEAR_TOTAL_LENGTH, "%s%s%s%s%s",
        STR_GRADIENT_FRAG_COMMON_VARIABLES,
        STR_LINEAR_GRADIENT_VARIABLES,
        STR_GRADIENT_FRAG_COMMON_FUNCTIONS,
        STR_LINEAR_GRADIENT_FUNCTIONS,
        STR_LINEAR_GRADIENT_MAIN
    );

    char radialGradientFragShader[RADIAL_TOTAL_LENGTH];
    snprintf(radialGradientFragShader, RADIAL_TOTAL_LENGTH, "%s%s%s%s%s",
        STR_GRADIENT_FRAG_COMMON_VARIABLES,
        STR_RADIAL_GRADIENT_VARIABLES,
        STR_GRADIENT_FRAG_COMMON_FUNCTIONS,
        STR_RADIAL_GRADIENT_FUNCTIONS,
        STR_RADIAL_GRADIENT_MAIN
    );

    mPrograms.push(new GlProgram(COLOR_VERT_SHADER, COLOR_FRAG_SHADER));
    mPrograms.push(new GlProgram(GRADIENT_VERT_SHADER, linearGradientFragShader));
    mPrograms.push(new GlProgram(GRADIENT_VERT_SHADER, radialGradientFragShader));
    mPrograms.push(new GlProgram(IMAGE_VERT_SHADER, IMAGE_FRAG_SHADER));

    // compose Renderer
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_ALPHA_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_INV_ALPHA_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_LUMA_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_INV_LUMA_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_ADD_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_SUB_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_INTERSECT_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_DIFF_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_LIGHTEN_FRAG_SHADER));
    mPrograms.push(new GlProgram(MASK_VERT_SHADER, MASK_DARKEN_FRAG_SHADER));

    // stencil Renderer
    mPrograms.push(new GlProgram(STENCIL_VERT_SHADER, STENCIL_FRAG_SHADER));

    // blit Renderer
    mPrograms.push(new GlProgram(BLIT_VERT_SHADER, BLIT_FRAG_SHADER));

    // blend programs: image (17) + scene (17) + shape solid (17) + shape linear (17) + shape radial (17)
    for (uint32_t i = 0; i < 85; ++i) mPrograms.push(nullptr);
}

RenderRegion GlRenderer::viewportRegion(const RenderRegion& vp, const RenderRegion& bbox)
{
    auto x = bbox.sx() - vp.sx();
    auto y = bbox.sy() - vp.sy();
    auto w = bbox.sw();
    auto h = bbox.sh();
    auto yGl = vp.sh() - y - h;

    return {{x, yGl}, {x + w, yGl + h}};
}

GlRenderTask* GlRenderer::createPrimitiveTask(RenderTypes type, BlendSource source, const RenderRegion& viewRegion, GlRenderTarget*& dstCopyFbo)
{
    dstCopyFbo = nullptr;

    if (mBlendMethod == BlendMethod::Normal) return new GlRenderTask(mPrograms[type]);

    if (mBlendPool.empty()) mBlendPool.push(new GlRenderTargetPool(surface.w, surface.h));
#if defined(THORVG_GL_TARGET_GL)
    dstCopyFbo = mBlendPool[0]->getRenderTarget(viewRegion);
#else  // TODO: create partial buffer when MSAA is disabled
    dstCopyFbo = mBlendPool[0]->getRenderTarget(currentPass()->getViewport());
#endif

    auto program = getBlendProgram(mBlendMethod, source);
    return new GlDirectBlendTask(program, currentPass()->getFbo(), dstCopyFbo, viewRegion);
}

GlRenderTask* GlRenderer::createStencilTask(GlRenderTask* task, GlStencilMode stencilMode, int32_t depth)
{
    if (stencilMode == GlStencilMode::None) return nullptr;

    auto stencilTask = new GlRenderTask(mPrograms[RT_Stencil], task);
    stencilTask->setDrawDepth(depth);

    return stencilTask;
}

void GlRenderer::bindBlendTarget(GlRenderTask* task, const GlRenderTarget* dstCopyFbo, const RenderRegion& viewRegion, uint32_t binding)
{
    if (!dstCopyFbo) return;

#if defined(THORVG_GL_TARGET_GL)
    float region[] = {float(viewRegion.sx()), float(viewRegion.sy()), float(dstCopyFbo->width), float(dstCopyFbo->height)};
#else  // TODO: create partial buffer when MSAA is disabled
    float region[] = {0.0f, 0.0f, float(dstCopyFbo->width), float(dstCopyFbo->height)};
#endif
    task->addBindResource(GlBindingResource{
        binding,
        task->getProgram()->getUniformBlockIndex("BlendRegion"),
        mGpuBuffer.getBufferId(),
        mGpuBuffer.push(region, 4 * sizeof(float), true),
        4 * sizeof(float),
    });
    task->addBindResource(GlBindingResource{0, dstCopyFbo->colorTex, task->getProgram()->getUniformLocation("uDstTexture")});
}

void GlRenderer::drawPrimitive(GlShape& sdata, const RenderColor& c, RenderUpdateFlag flag, int32_t depth)
{
    auto blendShape = (mBlendMethod != BlendMethod::Normal);
    auto vp = currentPass()->getViewport();
    auto bbox = blendShape ? sdata.geometry.getBounds() : sdata.geometry.viewport;

    bbox.intersect(vp);
    if (bbox.invalid()) return;

    auto viewRegion = viewportRegion(vp, bbox);
    auto stencilMode = sdata.geometry.getStencilMode(flag);

    if (!blendShape && stencilMode == GlStencilMode::None && sdata.clips.empty()) {
        mSolidBatch.draw(*this, sdata, c, depth, viewRegion);
        return;
    }

    if (!sdata.clips.empty()) mSolidBatch.clear();

    GlRenderTarget* dstCopyFbo = nullptr;
    auto task = createPrimitiveTask(RT_Color, BlendSource::Solid, viewRegion, dstCopyFbo);

    task->setViewMatrix(currentPass()->getViewMatrix());
    task->setDrawDepth(depth);

    if (!sdata.geometry.draw(task, &mGpuBuffer, flag)) {
        delete task;
        return;
    }

    auto a = MULTIPLY(c.a, sdata.opacity);
    if (flag & RenderUpdateFlag::Stroke) {
        auto strokeWidth = sdata.geometry.strokeRenderWidth;
        if (strokeWidth < MIN_GL_STROKE_WIDTH) {
            auto alpha = strokeWidth / MIN_GL_STROKE_WIDTH;
            a = MULTIPLY(a, static_cast<uint8_t>(alpha * 255));
        }
    }
    task->setVertexColor(c.r / 255.f, c.g / 255.f, c.b / 255.f, a / 255.f);
    task->setViewport(viewRegion);

    auto stencilTask = createStencilTask(task, stencilMode, depth);
    // Keep BlendRegion on the existing solid-shape blend UBO slot.
    bindBlendTarget(task, dstCopyFbo, viewRegion, 2);

    if (stencilTask) currentPass()->addRenderTask(new GlStencilCoverTask(stencilTask, task, stencilMode));
    else currentPass()->addRenderTask(task);
}

void GlRenderer::drawPrimitive(GlShape& sdata, const Fill* fill, RenderUpdateFlag flag, int32_t depth)
{
    auto blendShape = (mBlendMethod != BlendMethod::Normal);
    auto vp = currentPass()->getViewport();
    auto bbox = blendShape ? sdata.geometry.getBounds() : sdata.geometry.viewport;
    bbox.intersect(vp);
    if (bbox.invalid()) return;

    const Fill::ColorStop* stops = nullptr;
    auto stopCnt = min(fill->colorStops(&stops), static_cast<uint32_t>(MAX_GRADIENT_STOPS));
    if (stopCnt < 2) return;

    GlRenderTarget* dstCopyFbo = nullptr;
    auto radial = fill->type() == Type::RadialGradient;
    auto viewRegion = viewportRegion(vp, bbox);

    RenderTypes taskType = RT_None;
    auto blendSource = BlendSource::LinearGradient;

    if (fill->type() == Type::LinearGradient) {
        taskType = RT_LinGradient;
    } else if (radial) {
        taskType = RT_RadGradient;
        blendSource = BlendSource::RadialGradient;
    } else return;

    auto task = createPrimitiveTask(taskType, blendSource, viewRegion, dstCopyFbo);

    task->setViewMatrix(currentPass()->getViewMatrix());
    task->setDrawDepth(depth);

    if (!sdata.geometry.draw(task, &mGpuBuffer, flag)) {
        delete task;
        return;
    }

    task->setViewport(viewRegion);

    GlStencilMode stencilMode = sdata.geometry.getStencilMode(flag);
    auto stencilTask = createStencilTask(task, stencilMode, depth);

    // transform buffer (inverse fill-space transform)
    float invMat3[GL_MAT3_STD140_SIZE];
    Matrix inv;
    inverse(&fill->transform(), &inv);
    Matrix invShape;
    inverse(&sdata.geometry.matrix, &invShape);
    inv = inv * invShape;
    getMatrix3Std140(inv, invMat3);

    float transformInfo[GL_MAT3_STD140_SIZE];
    memcpy(transformInfo, invMat3, GL_MAT3_STD140_BYTES);
    auto transformOffset = mGpuBuffer.push(transformInfo, sizeof(transformInfo), true);

    task->addBindResource(GlBindingResource{
        0,
        task->getProgram()->getUniformBlockIndex("TransformInfo"),
        mGpuBuffer.getBufferId(),
        transformOffset,
        sizeof(transformInfo),
    });

    auto alpha = sdata.opacity / 255.f;

    if (flag & RenderUpdateFlag::GradientStroke) {
        auto strokeWidth = sdata.geometry.strokeRenderWidth;
        if (strokeWidth < MIN_GL_STROKE_WIDTH) {
            alpha = strokeWidth / MIN_GL_STROKE_WIDTH;
        }
    }

    // gradient block
    GlBindingResource gradientBinding{};
    auto loc = task->getProgram()->getUniformBlockIndex("GradientInfo");

    if (fill->type() == Type::LinearGradient) {
        auto linearFill = static_cast<const LinearGradient*>(fill);

        GlLinearGradientBlock gradientBlock;

        gradientBlock.nStops[1] = NOISE_LEVEL;
        gradientBlock.nStops[2] = static_cast<int32_t>(fill->spread()) * 1.f;
        uint32_t nStops = 0;
        for (uint32_t i = 0; i < stopCnt; ++i) {
            if (i > 0 && gradientBlock.stopPoints[nStops - 1] > stops[i].offset) continue;

            gradientBlock.stopPoints[i] = stops[i].offset;
            gradientBlock.stopColors[i * 4 + 0] = stops[i].r / 255.f;
            gradientBlock.stopColors[i * 4 + 1] = stops[i].g / 255.f;
            gradientBlock.stopColors[i * 4 + 2] = stops[i].b / 255.f;
            gradientBlock.stopColors[i * 4 + 3] = stops[i].a / 255.f * alpha;
            nStops++;
        }
        gradientBlock.nStops[0] = nStops * 1.f;

        float x1, x2, y1, y2;
        linearFill->linear(&x1, &y1, &x2, &y2);

        gradientBlock.startPos[0] = x1;
        gradientBlock.startPos[1] = y1;
        gradientBlock.stopPos[0] = x2;
        gradientBlock.stopPos[1] = y2;

        gradientBinding = GlBindingResource{
            2,
            loc,
            mGpuBuffer.getBufferId(),
            mGpuBuffer.push(&gradientBlock, sizeof(GlLinearGradientBlock), true),
            sizeof(GlLinearGradientBlock),
        };
    } else {
        auto radialFill = static_cast<const RadialGradient*>(fill);

        GlRadialGradientBlock gradientBlock;

        gradientBlock.nStops[1] = NOISE_LEVEL;
        gradientBlock.nStops[2] = static_cast<int32_t>(fill->spread()) * 1.f;

        uint32_t nStops = 0;
        for (uint32_t i = 0; i < stopCnt; ++i) {
            if (i > 0 && gradientBlock.stopPoints[nStops - 1] > stops[i].offset) continue;

            gradientBlock.stopPoints[i] = stops[i].offset;
            gradientBlock.stopColors[i * 4 + 0] = stops[i].r / 255.f;
            gradientBlock.stopColors[i * 4 + 1] = stops[i].g / 255.f;
            gradientBlock.stopColors[i * 4 + 2] = stops[i].b / 255.f;
            gradientBlock.stopColors[i * 4 + 3] = stops[i].a / 255.f * alpha;
            nStops++;
        }
        gradientBlock.nStops[0] = nStops * 1.f;

        float x, y, r, fx, fy, fr;
        radialFill->radial(&x, &y, &r, &fx, &fy, &fr);
        CONST_RADIAL(radialFill)->correct(fx, fy, fr);

        gradientBlock.centerPos[0] = fx;
        gradientBlock.centerPos[1] = fy;
        gradientBlock.centerPos[2] = x;
        gradientBlock.centerPos[3] = y;
        gradientBlock.radius[0] = fr;
        gradientBlock.radius[1] = r;

        gradientBinding = GlBindingResource{
            2,
            loc,
            mGpuBuffer.getBufferId(),
            mGpuBuffer.push(&gradientBlock, sizeof(GlRadialGradientBlock), true),
            sizeof(GlRadialGradientBlock),
        };
    }

    task->addBindResource(gradientBinding);

    // TransformInfo uses slot 0 and GradientInfo uses slot 2, so BlendRegion moves to 3.
    bindBlendTarget(task, dstCopyFbo, viewRegion, 3);

    if (stencilTask) {
        currentPass()->addRenderTask(new GlStencilCoverTask(stencilTask, task, stencilMode));
    } else {
        currentPass()->addRenderTask(task);
    }
}


void GlRenderer::drawClip(Array<RenderData>& clips)
{
    Array<float> identityVertex(4 * 2);
    float left = -1.f;
    float top = 1.f;
    float right = 1.f;
    float bottom = -1.f;

    identityVertex.push(left);
    identityVertex.push(top);
    identityVertex.push(left);
    identityVertex.push(bottom);
    identityVertex.push(right);
    identityVertex.push(top);
    identityVertex.push(right);
    identityVertex.push(bottom);

    Array<uint32_t> identityIndex(6);
    identityIndex.push(0);
    identityIndex.push(1);
    identityIndex.push(2);
    identityIndex.push(2);
    identityIndex.push(1);
    identityIndex.push(3);

    auto identityVertexOffset = mGpuBuffer.push(identityVertex.data, 8 * sizeof(float));
    auto identityIndexOffset = mGpuBuffer.pushIndex(identityIndex.data, 6 * sizeof(uint32_t));

    Array<int32_t> clipDepths(clips.count);
    clipDepths.count = clips.count;

    for (int32_t i = clips.count - 1; i >= 0; i--) {
        clipDepths[i] = currentPass()->nextDrawDepth();
    }

    const auto& vp = currentPass()->getViewport();
    const auto& viewMatrix = currentPass()->getViewMatrix();

    for (uint32_t i = 0; i < clips.count; ++i) {
        auto sdata = static_cast<GlShape*>(clips[i]);
        auto clipTask = new GlRenderTask(mPrograms[RT_Stencil]);
        clipTask->setDrawDepth(clipDepths[i]);
        clipTask->setViewMatrix(viewMatrix);

        auto flag = (sdata->geometry.stroke.vertex.count > 0) ? RenderUpdateFlag::Stroke : RenderUpdateFlag::Path;
        sdata->geometry.draw(clipTask, &mGpuBuffer, flag);

        auto bbox = sdata->geometry.viewport;
        bbox.intersect(vp);

        auto x = bbox.sx() - vp.sx();
        auto y = vp.sh() - (bbox.sy() - vp.sy()) - bbox.sh();
        clipTask->setViewport({{x, y}, {x + bbox.sw(), y + bbox.sh()}});

        auto maskTask = new GlRenderTask(mPrograms[RT_Stencil]);

        maskTask->setDrawDepth(clipDepths[i]);
        maskTask->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), identityVertexOffset});
        maskTask->setDrawRange(identityIndexOffset, 6);
        maskTask->setViewport({{0, 0}, {vp.sw(), vp.sh()}});

        currentPass()->addRenderTask(new GlClipTask(clipTask, maskTask));
    }
}

GlRenderPass* GlRenderer::currentPass()
{
    if (mRenderPassStack.empty()) return nullptr;
    return mRenderPassStack.last();
}

bool GlRenderer::beginComplexBlending(const RenderRegion& vp, RenderRegion bounds)
{
    if (vp.invalid()) return false;

    bounds.intersect(vp);
    if (bounds.invalid()) return false;

    if (mBlendMethod == BlendMethod::Normal) return false;

    if (mBlendPool.empty()) mBlendPool.push(new GlRenderTargetPool(surface.w, surface.h));

    auto blendFbo = mBlendPool[0]->getRenderTarget(bounds);

    mRenderPassStack.push(new GlRenderPass(blendFbo));

    return true;
}

void GlRenderer::endBlendingCompose(GlRenderTask* stencilTask)
{
    auto blendPass = mRenderPassStack.last();
    mRenderPassStack.pop();
    
    blendPass->setDrawDepth(currentPass()->nextDrawDepth());

    auto composeTask = blendPass->endRenderPass<GlComposeTask>(nullptr, currentPass()->getFboId());

    const auto& vp = blendPass->getViewport();
    if (mBlendPool.count < 2) mBlendPool.push(new GlRenderTargetPool(surface.w, surface.h));
#if defined(THORVG_GL_TARGET_GL)
    auto dstCopyFbo = mBlendPool[1]->getRenderTarget(vp);
#else // TODO: create partial buffer when MSAA is disabled        
    auto dstCopyFbo = mBlendPool[1]->getRenderTarget(currentPass()->getViewport());
#endif

    auto x = vp.sx();
    auto y = currentPass()->getViewport().sh() - vp.sy() - vp.sh();
    stencilTask->setViewport({{x, y}, {x + vp.sw(), y + vp.sh()}});

    stencilTask->setDrawDepth(currentPass()->nextDrawDepth());
    stencilTask->setViewMatrix(currentPass()->getViewMatrix());
    
    auto program = getBlendProgram(mBlendMethod, BlendSource::Image);
    auto task = new GlComplexBlendTask(program, currentPass()->getFbo(), dstCopyFbo, stencilTask, composeTask);
    prepareCmpTask(task, vp, blendPass->getFboWidth(), blendPass->getFboHeight());
    task->setDrawDepth(currentPass()->nextDrawDepth());

#if defined(THORVG_GL_TARGET_GLES)
    float region[] = {0.0f, 0.0f, float(dstCopyFbo->width), float(dstCopyFbo->height)};
    task->addBindResource(GlBindingResource{
        0,
        task->getProgram()->getUniformBlockIndex("BlendRegion"),
        mGpuBuffer.getBufferId(),
        mGpuBuffer.push(region, 4 * sizeof(float), true),
        4 * sizeof(float),
    });
#endif

    // src and dst texture
    task->addBindResource(GlBindingResource{1, blendPass->getFbo()->colorTex, task->getProgram()->getUniformLocation("uSrcTexture")});
    task->addBindResource(GlBindingResource{2, dstCopyFbo->colorTex, task->getProgram()->getUniformLocation("uDstTexture")});

    currentPass()->addRenderTask(task);

    delete(blendPass);
}


GlProgram* GlRenderer::getBlendProgram(BlendMethod method, BlendSource source)
{
    // custom blend shaders
    static const char* shaderFunc[17] {
        NORMAL_BLEND_FRAG,
        MULTIPLY_BLEND_FRAG,
        SCREEN_BLEND_FRAG,
        OVERLAY_BLEND_FRAG,
        DARKEN_BLEND_FRAG,
        LIGHTEN_BLEND_FRAG,
        COLOR_DODGE_BLEND_FRAG,
        COLOR_BURN_BLEND_FRAG,
        HARD_LIGHT_BLEND_FRAG,
        SOFT_LIGHT_BLEND_FRAG,
        DIFFERENCE_BLEND_FRAG,
        EXCLUSION_BLEND_FRAG,
        HUE_BLEND_FRAG,
        SATURATION_BLEND_FRAG,
        COLOR_BLEND_FRAG,
        LUMINOSITY_BLEND_FRAG,
        ADD_BLEND_FRAG
    };

    uint32_t methodInd = (uint32_t)method;
    uint32_t shaderInd = methodInd;

    switch (source) {
        case BlendSource::Scene: shaderInd += (uint32_t)RT_Blend_Scene_Normal; break;
        case BlendSource::Image: shaderInd += (uint32_t)RT_Blend_Image_Normal; break;
        case BlendSource::Solid: shaderInd += (uint32_t)RT_ShapeBlend_Solid_Normal; break;
        case BlendSource::LinearGradient: shaderInd += (uint32_t)RT_ShapeBlend_Linear_Normal; break;
        case BlendSource::RadialGradient: shaderInd += (uint32_t)RT_ShapeBlend_Radial_Normal; break;
    }

    if (mPrograms[shaderInd]) return mPrograms[shaderInd];

    const char* lumHelper = "";
    const char* satHelper = "";
    if (method == BlendMethod::Hue) {
        lumHelper = BLEND_FRAG_LUM_HELPER;
        satHelper = BLEND_FRAG_SAT_HELPER;
    } else if ((method == BlendMethod::Saturation) || (method == BlendMethod::Color) || (method == BlendMethod::Luminosity)) {
        lumHelper = BLEND_FRAG_LUM_HELPER;
    }

    const char* vertShader;
    char fragShader[BLEND_TOTAL_LENGTH];

    if (source == BlendSource::Scene || source == BlendSource::Image) {
        vertShader = BLIT_VERT_SHADER;
        const char* header = (source == BlendSource::Scene) ? BLEND_SCENE_FRAG_HEADER : BLEND_IMAGE_FRAG_HEADER;
        snprintf(fragShader, BLEND_TOTAL_LENGTH, "%s%s%s%s", header, lumHelper, satHelper, shaderFunc[methodInd]);
        mPrograms[shaderInd] = new GlProgram(vertShader, fragShader);
        return mPrograms[shaderInd];
    }

    vertShader = (source == BlendSource::Solid) ? COLOR_VERT_SHADER : GRADIENT_VERT_SHADER;
    switch (source) {
        case BlendSource::Solid:
            snprintf(fragShader, BLEND_TOTAL_LENGTH, "%s%s%s%s",
                     BLEND_SHAPE_SOLID_FRAG_HEADER,
                     lumHelper,
                     satHelper,
                     shaderFunc[methodInd]);
            break;
        case BlendSource::LinearGradient:
            snprintf(fragShader, BLEND_TOTAL_LENGTH, "%s%s%s%s%s%s%s%s",
                     STR_GRADIENT_FRAG_COMMON_VARIABLES,
                     STR_LINEAR_GRADIENT_VARIABLES,
                     STR_GRADIENT_FRAG_COMMON_FUNCTIONS,
                     STR_LINEAR_GRADIENT_FUNCTIONS,
                     BLEND_SHAPE_LINEAR_FRAG_HEADER,
                     lumHelper,
                     satHelper,
                     shaderFunc[methodInd]);
            break;
        case BlendSource::RadialGradient:
            snprintf(fragShader, BLEND_TOTAL_LENGTH, "%s%s%s%s%s%s%s%s",
                     STR_GRADIENT_FRAG_COMMON_VARIABLES,
                     STR_RADIAL_GRADIENT_VARIABLES,
                     STR_GRADIENT_FRAG_COMMON_FUNCTIONS,
                     STR_RADIAL_GRADIENT_FUNCTIONS,
                     BLEND_SHAPE_RADIAL_FRAG_HEADER,
                     lumHelper,
                     satHelper,
                     shaderFunc[methodInd]);
            break;
        default:
            TVGERR("RENDERER", "Unsupported blend source! = %d", (int)source);
            break;
    }

    mPrograms[shaderInd] = new GlProgram(vertShader, fragShader);
    return mPrograms[shaderInd];
}


void GlRenderer::prepareBlitTask(GlBlitTask* task)
{
    prepareCmpTask(task, {{0, 0}, {int32_t(surface.w), int32_t(surface.h)}}, surface.w, surface.h);
    task->addBindResource(GlBindingResource{0, task->getColorTexture(), task->getProgram()->getUniformLocation("uSrcTexture")});
}


void GlRenderer::prepareCmpTask(GlRenderTask* task, const RenderRegion& vp, uint32_t cmpWidth, uint32_t cmpHeight)
{
    const auto& passVp = currentPass()->getViewport();
    
    auto taskVp = vp;
    taskVp.intersect(passVp);

    auto x = taskVp.sx() - passVp.sx();
    auto y = taskVp.sy() - passVp.sy();
    auto w = taskVp.sw();
    auto h = taskVp.sh();

    float rw = static_cast<float>(passVp.w());
    float rh = static_cast<float>(passVp.h());

    float l = static_cast<float>(x);
    float t = static_cast<float>(rh - y);
    float r = static_cast<float>(x + w);
    float b = static_cast<float>(rh - y - h);

    // map vp ltrp to -1:1
    float left = (l / rw) * 2.f - 1.f;
    float top = (t / rh) * 2.f - 1.f;
    float right = (r / rw) * 2.f - 1.f;
    float bottom = (b / rh) * 2.f - 1.f;

    float uw = static_cast<float>(w) / static_cast<float>(cmpWidth);
    float uh = static_cast<float>(h) / static_cast<float>(cmpHeight);

    float vertices[4*4] {
        left, top,     0.f, uh,  // left top point
        left, bottom,  0.f, 0.f, // left bottom point
        right, top,    uw, uh,   // right top point
        right, bottom, uw, 0.f   // right bottom point
    };
    uint32_t indices[6]{0, 1, 2, 2, 1, 3};
    uint32_t vertexOffset = mGpuBuffer.push(vertices, sizeof(vertices));
    uint32_t indexOffset = mGpuBuffer.pushIndex(indices, sizeof(indices));

    task->addVertexLayout(GlVertexLayout{0, 2, 4 * sizeof(float), vertexOffset});
    task->addVertexLayout(GlVertexLayout{1, 2, 4 * sizeof(float), vertexOffset + 2 * sizeof(float)});
    task->setDrawRange(indexOffset, 6);
    y = (passVp.sh() - y - h);
    task->setViewport({{x, y}, {x + w, y + h}});
}


void GlRenderer::endRenderPass(RenderCompositor* cmp)
{
    auto glCmp = static_cast<GlCompositor*>(cmp);
    
    // setup masking and blending render pass configurations
    if ((glCmp->flags & (tvg::Blending | tvg::Masking)) == (tvg::Blending | tvg::Masking)) {
        // rearrange render tree
        auto selfPass = mRenderPassStack.last();
        mRenderPassStack.pop();
        auto prevPass = mRenderPassStack.last();
        mRenderPassStack.pop();
        auto maskPass = mRenderPassStack.last();
        mRenderPassStack.pop();
        mRenderPassStack.push(prevPass);
        mRenderPassStack.push(maskPass);
        mRenderPassStack.push(selfPass);
        // setup composition properties
        auto prevCompose = mComposeStack.last();
        auto opacity = glCmp->opacity;
        auto blendMethod = glCmp->blendMethod;
        // self scene task must be masked but not blended
        glCmp->method = prevCompose->method;
        glCmp->opacity = 255;
        glCmp->blendMethod = BlendMethod::Normal;
        // prev scene task must be blended but not masked
        prevCompose->method = MaskMethod::None;
        prevCompose->opacity = opacity;
        prevCompose->blendMethod = blendMethod;
    };

    if (cmp->method != MaskMethod::None) {
        auto selfPass = mRenderPassStack.last();
        mRenderPassStack.pop();

        // mask is pushed first
        auto maskPass = mRenderPassStack.last();
        mRenderPassStack.pop();

        GlProgram* program = nullptr;
        switch(cmp->method) {
            case MaskMethod::Alpha: program = mPrograms[RT_MaskAlpha]; break;
            case MaskMethod::InvAlpha: program = mPrograms[RT_MaskAlphaInv]; break;
            case MaskMethod::Luma: program = mPrograms[RT_MaskLuma]; break;
            case MaskMethod::InvLuma: program = mPrograms[RT_MaskLumaInv]; break;
            case MaskMethod::Add: program = mPrograms[RT_MaskAdd]; break;
            case MaskMethod::Subtract: program = mPrograms[RT_MaskSub]; break;
            case MaskMethod::Intersect: program = mPrograms[RT_MaskIntersect]; break;
            case MaskMethod::Difference: program = mPrograms[RT_MaskDifference]; break;
            case MaskMethod::Lighten: program = mPrograms[RT_MaskLighten]; break;
            case MaskMethod::Darken: program = mPrograms[RT_MaskDarken]; break;
            default: break;
        }
        if (program && !selfPass->isEmpty() && !maskPass->isEmpty()) {
            auto prev_task = maskPass->endRenderPass<GlComposeTask>(nullptr, currentPass()->getFboId());
            prev_task->setDrawDepth(currentPass()->nextDrawDepth());
            prev_task->setRenderSize(glCmp->bbox.w(), glCmp->bbox.h());
            prev_task->setViewport(glCmp->bbox);

            auto compose_task = selfPass->endRenderPass<GlDrawBlitTask>(program, currentPass()->getFboId());
            compose_task->setRenderSize(glCmp->bbox.w(), glCmp->bbox.h());
            compose_task->setPrevTask(prev_task);

            prepareCmpTask(compose_task, glCmp->bbox, selfPass->getFboWidth(), selfPass->getFboHeight());

            compose_task->addBindResource(GlBindingResource{0, selfPass->getTextureId(), program->getUniformLocation("uSrcTexture")});
            compose_task->addBindResource(GlBindingResource{1, maskPass->getTextureId(), program->getUniformLocation("uMaskTexture")});

            compose_task->setDrawDepth(currentPass()->nextDrawDepth());
            compose_task->setParentSize(currentPass()->getViewport().w(), currentPass()->getViewport().h());
            currentPass()->addRenderTask(compose_task);
        }
        delete(selfPass);
        delete(maskPass);
    } else if (glCmp->blendMethod != BlendMethod::Normal) {
        auto renderPass = mRenderPassStack.last();
        mRenderPassStack.pop();

        if (!renderPass->isEmpty()) {
            if (mBlendPool.count < 1) mBlendPool.push(new GlRenderTargetPool(surface.w, surface.h));
            if (mBlendPool.count < 2) mBlendPool.push(new GlRenderTargetPool(surface.w, surface.h));
#if defined(THORVG_GL_TARGET_GL)
            auto dstCopyFbo = mBlendPool[1]->getRenderTarget(renderPass->getViewport());
#else // TODO: create partial buffer when MSAA is disabled
            auto dstCopyFbo = mBlendPool[1]->getRenderTarget(currentPass()->getViewport());
#endif
            // image info
            uint32_t info[4] = {(uint32_t)ColorSpace::ABGR8888, 0, cmp->opacity, 0};

            auto program = getBlendProgram(glCmp->blendMethod, BlendSource::Scene);
            auto task = renderPass->endRenderPass<GlSceneBlendTask>(program, currentPass()->getFboId());
            task->setSrcTarget(currentPass()->getFbo());
            task->setDstCopy(dstCopyFbo);
            task->setRenderSize(glCmp->bbox.w(), glCmp->bbox.h());
            prepareCmpTask(task, glCmp->bbox, renderPass->getFboWidth(), renderPass->getFboHeight());
            task->setDrawDepth(currentPass()->nextDrawDepth());
#if defined(THORVG_GL_TARGET_GLES)
            float region[] = {0.0f, 0.0f, float(dstCopyFbo->width), float(dstCopyFbo->height)};
            task->addBindResource(GlBindingResource{
                1,
                task->getProgram()->getUniformBlockIndex("BlendRegion"),
                mGpuBuffer.getBufferId(),
                mGpuBuffer.push(region, 4 * sizeof(float), true),
                4 * sizeof(float),
            });
#endif
            // info
            task->addBindResource(GlBindingResource{0, task->getProgram()->getUniformBlockIndex("ColorInfo"), mGpuBuffer.getBufferId(), mGpuBuffer.push(info, sizeof(info), true), sizeof(info)});
            // textures
            task->addBindResource(GlBindingResource{0, renderPass->getTextureId(), task->getProgram()->getUniformLocation("uSrcTexture")});
            task->addBindResource(GlBindingResource{1, dstCopyFbo->colorTex, task->getProgram()->getUniformLocation("uDstTexture")});
            task->setParentSize(currentPass()->getViewport().w(), currentPass()->getViewport().h());
            currentPass()->addRenderTask(std::move(task));
        }
        delete(renderPass);
    } else {
        auto renderPass = mRenderPassStack.last();
        mRenderPassStack.pop();

        if (!renderPass->isEmpty()) {
            auto task = renderPass->endRenderPass<GlDrawBlitTask>(mPrograms[RT_Image], currentPass()->getFboId());
            task->setRenderSize(glCmp->bbox.w(), glCmp->bbox.h());
            prepareCmpTask(task, glCmp->bbox, renderPass->getFboWidth(), renderPass->getFboHeight());
            task->setDrawDepth(currentPass()->nextDrawDepth());
            task->setViewMatrix(tvg::identity());

            // image info
            uint32_t info[4] = {(uint32_t)ColorSpace::ABGR8888, 0, cmp->opacity, 0};

            task->addBindResource(GlBindingResource{
                1,
                task->getProgram()->getUniformBlockIndex("ColorInfo"),
                mGpuBuffer.getBufferId(),
                mGpuBuffer.push(info, 4 * sizeof(uint32_t), true),
                4 * sizeof(uint32_t),
            });

            // texture id
            task->addBindResource(GlBindingResource{0, renderPass->getTextureId(), task->getProgram()->getUniformLocation("uTexture")});
            task->setParentSize(currentPass()->getViewport().w(), currentPass()->getViewport().h());
            currentPass()->addRenderTask(std::move(task));
        }
        delete(renderPass);
    }
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

bool GlRenderer::clear()
{
    if (mRootTarget.invalid()) return false;

    mClearBuffer = true;
    return true;
}


bool GlRenderer::target(void* display, void* surface, void* context, int32_t id, uint32_t w, uint32_t h, ColorSpace cs)
{
    //assume the context zero is invalid
    if (!context || w == 0 || h == 0) return false;

    if (mContext) {
        currentContext();
        if (mContext != context) mTextures.clear();
    }

    flush();

    this->surface.stride = w;
    this->surface.w = w;
    this->surface.h = h;
    this->surface.cs = cs;

    mDisplay = display;
    mSurface = surface;
    mContext = context;
    mTargetFboId = static_cast<GLint>(id);

    auto ret = currentContext();

    mRootTarget.viewport = {{0, 0}, {int32_t(this->surface.w), int32_t(this->surface.h)}};
    mRootTarget.init(this->surface.w, this->surface.h, mTargetFboId);

    return ret;
}


bool GlRenderer::sync()
{
    //nothing to be done.
    if (mRenderPassStack.empty()) return true;

    currentContext();

    // Blend function for straight alpha
    GL_CHECK(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));
    GL_CHECK(glEnable(GL_BLEND));
    GL_CHECK(glEnable(GL_SCISSOR_TEST));
    GL_CHECK(glCullFace(GL_FRONT_AND_BACK));
    GL_CHECK(glFrontFace(GL_CCW));
    GL_CHECK(glEnable(GL_DEPTH_TEST));
    GL_CHECK(glDepthFunc(GL_GREATER));

    auto task = mRenderPassStack.first()->endRenderPass<GlBlitTask>(mPrograms[RT_Blit], mTargetFboId);

    prepareBlitTask(task);

    task->mClearBuffer = mClearBuffer;
    task->setTargetViewport({{0, 0}, {int32_t(surface.w), int32_t(surface.h)}});

    if (mGpuBuffer.flushToGPU()) {
        mGpuBuffer.bind();
        task->run();
    }

    mGpuBuffer.unbind();

    GL_CHECK(glDisable(GL_SCISSOR_TEST));

    clearDisposes();

    // Reset clear buffer flag to default (false) after use.    
    mClearBuffer = false; 

    delete task;

    return true;
}


bool GlRenderer::bounds(RenderData data, Point* pt4, const Matrix& m)
{
    if (data) {
        auto sdata = static_cast<GlShape*>(data);
        if (sdata->validStroke) {
            tvg::BBox bbox;
            bbox.init();
            auto& vertexes = sdata->geometry.stroke.vertex;
            if (m == sdata->geometry.matrix) {
                // Common AABB path: stroke vertices are already in world space.
                for (uint32_t i = 0; i < vertexes.count / 2; i++) {
                    Point vert = {vertexes[i * 2 + 0], vertexes[i * 2 + 1]};
                    bbox = { min(bbox.min, vert), max(bbox.max, vert)};
                }
            } else {
                // GL stroke vertices are generated in world space.
                // Normalize to local space first, then remap to the caller-requested space.
                // - OBB path passes m = identity -> result becomes local (caller applies model later).
                const auto inverseModel = sdata->geometry.inverseMatrix();

                for (uint32_t i = 0; i < vertexes.count / 2; i++) {
                    Point vert = {vertexes[i * 2 + 0], vertexes[i * 2 + 1]};
                    vert *= (*inverseModel) * m;
                    bbox = { min(bbox.min, vert), max(bbox.max, vert)};
                }
            }
            pt4[0] = bbox.min;
            pt4[1] = {bbox.max.x, bbox.min.y};
            pt4[2] = bbox.max;
            pt4[3] = {bbox.min.x, bbox.max.y};
            return true;
        }
    }
    return false;
}


RenderRegion GlRenderer::region(RenderData data)
{
    if (!data) return {};
    auto shape = reinterpret_cast<GlShape*>(data);
    return shape->geometry.getBounds();
}


bool GlRenderer::preRender()
{
    if (mRootTarget.invalid()) return false;

    currentContext();
    if (mPrograms.empty()) initShaders();
    mRenderPassStack.push(new GlRenderPass(&mRootTarget));

    return true;
}


bool GlRenderer::postRender()
{
    return true;
}


RenderCompositor* GlRenderer::target(const RenderRegion& region, TVG_UNUSED ColorSpace cs, TVG_UNUSED CompositionFlag flags)
{
    auto vp = region;
    if (currentPass()->isEmpty()) return nullptr;

    vp.intersect(currentPass()->getViewport());

    mComposeStack.push(new GlCompositor(vp, flags));
    return mComposeStack.last();
}


bool GlRenderer::beginComposite(RenderCompositor* cmp, MaskMethod method, uint8_t opacity)
{
    if (!cmp) return false;

    auto glCmp = static_cast<GlCompositor*>(cmp);
    glCmp->method = method;
    glCmp->opacity = opacity;
    glCmp->blendMethod = mBlendMethod;

    uint32_t index = mRenderPassStack.count - 1;
    if (index >= mComposePool.count) mComposePool.push( new GlRenderTargetPool(surface.w, surface.h));
    
    if (glCmp->bbox.valid()) mRenderPassStack.push(new GlRenderPass(mComposePool[index]->getRenderTarget(glCmp->bbox)));
    else mRenderPassStack.push(new GlRenderPass(nullptr));

    return true;
}


bool GlRenderer::endComposite(RenderCompositor* cmp)
{
    if (mComposeStack.empty()) return false;
    if (mComposeStack.last() != cmp) return false;

    // end current render pass;
    auto curCmp  = mComposeStack.last();
    mComposeStack.pop();

    assert(cmp == curCmp);

    endRenderPass(curCmp);

    delete(curCmp);

    return true;
}


void GlRenderer::prepare(RenderEffect* effect, const Matrix& transform)
{
    // we must be sure, that we have intermediate FBOs
    if (mBlendPool.count < 1) mBlendPool.push(new GlRenderTargetPool(surface.w, surface.h));
    if (mBlendPool.count < 2) mBlendPool.push(new GlRenderTargetPool(surface.w, surface.h));

    mEffect.update(effect, transform);
}


bool GlRenderer::region(RenderEffect* effect)
{
    return mEffect.region(effect);
}


bool GlRenderer::render(TVG_UNUSED RenderCompositor* cmp, const RenderEffect* effect, TVG_UNUSED bool direct)
{
    return mEffect.render(const_cast<RenderEffect*>(effect), currentPass(), mBlendPool);
}


void GlRenderer::dispose(RenderEffect* effect)
{
    tvg::free(effect->rd);
    effect->rd = nullptr;
}


ColorSpace GlRenderer::colorSpace()
{
    return surface.cs;
}


const RenderSurface* GlRenderer::mainSurface()
{
    return &surface;
}


bool GlRenderer::blend(BlendMethod method)
{
    if (method == mBlendMethod) return true;

    mBlendMethod = (method == BlendMethod::Composition ? BlendMethod::Normal : method);

    return true;
}


bool GlRenderer::renderImage(void* data)
{
    auto sdata = static_cast<GlShape*>(data);
    if (!sdata) return false;

    if (currentPass()->isEmpty() || !sdata->validFill) return true;

    auto vp = currentPass()->getViewport();
    auto bbox = sdata->geometry.viewport;
    bbox.intersect(vp);
    if (bbox.invalid()) return true;

    auto x = bbox.sx() - vp.sx();
    auto y = bbox.sy() - vp.sy();
    auto drawDepth = currentPass()->nextDrawDepth();

    if (!sdata->clips.empty()) drawClip(sdata->clips);

    auto task = new GlRenderTask(mPrograms[RT_Image]);
    task->setDrawDepth(drawDepth);

    if (!sdata->geometry.draw(task, &mGpuBuffer, RenderUpdateFlag::Image)) {
        delete task;
        return true;
    }

    bool complexBlend = beginComplexBlending(bbox, sdata->geometry.getBounds());
    if (complexBlend) vp = currentPass()->getViewport();
    task->setViewMatrix(currentPass()->getViewMatrix());

    // image info
    uint32_t info[4] = {(uint32_t)sdata->texColorSpace, sdata->texFlipY, sdata->opacity, 0};

    task->addBindResource(GlBindingResource{
        1,
        task->getProgram()->getUniformBlockIndex("ColorInfo"),
        mGpuBuffer.getBufferId(),
        mGpuBuffer.push(info, 4 * sizeof(uint32_t), true),
        4 * sizeof(uint32_t),
    });

    // texture id
    task->addBindResource(GlBindingResource{0, sdata->texId, task->getProgram()->getUniformLocation("uTexture")});

    y = vp.sh() - y - bbox.sh();
    auto x2 = x + bbox.sw();
    auto y2 = y + bbox.sh();

    task->setViewport({{x, y}, {x2, y2}});

    currentPass()->addRenderTask(task);

    if (complexBlend) {
        auto task = new GlRenderTask(mPrograms[RT_Stencil]);
        sdata->geometry.draw(task, &mGpuBuffer, RenderUpdateFlag::Image);
        endBlendingCompose(task);
    }

    return true;
}


bool GlRenderer::renderShape(RenderData data)
{
    auto sdata = static_cast<GlShape*>(data);
    if (currentPass()->isEmpty() || (!sdata->validFill && !sdata->validStroke)) return true;

    auto bbox = sdata->geometry.viewport;
    bbox.intersect(currentPass()->getViewport());
    if (bbox.invalid()) return true;

    int32_t drawDepth1 = 0, drawDepth2 = 0;
    if (sdata->validFill) drawDepth1 = currentPass()->nextDrawDepth();
    if (sdata->validStroke) drawDepth2 = currentPass()->nextDrawDepth();

    if (!sdata->clips.empty()) drawClip(sdata->clips);

    auto processFill = [&]() {
        if (sdata->validFill) {
            if (const auto& gradient = sdata->rshape->fill) {
                drawPrimitive(*sdata, gradient, RenderUpdateFlag::Gradient, drawDepth1);
            } else if (sdata->rshape->color.a > 0) {
                drawPrimitive(*sdata, sdata->rshape->color, RenderUpdateFlag::Color, drawDepth1);
            }
        }
    };

    auto processStroke = [&]() {
        if (sdata->validStroke) {
            if (const auto& gradient = sdata->rshape->strokeFill()) {
                drawPrimitive(*sdata, gradient, RenderUpdateFlag::GradientStroke, drawDepth2);
            } else if (sdata->rshape->stroke->color.a > 0) {
                drawPrimitive(*sdata, sdata->rshape->stroke->color, RenderUpdateFlag::Stroke, drawDepth2);
            }
        }
    };

    if (sdata->rshape->strokeFirst()) {
        processStroke();
        processFill();
    } else {
        processFill();
        processStroke();
    }

    return true;
}


void GlRenderer::dispose(RenderData data)
{
    auto sdata = static_cast<GlShape*>(data);
    if (!sdata) return;
    auto ownsTexture = sdata->texId && (sdata->texStamp == mTextures.stamp);
    if (ownsTexture) disposeTexture(mTextures.release(sdata->texSource, sdata->texFilter, sdata->texId));
    delete sdata;
}

RenderData GlRenderer::prepare(RenderSurface* image, RenderData data, const Matrix& transform, const Array<RenderData>& clips, uint8_t opacity, FilterMethod filter, RenderUpdateFlag flags)
{
    //TODO: redefine GlImage.
    if (opacity == 0) return data;

    auto sdata = static_cast<GlShape*>(data);
    if (!sdata) sdata = new GlShape;

    auto cacheStale = sdata->texId && (sdata->texStamp != mTextures.stamp);
    if (flags == RenderUpdateFlag::None && !cacheStale) return data;

    sdata->validFill = false;

    sdata->viewWd = static_cast<float>(surface.w);
    sdata->viewHt = static_cast<float>(surface.h);

    auto sourceChanged = (sdata->texSource != image) || (sdata->texFilter != filter);
    if (sdata->texId == 0 || sourceChanged || cacheStale) {
        auto ownsTexture = sdata->texId && (sdata->texStamp == mTextures.stamp);
        if (ownsTexture) disposeTexture(mTextures.release(sdata->texSource, sdata->texFilter, sdata->texId));
        sdata->texId = mTextures.retain(image, filter);
        sdata->texSource = image;
        sdata->texFilter = filter;
        sdata->texStamp = mTextures.stamp;
        sdata->geometry = GlGeometry();
    }

    sdata->texColorSpace = image->cs;
    sdata->texFlipY = 1;
    sdata->opacity = opacity;
    sdata->geometry.setMatrix(transform);
    sdata->geometry.viewport = vport;
    sdata->geometry.tesselateImage(image);
    sdata->validFill = true;

    if (flags & RenderUpdateFlag::Clip) {
        sdata->clips.clear();
        sdata->clips.push(clips);
    }

    return sdata;
}

RenderData GlRenderer::prepare(const RenderShape& rshape, RenderData data, const Matrix& transform, const Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags, bool clipper)
{
    auto sdata = static_cast<GlShape*>(data);
    if (!sdata) {
        sdata = new GlShape;
        sdata->rshape = &rshape;
        flags = RenderUpdateFlag::All;
    }

    if ((opacity == 0 && !clipper) || flags == RenderUpdateFlag::None) return sdata;

    sdata->viewWd = static_cast<float>(surface.w);
    sdata->viewHt = static_cast<float>(surface.h);
    sdata->opacity = opacity;

    if (flags & RenderUpdateFlag::Path) sdata->geometry = GlGeometry();

    sdata->geometry.setMatrix(transform);
    sdata->geometry.viewport = vport;
    if (flags & (RenderUpdateFlag::Path | RenderUpdateFlag::Transform)) sdata->geometry.prepare(rshape);
    
    //TODO: Please precisely update tessellation not to update only if the color is changed.
    if (flags & (RenderUpdateFlag::Color | RenderUpdateFlag::Gradient | RenderUpdateFlag::Transform | RenderUpdateFlag::Path)) {
        sdata->validFill = false;
        float opacityMultiplier = 1.0f;
        if (sdata->geometry.tesselateShape(*(sdata->rshape), &opacityMultiplier)) {
            sdata->opacity *= opacityMultiplier;
            sdata->validFill = true;
        }
    }

    //TODO: Please precisely update tessellation not to update only if the color is changed.
    if (flags & (RenderUpdateFlag::Color | RenderUpdateFlag::Stroke | RenderUpdateFlag::GradientStroke | RenderUpdateFlag::Transform | RenderUpdateFlag::Path)) {
        sdata->validStroke = false;
        if (sdata->geometry.tesselateStroke(*(sdata->rshape))) sdata->validStroke = true;
    }

    if (flags & RenderUpdateFlag::Clip) {
        sdata->clips.clear();
        sdata->clips.push(clips);
    }

    return sdata;
}


bool GlRenderer::preUpdate()
{
    if (mRootTarget.invalid()) return false;

    currentContext();
    return true;
}


bool GlRenderer::postUpdate()
{
    return true;
}


void GlRenderer::damage(TVG_UNUSED RenderData rd, TVG_UNUSED const RenderRegion& region)
{
    //TODO
}


bool GlRenderer::partial(bool disable)
{
    //TODO
    return false;
}


bool GlRenderer::intersectsShape(RenderData data, TVG_UNUSED const RenderRegion& region)
{
    if (!data) return false;
    auto shape = (GlShape*)data;
    const auto& bbox = shape->geometry.getBounds();
    if (region.intersected(bbox)) {
        if (region.contained(bbox)) return true;
        GlIntersector intersector;
        return intersector.intersectShape(RenderRegion::intersect(region, bbox), shape);
    }
    return false;
}


bool GlRenderer::intersectsImage(RenderData data, TVG_UNUSED const RenderRegion& region)
{
    if (!data) return false;
    auto shape = (GlShape*)data;
    const auto& bbox = shape->geometry.getBounds();
    if (region.intersected(bbox)) {
        if (region.contained(bbox)) return true;
        GlIntersector intersector;
        if (intersector.intersectImage(RenderRegion::intersect(region, bbox), shape)) return true;
    }
    return false;
}


bool GlRenderer::term()
{
    _rendererMtx.lock();

    if (_rendererCnt > 0) {
        _rendererMtx.unlock();
        return false;
    }

    glTerm();

    _rendererCnt = -1;
    _rendererMtx.unlock();

    return true;
}


GlRenderer* GlRenderer::gen(TVG_UNUSED uint32_t threads, TVG_UNUSED EngineOption op)
{
    //initialize engine
    _rendererMtx.lock();
    if (_rendererCnt == -1) {
        if (!glInit()) {
            TVGERR("GL_ENGINE", "Failed GL initialization!");
            _rendererMtx.unlock();
            return nullptr;
        }    
        _rendererCnt = 0;
    }
    ++_rendererCnt;
    _rendererMtx.unlock();

    return new GlRenderer;
}
