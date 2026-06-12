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

#ifndef _TVG_GL_EFFECT_H_
#define _TVG_GL_EFFECT_H_

#include "tvgRender.h"
#include "tvgGlProgram.h"


class GlEffect
{
private:
    GlStageBuffer* gpuBuffer; //shared resource with GlRenderer

    GlProgram* pBlurV{};
    GlProgram* pBlurH{};
    GlProgram* pDropShadow{};
    GlProgram* pFill{};
    GlProgram* pTint{};
    GlProgram* pTritone{};

    void update(RenderEffectGaussianBlur* effect, const Matrix& transform);
    void update(RenderEffectDropShadow* effect, const Matrix& transform);
    void update(RenderEffectFill* effect, const Matrix& transform);
    void update(RenderEffectTint* effect, const Matrix& transform);
    void update(RenderEffectTritone* effect, const Matrix& transform);

    bool region(RenderEffectGaussianBlur* effect);
    bool region(RenderEffectDropShadow* effect);

    GlRenderTask* render(RenderEffectGaussianBlur* effect, GlRenderTarget* dstFbo, Array<GlRenderTargetPool*>& blendPool, const RenderRegion& vp, uint32_t voffset, uint32_t ioffset);
    GlRenderTask* render(RenderEffectDropShadow* effect, GlRenderTarget* dstFbo, Array<GlRenderTargetPool*>& blendPool, const RenderRegion& vp, uint32_t voffset, uint32_t ioffset);
    GlRenderTask* render(RenderEffect* effect, GlRenderTarget* dstFbo, Array<GlRenderTargetPool*>& blendPool, const RenderRegion& vp, uint32_t voffset, uint32_t ioffset);

public:
    GlEffect(GlStageBuffer* buffer);
    ~GlEffect();

    void update(RenderEffect* effect, const Matrix& transform);
    bool region(RenderEffect* effect);
    bool render(RenderEffect* effect, GlRenderPass* pass, Array<GlRenderTargetPool*>& blendPool);
};

#endif /* _TVG_GL_EFFECT_H_ */
