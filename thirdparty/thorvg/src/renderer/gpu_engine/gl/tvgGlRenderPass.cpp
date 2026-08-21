/*
 * Copyright (c) 2023 - 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in
 all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "tvgGlCommon.h"
#include "tvgGlRenderPass.h"
#include "tvgGlRenderTask.h"

static Matrix _viewMatrix(const RenderRegion& vp)
{
    Matrix postMatrix = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    translate(&postMatrix, {(float)-vp.sx(), (float)-vp.sy()});

    Matrix mvp = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    mvp.e11 = 2.f / vp.w();
    mvp.e22 = -2.f / vp.h();
    mvp.e13 = -1.f;
    mvp.e23 = 1.f;
    return mvp * postMatrix;
}

GlRenderPass::GlRenderPass(GlRenderTarget* fbo): mFbo(fbo), mTasks(), mDrawDepth(0), mViewMatrix(tvg::identity())
{
    if (mFbo) mViewMatrix = _viewMatrix(mFbo->viewport);
}

GlRenderPass::GlRenderPass(GlRenderPass&& other): mFbo(other.mFbo), mTasks(), mDrawDepth(0), mViewMatrix(other.mViewMatrix)
{
    mTasks.push(other.mTasks);

    other.mTasks.clear();

    mDrawDepth = other.mDrawDepth;
}

GlRenderPass::~GlRenderPass()
{
    if (mTasks.empty()) return;

    ARRAY_FOREACH(p, mTasks) delete(*p);

    mTasks.clear();
}

void GlRenderPass::addRenderTask(GlRenderTask* task)
{
    mTasks.push(task);
}
