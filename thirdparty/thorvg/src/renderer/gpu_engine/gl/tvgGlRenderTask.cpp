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

#include "tvgGlRenderTask.h"
#include "tvgGlProgram.h"
#include "tvgGlRenderPass.h"

#if !defined(THORVG_GL_TARGET_GL)
static void clearColorTarget(uint32_t width, uint32_t height)
{
    GL_CHECK(glScissor(0, 0, width, height));
    GL_CHECK(glClearColor(0, 0, 0, 0));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
}
#endif

/************************************************************************/
/* GlRenderTask Class Implementation                                    */
/************************************************************************/

GlRenderTask::GlRenderTask(GlProgram* program, GlRenderTask* other): mProgram(program)
{
    mVertexLayout.push(other->mVertexLayout);
    mViewport = other->mViewport;
    mIndexOffset = other->mIndexOffset;
    mIndexCount = other->mIndexCount;
    mViewMatrix = other->mViewMatrix;
    mUseViewMatrix = other->mUseViewMatrix;
}


void GlRenderTask::run()
{
    // bind shader
    mProgram->load();

    int32_t dLoc = mProgram->getUniformLocation("uDepth");
    if (dLoc >= 0) {
        // fixme: prevent compiler warning: macro expands to multiple statements [-Wmultistatement-macros]
        GL_CHECK(glUniform1f(dLoc, mDrawDepth));
    }

    int32_t vLoc = mProgram->getUniformLocation("uViewMatrix");
    if (vLoc >= 0) {
        const auto& viewMatrix = mUseViewMatrix ? mViewMatrix : tvg::identity();
        float viewMat3[9];
        getMatrix3(viewMatrix, viewMat3);
        GL_CHECK(glUniformMatrix3fv(vLoc, 1, GL_FALSE, viewMat3));
    }

    // setup scissor rect
    GL_CHECK(glScissor(mViewport.sx(), mViewport.sy(), mViewport.sw(), mViewport.sh()));

    if (mUseVertexColor) {
        GL_CHECK(glDisableVertexAttribArray(1));
        GL_CHECK(glVertexAttrib4f(1, mVertexColor[0], mVertexColor[1], mVertexColor[2], mVertexColor[3]));
    }

    GLint defaultArrayBuffer = 0;
    GL_CHECK(glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &defaultArrayBuffer));

    // setup attribute layout
    for (uint32_t i = 0; i < mVertexLayout.count; i++) {
        const auto &layout = mVertexLayout[i];
        auto sourceBuffer = layout.arrayBufferId ? layout.arrayBufferId : static_cast<GLuint>(defaultArrayBuffer);
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, sourceBuffer));
        GL_CHECK(glEnableVertexAttribArray(layout.index));
        GL_CHECK(glVertexAttribPointer(layout.index, layout.size, layout.type,
                                       layout.normalized, layout.stride,
                                       reinterpret_cast<void*>(layout.offset)));
    }

    // binding uniforms
    for (uint32_t i = 0; i < mBindingResources.count; i++) {
        const auto& binding = mBindingResources[i];
        if (binding.type == GlBindingType::kTexture) {
            GL_CHECK(glActiveTexture(GL_TEXTURE0 + binding.bindPoint));
            GL_CHECK(glBindTexture(GL_TEXTURE_2D, binding.resourceId));

            mProgram->setUniform1Value(binding.location, 1, (int32_t*)&binding.bindPoint);
        } else if (binding.type == GlBindingType::kUniformBuffer) {

            GL_CHECK(glUniformBlockBinding(mProgram->getProgramId(), binding.location, binding.bindPoint));
            GL_CHECK(glBindBufferRange(GL_UNIFORM_BUFFER, binding.bindPoint, binding.resourceId,
                                       binding.bufferOffset, binding.bufferRange));
        }
    }

    GL_CHECK(glDrawElements(GL_TRIANGLES, mIndexCount, GL_UNSIGNED_INT, reinterpret_cast<void*>(mIndexOffset)));

    // setup attribute layout
    for (uint32_t i = 0; i < mVertexLayout.count; i++) {
        const auto &layout = mVertexLayout[i];
        GL_CHECK(glDisableVertexAttribArray(layout.index));
    }

    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(defaultArrayBuffer)));
}


void GlRenderTask::addVertexLayout(const GlVertexLayout &layout)
{
    mVertexLayout.push(layout);
}

void GlRenderTask::setVertexColor(float r, float g, float b, float a)
{
    mUseVertexColor = true;
    mVertexColor[0] = r;
    mVertexColor[1] = g;
    mVertexColor[2] = b;
    mVertexColor[3] = a;
}

void GlRenderTask::addBindResource(const GlBindingResource &binding)
{
    mBindingResources.push(binding);
}


void GlRenderTask::setDrawRange(uint32_t offset, uint32_t count)
{
    mIndexOffset = offset;
    mIndexCount = count;
}


void GlRenderTask::setViewport(const RenderRegion &viewport)
{
    mViewport = viewport;
    if (mViewport.max.x < mViewport.min.x) mViewport.max.x = mViewport.min.x;
    if (mViewport.max.y < mViewport.min.y) mViewport.max.y = mViewport.min.y;
}


/************************************************************************/
/* GlStencilCoverTask Class Implementation                              */
/************************************************************************/

GlStencilCoverTask::GlStencilCoverTask(GlRenderTask* stencil, GlRenderTask* cover, GlStencilMode mode)
 :GlRenderTask(nullptr), mStencilTask(stencil), mCoverTask(cover), mStencilMode(mode)
 {

 }


GlStencilCoverTask::~GlStencilCoverTask()
{
    delete mStencilTask;
    delete mCoverTask;
}


void GlStencilCoverTask::run()
{
    GL_CHECK(glEnable(GL_STENCIL_TEST));

    if (mStencilMode == GlStencilMode::Stroke) {
        GL_CHECK(glStencilFunc(GL_NOTEQUAL, 0x1, 0xFF));
        GL_CHECK(glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE));
    } else {
        GL_CHECK(glStencilFuncSeparate(GL_FRONT, GL_ALWAYS, 0x0, 0xFF));
        GL_CHECK(glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_INCR_WRAP));

        GL_CHECK(glStencilFuncSeparate(GL_BACK, GL_ALWAYS, 0x0, 0xFF));
        GL_CHECK(glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_DECR_WRAP));
    }
    GL_CHECK(glColorMask(0, 0, 0, 0));

    mStencilTask->run();

    if (mStencilMode == GlStencilMode::FillEvenOdd) {
        GL_CHECK(glStencilFunc(GL_NOTEQUAL, 0x00, 0x01));
        GL_CHECK(glStencilOp(GL_REPLACE, GL_KEEP, GL_REPLACE));
    } else {
        GL_CHECK(glStencilFunc(GL_NOTEQUAL, 0x0, 0xFF));
        GL_CHECK(glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE));
    }

    GL_CHECK(glColorMask(1, 1, 1, 1));

    mCoverTask->run();

    GL_CHECK(glDisable(GL_STENCIL_TEST));
}


void GlStencilCoverTask::normalizeDrawDepth(int32_t maxDepth)
{
    mCoverTask->normalizeDrawDepth(maxDepth);
    mStencilTask->normalizeDrawDepth(maxDepth);
}


/************************************************************************/
/* GlComposeTask Class Implementation                                   */
/************************************************************************/

GlComposeTask::GlComposeTask(GlProgram* program, GLuint target, GlRenderTarget* fbo, Array<GlRenderTask*>&& tasks)
 :GlRenderTask(program) ,mTargetFbo(target), mFbo(fbo), mTasks()
{
    mTasks.push(tasks);
    tasks.clear();
}


GlComposeTask::~GlComposeTask()
{
    ARRAY_FOREACH(p, mTasks) delete(*p);
    mTasks.clear();
}


void GlComposeTask::run()
{
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, getSelfFbo()));

    // we must clear all area of fbo
    GL_CHECK(glViewport(0, 0, mFbo->width, mFbo->height));
    GL_CHECK(glScissor(0, 0, mFbo->width, mFbo->height));
    GL_CHECK(glClearColor(0, 0, 0, 0));
    GL_CHECK(glClearStencil(0));
#ifdef THORVG_GL_TARGET_GLES
    GL_CHECK(glClearDepthf(0.0));
#else
    GL_CHECK(glClearDepth(0.0));
#endif
    GL_CHECK(glDepthMask(1));

    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
    GL_CHECK(glDepthMask(0));

    GL_CHECK(glViewport(0, 0, mRenderWidth, mRenderHeight));
    GL_CHECK(glScissor(0, 0, mRenderWidth, mRenderHeight));

    ARRAY_FOREACH(p, mTasks) {
        (*p)->run();
    }

#if defined(THORVG_GL_TARGET_GLES)
    // only OpenGLES has tiled base framebuffer and discard function
    GLenum attachments[2] = {GL_STENCIL_ATTACHMENT, GL_DEPTH_ATTACHMENT };
    GL_CHECK(glInvalidateFramebuffer(GL_FRAMEBUFFER, 2, attachments));
#endif
    // reset scissor box
    GL_CHECK(glScissor(0, 0, mFbo->width, mFbo->height));
    onResolve();
}


GLuint GlComposeTask::getSelfFbo()
{
    return mFbo->fbo;
}


GLuint GlComposeTask::getResolveFboId()
{
    return mFbo->resolvedFbo;
}


void GlComposeTask::onResolve()
{
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, getSelfFbo()));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, getResolveFboId()));
    GL_CHECK(glBlitFramebuffer(0, 0, mRenderWidth, mRenderHeight, 0, 0, mRenderWidth, mRenderHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}


/************************************************************************/
/* GlBlitTask Class Implementation                                      */
/************************************************************************/

GlBlitTask::GlBlitTask(GlProgram* program, GLuint target, GlRenderTarget* fbo, Array<GlRenderTask*>&& tasks)
 : GlComposeTask(program, target, fbo, std::move(tasks)), mColorTex(fbo->colorTex)
{
}


void GlBlitTask::run()
{
    GlComposeTask::run();

    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, getTargetFbo()));
    GL_CHECK(glViewport(mTargetViewport.x(), mTargetViewport.y(), mTargetViewport.w(), mTargetViewport.h()));

    if (mClearBuffer) {
        GL_CHECK(glClearColor(0, 0, 0, 0));
        GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
    }

    GL_CHECK(glDisable(GL_DEPTH_TEST));
    // make sure the blending is correct
    GL_CHECK(glEnable(GL_BLEND));
    GL_CHECK(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

    GlRenderTask::run();
}


/************************************************************************/
/* GlDrawBlitTask Class Implementation                                  */
/************************************************************************/


GlDrawBlitTask::GlDrawBlitTask(GlProgram* program, GLuint target, GlRenderTarget* fbo, Array<GlRenderTask*>&& tasks)
 : GlComposeTask(program, target, fbo, std::move(tasks))
{
}


GlDrawBlitTask::~GlDrawBlitTask()
{
    if (mPrevTask) delete mPrevTask;
}


void GlDrawBlitTask::run()
{
    if (mPrevTask) mPrevTask->run();

    GlComposeTask::run();

    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, getTargetFbo()));

    GL_CHECK(glViewport(0, 0, mParentWidth, mParentHeight));
    GL_CHECK(glScissor(0, 0, mParentWidth, mParentHeight));
    GlRenderTask::run();
}


/************************************************************************/
/* GlSceneBlendTask Class Implementation                                  */
/************************************************************************/


GlSceneBlendTask::GlSceneBlendTask(GlProgram* program, GLuint target, GlRenderTarget* fbo, Array<GlRenderTask*>&& tasks)
 : GlComposeTask(program, target, fbo, std::move(tasks))
{
}


GlSceneBlendTask::~GlSceneBlendTask()
{
}


void GlSceneBlendTask::run()
{
    GlComposeTask::run();

    const auto& vp = getViewport();
    const auto width = mSrcFbo->width;
    const auto height = mSrcFbo->height;
    if (width <= 0 || height <= 0) return;


#if defined(THORVG_GL_TARGET_GL)
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, getTargetFbo()));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo->resolvedFbo));
    GL_CHECK(glViewport(0, 0, mDstCopyFbo->width, mDstCopyFbo->height));
    GL_CHECK(glScissor(0, 0, mDstCopyFbo->width, mDstCopyFbo->height));
    GL_CHECK(glBlitFramebuffer(vp.min.x, vp.min.y, vp.max.x, vp.max.y, 0, 0, vp.w(), vp.h(), GL_COLOR_BUFFER_BIT, GL_LINEAR));
#else // TODO: create partial buffer when MSAA is disabled
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo->resolvedFbo));
    if (vp.min.x != 0 || vp.min.y != 0 || mDstCopyFbo->width != static_cast<uint32_t>(vp.w()) || mDstCopyFbo->height != static_cast<uint32_t>(vp.h())) clearColorTarget(mDstCopyFbo->width, mDstCopyFbo->height);
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mSrcFbo->fbo));
    GL_CHECK(glViewport(0, 0, width, height));
    GL_CHECK(glScissor(vp.min.x, vp.min.y, vp.w(), vp.h()));
    GL_CHECK(glBlitFramebuffer(vp.min.x, vp.min.y, vp.max.x, vp.max.y, vp.min.x, vp.min.y, vp.max.x, vp.max.y, GL_COLOR_BUFFER_BIT, GL_NEAREST));
#endif

    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, getTargetFbo()));
    GL_CHECK(glViewport(0, 0, mParentWidth, mParentHeight));
    GL_CHECK(glScissor(0, 0, mParentWidth, mParentHeight));

    GL_CHECK(glDisable(GL_DEPTH_TEST));
    GL_CHECK(glBlendFunc(GL_ONE, GL_ZERO));
    GlRenderTask::run();
    GL_CHECK(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));
    GL_CHECK(glEnable(GL_DEPTH_TEST));
}


/************************************************************************/
/* GlClipTask Class Implementation                                      */
/************************************************************************/

GlClipTask::GlClipTask(GlRenderTask* clip, GlRenderTask* mask)
 :GlRenderTask(nullptr), mClipTask(clip), mMaskTask(mask) {}


GlClipTask::~GlClipTask()
{
    delete mClipTask;
    delete mMaskTask;
}


void GlClipTask::run()
{
    GL_CHECK(glEnable(GL_STENCIL_TEST));
    GL_CHECK(glColorMask(0, 0, 0, 0));
    // draw clip path as normal stencil mask
    GL_CHECK(glStencilFuncSeparate(GL_FRONT, GL_ALWAYS, 0x1, 0xFF));
    GL_CHECK(glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_INCR_WRAP));

    GL_CHECK(glStencilFuncSeparate(GL_BACK, GL_ALWAYS, 0x1, 0xFF));
    GL_CHECK(glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_DECR_WRAP));

    mClipTask->run();


    // draw clip mask
    GL_CHECK(glDepthMask(1));
    GL_CHECK(glStencilFunc(GL_EQUAL, 0x0, 0xFF));
    GL_CHECK(glStencilOp(GL_REPLACE, GL_KEEP, GL_REPLACE));

    mMaskTask->run();

    GL_CHECK(glColorMask(1, 1, 1, 1));
    GL_CHECK(glDepthMask(0));
    GL_CHECK(glDisable(GL_STENCIL_TEST));
}


void GlClipTask::normalizeDrawDepth(int32_t maxDepth)
{
    mClipTask->normalizeDrawDepth(maxDepth);
    mMaskTask->normalizeDrawDepth(maxDepth);
}

/************************************************************************/
/* GlDirectBlendTask Class Implementation                               */
/************************************************************************/

GlDirectBlendTask::GlDirectBlendTask(GlProgram* program, GlRenderTarget* dstFbo, GlRenderTarget* dstCopyFbo, const RenderRegion& copyRegion)
    : GlRenderTask(program), mDstFbo(dstFbo), mDstCopyFbo(dstCopyFbo), mCopyRegion(copyRegion)
{
}


void GlDirectBlendTask::run()
{
    auto width = mCopyRegion.w();
    auto height = mCopyRegion.h();
    if (width <= 0 || height <= 0) return;
    auto x = mCopyRegion.sx();
    auto y = mCopyRegion.sy();
    const auto fboW = mDstFbo->width;
    const auto fboH = mDstFbo->height;
    if (fboW <= 0 || fboH <= 0) return;

    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mDstFbo->fbo));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo->resolvedFbo));

#if defined(THORVG_GL_TARGET_GL)
    GL_CHECK(glViewport(0, 0, width, height));
    GL_CHECK(glScissor(0, 0, width, height));
    GL_CHECK(glBlitFramebuffer(x, y, x + width, y + height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR));
#else // TODO: create partial buffer when MSAA is disabled
    if (x != 0 || y != 0 || mDstCopyFbo->width != static_cast<uint32_t>(width) || mDstCopyFbo->height != static_cast<uint32_t>(height)) clearColorTarget(mDstCopyFbo->width, mDstCopyFbo->height);
    GL_CHECK(glViewport(0, 0, fboW, fboH));
    GL_CHECK(glScissor(x, y, width, height));
    GL_CHECK(glBlitFramebuffer(x, y, x + width, y + height, x, y, x + width, y + height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
#endif
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstFbo->fbo));
    const auto& dstVp = mDstFbo->viewport;
    GL_CHECK(glViewport(0, 0, dstVp.w(), dstVp.h()));

    GL_CHECK(glBlendFunc(GL_ONE, GL_ZERO));
    GlRenderTask::run();
    GL_CHECK(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));
}


/************************************************************************/
/* GlComplexBlendTask Class Implementation                              */
/************************************************************************/


GlComplexBlendTask::GlComplexBlendTask(GlProgram* program, GlRenderTarget* dstFbo, GlRenderTarget* dstCopyFbo, GlRenderTask* stencilTask, GlComposeTask* composeTask)
 : GlRenderTask(program), mDstFbo(dstFbo), mDstCopyFbo(dstCopyFbo), mStencilTask(stencilTask), mComposeTask(composeTask)
 {
 }


GlComplexBlendTask::~GlComplexBlendTask()
{
    delete mStencilTask;
    delete mComposeTask;
}


void GlComplexBlendTask::run()
{
    mComposeTask->run();

    const auto& vp = getViewport();
    const auto width = mDstFbo->width;
    const auto height = mDstFbo->height;
    if (width <= 0 || height <= 0) return;

    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mDstFbo->fbo));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo->resolvedFbo));

#if defined(THORVG_GL_TARGET_GL)
    const auto& dstVp = mDstFbo->viewport;
    // copy the current fbo to the dstCopyFbo
    GL_CHECK(glViewport(0, 0, dstVp.w(), dstVp.h()));
    GL_CHECK(glScissor(0, 0, dstVp.w(), dstVp.h()));
    GL_CHECK(glBlitFramebuffer(vp.min.x, vp.min.y, vp.max.x, vp.max.y, 0, 0, vp.w(), vp.h(), GL_COLOR_BUFFER_BIT, GL_LINEAR));
#else // TODO: create partial buffer when MSAA is disabled
    if (vp.min.x != 0 || vp.min.y != 0 || mDstCopyFbo->width != static_cast<uint32_t>(vp.w()) || mDstCopyFbo->height != static_cast<uint32_t>(vp.h())) clearColorTarget(mDstCopyFbo->width, mDstCopyFbo->height);
    GL_CHECK(glViewport(0, 0, width, height));
    GL_CHECK(glScissor(vp.min.x, vp.min.y, vp.w(), vp.h()));
    GL_CHECK(glBlitFramebuffer(vp.min.x, vp.min.y, vp.max.x, vp.max.y, vp.min.x, vp.min.y, vp.max.x, vp.max.y, GL_COLOR_BUFFER_BIT, GL_NEAREST));
#endif

    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstFbo->fbo));

    GL_CHECK(glEnable(GL_STENCIL_TEST));
    GL_CHECK(glColorMask(0, 0, 0, 0));
    GL_CHECK(glStencilFuncSeparate(GL_FRONT, GL_ALWAYS, 0x0, 0xFF));
    GL_CHECK(glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_INCR_WRAP));

    GL_CHECK(glStencilFuncSeparate(GL_BACK, GL_ALWAYS, 0x0, 0xFF));
    GL_CHECK(glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_DECR_WRAP));


    mStencilTask->run();

    GL_CHECK(glColorMask(1, 1, 1, 1));
    GL_CHECK(glStencilFunc(GL_NOTEQUAL, 0x0, 0xFF));
    GL_CHECK(glStencilOp(GL_REPLACE, GL_KEEP, GL_REPLACE));

    GL_CHECK(glBlendFunc(GL_ONE, GL_ZERO));

    GlRenderTask::run();

    GL_CHECK(glDisable(GL_STENCIL_TEST));
    GL_CHECK(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));
}


void GlComplexBlendTask::normalizeDrawDepth(int32_t maxDepth)
{
    mStencilTask->normalizeDrawDepth(maxDepth);
    GlRenderTask::normalizeDrawDepth(maxDepth);
}

/************************************************************************/
/* GlGaussianBlurTask Class Implementation                              */
/************************************************************************/

void GlGaussianBlurTask::run()
{
    const auto vp = getViewport();
    const auto width = mDstFbo->width;
    const auto height = mDstFbo->height;

    // get targets handles
    GLuint dstCopyTexId0 = mDstCopyFbo0->colorTex;
    GLuint dstCopyTexId1 = mDstCopyFbo1->colorTex;
    // get programs properties
    GlProgram* programHorz = horzTask->getProgram();
    GlProgram* programVert = vertTask->getProgram();
    GLint horzSrcTextureLoc = programHorz->getUniformLocation("uSrcTexture");
    GLint vertSrcTextureLoc = programVert->getUniformLocation("uSrcTexture");

    GL_CHECK(glViewport(0, 0, width, height));
    GL_CHECK(glScissor(0, 0, width, height));
    // we need to make a full copy of dst to intermediate buffers to be sure that they don’t contain prev data.
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mDstFbo->fbo));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo0->resolvedFbo));
    GL_CHECK(glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstFbo->fbo));

    GL_CHECK(glDisable(GL_BLEND));
    if (effect->direction == 0) {
        GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mDstFbo->fbo));
        GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo1->resolvedFbo));
        GL_CHECK(glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
        // horizontal blur
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstCopyFbo1->resolvedFbo));
        horzTask->setViewport(vp);
        horzTask->addBindResource({ 0, dstCopyTexId0, horzSrcTextureLoc });
        horzTask->run();
        // vertical blur
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstFbo->fbo));
        vertTask->setViewport(vp);
        vertTask->addBindResource({ 0, dstCopyTexId1, vertSrcTextureLoc });
        vertTask->run();
    } // horizontal
    else if (effect->direction == 1) {
        horzTask->setViewport(vp);
        horzTask->addBindResource({ 0, dstCopyTexId0, horzSrcTextureLoc });
        horzTask->run();
    } // vertical
    else if (effect->direction == 2) {
        vertTask->setViewport(vp);
        vertTask->addBindResource({ 0, dstCopyTexId0, vertSrcTextureLoc });
        vertTask->run();
    }
    GL_CHECK(glEnable(GL_BLEND));
}

/************************************************************************/
/* GlEffectDropShadowTask Class Implementation                          */
/************************************************************************/

void GlEffectDropShadowTask::run()
{
    const auto vp = getViewport();
    const auto width = mDstFbo->width;
    const auto height = mDstFbo->height;

    // get targets handles
    GLuint dstCopyTexId0 = mDstCopyFbo0->colorTex;
    GLuint dstCopyTexId1 = mDstCopyFbo1->colorTex;
    // get programs properties
    GlProgram* programHorz = horzTask->getProgram();
    GlProgram* programVert = vertTask->getProgram();
    GLint horzSrcTextureLoc = programHorz->getUniformLocation("uSrcTexture");
    GLint vertSrcTextureLoc = programVert->getUniformLocation("uSrcTexture");

    GLint srcTextureLoc = getProgram()->getUniformLocation("uSrcTexture");
    GLint blrTextureLoc = getProgram()->getUniformLocation("uBlrTexture");
    addBindResource({ 0, dstCopyTexId0, srcTextureLoc });
    addBindResource({ 1, dstCopyTexId1, blrTextureLoc });

    GL_CHECK(glViewport(0, 0, width, height));
    GL_CHECK(glScissor(0, 0, width, height));

    // we need to make a full copy of dst to intermediate buffers to be sure that they don’t contain prev data.
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mDstFbo->fbo));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo0->resolvedFbo));
    GL_CHECK(glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mDstFbo->fbo));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo1->resolvedFbo));
    GL_CHECK(glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
    
    GL_CHECK(glDisable(GL_BLEND));
    // when sigma is 0, no blur is applied, and the original image is used directly as the shadow.
    if (!tvg::zero(effect->sigma)) {
        // horizontal blur
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstCopyFbo0->resolvedFbo));
        horzTask->setViewport(vp);
        horzTask->addBindResource({ 0, dstCopyTexId1, horzSrcTextureLoc });
        horzTask->run();
        // vertical blur
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstCopyFbo1->resolvedFbo));
        vertTask->setViewport(vp);
        vertTask->addBindResource({ 0, dstCopyTexId0, vertSrcTextureLoc });
        vertTask->run();
        // copy original image to intermediate buffer
        GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mDstFbo->fbo));
        GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo0->resolvedFbo));
        GL_CHECK(glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
    }
    // run drop shadow effect
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstFbo->fbo));
    GlRenderTask::run();
    GL_CHECK(glEnable(GL_BLEND));
}

/************************************************************************/
/* GlEffectColorTransformTask Class Implementation                      */
/************************************************************************/

void GlEffectColorTransformTask::run()
{
    const auto width = mDstFbo->width;
    const auto height = mDstFbo->height;
    // get targets handles and pass to shader
    GLuint dstCopyTexId = mDstCopyFbo->colorTex;
    GLint srcTextureLoc = getProgram()->getUniformLocation("uSrcTexture");
    addBindResource({ 0, dstCopyTexId, srcTextureLoc });

    GL_CHECK(glViewport(0, 0, width, height));
    GL_CHECK(glScissor(0, 0, width, height));
    // we need to make a full copy of dst to intermediate buffers to be sure that they don’t contain prev data.
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, mDstFbo->fbo));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDstCopyFbo->resolvedFbo));
    GL_CHECK(glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, mDstFbo->fbo));

    // run transform
    GL_CHECK(glDisable(GL_BLEND));
    GlRenderTask::run();
    GL_CHECK(glEnable(GL_BLEND));
}
