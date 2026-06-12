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

#ifndef _TVG_GL_RENDER_TASK_H_
#define _TVG_GL_RENDER_TASK_H_

#include "tvgGlCommon.h"
#include "tvgGlProgram.h"


struct GlVertexLayout
{
    uint32_t index;
    uint32_t size;
    uint32_t stride;
    size_t   offset;
    GLenum type = GL_FLOAT;
    GLboolean normalized = GL_FALSE;
    // Optional VBO for this attribute. 0 means use the GL_ARRAY_BUFFER binding
    // captured at the start of GlRenderTask::run().
    GLuint arrayBufferId = 0;
};

enum class GlBindingType
{
    kUniformBuffer,
    kTexture,
};


struct GlBindingResource
{
    GlBindingType type;
    /**
     * Binding point index.
     * Can be a uniform location for a texture
     * Can be a uniform buffer binding index for a uniform block
     */
    uint32_t        bindPoint = 0;
    GLint           location = 0;
    // GL object id used by this binding: texture id for kTexture, UBO id for kUniformBuffer.
    GLuint resourceId = 0;
    uint32_t        bufferOffset = 0;
    uint32_t        bufferRange = 0;

    GlBindingResource() = default;

    GlBindingResource(uint32_t index, GLint location, GLuint uniformBufferId, uint32_t offset, uint32_t range) : type(GlBindingType::kUniformBuffer), bindPoint(index), location(location), resourceId(uniformBufferId), bufferOffset(offset), bufferRange(range)
    {
    }

    GlBindingResource(uint32_t bindPoint, GLuint textureId, GLint location) : type(GlBindingType::kTexture), bindPoint(bindPoint), location(location), resourceId(textureId)
    {
    }
};

class GlRenderTask
{
public:
    GlRenderTask(GlProgram* program): mProgram(program) {}
    GlRenderTask(GlProgram* program, GlRenderTask* other);

    virtual ~GlRenderTask() = default;

    virtual void run();

    void addVertexLayout(const GlVertexLayout& layout);
    void setVertexColor(float r, float g, float b, float a);
    void addBindResource(const GlBindingResource& binding);
    void setDrawRange(uint32_t offset, uint32_t count);
    void setViewport(const RenderRegion& viewport);
    void setDrawDepth(int32_t depth) { mDrawDepth = static_cast<float>(depth); }
    void setViewMatrix(const Matrix& matrix) { mViewMatrix = matrix; mUseViewMatrix = true; }
    virtual void normalizeDrawDepth(int32_t maxDepth) { mDrawDepth /= static_cast<float>(maxDepth);  }

    GlProgram* getProgram() { return mProgram; }
    const RenderRegion& getViewport() const { return mViewport; }
    float getDrawDepth() const { return mDrawDepth; }
    const Array<GlVertexLayout>& getVertexLayout() const { return mVertexLayout; }
    uint32_t getIndexOffset() const { return mIndexOffset; }
    uint32_t getIndexCount() const { return mIndexCount; }

private:
    GlProgram* mProgram;
    RenderRegion mViewport = {};
    uint32_t mIndexOffset = {};
    uint32_t mIndexCount = {};
    Array<GlVertexLayout> mVertexLayout = {};
    Array<GlBindingResource> mBindingResources = {};
    float mDrawDepth = 0.f;
    Matrix mViewMatrix = {};
    bool mUseViewMatrix = false;
    bool mUseVertexColor = false;
    float mVertexColor[4] = {0.f, 0.f, 0.f, 0.f};
};

class GlStencilCoverTask : public GlRenderTask
{
public:
    GlStencilCoverTask(GlRenderTask* stencil, GlRenderTask* cover, GlStencilMode mode);
    ~GlStencilCoverTask() override;

    void run() override;

    void normalizeDrawDepth(int32_t maxDepth) override;
private:
    GlRenderTask* mStencilTask;
    GlRenderTask* mCoverTask;
    GlStencilMode mStencilMode;
};

struct GlRenderTarget;

class GlComposeTask : public GlRenderTask 
{
public:
    GlComposeTask(GlProgram* program, GLuint target, GlRenderTarget* fbo, Array<GlRenderTask*>&& tasks);
    ~GlComposeTask() override;

    void run() override;

    void setRenderSize(uint32_t width, uint32_t height) { mRenderWidth = width; mRenderHeight = height; }

    bool mClearBuffer = true;

protected:
    GLuint getTargetFbo() { return mTargetFbo; }
    GLuint getSelfFbo();
    GLuint getResolveFboId();
    void onResolve();

private:
    GLuint mTargetFbo;
    GlRenderTarget* mFbo;
    Array<GlRenderTask*> mTasks;
    uint32_t mRenderWidth = 0;
    uint32_t mRenderHeight = 0;
};

class GlBlitTask : public GlComposeTask
{
public:
    GlBlitTask(GlProgram*, GLuint target, GlRenderTarget* fbo, Array<GlRenderTask*>&& tasks);
    ~GlBlitTask() override = default;

    void run() override;

    GLuint getColorTexture() const { return mColorTex; }

    void setTargetViewport(const RenderRegion& vp) { mTargetViewport = vp; }
private:
    GLuint mColorTex = 0;
    RenderRegion mTargetViewport = {};
};

class GlDrawBlitTask : public GlComposeTask
{
public:
    GlDrawBlitTask(GlProgram*, GLuint target, GlRenderTarget* fbo, Array<GlRenderTask*>&& tasks);
    ~GlDrawBlitTask() override;

    void setPrevTask(GlRenderTask* task) { mPrevTask = task; }

    void setParentSize(uint32_t width, uint32_t height) { mParentWidth = width; mParentHeight = height; }

    void run() override;

private:
    GlRenderTask* mPrevTask = nullptr;
    uint32_t mParentWidth = 0;
    uint32_t mParentHeight = 0;
};

class GlSceneBlendTask : public GlComposeTask
{
public:
    GlSceneBlendTask(GlProgram*, GLuint target, GlRenderTarget* fbo, Array<GlRenderTask*>&& tasks);
    ~GlSceneBlendTask() override;

    void setParentSize(uint32_t width, uint32_t height) { mParentWidth = width; mParentHeight = height; }
    void setSrcTarget(GlRenderTarget* srcFbo) { mSrcFbo = srcFbo; }
    void setDstCopy(GlRenderTarget* dstCopyFbo) { mDstCopyFbo = dstCopyFbo; }

    void run() override;

private:
    GlRenderTarget* mSrcFbo = nullptr;
    GlRenderTarget* mDstCopyFbo = nullptr;
    uint32_t mParentWidth = 0;
    uint32_t mParentHeight = 0;
};

class GlClipTask : public GlRenderTask
{
public:
    GlClipTask(GlRenderTask* clip, GlRenderTask* mask);
    ~GlClipTask() override;

    void run() override;

    void normalizeDrawDepth(int32_t maxDepth) override;
private:
    GlRenderTask* mClipTask;
    GlRenderTask* mMaskTask;
};

class GlDirectBlendTask : public GlRenderTask
{
public:
    GlDirectBlendTask(GlProgram* program, GlRenderTarget* dstFbo, GlRenderTarget* dstCopyFbo, const RenderRegion& copyRegion);
    ~GlDirectBlendTask() override = default;

    void run() override;
private:
    GlRenderTarget* mDstFbo = nullptr;
    GlRenderTarget* mDstCopyFbo = nullptr;
    RenderRegion mCopyRegion{};
};

class GlComplexBlendTask: public GlRenderTask
{
public:
    GlComplexBlendTask(GlProgram* program, GlRenderTarget* dstFbo, GlRenderTarget* dstCopyFbo, GlRenderTask* stencilTask, GlComposeTask* composeTask);
    ~GlComplexBlendTask() override;

    void run() override;

    void normalizeDrawDepth(int32_t maxDepth) override;
private:
    GlRenderTarget* mDstFbo;
    GlRenderTarget* mDstCopyFbo;
    GlRenderTask* mStencilTask;
    GlComposeTask* mComposeTask;
};

class GlGaussianBlurTask: public GlRenderTask
{
public:
    GlGaussianBlurTask(GlRenderTarget* dstFbo, GlRenderTarget* dstCopyFbo0, GlRenderTarget* dstCopyFbo1): 
        GlRenderTask(nullptr), mDstFbo(dstFbo), mDstCopyFbo0(dstCopyFbo0), mDstCopyFbo1(dstCopyFbo1) {};
    ~GlGaussianBlurTask(){ delete horzTask; delete vertTask; };

    void run() override;

    GlRenderTask* horzTask;
    GlRenderTask* vertTask;
    RenderEffectGaussianBlur* effect;
private:
    GlRenderTarget* mDstFbo;
    GlRenderTarget* mDstCopyFbo0;
    GlRenderTarget* mDstCopyFbo1;
};

class GlEffectDropShadowTask: public GlRenderTask
{
public:
    GlEffectDropShadowTask(GlProgram* program, GlRenderTarget* dstFbo, GlRenderTarget* dstCopyFbo0, GlRenderTarget* dstCopyFbo1): 
        GlRenderTask(program), mDstFbo(dstFbo), mDstCopyFbo0(dstCopyFbo0), mDstCopyFbo1(dstCopyFbo1) {};
    ~GlEffectDropShadowTask(){ delete horzTask; delete vertTask; };

    void run() override;

    GlRenderTask* horzTask;
    GlRenderTask* vertTask;
    RenderEffectDropShadow* effect;
private:
    GlRenderTarget* mDstFbo;
    GlRenderTarget* mDstCopyFbo0;
    GlRenderTarget* mDstCopyFbo1;
};

class GlEffectColorTransformTask: public GlRenderTask
{
public:
    GlEffectColorTransformTask(GlProgram* program, GlRenderTarget* dstFbo, GlRenderTarget* dstCopyFbo):
        GlRenderTask(program), mDstFbo(dstFbo), mDstCopyFbo(dstCopyFbo) {};
    ~GlEffectColorTransformTask() {};

    void run() override;
private:
    GlRenderTarget* mDstFbo;
    GlRenderTarget* mDstCopyFbo;
};

#endif /* _TVG_GL_RENDER_TASK_H_ */
