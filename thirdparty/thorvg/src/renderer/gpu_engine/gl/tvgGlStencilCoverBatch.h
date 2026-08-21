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

#ifndef _TVG_GL_STENCIL_COVER_BATCH_H_
#define _TVG_GL_STENCIL_COVER_BATCH_H_

#include "tvgGlCommon.h"

class GlRenderPass;
class GlRenderTask;
class GlStencilCoverTask;
class GlProgram;
class GlStageBuffer;

class GlStencilCoverBatch
{
public:
    void clear();
    GlRenderTask* prepare(GlProgram* stencilProgram, GlRenderPass* pass, GlRenderTask* coverTask,
                          const GlGeometry& geometry, GlStageBuffer* gpuBuffer, RenderUpdateFlag flag,
                          GlStencilMode stencilMode, bool clipped, int32_t depth, const Matrix& viewMatrix,
                          const RenderRegion& passViewport, const RenderColor* color,
                          const RenderRegion& viewBounds, RenderRegion& geometryBounds,
                          const GlGeometryBuffer*& stencilBuffer, uint32_t*& stencilIndices,
                          bool& merge);
    bool mergeable(const GlRenderPass* pass, GlStencilMode mode, bool clipped, const RenderRegion& bounds, const GlGeometryBuffer* stencilBuffer) const;
    void draw(GlRenderPass* pass, GlRenderTask* stencil, GlRenderTask* cover, bool merge, GlStencilMode mode, bool clipped, const RenderRegion& bounds, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer, uint32_t* stencilIndices);

private:
    void emitSingle(GlRenderPass* pass, GlRenderTask* stencil, GlRenderTask* cover, GlStencilMode mode, bool clipped, const RenderRegion& bounds, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer);
    void append(GlRenderTask* stencil, GlRenderTask* cover, const RenderRegion& bounds, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer, uint32_t* stencilIndices);
    void setStencilMergeTarget(GlRenderTask* stencil, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer);
    bool merge(GlRenderTask* stencil, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer, uint32_t* stencilIndices);
    bool mergeCover(GlRenderTask* cover, const RenderRegion& viewBounds);
    bool intersects(const RenderRegion& bounds) const;
    void addBounds(const RenderRegion& bounds);

    GlRenderPass* pass = nullptr;
    GlStencilCoverTask* task = nullptr;
    GlRenderTask* stencilTask = nullptr;
    GlStencilMode mode = GlStencilMode::None;
    Array<RenderRegion> bounds = {};
    RenderRegion stencilViewBounds = {};
    RenderRegion coverViewBounds = {};
    uint32_t vertexCount = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    bool clipped = false;
    bool ySorted = false;
    bool open = false;
};

#endif /* _TVG_GL_STENCIL_COVER_BATCH_H_ */
