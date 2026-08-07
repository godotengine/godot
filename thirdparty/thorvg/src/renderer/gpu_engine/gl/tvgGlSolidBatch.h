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

#ifndef _TVG_GL_SOLID_BATCH_H_
#define _TVG_GL_SOLID_BATCH_H_

#include "tvgGlCommon.h"

struct GlRenderer;
class GlRenderPass;
class GlRenderTask;

class GlSolidBatch
{
public:
    void clear() { *this = {}; }
    void draw(GlRenderer& renderer, GlShape& sdata, const RenderColor& color, int32_t depth, const RenderRegion& viewRegion);

private:
    bool appendable(const GlRenderer& renderer, const GlRenderPass* pass, const RenderRegion& viewRegion) const;
    void emitSingle(GlRenderer& renderer, GlRenderPass* pass, GlShape& sdata, const RenderColor& color, int32_t depth, const RenderRegion& viewRegion, uint32_t vertexCount, uint32_t indexCount);
    bool promote(GlRenderer& renderer, GlRenderPass* pass, const RenderColor& solidColor, int32_t depth, const RenderRegion& viewRegion, const GlGeometryBuffer* buffer, uint32_t vertexCount, uint32_t indexCount);
    void append(GlRenderer& renderer, const RenderColor& solidColor, const RenderRegion& viewRegion, const GlGeometryBuffer* buffer, uint32_t vertexCount, uint32_t indexCount, int32_t depth);
    static RenderColor solidColor(const GlShape& sdata, const RenderColor& color, RenderUpdateFlag flag);
    static void buildPositions(float* out, const GlGeometryBuffer* src, uint32_t count);
    static void buildColors(tvg::RGBA* out, uint32_t count, const RenderColor& color);
    static void buildIndices(uint32_t* out, const GlGeometryBuffer* src, uint32_t baseVertex);

    GlRenderPass* pass = nullptr;
    GlRenderTask* task = nullptr;
    GlShape* shape = nullptr;
    RenderColor color = {};
    RenderUpdateFlag flag = RenderUpdateFlag::None;
    int32_t depth = 0;
    uint32_t vertexCount = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    bool promoted = false;
};

#endif /* _TVG_GL_SOLID_BATCH_H_ */
