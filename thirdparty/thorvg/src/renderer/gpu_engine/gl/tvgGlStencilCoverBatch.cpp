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

#include "tvgGlRenderPass.h"
#include "tvgGlRenderTask.h"
#include "tvgGlGpuBuffer.h"
#include "tvgGlStencilCoverBatch.h"

#include <cassert>
#include <string.h>

// WebGL uses smaller batch caps to limit wasm memory. Native GL keeps larger caps
// for fewer render tasks. BATCH_REGION_RESET_THRESHOLD only releases cached bounds.
#ifdef __EMSCRIPTEN__
static constexpr uint32_t BATCH_REGION_MAX_COUNT = 256;
static constexpr uint32_t BATCH_REGION_RESET_THRESHOLD = 64;
static constexpr uint32_t BATCH_VERTEX_MAX_COUNT = 65536;
static constexpr uint32_t BATCH_INDEX_MAX_COUNT = 262144;
#else
static constexpr uint32_t BATCH_REGION_MAX_COUNT = 512;
static constexpr uint32_t BATCH_REGION_RESET_THRESHOLD = 128;
static constexpr uint32_t BATCH_VERTEX_MAX_COUNT = 262144;
static constexpr uint32_t BATCH_INDEX_MAX_COUNT = 1048576;
#endif

static constexpr uint32_t COVER_VERTEX_COUNT = 6;

struct StencilCoverVertex
{
    float x;
    float y;
    tvg::RGBA color;
};

static void addStencilCoverPositionLayout(GlRenderTask* task, GlStageBuffer* gpuBuffer, const RenderRegion& bbox)
{
    float vertex[] = {
        float(bbox.min.x), float(bbox.min.y),
        float(bbox.min.x), float(bbox.max.y),
        float(bbox.max.x), float(bbox.min.y),
        float(bbox.max.x), float(bbox.min.y),
        float(bbox.min.x), float(bbox.max.y),
        float(bbox.max.x), float(bbox.max.y)};
    task->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), gpuBuffer->pushAux(vertex, sizeof(vertex)), GL_FLOAT, GL_FALSE, gpuBuffer->getAuxBufferId()});
}

static void addStencilCoverSolidLayout(GlRenderTask* task, GlStageBuffer* gpuBuffer, const RenderRegion& bbox, const RenderColor& color)
{
    StencilCoverVertex vertex[] = {
        {float(bbox.min.x), float(bbox.min.y), color},
        {float(bbox.min.x), float(bbox.max.y), color},
        {float(bbox.max.x), float(bbox.min.y), color},
        {float(bbox.max.x), float(bbox.min.y), color},
        {float(bbox.min.x), float(bbox.max.y), color},
        {float(bbox.max.x), float(bbox.max.y), color}};
    auto vertexOffset = gpuBuffer->pushAux(vertex, sizeof(vertex));
    auto auxBuffer = gpuBuffer->getAuxBufferId();
    task->addVertexLayout(GlVertexLayout{0, 2, sizeof(StencilCoverVertex), vertexOffset, GL_FLOAT, GL_FALSE, auxBuffer});
    task->addVertexLayout(GlVertexLayout{1, 4, sizeof(StencilCoverVertex), vertexOffset + 2 * sizeof(float), GL_UNSIGNED_BYTE, GL_TRUE, auxBuffer});
}

static uint32_t* drawStencilGeometry(GlRenderTask* task, GlStageBuffer* gpuBuffer, const GlGeometryBuffer* buffer)
{
    auto vertexOffset = gpuBuffer->push(buffer->vertex.data, buffer->vertex.count * sizeof(float));

    uint32_t* indices = nullptr;
    auto indexOffset = gpuBuffer->reserveIndex(buffer->index.count * sizeof(uint32_t), reinterpret_cast<void**>(&indices));
    if (buffer->index.count > 0) memcpy(indices, buffer->index.data, buffer->index.count * sizeof(uint32_t));

    task->addVertexLayout(GlVertexLayout{0, 2, 2 * sizeof(float), vertexOffset});
    task->setDrawRange(indexOffset, buffer->index.count);
    return indices;
}

static void buildIndices(uint32_t* out, const GlGeometryBuffer* src, uint32_t baseVertex)
{
    for (uint32_t i = 0; i < src->index.count; ++i) out[i] = src->index[i] + baseVertex;
}

static bool solidCoverLayout(const Array<GlVertexLayout>& layouts)
{
    if (layouts.count != 2) return false;

    const auto& position = layouts[0];
    const auto& color = layouts[1];
    if (position.index != 0 || position.size != 2 || position.type != GL_FLOAT || position.normalized != GL_FALSE) return false;
    if (color.index != 1 || color.size != 4 || color.type != GL_UNSIGNED_BYTE || color.normalized != GL_TRUE) return false;
    if (position.stride != color.stride) return false;
    if (position.arrayBufferId == 0 || position.arrayBufferId != color.arrayBufferId) return false;
    if (color.offset != position.offset + 2 * sizeof(float)) return false;

    return true;
}


void GlStencilCoverBatch::clear()
{
    pass = nullptr;
    task = nullptr;
    stencilTask = nullptr;
    mode = GlStencilMode::None;
    if (bounds.reserved > BATCH_REGION_RESET_THRESHOLD) bounds.reset();
    else bounds.clear();
    stencilViewBounds = {};
    coverViewBounds = {};
    vertexCount = 0;
    indexOffset = 0;
    indexCount = 0;
    clipped = false;
    ySorted = false;
    open = false;
}

GlRenderTask* GlStencilCoverBatch::prepare(GlProgram* stencilProgram, GlRenderPass* pass, GlRenderTask* coverTask,
                                           const GlGeometry& geometry, GlStageBuffer* gpuBuffer, RenderUpdateFlag flag,
                                           GlStencilMode stencilMode, bool clipped, int32_t depth, const Matrix& viewMatrix,
                                           const RenderRegion& passViewport, const RenderColor* color,
                                           const RenderRegion& viewBounds, RenderRegion& geometryBounds,
                                           const GlGeometryBuffer*& stencilBuffer, uint32_t*& stencilIndices,
                                           bool& merge)
{
    auto stroke = (flag & RenderUpdateFlag::Stroke) || (flag & RenderUpdateFlag::GradientStroke);
    auto bbox = stroke ? geometry.strokeBounds : geometry.fillBounds;
    geometryBounds = stroke ? gpuTransformBounds(bbox, geometry.matrix) : bbox;
    geometryBounds.intersect(viewBounds);

    auto x = geometryBounds.sx() - passViewport.sx();
    auto y = geometryBounds.sy() - passViewport.sy();
    auto w = geometryBounds.sw();
    auto h = geometryBounds.sh();
    auto yGl = passViewport.sh() - y - h;
    auto stencilViewport = RenderRegion{{x, yGl}, {x + w, yGl + h}};
    coverTask->setViewport(stencilViewport);

    if (color) addStencilCoverSolidLayout(coverTask, gpuBuffer, bbox, *color);
    else addStencilCoverPositionLayout(coverTask, gpuBuffer, bbox);
    coverTask->mUseDrawArrays = true;
    coverTask->mArrayMode = GL_TRIANGLES;
    coverTask->mArrayOffset = 0;
    coverTask->mIndexCount = COVER_VERTEX_COUNT;
    stencilBuffer = stroke ? &geometry.stroke : &geometry.fill;
    // Cache this before writing stencil indices; the batch needs the
    // pre-mutation answer to keep the index stream mergeable.
    merge = mergeable(pass, stencilMode, clipped, geometryBounds, stencilBuffer);

    auto stencilTask = new GlRenderTask(stencilProgram);
    stencilTask->setViewMatrix(viewMatrix);
    stencilTask->setDrawDepth(depth);
    stencilIndices = drawStencilGeometry(stencilTask, gpuBuffer, stencilBuffer);
    stencilTask->setViewport(stencilViewport);
    return stencilTask;
}

bool GlStencilCoverBatch::intersects(const RenderRegion& bounds) const
{
    auto min = ySorted ? bounds.min.y : bounds.min.x;
    auto max = ySorted ? bounds.max.y : bounds.max.x;
    ARRAY_FOREACH(p, this->bounds) {
        auto pMin = ySorted ? (*p).min.y : (*p).min.x;
        if ((ySorted ? (*p).max.y : (*p).max.x) <= min) continue;
        if (pMin >= max) break;
        if ((*p).intersected(bounds)) return true;
    }

    return false;
}


void GlStencilCoverBatch::addBounds(const RenderRegion& bounds)
{
    auto p = this->bounds.count;
    auto min = ySorted ? bounds.min.y : bounds.min.x;
    this->bounds.grow(1);
    while (p > 0 && (ySorted ? this->bounds[p - 1].min.y : this->bounds[p - 1].min.x) > min) {
        this->bounds[p] = this->bounds[p - 1];
        --p;
    }
    this->bounds[p] = bounds;
    ++this->bounds.count;
}


bool GlStencilCoverBatch::mergeable(const GlRenderPass* pass, GlStencilMode mode, bool clipped, const RenderRegion& bounds, const GlGeometryBuffer* stencilBuffer) const
{
    if (!open) return false;
    // A new current pass is a hard batch boundary; fail before touching the old pass/task pair.
    if (this->pass != pass) return false;
    if (pass->lastTask() != task) return false;
    if (this->mode != mode) return false;
    // drawClip() clears the batch for every clipped paint, so different clip
    // chains cannot continue the same batch. Only clipped vs unclipped needs
    // an extra merge guard here.
    if (this->clipped != clipped) return false;
    if (bounds.invalid()) return false;
    if (this->bounds.count >= BATCH_REGION_MAX_COUNT) return false;
    auto incomingVertexCount = stencilBuffer ? stencilBuffer->vertex.count / 2 : 0;
    auto incomingIndexCount = stencilBuffer ? stencilBuffer->index.count : 0;
    if (incomingVertexCount > BATCH_VERTEX_MAX_COUNT || vertexCount > BATCH_VERTEX_MAX_COUNT - incomingVertexCount) return false;
    if (incomingIndexCount > BATCH_INDEX_MAX_COUNT || indexCount > BATCH_INDEX_MAX_COUNT - incomingIndexCount) return false;
    return !intersects(bounds);
}


void GlStencilCoverBatch::draw(GlRenderPass* pass, GlRenderTask* stencil, GlRenderTask* cover, bool merge, GlStencilMode mode, bool clipped, const RenderRegion& bounds, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer, uint32_t* stencilIndices)
{
    if (!stencil || !cover) {
        delete stencil;
        delete cover;
        return;
    }

    if (merge) this->append(stencil, cover, bounds, viewBounds, stencilBuffer, stencilIndices);
    else emitSingle(pass, stencil, cover, mode, clipped, bounds, viewBounds, stencilBuffer);
}


void GlStencilCoverBatch::emitSingle(GlRenderPass* pass, GlRenderTask* stencil, GlRenderTask* cover, GlStencilMode mode, bool clipped, const RenderRegion& bounds, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer)
{
    auto task = new GlStencilCoverTask(stencil, cover, mode);
    pass->addRenderTask(task);

    this->pass = pass;
    this->task = task;
    this->mode = mode;
    this->clipped = clipped;
    ySorted = pass->getViewport().sh() > pass->getViewport().sw();
    coverViewBounds = viewBounds;
    if (this->bounds.reserved > BATCH_REGION_RESET_THRESHOLD) this->bounds.reset();
    else this->bounds.clear();
    setStencilMergeTarget(stencil, viewBounds, stencilBuffer);
    open = bounds.valid();
    if (open) addBounds(bounds);
}


void GlStencilCoverBatch::append(GlRenderTask* stencil, GlRenderTask* cover, const RenderRegion& bounds, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer, uint32_t* stencilIndices)
{
    if (!mergeCover(cover, viewBounds)) {
        task->mCoverTasks.push(cover);
        coverViewBounds = viewBounds;
    }
    if (!merge(stencil, viewBounds, stencilBuffer, stencilIndices)) {
        task->mStencilTasks.push(stencil);
        setStencilMergeTarget(stencil, viewBounds, stencilBuffer);
    }
    addBounds(bounds);
}


void GlStencilCoverBatch::setStencilMergeTarget(GlRenderTask* stencil, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer)
{
    stencilTask = stencil;
    stencilViewBounds = viewBounds;
    vertexCount = stencilBuffer ? stencilBuffer->vertex.count / 2 : 0;
    indexOffset = stencil->mIndexOffset;
    indexCount = stencil->mIndexCount;
}


bool GlStencilCoverBatch::merge(GlRenderTask* stencil, const RenderRegion& viewBounds, const GlGeometryBuffer* stencilBuffer, uint32_t* stencilIndices)
{
    auto incomingVertexCount = stencilBuffer ? stencilBuffer->vertex.count / 2 : 0;
    if (!stencilTask || vertexCount == 0 || incomingVertexCount == 0) return false;
    if (!(stencilViewBounds == viewBounds)) return false;
    if (!stencilIndices) return false;
    if (stencilTask->mProgram != stencil->mProgram) return false;
    if (stencilTask->mUseViewMatrix != stencil->mUseViewMatrix) return false;
    if (stencilTask->mUseViewMatrix && !(stencilTask->mViewMatrix == stencil->mViewMatrix)) return false;

    const auto& layouts = stencilTask->mVertexLayout;
    const auto& appendLayouts = stencil->mVertexLayout;
    if (layouts.count != 1 || appendLayouts.count != 1) return false;

    const auto& layout = layouts[0];
    const auto& appendLayout = appendLayouts[0];
    if (layout.index != appendLayout.index || layout.size != appendLayout.size || layout.stride != appendLayout.stride) return false;
    if (layout.offset + vertexCount * layout.stride != appendLayout.offset || layout.type != appendLayout.type || layout.normalized != appendLayout.normalized) return false;
    if (layout.arrayBufferId != appendLayout.arrayBufferId) return false;

    auto expectedIndexOffset = indexOffset + indexCount * sizeof(uint32_t);
    if (stencil->mIndexOffset != expectedIndexOffset) return false;

    buildIndices(stencilIndices, stencilBuffer, vertexCount);
    indexCount += stencil->mIndexCount;
    vertexCount += incomingVertexCount;
    stencilTask->mIndexOffset = indexOffset;
    stencilTask->mIndexCount = indexCount;
    stencilTask->mDrawDepth = stencil->mDrawDepth;
    stencilTask->mViewport.add(stencil->mViewport);

    delete stencil;
    return true;
}


bool GlStencilCoverBatch::mergeCover(GlRenderTask* cover, const RenderRegion& viewBounds)
{
    if (task->mCoverTasks.empty()) return false;
    if (!(coverViewBounds == viewBounds)) return false;

    auto dst = task->mCoverTasks.last();
    if (dst->mProgram != cover->mProgram) return false;
    if (dst->mBindingResources.count > 0 || cover->mBindingResources.count > 0) return false;
    if (!dst->mUseDrawArrays || !cover->mUseDrawArrays) return false;
    if (dst->mArrayMode != GL_TRIANGLES || cover->mArrayMode != GL_TRIANGLES) return false;
    if (dst->mUseVertexColor || cover->mUseVertexColor) return false;
    if (dst->mUseViewMatrix != cover->mUseViewMatrix) return false;
    if (dst->mUseViewMatrix && !(dst->mViewMatrix == cover->mViewMatrix)) return false;

    const auto& layouts = dst->mVertexLayout;
    const auto& coverLayouts = cover->mVertexLayout;
    if (!solidCoverLayout(layouts) || !solidCoverLayout(coverLayouts)) return false;

    const auto& position = layouts[0];
    const auto& coverPosition = coverLayouts[0];
    const auto& color = layouts[1];
    const auto& coverColor = coverLayouts[1];
    if (position.stride != coverPosition.stride || position.arrayBufferId != coverPosition.arrayBufferId) return false;
    if (color.stride != coverColor.stride || color.arrayBufferId != coverColor.arrayBufferId) return false;

    auto dstEnd = position.offset + (static_cast<size_t>(dst->mArrayOffset) + dst->mIndexCount) * position.stride;
    auto coverStart = coverPosition.offset + static_cast<size_t>(cover->mArrayOffset) * coverPosition.stride;
    auto colorEnd = color.offset + (static_cast<size_t>(dst->mArrayOffset) + dst->mIndexCount) * color.stride;
    auto coverColorStart = coverColor.offset + static_cast<size_t>(cover->mArrayOffset) * coverColor.stride;
    assert(dstEnd == coverStart && colorEnd == coverColorStart);
    if (dstEnd != coverStart || colorEnd != coverColorStart) return false;

    dst->mIndexCount += cover->mIndexCount;
    dst->mViewport.add(cover->mViewport);
    dst->mDrawDepth = cover->mDrawDepth;

    delete cover;
    return true;
}
