/*
 * Copyright (c) 2024 - 2026 ThorVG project. All rights reserved.

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

#include "tvgWgCompositor.h"
#include "tvgWgShaderTypes.h"
#include <iostream>

void WgCompositor::updateViewMat(WgContext& context, uint32_t width, uint32_t height)
{
    if (bindGroupViewMat && viewMatWidth == width && viewMatHeight == height) return;

    WgShaderTypeMat4x4f viewMat(width, height);
    bool bufferChanged = context.allocateBufferUniform(bufferViewMat, &viewMat, sizeof(viewMat));

    if (bufferChanged || !bindGroupViewMat) {
        context.layouts.releaseBindGroup(bindGroupViewMat);
        bindGroupViewMat = context.layouts.createBindGroupBuffer1Un(bufferViewMat);
    }

    viewMatWidth = width;
    viewMatHeight = height;
}


void WgCompositor::initialize(WgContext& context, uint32_t width, uint32_t height)
{
    // pipelines (external handle, do not release)
    pipelines.initialize(context);
    stageBufferGeometry.initialize(context);
    // initialize opacity pool
    initPools(context);
    // allocate global view matrix handles
    updateViewMat(context, width, height);
    // create render targets handles
    resize(context, width, height);
    // composition and blend geometries
    meshDataBlit.blitBox();
    // force stage buffers initialization
    flush(context);
}


void WgCompositor::initPools(WgContext& context)
{
    for (uint32_t i = 0; i < 256; i++) {
        float opacity = i / 255.0f;
        context.allocateBufferUniform(bufferOpacities[i], &opacity, sizeof(float));
        bindGroupOpacities[i] = context.layouts.createBindGroupBuffer1Un(bufferOpacities[i]);
    }
}


void WgCompositor::release(WgContext& context)
{
    // release render targets habdles
    resize(context, 0, 0);
    // release opacity pool
    releasePools(context);
    // release global view matrix handles
    context.layouts.releaseBindGroup(bindGroupViewMat);
    context.releaseBuffer(bufferViewMat);
    viewMatWidth = 0;
    viewMatHeight = 0;
    // release stage buffer
    stageBufferSolidColor.release(context);
    stageBufferPaint.release(context);
    stageBufferGeometry.release(context);
    // release pipelines
    pipelines.release(context);
}


void WgCompositor::releasePools(WgContext& context)
{
    // release opacity pool
    for (uint32_t i = 0; i < 256; i++) {
        context.layouts.releaseBindGroup(bindGroupOpacities[i]);
        context.releaseBuffer(bufferOpacities[i]);
    }
}


void WgCompositor::resize(WgContext& context, uint32_t width, uint32_t height) {
    // release existig handles
    if ((this->width != width) || (this->height != height)) {
        context.layouts.releaseBindGroup(bindGroupStorageTemp);
        // release intermediate render target
        targetTemp1.release(context);
        targetTemp0.release(context);
        // release global stencil buffer handles
        context.releaseTextureView(texViewDepthStencilMS);
        context.releaseTexture(texDepthStencilMS);
        context.releaseTextureView(texViewDepthStencil);
        context.releaseTexture(texDepthStencil);
        // store render target dimensions
        this->height = height;
        this->width = width;
    }

    // create render targets handles
    if ((width != 0) && (height != 0)) {
        // store render target dimensions
        this->width = width;
        this->height = height;
        // update global view matrix handles
        updateViewMat(context, width, height);
        // allocate global stencil buffer handles
        texDepthStencil = context.createTexAttachement(width, height, WGPUTextureFormat_Depth24PlusStencil8, 1);
        texViewDepthStencil = context.createTextureView(texDepthStencil);
        texDepthStencilMS = context.createTexAttachement(width, height, WGPUTextureFormat_Depth24PlusStencil8, 4);
        texViewDepthStencilMS = context.createTextureView(texDepthStencilMS);
        // initialize intermediate render targets
        targetTemp0.initialize(context, width, height);
        targetTemp1.initialize(context, width, height);
        bindGroupStorageTemp = context.layouts.createBindGroupStrorage2RO(targetTemp0.texView, targetTemp1.texView);
    }
}


RenderRegion WgCompositor::shrinkRenderRegion(const RenderRegion& rect)
{
    return {
        {std::max(0, std::min((int32_t)width, rect.min.x)), std::max(0, std::min((int32_t)height, rect.min.y))},
        {std::max(0, std::min((int32_t)width, rect.max.x)), std::max(0, std::min((int32_t)height, rect.max.y))}
    };
}


void WgCompositor::copyTexture(const WgRenderTarget* dst, const WgRenderTarget* src)
{
    const RenderRegion region = {{0, 0}, {(int32_t)src->width, (int32_t)src->height}};
    copyTexture(dst, src, region);
}


void WgCompositor::copyTexture(const WgRenderTarget* dst, const WgRenderTarget* src, const RenderRegion& region)
{
    assert(dst);
    assert(src);
    assert(commandEncoder);
    const WGPUTexelCopyTextureInfo texSrc { .texture = src->texture, .origin = { .x = (uint32_t)region.min.x, .y = (uint32_t)region.min.y } };
    const WGPUTexelCopyTextureInfo texDst { .texture = dst->texture, .origin = { .x = (uint32_t)region.min.x, .y = (uint32_t)region.min.y } };
    const WGPUExtent3D copySize { .width = region.w(), .height = region.h(), .depthOrArrayLayers = 1 };
    wgpuCommandEncoderCopyTextureToTexture(commandEncoder, &texSrc, &texDst, &copySize);
}


void WgCompositor::beginRenderPassMS(WGPUCommandEncoder commandEncoder, WgRenderTarget* target, bool clear, WGPUColor clearColor)
{
    assert(target);
    assert(commandEncoder);
    // do not start same render bass
    if (target == currentTarget) return;
    // we must to end render pass first
    endRenderPass();
    this->currentTarget = target;
    // start new render pass
    this->commandEncoder = commandEncoder;
    const WGPURenderPassDepthStencilAttachment depthStencilAttachment{ 
        .view = texViewDepthStencilMS,
        .depthLoadOp = WGPULoadOp_Clear,
        .depthStoreOp = WGPUStoreOp_Discard,
        .depthClearValue = 1.0f,
        .stencilLoadOp = WGPULoadOp_Clear,
        .stencilStoreOp = WGPUStoreOp_Discard,
        .stencilClearValue = 0
    };
    const WGPURenderPassColorAttachment colorAttachment{
        .view = target->texViewMS,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .resolveTarget = target->texView,
        .loadOp = clear ? WGPULoadOp_Clear : WGPULoadOp_Load,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = clearColor
    };
    WGPURenderPassDescriptor renderPassDesc{ .colorAttachmentCount = 1, .colorAttachments = &colorAttachment, .depthStencilAttachment = &depthStencilAttachment };
    renderPassEncoder = wgpuCommandEncoderBeginRenderPass(commandEncoder, &renderPassDesc);
    assert(renderPassEncoder);
}


void WgCompositor::beginRenderPass(WGPUCommandEncoder encoder, WgRenderTarget* target)
{
    assert(!renderPassEncoder);
    currentTarget = target;
    const WGPURenderPassDepthStencilAttachment depthStencilAttachment{
        .view = texViewDepthStencil,
        .depthLoadOp = WGPULoadOp_Load,
        .depthStoreOp = WGPUStoreOp_Discard,
        .stencilLoadOp = WGPULoadOp_Load,
        .stencilStoreOp = WGPUStoreOp_Discard
    };
    const WGPURenderPassColorAttachment colorAttachment {
        .view = target->texView,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .loadOp = WGPULoadOp_Load,
        .storeOp = WGPUStoreOp_Store,
    };
    const WGPURenderPassDescriptor renderPassDesc{ .colorAttachmentCount = 1, .colorAttachments = &colorAttachment, .depthStencilAttachment = &depthStencilAttachment };
    renderPassEncoder = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
}


void WgCompositor::endRenderPass()
{
    if (currentTarget) {
        assert(renderPassEncoder);
        wgpuRenderPassEncoderEnd(renderPassEncoder);
        wgpuRenderPassEncoderRelease(renderPassEncoder);
        renderPassEncoder = nullptr;
        currentTarget = nullptr;
    }
}

void WgCompositor::reset(WgContext& context)
{
    stageBufferGeometry.clear();
    stageBufferSolidColor.clear();
    stageBufferPaint.clear();
}


void WgCompositor::flush(WgContext& context)
{
    stageBufferGeometry.append(&meshDataBlit);
    stageBufferGeometry.flush(context);
    stageBufferSolidColor.flush(context);
    stageBufferPaint.flush(context);
    context.submit();
}


void WgCompositor::requestShape(WgRenderDataShape* renderData)
{
    stageBufferGeometry.append(renderData);

    auto& shapeSettings = renderData->renderSettingsShape;
    if (shapeSettings.fillType == WgRenderSettingsType::Solid) shapeSettings.solidColorInd = stageBufferSolidColor.append(shapeSettings.settings.color);
    else shapeSettings.bindGroupInd = stageBufferPaint.append(shapeSettings.settings);

    if (!renderData->renderSettingsStroke.skip && renderData->meshStrokes.vbuffer.count > 0) {
        auto& strokeSettings = renderData->renderSettingsStroke;
        if (strokeSettings.fillType == WgRenderSettingsType::Solid) strokeSettings.solidColorInd = stageBufferSolidColor.append(strokeSettings.settings.color);
        else strokeSettings.bindGroupInd = stageBufferPaint.append(strokeSettings.settings);
    }
    ARRAY_FOREACH(p, renderData->clips)
        requestShape((WgRenderDataShape*)(*p));
}


void WgCompositor::requestImage(WgRenderDataPicture* renderData)
{
    stageBufferGeometry.append(renderData);
    renderData->renderSettings.bindGroupInd = stageBufferPaint.append(renderData->renderSettings.settings);
    ARRAY_FOREACH(p, renderData->clips)
        requestShape((WgRenderDataShape*)(*p));
}


void WgCompositor::renderShape(WgContext& context, WgRenderDataShape* renderData, BlendMethod blendMethod)
{
    assert(renderData);
    assert(renderPassEncoder);
    // apply clip path if necessary
    if (!renderData->clips.empty()) {
        renderClipPath(context, renderData);
        if (renderData->strokeFirst) {
            clipStrokes(context, renderData);
            clipShape(context, renderData);
        } else {
            clipShape(context, renderData);
            clipStrokes(context, renderData);
        }
        clearClipPath(context, renderData);
    // use custom blending
    } else if (blendMethod != BlendMethod::Normal) {
        if (renderData->strokeFirst) {
            blendStrokes(context, renderData, blendMethod);
            blendShape(context, renderData, blendMethod);
        } else {
            blendShape(context, renderData, blendMethod);
            blendStrokes(context, renderData, blendMethod);
        }
    // use direct hardware blending
    } else {
        if (renderData->strokeFirst) {
            drawStrokes(context, renderData);
            drawShape(context, renderData);
        } else {
            drawShape(context, renderData);
            drawStrokes(context, renderData);
        }
    }
}


void WgCompositor::renderImage(WgContext& context, WgRenderDataPicture* renderData, BlendMethod blendMethod)
{
    assert(renderData);
    assert(renderPassEncoder);
    // apply clip path if necessary
    if (renderData->clips.count != 0) {
        renderClipPath(context, renderData);
        clipImage(context, renderData);
        clearClipPath(context, renderData);
    // use custom blending
    } else if (blendMethod != BlendMethod::Normal)
        blendImage(context, renderData, blendMethod);
    // use direct hardware blending
    else drawImage(context, renderData);
}


void WgCompositor::renderScene(WgContext& context, WgRenderTarget* scene, WgCompose* compose)
{
    assert(scene);
    assert(compose);
    assert(renderPassEncoder);
    // use custom blending
    if (compose->blend != BlendMethod::Normal)
        blendScene(context, scene, compose);
    // use direct hardware blending
    else drawScene(context, scene, compose);
}


void WgCompositor::composeScene(WgContext& context, WgRenderTarget* src, WgRenderTarget* mask, WgCompose* cmp)
{
    assert(cmp);
    assert(src);
    assert(mask);
    assert(renderPassEncoder);
    RenderRegion rect = shrinkRenderRegion(cmp->aabb);
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, rect.x(), rect.y(), rect.w(), rect.h());
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, src->bindGroupTexture, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, mask->bindGroupTexture, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.scene_compose[(uint32_t)cmp->method]);
    drawMeshImage(context, &meshDataBlit);
}


void WgCompositor::blit(WgContext& context, WGPUCommandEncoder encoder, WgRenderTarget* src, WGPUTextureView dstView)
{
    assert(!renderPassEncoder);
    const WGPURenderPassDepthStencilAttachment depthStencilAttachment{
        .view = texViewDepthStencil,
        .depthLoadOp = WGPULoadOp_Load,
        .depthStoreOp = WGPUStoreOp_Discard,
        .stencilLoadOp = WGPULoadOp_Load,
        .stencilStoreOp = WGPUStoreOp_Discard
    };
    const WGPURenderPassColorAttachment colorAttachment { 
        .view = dstView,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .loadOp = WGPULoadOp_Load,
        .storeOp = WGPUStoreOp_Store,
    };
    const WGPURenderPassDescriptor renderPassDesc{ .colorAttachmentCount = 1, .colorAttachments = &colorAttachment, .depthStencilAttachment = &depthStencilAttachment };
    renderPassEncoder = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, src->bindGroupTexture, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.blit);
    drawMeshImage(context, &meshDataBlit);
    wgpuRenderPassEncoderEnd(renderPassEncoder);
    wgpuRenderPassEncoderRelease(renderPassEncoder);
    renderPassEncoder = nullptr;
}


void WgCompositor::drawMesh(WgContext& context, WgMeshData* meshData)
{
    assert(meshData);
    assert(renderPassEncoder);
    uint64_t icount = meshData->ibuffer.count;
    uint64_t vsize = meshData->vbuffer.count * sizeof(Point);
    uint64_t isize = icount * sizeof(uint32_t);
    wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 0, stageBufferGeometry.vbuffer_gpu, meshData->voffset, vsize);
    wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder, stageBufferGeometry.ibuffer_gpu, WGPUIndexFormat_Uint32, meshData->ioffset, isize);
    wgpuRenderPassEncoderDrawIndexed(renderPassEncoder, icount, 1, 0, 0, 0);
};


void WgCompositor::drawMeshSolid(WgContext& context, WgMeshData* meshData, uint32_t solidColorInd)
{
    const uint64_t icount = meshData->ibuffer.count;
    const uint64_t vsize = meshData->vbuffer.count * sizeof(Point);
    const uint64_t isize = icount * sizeof(uint32_t);
    const uint64_t csize = sizeof(WgShaderTypeVec4f);
    // One instance (instanceCount = 1): select this draw's single vec4 solid color by offset.
    const uint64_t coffset = solidColorInd * csize;
    wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 0, stageBufferGeometry.vbuffer_gpu, meshData->voffset, vsize);
    wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 1, stageBufferSolidColor.vbuffer_gpu, coffset, csize);
    wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder, stageBufferGeometry.ibuffer_gpu, WGPUIndexFormat_Uint32, meshData->ioffset, isize);
    wgpuRenderPassEncoderDrawIndexed(renderPassEncoder, icount, 1, 0, 0, 0);
}


void WgCompositor::drawMeshImage(WgContext& context, WgMeshData* meshData)
{
    assert(meshData);
    assert(renderPassEncoder);
    uint64_t icount = meshData->ibuffer.count;
    uint64_t vsize = meshData->vbuffer.count * sizeof(Point);
    uint64_t isize = icount * sizeof(uint32_t);
    wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 0, stageBufferGeometry.vbuffer_gpu, meshData->voffset, vsize);
    wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 1, stageBufferGeometry.vbuffer_gpu, meshData->toffset, vsize);
    wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder, stageBufferGeometry.ibuffer_gpu, WGPUIndexFormat_Uint32, meshData->ioffset, isize);
    wgpuRenderPassEncoderDrawIndexed(renderPassEncoder, icount, 1, 0, 0, 0);
};


void WgCompositor::drawShape(WgContext& context, WgRenderDataShape* renderData)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->renderSettingsShape.skip || renderData->meshShape.vbuffer.count == 0 || renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettingsShape;
    const bool convex = renderData->convex;
    WgMeshData* mesh = renderData->convex ? &renderData->meshShape : &renderData->meshBBox;

    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());

    // setup stencil rules
    if (!convex) {
        WGPURenderPipeline stencilPipeline = (renderData->fillRule == FillRule::NonZero) ? pipelines.nonzero : pipelines.evenodd;
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, stencilPipeline);
        // draw to stencil (first pass)
        drawMesh(context, &renderData->meshShape);
    }

    // setup fill rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);

    if (settings.fillType == WgRenderSettingsType::Solid) {
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, convex ? pipelines.solid_conv : pipelines.solid);
        drawMeshSolid(context, mesh, settings.solidColorInd);
    } else if (settings.fillType == WgRenderSettingsType::Linear) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, convex ? pipelines.linear_conv : pipelines.linear);
        drawMesh(context, mesh);
    } else if (settings.fillType == WgRenderSettingsType::Radial) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, convex ? pipelines.radial_conv : pipelines.radial);
        drawMesh(context, mesh);
    }
}


void WgCompositor::blendShape(WgContext& context, WgRenderDataShape* renderData, BlendMethod blendMethod)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->renderSettingsShape.skip || renderData->meshShape.vbuffer.count == 0 || renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettingsShape;
    // copy current render target data to dst target
    WgRenderTarget *target = currentTarget;
    endRenderPass();
    copyTexture(&targetTemp0, target);
    beginRenderPassMS(commandEncoder, target, false);
    // render shape with blend settings
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    // setup stencil rules
    WGPURenderPipeline stencilPipeline = (renderData->fillRule == FillRule::NonZero) ? pipelines.nonzero : pipelines.evenodd;
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, stencilPipeline);
    // draw to stencil (first pass)
    drawMesh(context, &renderData->meshShape);
    // setup fill rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    uint32_t blendMethodInd = (uint32_t)blendMethod;
    if (settings.fillType == WgRenderSettingsType::Solid) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.solid_blend[blendMethodInd]);
        drawMeshSolid(context, &renderData->meshBBox, settings.solidColorInd);
    } else if (settings.fillType == WgRenderSettingsType::Linear) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 3, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.linear_blend[blendMethodInd]);
        drawMesh(context, &renderData->meshBBox);
    } else if (settings.fillType == WgRenderSettingsType::Radial) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 3, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.radial_blend[blendMethodInd]);
        drawMesh(context, &renderData->meshBBox);
    }
}


void WgCompositor::clipShape(WgContext& context, WgRenderDataShape* renderData)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->renderSettingsShape.skip || renderData->meshShape.vbuffer.count == 0 || renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettingsShape;
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    // setup stencil rules
    WGPURenderPipeline stencilPipeline = (renderData->fillRule == FillRule::NonZero) ? pipelines.nonzero : pipelines.evenodd;
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, stencilPipeline);
    // draw to stencil (first pass)
    drawMesh(context, &renderData->meshShape);
    // merge depth and stencil buffer
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[128], 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.merge_depth_stencil);
    drawMesh(context, &renderData->meshBBox);
    // setup fill rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    if (settings.fillType == WgRenderSettingsType::Solid) {
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.solid);
        drawMeshSolid(context, &renderData->meshBBox, settings.solidColorInd);
    } else if (settings.fillType == WgRenderSettingsType::Linear) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.linear);
        drawMesh(context, &renderData->meshBBox);
    } else if (settings.fillType == WgRenderSettingsType::Radial) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.radial);
        drawMesh(context, &renderData->meshBBox);
    }
}


void WgCompositor::drawStrokes(WgContext& context, WgRenderDataShape* renderData)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->renderSettingsStroke.skip || renderData->meshStrokes.vbuffer.count == 0 || renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettingsStroke;
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    // draw strokes to stencil (first pass)
    // setup stencil rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 255);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.direct);
    // draw to stencil (first pass)
    drawMesh(context, &renderData->meshStrokes);
    // setup fill rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    if (settings.fillType == WgRenderSettingsType::Solid) {
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.solid);
        drawMeshSolid(context, &renderData->meshStrokesBBox, settings.solidColorInd);
    } else if (settings.fillType == WgRenderSettingsType::Linear) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.linear);
        drawMesh(context, &renderData->meshStrokesBBox);
    } else if (settings.fillType == WgRenderSettingsType::Radial) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.radial);
        drawMesh(context, &renderData->meshStrokesBBox);
    }
}


void WgCompositor::blendStrokes(WgContext& context, WgRenderDataShape* renderData, BlendMethod blendMethod)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->renderSettingsStroke.skip || renderData->meshStrokes.vbuffer.count == 0 || renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettingsStroke;
    // copy current render target data to dst target
    WgRenderTarget *target = currentTarget;
    endRenderPass();
    copyTexture(&targetTemp0, target);
    beginRenderPassMS(commandEncoder, target, false);
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    // draw strokes to stencil (first pass)
    // setup stencil rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 255);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.direct);
    // draw to stencil (first pass)
    drawMesh(context, &renderData->meshStrokes);
    // setup fill rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    uint32_t blendMethodInd = (uint32_t)blendMethod;
    if (settings.fillType == WgRenderSettingsType::Solid) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.solid_blend[blendMethodInd]);
        drawMeshSolid(context, &renderData->meshStrokesBBox, settings.solidColorInd);
    } else if (settings.fillType == WgRenderSettingsType::Linear) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 3, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.linear_blend[blendMethodInd]);
        drawMesh(context, &renderData->meshStrokesBBox);
    } else if (settings.fillType == WgRenderSettingsType::Radial) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 3, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.radial_blend[blendMethodInd]);
        drawMesh(context, &renderData->meshStrokesBBox);
    }
};


void WgCompositor::clipStrokes(WgContext& context, WgRenderDataShape* renderData)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->renderSettingsStroke.skip || renderData->meshStrokes.vbuffer.count == 0 || renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettingsStroke;
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    // draw strokes to stencil (first pass)
    // setup stencil rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 255);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.direct);
    // draw to stencil (first pass)
    drawMesh(context, &renderData->meshStrokes);
    // merge depth and stencil buffer
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[128], 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.merge_depth_stencil);
    drawMesh(context, &renderData->meshBBox);
    // setup fill rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    if (settings.fillType == WgRenderSettingsType::Solid) {
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.solid);
        drawMeshSolid(context, &renderData->meshStrokesBBox, settings.solidColorInd);
    } else if (settings.fillType == WgRenderSettingsType::Linear) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.linear);
        drawMesh(context, &renderData->meshStrokesBBox);
    } else if (settings.fillType == WgRenderSettingsType::Radial) {
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, settings.gradientData.bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.radial);
        drawMesh(context, &renderData->meshStrokesBBox);
    }
}


void WgCompositor::drawImage(WgContext& context, WgRenderDataPicture* renderData)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettings;
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    // draw stencil
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 255);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.direct);
    drawMeshImage(context, &renderData->meshData);
    // draw image
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, renderData->imageData.bindGroup, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.image);
    drawMeshImage(context, &renderData->meshData);
}


void WgCompositor::blendImage(WgContext& context, WgRenderDataPicture* renderData, BlendMethod blendMethod)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettings;
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    // copy current render target data to dst target
    WgRenderTarget *target = currentTarget;
    endRenderPass();
    copyTexture(&targetTemp0, target);
    beginRenderPassMS(commandEncoder, target, false);
    // setup stencil rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 255);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.direct);
    drawMeshImage(context, &renderData->meshData);
    // blend image
    uint32_t blendMethodInd = (uint32_t)blendMethod;
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, renderData->imageData.bindGroup, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 3, targetTemp0.bindGroupTexture, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.image_blend[blendMethodInd]);
    drawMeshImage(context, &renderData->meshData);
};


void WgCompositor::clipImage(WgContext& context, WgRenderDataPicture* renderData)
{
    assert(renderData);
    assert(renderPassEncoder);
    if (renderData->viewport.invalid()) return;
    WgRenderSettings& settings = renderData->renderSettings;
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    // setup stencil rules
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 255);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.direct);
    drawMeshImage(context, &renderData->meshData);
    // merge depth and stencil buffer
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[128], 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.merge_depth_stencil);
    drawMeshImage(context, &renderData->meshData);
    // draw image
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, stageBufferPaint[settings.bindGroupInd], 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, renderData->imageData.bindGroup, 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.image);
    drawMeshImage(context, &renderData->meshData);
}


void WgCompositor::drawScene(WgContext& context, WgRenderTarget* scene, WgCompose* compose)
{
    assert(scene);
    assert(compose);
    assert(currentTarget);
    // draw scene
    RenderRegion rect = shrinkRenderRegion(compose->aabb);
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, rect.x(), rect.y(), rect.w(), rect.h());
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, scene->bindGroupTexture, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[compose->opacity], 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.scene);
    drawMeshImage(context, &meshDataBlit);
}


void WgCompositor::blendScene(WgContext& context, WgRenderTarget* scene, WgCompose* compose)
{
    assert(scene);
    assert(compose);
    assert(currentTarget);
    // copy current render target data to dst target
    WgRenderTarget *target = currentTarget;
    endRenderPass();
    copyTexture(&targetTemp0, target);
    beginRenderPassMS(commandEncoder, target, false);
    // blend scene
    uint32_t blendMethodInd = (uint32_t)compose->blend;
    RenderRegion rect = shrinkRenderRegion(compose->aabb);
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, rect.x(), rect.y(), rect.w(), rect.h());
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, scene->bindGroupTexture, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, targetTemp0.bindGroupTexture, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, bindGroupOpacities[compose->opacity], 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.scene_blend[blendMethodInd]);
    drawMeshImage(context, &meshDataBlit);
}


void WgCompositor::markupClipPath(WgContext& context, WgRenderDataShape* renderData)
{
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, renderData->viewport.x(), renderData->viewport.y(), renderData->viewport.w(), renderData->viewport.h());
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    // markup stencil
    if (renderData->meshStrokes.vbuffer.count > 0) {
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 255);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.direct);
        drawMesh(context, &renderData->meshStrokes);
    } else if (renderData->meshShape.vbuffer.count > 0) {
        WGPURenderPipeline stencilPipeline = (renderData->fillRule == FillRule::NonZero) ? pipelines.nonzero : pipelines.evenodd;
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, stencilPipeline);
        drawMesh(context, &renderData->meshShape);
    }
}


void WgCompositor::renderClipPath(WgContext& context, WgRenderDataPaint* paint)
{
    assert(paint);
    assert(renderPassEncoder);
    assert(paint->clips.count > 0);
    // reset scissor recr to full screen
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, 0, 0, width, height);
    // get render data
    WgRenderDataShape* renderData0 = (WgRenderDataShape*)paint->clips[0];
    // markup stencil
    markupClipPath(context, renderData0);
    // copy stencil to depth
    wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[128], 0, nullptr);
    wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.copy_stencil_to_depth);
    drawMesh(context, &renderData0->meshBBox);
    // merge clip paths with AND logic
    for (auto p = paint->clips.begin() + 1; p < paint->clips.end(); ++p) {
        // get render data
        WgRenderDataShape* renderData = (WgRenderDataShape*)(*p);
        // markup stencil
        markupClipPath(context, renderData);
        // copy stencil to depth (clear stencil)
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[190], 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.copy_stencil_to_depth_interm);
        drawMesh(context, &renderData->meshBBox);
        // copy depth to stencil
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 1);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[190], 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.copy_depth_to_stencil);
        drawMesh(context, &renderData->meshBBox);
        // clear depth current (keep stencil)
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[255], 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.clear_depth);
        drawMesh(context, &renderData->meshBBox);
        // clear depth original (keep stencil)
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[255], 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.clear_depth);
        drawMesh(context, &renderData0->meshBBox);
        // copy stencil to depth (clear stencil)
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[128], 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.copy_stencil_to_depth);
        drawMesh(context, &renderData->meshBBox);
    }
}


void WgCompositor::clearClipPath(WgContext& context, WgRenderDataPaint* paint)
{
    assert(paint);
    assert(renderPassEncoder);
    assert(paint->clips.count > 0);
    // reset scissor recr to full screen
    wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, 0, 0, width, height);
    // get render data
    ARRAY_FOREACH(p, paint->clips) {
        WgRenderDataShape* renderData = (WgRenderDataShape*)(*p);
        // set transformations
        wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, 0);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroupViewMat, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, bindGroupOpacities[255], 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.clear_depth);
        drawMesh(context, &renderData->meshBBox);
    }
}


bool WgCompositor::gaussianBlur(WgContext& context, WgRenderTarget* dst, const RenderEffectGaussianBlur* params, const WgCompose* compose)
{
    assert(dst);
    assert(params);
    assert(params->rd);
    assert(!renderPassEncoder);

    auto renderDataParams = (WgRenderDataEffectParams*)params->rd;
    auto aabb = shrinkRenderRegion(compose->aabb);

    copyTexture(&targetTemp0, dst);
    if (params->direction == 0) { // both
        beginRenderPass(commandEncoder, &targetTemp0); {
            wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, dst->bindGroupTexture, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
            wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.gaussian_horz);
            drawMeshImage(context, &meshDataBlit);
        } endRenderPass();
        beginRenderPass(commandEncoder, dst); {
            wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, targetTemp0.bindGroupTexture, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
            wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.gaussian_vert);
            drawMeshImage(context, &meshDataBlit);
        } endRenderPass();
    } else if (params->direction == 1) { // horizontal
        beginRenderPass(commandEncoder, dst); {
            wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, targetTemp0.bindGroupTexture, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
            wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.gaussian_horz);
            drawMeshImage(context, &meshDataBlit);
        } endRenderPass();
    } else if (params->direction == 2) { // vertical
        beginRenderPass(commandEncoder, dst); {
            wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, targetTemp0.bindGroupTexture, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
            wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.gaussian_vert);
            drawMeshImage(context, &meshDataBlit);
        } endRenderPass();
    }

    return true;
}


bool WgCompositor::dropShadow(WgContext& context, WgRenderTarget* dst, const RenderEffectDropShadow* params, const WgCompose* compose)
{
    assert(dst);
    assert(params);
    assert(params->rd);
    assert(!renderPassEncoder);

    auto renderDataParams = (WgRenderDataEffectParams*)params->rd;
    auto aabb = shrinkRenderRegion(compose->aabb);

    copyTexture(&targetTemp0, dst);
    copyTexture(&targetTemp1, dst);
    if (!tvg::zero(params->sigma)) {
        // horizontal
        beginRenderPass(commandEncoder, &targetTemp0); {
            wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, dst->bindGroupTexture, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
            wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.gaussian_horz);
            drawMeshImage(context, &meshDataBlit);
        } endRenderPass();
        // vertical
        beginRenderPass(commandEncoder, &targetTemp1); {
            wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, targetTemp0.bindGroupTexture, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
            wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.gaussian_vert);
            drawMeshImage(context, &meshDataBlit);
        } endRenderPass();
    }
    // drop shadow
    copyTexture(&targetTemp0, dst, aabb);
    beginRenderPass(commandEncoder, dst); {
        wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, targetTemp1.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 2, renderDataParams->bindGroupParams, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.dropshadow);
        drawMeshImage(context, &meshDataBlit);
    } endRenderPass();
    return true;
}


bool WgCompositor::fillEffect(WgContext& context, WgRenderTarget* dst, const RenderEffectFill* params, const WgCompose* compose)
{
    assert(dst);
    assert(params);
    assert(params->rd);
    assert(!renderPassEncoder);

    auto renderDataParams = (WgRenderDataEffectParams*)params->rd;
    auto aabb = shrinkRenderRegion(compose->aabb);

    copyTexture(&targetTemp0, dst, aabb);
    beginRenderPass(commandEncoder, dst); {
        wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.fill_effect);
        drawMeshImage(context, &meshDataBlit);
    } endRenderPass();

    return true;
}


bool WgCompositor::tintEffect(WgContext& context, WgRenderTarget* dst, const RenderEffectTint* params, const WgCompose* compose)
{
    assert(dst);
    assert(params);
    assert(params->rd);
    assert(!renderPassEncoder);

    auto renderDataParams = (WgRenderDataEffectParams*)params->rd;
    auto aabb = shrinkRenderRegion(compose->aabb);

    copyTexture(&targetTemp0, dst, aabb);
    beginRenderPass(commandEncoder, dst); {
        wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.tint_effect);
        drawMeshImage(context, &meshDataBlit);
    } endRenderPass();

    return true;
}

bool WgCompositor::tritoneEffect(WgContext& context, WgRenderTarget* dst, const RenderEffectTritone* params, const WgCompose* compose)
{
    assert(dst);
    assert(params);
    assert(params->rd);
    assert(!renderPassEncoder);

    auto renderDataParams = (WgRenderDataEffectParams*)params->rd;
    auto aabb = shrinkRenderRegion(compose->aabb);

    copyTexture(&targetTemp0, dst, aabb);
    beginRenderPass(commandEncoder, dst); {
        wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, aabb.x(), aabb.y(), aabb.w(), aabb.h());
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, targetTemp0.bindGroupTexture, 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 1, renderDataParams->bindGroupParams, 0, nullptr);
        wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipelines.tritone_effect);
        drawMeshImage(context, &meshDataBlit);
    } endRenderPass();

    return true;
}
