/*
 * Copyright (c) 2023 - 2026 ThorVG project. All rights reserved.

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

#include <cassert>
#include "tvgWgCommon.h"
#include "tvgArray.h"

void WgContext::initialize(WGPUInstance instance, WGPUDevice device)
{
    this->instance = instance;
    this->device = device;

    queue = wgpuDeviceGetQueue(device);

    samplerNearestClamp = createSampler(WGPUFilterMode_Nearest, WGPUMipmapFilterMode_Nearest, WGPUAddressMode_ClampToEdge, 1);
    samplerNearestRepeat = createSampler(WGPUFilterMode_Nearest, WGPUMipmapFilterMode_Nearest, WGPUAddressMode_Repeat);
    samplerLinearRepeat = createSampler(WGPUFilterMode_Linear, WGPUMipmapFilterMode_Linear, WGPUAddressMode_Repeat, 4);
    samplerLinearMirror = createSampler(WGPUFilterMode_Linear, WGPUMipmapFilterMode_Linear, WGPUAddressMode_MirrorRepeat, 4);
    samplerLinearClamp = createSampler(WGPUFilterMode_Linear, WGPUMipmapFilterMode_Linear, WGPUAddressMode_ClampToEdge, 4);

    layouts.initialize(device);
}


void WgContext::release()
{
    layouts.release();
    releaseSampler(samplerLinearClamp);
    releaseSampler(samplerLinearMirror);
    releaseSampler(samplerLinearRepeat);
    releaseSampler(samplerNearestRepeat);
    releaseSampler(samplerNearestClamp);
    releaseQueue(queue);
}


WGPUSampler WgContext::createSampler(WGPUFilterMode filter, WGPUMipmapFilterMode mipmapFilter, WGPUAddressMode addrMode, uint16_t anisotropy)
{
    const WGPUSamplerDescriptor samplerDesc {
        .addressModeU = addrMode, .addressModeV = addrMode, .addressModeW = addrMode,
        .magFilter = filter, .minFilter = filter, .mipmapFilter = mipmapFilter,
        .lodMinClamp = 0.0f, .lodMaxClamp = 32.0f, .maxAnisotropy = anisotropy
    };
    return wgpuDeviceCreateSampler(device, &samplerDesc);
}


bool WgContext::allocateTexture(WGPUTexture& texture, uint32_t width, uint32_t height, WGPUTextureFormat format, void* data)
{
    if ((texture) && (wgpuTextureGetWidth(texture) == width) && (wgpuTextureGetHeight(texture) == height)) {
        // update texture data
        const WGPUTexelCopyTextureInfo copyTextureInfo{ .texture = texture };
        const WGPUTexelCopyBufferLayout copyBufferLayout{ .bytesPerRow = 4 * width, .rowsPerImage = height };
        const WGPUExtent3D writeSize{ .width = width, .height = height, .depthOrArrayLayers = 1 };
        wgpuQueueWriteTexture(queue, &copyTextureInfo, data, 4 * width * height, &copyBufferLayout, &writeSize);
    } else {
        releaseTexture(texture);
        texture = createTexture(width, height, format);
        // update texture data
        const WGPUTexelCopyTextureInfo copyTextureInfo{ .texture = texture };
        const WGPUTexelCopyBufferLayout copyBufferLayout{ .bytesPerRow = 4 * width, .rowsPerImage = height };
        const WGPUExtent3D writeSize{ .width = width, .height = height, .depthOrArrayLayers = 1 };
        wgpuQueueWriteTexture(queue, &copyTextureInfo, data, 4 * width * height, &copyBufferLayout, &writeSize);
        return true;
    }
    return false;

}


WGPUTexture WgContext::createTexture(uint32_t width, uint32_t height, WGPUTextureFormat format)
{
    const WGPUTextureDescriptor textureDesc {
        .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
        .dimension = WGPUTextureDimension_2D, .size = { width, height, 1 },
        .format = format, .mipLevelCount = 1, .sampleCount = 1
    };
    return wgpuDeviceCreateTexture(device, &textureDesc);
}


WGPUTexture WgContext::createTexStorage(uint32_t width, uint32_t height, WGPUTextureFormat format)
{
    const WGPUTextureDescriptor textureDesc {
        .usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_RenderAttachment,
        .dimension = WGPUTextureDimension_2D, .size = { width, height, 1 },
        .format = format, .mipLevelCount = 1, .sampleCount = 1
    };
    return wgpuDeviceCreateTexture(device, &textureDesc);
}


WGPUTexture WgContext::createTexAttachement(uint32_t width, uint32_t height, WGPUTextureFormat format, uint32_t sc)
{
    const WGPUTextureDescriptor textureDesc {
        .usage = WGPUTextureUsage_RenderAttachment,
        .dimension = WGPUTextureDimension_2D, .size = { width, height, 1 },
        .format = format, .mipLevelCount = 1, .sampleCount = sc
    };
    return wgpuDeviceCreateTexture(device, &textureDesc);
}


WGPUTextureView WgContext::createTextureView(WGPUTexture texture)
{
    const WGPUTextureViewDescriptor textureViewDesc {
        .format = wgpuTextureGetFormat(texture),
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All
    };
    return wgpuTextureCreateView(texture, &textureViewDesc);
}


void WgContext::releaseTextureView(WGPUTextureView& textureView)
{
    if (textureView) {
        wgpuTextureViewRelease(textureView);
        textureView = nullptr;
    }
}


void WgContext::releaseTexture(WGPUTexture& texture)
{
    if (texture) {
        wgpuTextureDestroy(texture);
        wgpuTextureRelease(texture);
        texture = nullptr;
    }
    
}


void WgContext::releaseSampler(WGPUSampler& sampler)
{
    if (sampler) {
        wgpuSamplerRelease(sampler);
        sampler = nullptr;
    }
}


bool WgContext::allocateBufferUniform(WGPUBuffer& buffer, const void* data, uint64_t size)
{
    if ((buffer) && (wgpuBufferGetSize(buffer) >= size))
        wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
    else {
        releaseBuffer(buffer);
        const WGPUBufferDescriptor bufferDesc { .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, .size = size };
        buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
        wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
        return true;
    }
    return false;
}


bool WgContext::allocateBufferVertex(WGPUBuffer& buffer, const float* data, uint64_t size)
{
    if ((buffer) && (wgpuBufferGetSize(buffer) >= size))
        wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
    else {
        releaseBuffer(buffer);
        const WGPUBufferDescriptor bufferDesc { .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex, .size = size };
        buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
        wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
        return true;
    }
    return false;
}


bool WgContext::allocateBufferIndex(WGPUBuffer& buffer, const uint32_t* data, uint64_t size)
{
    if ((buffer) && (wgpuBufferGetSize(buffer) >= size))
        wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
    else {
        releaseBuffer(buffer);
        const WGPUBufferDescriptor bufferDesc { .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index, .size = size };
        buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
        wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
        return true;
    }
    return false;
}


void WgContext::releaseBuffer(WGPUBuffer& buffer)
{
    if (buffer) { 
        wgpuBufferDestroy(buffer);
        wgpuBufferRelease(buffer);
        buffer = nullptr;
    }
}

void WgContext::releaseQueue(WGPUQueue& queue)
{
    if (queue) {
        wgpuQueueRelease(queue);
        queue = nullptr;
    }
}


WGPUCommandEncoder WgContext::createCommandEncoder()
{
    WGPUCommandEncoderDescriptor commandEncoderDesc{};
    return wgpuDeviceCreateCommandEncoder(device, &commandEncoderDesc);
}


void WgContext::submitCommandEncoder(WGPUCommandEncoder commandEncoder)
{
    const WGPUCommandBufferDescriptor commandBufferDesc{};
    WGPUCommandBuffer commandsBuffer = wgpuCommandEncoderFinish(commandEncoder, &commandBufferDesc);
    wgpuQueueSubmit(queue, 1, &commandsBuffer);
    wgpuCommandBufferRelease(commandsBuffer);
}


void WgContext::releaseCommandEncoder(WGPUCommandEncoder& commandEncoder)
{
    if (commandEncoder) {
        wgpuCommandEncoderRelease(commandEncoder);
        commandEncoder = nullptr;
    }
}


void WgContext::submit()
{
    wgpuQueueSubmit(queue, 0, nullptr);
}


bool WgContext::invalid()
{
    return !instance || !device;
}
