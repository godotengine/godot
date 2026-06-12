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

#include "tvgWgBindGroups.h"
#include <cassert>

WGPUBindGroup WgBindGroupLayouts::createBindGroupTexSampled(WGPUSampler sampler, WGPUTextureView texView)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .sampler = sampler },
        { .binding = 1, .textureView = texView }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutTexSampled, .entryCount = 2, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupTexSampledBuff1Un(WGPUSampler sampler, WGPUTextureView texView, WGPUBuffer buff)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .sampler = sampler },
        { .binding = 1, .textureView = texView },
        { .binding = 2, .buffer = buff, .size = wgpuBufferGetSize(buff) }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutTexSampledBuff1Un, .entryCount = 3, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupTexSampledBuff2Un(WGPUSampler sampler, WGPUTextureView texView, WGPUBuffer buff0, WGPUBuffer buff1)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .sampler = sampler },
        { .binding = 1, .textureView = texView },
        { .binding = 2, .buffer = buff0, .size = wgpuBufferGetSize(buff0) },
        { .binding = 3, .buffer = buff1, .size = wgpuBufferGetSize(buff1) }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutTexSampledBuff2Un, .entryCount = 4, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupStrorage1WO(WGPUTextureView texView)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .textureView = texView }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutTexStrorage1WO, .entryCount = 1, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupStrorage1RO(WGPUTextureView texView)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .textureView = texView }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutTexStrorage1RO, .entryCount = 1, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupStrorage2RO(WGPUTextureView texView0, WGPUTextureView texView1)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .textureView = texView0 },
        { .binding = 1, .textureView = texView1 }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutTexStrorage2RO, .entryCount = 2, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupStrorage3RO(WGPUTextureView texView0, WGPUTextureView texView1, WGPUTextureView texView2)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .textureView = texView0 },
        { .binding = 1, .textureView = texView1 },
        { .binding = 2, .textureView = texView2 }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutTexStrorage3RO, .entryCount = 3, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupBuffer1Un(WGPUBuffer buff)
{
    return createBindGroupBuffer1Un(buff, 0, wgpuBufferGetSize(buff));
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupBuffer1Un(WGPUBuffer buff, uint64_t offset, uint64_t size)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .buffer = buff, .offset = offset, .size = size }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutBuffer1Un, .entryCount = 1, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupBuffer2Un(WGPUBuffer buff0, WGPUBuffer buff1)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .buffer = buff0, .size = wgpuBufferGetSize(buff0) },
        { .binding = 1, .buffer = buff1, .size = wgpuBufferGetSize(buff1) }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutBuffer2Un, .entryCount = 2, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


WGPUBindGroup WgBindGroupLayouts::createBindGroupBuffer3Un(WGPUBuffer buff0, WGPUBuffer buff1, WGPUBuffer buff2)
{
    const WGPUBindGroupEntry bindGroupEntrys[] = {
        { .binding = 0, .buffer = buff0, .size = wgpuBufferGetSize(buff0) },
        { .binding = 1, .buffer = buff1, .size = wgpuBufferGetSize(buff1) },
        { .binding = 2, .buffer = buff2, .size = wgpuBufferGetSize(buff2) }
    };
    const WGPUBindGroupDescriptor bindGroupDesc { .layout = layoutBuffer3Un, .entryCount = 3, .entries = bindGroupEntrys };
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}


void WgBindGroupLayouts::releaseBindGroup(WGPUBindGroup& bindGroup)
{
    if (bindGroup) wgpuBindGroupRelease(bindGroup);
    bindGroup = nullptr;
}


void WgBindGroupLayouts::releaseBindGroupLayout(WGPUBindGroupLayout& bindGroupLayout)
{
    if (bindGroupLayout) wgpuBindGroupLayoutRelease(bindGroupLayout);
    bindGroupLayout = nullptr;
}


void WgBindGroupLayouts::initialize(WGPUDevice device)
{
    // store device handle
    assert(device);
    this->device = device;

    // common bind group settings
    const WGPUShaderStage visibility_vert = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute;
    const WGPUShaderStage visibility_frag = WGPUShaderStage_Fragment | WGPUShaderStage_Compute;
    const WGPUSamplerBindingLayout sampler = { .type = WGPUSamplerBindingType_Filtering };
    const WGPUTextureBindingLayout texture = { .sampleType = WGPUTextureSampleType_Float, .viewDimension = WGPUTextureViewDimension_2D };
    const WGPUStorageTextureBindingLayout storageTextureWO { .access = WGPUStorageTextureAccess_WriteOnly, .format = WGPUTextureFormat_RGBA8Unorm, .viewDimension = WGPUTextureViewDimension_2D };
    const WGPUBufferBindingLayout bufferUniform { .type = WGPUBufferBindingType_Uniform };

    // bind group layout tex sampled with buffer uniforms
    const WGPUBindGroupLayoutEntry entriesTexSampledBufferUniforms[] {
        { .binding = 0, .visibility = visibility_frag, .sampler = sampler },
        { .binding = 1, .visibility = visibility_frag, .texture = texture },
        { .binding = 2, .visibility = visibility_vert, .buffer = bufferUniform },
        { .binding = 3, .visibility = visibility_vert, .buffer = bufferUniform }
    };
    const WGPUBindGroupLayoutDescriptor layoutDescTexSambled        { .entryCount = 2, .entries = entriesTexSampledBufferUniforms };
    const WGPUBindGroupLayoutDescriptor layoutDescTexSampledBuff1Un { .entryCount = 3, .entries = entriesTexSampledBufferUniforms };
    const WGPUBindGroupLayoutDescriptor layoutDescTexSampledBuff2Un { .entryCount = 4, .entries = entriesTexSampledBufferUniforms };
    layoutTexSampled        = wgpuDeviceCreateBindGroupLayout(device, &layoutDescTexSambled);
    layoutTexSampledBuff1Un = wgpuDeviceCreateBindGroupLayout(device, &layoutDescTexSampledBuff1Un);
    layoutTexSampledBuff2Un = wgpuDeviceCreateBindGroupLayout(device, &layoutDescTexSampledBuff2Un);
    assert(layoutTexSampled);
    assert(layoutTexSampledBuff1Un);
    assert(layoutTexSampledBuff2Un);

    // bind group layout tex storages WO
    const WGPUBindGroupLayoutEntry entriesTexStoragesWO[] {
        { .binding = 0, .visibility = visibility_frag, .storageTexture = storageTextureWO }
    };
    const WGPUBindGroupLayoutDescriptor layoutDescTexStrorage1WO { .entryCount = 1, .entries = entriesTexStoragesWO };
    layoutTexStrorage1WO = wgpuDeviceCreateBindGroupLayout(device, &layoutDescTexStrorage1WO);
    assert(layoutTexStrorage1WO);

    // bind group layout tex storages RO
    const WGPUBindGroupLayoutEntry entriesTexStoragesRO[] {
        { .binding = 0, .visibility = visibility_frag, .texture = texture },
        { .binding = 1, .visibility = visibility_frag, .texture = texture },
        { .binding = 2, .visibility = visibility_frag, .texture = texture }
    };
    const WGPUBindGroupLayoutDescriptor layoutDescTexStorages1RO { .entryCount = 1, .entries = entriesTexStoragesRO };
    const WGPUBindGroupLayoutDescriptor layoutDescTexStorages2RO { .entryCount = 2, .entries = entriesTexStoragesRO };
    const WGPUBindGroupLayoutDescriptor layoutDescTexStorages3RO { .entryCount = 3, .entries = entriesTexStoragesRO };
    layoutTexStrorage1RO = wgpuDeviceCreateBindGroupLayout(device, &layoutDescTexStorages1RO);
    layoutTexStrorage2RO = wgpuDeviceCreateBindGroupLayout(device, &layoutDescTexStorages2RO);
    layoutTexStrorage3RO = wgpuDeviceCreateBindGroupLayout(device, &layoutDescTexStorages3RO);
    assert(layoutTexStrorage1RO);
    assert(layoutTexStrorage2RO);
    assert(layoutTexStrorage3RO);

    // bind group layout buffer uniforms
    const WGPUBindGroupLayoutEntry entriesBufferUniform[] {
        { .binding = 0, .visibility = visibility_vert, .buffer = bufferUniform },
        { .binding = 1, .visibility = visibility_vert, .buffer = bufferUniform },
        { .binding = 2, .visibility = visibility_vert, .buffer = bufferUniform }
    };
    const WGPUBindGroupLayoutDescriptor layoutDescBufferUniforms1Un { .entryCount = 1, .entries = entriesBufferUniform };
    const WGPUBindGroupLayoutDescriptor layoutDescBufferUniforms2Un { .entryCount = 2, .entries = entriesBufferUniform };
    const WGPUBindGroupLayoutDescriptor layoutDescBufferUniforms3Un { .entryCount = 3, .entries = entriesBufferUniform };
    layoutBuffer1Un = wgpuDeviceCreateBindGroupLayout(device, &layoutDescBufferUniforms1Un);
    layoutBuffer2Un = wgpuDeviceCreateBindGroupLayout(device, &layoutDescBufferUniforms2Un);
    layoutBuffer3Un = wgpuDeviceCreateBindGroupLayout(device, &layoutDescBufferUniforms3Un);
    assert(layoutBuffer1Un);
    assert(layoutBuffer2Un);
    assert(layoutBuffer3Un);
}


void WgBindGroupLayouts::release()
{
    releaseBindGroupLayout(layoutBuffer3Un);
    releaseBindGroupLayout(layoutBuffer2Un);
    releaseBindGroupLayout(layoutBuffer1Un);
    releaseBindGroupLayout(layoutTexStrorage3RO);
    releaseBindGroupLayout(layoutTexStrorage2RO);
    releaseBindGroupLayout(layoutTexStrorage1RO);
    releaseBindGroupLayout(layoutTexStrorage1WO);
    releaseBindGroupLayout(layoutTexSampledBuff1Un);
    releaseBindGroupLayout(layoutTexSampledBuff2Un);
    releaseBindGroupLayout(layoutTexSampled);
    device = nullptr;
}
