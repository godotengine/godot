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

#ifndef _TVG_WG_BIND_GROUPS_H_
#define _TVG_WG_BIND_GROUPS_H_

#include <webgpu/webgpu.h>

class WgBindGroupLayouts {
private:
    WGPUDevice device{};
public:
    WGPUBindGroupLayout layoutTexSampled{};
    WGPUBindGroupLayout layoutTexSampledBuff1Un{};
    WGPUBindGroupLayout layoutTexSampledBuff2Un{};
    WGPUBindGroupLayout layoutTexStrorage1WO{};
    WGPUBindGroupLayout layoutTexStrorage1RO{};
    WGPUBindGroupLayout layoutTexStrorage2RO{};
    WGPUBindGroupLayout layoutTexStrorage3RO{};
    WGPUBindGroupLayout layoutBuffer1Un{};
    WGPUBindGroupLayout layoutBuffer2Un{};
    WGPUBindGroupLayout layoutBuffer3Un{};

    WGPUBindGroup createBindGroupTexSampled(WGPUSampler sampler, WGPUTextureView texView);
    WGPUBindGroup createBindGroupTexSampledBuff1Un(WGPUSampler sampler, WGPUTextureView texView, WGPUBuffer buff);
    WGPUBindGroup createBindGroupTexSampledBuff2Un(WGPUSampler sampler, WGPUTextureView texView, WGPUBuffer buff0, WGPUBuffer buff1);
    WGPUBindGroup createBindGroupStrorage1WO(WGPUTextureView texView);
    // for read-only access in compute shaders, use texture_2d<f32> instead of texture_storage_2d<rgba8unorm, read>
    WGPUBindGroup createBindGroupStrorage1RO(WGPUTextureView texView);
    WGPUBindGroup createBindGroupStrorage2RO(WGPUTextureView texView0, WGPUTextureView texView1);
    WGPUBindGroup createBindGroupStrorage3RO(WGPUTextureView texView0, WGPUTextureView texView1, WGPUTextureView texView2);
    WGPUBindGroup createBindGroupBuffer1Un(WGPUBuffer buff);
    WGPUBindGroup createBindGroupBuffer1Un(WGPUBuffer buff, uint64_t offset, uint64_t size);
    WGPUBindGroup createBindGroupBuffer2Un(WGPUBuffer buff0, WGPUBuffer buff1);
    WGPUBindGroup createBindGroupBuffer3Un(WGPUBuffer buff0, WGPUBuffer buff1, WGPUBuffer buff2);
    void releaseBindGroup(WGPUBindGroup& bindGroup);
    void releaseBindGroupLayout(WGPUBindGroupLayout& bindGroupLayout);

    void initialize(WGPUDevice device);
    void release();
};

#endif // _TVG_WG_BIND_GROUPS_H_
