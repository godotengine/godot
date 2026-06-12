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

#include "tvgWgRenderTarget.h"

void WgRenderTarget::initialize(WgContext& context, uint32_t width, uint32_t height)
{
    this->width = width;
    this->height = height;
    texture = context.createTexStorage(width, height, WGPUTextureFormat_RGBA8Unorm);
    textureMS = context.createTexAttachement(width, height, WGPUTextureFormat_RGBA8Unorm, 4);
    texView = context.createTextureView(texture);
    texViewMS = context.createTextureView(textureMS);
    bindGroupRead = context.layouts.createBindGroupStrorage1RO(texView);
    bindGroupWrite = context.layouts.createBindGroupStrorage1WO(texView);
    bindGroupTexture = context.layouts.createBindGroupTexSampled(context.samplerNearestRepeat, texView);
}


void WgRenderTarget::release(WgContext& context)
{
    context.layouts.releaseBindGroup(bindGroupTexture);
    context.layouts.releaseBindGroup(bindGroupWrite);
    context.layouts.releaseBindGroup(bindGroupRead);
    context.releaseTextureView(texViewMS);
    context.releaseTexture(textureMS);
    context.releaseTextureView(texView);
    context.releaseTexture(texture);
    height = 0;
    width = 0;
}

//*****************************************************************************
// render target pool
//*****************************************************************************

WgRenderTarget* WgRenderTargetPool::allocate(WgContext& context)
{
    WgRenderTarget* renderTarget{};
    if (pool.count > 0) {
        renderTarget = pool.last();
        pool.pop();
    } else {
        renderTarget = new WgRenderTarget;
        renderTarget->initialize(context, width, height);
        list.push(renderTarget);
    }
    return renderTarget;
};


void WgRenderTargetPool::free(WgContext& context, WgRenderTarget* renderTarget)
{
    pool.push(renderTarget);
};


void WgRenderTargetPool::initialize(WgContext& context, uint32_t width, uint32_t height)
{
    this->width = width;
    this->height = height;
}


void WgRenderTargetPool::release(WgContext& context)
{
    ARRAY_FOREACH(p, list) {
       (*p)->release(context);
       delete(*p);
    }
    list.clear();
    pool.clear();
    height = 0;
    width = 0;
};
