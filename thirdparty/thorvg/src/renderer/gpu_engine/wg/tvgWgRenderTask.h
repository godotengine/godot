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

#ifndef _TVG_WG_RENDER_TASK_H_
#define _TVG_WG_RENDER_TASK_H_
 
#include "tvgWgCompositor.h"

// base class for any renderable objects 
struct WgRenderTask {
    virtual ~WgRenderTask() {}
    virtual void run(WgContext& context, WgCompositor& compositor, WGPUCommandEncoder encoder) = 0;
};

// task for single shape rendering
struct WgPaintTask: public WgRenderTask {
    // shape render properties
    WgRenderDataPaint* renderData{};
    BlendMethod blendMethod{};

    WgPaintTask(WgRenderDataPaint* renderData, BlendMethod blendMethod) : 
        renderData(renderData), blendMethod(blendMethod) {}
    // apply shape execution, including custom blending and clipping
    void run(WgContext& context, WgCompositor& compositor, WGPUCommandEncoder encoder) override;
};

// task for scene rendering with blending, composition and effect
struct WgSceneTask: public WgRenderTask {
public:
    // parent scene (nullptr for root scene)
    WgSceneTask* parent{};
    // children can be shapes or scenes tasks
    Array<WgRenderTask*> children;
    // scene blend/compose targets
    WgRenderTarget* renderTarget{};
    WgRenderTarget* renderTargetMsk{};
    WgRenderTarget* renderTargetDst{};
    // scene blend/compose properties
    WgCompose* compose{};
    // scene effect properties
    const RenderEffect* effect{};

    WgSceneTask(WgRenderTarget* renderTarget, WgCompose* compose, WgSceneTask* parent) :
        parent(parent), renderTarget(renderTarget), compose(compose) {}
    // run all, including all shapes drawing, blending, composition and effect
    void run(WgContext& context, WgCompositor& compositor, WGPUCommandEncoder encoder) override;
private:
    void runChildren(WgContext& context, WgCompositor& compositor, WGPUCommandEncoder encoder);
    void runEffect(WgContext& context, WgCompositor& compositor, WGPUCommandEncoder encoder);
};
 
 #endif // _TVG_WG_RENDER_TASK_H_
 
