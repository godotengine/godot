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

#include "tvgCanvas.h"
#include "tvgTaskScheduler.h"
#include "tvgLoadModule.h"

#ifdef THORVG_SW_RASTER_SUPPORT
    #include "tvgSwRenderer.h"
#endif

#ifdef THORVG_GL_RASTER_SUPPORT
    #include "tvgGlRenderer.h"
#endif

#ifdef THORVG_WG_RASTER_SUPPORT
    #include "tvgWgRenderer.h"
#endif


/************************************************************************/
/* Canvas Class Implementation                                          */
/************************************************************************/

Canvas::Canvas():pImpl(new Impl)
{
}


Canvas::~Canvas()
{
    delete(pImpl);
}


const list<Paint*>& Canvas::paints() const noexcept
{
    return pImpl->scene->paints();
}


Result Canvas::add(Paint* target, Paint* at) noexcept
{
    if (target) return pImpl->add(target, at);
    return Result::InvalidArguments;
}


Result Canvas::draw(bool clear) noexcept
{
    TVGLOG("RENDERER", "Draw S. -------------------------------- Canvas(%p)", this);
    auto ret = pImpl->draw(clear);
    TVGLOG("RENDERER", "Draw E. -------------------------------- Canvas(%p)", this);

    return ret;
}


Result Canvas::update() noexcept
{
    TVGLOG("RENDERER", "Update S. ------------------------------ Canvas(%p)", this);
    auto ret = pImpl->update();
    TVGLOG("RENDERER", "Update E. ------------------------------ Canvas(%p)", this);

    return ret;
}


Result Canvas::remove(Paint* paint) noexcept
{
    return pImpl->remove(paint);
}


Result Canvas::viewport(int32_t x, int32_t y, int32_t w, int32_t h) noexcept
{
    return pImpl->viewport(x, y, w, h);
}


Result Canvas::sync() noexcept
{
    return pImpl->sync();
}


/************************************************************************/
/* SwCanvas Class Implementation                                        */
/************************************************************************/

SwCanvas::SwCanvas() = default;

SwCanvas::~SwCanvas()
{
#ifdef THORVG_SW_RASTER_SUPPORT
    SwRenderer::term();
#endif
}


Result SwCanvas::target(uint32_t* buffer, uint32_t stride, uint32_t w, uint32_t h, ColorSpace cs) noexcept
{
#ifdef THORVG_SW_RASTER_SUPPORT
    if (cs == ColorSpace::Unknown) return Result::InvalidArguments;
    if (cs == ColorSpace::Grayscale8) return Result::NonSupport;

    if (pImpl->status == Status::Updating || pImpl->status == Status::Drawing) {
        return Result::InsufficientCondition;
    }

    //We know renderer type, avoid dynamic_cast for performance.
    auto renderer = static_cast<SwRenderer*>(pImpl->renderer);
    if (!renderer) return Result::MemoryCorruption;

    if (!renderer->target(buffer, stride, w, h, cs)) return Result::InvalidArguments;
    pImpl->vport = {{0, 0}, {(int32_t)w, (int32_t)h}};
    renderer->viewport(pImpl->vport);

    //FIXME: The value must be associated with an individual canvas instance.
    ImageLoader::cs = static_cast<ColorSpace>(cs);

    //Paints must be updated again with this new target.
    pImpl->status = Status::Damaged;

    return Result::Success;
#endif
    return Result::NonSupport;
}


SwCanvas* SwCanvas::gen(EngineOption op) noexcept
{
#ifdef THORVG_SW_RASTER_SUPPORT
    if (engineInit > 0) {
        auto renderer = new SwRenderer(TaskScheduler::threads(), op);
        renderer->ref();
        auto ret = new SwCanvas;
        ret->pImpl->renderer = renderer;
        return ret;
    }
#endif
    return nullptr;
}


/************************************************************************/
/* GlCanvas Class Implementation                                        */
/************************************************************************/

GlCanvas::GlCanvas() = default;

GlCanvas::~GlCanvas()
{
#ifdef THORVG_GL_RASTER_SUPPORT
    GlRenderer::term();
#endif
}


Result GlCanvas::target(void* display, void* surface, void* context, int32_t id, uint32_t w, uint32_t h, ColorSpace cs) noexcept
{
#ifdef THORVG_GL_RASTER_SUPPORT
    if (cs != ColorSpace::ABGR8888S) return Result::NonSupport;

    if (pImpl->status == Status::Updating || pImpl->status == Status::Drawing) {
        return Result::InsufficientCondition;
    }

    //We know renderer type, avoid dynamic_cast for performance.
    auto renderer = static_cast<GlRenderer*>(pImpl->renderer);
    if (!renderer) return Result::MemoryCorruption;

    if (!renderer->target(display, surface, context, id, w, h, cs)) return Result::Unknown;
    pImpl->vport = {{0, 0}, {(int32_t)w, (int32_t)h}};
    renderer->viewport(pImpl->vport);

    //Paints must be updated again with this new target.
    pImpl->status = Status::Damaged;

    return Result::Success;
#endif
    return Result::NonSupport;
}


GlCanvas* GlCanvas::gen(EngineOption op) noexcept
{
#ifdef THORVG_GL_RASTER_SUPPORT
    if (engineInit > 0) {
        if (op == EngineOption::SmartRender) TVGLOG("RENDERER", "GlCanvas doesn't support Smart Rendering");
        auto renderer = GlRenderer::gen(TaskScheduler::threads());
        if (!renderer) return nullptr;
        renderer->ref();
        auto ret = new GlCanvas;
        ret->pImpl->renderer = renderer;
        return ret;
    }
#endif
    return nullptr;
}


/************************************************************************/
/* WgCanvas Class Implementation                                        */
/************************************************************************/

WgCanvas::WgCanvas() = default;

WgCanvas::~WgCanvas()
{
#ifdef THORVG_WG_RASTER_SUPPORT
    auto renderer = static_cast<WgRenderer*>(pImpl->renderer);
    renderer->target(nullptr, nullptr, nullptr, 0, 0, ColorSpace::Unknown);

    WgRenderer::term();
#endif
}


Result WgCanvas::target(void* device, void* instance, void* target, uint32_t w, uint32_t h, ColorSpace cs, int type) noexcept
{
#ifdef THORVG_WG_RASTER_SUPPORT
    if (cs != ColorSpace::ABGR8888S) return Result::NonSupport;

    if (pImpl->status == Status::Updating || pImpl->status == Status::Drawing) {
        return Result::InsufficientCondition;
    }

    //We know renderer type, avoid dynamic_cast for performance.
    auto renderer = static_cast<WgRenderer*>(pImpl->renderer);
    if (!renderer) return Result::MemoryCorruption;

    if (!renderer->target((WGPUDevice)device, (WGPUInstance)instance, target, w, h, cs, type)) return Result::Unknown;
    pImpl->vport = {{0, 0}, {(int32_t)w, (int32_t)h}};
    renderer->viewport(pImpl->vport);

    //Paints must be updated again with this new target.
    pImpl->status = Status::Damaged;

    return Result::Success;
#endif
    return Result::NonSupport;
}


WgCanvas* WgCanvas::gen(EngineOption op) noexcept
{
#ifdef THORVG_WG_RASTER_SUPPORT
    if (engineInit > 0) {
        if (op == EngineOption::SmartRender) TVGLOG("RENDERER", "WgCanvas doesn't support Smart Rendering");
        auto renderer = WgRenderer::gen(TaskScheduler::threads());
        renderer->ref();
        auto ret = new WgCanvas;
        ret->pImpl->renderer = renderer;
        return ret;
    }
#endif
    return nullptr;
}
