/*
 * Copyright (c) 2020 - 2022 Samsung Electronics Co., Ltd. All rights reserved.

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
#include "tvgMath.h"
#include "tvgSwCommon.h"
#include "tvgTaskScheduler.h"
#include "tvgSwRenderer.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/
static int32_t initEngineCnt = false;
static int32_t rendererCnt = 0;
static SwMpool* globalMpool = nullptr;
static uint32_t threadsCnt = 0;

struct SwTask : Task
{
    Matrix* transform = nullptr;
    SwSurface* surface = nullptr;
    SwMpool* mpool = nullptr;
    RenderUpdateFlag flags = RenderUpdateFlag::None;
    Array<RenderData> clips;
    uint32_t opacity;
    SwBBox bbox = {{0, 0}, {0, 0}};       //Whole Rendering Region
    bool pushed = false;                  //Pushed into task list?
    bool disposed = false;                //Disposed task?

    RenderRegion bounds() const
    {
        RenderRegion region;

        //Range over?
        region.x = bbox.min.x > 0 ? bbox.min.x : 0;
        region.y = bbox.min.y > 0 ? bbox.min.y : 0;
        region.w = bbox.max.x - region.x;
        region.h = bbox.max.y - region.y;
        if (region.w < 0) region.w = 0;
        if (region.h < 0) region.h = 0;

        return region;
    }

    virtual bool dispose() = 0;

    virtual ~SwTask()
    {
        free(transform);
    }
};


struct SwShapeTask : SwTask
{
    SwShape shape;
    const Shape* sdata = nullptr;
    bool cmpStroking = false;

    void run(unsigned tid) override
    {
        if (opacity == 0) return;  //Invisible

        uint8_t strokeAlpha = 0;
        auto visibleStroke = false;
        bool visibleFill = false;
        auto clipRegion = bbox;

        if (HALF_STROKE(sdata->strokeWidth()) > 0) {
            sdata->strokeColor(nullptr, nullptr, nullptr, &strokeAlpha);
            visibleStroke = sdata->strokeFill() || (static_cast<uint32_t>(strokeAlpha * opacity / 255) > 0);
        }

        //This checks also for the case, if the invisible shape turned to visible by alpha.
        auto prepareShape = false;
        if (!shapePrepared(&shape) && (flags & RenderUpdateFlag::Color)) prepareShape = true;

        //Shape
        if (flags & (RenderUpdateFlag::Path | RenderUpdateFlag::Transform) || prepareShape) {
            uint8_t alpha = 0;
            sdata->fillColor(nullptr, nullptr, nullptr, &alpha);
            alpha = static_cast<uint8_t>(static_cast<uint32_t>(alpha) * opacity / 255);
            visibleFill = (alpha > 0 || sdata->fill());
            if (visibleFill || visibleStroke) {
                shapeReset(&shape);
                if (!shapePrepare(&shape, sdata, transform, clipRegion, bbox, mpool, tid, clips.count > 0 ? true : false)) goto err;
            }
        }

        //Decide Stroking Composition
        if (visibleStroke && visibleFill && opacity < 255) cmpStroking = true;
        else cmpStroking = false;

        //Fill
        if (flags & (RenderUpdateFlag::Gradient | RenderUpdateFlag::Transform | RenderUpdateFlag::Color)) {
            if (visibleFill) {
                /* We assume that if stroke width is bigger than 2,
                   shape outline below stroke could be full covered by stroke drawing.
                   Thus it turns off antialising in that condition.
                   Also, it shouldn't be dash style. */
                auto antiAlias = (strokeAlpha == 255 && sdata->strokeWidth() > 2 && sdata->strokeDash(nullptr) == 0) ? false : true;

                if (!shapeGenRle(&shape, sdata, antiAlias)) goto err;
            }
            if (auto fill = sdata->fill()) {
                auto ctable = (flags & RenderUpdateFlag::Gradient) ? true : false;
                if (ctable) shapeResetFill(&shape);
                if (!shapeGenFillColors(&shape, fill, transform, surface, cmpStroking ? 255 : opacity, ctable)) goto err;
            } else {
                shapeDelFill(&shape);
            }
        }

        //Stroke
        if (flags & (RenderUpdateFlag::Stroke | RenderUpdateFlag::Transform)) {
            if (visibleStroke) {
                shapeResetStroke(&shape, sdata, transform);
                if (!shapeGenStrokeRle(&shape, sdata, transform, clipRegion, bbox, mpool, tid)) goto err;

                if (auto fill = sdata->strokeFill()) {
                    auto ctable = (flags & RenderUpdateFlag::GradientStroke) ? true : false;
                    if (ctable) shapeResetStrokeFill(&shape);
                    if (!shapeGenStrokeFillColors(&shape, fill, transform, surface, cmpStroking ? 255 : opacity, ctable)) goto err;
                } else {
                    shapeDelStrokeFill(&shape);
                }
            } else {
                shapeDelStroke(&shape);
            }
        }

        //Clip Path
        for (auto clip = clips.data; clip < (clips.data + clips.count); ++clip) {
            auto clipper = &static_cast<SwShapeTask*>(*clip)->shape;
            //Clip shape rle
            if (shape.rle) {
                if (clipper->fastTrack) rleClipRect(shape.rle, &clipper->bbox);
                else if (clipper->rle) rleClipPath(shape.rle, clipper->rle);
                else goto err;
            }
            //Clip stroke rle
            if (shape.strokeRle) {
                if (clipper->fastTrack) rleClipRect(shape.strokeRle, &clipper->bbox);
                else if (clipper->rle) rleClipPath(shape.strokeRle, clipper->rle);
                else goto err;
            }
        }
        goto end;

    err:
        shapeReset(&shape);
    end:
        shapeDelOutline(&shape, mpool, tid);
    }

    bool dispose() override
    {
       shapeFree(&shape);
       return true;
    }
};


struct SwImageTask : SwTask
{
    SwImage image;

    void run(unsigned tid) override
    {
        auto clipRegion = bbox;

        //Invisible shape turned to visible by alpha.
        if ((flags & (RenderUpdateFlag::Image | RenderUpdateFlag::Transform | RenderUpdateFlag::Color)) && (opacity > 0)) {
            imageReset(&image);
            if (!image.data || image.w == 0 || image.h == 0) goto end;

            if (!imagePrepare(&image, transform, clipRegion, bbox, mpool, tid)) goto end;

            if (clips.count > 0) {
                if (!imageGenRle(&image, bbox, false)) goto end;
                if (image.rle) {
                    for (auto clip = clips.data; clip < (clips.data + clips.count); ++clip) {
                        auto clipper = &static_cast<SwShapeTask*>(*clip)->shape;
                        if (clipper->fastTrack) rleClipRect(image.rle, &clipper->bbox);
                        else if (clipper->rle) rleClipPath(image.rle, clipper->rle);
                        else goto err;
                    }
                }
            }
        }
        goto end;

    err:
        rleReset(image.rle);
    end:
        imageDelOutline(&image, mpool, tid);
    }

    bool dispose() override
    {
       imageFree(&image);
       return true;
    }
};


static void _termEngine()
{
    if (rendererCnt > 0) return;

    mpoolTerm(globalMpool);
    globalMpool = nullptr;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

SwRenderer::~SwRenderer()
{
    clearCompositors();

    if (surface) delete(surface);

    if (!sharedMpool) mpoolTerm(mpool);

    --rendererCnt;

    if (rendererCnt == 0 && initEngineCnt == 0) _termEngine();
}


bool SwRenderer::clear()
{
    for (auto task = tasks.data; task < (tasks.data + tasks.count); ++task) {
        if ((*task)->disposed) {
            delete(*task);
        } else {
            (*task)->done();
            (*task)->pushed = false;
        }
    }
    tasks.clear();

    if (!sharedMpool) mpoolClear(mpool);

    if (surface) {
        vport.x = vport.y = 0;
        vport.w = surface->w;
        vport.h = surface->h;
    }

    return true;
}


bool SwRenderer::sync()
{
    return true;
}


RenderRegion SwRenderer::viewport()
{
    return vport;
}


bool SwRenderer::viewport(const RenderRegion& vp)
{
    vport = vp;
    return true;
}


bool SwRenderer::target(uint32_t* buffer, uint32_t stride, uint32_t w, uint32_t h, uint32_t cs)
{
    if (!buffer || stride == 0 || w == 0 || h == 0 || w > stride) return false;

    if (!surface) surface = new SwSurface;

    surface->buffer = buffer;
    surface->stride = stride;
    surface->w = w;
    surface->h = h;
    surface->cs = cs;

    vport.x = vport.y = 0;
    vport.w = surface->w;
    vport.h = surface->h;

    return rasterCompositor(surface);
}


bool SwRenderer::preRender()
{
    return rasterClear(surface);
}

void SwRenderer::clearCompositors()
{
    //Free Composite Caches
    for (auto comp = compositors.data; comp < (compositors.data + compositors.count); ++comp) {
        free((*comp)->compositor->image.data);
        delete((*comp)->compositor);
        delete(*comp);
    }
    compositors.reset();
}


bool SwRenderer::postRender()
{
    //Unmultiply alpha if needed
    if (surface->cs == SwCanvas::ABGR8888_STRAIGHT || surface->cs == SwCanvas::ARGB8888_STRAIGHT) {
        rasterUnpremultiply(surface);
    }

    for (auto task = tasks.data; task < (tasks.data + tasks.count); ++task) {
        (*task)->pushed = false;
    }
    tasks.clear();

    clearCompositors();
    return true;
}


bool SwRenderer::renderImage(RenderData data)
{
    auto task = static_cast<SwImageTask*>(data);
    task->done();

    if (task->opacity == 0) return true;

    return rasterImage(surface, &task->image, task->transform, task->bbox, task->opacity);
}


bool SwRenderer::renderShape(RenderData data)
{
    auto task = static_cast<SwShapeTask*>(data);
    if (!task) return false;

    task->done();

    if (task->opacity == 0) return true;

    uint32_t opacity;
    Compositor* cmp = nullptr;

    //Do Stroking Composition
    if (task->cmpStroking) {
        opacity = 255;
        cmp = target(task->bounds());
        beginComposite(cmp, CompositeMethod::None, task->opacity);
    //No Stroking Composition
    } else {
        opacity = task->opacity;
    }

    //Main raster stage
    uint8_t r, g, b, a;

    if (auto fill = task->sdata->fill()) {
        rasterGradientShape(surface, &task->shape, fill->identifier());
    } else {
        task->sdata->fillColor(&r, &g, &b, &a);
        a = static_cast<uint8_t>((opacity * (uint32_t) a) / 255);
        if (a > 0) rasterShape(surface, &task->shape, r, g, b, a);
    }

    if (auto strokeFill = task->sdata->strokeFill()) {
        rasterGradientStroke(surface, &task->shape, strokeFill->identifier());
    } else {
        if (task->sdata->strokeColor(&r, &g, &b, &a) == Result::Success) {
            a = static_cast<uint8_t>((opacity * (uint32_t) a) / 255);
            if (a > 0) rasterStroke(surface, &task->shape, r, g, b, a);
        }
    }

    if (task->cmpStroking) endComposite(cmp);

    return true;
}


RenderRegion SwRenderer::region(RenderData data)
{
    return static_cast<SwTask*>(data)->bounds();
}


bool SwRenderer::beginComposite(Compositor* cmp, CompositeMethod method, uint32_t opacity)
{
    if (!cmp) return false;
    auto p = static_cast<SwCompositor*>(cmp);

    p->method = method;
    p->opacity = opacity;

    //Current Context?
    if (p->method != CompositeMethod::None) {
        surface = p->recoverSfc;
        surface->compositor = p;
    }

    return true;
}


bool SwRenderer::mempool(bool shared)
{
    if (shared == sharedMpool) return true;

    if (shared) {
        if (!sharedMpool) {
            if (!mpoolTerm(mpool)) return false;
            mpool = globalMpool;
        }
    } else {
        if (sharedMpool) mpool = mpoolInit(threadsCnt);
    }

    sharedMpool = shared;

    if (mpool) return true;
    return false;
}


Compositor* SwRenderer::target(const RenderRegion& region)
{
    auto x = region.x;
    auto y = region.y;
    auto w = region.w;
    auto h = region.h;
    auto sw = static_cast<int32_t>(surface->w);
    auto sh = static_cast<int32_t>(surface->h);

    //Out of boundary
    if (x > sw || y > sh) return nullptr;

    SwSurface* cmp = nullptr;

    //Use cached data
    for (auto p = compositors.data; p < (compositors.data + compositors.count); ++p) {
        if ((*p)->compositor->valid) {
            cmp = *p;
            break;
        }
    }

    //New Composition
    if (!cmp) {
        cmp = new SwSurface;
        if (!cmp) goto err;

        //Inherits attributes from main surface
        *cmp = *surface;

        cmp->compositor = new SwCompositor;
        if (!cmp->compositor) goto err;

        //SwImage, Optimize Me: Surface size from MainSurface(WxH) to Parameter W x H
        cmp->compositor->image.data = (uint32_t*) malloc(sizeof(uint32_t) * surface->stride * surface->h);
        if (!cmp->compositor->image.data) goto err;
        compositors.push(cmp);
    }

    //Boundary Check
    if (x + w > sw) w = (sw - x);
    if (y + h > sh) h = (sh - y);

    TVGLOG("SW_ENGINE", "Using intermediate composition [Region: %d %d %d %d]", x, y, w, h);

    cmp->compositor->recoverSfc = surface;
    cmp->compositor->recoverCmp = surface->compositor;
    cmp->compositor->valid = false;
    cmp->compositor->bbox.min.x = x;
    cmp->compositor->bbox.min.y = y;
    cmp->compositor->bbox.max.x = x + w;
    cmp->compositor->bbox.max.y = y + h;
    cmp->compositor->image.stride = surface->stride;
    cmp->compositor->image.w = surface->w;
    cmp->compositor->image.h = surface->h;
    cmp->compositor->image.direct = true;

    //We know partial clear region
    cmp->buffer = cmp->compositor->image.data + (cmp->stride * y + x);
    cmp->w = w;
    cmp->h = h;

    rasterClear(cmp);

    //Recover context
    cmp->buffer = cmp->compositor->image.data;
    cmp->w = cmp->compositor->image.w;
    cmp->h = cmp->compositor->image.h;

    //Switch render target
    surface = cmp;

    return cmp->compositor;

err:
    if (cmp) {
        if (cmp->compositor) delete(cmp->compositor);
        delete(cmp);
    }

    return nullptr;
}


bool SwRenderer::endComposite(Compositor* cmp)
{
    if (!cmp) return false;

    auto p = static_cast<SwCompositor*>(cmp);
    p->valid = true;

    //Recover Context
    surface = p->recoverSfc;
    surface->compositor = p->recoverCmp;

    //Default is alpha blending
    if (p->method == CompositeMethod::None) {
        return rasterImage(surface, &p->image, nullptr, p->bbox, p->opacity);
    }

    return true;
}


bool SwRenderer::dispose(RenderData data)
{
    auto task = static_cast<SwTask*>(data);
    if (!task) return true;
    task->done();
    task->dispose();

    if (task->pushed) task->disposed = true;
    else delete(task);

    return true;
}


void* SwRenderer::prepareCommon(SwTask* task, const RenderTransform* transform, uint32_t opacity, const Array<RenderData>& clips, RenderUpdateFlag flags)
{
    if (!surface) return task;
    if (flags == RenderUpdateFlag::None) return task;

    //Finish previous task if it has duplicated request.
    task->done();

    if (clips.count > 0) {
        //Guarantee composition targets get ready.
        for (auto clip = clips.data; clip < (clips.data + clips.count); ++clip) {
            static_cast<SwShapeTask*>(*clip)->done();
        }
        task->clips = clips;
    }

    if (transform) {
        if (!task->transform) task->transform = static_cast<Matrix*>(malloc(sizeof(Matrix)));
        *task->transform = transform->m;
    } else {
        if (task->transform) free(task->transform);
        task->transform = nullptr;
    }

    task->opacity = opacity;
    task->surface = surface;
    task->mpool = mpool;
    task->flags = flags;
    task->bbox.min.x = mathMax(static_cast<SwCoord>(0), static_cast<SwCoord>(vport.x));
    task->bbox.min.y = mathMax(static_cast<SwCoord>(0), static_cast<SwCoord>(vport.y));
    task->bbox.max.x = mathMin(static_cast<SwCoord>(surface->w), static_cast<SwCoord>(vport.x + vport.w));
    task->bbox.max.y = mathMin(static_cast<SwCoord>(surface->h), static_cast<SwCoord>(vport.y + vport.h));

    if (!task->pushed) {
        task->pushed = true;
        tasks.push(task);
    }

    TaskScheduler::request(task);

    return task;
}


RenderData SwRenderer::prepare(Surface* image, RenderData data, const RenderTransform* transform, uint32_t opacity, Array<RenderData>& clips, RenderUpdateFlag flags)
{
    //prepare task
    auto task = static_cast<SwImageTask*>(data);
    if (!task) {
        task = new SwImageTask;
        if (flags & RenderUpdateFlag::Image) {
            task->image.data = image->buffer;
            task->image.w = image->w;
            task->image.h = image->h;
            task->image.stride = image->stride;
        }
    }
    return prepareCommon(task, transform, opacity, clips, flags);
}


RenderData SwRenderer::prepare(const Shape& sdata, RenderData data, const RenderTransform* transform, uint32_t opacity, Array<RenderData>& clips, RenderUpdateFlag flags)
{
    //prepare task
    auto task = static_cast<SwShapeTask*>(data);
    if (!task) {
        task = new SwShapeTask;
        task->sdata = &sdata;
    }
    return prepareCommon(task, transform, opacity, clips, flags);
}


SwRenderer::SwRenderer():mpool(globalMpool)
{
}


bool SwRenderer::init(uint32_t threads)
{
    if ((initEngineCnt++) > 0) return true;

    threadsCnt = threads;

    //Share the memory pool among the renderer
    globalMpool = mpoolInit(threads);
    if (!globalMpool) {
        --initEngineCnt;
        return false;
    }

    return true;
}


int32_t SwRenderer::init()
{
    return initEngineCnt;
}


bool SwRenderer::term()
{
    if ((--initEngineCnt) > 0) return true;

    initEngineCnt = 0;

   _termEngine();

    return true;
}

SwRenderer* SwRenderer::gen()
{
    ++rendererCnt;
    return new SwRenderer();
}
