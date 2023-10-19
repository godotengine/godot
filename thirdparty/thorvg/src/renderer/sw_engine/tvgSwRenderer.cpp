/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

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
    SwSurface* surface = nullptr;
    SwMpool* mpool = nullptr;
    SwBBox bbox = {{0, 0}, {0, 0}};       //Whole Rendering Region
    Matrix* transform = nullptr;
    Array<RenderData> clips;
    RenderUpdateFlag flags = RenderUpdateFlag::None;
    uint8_t opacity;
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
    virtual bool clip(SwRleData* target) = 0;
    virtual SwRleData* rle() = 0;

    virtual ~SwTask()
    {
        free(transform);
    }
};


struct SwShapeTask : SwTask
{
    SwShape shape;
    const RenderShape* rshape = nullptr;
    bool cmpStroking = false;
    bool clipper = false;

    /* We assume that if the stroke width is greater than 2,
       the shape's outline beneath the stroke could be adequately covered by the stroke drawing.
       Therefore, antialiasing is disabled under this condition.
       Additionally, the stroke style should not be dashed. */
    bool antialiasing(float strokeWidth)
    {
        return strokeWidth < 2.0f || rshape->stroke->dashCnt > 0 || rshape->stroke->strokeFirst;
    }

    float validStrokeWidth()
    {
        if (!rshape->stroke) return 0.0f;

        auto width = rshape->stroke->width;
        if (mathZero(width)) return 0.0f;

        if (!rshape->stroke->fill && (MULTIPLY(rshape->stroke->color[3], opacity) == 0)) return 0.0f;
        if (mathZero(rshape->stroke->trim.begin - rshape->stroke->trim.end)) return 0.0f;

        if (transform) return (width * sqrt(transform->e11 * transform->e11 + transform->e12 * transform->e12));
        else return width;
    }


    bool clip(SwRleData* target) override
    {
        if (shape.fastTrack) rleClipRect(target, &bbox);
        else if (shape.rle) rleClipPath(target, shape.rle);
        else return false;

        return true;
    }

    SwRleData* rle() override
    {
        if (!shape.rle && shape.fastTrack) {
            shape.rle = rleRender(&shape.bbox);
        }
        return shape.rle;
    }

    void run(unsigned tid) override
    {
        if (opacity == 0 && !clipper) return;  //Invisible

        auto strokeWidth = validStrokeWidth();
        bool visibleFill = false;
        auto clipRegion = bbox;

        //This checks also for the case, if the invisible shape turned to visible by alpha.
        auto prepareShape = false;
        if (!shapePrepared(&shape) && (flags & RenderUpdateFlag::Color)) prepareShape = true;

        //Shape
        if (flags & (RenderUpdateFlag::Path | RenderUpdateFlag::Transform) || prepareShape) {
            uint8_t alpha = 0;
            rshape->fillColor(nullptr, nullptr, nullptr, &alpha);
            alpha = MULTIPLY(alpha, opacity);
            visibleFill = (alpha > 0 || rshape->fill);
            if (visibleFill || clipper) {
                shapeReset(&shape);
                if (!shapePrepare(&shape, rshape, transform, clipRegion, bbox, mpool, tid, clips.count > 0 ? true : false)) goto err;
            }
        }
        //Fill
        if (flags & (RenderUpdateFlag::Gradient | RenderUpdateFlag::Transform | RenderUpdateFlag::Color)) {
            if (visibleFill || clipper) {
                if (!shapeGenRle(&shape, rshape, antialiasing(strokeWidth))) goto err;
            }
            if (auto fill = rshape->fill) {
                auto ctable = (flags & RenderUpdateFlag::Gradient) ? true : false;
                if (ctable) shapeResetFill(&shape);
                if (!shapeGenFillColors(&shape, fill, transform, surface, opacity, ctable)) goto err;
            } else {
                shapeDelFill(&shape);
            }
        }
        //Stroke
        if (flags & (RenderUpdateFlag::Stroke | RenderUpdateFlag::Transform)) {
            if (strokeWidth > 0.0f) {
                shapeResetStroke(&shape, rshape, transform);
                if (!shapeGenStrokeRle(&shape, rshape, transform, clipRegion, bbox, mpool, tid)) goto err;

                if (auto fill = rshape->strokeFill()) {
                    auto ctable = (flags & RenderUpdateFlag::GradientStroke) ? true : false;
                    if (ctable) shapeResetStrokeFill(&shape);
                    if (!shapeGenStrokeFillColors(&shape, fill, transform, surface, opacity, ctable)) goto err;
                } else {
                    shapeDelStrokeFill(&shape);
                }
            } else {
                shapeDelStroke(&shape);
            }
        }

        //Clear current task memorypool here if the clippers would use the same memory pool
        shapeDelOutline(&shape, mpool, tid);

        //Clip Path
        for (auto clip = clips.data; clip < clips.end(); ++clip) {
            auto clipper = static_cast<SwTask*>(*clip);
            //Clip shape rle
            if (shape.rle && !clipper->clip(shape.rle)) goto err;
            //Clip stroke rle
            if (shape.strokeRle && !clipper->clip(shape.strokeRle)) goto err;
        }
        return;

    err:
        shapeReset(&shape);
        shapeDelOutline(&shape, mpool, tid);
    }

    bool dispose() override
    {
       shapeFree(&shape);
       return true;
    }
};


struct SwSceneTask : SwTask
{
    Array<RenderData> scene;    //list of paints render data (SwTask)
    SwRleData* sceneRle = nullptr;

    bool clip(SwRleData* target) override
    {
        //Only one shape
        if (scene.count == 1) {
            return static_cast<SwTask*>(*scene.data)->clip(target);
        }

        //More than one shapes
        if (sceneRle) rleClipPath(target, sceneRle);
        else TVGLOG("SW_ENGINE", "No clippers in a scene?");

        return true;
    }

    SwRleData* rle() override
    {
        return sceneRle;
    }

    void run(unsigned tid) override
    {
        //TODO: Skip the run if the scene hans't changed.
        if (!sceneRle) sceneRle = static_cast<SwRleData*>(calloc(1, sizeof(SwRleData)));
        else rleReset(sceneRle);

        //Merge shapes if it has more than one shapes
        if (scene.count > 1) {
            //Merge first two clippers
            auto clipper1 = static_cast<SwTask*>(*scene.data);
            auto clipper2 = static_cast<SwTask*>(*(scene.data + 1));

            rleMerge(sceneRle, clipper1->rle(), clipper2->rle());

            //Unify the remained clippers
            for (auto rd = scene.data + 2; rd < scene.end(); ++rd) {
                auto clipper = static_cast<SwTask*>(*rd);
                rleMerge(sceneRle, sceneRle, clipper->rle());
            }
        }
    }

    bool dispose() override
    {
        rleFree(sceneRle);
        return true;
    }
};


struct SwImageTask : SwTask
{
    SwImage image;
    Surface* source;                            //Image source
    const RenderMesh* mesh = nullptr;           //Should be valid ptr in action

    bool clip(SwRleData* target) override
    {
        TVGERR("SW_ENGINE", "Image is used as ClipPath?");
        return true;
    }

    SwRleData* rle() override
    {
        TVGERR("SW_ENGINE", "Image is used as Scene ClipPath?");
        return nullptr;
    }

    void run(unsigned tid) override
    {
        auto clipRegion = bbox;

        //Convert colorspace if it's not aligned.
        if (source->owner) {
            if (source->cs != surface->cs) rasterConvertCS(source, surface->cs);
            if (!source->premultiplied) rasterPremultiply(source);
        }

        image.data = source->data;
        image.w = source->w;
        image.h = source->h;
        image.stride = source->stride;
        image.channelSize = source->channelSize;

        //Invisible shape turned to visible by alpha.
        if ((flags & (RenderUpdateFlag::Image | RenderUpdateFlag::Transform | RenderUpdateFlag::Color)) && (opacity > 0)) {
            imageReset(&image);
            if (!image.data || image.w == 0 || image.h == 0) goto end;

            if (!imagePrepare(&image, mesh, transform, clipRegion, bbox, mpool, tid)) goto end;

            // TODO: How do we clip the triangle mesh? Only clip non-meshed images for now
            if (mesh->triangleCnt == 0 && clips.count > 0) {
                if (!imageGenRle(&image, bbox, false)) goto end;
                if (image.rle) {
                    //Clear current task memorypool here if the clippers would use the same memory pool
                    imageDelOutline(&image, mpool, tid);
                    for (auto clip = clips.data; clip < clips.end(); ++clip) {
                        auto clipper = static_cast<SwTask*>(*clip);
                        if (!clipper->clip(image.rle)) goto err;
                    }
                    return;
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


static void _renderFill(SwShapeTask* task, SwSurface* surface, uint8_t opacity)
{
    uint8_t r, g, b, a;
    if (auto fill = task->rshape->fill) {
        rasterGradientShape(surface, &task->shape, fill->identifier());
    } else {
        task->rshape->fillColor(&r, &g, &b, &a);
        a = MULTIPLY(opacity, a);
        if (a > 0) rasterShape(surface, &task->shape, r, g, b, a);
    }
}

static void _renderStroke(SwShapeTask* task, SwSurface* surface, uint8_t opacity)
{
    uint8_t r, g, b, a;
    if (auto strokeFill = task->rshape->strokeFill()) {
        rasterGradientStroke(surface, &task->shape, strokeFill->identifier());
    } else {
        if (task->rshape->strokeColor(&r, &g, &b, &a)) {
            a = MULTIPLY(opacity, a);
            if (a > 0) rasterStroke(surface, &task->shape, r, g, b, a);
        }
    }
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

SwRenderer::~SwRenderer()
{
    clearCompositors();

    delete(surface);

    if (!sharedMpool) mpoolTerm(mpool);

    --rendererCnt;

    if (rendererCnt == 0 && initEngineCnt == 0) _termEngine();
}


bool SwRenderer::clear()
{
    for (auto task = tasks.data; task < tasks.end(); ++task) {
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


bool SwRenderer::target(pixel_t* data, uint32_t stride, uint32_t w, uint32_t h, ColorSpace cs)
{
    if (!data || stride == 0 || w == 0 || h == 0 || w > stride) return false;

    if (!surface) surface = new SwSurface;

    surface->data = data;
    surface->stride = stride;
    surface->w = w;
    surface->h = h;
    surface->cs = cs;
    surface->channelSize = CHANNEL_SIZE(cs);
    surface->premultiplied = true;
    surface->owner = true;

    vport.x = vport.y = 0;
    vport.w = surface->w;
    vport.h = surface->h;

    return rasterCompositor(surface);
}


bool SwRenderer::preRender()
{
    return rasterClear(surface, 0, 0, surface->w, surface->h);
}


void SwRenderer::clearCompositors()
{
    //Free Composite Caches
    for (auto comp = compositors.data; comp < compositors.end(); ++comp) {
        free((*comp)->compositor->image.data);
        delete((*comp)->compositor);
        delete(*comp);
    }
    compositors.reset();
}


bool SwRenderer::postRender()
{
    //Unmultiply alpha if needed
    if (surface->cs == ColorSpace::ABGR8888S || surface->cs == ColorSpace::ARGB8888S) {
        rasterUnpremultiply(surface);
    }

    for (auto task = tasks.data; task < tasks.end(); ++task) {
        if ((*task)->disposed) delete(*task);
        else (*task)->pushed = false;
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

    return rasterImage(surface, &task->image, task->mesh, task->transform, task->bbox, task->opacity);
}


bool SwRenderer::renderShape(RenderData data)
{
    auto task = static_cast<SwShapeTask*>(data);
    if (!task) return false;

    task->done();

    if (task->opacity == 0) return true;

    //Main raster stage
    if (task->rshape->stroke && task->rshape->stroke->strokeFirst) {
        _renderStroke(task, surface, task->opacity);
        _renderFill(task, surface, task->opacity);
    } else {
        _renderFill(task, surface, task->opacity);
        _renderStroke(task, surface, task->opacity);
    }

    return true;
}


bool SwRenderer::blend(BlendMethod method)
{
    if (surface->blendMethod == method) return true;
    surface->blendMethod = method;

    switch (method) {
        case BlendMethod::Add:
            surface->blender = opBlendAdd;
            break;
        case BlendMethod::Screen:
            surface->blender = opBlendScreen;
            break;
        case BlendMethod::Multiply:
            surface->blender = opBlendMultiply;
            break;
        case BlendMethod::Overlay:
            surface->blender = opBlendOverlay;
            break;
        case BlendMethod::Difference:
            surface->blender = opBlendDifference;
            break;
        case BlendMethod::Exclusion:
            surface->blender = opBlendExclusion;
            break;
        case BlendMethod::SrcOver:
            surface->blender = opBlendSrcOver;
            break;
        case BlendMethod::Darken:
            surface->blender = opBlendDarken;
            break;
        case BlendMethod::Lighten:
            surface->blender = opBlendLighten;
            break;
        case BlendMethod::ColorDodge:
            surface->blender = opBlendColorDodge;
            break;
        case BlendMethod::ColorBurn:
            surface->blender = opBlendColorBurn;
            break;
        case BlendMethod::HardLight:
            surface->blender = opBlendHardLight;
            break;
        case BlendMethod::SoftLight:
            surface->blender = opBlendSoftLight;
            break;
        default:
            surface->blender = nullptr;
            break;
    }
    return false;
}


RenderRegion SwRenderer::region(RenderData data)
{
    return static_cast<SwTask*>(data)->bounds();
}


bool SwRenderer::beginComposite(Compositor* cmp, CompositeMethod method, uint8_t opacity)
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


Compositor* SwRenderer::target(const RenderRegion& region, ColorSpace cs)
{
    auto x = region.x;
    auto y = region.y;
    auto w = region.w;
    auto h = region.h;
    auto sw = static_cast<int32_t>(surface->w);
    auto sh = static_cast<int32_t>(surface->h);

    //Out of boundary
    if (x >= sw || y >= sh || x + w < 0 || y + h < 0) return nullptr;

    SwSurface* cmp = nullptr;

    auto reqChannelSize = CHANNEL_SIZE(cs);

    //Use cached data
    for (auto p = compositors.data; p < compositors.end(); ++p) {
        if ((*p)->compositor->valid && (*p)->compositor->image.channelSize == reqChannelSize) {
            cmp = *p;
            break;
        }
    }

    //New Composition
    if (!cmp) {
        cmp = new SwSurface;

        //Inherits attributes from main surface
        *cmp = *surface;

        cmp->compositor = new SwCompositor;

        //TODO: We can optimize compositor surface size from (surface->stride x surface->h) to Parameter(w x h)
        cmp->compositor->image.data = (pixel_t*)malloc(reqChannelSize * surface->stride * surface->h);
        cmp->channelSize = cmp->compositor->image.channelSize = reqChannelSize;

        compositors.push(cmp);
    }

    //Boundary Check
    if (x + w > sw) w = (sw - x);
    if (y + h > sh) h = (sh - y);

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

    cmp->data = cmp->compositor->image.data;
    cmp->w = cmp->compositor->image.w;
    cmp->h = cmp->compositor->image.h;

    rasterClear(cmp, x, y, w, h);

    //Switch render target
    surface = cmp;

    return cmp->compositor;
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
        return rasterImage(surface, &p->image, nullptr, nullptr, p->bbox, p->opacity);
    }

    return true;
}


ColorSpace SwRenderer::colorSpace()
{
    if (surface) return surface->cs;
    else return ColorSpace::Unsupported;
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


void* SwRenderer::prepareCommon(SwTask* task, const RenderTransform* transform, const Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags)
{
    if (!surface) return task;
    if (flags == RenderUpdateFlag::None) return task;

    //Finish previous task if it has duplicated request.
    task->done();

    //TODO: Failed threading them. It would be better if it's possible.
    //See: https://github.com/thorvg/thorvg/issues/1409
    //Guarantee composition targets get ready.
    for (auto clip = clips.data; clip < clips.end(); ++clip) {
        static_cast<SwTask*>(*clip)->done();
    }

    task->clips = clips;

    if (transform) {
        if (!task->transform) task->transform = static_cast<Matrix*>(malloc(sizeof(Matrix)));
        *task->transform = transform->m;
    } else {
        if (task->transform) free(task->transform);
        task->transform = nullptr;
    }

    //zero size?
    if (task->transform) {
        if (task->transform->e11 == 0.0f && task->transform->e12 == 0.0f) return task; //zero width
        if (task->transform->e21 == 0.0f && task->transform->e22 == 0.0f) return task; //zero height
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


RenderData SwRenderer::prepare(Surface* surface, const RenderMesh* mesh, RenderData data, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags)
{
    //prepare task
    auto task = static_cast<SwImageTask*>(data);
    if (!task) task = new SwImageTask;
    task->source = surface;
    task->mesh = mesh;
    return prepareCommon(task, transform, clips, opacity, flags);
}


RenderData SwRenderer::prepare(const Array<RenderData>& scene, RenderData data, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags)
{
    //prepare task
    auto task = static_cast<SwSceneTask*>(data);
    if (!task) task = new SwSceneTask;
    task->scene = scene;

    //TODO: Failed threading them. It would be better if it's possible.
    //See: https://github.com/thorvg/thorvg/issues/1409
    //Guarantee composition targets get ready.
    for (auto task = scene.data; task < scene.end(); ++task) {
        static_cast<SwTask*>(*task)->done();
    }
    return prepareCommon(task, transform, clips, opacity, flags);
}


RenderData SwRenderer::prepare(const RenderShape& rshape, RenderData data, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flags, bool clipper)
{
    //prepare task
    auto task = static_cast<SwShapeTask*>(data);
    if (!task) {
        task = new SwShapeTask;
        task->rshape = &rshape;
    }
    task->clipper = clipper;

    return prepareCommon(task, transform, clips, opacity, flags);
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
