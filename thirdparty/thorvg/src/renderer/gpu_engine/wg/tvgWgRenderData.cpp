
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

#include <algorithm>
#include <cmath>
#include "tvgCommon.h"
#include "tvgWgTessellator.h"
#include "tvgWgRenderData.h"
#include "tvgWgShaderTypes.h"

//***********************************************************************
// WgImageData
//***********************************************************************

void WgImageData::update(WgContext& context, const RenderSurface* surface, FilterMethod filter)
{
    // get appropriate texture format from color space
    WGPUTextureFormat texFormat = WGPUTextureFormat_BGRA8Unorm;
    if (surface->cs == ColorSpace::ABGR8888S)
        texFormat = WGPUTextureFormat_RGBA8Unorm;
    if (surface->cs == ColorSpace::Grayscale8)
        texFormat = WGPUTextureFormat_R8Unorm;
    // allocate new texture handle
    bool texHandleChanged = context.allocateTexture(texture, surface->w, surface->h, texFormat, surface->data);
    // update texture view of texture handle was changed
    if (texHandleChanged) {
        context.releaseTextureView(textureView);
        textureView = context.createTextureView(texture);
        // update bind group
        context.layouts.releaseBindGroup(bindGroup);
        auto sampler = (filter == FilterMethod::Bilinear) ? context.samplerLinearClamp : context.samplerNearestClamp;
        bindGroup = context.layouts.createBindGroupTexSampled(sampler, textureView);
    }
};


void WgImageData::update(WgContext& context, const Fill* fill)
{
    // compute gradient data
    WgShaderTypeGradientData gradientData;
    gradientData.update(fill);
    // allocate new texture handle
    bool texHandleChanged = context.allocateTexture(texture, WG_TEXTURE_GRADIENT_SIZE, 1, WGPUTextureFormat_RGBA8Unorm, gradientData.data);
    // update texture view of texture handle was changed
    if (texHandleChanged) {
        context.releaseTextureView(textureView);
        textureView = context.createTextureView(texture);
        // get sampler by spread type
        WGPUSampler sampler = context.samplerLinearClamp;
        if (fill->spread() == FillSpread::Reflect) sampler = context.samplerLinearMirror;
        if (fill->spread() == FillSpread::Repeat) sampler = context.samplerLinearRepeat;
        // update bind group
        context.layouts.releaseBindGroup(bindGroup);
        bindGroup = context.layouts.createBindGroupTexSampled(sampler, textureView);
    }
};


void WgImageData::release(WgContext& context)
{
    context.layouts.releaseBindGroup(bindGroup);
    context.releaseTextureView(textureView);
    context.releaseTexture(texture);
};

//***********************************************************************
// WgRenderSettings
//***********************************************************************

void WgRenderSettings::bakeSolidColor()
{
    if (fillType != WgRenderSettingsType::Solid) return;
    settings.color = solidColor;
    settings.color.vec[3] *= opacity;
    settings.options.vec[3] = 1.0f;
}


void WgRenderSettings::update(TVG_UNUSED WgContext& context, tvg::ColorSpace cs, uint8_t opacity)
{
    //TODO: Update separately according to the RenderUpdateFlag
    settings.options.update(cs, opacity * opacityMultiplier);
    this->opacity = settings.options.vec[3];
    bakeSolidColor();
}

void WgRenderSettings::update(WgContext& context, const Fill* fill, const Matrix* modelTransform, bool updateColorRamp)
{
    assert(fill);
    settings.gradient.update(fill, modelTransform);
    if (updateColorRamp) gradientData.update(context, fill);
    // get gradient rasterisation settings
    rasterType = WgRenderRasterType::Gradient;
    if (fill->type() == Type::LinearGradient)
        fillType = WgRenderSettingsType::Linear;
    else if (fill->type() == Type::RadialGradient)
        fillType = WgRenderSettingsType::Radial;
    settings.options.vec[3] = opacity;
};


void WgRenderSettings::update(TVG_UNUSED WgContext& context, const RenderColor& c)
{
    solidColor.update(c);
    settings.color = solidColor;
    rasterType = WgRenderRasterType::Solid;
    fillType = WgRenderSettingsType::Solid;
    bakeSolidColor();
};


void WgRenderSettings::release(WgContext& context)
{
    gradientData.release(context);
};

//***********************************************************************
// WgRenderDataPaint
//***********************************************************************

void WgRenderDataPaint::release(WgContext& context)
{
    clips.clear();
};

void WgRenderDataPaint::updateClips(const Array<RenderData>& clips)
{
    this->clips.clear();
    // RenderData == WgRenderDataPaint*, just copy it.
    this->clips = *((Array<WgRenderDataPaint*>*)&clips);
}

//***********************************************************************
// WgRenderDataShape
//***********************************************************************

void WgRenderDataShape::updateBBox(BBox bb)
{
    bbox.min = tvg::min(bbox.min, bb.min);
    bbox.max = tvg::max(bbox.max, bb.max);
}


void WgRenderDataShape::updateVisibility(const RenderShape& rshape, uint8_t opacity)
{
    renderSettingsShape.skip = (rshape.color.a * opacity == 0) && (!rshape.fill);
    renderSettingsStroke.skip = rshape.stroke ? (rshape.stroke->color.a * opacity == 0) && (!rshape.stroke->fill) : true;
}


void WgRenderDataShape::updateMeshes(const RenderShape &rshape, RenderUpdateFlag flag, const Matrix& matrix)
{
    releaseMeshes();  //Optimize: bad idea to reset meshes always. it could re-use the meshes if there haven't been any path changes.

    convex = false;
    strokeFirst = rshape.strokeFirst();
    renderSettingsShape.opacityMultiplier = 1.0f;
    renderSettingsStroke.opacityMultiplier = 1.0f;

    // optimize path
    auto& optPath = RenderPath::scratch();
    bool optPathThin = false;
    bool optPathSkipFill = false;
    if (rshape.trimpath()) {
        auto& trimmed = RenderPath::scratch();
        if (rshape.stroke->trim.trim(rshape.path, trimmed)) gpuOptimize(trimmed, optPath, matrix, optPathThin, optPathSkipFill);
        else optPath.clear();
    } else {
        gpuOptimize(rshape.path, optPath, matrix, optPathThin, optPathSkipFill);
    }

    auto updatePath = flag & (RenderUpdateFlag::Transform | RenderUpdateFlag::Path);

    // update fill shapes
    if (updatePath || (flag & (RenderUpdateFlag::Color | RenderUpdateFlag::Gradient))) {
        if (optPathSkipFill) {
            // Too-thin fills are suppressed instead of going through thin fallback.
            meshShape.clear();
        } else {
            BBox bbox;
            // Drawable thin fills are tessellated as a minimal-width stroke.
            if (optPathThin && tvg::zero(rshape.strokeWidth())) {
                WgStroker stroker(&meshShape, MIN_WG_STROKE_WIDTH, StrokeCap::Butt, StrokeJoin::Bevel);
                stroker.run(optPath);
                bbox = stroker.getBBox();
                renderSettingsShape.opacityMultiplier = MIN_WG_STROKE_ALPHA;
            } else {
                WgBWTessellator bwTess{&meshShape};
                bwTess.tessellate(optPath);
                convex = bwTess.convex;
                bbox = bwTess.getBBox();
            }
            if (meshShape.ibuffer.empty()) {
                meshShape.clear();
            } else {
                meshShapeBBox.bbox(bbox.min, bbox.max);
                updateBBox(bbox);
            }
        }
    }
    // update strokes shapes
    if (rshape.stroke && (updatePath || (flag & (RenderUpdateFlag::Stroke | RenderUpdateFlag::GradientStroke)))) {
        auto strokeWidth = rshape.strokeWidth();
        auto strokeWidthWorld = strokeWidth * scaling(matrix);
        if (!std::isfinite(strokeWidthWorld)) strokeWidthWorld = strokeWidth;
        if (!std::isfinite(strokeWidthWorld)) strokeWidthWorld = 0.0f;

        //run stroking only if it's valid
        if (!tvg::zero(strokeWidthWorld)) {
            WgStroker stroker(&meshStrokes, strokeWidthWorld, rshape.strokeCap(), rshape.strokeJoin(), rshape.strokeMiterlimit());
            auto& dashed = RenderPath::scratch();
            if (gpuStrokeDash(rshape, dashed, &matrix)) stroker.run(dashed);
            else stroker.run(optPath);
            renderSettingsStroke.opacityMultiplier = 1.0f;
            if (meshStrokes.ibuffer.empty()) {
                meshStrokes.clear();
            } else {
                auto bbox = stroker.getBBox();
                meshStrokesBBox.bbox(bbox.min, bbox.max);
                updateBBox(bbox);
            }
        }
    }
    // update shapes bbox (with empty path handling)
    if (!meshShape.vbuffer.empty() || !meshStrokes.vbuffer.empty()) updateAABB();
    else bbox = aabb = {{0, 0}, {0, 0}};
    meshBBox.bbox(bbox.min, bbox.max);
}


void WgRenderDataShape::releaseMeshes()
{
    meshStrokes.clear();
    meshStrokesBBox.clear();
    meshShape.clear();
    meshShapeBBox.clear();
    meshBBox.clear();
    bbox.min = {FLT_MAX, FLT_MAX};
    bbox.max = {0.0f, 0.0f};
    aabb = {{0, 0}, {0, 0}};
    clips.clear();
}


void WgRenderDataShape::release(WgContext& context)
{
    releaseMeshes();
    renderSettingsStroke.release(context);
    renderSettingsShape.release(context);
    WgRenderDataPaint::release(context);
};

//***********************************************************************
// WgRenderDataShapePool
//***********************************************************************

WgRenderDataShape* WgRenderDataShapePool::allocate(WgContext& context)
{
    WgRenderDataShape* renderData{};
    if (mPool.count > 0) {
        renderData = mPool.last();
        mPool.pop();
    } else {
        renderData = new WgRenderDataShape();
        mList.push(renderData);
    }
    return renderData;
}


void WgRenderDataShapePool::free(WgContext& context, WgRenderDataShape* renderData)
{
    renderData->releaseMeshes();
    renderData->clips.clear();
    mPool.push(renderData);
}


void WgRenderDataShapePool::release(WgContext& context)
{
    ARRAY_FOREACH(p, mList) {
        (*p)->release(context);
        delete(*p);
    }
    mPool.clear();
    mList.clear();
}

//***********************************************************************
// WgRenderDataPicture
//***********************************************************************

void WgRenderDataPicture::updateSurface(WgContext& context, const RenderSurface* surface, const Matrix& transform, FilterMethod filter, bool updateTexture)
{
    meshData.imageBox(surface->w, surface->h, transform);
    if (updateTexture) imageData.update(context, surface, filter);
}


void WgRenderDataPicture::release(WgContext& context)
{
    renderSettings.release(context);
    imageData.release(context);
    WgRenderDataPaint::release(context);
}

//***********************************************************************
// WgRenderDataPicturePool
//***********************************************************************

WgRenderDataPicture* WgRenderDataPicturePool::allocate(WgContext& context)
{
    WgRenderDataPicture* renderData{};
    if (mPool.count > 0) {
        renderData = mPool.last();
        mPool.pop();
    } else {
        renderData = new WgRenderDataPicture();
        mList.push(renderData);
    }
    return renderData;
}


void WgRenderDataPicturePool::free(WgContext& context, WgRenderDataPicture* renderData)
{
    renderData->clips.clear();
    mPool.push(renderData);
}


void WgRenderDataPicturePool::release(WgContext& context)
{
    ARRAY_FOREACH(p, mList) {
        (*p)->release(context);
        delete(*p);
    }
    mPool.clear();
    mList.clear();
}

//***********************************************************************
// WgRenderDataEffectParams
//***********************************************************************

void WgRenderDataEffectParams::update(WgContext& context, WgShaderTypeEffectParams& effectParams)
{
    if (context.allocateBufferUniform(bufferParams, &effectParams.params, sizeof(effectParams.params))) {
        context.layouts.releaseBindGroup(bindGroupParams);
        bindGroupParams = context.layouts.createBindGroupBuffer1Un(bufferParams);
    }
}


void WgRenderDataEffectParams::update(WgContext& context, RenderEffectGaussianBlur* gaussian, const Matrix& transform)
{
    assert(gaussian);
    WgShaderTypeEffectParams effectParams;
    if (!effectParams.update(gaussian, transform)) return;
    update(context, effectParams);
    extend = effectParams.extend;
}


void WgRenderDataEffectParams::update(WgContext& context, RenderEffectDropShadow* dropShadow, const Matrix& transform)
{
    assert(dropShadow);
    WgShaderTypeEffectParams effectParams;
    if (!effectParams.update(dropShadow, transform)) return;
    update(context, effectParams);
    extend = effectParams.extend;
    offset = effectParams.offset;
}


void WgRenderDataEffectParams::update(WgContext& context, RenderEffectFill* fill)
{
    assert(fill);
    WgShaderTypeEffectParams effectParams;
    if (!effectParams.update(fill)) return;
    update(context, effectParams);
}


void WgRenderDataEffectParams::update(WgContext& context, RenderEffectTint* tint)
{
    assert(tint);
    WgShaderTypeEffectParams effectParams;
    if (!effectParams.update(tint)) return;
    update(context, effectParams);
}


void WgRenderDataEffectParams::update(WgContext& context, RenderEffectTritone* tritone)
{
    assert(tritone);
    WgShaderTypeEffectParams effectParams;
    if (!effectParams.update(tritone)) return;
    update(context, effectParams);
}


void WgRenderDataEffectParams::release(WgContext& context)
{
    context.releaseBuffer(bufferParams);
    context.layouts.releaseBindGroup(bindGroupParams);
}

//***********************************************************************
// WgRenderDataColorsPool
//***********************************************************************

WgRenderDataEffectParams* WgRenderDataEffectParamsPool::allocate(WgContext& context)
{
    WgRenderDataEffectParams* renderData{};
    if (mPool.count > 0) {
        renderData = mPool.last();
        mPool.pop();
    } else {
        renderData = new WgRenderDataEffectParams();
        mList.push(renderData);
    }
    return renderData;
}


void WgRenderDataEffectParamsPool::free(WgContext& context, WgRenderDataEffectParams* renderData)
{
    if (renderData) mPool.push(renderData);
}


void WgRenderDataEffectParamsPool::release(WgContext& context)
{
    ARRAY_FOREACH(p, mList) {
        (*p)->release(context);
        delete(*p);
    }
    mPool.clear();
    mList.clear();
}

//***********************************************************************
// WgStageBufferGeometry
//***********************************************************************

void WgStageBufferGeometry::append(WgMeshData* meshData)
{
    assert(meshData);
    uint32_t vsize = meshData->vbuffer.count * sizeof(meshData->vbuffer[0]);
    uint32_t tsize = meshData->tbuffer.count * sizeof(meshData->tbuffer[0]);
    uint32_t isize = meshData->ibuffer.count * sizeof(meshData->ibuffer[0]);
    // append vertex data
    if (vbuffer.reserved < vbuffer.count + vsize)
        vbuffer.grow(std::max(vsize, vbuffer.reserved));
    if (meshData->vbuffer.count > 0) {
        meshData->voffset = vbuffer.count;
        memcpy(vbuffer.data + vbuffer.count, meshData->vbuffer.data, vsize);
        vbuffer.count += vsize;
    }
    // append tex coords data
    if (vbuffer.reserved < vbuffer.count + tsize)
        vbuffer.grow(std::max(tsize, vbuffer.reserved));
    if (meshData->tbuffer.count > 0) {
        meshData->toffset = vbuffer.count;
        memcpy(vbuffer.data + vbuffer.count, meshData->tbuffer.data, tsize);
        vbuffer.count += tsize;
    }
    // append index data
    if (ibuffer.reserved < ibuffer.count + isize)
        ibuffer.grow(std::max(isize, ibuffer.reserved));
    if (meshData->ibuffer.count > 0) {
        meshData->ioffset = ibuffer.count;
        memcpy(ibuffer.data + ibuffer.count, meshData->ibuffer.data, isize);
        ibuffer.count += isize;
    }
}


void WgStageBufferGeometry::append(WgRenderDataShape* renderDataShape)
{
    append(&renderDataShape->meshShape);
    append(&renderDataShape->meshShapeBBox);
    append(&renderDataShape->meshStrokes);
    append(&renderDataShape->meshStrokesBBox);
    append(&renderDataShape->meshBBox);
}


void WgStageBufferGeometry::append(WgRenderDataPicture* renderDataPicture)
{
    append(&renderDataPicture->meshData);
}


void WgStageBufferGeometry::release(WgContext& context)
{
    context.releaseBuffer(vbuffer_gpu);
    context.releaseBuffer(ibuffer_gpu);
}


void WgStageBufferGeometry::clear()
{
    vbuffer.clear();
    ibuffer.clear();
}


void WgStageBufferGeometry::flush(WgContext& context) 
{
    context.allocateBufferVertex(vbuffer_gpu, (float *)vbuffer.data, vbuffer.count);
    context.allocateBufferIndex(ibuffer_gpu, (uint32_t *)ibuffer.data, ibuffer.count);
}

//***********************************************************************
// WgStageBufferSolidColor
//***********************************************************************

void WgStageBufferSolidColor::release(WgContext& context)
{
    context.releaseBuffer(vbuffer_gpu);
}


void WgStageBufferSolidColor::clear()
{
    vbuffer.clear();
}


void WgStageBufferSolidColor::flush(WgContext& context)
{
    if (vbuffer.count > 0)
        context.allocateBufferVertex(vbuffer_gpu, (float*)vbuffer.data, vbuffer.count * sizeof(WgShaderTypeVec4f));
}

//***********************************************************************
// WgIntersector
//***********************************************************************

bool WgIntersector::isPointInTriangle(const Point& p, const Point& a, const Point& b, const Point& c)
{
    auto d1 = tvg::cross(p - a, p - b);
    auto d2 = tvg::cross(p - b, p - c);
    auto d3 = tvg::cross(p - c, p - a);
    auto has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    auto has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(has_neg && has_pos);
}


// triangle list
bool WgIntersector::isPointInTris(const Point& p, const WgMeshData& mesh)
{
    for (uint32_t i = 0; i < mesh.ibuffer.count; i += 3) {
        auto p0 = mesh.vbuffer[mesh.ibuffer[i+0]];
        auto p1 = mesh.vbuffer[mesh.ibuffer[i+1]];
        auto p2 = mesh.vbuffer[mesh.ibuffer[i+2]];
        if (isPointInTriangle(p, p0, p1, p2)) return true;
    }
    return false;
}


// even-odd triangle list
bool WgIntersector::isPointInMesh(const Point& p, const WgMeshData& mesh)
{
    uint32_t crossings = 0;
    for (uint32_t i = 0; i < mesh.ibuffer.count; i += 3) {
        Point triangle[3] = {
            mesh.vbuffer[mesh.ibuffer[i+0]],
            mesh.vbuffer[mesh.ibuffer[i+1]],
            mesh.vbuffer[mesh.ibuffer[i+2]]
        };
        for (uint32_t j = 0; j < 3; j++) {
            auto p1 = triangle[j];
            auto p2 = triangle[(j + 1) % 3];
            if (p1.y == p2.y) continue;
            if (p1.y > p2.y) std::swap(p1, p2);
            if ((p.y > p1.y) && (p.y <= p2.y)) {
                auto intersectionX = (p2.x - p1.x) * (p.y - p1.y) / (p2.y - p1.y) + p1.x;
                if (intersectionX > p.x) crossings++;
            }
        }
    }
    return (crossings % 2) == 1;
}


bool WgIntersector::intersectClips(const Point& pt, const Array<WgRenderDataPaint*>& clips)
{
    for (uint32_t i = 0; i < clips.count; i++) {
        auto clip = (WgRenderDataShape*)clips[i];
        if (!isPointInMesh(pt, clip->meshShape)) return false;
    }
    return true;
}


bool WgIntersector::intersectShape(const RenderRegion region, const WgRenderDataShape* shape)
{
    if (!shape || ((shape->meshShape.ibuffer.count == 0) && (shape->meshStrokes.ibuffer.count == 0))) return false;
    auto sizeX = region.sw();
    auto sizeY = region.sh();
    for (int32_t y = 0; y <= sizeY; y++) {
        for (int32_t x = 0; x <= sizeX; x++) {
            Point pt{(float)x + region.min.x, (float)y + region.min.y};
            if (y % 2 == 1) pt.y = (float) sizeY - y - sizeY % 2 + region.min.y;
            if (intersectClips(pt, shape->clips)) {
                if (!shape->renderSettingsShape.skip && isPointInMesh(pt, shape->meshShape)) return true;
                if (!shape->renderSettingsStroke.skip && isPointInTris(pt, shape->meshStrokes)) return true;
            }
        }
    }
    return false;
}


bool WgIntersector::intersectImage(const RenderRegion region, const WgRenderDataPicture* image)
{
    if (!image) return false;
    auto sizeX = region.sw();
    auto sizeY = region.sh();
    for (int32_t y = 0; y <= sizeY; y++) {
        for (int32_t x = 0; x <= sizeX; x++) {
            Point pt{(float)x + region.min.x, (float)y + region.min.y};
            if (y % 2 == 1) pt.y = (float) sizeY - y - sizeY % 2 + region.min.y;
            if (intersectClips(pt, image->clips)) {
                if (isPointInTris(pt, image->meshData)) return true;
            }
        }
    }
    return false;
}
