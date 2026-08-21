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

#include "tvgWgTextureMgr.h"

static tvg::Inlist<WgTextureEntry>& _entries(WgTextureMgr::SurfaceEntry& surfaceEntry, FilterMethod filter)
{
    return (filter == FilterMethod::Bilinear) ? surfaceEntry.bilinear : surfaceEntry.nearest;
}

static WgTextureEntry* _findEntry(tvg::Inlist<WgTextureEntry>& entries, WGPUTexture texture)
{
    INLIST_FOREACH(entries, entry)
    {
        if (entry->texture == texture) return entry;
    }
    return nullptr;
}

static bool _matches(const WgTextureEntry& entry, const RenderSurface* surface, WGPUTextureFormat format)
{
    return entry.texture && (wgpuTextureGetWidth(entry.texture) == surface->w) && (wgpuTextureGetHeight(entry.texture) == surface->h) && (wgpuTextureGetFormat(entry.texture) == format);
}

WgTextureMgr::SurfaceEntry* WgTextureMgr::find(const RenderSurface* surface)
{
    INLIST_FOREACH(surfaces, entry)
    {
        if (entry->surface == surface) return entry;
    }
    return nullptr;
}

WGPUTextureFormat WgTextureMgr::textureFormat(const RenderSurface* surface)
{
    if (surface->cs == ColorSpace::ABGR8888 || surface->cs == ColorSpace::ABGR8888S) return WGPUTextureFormat_RGBA8Unorm;
    if (surface->cs == ColorSpace::ARGB8888 || surface->cs == ColorSpace::ARGB8888S) return WGPUTextureFormat_BGRA8Unorm;
    return WGPUTextureFormat_R8Unorm;  // must be
}

void WgTextureMgr::upload(WgContext& context, WgTextureEntry& entry, const RenderSurface* surface, FilterMethod filter)
{
    auto bytesPerRow = surface->stride * CHANNEL_SIZE(surface->cs);
    auto dataSize = static_cast<uint64_t>(bytesPerRow) * surface->h;
    if (!context.allocateTexture(entry.texture, surface->w, surface->h, textureFormat(surface), surface->data, bytesPerRow, dataSize)) return;

    context.releaseTextureView(entry.textureView);
    entry.textureView = context.createTextureView(entry.texture);

    context.layouts.releaseBindGroup(entry.bindGroup);
    auto sampler = (filter == FilterMethod::Bilinear) ? context.samplerLinearClamp : context.samplerNearestClamp;
    entry.bindGroup = context.layouts.createBindGroupTexSampled(sampler, entry.textureView);
}

void WgTextureMgr::releaseEntry(WgContext& context, WgTextureEntry& entry)
{
    context.layouts.releaseBindGroup(entry.bindGroup);
    context.releaseTextureView(entry.textureView);
    context.releaseTexture(entry.texture);
    entry.refCnt = 0;
}

const WgTextureEntry* WgTextureMgr::retain(WgContext& context, const RenderSurface* surface, FilterMethod filter, bool refreshTexture)
{
    auto* surfaceEntry = find(surface);
    if (!surfaceEntry) {
        surfaceEntry = new SurfaceEntry;
        surfaceEntry->surface = surface;
        surfaces.back(surfaceEntry);
    }

    auto& entries = _entries(*surfaceEntry, filter);
    auto* entry = entries.tail;
    auto format = textureFormat(surface);
    if (entry && refreshTexture && !_matches(*entry, surface, format) && entry->refCnt > 0) {
        entry = new WgTextureEntry;
        entries.back(entry);
    } else if (!entry) {
        entry = new WgTextureEntry;
        entries.back(entry);
    }
    if (!entry->texture || refreshTexture) upload(context, *entry, surface, filter);

    ++entry->refCnt;
    return entry;
}

void WgTextureMgr::release(WgContext& context, const RenderSurface* surface, FilterMethod filter, WGPUTexture texture)
{
    auto* surfaceEntry = find(surface);
    if (!surfaceEntry) return;

    auto& entries = _entries(*surfaceEntry, filter);
    auto* entry = _findEntry(entries, texture);
    if (!entry) return;

    if (entry->refCnt > 0) --entry->refCnt;
    if (entry->refCnt > 0) return;

    releaseEntry(context, *entry);
    entries.remove(entry);
    delete (entry);
    if (surfaceEntry->bilinear.empty() && surfaceEntry->nearest.empty()) {
        surfaces.remove(surfaceEntry);
        delete (surfaceEntry);
    }
}

void WgTextureMgr::clear(WgContext& context)
{
    while (auto* surfaceEntry = surfaces.front()) {
        while (auto* entry = surfaceEntry->bilinear.front()) {
            releaseEntry(context, *entry);
            delete (entry);
        }
        while (auto* entry = surfaceEntry->nearest.front()) {
            releaseEntry(context, *entry);
            delete (entry);
        }
        delete (surfaceEntry);
    }
    if (++stamp == 0) stamp = 1;
}
