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

#include <memory.h>
#include <webp/decode.h>

#include "tvgWebpLoader.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

void WebpLoader::run(unsigned tid)
{
    WebPDecoderConfig config;
    if (!WebPInitDecoderConfig(&config)) return;

    ColorSpace cs;

    // request premultiplied image data
    if (BitmapLoader::cs == ColorSpace::ARGB8888 || BitmapLoader::cs == ColorSpace::ARGB8888S) {
        config.output.colorspace = MODE_bgrA;
        if (WebPDecode(data, size, &config) != VP8_STATUS_OK) return;
        cs = ColorSpace::ARGB8888;
    } else {
        config.output.colorspace = MODE_rgbA;
        if (WebPDecode(data, size, &config) != VP8_STATUS_OK) return;
        cs = ColorSpace::ABGR8888;
    }
    surface.setup((pixel_t*)config.output.u.RGBA.rgba, (uint32_t)w, (uint32_t)w, (uint32_t)h, sizeof(uint32_t), cs);
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

WebpLoader::~WebpLoader()
{
    done();

    if (owner != Ownership::Borrow) tvg::free(data);
    data = nullptr;
    size = 0;
    owner = Ownership::Borrow;
    WebPFree(surface.buf8);
}

bool WebpLoader::open(const char* path, const LoaderOps& ops)
{
#ifdef THORVG_FILE_IO_SUPPORT
    if (!(data = (unsigned char*)Loader::open(path, size))) return false;
    owner = Ownership::Transfer;

    WebPBitstreamFeatures features;
    if (WebPGetFeatures(data, size, &features)) return false;
    w = static_cast<float>(features.width);
    h = static_cast<float>(features.height);
    surface.alphaIgnored = !features.has_alpha;
    return true;
#else
    return false;
#endif
}

bool WebpLoader::open(const char* data, uint32_t size, const LoaderOps& ops)
{
    if (ops.owner == Ownership::Copy) {
        this->data = tvg::malloc<unsigned char>(size);
        memcpy((unsigned char *)this->data, data, size);
    } else {
        this->data = (unsigned char *) data;
    }
    owner = ops.owner;

    WebPBitstreamFeatures features;
    if (WebPGetFeatures(this->data, size, &features)) return false;
    w = static_cast<float>(features.width);
    h = static_cast<float>(features.height);
    surface.alphaIgnored = !features.has_alpha;
    this->size = size;
    return true;
}


bool WebpLoader::read()
{
    if (!Loader::read()) return true;

    if (!data || w == 0 || h == 0) return false;

    TaskScheduler::request(this);

    return true;
}


RenderSurface* WebpLoader::bitmap()
{
    this->done();

    return BitmapLoader::bitmap();
}
