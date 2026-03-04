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
    //TODO: acquire the current colorspace format & pre-multiplied alpha image.
    surface.buf8 = WebPDecodeBGRA(data, size, nullptr, nullptr);
    surface.stride = (uint32_t)w;
    surface.w = (uint32_t)w;
    surface.h = (uint32_t)h;
    surface.channelSize = sizeof(uint32_t);
    surface.cs = ColorSpace::ARGB8888;
    surface.premultiplied = false;

}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

WebpLoader::WebpLoader() : ImageLoader(FileType::Webp)
{
}


WebpLoader::~WebpLoader()
{
    done();

    if (freeData) tvg::free(data);
    data = nullptr;
    size = 0;
    freeData = false;
    WebPFree(surface.buf8);
}


bool WebpLoader::open(const char* path)
{
#ifdef THORVG_FILE_IO_SUPPORT
    if (!(data = (unsigned char*)LoadModule::open(path, size))) return false;

    int width, height;
    if (!WebPGetInfo(data, size, &width, &height)) return false;
    w = static_cast<float>(width);
    h = static_cast<float>(height);
    freeData = true;
    return true;
#else
    return false;
#endif
}


bool WebpLoader::open(const char* data, uint32_t size, TVG_UNUSED const char* rpath, bool copy)
{
    if (copy) {
        this->data = tvg::malloc<unsigned char>(size);
        if (!this->data) return false;
        memcpy((unsigned char *)this->data, data, size);
        freeData = true;
    } else {
        this->data = (unsigned char *) data;
        freeData = false;
    }

    int width, height;
    if (!WebPGetInfo(this->data, size, &width, &height)) return false;

    w = static_cast<float>(width);
    h = static_cast<float>(height);
    surface.cs = ColorSpace::ARGB8888;
    this->size = size;
    return true;
}


bool WebpLoader::read()
{
    if (!LoadModule::read()) return true;

    if (!data || w == 0 || h == 0) return false;

    TaskScheduler::request(this);

    return true;
}


RenderSurface* WebpLoader::bitmap()
{
    this->done();

    return ImageLoader::bitmap();
}
