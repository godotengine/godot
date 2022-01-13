/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All rights reserved.

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
#include "tvgLoader.h"
#include "tvgPngLoader.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/


static inline uint32_t PREMULTIPLY(uint32_t c)
{
    auto a = (c >> 24);
    return (c & 0xff000000) + ((((c >> 8) & 0xff) * a) & 0xff00) + ((((c & 0x00ff00ff) * a) >> 8) & 0x00ff00ff);
}


static void _premultiply(uint32_t* data, uint32_t w, uint32_t h)
{
    auto buffer = data;
    for (uint32_t y = 0; y < h; ++y, buffer += w) {
        auto src = buffer;
        for (uint32_t x = 0; x < w; ++x, ++src) {
            *src = PREMULTIPLY(*src);
        }
    }
}


void PngLoader::clear()
{
    lodepng_state_cleanup(&state);

    if (freeData) free(data);
    data = nullptr;
    size = 0;
    freeData = false;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

PngLoader::PngLoader()
{
    lodepng_state_init(&state);
}


PngLoader::~PngLoader()
{
    if (freeData) free(data);
}


bool PngLoader::open(const string& path)
{
    clear();

    auto pngFile = fopen(path.c_str(), "rb");
    if (!pngFile) return false;

    auto ret = false;

    //determine size
    if (fseek(pngFile, 0, SEEK_END) < 0) goto finalize;
    if (((size = ftell(pngFile)) < 1)) goto finalize;
    if (fseek(pngFile, 0, SEEK_SET)) goto finalize;

    data = (unsigned char *) malloc(size);
    if (!data) goto finalize;

    freeData = true;

    if (fread(data, size, 1, pngFile) < 1) goto failure;

    lodepng_state_init(&state);

    unsigned int width, height;
    if (lodepng_inspect(&width, &height, &state, data, size) > 0) goto failure;

    w = static_cast<float>(width);
    h = static_cast<float>(height);
    ret = true;

    goto finalize;

failure:
    clear();

finalize:
    fclose(pngFile);
    return ret;
}


bool PngLoader::open(const char* data, uint32_t size, bool copy)
{
    clear();

    lodepng_state_init(&state);
    
    unsigned int width, height;
    if (lodepng_inspect(&width, &height, &state, (unsigned char*)(data), size) > 0) return false;

    if (copy) {
        this->data = (unsigned char *) malloc(size);
        if (!this->data) return false;
        memcpy((unsigned char *)this->data, data, size);
        freeData = true;
    } else {
        this->data = (unsigned char *) data;
        freeData = false;
    }

    w = static_cast<float>(width);
    h = static_cast<float>(height);
    this->size = size;

    return true;
}


bool PngLoader::read()
{
    if (!data || w <= 0 || h <= 0) return false;

    TaskScheduler::request(this);

    return true;
}


bool PngLoader::close()
{
    this->done();
    clear();
    return true;
}


unique_ptr<Surface> PngLoader::bitmap()
{
    this->done();

    if (!image) return nullptr;

    auto surface = static_cast<Surface*>(malloc(sizeof(Surface)));
    surface->buffer = (uint32_t*)(image);
    surface->stride = w;
    surface->w = w;
    surface->h = h;
    surface->cs = SwCanvas::ARGB8888;

    return unique_ptr<Surface>(surface);
}


void PngLoader::run(unsigned tid)
{
    auto width = static_cast<unsigned>(w);
    auto height = static_cast<unsigned>(h);

    lodepng_decode(&image, &width, &height, &state, data, size);

    _premultiply((uint32_t*)(image), width, height);
}