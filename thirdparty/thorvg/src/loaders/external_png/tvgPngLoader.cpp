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

#include "tvgLoader.h"
#include "tvgPngLoader.h"

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


PngLoader::PngLoader()
{
    image = static_cast<png_imagep>(calloc(1, sizeof(png_image)));
    image->version = PNG_IMAGE_VERSION;
    image->opaque = NULL;
}

PngLoader::~PngLoader()
{
    if (content) {
        free((void*)content);
        content = nullptr;
    }
    free(image);
}

bool PngLoader::open(const string& path)
{
    image->opaque = NULL;

    if (!png_image_begin_read_from_file(image, path.c_str())) return false;

    w = (float)image->width;
    h = (float)image->height;

    return true;
}

bool PngLoader::open(const char* data, uint32_t size, bool copy)
{
    image->opaque = NULL;

    if (!png_image_begin_read_from_memory(image, data, size)) return false;

    w = (float)image->width;
    h = (float)image->height;

    return true;
}


bool PngLoader::read()
{
    png_bytep buffer;
    image->format = PNG_FORMAT_BGRA;
    buffer = static_cast<png_bytep>(malloc(PNG_IMAGE_SIZE((*image))));
    if (!buffer) {
        //out of memory, only time when libpng doesnt free its data
        png_image_free(image);
        return false;
    }
    if (!png_image_finish_read(image, NULL, buffer, 0, NULL)) {
        free(buffer);
        return false;
    }
    content = reinterpret_cast<uint32_t*>(buffer);

    _premultiply(reinterpret_cast<uint32_t*>(buffer), image->width, image->height);

    return true;
}

bool PngLoader::close()
{
    png_image_free(image);
    return true;
}

unique_ptr<Surface> PngLoader::bitmap()
{
    if (!content) return nullptr;

    auto surface = static_cast<Surface*>(malloc(sizeof(Surface)));
    surface->buffer = (uint32_t*)(content);
    surface->stride = w;
    surface->w = w;
    surface->h = h;
    surface->cs = SwCanvas::ARGB8888;

    return unique_ptr<Surface>(surface);
}
