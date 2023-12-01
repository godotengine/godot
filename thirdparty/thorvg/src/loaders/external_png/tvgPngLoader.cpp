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

#include "tvgLoader.h"
#include "tvgPngLoader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

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
    cs = ColorSpace::ARGB8888;

    return true;
}

bool PngLoader::open(const char* data, uint32_t size, bool copy)
{
    image->opaque = NULL;

    if (!png_image_begin_read_from_memory(image, data, size)) return false;

    w = (float)image->width;
    h = (float)image->height;
    cs = ColorSpace::ARGB8888;

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

    //TODO: It's better to keep this surface instance in the loader side
    auto surface = new Surface;
    surface->buf32 = content;
    surface->stride = w;
    surface->w = w;
    surface->h = h;
    surface->cs = cs;
    surface->channelSize = sizeof(uint32_t);
    surface->owner = true;
    surface->premultiplied = false;

    return unique_ptr<Surface>(surface);
}

