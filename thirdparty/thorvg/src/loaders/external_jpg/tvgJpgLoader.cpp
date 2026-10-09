/*
 * Copyright (c) 2021 - 2026 ThorVG project. All rights reserved.

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

#include <climits>
#include <turbojpeg.h>
#include "tvgJpgLoader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

void JpgLoader::clear()
{
    if (owner != Ownership::Borrow) tvg::free(data);
    data = nullptr;
    size = 0;
    owner = Ownership::Borrow;
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

JpgLoader::JpgLoader() : BitmapLoader(FileType::Jpg), jpegDecompressor(tjInitDecompress())
{
}

JpgLoader::~JpgLoader()
{
    clear();
    tjDestroy(jpegDecompressor);

    //This image is shared with raster engine.
    tjFree(surface.buf8);
}

bool JpgLoader::open(const char* path, const LoaderOps& ops)
{
#ifdef THORVG_FILE_IO_SUPPORT
    if (!(data = (unsigned char*)Loader::open(path, size))) return false;
    owner = Ownership::Transfer;

    int width, height, subSample, colorSpace;
    if (tjDecompressHeader3(jpegDecompressor, data, size, &width, &height, &subSample, &colorSpace) < 0) return false;
    w = static_cast<float>(width);
    h = static_cast<float>(height);
    return true;
#else
    return false;
#endif
}

bool JpgLoader::open(const char* data, uint32_t size, const LoaderOps& ops)
{
    int width, height, subSample, colorSpace;
    if (tjDecompressHeader3(jpegDecompressor, (unsigned char *) data, size, &width, &height, &subSample, &colorSpace) < 0) return false;

    if (ops.owner == Ownership::Copy) {
        this->data = tvg::malloc<unsigned char>(size);
        if (!this->data) return false;
        memcpy((unsigned char *)this->data, data, size);
    } else {
        this->data = (unsigned char *) data;
    }
    owner = ops.owner;
    w = static_cast<float>(width);
    h = static_cast<float>(height);
    this->size = size;

    return true;
}


bool JpgLoader::read()
{
    if (!Loader::read()) return true;

    if (w == 0 || h == 0) return false;

    //determine the image format
    ColorSpace cs;
    TJPF format;
    if (BitmapLoader::cs == ColorSpace::ARGB8888 || BitmapLoader::cs == ColorSpace::ARGB8888S) {
        format = TJPF_BGRX;
        cs = ColorSpace::ARGB8888;
    } else {
        format = TJPF_RGBX;
        cs = ColorSpace::ABGR8888;
    }

    if (static_cast<int>(w) > INT_MAX / static_cast<int>(h) / tjPixelSize[format]) return false;

    auto image = (unsigned char *)tjAlloc(static_cast<int>(w) * static_cast<int>(h) * tjPixelSize[format]);

    //decompress jpg image
    if (tjDecompress2(jpegDecompressor, data, size, image, static_cast<int>(w), 0, static_cast<int>(h), format, 0) < 0) {
        TVGERR("JPG LOADER", "%s", tjGetErrorStr());
        tjFree(image);
        image = nullptr;
        return false;
    }

    surface.setup((pixel_t*)image, w, w, h, sizeof(uint32_t), cs, true);
    clear();
    return true;
}
