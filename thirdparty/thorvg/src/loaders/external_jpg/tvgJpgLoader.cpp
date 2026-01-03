/*
 * Copyright (c) 2021 - 2024 the ThorVG project. All rights reserved.

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
#include <turbojpeg.h>
#include "tvgJpgLoader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

void JpgLoader::clear()
{
    if (freeData) free(data);
    data = nullptr;
    size = 0;
    freeData = false;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

JpgLoader::JpgLoader() : ImageLoader(FileType::Jpg)
{
    jpegDecompressor = tjInitDecompress();
}


JpgLoader::~JpgLoader()
{
    clear();
    tjDestroy(jpegDecompressor);

    //This image is shared with raster engine.
    tjFree(surface.buf8);
}


bool JpgLoader::open(const string& path)
{
#ifdef THORVG_FILE_IO_SUPPORT
    auto jpegFile = fopen(path.c_str(), "rb");
    if (!jpegFile) return false;

    auto ret = false;

    //determine size
    if (fseek(jpegFile, 0, SEEK_END) < 0) goto finalize;
    if (((size = ftell(jpegFile)) < 1)) goto finalize;
    if (fseek(jpegFile, 0, SEEK_SET)) goto finalize;

    data = (unsigned char *) malloc(size);
    if (!data) goto finalize;

    freeData = true;

    if (fread(data, size, 1, jpegFile) < 1) goto failure;

    int width, height, subSample, colorSpace;
    if (tjDecompressHeader3(jpegDecompressor, data, size, &width, &height, &subSample, &colorSpace) < 0) {
        TVGERR("JPG LOADER", "%s", tjGetErrorStr());
        goto failure;
    }

    w = static_cast<float>(width);
    h = static_cast<float>(height);
    ret = true;

    goto finalize;

failure:
    clear();

finalize:
    fclose(jpegFile);
    return ret;
#else
    return false;
#endif
}


bool JpgLoader::open(const char* data, uint32_t size, bool copy)
{
    int width, height, subSample, colorSpace;
    if (tjDecompressHeader3(jpegDecompressor, (unsigned char *) data, size, &width, &height, &subSample, &colorSpace) < 0) return false;

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


bool JpgLoader::read()
{
    if (!LoadModule::read()) return true;

    if (w == 0 || h == 0) return false;

    //determine the image format
    TJPF format;
    if (ImageLoader::cs == ColorSpace::ARGB8888 || ImageLoader::cs == ColorSpace::ARGB8888S) {
        format = TJPF_BGRX;
        surface.cs = ColorSpace::ARGB8888;
    } else {
        format = TJPF_RGBX;
        surface.cs = ColorSpace::ABGR8888;
    }

    auto image = (unsigned char *)tjAlloc(static_cast<int>(w) * static_cast<int>(h) * tjPixelSize[format]);
    if (!image) return false;

    //decompress jpg image
    if (tjDecompress2(jpegDecompressor, data, size, image, static_cast<int>(w), 0, static_cast<int>(h), format, 0) < 0) {
        TVGERR("JPG LOADER", "%s", tjGetErrorStr());
        tjFree(image);
        image = nullptr;
        return false;
    }

    //setup the surface
    surface.buf8 = image;
    surface.stride = w;
    surface.w = w;
    surface.h = h;
    surface.channelSize = sizeof(uint32_t);
    surface.premultiplied = true;

    clear();
    return true;
}
