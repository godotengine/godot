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
#include "tvgJpgLoader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

void JpgLoader::clear()
{
    jpgdDelete(decoder);
    if (freeData) free(data);
    decoder = nullptr;
    data = nullptr;
    freeData = false;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/


JpgLoader::~JpgLoader()
{
    jpgdDelete(decoder);
    if (freeData) free(data);
}


bool JpgLoader::open(const string& path)
{
    clear();

    int width, height;
    decoder = jpgdHeader(path.c_str(), &width, &height);
    if (!decoder) return false;

    w = static_cast<float>(width);
    h = static_cast<float>(height);

    return true;
}


bool JpgLoader::open(const char* data, uint32_t size, bool copy)
{
    clear();

    if (copy) {
        this->data = (char *) malloc(size);
        if (!this->data) return false;
        memcpy((char *)this->data, data, size);
        freeData = true;
    } else {
        this->data = (char *) data;
        freeData = false;
    }

    int width, height;
    decoder = jpgdHeader(this->data, size, &width, &height);
    if (!decoder) return false;

    w = static_cast<float>(width);
    h = static_cast<float>(height);

    return true;
}



bool JpgLoader::read()
{
    if (!decoder || w <= 0 || h <= 0) return false;

    TaskScheduler::request(this);

    return true;
}


bool JpgLoader::close()
{
    this->done();
    clear();
    return true;
}


unique_ptr<Surface> JpgLoader::bitmap()
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


void JpgLoader::run(unsigned tid)
{
    image = jpgdDecompress(decoder);
}