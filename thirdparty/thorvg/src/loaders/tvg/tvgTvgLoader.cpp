/*
 * Copyright (c) 2021 - 2022 Samsung Electronics Co., Ltd. All rights reserved.

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
#include <fstream>
#include "tvgLoader.h"
#include "tvgTvgLoader.h"
#include "tvgLzw.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/


void TvgLoader::clear()
{
    if (copy) free((char*)data);
    ptr = data = nullptr;
    size = 0;
    copy = false;

    if (interpreter) {
        delete(interpreter);
        interpreter = nullptr;
    }
}


/* WARNING: Header format shall not change! */
bool TvgLoader::readHeader()
{
    if (!ptr) return false;

    //Make sure the size is large enough to hold the header
    if (size < TVG_HEADER_SIZE) return false;

    //1. Signature
    if (memcmp(ptr, TVG_HEADER_SIGNATURE, TVG_HEADER_SIGNATURE_LENGTH)) return false;
    ptr += TVG_HEADER_SIGNATURE_LENGTH;

    //2. Version
    char version[TVG_HEADER_VERSION_LENGTH + 1];
    memcpy(version, ptr, TVG_HEADER_VERSION_LENGTH);
    version[TVG_HEADER_VERSION_LENGTH - 1] = '\0';
    ptr += TVG_HEADER_VERSION_LENGTH;
    this->version = atoi(version);
    if (this->version > THORVG_VERSION_NUMBER()) {
        TVGLOG("TVG", "This TVG file expects a higher version(%d) of ThorVG symbol(%d)", this->version, THORVG_VERSION_NUMBER());
    }

    //3. View Size
    READ_FLOAT(&w, ptr);
    ptr += SIZE(float);
    READ_FLOAT(&h, ptr);
    ptr += SIZE(float);

    //4. Reserved
    if (*ptr & TVG_HEAD_FLAG_COMPRESSED) compressed = true;
    ptr += TVG_HEADER_RESERVED_LENGTH;

    //5. Compressed Size if any
    if (compressed) {
        auto p = ptr;

        //TVG_HEADER_UNCOMPRESSED_SIZE
        memcpy(&uncompressedSize, p, sizeof(uint32_t));
        p += SIZE(uint32_t);

        //TVG_HEADER_COMPRESSED_SIZE
        memcpy(&compressedSize, p, sizeof(uint32_t));
        p += SIZE(uint32_t);

        //TVG_HEADER_COMPRESSED_SIZE_BITS
        memcpy(&compressedSizeBits, p, sizeof(uint32_t));
    }

    ptr += TVG_HEADER_COMPRESS_SIZE;

    //Decide the proper Tvg Binary Interpreter based on the current file version
    if (this->version >= 0) interpreter = new TvgBinInterpreter;

    return true;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

TvgLoader::~TvgLoader()
{
    close();
}


bool TvgLoader::open(const string &path)
{
    clear();

    ifstream f;
    f.open(path, ifstream::in | ifstream::binary | ifstream::ate);

    if (!f.is_open()) return false;

    size = f.tellg();
    f.seekg(0, ifstream::beg);

    copy = true;
    data = (char*)malloc(size);
    if (!data) {
        clear();
        f.close();
        return false;
    }

    if (!f.read((char*)data, size))
    {
        clear();
        f.close();
        return false;
    }

    f.close();

    ptr = data;

    return readHeader();
}


bool TvgLoader::open(const char *data, uint32_t size, bool copy)
{
    clear();

    if (copy) {
        this->data = (char*)malloc(size);
        if (!this->data) return false;
        memcpy((char*)this->data, data, size);
    } else this->data = data;

    this->ptr = this->data;
    this->size = size;
    this->copy = copy;

    return readHeader();
}


bool TvgLoader::resize(Paint* paint, float w, float h)
{
    if (!paint) return false;

    auto sx = w / this->w;
    auto sy = h / this->h;

    //Scale
    auto scale = sx < sy ? sx : sy;
    paint->scale(scale);

    //Align
    float tx = 0, ty = 0;
    auto sw = this->w * scale;
    auto sh = this->h * scale;
    if (sw > sh) ty -= (h - sh) * 0.5f;
    else tx -= (w - sw) * 0.5f;
    paint->translate(-tx, -ty);

    return true;
}


bool TvgLoader::read()
{
    if (!ptr || size == 0) return false;

    TaskScheduler::request(this);

    return true;
}


bool TvgLoader::close()
{
    this->done();
    clear();
    return true;
}


void TvgLoader::run(unsigned tid)
{
    if (root) root.reset();

    auto data = const_cast<char*>(ptr);

    if (compressed) {
        data = (char*) lzwDecode((uint8_t*) data, compressedSize, compressedSizeBits, uncompressedSize);
        root = interpreter->run(data, data + uncompressedSize);
        free(data);
    } else {
        root = interpreter->run(data, this->data + size);
    }

    if (!root) clear();
}


unique_ptr<Paint> TvgLoader::paint()
{
    this->done();
    if (root) return move(root);
    return nullptr;
}
