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

#include <fstream>
#include <memory.h>
#include "tvgLoaderMgr.h"
#include "tvgTvgLoader.h"
#include "tvgTvgLoadParser.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

void TvgLoader::clear()
{
    if (copy) free((char*)data);
    data = nullptr;
    pointer = nullptr;
    size = 0;
    copy = false;
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

    pointer = data;

    return tvgValidateData(pointer, size);
}

bool TvgLoader::open(const char *data, uint32_t size, bool copy)
{
    clear();

    if (copy) {
        this->data = (char*)malloc(size);
        if (!this->data) return false;
        memcpy((char*)this->data, data, size);
    } else this->data = data;

    this->pointer = this->data;
    this->size = size;
    this->copy = copy;

    return tvgValidateData(pointer, size);
}

bool TvgLoader::read()
{
    if (!pointer || size == 0) return false;

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
    root = tvgLoadData(pointer, size);
    if (!root) clear();
}

unique_ptr<Scene> TvgLoader::scene()
{
    this->done();
    if (root) return move(root);
    return nullptr;
}
