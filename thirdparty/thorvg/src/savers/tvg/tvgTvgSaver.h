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
#ifndef _TVG_TVGSAVER_H_
#define _TVG_TVGSAVER_H_

#include "tvgArray.h"
#include "tvgBinaryDesc.h"
#include "tvgTaskScheduler.h"

namespace tvg
{

class TvgSaver : public SaveModule, public Task
{
private:
    Array<TvgBinByte> buffer;
    Paint* paint = nullptr;
    char *path = nullptr;
    uint32_t headerSize;
    float vsize[2] = {0.0f, 0.0f};
    bool compress;

    bool flushTo(const std::string& path);
    bool saveEncoding(const std::string& path);
    void reserveCount();

    bool writeHeader();
    bool writeViewSize();
    void writeTag(TvgBinTag tag);
    void writeCount(TvgBinCounter cnt);
    void writeReservedCount(TvgBinCounter cnt);
    TvgBinCounter writeData(const void* data, TvgBinCounter cnt);
    TvgBinCounter writeTagProperty(TvgBinTag tag, TvgBinCounter cnt, const void* data);
    TvgBinCounter writeTransform(const Matrix* transform, TvgBinTag tag);

    TvgBinCounter serialize(const Paint* paint, const Matrix* pTransform, bool compTarget = false);
    TvgBinCounter serializeScene(const Scene* scene, const Matrix* pTransform, const Matrix* cTransform);
    TvgBinCounter serializeShape(const Shape* shape, const Matrix* pTransform, const Matrix* cTransform);
    TvgBinCounter serializePicture(const Picture* picture, const Matrix* pTransform, const Matrix* cTransform);
    TvgBinCounter serializePaint(const Paint* paint, const Matrix* pTransform);
    TvgBinCounter serializeFill(const Fill* fill, TvgBinTag tag, const Matrix* pTransform);
    TvgBinCounter serializeStroke(const Shape* shape, const Matrix* pTransform, bool preTransform);
    TvgBinCounter serializePath(const Shape* shape, const Matrix* transform, bool preTransform);
    TvgBinCounter serializeComposite(const Paint* cmpTarget, CompositeMethod cmpMethod, const Matrix* pTransform);
    TvgBinCounter serializeChildren(Iterator* it, const Matrix* transform, bool reserved);
    TvgBinCounter serializeChild(const Paint* parent, const Paint* child, const Matrix* pTransform);

public:
    ~TvgSaver();

    bool save(Paint* paint, const string& path, bool compress) override;
    bool close() override;
    void run(unsigned tid) override;
};

}

#endif  //_TVG_SAVE_MODULE_H_
