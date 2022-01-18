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
#include "tvgMath.h"
#include "tvgSaveModule.h"
#include "tvgTvgSaver.h"
#include "tvgLzw.h"

#include <cstring>

#ifdef _WIN32
    #include <malloc.h>
#else
    #include <alloca.h>
#endif

static FILE* _fopen(const char* filename, const char* mode)
{
#if defined(_MSC_VER) && defined(__clang__)
    FILE *fp;
    auto err = fopen_s(&fp, filename, mode);
    if (err != 0) return nullptr;
    return fp;
#else
    auto fp = fopen(filename, mode);
    if (!fp) return nullptr;
    return fp;
#endif
}

#define SIZE(A) sizeof(A)

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static inline TvgBinCounter SERIAL_DONE(TvgBinCounter cnt)
{
    return SIZE(TvgBinTag) + SIZE(TvgBinCounter) + cnt;
}


/* if the properties are identical, we can merge the shapes. */
static bool _merge(Shape* from, Shape* to)
{
    uint8_t r, g, b, a;
    uint8_t r2, g2, b2, a2;

    //fill
    if (from->fill() || to->fill()) return false;

    r = g = b = a = r2 = g2 = b2 = a2 = 0;

    from->fillColor(&r, &g, &b, &a);
    to->fillColor(&r2, &g2, &b2, &a2);

    if (r != r2 || g != g2 || b != b2 || a != a2) return false;

    //composition
    if (from->composite(nullptr) != CompositeMethod::None) return false;
    if (to->composite(nullptr) != CompositeMethod::None) return false;

    //opacity
    if (from->opacity() != to->opacity()) return false;

    //transform
    auto t1 = from->transform();
    auto t2 = to->transform();

    if (!mathEqual(t1.e11, t2.e11) || !mathEqual(t1.e12, t2.e12) || !mathEqual(t1.e13, t2.e13) ||
        !mathEqual(t1.e21, t2.e21) || !mathEqual(t1.e22, t2.e22) || !mathEqual(t1.e23, t2.e23) ||
        !mathEqual(t1.e31, t2.e31) || !mathEqual(t1.e32, t2.e32) || !mathEqual(t1.e33, t2.e33)) {
       return false;
    }

    //stroke
    r = g = b = a = r2 = g2 = b2 = a2 = 0;

    from->strokeColor(&r, &g, &b, &a);
    to->strokeColor(&r2, &g2, &b2, &a2);

    if (r != r2 || g != g2 || b != b2 || a != a2) return false;

    if (fabs(from->strokeWidth() - to->strokeWidth()) > FLT_EPSILON) return false;

    //OPTIMIZE: Yet we can't merge outlining shapes unless we can support merging shapes feature.
    if (from->strokeWidth() > 0 || to->strokeWidth() > 0) return false;

    if (from->strokeCap() != to->strokeCap()) return false;
    if (from->strokeJoin() != to->strokeJoin()) return false;
    if (from->strokeDash(nullptr) > 0 || to->strokeDash(nullptr) > 0) return false;
    if (from->strokeFill() || to->strokeFill()) return false;

    //fill rule
    if (from->fillRule() != to->fillRule()) return false;

    //Good, identical shapes, we can merge them.
    const PathCommand* cmds = nullptr;
    auto cmdCnt = from->pathCommands(&cmds);

    const Point* pts = nullptr;
    auto ptsCnt = from->pathCoords(&pts);

    to->appendPath(cmds, cmdCnt, pts, ptsCnt);

    return true;
}


bool TvgSaver::saveEncoding(const std::string& path)
{
    if (!compress) return flushTo(path);

    //Try encoding
    auto uncompressed = buffer.data + headerSize;
    auto uncompressedSize = buffer.count - headerSize;

    uint32_t compressedSize, compressedSizeBits;

    auto compressed = lzwEncode(uncompressed, uncompressedSize, &compressedSize, &compressedSizeBits);

    //Failed compression.
    if (!compressed) return flushTo(path);

    //Optimization is ineffective.
    if (compressedSize >= uncompressedSize) {
        free(compressed);
        return flushTo(path);
    }

    TVGLOG("TVG_SAVER", "%s, compressed: %d -> %d, saved rate: %3.2f%%", path.c_str(), uncompressedSize, compressedSize, (1 - ((float) compressedSize / (float) uncompressedSize)) * 100);

    //Update compress size in the header.
    uncompressed -= (TVG_HEADER_COMPRESS_SIZE + TVG_HEADER_RESERVED_LENGTH);

    //Compression Flag
    *uncompressed |= TVG_HEAD_FLAG_COMPRESSED;
    uncompressed += TVG_HEADER_RESERVED_LENGTH;

    //Uncompressed Size
    memcpy(uncompressed, &uncompressedSize, TVG_HEADER_UNCOMPRESSED_SIZE);
    uncompressed += TVG_HEADER_UNCOMPRESSED_SIZE;

    //Comprssed Size
    memcpy(uncompressed, &compressedSize, TVG_HEADER_COMPRESSED_SIZE);
    uncompressed += TVG_HEADER_COMPRESSED_SIZE;

    //Compressed Size Bits
    memcpy(uncompressed, &compressedSizeBits, TVG_HEADER_COMPRESSED_SIZE_BITS);

    //Good optimization, flush to file.
    auto fp = _fopen(path.c_str(), "w+");
    if (!fp) goto fail;

    //write header
    if (fwrite(buffer.data, SIZE(uint8_t), headerSize, fp) == 0) goto fail;

    //write compressed data
    if (fwrite(compressed, SIZE(uint8_t), compressedSize, fp) == 0) goto fail;

    fclose(fp);
    free(compressed);

    return true;

fail:
    if (fp) fclose(fp);
    if (compressed) free(compressed);
    return false;
}


bool TvgSaver::flushTo(const std::string& path)
{
    auto fp = _fopen(path.c_str(), "w+");
    if (!fp) return false;

    if (fwrite(buffer.data, SIZE(uint8_t), buffer.count, fp) == 0) {
        fclose(fp);
        return false;
    }
    fclose(fp);

    return true;
}


/* WARNING: Header format shall not changed! */
bool TvgSaver::writeHeader()
{
    headerSize = TVG_HEADER_SIGNATURE_LENGTH + TVG_HEADER_VERSION_LENGTH + SIZE(vsize) + TVG_HEADER_RESERVED_LENGTH + TVG_HEADER_COMPRESS_SIZE;

    buffer.grow(headerSize);

    //1. Signature
    auto ptr = buffer.ptr();
    memcpy(ptr, TVG_HEADER_SIGNATURE, TVG_HEADER_SIGNATURE_LENGTH);
    ptr += TVG_HEADER_SIGNATURE_LENGTH;

    //2. Version
    memcpy(ptr, TVG_HEADER_VERSION, TVG_HEADER_VERSION_LENGTH);
    ptr += TVG_HEADER_VERSION_LENGTH;

    buffer.count += (TVG_HEADER_SIGNATURE_LENGTH + TVG_HEADER_VERSION_LENGTH);

    //3. View Size
    writeData(vsize, SIZE(vsize));
    ptr += SIZE(vsize);

    //4. Reserved data + Compress size
    memset(ptr, 0x00, TVG_HEADER_RESERVED_LENGTH + TVG_HEADER_COMPRESS_SIZE);
    buffer.count += (TVG_HEADER_RESERVED_LENGTH + TVG_HEADER_COMPRESS_SIZE);

    return true;
}


void TvgSaver::writeTag(TvgBinTag tag)
{
    buffer.grow(SIZE(TvgBinTag));
    memcpy(buffer.ptr(), &tag, SIZE(TvgBinTag));
    buffer.count += SIZE(TvgBinTag);
}


void TvgSaver::writeCount(TvgBinCounter cnt)
{
    buffer.grow(SIZE(TvgBinCounter));
    memcpy(buffer.ptr(), &cnt, SIZE(TvgBinCounter));
    buffer.count += SIZE(TvgBinCounter);
}


void TvgSaver::writeReservedCount(TvgBinCounter cnt)
{
    memcpy(buffer.ptr() - cnt - SIZE(TvgBinCounter), &cnt, SIZE(TvgBinCounter));
}


void TvgSaver::reserveCount()
{
    buffer.grow(SIZE(TvgBinCounter));
    buffer.count += SIZE(TvgBinCounter);
}


TvgBinCounter TvgSaver::writeData(const void* data, TvgBinCounter cnt)
{
    buffer.grow(cnt);
    memcpy(buffer.ptr(), data, cnt);
    buffer.count += cnt;

    return cnt;
}


TvgBinCounter TvgSaver::writeTagProperty(TvgBinTag tag, TvgBinCounter cnt, const void* data)
{
    auto growCnt = SERIAL_DONE(cnt);

    buffer.grow(growCnt);

    auto ptr = buffer.ptr();

    *ptr = tag;
    ++ptr;

    memcpy(ptr, &cnt, SIZE(TvgBinCounter));
    ptr += SIZE(TvgBinCounter);

    memcpy(ptr, data, cnt);
    ptr += cnt;

    buffer.count += growCnt;

    return growCnt;
}


TvgBinCounter TvgSaver::writeTransform(const Matrix* transform, TvgBinTag tag)
{
    if (!mathIdentity(transform)) return writeTagProperty(tag, SIZE(Matrix), transform);
    return 0;
}


TvgBinCounter TvgSaver::serializePaint(const Paint* paint, const Matrix* pTransform)
{
    TvgBinCounter cnt = 0;

    //opacity
    auto opacity = paint->opacity();
    if (opacity < 255) {
        cnt += writeTagProperty(TVG_TAG_PAINT_OPACITY, SIZE(opacity), &opacity);
    }

    //composite
    const Paint* cmpTarget = nullptr;
    auto cmpMethod = paint->composite(&cmpTarget);
    if (cmpMethod != CompositeMethod::None && cmpTarget) {
        cnt += serializeComposite(cmpTarget, cmpMethod, pTransform);
    }

    return cnt;
}


/* Propagate parents properties to the child so that we can skip saving the parent. */
TvgBinCounter TvgSaver::serializeChild(const Paint* parent, const Paint* child, const Matrix* transform)
{
    const Paint* compTarget = nullptr;
    auto compMethod = parent->composite(&compTarget);

    /* If the parent & the only child have composition, we can't skip the parent...
       Or if the parent has the transform and composition, we can't skip the parent... */
    if (compMethod != CompositeMethod::None) {
        if (transform || child->composite(nullptr) != CompositeMethod::None) return 0;
    }

    //propagate opacity
    uint32_t opacity = parent->opacity();

    if (opacity < 255) {
        uint32_t tmp = (child->opacity() * opacity);
        if (tmp > 0) tmp /= 255;
        const_cast<Paint*>(child)->opacity(tmp);
    }

    //propagate composition
    if (compTarget) const_cast<Paint*>(child)->composite(unique_ptr<Paint>(compTarget->duplicate()), compMethod);

    return serialize(child, transform);
}


TvgBinCounter TvgSaver::serializeScene(const Scene* scene, const Matrix* pTransform, const Matrix* cTransform)
{
    auto it = this->iterator(scene);
    if (it->count() == 0) {
        delete(it);
        return 0;
    }

    //Case - Only Child: Skip saving this scene.
    if (it->count() == 1) {
        auto cnt = serializeChild(scene, it->next(), cTransform);
        if (cnt > 0) {
            delete(it);
            return cnt;
        }
    }

    it->begin();

    //Case - Delegator Scene: This scene is just a delegator, we can skip this:
    if (scene->composite(nullptr) == CompositeMethod::None && scene->opacity() == 255) {
        auto ret = serializeChildren(it, cTransform, false);
        delete(it);
        return ret;
    }

    //Case - Serialize Scene & its children
    writeTag(TVG_TAG_CLASS_SCENE);
    reserveCount();

    auto cnt = serializeChildren(it, cTransform, true) + serializePaint(scene, pTransform);

    delete(it);

    writeReservedCount(cnt);

    return SERIAL_DONE(cnt);
}


TvgBinCounter TvgSaver::serializeFill(const Fill* fill, TvgBinTag tag, const Matrix* pTransform)
{
    const Fill::ColorStop* stops = nullptr;
    auto stopsCnt = fill->colorStops(&stops);
    if (!stops || stopsCnt == 0) return 0;

    writeTag(tag);
    reserveCount();

    TvgBinCounter cnt = 0;

    //radial fill
    if (fill->identifier() == TVG_CLASS_ID_RADIAL) {
        float args[3];
        static_cast<const RadialGradient*>(fill)->radial(args, args + 1, args + 2);
        cnt += writeTagProperty(TVG_TAG_FILL_RADIAL_GRADIENT, SIZE(args), args);
    //linear fill
    } else {
        float args[4];
        static_cast<const LinearGradient*>(fill)->linear(args, args + 1, args + 2, args + 3);
        cnt += writeTagProperty(TVG_TAG_FILL_LINEAR_GRADIENT, SIZE(args), args);
    }

    if (auto flag = static_cast<TvgBinFlag>(fill->spread()))
        cnt += writeTagProperty(TVG_TAG_FILL_FILLSPREAD, SIZE(TvgBinFlag), &flag);
    cnt += writeTagProperty(TVG_TAG_FILL_COLORSTOPS, stopsCnt * SIZE(Fill::ColorStop), stops);

    auto gTransform = fill->transform();
    if (pTransform) gTransform = mathMultiply(pTransform, &gTransform);

    cnt += writeTransform(&gTransform, TVG_TAG_FILL_TRANSFORM);

    writeReservedCount(cnt);

    return SERIAL_DONE(cnt);
}


TvgBinCounter TvgSaver::serializeStroke(const Shape* shape, const Matrix* pTransform, bool preTransform)
{
    writeTag(TVG_TAG_SHAPE_STROKE);
    reserveCount();

    //width
    auto width = shape->strokeWidth();
    if (preTransform) width *= sqrtf(powf(pTransform->e11, 2.0f) + powf(pTransform->e21, 2.0f));  //we know x/y scaling factors are same.
    auto cnt = writeTagProperty(TVG_TAG_SHAPE_STROKE_WIDTH, SIZE(width), &width);

    //cap
    if (auto flag = static_cast<TvgBinFlag>(shape->strokeCap()))
        cnt += writeTagProperty(TVG_TAG_SHAPE_STROKE_CAP, SIZE(TvgBinFlag), &flag);

    //join
    if (auto flag = static_cast<TvgBinFlag>(shape->strokeJoin()))
        cnt += writeTagProperty(TVG_TAG_SHAPE_STROKE_JOIN, SIZE(TvgBinFlag), &flag);

    //fill
    if (auto fill = shape->strokeFill()) {
        cnt += serializeFill(fill, TVG_TAG_SHAPE_STROKE_FILL, (preTransform ? pTransform : nullptr));
    } else {
        uint8_t color[4] = {0, 0, 0, 0};
        shape->strokeColor(color, color + 1, color + 2, color + 3);
        cnt += writeTagProperty(TVG_TAG_SHAPE_STROKE_COLOR, SIZE(color), &color);
    }

    //dash
    const float* dashPattern = nullptr;
    auto dashCnt = shape->strokeDash(&dashPattern);
    if (dashPattern && dashCnt > 0) {
        TvgBinCounter dashCntSize = SIZE(dashCnt);
        TvgBinCounter dashPtrnSize = dashCnt * SIZE(dashPattern[0]);

        writeTag(TVG_TAG_SHAPE_STROKE_DASHPTRN);
        writeCount(dashCntSize + dashPtrnSize);
        cnt += writeData(&dashCnt, dashCntSize);
        cnt += writeData(dashPattern, dashPtrnSize);
        cnt += SIZE(TvgBinTag) + SIZE(TvgBinCounter);
    }

    writeReservedCount(cnt);

    return SERIAL_DONE(cnt);
}


TvgBinCounter TvgSaver::serializePath(const Shape* shape, const Matrix* transform, bool preTransform)
{
    const PathCommand* cmds = nullptr;
    auto cmdCnt = shape->pathCommands(&cmds);
    const Point* pts = nullptr;
    auto ptsCnt = shape->pathCoords(&pts);

    if (!cmds || !pts || cmdCnt == 0 || ptsCnt == 0) return 0;

    writeTag(TVG_TAG_SHAPE_PATH);
    reserveCount();

    /* Reduce the binary size.
       Convert PathCommand(4 bytes) to TvgBinFlag(1 byte) */
    TvgBinFlag* outCmds = (TvgBinFlag*)alloca(SIZE(TvgBinFlag) * cmdCnt);
    for (uint32_t i = 0; i < cmdCnt; ++i) {
        outCmds[i] = static_cast<TvgBinFlag>(cmds[i]);
    }

    auto cnt = writeData(&cmdCnt, SIZE(cmdCnt));
    cnt += writeData(&ptsCnt, SIZE(ptsCnt));
    cnt += writeData(outCmds, SIZE(TvgBinFlag) * cmdCnt);

    //transform?
    if (preTransform) {
        if (!mathEqual(transform->e11, 1.0f) || !mathZero(transform->e12) || !mathZero(transform->e13) ||
            !mathZero(transform->e21) || !mathEqual(transform->e22, 1.0f) || !mathZero(transform->e23) ||
            !mathZero(transform->e31) || !mathZero(transform->e32) || !mathEqual(transform->e33, 1.0f)) {
            auto p = const_cast<Point*>(pts);
            for (uint32_t i = 0; i < ptsCnt; ++i) mathMultiply(p++, transform);
        }
    }

    cnt += writeData(pts, ptsCnt * SIZE(pts[0]));

    writeReservedCount(cnt);

    return SERIAL_DONE(cnt);
}


TvgBinCounter TvgSaver::serializeShape(const Shape* shape, const Matrix* pTransform, const Matrix* cTransform)
{
    writeTag(TVG_TAG_CLASS_SHAPE);
    reserveCount();
    TvgBinCounter cnt = 0;

    //fill rule
    if (auto flag = static_cast<TvgBinFlag>(shape->fillRule())) {
        cnt = writeTagProperty(TVG_TAG_SHAPE_FILLRULE, SIZE(TvgBinFlag), &flag);
    }

    //the pre-transformation can't be applied in the case when the stroke is dashed or irregulary scaled
    bool preTransform = true;

    //stroke
    if (shape->strokeWidth() > 0) {
        uint8_t color[4] = {0, 0, 0, 0};
        shape->strokeColor(color, color + 1, color + 2, color + 3);
        auto fill = shape->strokeFill();
        if (fill || color[3] > 0) {
            if (!mathEqual(cTransform->e11, cTransform->e22) || (mathZero(cTransform->e11) && !mathEqual(cTransform->e12, cTransform->e21)) || shape->strokeDash(nullptr) > 0) preTransform = false;
            cnt += serializeStroke(shape, cTransform, preTransform);
        }
    }

    //fill
    if (auto fill = shape->fill()) {
        cnt += serializeFill(fill, TVG_TAG_SHAPE_FILL, (preTransform ? cTransform : nullptr));
    } else {
        uint8_t color[4] = {0, 0, 0, 0};
        shape->fillColor(color, color + 1, color + 2, color + 3);
        if (color[3] > 0) cnt += writeTagProperty(TVG_TAG_SHAPE_COLOR, SIZE(color), color);
    }

    cnt += serializePath(shape, cTransform, preTransform);

    if (!preTransform) cnt += writeTransform(cTransform, TVG_TAG_PAINT_TRANSFORM);
    cnt += serializePaint(shape, pTransform);

    writeReservedCount(cnt);

    return SERIAL_DONE(cnt);
}


/* Picture has either a vector scene or a bitmap. */
TvgBinCounter TvgSaver::serializePicture(const Picture* picture, const Matrix* pTransform, const Matrix* cTransform)
{
    auto it = this->iterator(picture);

    //Case - Vector Scene:
    if (it->count() == 1) {
        auto cnt = serializeChild(picture, it->next(), cTransform);
        //Only child, Skip to save Picture...
        if (cnt > 0) {
            delete(it);
            return cnt;
        /* Unfortunately, we can't skip the Picture because it might have a compositor,
           Serialize Scene(instead of the Picture) & its scene. */
        } else {
            writeTag(TVG_TAG_CLASS_SCENE);
            reserveCount();
            auto cnt = serializeChildren(it, cTransform, true) + serializePaint(picture, pTransform);
            writeReservedCount(cnt);
            delete(it);
            return SERIAL_DONE(cnt);
        }
    }
    delete(it);

    //Case - Bitmap Image:
    uint32_t w, h;
    auto pixels = picture->data(&w, &h);
    if (!pixels) return 0;

    writeTag(TVG_TAG_CLASS_PICTURE);
    reserveCount();

    TvgBinCounter cnt = 0;
    TvgBinCounter sizeCnt = SIZE(w);
    TvgBinCounter imgSize = w * h * SIZE(pixels[0]);

    writeTag(TVG_TAG_PICTURE_RAW_IMAGE);
    writeCount(2 * sizeCnt + imgSize);

    cnt += writeData(&w, sizeCnt);
    cnt += writeData(&h, sizeCnt);
    cnt += writeData(pixels, imgSize);
    cnt += SIZE(TvgBinTag) + SIZE(TvgBinCounter);

    //Bitmap picture needs the transform info.
    cnt += writeTransform(cTransform, TVG_TAG_PAINT_TRANSFORM);

    cnt += serializePaint(picture, pTransform);

    writeReservedCount(cnt);

    return SERIAL_DONE(cnt);
}


TvgBinCounter TvgSaver::serializeComposite(const Paint* cmpTarget, CompositeMethod cmpMethod, const Matrix* pTransform)
{
    writeTag(TVG_TAG_PAINT_CMP_TARGET);
    reserveCount();

    auto flag = static_cast<TvgBinFlag>(cmpMethod);
    auto cnt = writeTagProperty(TVG_TAG_PAINT_CMP_METHOD, SIZE(TvgBinFlag), &flag);

    cnt += serialize(cmpTarget, pTransform, true);

    writeReservedCount(cnt);

    return SERIAL_DONE(cnt);
}


TvgBinCounter TvgSaver::serializeChildren(Iterator* it, const Matrix* pTransform, bool reserved)
{
    TvgBinCounter cnt = 0;

    //Merging shapes. the result is written in the children.
    Array<const Paint*> children;
    children.reserve(it->count());
    children.push(it->next());

    while (auto child = it->next()) {
        if (child->identifier() == TVG_CLASS_ID_SHAPE) {
            //only dosable if the previous child is a shape.
            auto target = children.ptr() - 1;
            if ((*target)->identifier() == TVG_CLASS_ID_SHAPE) {
                if (_merge((Shape*)child, (Shape*)*target)) {
                    continue;
                }
            }
        }
        children.push(child);
    }

    //The children of a reserved scene
    if (reserved && children.count > 1) {
        cnt += writeTagProperty(TVG_TAG_SCENE_RESERVEDCNT, SIZE(children.count), &children.count);
    }

    //Serialize merged children.
    auto child = children.data;
    for (uint32_t i = 0; i < children.count; ++i, ++child) {
        cnt += serialize(*child, pTransform);
    }

    return cnt;
}


TvgBinCounter TvgSaver::serialize(const Paint* paint, const Matrix* pTransform, bool compTarget)
{
    if (!paint) return 0;

    //Invisible paint, no point to save it if the paint is not the composition target...
    if (!compTarget && paint->opacity() == 0) return 0;

    auto transform = const_cast<Paint*>(paint)->transform();
    if (pTransform) transform = mathMultiply(pTransform, &transform);

    switch (paint->identifier()) {
        case TVG_CLASS_ID_SHAPE: return serializeShape(static_cast<const Shape*>(paint), pTransform, &transform);
        case TVG_CLASS_ID_SCENE: return serializeScene(static_cast<const Scene*>(paint), pTransform, &transform);
        case TVG_CLASS_ID_PICTURE: return serializePicture(static_cast<const Picture*>(paint), pTransform, &transform);
    }

    return 0;
}


void TvgSaver::run(unsigned tid)
{
    if (!writeHeader()) return;

    //Serialize Root Paint, without its transform.
    Matrix transform = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    if (paint->opacity() > 0) {
        switch (paint->identifier()) {
            case TVG_CLASS_ID_SHAPE: {
                serializeShape(static_cast<const Shape*>(paint), nullptr, &transform);
                break;
            }
            case TVG_CLASS_ID_SCENE: {
                serializeScene(static_cast<const Scene*>(paint), nullptr, &transform);
                break;
            }
            case TVG_CLASS_ID_PICTURE: {
                serializePicture(static_cast<const Picture*>(paint), nullptr, &transform);
                break;
            }
        }
    }

    if (!saveEncoding(path)) return;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

TvgSaver::~TvgSaver()
{
    close();
}


bool TvgSaver::close()
{
    this->done();

    if (paint) {
        delete(paint);
        paint = nullptr;
    }
    if (path) {
        free(path);
        path = nullptr;
    }
    buffer.reset();
    return true;
}


bool TvgSaver::save(Paint* paint, const string& path, bool compress)
{
    close();

    float x, y;
    x = y = 0;
    paint->bounds(&x, &y, &vsize[0], &vsize[1], false);

    //cut off the negative space
    if (x < 0) vsize[0] += x;
    if (y < 0) vsize[1] += y;

    if (vsize[0] < FLT_EPSILON || vsize[1] < FLT_EPSILON) {
        TVGLOG("TVG_SAVER", "Saving paint(%p) has zero view size.", paint);
        return false;
    }

    this->path = strdup(path.c_str());
    if (!this->path) return false;

    this->paint = paint;
    this->compress = compress;

    TaskScheduler::request(this);

    return true;
}
