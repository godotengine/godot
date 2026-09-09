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

#include <cstring>
#include "tvgCommon.h"
#include "tvgStr.h"
#include "tvgSaveModule.h"

#ifdef THORVG_GIF_SAVER_SUPPORT
    #include "tvgGifSaver.h"
#endif

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct Saver::Impl
{
    SaveModule* saveModule = nullptr;
    Paint* bg = nullptr;

    ~Impl()
    {
        delete(saveModule);
        if (bg) bg->unref();
    }
};


static SaveModule* _find(FileType type)
{
    switch(type) {
        case FileType::Gif: {
#ifdef THORVG_GIF_SAVER_SUPPORT
            return new GifSaver;
#endif
            break;
        }
        default: {
            break;
        }
    }

#ifdef THORVG_LOG_ENABLED
    const char *format;
    switch(type) {
        case FileType::Gif: {
            format = "GIF";
            break;
        }
        default: {
            format = "???";
            break;
        }
    }
    TVGLOG("RENDERER", "%s format is not supported", format);
#endif
    return nullptr;
}


static SaveModule* _find(const char* filename)
{
    auto ext = fileext(filename);
    if (ext && !strcmp(ext, "gif")) return _find(FileType::Gif);
    return nullptr;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Saver::Saver() : pImpl(new Impl())
{
}


Saver::~Saver()
{
    delete(pImpl);
}


Result Saver::save(Paint* paint, const char* filename, uint32_t quality) noexcept
{
    if (!paint) return Result::InvalidArguments;

    //Already on saving another resource.
    if (pImpl->saveModule) {
        Paint::rel(paint);
        return Result::InsufficientCondition;
    }

    if (auto saveModule = _find(filename)) {
        if (saveModule->save(paint, pImpl->bg, filename, quality)) {
            pImpl->saveModule = saveModule;
            return Result::Success;
        } else {
            Paint::rel(paint);
            delete(saveModule);
            return Result::Unknown;
        }
    }
    Paint::rel(paint);
    return Result::NonSupport;
}


Result Saver::background(Paint* paint) noexcept
{
    if (!paint) return Result::InvalidArguments;

    if (pImpl->bg) pImpl->bg->unref();
    paint->ref();
    pImpl->bg = paint;

    return Result::Success;
}


Result Saver::save(Animation* animation, const char* filename, uint32_t quality, uint32_t fps) noexcept
{
    if (!animation) return Result::InvalidArguments;

    //animation holds the picture, it must be 1 at the bottom.
    auto remove = animation->picture()->refCnt() <= 1 ? true : false;

    if (tvg::zero(animation->totalFrame())) {
        if (remove) delete(animation);
        return Result::InsufficientCondition;
    }

    //Already on saving another resource.
    if (pImpl->saveModule) {
        if (remove) delete(animation);
        return Result::InsufficientCondition;
    }

    if (auto saveModule = _find(filename)) {
        if (saveModule->save(animation, pImpl->bg, filename, quality, fps)) {
            pImpl->saveModule = saveModule;
            return Result::Success;
        } else {
            if (remove) delete(animation);
            delete(saveModule);
            return Result::Unknown;
        }
    }
    if (remove) delete(animation);
    return Result::NonSupport;
}


Result Saver::sync() noexcept
{
    if (!pImpl->saveModule) return Result::InsufficientCondition;
    pImpl->saveModule->close();
    delete(pImpl->saveModule);
    pImpl->saveModule = nullptr;

    return Result::Success;
}


Saver* Saver::gen() noexcept
{
    return new Saver;
}
