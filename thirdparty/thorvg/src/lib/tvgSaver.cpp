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
#include "tvgCommon.h"
#include "tvgSaveModule.h"

#ifdef THORVG_TVG_SAVER_SUPPORT
    #include "tvgTvgSaver.h"
#endif

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct Saver::Impl
{
    SaveModule* saveModule = nullptr;
    ~Impl()
    {
        if (saveModule) delete(saveModule);
    }
};


static SaveModule* _find(FileType type)
{
    switch(type) {
        case FileType::Tvg: {
#ifdef THORVG_TVG_SAVER_SUPPORT
            return new TvgSaver;
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
        case FileType::Tvg: {
            format = "TVG";
            break;
        }
        default: {
            format = "???";
            break;
        }
    }
    TVGLOG("SAVER", "%s format is not supported", format);
#endif
    return nullptr;
}


static SaveModule* _find(const string& path)
{
    auto ext = path.substr(path.find_last_of(".") + 1);
    if (!ext.compare("tvg")) {
        return _find(FileType::Tvg);
    }
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


Result Saver::save(std::unique_ptr<Paint> paint, const string& path, bool compress) noexcept
{
    auto p = paint.release();
    if (!p) return Result::MemoryCorruption;

    //Already on saving an other resource.
    if (pImpl->saveModule) {
        delete(p);
        return Result::InsufficientCondition;
    }

    if (auto saveModule = _find(path)) {
        if (saveModule->save(p, path, compress)) {
            pImpl->saveModule = saveModule;
            return Result::Success;
        } else {
            delete(p);
            delete(saveModule);
            return Result::Unknown;
        }
    }
    delete(p);
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


unique_ptr<Saver> Saver::gen() noexcept
{
    return unique_ptr<Saver>(new Saver);
}
