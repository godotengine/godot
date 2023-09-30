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

#ifdef THORVG_SVG_LOADER_SUPPORT
    #include "tvgSvgLoader.h"
#endif

#ifdef THORVG_PNG_LOADER_SUPPORT
    #include "tvgPngLoader.h"
#endif

#ifdef THORVG_TVG_LOADER_SUPPORT
    #include "tvgTvgLoader.h"
#endif

#ifdef THORVG_JPG_LOADER_SUPPORT
    #include "tvgJpgLoader.h"
#endif

#ifdef THORVG_WEBP_LOADER_SUPPORT
    #include "tvgWebpLoader.h"
#endif

#ifdef THORVG_LOTTIE_LOADER_SUPPORT
    #include "tvgLottieLoader.h"
#endif

#include "tvgRawLoader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static LoadModule* _find(FileType type)
{
    switch(type) {
        case FileType::Tvg: {
#ifdef THORVG_TVG_LOADER_SUPPORT
            return new TvgLoader;
#endif
            break;
        }
        case FileType::Svg: {
#ifdef THORVG_SVG_LOADER_SUPPORT
            return new SvgLoader;
#endif
            break;
        }
        case FileType::Lottie: {
#ifdef THORVG_LOTTIE_LOADER_SUPPORT
            return new LottieLoader;
#endif
            break;
        }
        case FileType::Raw: {
            return new RawLoader;
            break;
        }
        case FileType::Png: {
#ifdef THORVG_PNG_LOADER_SUPPORT
            return new PngLoader;
#endif
            break;
        }
        case FileType::Jpg: {
#ifdef THORVG_JPG_LOADER_SUPPORT
            return new JpgLoader;
#endif
            break;
        }
        case FileType::Webp: {
#ifdef THORVG_WEBP_LOADER_SUPPORT
            return new WebpLoader;
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
        case FileType::Svg: {
            format = "SVG";
            break;
        }
        case FileType::Lottie: {
            format = "lottie(json)";
            break;
        }
        case FileType::Raw: {
            format = "RAW";
            break;
        }
        case FileType::Png: {
            format = "PNG";
            break;
        }
        case FileType::Jpg: {
            format = "JPG";
            break;
        }
        case FileType::Webp: {
            format = "WEBP";
            break;
        }
        default: {
            format = "???";
            break;
        }
    }
    TVGLOG("LOADER", "%s format is not supported", format);
#endif
    return nullptr;
}


static LoadModule* _findByPath(const string& path)
{
    auto ext = path.substr(path.find_last_of(".") + 1);
    if (!ext.compare("tvg")) return _find(FileType::Tvg);
    if (!ext.compare("svg")) return _find(FileType::Svg);
    if (!ext.compare("json")) return _find(FileType::Lottie);
    if (!ext.compare("lottie")) return _find(FileType::Lottie);
    if (!ext.compare("png")) return _find(FileType::Png);
    if (!ext.compare("jpg")) return _find(FileType::Jpg);
    if (!ext.compare("webp")) return _find(FileType::Webp);
    return nullptr;
}


static LoadModule* _findByType(const string& mimeType)
{
    if (mimeType.empty()) return nullptr;

    auto type = FileType::Unknown;

    if (mimeType == "tvg") type = FileType::Tvg;
    else if (mimeType == "svg" || mimeType == "svg+xml") type = FileType::Svg;
    else if (mimeType == "lottie") type = FileType::Lottie;
    else if (mimeType == "raw") type = FileType::Raw;
    else if (mimeType == "png") type = FileType::Png;
    else if (mimeType == "jpg" || mimeType == "jpeg") type = FileType::Jpg;
    else if (mimeType == "webp") type = FileType::Webp;
    else {
        TVGLOG("LOADER", "Given mimetype is unknown = \"%s\".", mimeType.c_str());
        return nullptr;
    }

    return _find(type);
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/


bool LoaderMgr::init()
{
    //TODO:

    return true;
}


bool LoaderMgr::term()
{
    //TODO:

    return true;
}


shared_ptr<LoadModule> LoaderMgr::loader(const string& path, bool* invalid)
{
    *invalid = false;

    if (auto loader = _findByPath(path)) {
        if (loader->open(path)) return shared_ptr<LoadModule>(loader);
        else delete(loader);
        *invalid = true;
    }
    return nullptr;
}


shared_ptr<LoadModule> LoaderMgr::loader(const char* data, uint32_t size, const string& mimeType, bool copy)
{
    //Try with the given MimeType
    if (!mimeType.empty()) {
        if (auto loader = _findByType(mimeType)) {
            if (loader->open(data, size, copy)) {
                return shared_ptr<LoadModule>(loader);
            } else {
                TVGLOG("LOADER", "Given mimetype \"%s\" seems incorrect or not supported.", mimeType.c_str());
                delete(loader);
            }
        }
    //Unkown MimeType. Try with the candidates in the order
    } else {
        for (int i = 0; i < static_cast<int>(FileType::Unknown); i++) {
            auto loader = _find(static_cast<FileType>(i));
            if (loader) {
                if (loader->open(data, size, copy)) return shared_ptr<LoadModule>(loader);
                else delete(loader);
            }
        }
    }
    return nullptr;
}


shared_ptr<LoadModule> LoaderMgr::loader(const uint32_t *data, uint32_t w, uint32_t h, bool copy)
{
    //function is dedicated for raw images only
    auto loader = new RawLoader;
    if (loader->open(data, w, h, copy)) return shared_ptr<LoadModule>(loader);
    else delete(loader);

    return nullptr;
}
