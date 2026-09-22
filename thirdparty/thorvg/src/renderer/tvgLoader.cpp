/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

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

#include <atomic>
#include "tvgInlist.h"
#include "tvgStr.h"
#include "tvgLoader.h"
#include "tvgLock.h"

#ifdef THORVG_SVG_LOADER_SUPPORT
    #include "tvgSvgLoader.h"
#endif

#ifdef THORVG_PNG_LOADER_SUPPORT
    #include "tvgPngLoader.h"
#endif

#ifdef THORVG_JPG_LOADER_SUPPORT
    #include "tvgJpgLoader.h"
#endif

#ifdef THORVG_WEBP_LOADER_SUPPORT
    #include "tvgWebpLoader.h"
#endif

#ifdef THORVG_TTF_LOADER_SUPPORT
    #include "tvgTtfLoader.h"
#endif

#ifdef THORVG_LOTTIE_LOADER_SUPPORT
    #include "tvgLottieLoader.h"
#endif

#include "tvgRawLoader.h"


uintptr_t HASH_KEY(const char* data)
{
    return reinterpret_cast<uintptr_t>(data);
}

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

//TODO: remove it.
atomic<ColorSpace> ImageLoader::cs{ColorSpace::ARGB8888};

static Key _key;
static Inlist<tvg::LoadModule> _activeLoaders;


static tvg::LoadModule* _find(FileType type)
{
    switch(type) {
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
        case FileType::Svg: {
#ifdef THORVG_SVG_LOADER_SUPPORT
            return new SvgLoader;
#endif
            break;
        }
        case FileType::Ttf: {
#ifdef THORVG_TTF_LOADER_SUPPORT
            return new TtfLoader;
#endif
            break;
        }
        case FileType::Lot: {
#ifdef THORVG_LOTTIE_LOADER_SUPPORT
            return new LottieLoader;
#endif
            break;
        }
        case FileType::Raw: {
            return new RawLoader;
            break;
        }
        default: {
            break;
        }
    }

#ifdef THORVG_LOG_ENABLED
    const char *format;
    switch(type) {
        case FileType::Svg: {
            format = "SVG";
            break;
        }
        case FileType::Ttf: {
            format = "TTF";
            break;
        }
        case FileType::Lot: {
            format = "LOT";
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
    TVGLOG("RENDERER", "%s format is not supported", format);
#endif
    return nullptr;
}


#ifdef THORVG_FILE_IO_SUPPORT
static tvg::LoadModule* _findByPath(const char* filename)
{
    auto ext = fileext(filename);
    if (!ext) return nullptr;

    if (!strcmp(ext, "svg")) return _find(FileType::Svg);
    if (!strcmp(ext, "lot") || !strcmp(ext, "json")) return _find(FileType::Lot);
    if (!strcmp(ext, "png")) return _find(FileType::Png);
    if (!strcmp(ext, "jpg")) return _find(FileType::Jpg);
    if (!strcmp(ext, "webp")) return _find(FileType::Webp);
    if (!strcmp(ext, "ttf") || !strcmp(ext, "ttc")) return _find(FileType::Ttf);
    if (!strcmp(ext, "otf") || !strcmp(ext, "otc")) return _find(FileType::Ttf);
    return nullptr;
}
#endif


static FileType _convert(const char* mimeType)
{
    if (!mimeType) return FileType::Unknown;

    auto type = FileType::Unknown;

    if (!strcmp(mimeType, "svg") || !strcmp(mimeType, "svg+xml")) type = FileType::Svg;
    else if (!strcmp(mimeType, "ttf") || !strcmp(mimeType, "otf")) type = FileType::Ttf;
    else if (!strcmp(mimeType, "lot") || !strcmp(mimeType, "lottie+json")) type = FileType::Lot;
    else if (!strcmp(mimeType, "raw")) type = FileType::Raw;
    else if (!strcmp(mimeType, "png")) type = FileType::Png;
    else if (!strcmp(mimeType, "jpg") || !strcmp(mimeType, "jpeg")) type = FileType::Jpg;
    else if (!strcmp(mimeType, "webp")) type = FileType::Webp;
    else TVGLOG("RENDERER", "Given mimetype is unknown = \"%s\".", mimeType);

    return type;
}


static tvg::LoadModule* _findByType(const char* mimeType)
{
    return _find(_convert(mimeType));
}


static tvg::LoadModule* _findFromCache(const char* filename)
{
    ScopedLock lock(_key);
    INLIST_FOREACH(_activeLoaders, loader) {
        if (loader->cached && loader->hashpath && !strcmp(loader->hashpath, filename)) {
            ++loader->sharing;
            return loader;
        }
    }
    return nullptr;
}


static tvg::LoadModule* _findFromCache(const char* data, uint32_t size, const char* mimeType)
{
    auto type = _convert(mimeType);
    if (type == FileType::Unknown) return nullptr;

    auto key = HASH_KEY(data);

    ScopedLock lock(_key);

    INLIST_FOREACH(_activeLoaders, loader) {
        if (loader->type == type && loader->hashkey == key) {
            ++loader->sharing;
            return loader;
        }
    }
    return nullptr;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/


bool LoaderMgr::init()
{
    return true;
}


bool LoaderMgr::term()
{
    //clean up the remained font loaders which is globally used.
    INLIST_SAFE_FOREACH(_activeLoaders, loader) {
        if (loader->type != FileType::Ttf) continue;
        auto ret = loader->close();
        _activeLoaders.remove(loader);
        if (ret) delete(loader);
    }
    return true;
}


bool LoaderMgr::retrieve(LoadModule* loader)
{
    if (!loader) return false;

    if (loader->close()) {
        if (loader->cached) {
            _activeLoaders.remove(loader);
        }
        delete(loader);
    }
    return true;
}


tvg::LoadModule* LoaderMgr::loader(const char* filename, bool* invalid)
{
#ifdef THORVG_FILE_IO_SUPPORT
    *invalid = false;

    //TODO: make lottie sharable.
    auto allowCache = true;
    auto ext = fileext(filename);
    if (ext && (!strcmp(ext, "json") || !strcmp(ext, "lot"))) allowCache = false;

    if (allowCache) {
        if (auto loader = _findFromCache(filename)) return loader;
    }

    if (auto loader = _findByPath(filename)) {
        if (loader->open(filename)) {
            if (allowCache) {
                loader->cache(duplicate(filename));
                {
                    ScopedLock lock(_key);
                    _activeLoaders.back(loader);
                }
            }
            return loader;
        }
        delete(loader);
    }
    //Unknown MimeType. Try with the candidates in the order
    for (int i = 0; i < static_cast<int>(FileType::Raw); i++) {
        if (auto loader = _find(static_cast<FileType>(i))) {
            if (loader->open(filename)) {
                if (allowCache) {
                    loader->cache(duplicate(filename));
                    {
                        ScopedLock lock(_key);
                        _activeLoaders.back(loader);
                    }
                }
                return loader;
            }
            delete(loader);
        }
    }
    *invalid = true;
#endif
    return nullptr;
}


bool LoaderMgr::retrieve(const char* filename)
{
    return retrieve(_findFromCache(filename));
}


tvg::LoadModule* LoaderMgr::loader(const char* data, uint32_t size, const char* mimeType, const char* rpath, bool copy)
{
    //Note that users could use the same data pointer with the different content.
    //Thus caching is only valid for shareable.
    auto allowCache = !copy;

    //TODO: make lottie shareable.
    if (allowCache) {
        auto type = _convert(mimeType);
        if (type == FileType::Lot) allowCache = false;
    }

    if (allowCache) {
        if (auto loader = _findFromCache(data, size, mimeType)) return loader;
    }

    //Try with the given MimeType
    if (mimeType) {
        if (auto loader = _findByType(mimeType)) {
            if (loader->open(data, size, rpath, copy)) {
                if (allowCache) {
                    loader->cache(HASH_KEY(data));
                    ScopedLock lock(_key);
                    _activeLoaders.back(loader);
                }
                return loader;
            } else {
                TVGLOG("LOADER", "Given mimetype \"%s\" seems incorrect or not supported.", mimeType);
                delete(loader);
            }
        }
    }
    //Unknown MimeType. Try with the candidates in the order
    for (int i = 0; i < static_cast<int>(FileType::Raw); i++) {
        auto loader = _find(static_cast<FileType>(i));
        if (loader) {
            if (loader->open(data, size, rpath, copy)) {
                if (allowCache) {
                    loader->cache(HASH_KEY(data));
                    ScopedLock lock(_key);
                    _activeLoaders.back(loader);
                }
                return loader;
            }
            delete(loader);
        }
    }
    return nullptr;
}


tvg::LoadModule* LoaderMgr::loader(const uint32_t *data, uint32_t w, uint32_t h, ColorSpace cs, bool copy)
{
    //Note that users could use the same data pointer with the different content.
    //Thus caching is only valid for shareable.
    if (!copy) {
        //TODO: should we check premultiplied??
        if (auto loader = _findFromCache((const char*)(data), w * h, "raw")) return loader;
    }

    //function is dedicated for raw images only
    auto loader = new RawLoader;
    if (loader->open(data, w, h, cs, copy)) {
        if (!copy) {
            loader->cache(HASH_KEY((const char*)data));
            ScopedLock lock(_key);
            _activeLoaders.back(loader);
        }
        return loader;
    }
    delete(loader);
    return nullptr;
}


//loads fonts from memory - loader is cached (regardless of copy value) in order to access it while setting font
tvg::LoadModule* LoaderMgr::loader(const char* name, const char* data, uint32_t size, TVG_UNUSED const char* mimeType, bool copy)
{
#ifdef THORVG_TTF_LOADER_SUPPORT
    //TODO: add check for mimetype ?
    if (auto loader = font(name)) return loader;

    //function is dedicated for ttf loader (the only supported font loader)
    auto loader = new TtfLoader;
    if (loader->open(data, size, "", copy)) {
        loader->name = duplicate(name);
        loader->cached = true;  //force it.
        ScopedLock lock(_key);
        _activeLoaders.back(loader);
        return loader;
    }

    TVGLOG("LOADER", "The font data \"%s\" could not be loaded.", name);
    delete(loader);
#endif
    return nullptr;
}


tvg::LoadModule* LoaderMgr::font(const char* name)
{
    ScopedLock lock(_key);
    INLIST_FOREACH(_activeLoaders, loader) {
        if (loader->type != FileType::Ttf) continue;
        if (loader->cached && tvg::equal(name, static_cast<FontLoader*>(loader)->name)) {
            ++loader->sharing;
            return loader;
        }
    }
    return nullptr;
}


tvg::LoadModule* LoaderMgr::anyfont()
{
    ScopedLock lock(_key);
    INLIST_FOREACH(_activeLoaders, loader) {
        if (loader->cached && loader->type == FileType::Ttf) {
            ++loader->sharing;
            return loader;
        }
    }
    return nullptr;
}
