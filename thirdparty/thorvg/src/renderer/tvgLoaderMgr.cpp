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
#include "tvgLoaderMgr.h"
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

#ifdef THORVG_SFNT_LOADER_SUPPORT
    #include "tvgSfntLoader.h"
#endif

#ifdef THORVG_LOTTIE_LOADER_SUPPORT
    #include "tvgLottieLoader.h"
#endif

#ifdef THORVG_MEDIA_LOADER_SUPPORT
    #include "tvgMediaLoader.h"
#endif

#include "tvgRawLoader.h"

uintptr_t HASH_KEY(const char* data)
{
    return reinterpret_cast<uintptr_t>(data);
}

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

// TODO: remove it.
atomic<ColorSpace> BitmapLoader::cs{ColorSpace::ARGB8888};

static Key _key;
static Inlist<tvg::Loader> _activeLoaders;

static tvg::Loader* _find(FileType type)
{
    switch (type) {
#ifdef THORVG_SVG_LOADER_SUPPORT
        case FileType::Svg: return new SvgLoader;
#endif
#ifdef THORVG_LOTTIE_LOADER_SUPPORT
        case FileType::Lot: return new LottieLoader;
#endif
#ifdef THORVG_SFNT_LOADER_SUPPORT
        case FileType::Sfnt: return new SfntLoader;
#endif
#ifdef THORVG_PNG_LOADER_SUPPORT
        case FileType::Png: return new PngLoader;
#endif
#ifdef THORVG_JPG_LOADER_SUPPORT
        case FileType::Jpg: return new JpgLoader;
#endif
#ifdef THORVG_WEBP_LOADER_SUPPORT
        case FileType::Webp: return new WebpLoader;
#endif
#ifdef THORVG_MEDIA_LOADER_SUPPORT
        case FileType::Media: return MediaLoader::gen();
#endif
        case FileType::Raw: return new RawLoader;
        default: break;
    }

#ifdef THORVG_LOG_ENABLED
    auto toString = [](FileType type) {
        switch (type) {
            case FileType::Svg:  return "SVG";
            case FileType::Lot:  return "LOT";
            case FileType::Sfnt: return "SFNT";
            case FileType::Png:  return "PNG";
            case FileType::Jpg:  return "JPG";
            case FileType::Webp: return "WEBP";
            case FileType::Media: return "MEDIA";
            case FileType::Raw:  return "RAW";
            case FileType::Gif:  return "GIF";
            default:             return "???";
        }
    };
    TVGLOG("RENDERER", "%s format is not supported", toString(type));
#endif
    return nullptr;
}

#ifdef THORVG_FILE_IO_SUPPORT
static tvg::Loader* _findByPath(const char* filename, Result& ret)
{
    auto ext = fileext(filename);
    if (ext) {
        auto type = FileType::Unknown;
        if (!strcmp(ext, "svg")) type = FileType::Svg;
        else if (!strcmp(ext, "lot") || !strcmp(ext, "json")) type = FileType::Lot;
        else if (!strcmp(ext, "ttf") || !strcmp(ext, "ttc") || !strcmp(ext, "otf") || !strcmp(ext, "otc")) type = FileType::Sfnt;
        else if (!strcmp(ext, "png")) type = FileType::Png;
        else if (!strcmp(ext, "jpg")) type = FileType::Jpg;
        else if (!strcmp(ext, "webp")) type = FileType::Webp;
        else if (!strcmp(ext, "mp4")) type = FileType::Media;  // TODO: add common media formats
        if (type != FileType::Unknown) {
            ret = Result::NonSupport;
            return _find(type);
        }
    }
    return nullptr;
}
#endif

static FileType _convert(const char* mimeType)
{
    if (!mimeType) return FileType::Unknown;

    auto type = FileType::Unknown;

    if (!strcmp(mimeType, "svg") || !strcmp(mimeType, "svg+xml")) type = FileType::Svg;
    else if (!strcmp(mimeType, "lot") || !strcmp(mimeType, "lottie+json")) type = FileType::Lot;
    else if (!strcmp(mimeType, "ttf") || !strcmp(mimeType, "otf")) type = FileType::Sfnt;
    else if (!strcmp(mimeType, "png")) type = FileType::Png;
    else if (!strcmp(mimeType, "jpg") || !strcmp(mimeType, "jpeg")) type = FileType::Jpg;
    else if (!strcmp(mimeType, "webp")) type = FileType::Webp;
    else if (!strcmp(mimeType, "mp4")) type = FileType::Media;  // TODO: add common media formats
    else if (!strcmp(mimeType, "raw")) type = FileType::Raw;
    else TVGLOG("RENDERER", "Given mimetype is unknown = \"%s\".", mimeType);

    return type;
}

static tvg::Loader* _findFromCache(const char* filename)
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

static tvg::Loader* _findFromCache(const char* data, uint32_t size, const char* mimeType)
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
    // force clean up the font loaders which is globally used.
    INLIST_SAFE_FOREACH(_activeLoaders, loader) {
        if (loader->type != FileType::Sfnt) continue;
        _activeLoaders.remove(loader);
        delete (loader);
    }
    return true;
}

bool LoaderMgr::retrieve(Loader* loader)
{
    if (!loader) return false;

    if (loader->close()) {
        if (loader->cached) _activeLoaders.remove(loader);
        delete (loader);
    }
    return true;
}

tvg::Loader* LoaderMgr::loader(const char* filename, LoaderOps& ops, Result& ret)
{
#ifdef THORVG_FILE_IO_SUPPORT
    if (!filename) return nullptr;

    if (auto loader = _findFromCache(filename)) return loader;

    ret = Result::InvalidArguments;

    if (auto loader = _findByPath(filename, ret)) {
        // respect the return value here, but ignore trials with unknown MIME types.
        ret = loader->open(filename, ops);
        if (ret == Result::Success) {
            if (loader->cache(filename)) {
                ScopedLock lock(_key);
                _activeLoaders.back(loader);
            }
            return loader;
        }
        delete (loader);
    }

    if (ret == Result::NonSupport) return nullptr;

    // Unknown MimeType. Try with the candidates in the order
    for (int i = 0; i < static_cast<int>(FileType::Unknown); i++) {
        if (auto loader = _find(static_cast<FileType>(i))) {
            if (loader->open(filename, ops) == Result::Success) {
                if (loader->cache(filename)) {
                    ScopedLock lock(_key);
                    _activeLoaders.back(loader);
                }
                return loader;
            }
            delete (loader);
        }
    }
#else
    TVGLOG("RENDERER", "FILE IO is disabled!");
    ret = Result::NonSupport;
#endif
    return nullptr;
}

bool LoaderMgr::retrieve(const char* filename)
{
    return retrieve(_findFromCache(filename));
}

tvg::Loader* LoaderMgr::loader(const char* data, uint32_t size, const char* mimeType, LoaderOps& ops, Result& ret)
{
    // Note that users could use the same data pointer with the different content.
    // Thus caching is only valid for shareable.
    if (ops.owner == Ownership::Borrow) {
        if (auto loader = _findFromCache(data, size, mimeType)) return loader;
    }

    // Try with the given MimeType
    if (mimeType) {
        auto type = _convert(mimeType);
        if (type != FileType::Unknown) ret = Result::NonSupport;
        if (auto loader = _find(type)) {
            // respect the return value here, but ignore trials with unknown MIME types.
            ret = loader->open(data, size, ops);
            if (ret == Result::Success) {
                if (ops.owner == Ownership::Borrow && loader->cache(HASH_KEY(data))) {
                    ScopedLock lock(_key);
                    _activeLoaders.back(loader);
                }
                return loader;
            } else {
                TVGLOG("LOADER", "Given mimetype \"%s\" seems incorrect or not supported.", mimeType);
                delete (loader);
            }
        }
    }

    if (ret == Result::NonSupport) return nullptr;

    // Unknown MimeType. Try with the candidates in the order
    for (int i = 0; i < static_cast<int>(FileType::Unknown); i++) {
        auto loader = _find(static_cast<FileType>(i));
        if (!loader) continue;
        if (loader->open(data, size, ops) == Result::Success) {
            if (ops.owner == Ownership::Borrow && loader->cache(HASH_KEY(data))) {
                ScopedLock lock(_key);
                _activeLoaders.back(loader);
            }
            return loader;
        }
        delete (loader);
    }
    return nullptr;
}

tvg::Loader* LoaderMgr::loader(const uint32_t* data, uint32_t w, uint32_t h, ColorSpace cs, Ownership owner)
{
    // Note that users could use the same data pointer with the different content.
    // Thus caching is only valid for shareable.
    if (owner == Ownership::Borrow) {
        // TODO: should we check premultiplied??
        if (auto loader = _findFromCache((const char*)(data), w * h, "raw")) return loader;
    }

    // function is dedicated for raw images only
    auto loader = new RawLoader;
    if (loader->open(data, w, h, cs, owner)) {
        if (owner == Ownership::Borrow && loader->cache(HASH_KEY((const char*)data))) {
            ScopedLock lock(_key);
            _activeLoaders.back(loader);
        }
        return loader;
    }
    delete (loader);
    return nullptr;
}

// Loads fonts from memory. The loader is always cached so it remains available while setting the font.
tvg::Loader* LoaderMgr::loader(const char* name, const char* data, uint32_t size, TVG_UNUSED const char* mimeType, const LoaderOps& ops)
{
#ifdef THORVG_SFNT_LOADER_SUPPORT
    if (auto loader = font(name)) {
        // user owned memory can be freed anytime.
        if (loader->owner != Ownership::Borrow) {
            if (ops.owner == Ownership::Transfer) tvg::free((char*)data);
            return loader;
        }
    }

    // function is dedicated for SFNT-based font loading
    auto loader = new SfntLoader;
    if (loader->open(data, size, ops) == Result::Success) {
        loader->name = duplicate(name);
        loader->cached = true;  // force it.
        ScopedLock lock(_key);
        _activeLoaders.back(loader);
        return loader;
    }

    TVGLOG("LOADER", "The font data \"%s\" could not be loaded.", name);
    delete (loader);
#endif
    return nullptr;
}

tvg::Loader* LoaderMgr::font(const char* name)
{
    ScopedLock lock(_key);
    INLIST_FOREACH(_activeLoaders, loader) {
        if (loader->type != FileType::Sfnt) continue;
        if (loader->cached && tvg::equal(name, static_cast<FontLoader*>(loader)->name)) {
            ++loader->sharing;
            return loader;
        }
    }
    return nullptr;
}

tvg::Loader* LoaderMgr::anyfont()
{
    ScopedLock lock(_key);
    INLIST_FOREACH(_activeLoaders, loader) {
        if (loader->cached && loader->type == FileType::Sfnt) {
            ++loader->sharing;
            return loader;
        }
    }
    return nullptr;
}
