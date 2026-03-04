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

#ifndef _TVG_LOAD_MODULE_H_
#define _TVG_LOAD_MODULE_H_

#include <atomic>
#include "tvgCommon.h"
#include "tvgRender.h"
#include "tvgInlist.h"


struct AssetResolver
{
    std::function<bool(Paint* paint, const char* src, void* data)> func;
    void* data;
};


namespace tvg
{

struct LoadModule
{
    INLIST_ITEM(LoadModule);

    //Use either hashkey(data) or hashpath(path)
    uintptr_t hashkey = 0;
    char* hashpath = nullptr;

    FileType type;                                  //current loader file type
    atomic<uint16_t> sharing{};                     //reference count
    bool readied = false;                           //read done already.
    bool cached = false;                            //cached for sharing

    LoadModule(FileType type) : type(type) {}
    virtual ~LoadModule()
    {
        tvg::free(hashpath);
    }

    void cache(uintptr_t data)
    {
        hashkey = data;
        cached = true;
    }

    void cache(char* data)
    {
        hashpath = data;
        cached = true;
    }

    virtual bool open(const char* path) { return false; }
    virtual bool open(const char* data, uint32_t size, const char* rpath, bool copy) { return false; }
    virtual bool resize(Paint* paint, float w, float h) { return false; }
    virtual void sync() {};  //finish immediately if any async update jobs.

    virtual bool read()
    {
        if (readied) return false;
        readied = true;
        return true;
    }

    virtual bool close()
    {
        if (sharing == 0) return true;
        --sharing;
        return false;
    }

    char* open(const char* path, uint32_t& size, bool text = false)
    {
#ifdef THORVG_FILE_IO_SUPPORT
        auto f = fopen(path, text ? "r" : "rb");
        if (!f) return nullptr;

        fseek(f, 0, SEEK_END);

        size = ftell(f);
        if (size == 0) {
            fclose(f);
            return nullptr;
        }

        auto content = tvg::malloc<char>(sizeof(char) * (text ? size + 1 : size));
        fseek(f, 0, SEEK_SET);
        size = fread(content, sizeof(char), size, f);
        if (text) content[size] = '\0';

        fclose(f);

        return content;
#endif
        return nullptr;
    }
};


struct ImageLoader : LoadModule
{
    static atomic<ColorSpace> cs;                   //desired value

    float w = 0, h = 0;                             //default image size
    RenderSurface surface;

    ImageLoader(FileType type) : LoadModule(type) {}

    virtual bool animatable() { return false; }  //true if this loader supports animation.
    virtual Paint* paint() { return nullptr; }
    virtual void set(const AssetResolver* resolver) {}

    virtual RenderSurface* bitmap()
    {
        if (surface.data) return &surface;
        return nullptr;
    }
};


struct FontMetrics
{
    Point size;  //text width, height
    float scale;
    Point align{}, box{}, spacing{1.0f, 1.0f};
    float fontSize = 0.0f;
    TextWrap wrap = TextWrap::None;

    void *engine = nullptr;  //engine extension

    ~FontMetrics()
    {
        tvg::free(engine);
    }
};


struct FontLoader : LoadModule
{
    static constexpr const float DPI = 96.0f / 72.0f;   //dpi base?

    char* name = nullptr;

    FontLoader(FileType type) : LoadModule(type) {}

    using LoadModule::read;

    virtual bool get(FontMetrics& fm, char* text, RenderPath& out) = 0;
    virtual void transform(Paint* paint, FontMetrics& fm, float italicShear) = 0;
    virtual void release(FontMetrics& fm) = 0;
    virtual void metrics(const FontMetrics& fm, TextMetrics& out) = 0;
    virtual void copy(const FontMetrics& in, FontMetrics& out) = 0;
};

}

#endif //_TVG_LOAD_MODULE_H_
