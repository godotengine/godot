/*
 * Copyright (c) 2023 - 2024 the ThorVG project. All rights reserved.

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


#include "tvgText.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/



/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/


Text::Text() : pImpl(new Impl)
{
    Paint::pImpl->id = TVG_CLASS_ID_TEXT;
}


Text::~Text()
{
    delete(pImpl);
}


Result Text::text(const char* text) noexcept
{
    return pImpl->text(text);
}


Result Text::font(const char* name, float size, const char* style) noexcept
{
    return pImpl->font(name, size, style);
}


Result Text::load(const std::string& path) noexcept
{
    bool invalid; //invalid path
    if (!LoaderMgr::loader(path, &invalid)) {
        if (invalid) return Result::InvalidArguments;
        else return Result::NonSupport;
    }

    return Result::Success;
}


Result Text::unload(const std::string& path) noexcept
{
    if (LoaderMgr::retrieve(path)) return Result::Success;
    return Result::InsufficientCondition;
}


Result Text::fill(uint8_t r, uint8_t g, uint8_t b) noexcept
{
    if (!pImpl->paint) return Result::InsufficientCondition;

    return pImpl->fill(r, g, b);
}


Result Text::fill(unique_ptr<Fill> f) noexcept
{
    if (!pImpl->paint) return Result::InsufficientCondition;

    auto p = f.release();
    if (!p) return Result::MemoryCorruption;

    return pImpl->fill(p);
}


unique_ptr<Text> Text::gen() noexcept
{
    return unique_ptr<Text>(new Text);
}


uint32_t Text::identifier() noexcept
{
    return TVG_CLASS_ID_TEXT;
}
