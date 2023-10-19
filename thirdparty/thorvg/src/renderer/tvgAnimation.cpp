/*
 * Copyright (c) 2023 the ThorVG project. All rights reserved.

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
#include "tvgFrameModule.h"
#include "tvgPaint.h"
#include "tvgPicture.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct Animation::Impl
{
    Picture* picture = nullptr;

    Impl()
    {
        picture = Picture::gen().release();
        PP(picture)->ref();
    }

    ~Impl()
    {
        if (PP(picture)->unref() == 0) {
            delete(picture);
        }
    }
};

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Animation::~Animation()
{
    delete(pImpl);
}


Animation::Animation() : pImpl(new Impl)
{
}


Result Animation::frame(uint32_t no) noexcept
{
    auto loader = pImpl->picture->pImpl->loader.get();

    if (!loader) return Result::InsufficientCondition;
    if (!loader->animatable()) return Result::NonSupport;

    if (static_cast<FrameModule*>(loader)->frame(no)) return Result::Success;
    return Result::InsufficientCondition;
}


Picture* Animation::picture() const noexcept
{
    return pImpl->picture;
}


uint32_t Animation::curFrame() const noexcept
{
    auto loader = pImpl->picture->pImpl->loader.get();

    if (!loader) return 0;
    if (!loader->animatable()) return 0;

    return static_cast<FrameModule*>(loader)->curFrame();
}


uint32_t Animation::totalFrame() const noexcept
{
    auto loader = pImpl->picture->pImpl->loader.get();

    if (!loader) return 0;
    if (!loader->animatable()) return 0;

    return static_cast<FrameModule*>(loader)->totalFrame();
}


float Animation::duration() const noexcept
{
    auto loader = pImpl->picture->pImpl->loader.get();

    if (!loader) return 0;
    if (!loader->animatable()) return 0;

    return static_cast<FrameModule*>(loader)->duration();
}


unique_ptr<Animation> Animation::gen() noexcept
{
    return unique_ptr<Animation>(new Animation);
}
