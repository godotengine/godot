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

#include "tvgFrameModule.h"
#include "tvgAnimation.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

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


Result Animation::frame(float no) noexcept
{
    auto loader = pImpl->picture->pImpl->loader;

    if (!loader) return Result::InsufficientCondition;
    if (!loader->animatable()) return Result::NonSupport;

    if (static_cast<FrameModule*>(loader)->frame(no)) return Result::Success;
    return Result::InsufficientCondition;
}


Picture* Animation::picture() const noexcept
{
    return pImpl->picture;
}


float Animation::curFrame() const noexcept
{
    auto loader = pImpl->picture->pImpl->loader;

    if (!loader) return 0;
    if (!loader->animatable()) return 0;

    return static_cast<FrameModule*>(loader)->curFrame();
}


float Animation::totalFrame() const noexcept
{
    auto loader = pImpl->picture->pImpl->loader;

    if (!loader) return 0;
    if (!loader->animatable()) return 0;

    return static_cast<FrameModule*>(loader)->totalFrame();
}


float Animation::duration() const noexcept
{
    auto loader = pImpl->picture->pImpl->loader;

    if (!loader) return 0;
    if (!loader->animatable()) return 0;

    return static_cast<FrameModule*>(loader)->duration();
}


Result Animation::segment(float begin, float end) noexcept
{
    if (begin < 0.0 || end > 1.0 || begin >= end) return Result::InvalidArguments;

    auto loader = pImpl->picture->pImpl->loader;
    if (!loader) return Result::InsufficientCondition;
    if (!loader->animatable()) return Result::NonSupport;

    static_cast<FrameModule*>(loader)->segment(begin, end);

    return Result::Success;
}


Result Animation::segment(float *begin, float *end) noexcept
{
    auto loader = pImpl->picture->pImpl->loader;
    if (!loader) return Result::InsufficientCondition;
    if (!loader->animatable()) return Result::NonSupport;
    if (!begin && !end) return Result::InvalidArguments;

    static_cast<FrameModule*>(loader)->segment(begin, end);

    return Result::Success;
}


unique_ptr<Animation> Animation::gen() noexcept
{
    return unique_ptr<Animation>(new Animation);
}
