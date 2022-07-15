/*
 * Copyright (c) 2020 - 2022 Samsung Electronics Co., Ltd. All rights reserved.

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
#include <float.h>
#include "tvgFill.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct RadialGradient::Impl
{
    float cx = 0;
    float cy = 0;
    float radius = 0;

    Fill* duplicate()
    {
        auto ret = RadialGradient::gen();
        if (!ret) return nullptr;

        ret->pImpl->cx = cx;
        ret->pImpl->cy = cy;
        ret->pImpl->radius = radius;

        return ret.release();
    }
};


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

RadialGradient::RadialGradient():pImpl(new Impl())
{
    Fill::pImpl->id = TVG_CLASS_ID_RADIAL;
    Fill::pImpl->method(new FillDup<RadialGradient::Impl>(pImpl));
}


RadialGradient::~RadialGradient()
{
    delete(pImpl);
}


Result RadialGradient::radial(float cx, float cy, float radius) noexcept
{
    if (radius < 0) return Result::InvalidArguments;

    pImpl->cx = cx;
    pImpl->cy = cy;
    pImpl->radius = radius;

    return Result::Success;
}


Result RadialGradient::radial(float* cx, float* cy, float* radius) const noexcept
{
    if (cx) *cx = pImpl->cx;
    if (cy) *cy = pImpl->cy;
    if (radius) *radius = pImpl->radius;

    return Result::Success;
}


unique_ptr<RadialGradient> RadialGradient::gen() noexcept
{
    return unique_ptr<RadialGradient>(new RadialGradient);
}


uint32_t RadialGradient::identifier() noexcept
{
    return TVG_CLASS_ID_RADIAL;
}
