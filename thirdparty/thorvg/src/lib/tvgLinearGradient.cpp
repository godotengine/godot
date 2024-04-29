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

#include <float.h>
#include <math.h>
#include "tvgFill.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct LinearGradient::Impl
{
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;

    Fill* duplicate()
    {
        auto ret = LinearGradient::gen();
        if (!ret) return nullptr;

        ret->pImpl->x1 = x1;
        ret->pImpl->y1 = y1;
        ret->pImpl->x2 = x2;
        ret->pImpl->y2 = y2;

        return ret.release();
    }
};

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

LinearGradient::LinearGradient():pImpl(new Impl())
{
    Fill::pImpl->id = TVG_CLASS_ID_LINEAR;
    Fill::pImpl->method(new FillDup<LinearGradient::Impl>(pImpl));
}


LinearGradient::~LinearGradient()
{
    delete(pImpl);
}


Result LinearGradient::linear(float x1, float y1, float x2, float y2) noexcept
{
    pImpl->x1 = x1;
    pImpl->y1 = y1;
    pImpl->x2 = x2;
    pImpl->y2 = y2;

    return Result::Success;
}


Result LinearGradient::linear(float* x1, float* y1, float* x2, float* y2) const noexcept
{
    if (x1) *x1 = pImpl->x1;
    if (x2) *x2 = pImpl->x2;
    if (y1) *y1 = pImpl->y1;
    if (y2) *y2 = pImpl->y2;

    return Result::Success;
}


unique_ptr<LinearGradient> LinearGradient::gen() noexcept
{
    return unique_ptr<LinearGradient>(new LinearGradient);
}


uint32_t LinearGradient::identifier() noexcept
{
    return TVG_CLASS_ID_LINEAR;
}
