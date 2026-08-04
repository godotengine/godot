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

#include "tvgFill.h"


/************************************************************************/
/* Fill Class Implementation                                            */
/************************************************************************/

Fill::Fill() = default;
Fill::~Fill() = default;

Result Fill::colorStops(const ColorStop* colorStops, uint32_t cnt) noexcept
{
    return pImpl->update(colorStops, cnt);
}


uint32_t Fill::colorStops(const ColorStop** colorStops) const noexcept
{
    if (colorStops) *colorStops = pImpl->colorStops;
    return pImpl->cnt;
}


Result Fill::spread(FillSpread s) noexcept
{
    pImpl->spread = s;
    return Result::Success;
}


FillSpread Fill::spread() const noexcept
{
    return pImpl->spread;
}


Result Fill::transform(const Matrix& m) noexcept
{
    pImpl->transform = m;
    return Result::Success;
}


Matrix& Fill::transform() const noexcept
{
    return pImpl->transform;
}


Fill* Fill::duplicate() const noexcept
{
    if (type() == Type::LinearGradient) return CONST_LINEAR(this)->duplicate();
    else if (type() == Type::RadialGradient) return CONST_RADIAL(this)->duplicate();
    return nullptr;
}


/************************************************************************/
/* RadialGradient Class Implementation                                  */
/************************************************************************/

RadialGradient::RadialGradient() = default;


Result RadialGradient::radial(float cx, float cy, float r, float fx, float fy, float fr) noexcept
{
    return RADIAL(this)->radial(cx, cy, r, fx, fy, fr);
}


Result RadialGradient::radial(float* cx, float* cy, float* r, float* fx, float* fy, float* fr) const noexcept
{
    return CONST_RADIAL(this)->radial(cx, cy, r, fx, fy, fr);
}


RadialGradient* RadialGradient::gen() noexcept
{
    return new RadialGradientImpl;
}


Type RadialGradient::type() const noexcept
{
    return Type::RadialGradient;
}


/************************************************************************/
/* LinearGradient Class Implementation                                  */
/************************************************************************/

LinearGradient::LinearGradient() = default;


Result LinearGradient::linear(float x1, float y1, float x2, float y2) noexcept
{
    return LINEAR(this)->linear(x1, y1, x2, y2);
}


Result LinearGradient::linear(float* x1, float* y1, float* x2, float* y2) const noexcept
{
    return CONST_LINEAR(this)->linear(x1, y1, x2, y2);
}


LinearGradient* LinearGradient::gen() noexcept
{
    return new LinearGradientImpl;
}


Type LinearGradient::type() const noexcept
{
    return Type::LinearGradient;
}
