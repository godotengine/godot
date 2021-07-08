//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// imageformats.cpp: Defines image format types with functions for mip generation
// and copying.

#include "image_util/imageformats.h"

#include "common/mathutil.h"

namespace angle
{

void L8::readColor(gl::ColorF *dst, const L8 *src)
{
    const float lum = gl::normalizedToFloat(src->L);
    dst->red        = lum;
    dst->green      = lum;
    dst->blue       = lum;
    dst->alpha      = 1.0f;
}

void L8::writeColor(L8 *dst, const gl::ColorF *src)
{
    dst->L = gl::floatToNormalized<uint8_t>(src->red);
}

void L8::average(L8 *dst, const L8 *src1, const L8 *src2)
{
    dst->L = gl::average(src1->L, src2->L);
}

void R8::readColor(gl::ColorUI *dst, const R8 *src)
{
    dst->red   = src->R;
    dst->green = 0;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R8::readColor(gl::ColorF *dst, const R8 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R8::writeColor(R8 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint8_t>(src->red);
}

void R8::writeColor(R8 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
}

void R8::average(R8 *dst, const R8 *src1, const R8 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
}

void A8::readColor(gl::ColorF *dst, const A8 *src)
{
    dst->red   = 0.0f;
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = gl::normalizedToFloat(src->A);
}

void A8::writeColor(A8 *dst, const gl::ColorF *src)
{
    dst->A = gl::floatToNormalized<uint8_t>(src->alpha);
}

void A8::average(A8 *dst, const A8 *src1, const A8 *src2)
{
    dst->A = gl::average(src1->A, src2->A);
}

void L8A8::readColor(gl::ColorF *dst, const L8A8 *src)
{
    const float lum = gl::normalizedToFloat(src->L);
    dst->red        = lum;
    dst->green      = lum;
    dst->blue       = lum;
    dst->alpha      = gl::normalizedToFloat(src->A);
}

void L8A8::writeColor(L8A8 *dst, const gl::ColorF *src)
{
    dst->L = gl::floatToNormalized<uint8_t>(src->red);
    dst->A = gl::floatToNormalized<uint8_t>(src->alpha);
}

void L8A8::average(L8A8 *dst, const L8A8 *src1, const L8A8 *src2)
{
    *(uint16_t *)dst = (((*(uint16_t *)src1 ^ *(uint16_t *)src2) & 0xFEFE) >> 1) +
                       (*(uint16_t *)src1 & *(uint16_t *)src2);
}

void A8L8::readColor(gl::ColorF *dst, const A8L8 *src)
{
    const float lum = gl::normalizedToFloat(src->L);
    dst->red        = lum;
    dst->green      = lum;
    dst->blue       = lum;
    dst->alpha      = gl::normalizedToFloat(src->A);
}

void A8L8::writeColor(A8L8 *dst, const gl::ColorF *src)
{
    dst->L = gl::floatToNormalized<uint8_t>(src->red);
    dst->A = gl::floatToNormalized<uint8_t>(src->alpha);
}

void A8L8::average(A8L8 *dst, const A8L8 *src1, const A8L8 *src2)
{
    *(uint16_t *)dst = (((*(uint16_t *)src1 ^ *(uint16_t *)src2) & 0xFEFE) >> 1) +
                       (*(uint16_t *)src1 & *(uint16_t *)src2);
}

void R8G8::readColor(gl::ColorUI *dst, const R8G8 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R8G8::readColor(gl::ColorF *dst, const R8G8 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R8G8::writeColor(R8G8 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint8_t>(src->red);
    dst->G = static_cast<uint8_t>(src->green);
}

void R8G8::writeColor(R8G8 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
    dst->G = gl::floatToNormalized<uint8_t>(src->green);
}

void R8G8::average(R8G8 *dst, const R8G8 *src1, const R8G8 *src2)
{
    *(uint16_t *)dst = (((*(uint16_t *)src1 ^ *(uint16_t *)src2) & 0xFEFE) >> 1) +
                       (*(uint16_t *)src1 & *(uint16_t *)src2);
}

void R8G8B8::readColor(gl::ColorUI *dst, const R8G8B8 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 1;
}

void R8G8B8::readColor(gl::ColorF *dst, const R8G8B8 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = 1.0f;
}

void R8G8B8::writeColor(R8G8B8 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint8_t>(src->red);
    dst->G = static_cast<uint8_t>(src->green);
    dst->B = static_cast<uint8_t>(src->blue);
}

void R8G8B8::writeColor(R8G8B8 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
    dst->G = gl::floatToNormalized<uint8_t>(src->green);
    dst->B = gl::floatToNormalized<uint8_t>(src->blue);
}

void R8G8B8::average(R8G8B8 *dst, const R8G8B8 *src1, const R8G8B8 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
}

void B8G8R8::readColor(gl::ColorUI *dst, const B8G8R8 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->G;
    dst->alpha = 1;
}

void B8G8R8::readColor(gl::ColorF *dst, const B8G8R8 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = 1.0f;
}

void B8G8R8::writeColor(B8G8R8 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint8_t>(src->red);
    dst->G = static_cast<uint8_t>(src->green);
    dst->B = static_cast<uint8_t>(src->blue);
}

void B8G8R8::writeColor(B8G8R8 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
    dst->G = gl::floatToNormalized<uint8_t>(src->green);
    dst->B = gl::floatToNormalized<uint8_t>(src->blue);
}

void B8G8R8::average(B8G8R8 *dst, const B8G8R8 *src1, const B8G8R8 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
}

void R5G6B5::readColor(gl::ColorF *dst, const R5G6B5 *src)
{
    dst->red   = gl::normalizedToFloat<5>(gl::getShiftedData<5, 11>(src->RGB));
    dst->green = gl::normalizedToFloat<6>(gl::getShiftedData<6, 5>(src->RGB));
    dst->blue  = gl::normalizedToFloat<5>(gl::getShiftedData<5, 0>(src->RGB));
    dst->alpha = 1.0f;
}

void R5G6B5::writeColor(R5G6B5 *dst, const gl::ColorF *src)
{
    dst->RGB = gl::shiftData<5, 11>(gl::floatToNormalized<5, uint16_t>(src->red)) |
               gl::shiftData<6, 5>(gl::floatToNormalized<6, uint16_t>(src->green)) |
               gl::shiftData<5, 0>(gl::floatToNormalized<5, uint16_t>(src->blue));
}

void R5G6B5::average(R5G6B5 *dst, const R5G6B5 *src1, const R5G6B5 *src2)
{
    dst->RGB = gl::shiftData<5, 11>(gl::average(gl::getShiftedData<5, 11>(src1->RGB),
                                                gl::getShiftedData<5, 11>(src2->RGB))) |
               gl::shiftData<6, 5>(gl::average(gl::getShiftedData<6, 5>(src1->RGB),
                                               gl::getShiftedData<6, 5>(src2->RGB))) |
               gl::shiftData<5, 0>(gl::average(gl::getShiftedData<5, 0>(src1->RGB),
                                               gl::getShiftedData<5, 0>(src2->RGB)));
}

void B5G6R5::readColor(gl::ColorF *dst, const B5G6R5 *src)
{
    dst->red   = gl::normalizedToFloat<5>(gl::getShiftedData<5, 11>(src->BGR));
    dst->green = gl::normalizedToFloat<6>(gl::getShiftedData<6, 5>(src->BGR));
    dst->blue  = gl::normalizedToFloat<5>(gl::getShiftedData<5, 0>(src->BGR));
    dst->alpha = 1.0f;
}

void B5G6R5::writeColor(B5G6R5 *dst, const gl::ColorF *src)
{
    dst->BGR = gl::shiftData<5, 0>(gl::floatToNormalized<5, unsigned short>(src->blue)) |
               gl::shiftData<6, 5>(gl::floatToNormalized<6, unsigned short>(src->green)) |
               gl::shiftData<5, 11>(gl::floatToNormalized<5, unsigned short>(src->red));
}

void B5G6R5::average(B5G6R5 *dst, const B5G6R5 *src1, const B5G6R5 *src2)
{
    dst->BGR = gl::shiftData<5, 11>(gl::average(gl::getShiftedData<5, 11>(src1->BGR),
                                                gl::getShiftedData<5, 11>(src2->BGR))) |
               gl::shiftData<6, 5>(gl::average(gl::getShiftedData<6, 5>(src1->BGR),
                                               gl::getShiftedData<6, 5>(src2->BGR))) |
               gl::shiftData<5, 0>(gl::average(gl::getShiftedData<5, 0>(src1->BGR),
                                               gl::getShiftedData<5, 0>(src2->BGR)));
}

void A8R8G8B8::readColor(gl::ColorUI *dst, const A8R8G8B8 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void A8R8G8B8::readColor(gl::ColorF *dst, const A8R8G8B8 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void A8R8G8B8::writeColor(A8R8G8B8 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint8_t>(src->red);
    dst->G = static_cast<uint8_t>(src->green);
    dst->B = static_cast<uint8_t>(src->blue);
    dst->A = static_cast<uint8_t>(src->alpha);
}

void A8R8G8B8::writeColor(A8R8G8B8 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
    dst->G = gl::floatToNormalized<uint8_t>(src->green);
    dst->B = gl::floatToNormalized<uint8_t>(src->blue);
    dst->A = gl::floatToNormalized<uint8_t>(src->alpha);
}

void A8R8G8B8::average(A8R8G8B8 *dst, const A8R8G8B8 *src1, const A8R8G8B8 *src2)
{
    *(uint32_t *)dst = (((*(uint32_t *)src1 ^ *(uint32_t *)src2) & 0xFEFEFEFE) >> 1) +
                       (*(uint32_t *)src1 & *(uint32_t *)src2);
}

void R8G8B8A8::readColor(gl::ColorUI *dst, const R8G8B8A8 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R8G8B8A8::readColor(gl::ColorF *dst, const R8G8B8A8 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void R8G8B8A8::writeColor(R8G8B8A8 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint8_t>(src->red);
    dst->G = static_cast<uint8_t>(src->green);
    dst->B = static_cast<uint8_t>(src->blue);
    dst->A = static_cast<uint8_t>(src->alpha);
}

void R8G8B8A8::writeColor(R8G8B8A8 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
    dst->G = gl::floatToNormalized<uint8_t>(src->green);
    dst->B = gl::floatToNormalized<uint8_t>(src->blue);
    dst->A = gl::floatToNormalized<uint8_t>(src->alpha);
}

void R8G8B8A8::average(R8G8B8A8 *dst, const R8G8B8A8 *src1, const R8G8B8A8 *src2)
{
    *(uint32_t *)dst = (((*(uint32_t *)src1 ^ *(uint32_t *)src2) & 0xFEFEFEFE) >> 1) +
                       (*(uint32_t *)src1 & *(uint32_t *)src2);
}

void R8G8B8A8SRGB::readColor(gl::ColorF *dst, const R8G8B8A8SRGB *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void R8G8B8A8SRGB::writeColor(R8G8B8A8SRGB *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
    dst->G = gl::floatToNormalized<uint8_t>(src->green);
    dst->B = gl::floatToNormalized<uint8_t>(src->blue);
    dst->A = gl::floatToNormalized<uint8_t>(src->alpha);
}

void R8G8B8A8SRGB::average(R8G8B8A8SRGB *dst, const R8G8B8A8SRGB *src1, const R8G8B8A8SRGB *src2)
{
    dst->R =
        gl::linearToSRGB(static_cast<uint8_t>((static_cast<uint16_t>(gl::sRGBToLinear(src1->R)) +
                                               static_cast<uint16_t>(gl::sRGBToLinear(src2->R))) >>
                                              1));
    dst->G =
        gl::linearToSRGB(static_cast<uint8_t>((static_cast<uint16_t>(gl::sRGBToLinear(src1->G)) +
                                               static_cast<uint16_t>(gl::sRGBToLinear(src2->G))) >>
                                              1));
    dst->B =
        gl::linearToSRGB(static_cast<uint8_t>((static_cast<uint16_t>(gl::sRGBToLinear(src1->B)) +
                                               static_cast<uint16_t>(gl::sRGBToLinear(src2->B))) >>
                                              1));
    dst->A = static_cast<uint8_t>(
        (static_cast<uint16_t>(src1->A) + static_cast<uint16_t>(src2->A)) >> 1);
}

void B8G8R8A8::readColor(gl::ColorUI *dst, const B8G8R8A8 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void B8G8R8A8::readColor(gl::ColorF *dst, const B8G8R8A8 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void B8G8R8A8::writeColor(B8G8R8A8 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint8_t>(src->red);
    dst->G = static_cast<uint8_t>(src->green);
    dst->B = static_cast<uint8_t>(src->blue);
    dst->A = static_cast<uint8_t>(src->alpha);
}

void B8G8R8A8::writeColor(B8G8R8A8 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
    dst->G = gl::floatToNormalized<uint8_t>(src->green);
    dst->B = gl::floatToNormalized<uint8_t>(src->blue);
    dst->A = gl::floatToNormalized<uint8_t>(src->alpha);
}

void B8G8R8A8::average(B8G8R8A8 *dst, const B8G8R8A8 *src1, const B8G8R8A8 *src2)
{
    *(uint32_t *)dst = (((*(uint32_t *)src1 ^ *(uint32_t *)src2) & 0xFEFEFEFE) >> 1) +
                       (*(uint32_t *)src1 & *(uint32_t *)src2);
}

void B8G8R8X8::readColor(gl::ColorUI *dst, const B8G8R8X8 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 1;
}

void B8G8R8X8::readColor(gl::ColorF *dst, const B8G8R8X8 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = 1.0f;
}

void B8G8R8X8::writeColor(B8G8R8X8 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint8_t>(src->red);
    dst->G = static_cast<uint8_t>(src->green);
    dst->B = static_cast<uint8_t>(src->blue);
    dst->X = 255;
}

void B8G8R8X8::writeColor(B8G8R8X8 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint8_t>(src->red);
    dst->G = gl::floatToNormalized<uint8_t>(src->green);
    dst->B = gl::floatToNormalized<uint8_t>(src->blue);
    dst->X = 255;
}

void B8G8R8X8::average(B8G8R8X8 *dst, const B8G8R8X8 *src1, const B8G8R8X8 *src2)
{
    *(uint32_t *)dst = (((*(uint32_t *)src1 ^ *(uint32_t *)src2) & 0xFEFEFEFE) >> 1) +
                       (*(uint32_t *)src1 & *(uint32_t *)src2);
    dst->X = 255;
}

void A1R5G5B5::readColor(gl::ColorF *dst, const A1R5G5B5 *src)
{
    dst->alpha = gl::normalizedToFloat<1>(gl::getShiftedData<1, 15>(src->ARGB));
    dst->red   = gl::normalizedToFloat<5>(gl::getShiftedData<5, 10>(src->ARGB));
    dst->green = gl::normalizedToFloat<5>(gl::getShiftedData<5, 5>(src->ARGB));
    dst->blue  = gl::normalizedToFloat<5>(gl::getShiftedData<5, 0>(src->ARGB));
}

void A1R5G5B5::writeColor(A1R5G5B5 *dst, const gl::ColorF *src)
{
    dst->ARGB = gl::shiftData<1, 15>(gl::floatToNormalized<1, uint16_t>(src->alpha)) |
                gl::shiftData<5, 10>(gl::floatToNormalized<5, uint16_t>(src->red)) |
                gl::shiftData<5, 5>(gl::floatToNormalized<5, uint16_t>(src->green)) |
                gl::shiftData<5, 0>(gl::floatToNormalized<5, uint16_t>(src->blue));
}

void A1R5G5B5::average(A1R5G5B5 *dst, const A1R5G5B5 *src1, const A1R5G5B5 *src2)
{
    dst->ARGB = gl::shiftData<1, 15>(gl::average(gl::getShiftedData<1, 15>(src1->ARGB),
                                                 gl::getShiftedData<1, 15>(src2->ARGB))) |
                gl::shiftData<5, 10>(gl::average(gl::getShiftedData<5, 10>(src1->ARGB),
                                                 gl::getShiftedData<5, 10>(src2->ARGB))) |
                gl::shiftData<5, 5>(gl::average(gl::getShiftedData<5, 5>(src1->ARGB),
                                                gl::getShiftedData<5, 5>(src2->ARGB))) |
                gl::shiftData<5, 0>(gl::average(gl::getShiftedData<5, 0>(src1->ARGB),
                                                gl::getShiftedData<5, 0>(src2->ARGB)));
}

void R5G5B5A1::readColor(gl::ColorF *dst, const R5G5B5A1 *src)
{
    dst->red   = gl::normalizedToFloat<5>(gl::getShiftedData<5, 11>(src->RGBA));
    dst->green = gl::normalizedToFloat<5>(gl::getShiftedData<5, 6>(src->RGBA));
    dst->blue  = gl::normalizedToFloat<5>(gl::getShiftedData<5, 1>(src->RGBA));
    dst->alpha = gl::normalizedToFloat<1>(gl::getShiftedData<1, 0>(src->RGBA));
}

void R5G5B5A1::writeColor(R5G5B5A1 *dst, const gl::ColorF *src)
{
    dst->RGBA = gl::shiftData<5, 11>(gl::floatToNormalized<5, uint16_t>(src->red)) |
                gl::shiftData<5, 6>(gl::floatToNormalized<5, uint16_t>(src->green)) |
                gl::shiftData<5, 1>(gl::floatToNormalized<5, uint16_t>(src->blue)) |
                gl::shiftData<1, 0>(gl::floatToNormalized<1, uint16_t>(src->alpha));
}

void R5G5B5A1::average(R5G5B5A1 *dst, const R5G5B5A1 *src1, const R5G5B5A1 *src2)
{
    dst->RGBA = gl::shiftData<5, 11>(gl::average(gl::getShiftedData<5, 11>(src1->RGBA),
                                                 gl::getShiftedData<5, 11>(src2->RGBA))) |
                gl::shiftData<5, 6>(gl::average(gl::getShiftedData<5, 6>(src1->RGBA),
                                                gl::getShiftedData<5, 6>(src2->RGBA))) |
                gl::shiftData<5, 1>(gl::average(gl::getShiftedData<5, 1>(src1->RGBA),
                                                gl::getShiftedData<5, 1>(src2->RGBA))) |
                gl::shiftData<1, 0>(gl::average(gl::getShiftedData<1, 0>(src1->RGBA),
                                                gl::getShiftedData<1, 0>(src2->RGBA)));
}

void R4G4B4A4::readColor(gl::ColorF *dst, const R4G4B4A4 *src)
{
    dst->red   = gl::normalizedToFloat<4>(gl::getShiftedData<4, 12>(src->RGBA));
    dst->green = gl::normalizedToFloat<4>(gl::getShiftedData<4, 8>(src->RGBA));
    dst->blue  = gl::normalizedToFloat<4>(gl::getShiftedData<4, 4>(src->RGBA));
    dst->alpha = gl::normalizedToFloat<4>(gl::getShiftedData<4, 0>(src->RGBA));
}

void R4G4B4A4::writeColor(R4G4B4A4 *dst, const gl::ColorF *src)
{
    dst->RGBA = gl::shiftData<4, 12>(gl::floatToNormalized<4, uint16_t>(src->red)) |
                gl::shiftData<4, 8>(gl::floatToNormalized<4, uint16_t>(src->green)) |
                gl::shiftData<4, 4>(gl::floatToNormalized<4, uint16_t>(src->blue)) |
                gl::shiftData<4, 0>(gl::floatToNormalized<4, uint16_t>(src->alpha));
}

void R4G4B4A4::average(R4G4B4A4 *dst, const R4G4B4A4 *src1, const R4G4B4A4 *src2)
{
    dst->RGBA = gl::shiftData<4, 12>(gl::average(gl::getShiftedData<4, 12>(src1->RGBA),
                                                 gl::getShiftedData<4, 12>(src2->RGBA))) |
                gl::shiftData<4, 8>(gl::average(gl::getShiftedData<4, 8>(src1->RGBA),
                                                gl::getShiftedData<4, 8>(src2->RGBA))) |
                gl::shiftData<4, 4>(gl::average(gl::getShiftedData<4, 4>(src1->RGBA),
                                                gl::getShiftedData<4, 4>(src2->RGBA))) |
                gl::shiftData<4, 0>(gl::average(gl::getShiftedData<4, 0>(src1->RGBA),
                                                gl::getShiftedData<4, 0>(src2->RGBA)));
}

void A4R4G4B4::readColor(gl::ColorF *dst, const A4R4G4B4 *src)
{
    dst->alpha = gl::normalizedToFloat<4>(gl::getShiftedData<4, 12>(src->ARGB));
    dst->red   = gl::normalizedToFloat<4>(gl::getShiftedData<4, 8>(src->ARGB));
    dst->green = gl::normalizedToFloat<4>(gl::getShiftedData<4, 4>(src->ARGB));
    dst->blue  = gl::normalizedToFloat<4>(gl::getShiftedData<4, 0>(src->ARGB));
}

void A4R4G4B4::writeColor(A4R4G4B4 *dst, const gl::ColorF *src)
{
    dst->ARGB = gl::shiftData<4, 12>(gl::floatToNormalized<4, uint16_t>(src->alpha)) |
                gl::shiftData<4, 8>(gl::floatToNormalized<4, uint16_t>(src->red)) |
                gl::shiftData<4, 4>(gl::floatToNormalized<4, uint16_t>(src->green)) |
                gl::shiftData<4, 0>(gl::floatToNormalized<4, uint16_t>(src->blue));
}

void A4R4G4B4::average(A4R4G4B4 *dst, const A4R4G4B4 *src1, const A4R4G4B4 *src2)
{
    dst->ARGB = gl::shiftData<4, 12>(gl::average(gl::getShiftedData<4, 12>(src1->ARGB),
                                                 gl::getShiftedData<4, 12>(src2->ARGB))) |
                gl::shiftData<4, 8>(gl::average(gl::getShiftedData<4, 8>(src1->ARGB),
                                                gl::getShiftedData<4, 8>(src2->ARGB))) |
                gl::shiftData<4, 4>(gl::average(gl::getShiftedData<4, 4>(src1->ARGB),
                                                gl::getShiftedData<4, 4>(src2->ARGB))) |
                gl::shiftData<4, 0>(gl::average(gl::getShiftedData<4, 0>(src1->ARGB),
                                                gl::getShiftedData<4, 0>(src2->ARGB)));
}

void R16::readColor(gl::ColorUI *dst, const R16 *src)
{
    dst->red   = src->R;
    dst->green = 0;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R16::readColor(gl::ColorF *dst, const R16 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R16::writeColor(R16 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint16_t>(src->red);
}

void R16::writeColor(R16 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint16_t>(src->red);
}

void R16::average(R16 *dst, const R16 *src1, const R16 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
}

void R16G16::readColor(gl::ColorUI *dst, const R16G16 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R16G16::readColor(gl::ColorF *dst, const R16G16 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R16G16::writeColor(R16G16 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint16_t>(src->red);
    dst->G = static_cast<uint16_t>(src->green);
}

void R16G16::writeColor(R16G16 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint16_t>(src->red);
    dst->G = gl::floatToNormalized<uint16_t>(src->green);
}

void R16G16::average(R16G16 *dst, const R16G16 *src1, const R16G16 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
}

void R16G16B16::readColor(gl::ColorUI *dst, const R16G16B16 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 1;
}

void R16G16B16::readColor(gl::ColorF *dst, const R16G16B16 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = 1.0f;
}

void R16G16B16::writeColor(R16G16B16 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint16_t>(src->red);
    dst->G = static_cast<uint16_t>(src->green);
    dst->B = static_cast<uint16_t>(src->blue);
}

void R16G16B16::writeColor(R16G16B16 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint16_t>(src->red);
    dst->G = gl::floatToNormalized<uint16_t>(src->green);
    dst->B = gl::floatToNormalized<uint16_t>(src->blue);
}

void R16G16B16::average(R16G16B16 *dst, const R16G16B16 *src1, const R16G16B16 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
}

void R16G16B16A16::readColor(gl::ColorUI *dst, const R16G16B16A16 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R16G16B16A16::readColor(gl::ColorF *dst, const R16G16B16A16 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void R16G16B16A16::writeColor(R16G16B16A16 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint16_t>(src->red);
    dst->G = static_cast<uint16_t>(src->green);
    dst->B = static_cast<uint16_t>(src->blue);
    dst->A = static_cast<uint16_t>(src->alpha);
}

void R16G16B16A16::writeColor(R16G16B16A16 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint16_t>(src->red);
    dst->G = gl::floatToNormalized<uint16_t>(src->green);
    dst->B = gl::floatToNormalized<uint16_t>(src->blue);
    dst->A = gl::floatToNormalized<uint16_t>(src->alpha);
}

void R16G16B16A16::average(R16G16B16A16 *dst, const R16G16B16A16 *src1, const R16G16B16A16 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
    dst->A = gl::average(src1->A, src2->A);
}

void R32::readColor(gl::ColorUI *dst, const R32 *src)
{
    dst->red   = src->R;
    dst->green = 0;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R32::readColor(gl::ColorF *dst, const R32 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R32::writeColor(R32 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint32_t>(src->red);
}

void R32::writeColor(R32 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint32_t>(src->red);
}

void R32::average(R32 *dst, const R32 *src1, const R32 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
}

void R32G32::readColor(gl::ColorUI *dst, const R32G32 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R32G32::readColor(gl::ColorF *dst, const R32G32 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R32G32::writeColor(R32G32 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint32_t>(src->red);
    dst->G = static_cast<uint32_t>(src->green);
}

void R32G32::writeColor(R32G32 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint32_t>(src->red);
    dst->G = gl::floatToNormalized<uint32_t>(src->green);
}

void R32G32::average(R32G32 *dst, const R32G32 *src1, const R32G32 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
}

void R32G32B32::readColor(gl::ColorUI *dst, const R32G32B32 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 1;
}

void R32G32B32::readColor(gl::ColorF *dst, const R32G32B32 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = 1.0f;
}

void R32G32B32::writeColor(R32G32B32 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint32_t>(src->red);
    dst->G = static_cast<uint32_t>(src->green);
    dst->B = static_cast<uint32_t>(src->blue);
}

void R32G32B32::writeColor(R32G32B32 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint32_t>(src->red);
    dst->G = gl::floatToNormalized<uint32_t>(src->green);
    dst->B = gl::floatToNormalized<uint32_t>(src->blue);
}

void R32G32B32::average(R32G32B32 *dst, const R32G32B32 *src1, const R32G32B32 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
}

void R32G32B32A32::readColor(gl::ColorUI *dst, const R32G32B32A32 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R32G32B32A32::readColor(gl::ColorF *dst, const R32G32B32A32 *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void R32G32B32A32::writeColor(R32G32B32A32 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint32_t>(src->red);
    dst->G = static_cast<uint32_t>(src->green);
    dst->B = static_cast<uint32_t>(src->blue);
    dst->A = static_cast<uint32_t>(src->alpha);
}

void R32G32B32A32::writeColor(R32G32B32A32 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<uint32_t>(src->red);
    dst->G = gl::floatToNormalized<uint32_t>(src->green);
    dst->B = gl::floatToNormalized<uint32_t>(src->blue);
    dst->A = gl::floatToNormalized<uint32_t>(src->alpha);
}

void R32G32B32A32::average(R32G32B32A32 *dst, const R32G32B32A32 *src1, const R32G32B32A32 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
    dst->A = gl::average(src1->A, src2->A);
}

void R8S::readColor(gl::ColorI *dst, const R8S *src)
{
    dst->red   = src->R;
    dst->green = 0;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R8S::readColor(gl::ColorF *dst, const R8S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R8S::writeColor(R8S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int8_t>(src->red);
}

void R8S::writeColor(R8S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int8_t>(src->red);
}

void R8S::average(R8S *dst, const R8S *src1, const R8S *src2)
{
    dst->R = static_cast<int8_t>(gl::average(src1->R, src2->R));
}

void R8G8S::readColor(gl::ColorI *dst, const R8G8S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R8G8S::readColor(gl::ColorF *dst, const R8G8S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R8G8S::writeColor(R8G8S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int8_t>(src->red);
    dst->G = static_cast<int8_t>(src->green);
}

void R8G8S::writeColor(R8G8S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int8_t>(src->red);
    dst->G = gl::floatToNormalized<int8_t>(src->green);
}

void R8G8S::average(R8G8S *dst, const R8G8S *src1, const R8G8S *src2)
{
    dst->R = static_cast<int8_t>(gl::average(src1->R, src2->R));
    dst->G = static_cast<int8_t>(gl::average(src1->G, src2->G));
}

void R8G8B8S::readColor(gl::ColorI *dst, const R8G8B8S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 1;
}

void R8G8B8S::readColor(gl::ColorF *dst, const R8G8B8S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = 1.0f;
}

void R8G8B8S::writeColor(R8G8B8S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int8_t>(src->red);
    dst->G = static_cast<int8_t>(src->green);
    dst->B = static_cast<int8_t>(src->blue);
}

void R8G8B8S::writeColor(R8G8B8S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int8_t>(src->red);
    dst->G = gl::floatToNormalized<int8_t>(src->green);
    dst->B = gl::floatToNormalized<int8_t>(src->blue);
}

void R8G8B8S::average(R8G8B8S *dst, const R8G8B8S *src1, const R8G8B8S *src2)
{
    dst->R = static_cast<int8_t>(gl::average(src1->R, src2->R));
    dst->G = static_cast<int8_t>(gl::average(src1->G, src2->G));
    dst->B = static_cast<int8_t>(gl::average(src1->B, src2->B));
}

void R8G8B8A8S::readColor(gl::ColorI *dst, const R8G8B8A8S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R8G8B8A8S::readColor(gl::ColorF *dst, const R8G8B8A8S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void R8G8B8A8S::writeColor(R8G8B8A8S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int8_t>(src->red);
    dst->G = static_cast<int8_t>(src->green);
    dst->B = static_cast<int8_t>(src->blue);
    dst->A = static_cast<int8_t>(src->alpha);
}

void R8G8B8A8S::writeColor(R8G8B8A8S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int8_t>(src->red);
    dst->G = gl::floatToNormalized<int8_t>(src->green);
    dst->B = gl::floatToNormalized<int8_t>(src->blue);
    dst->A = gl::floatToNormalized<int8_t>(src->alpha);
}

void R8G8B8A8S::average(R8G8B8A8S *dst, const R8G8B8A8S *src1, const R8G8B8A8S *src2)
{
    dst->R = static_cast<int8_t>(gl::average(src1->R, src2->R));
    dst->G = static_cast<int8_t>(gl::average(src1->G, src2->G));
    dst->B = static_cast<int8_t>(gl::average(src1->B, src2->B));
    dst->A = static_cast<int8_t>(gl::average(src1->A, src2->A));
}

void R16S::readColor(gl::ColorI *dst, const R16S *src)
{
    dst->red   = src->R;
    dst->green = 0;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R16S::readColor(gl::ColorF *dst, const R16S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R16S::writeColor(R16S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int16_t>(src->red);
}

void R16S::writeColor(R16S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int16_t>(src->red);
}

void R16S::average(R16S *dst, const R16S *src1, const R16S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
}

void R16G16S::readColor(gl::ColorI *dst, const R16G16S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R16G16S::readColor(gl::ColorF *dst, const R16G16S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R16G16S::writeColor(R16G16S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int16_t>(src->red);
    dst->G = static_cast<int16_t>(src->green);
}

void R16G16S::writeColor(R16G16S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int16_t>(src->red);
    dst->G = gl::floatToNormalized<int16_t>(src->green);
}

void R16G16S::average(R16G16S *dst, const R16G16S *src1, const R16G16S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
}

void R16G16B16S::readColor(gl::ColorI *dst, const R16G16B16S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 1;
}

void R16G16B16S::readColor(gl::ColorF *dst, const R16G16B16S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = 1.0f;
}

void R16G16B16S::writeColor(R16G16B16S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int16_t>(src->red);
    dst->G = static_cast<int16_t>(src->green);
    dst->B = static_cast<int16_t>(src->blue);
}

void R16G16B16S::writeColor(R16G16B16S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int16_t>(src->red);
    dst->G = gl::floatToNormalized<int16_t>(src->green);
    dst->B = gl::floatToNormalized<int16_t>(src->blue);
}

void R16G16B16S::average(R16G16B16S *dst, const R16G16B16S *src1, const R16G16B16S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
}

void R16G16B16A16S::readColor(gl::ColorI *dst, const R16G16B16A16S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R16G16B16A16S::readColor(gl::ColorF *dst, const R16G16B16A16S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void R16G16B16A16S::writeColor(R16G16B16A16S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int16_t>(src->red);
    dst->G = static_cast<int16_t>(src->green);
    dst->B = static_cast<int16_t>(src->blue);
    dst->A = static_cast<int16_t>(src->alpha);
}

void R16G16B16A16S::writeColor(R16G16B16A16S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int16_t>(src->red);
    dst->G = gl::floatToNormalized<int16_t>(src->green);
    dst->B = gl::floatToNormalized<int16_t>(src->blue);
    dst->A = gl::floatToNormalized<int16_t>(src->alpha);
}

void R16G16B16A16S::average(R16G16B16A16S *dst,
                            const R16G16B16A16S *src1,
                            const R16G16B16A16S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
    dst->A = gl::average(src1->A, src2->A);
}

void R32S::readColor(gl::ColorI *dst, const R32S *src)
{
    dst->red   = src->R;
    dst->green = 0;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R32S::readColor(gl::ColorF *dst, const R32S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R32S::writeColor(R32S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int32_t>(src->red);
}

void R32S::writeColor(R32S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int32_t>(src->red);
}

void R32S::average(R32S *dst, const R32S *src1, const R32S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
}

void R32G32S::readColor(gl::ColorI *dst, const R32G32S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = 0;
    dst->alpha = 1;
}

void R32G32S::readColor(gl::ColorF *dst, const R32G32S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R32G32S::writeColor(R32G32S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int32_t>(src->red);
    dst->G = static_cast<int32_t>(src->green);
}

void R32G32S::writeColor(R32G32S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int32_t>(src->red);
    dst->G = gl::floatToNormalized<int32_t>(src->green);
}

void R32G32S::average(R32G32S *dst, const R32G32S *src1, const R32G32S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
}

void R32G32B32S::readColor(gl::ColorI *dst, const R32G32B32S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 1;
}

void R32G32B32S::readColor(gl::ColorF *dst, const R32G32B32S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = 1.0f;
}

void R32G32B32S::writeColor(R32G32B32S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int32_t>(src->red);
    dst->G = static_cast<int32_t>(src->green);
    dst->B = static_cast<int32_t>(src->blue);
}

void R32G32B32S::writeColor(R32G32B32S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int32_t>(src->red);
    dst->G = gl::floatToNormalized<int32_t>(src->green);
    dst->B = gl::floatToNormalized<int32_t>(src->blue);
}

void R32G32B32S::average(R32G32B32S *dst, const R32G32B32S *src1, const R32G32B32S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
}

void R32G32B32A32S::readColor(gl::ColorI *dst, const R32G32B32A32S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R32G32B32A32S::readColor(gl::ColorF *dst, const R32G32B32A32S *src)
{
    dst->red   = gl::normalizedToFloat(src->R);
    dst->green = gl::normalizedToFloat(src->G);
    dst->blue  = gl::normalizedToFloat(src->B);
    dst->alpha = gl::normalizedToFloat(src->A);
}

void R32G32B32A32S::writeColor(R32G32B32A32S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int32_t>(src->red);
    dst->G = static_cast<int32_t>(src->green);
    dst->B = static_cast<int32_t>(src->blue);
    dst->A = static_cast<int32_t>(src->alpha);
}

void R32G32B32A32S::writeColor(R32G32B32A32S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<int32_t>(src->red);
    dst->G = gl::floatToNormalized<int32_t>(src->green);
    dst->B = gl::floatToNormalized<int32_t>(src->blue);
    dst->A = gl::floatToNormalized<int32_t>(src->alpha);
}

void R32G32B32A32S::average(R32G32B32A32S *dst,
                            const R32G32B32A32S *src1,
                            const R32G32B32A32S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
    dst->A = gl::average(src1->A, src2->A);
}

void A16B16G16R16F::readColor(gl::ColorF *dst, const A16B16G16R16F *src)
{
    dst->red   = gl::float16ToFloat32(src->R);
    dst->green = gl::float16ToFloat32(src->G);
    dst->blue  = gl::float16ToFloat32(src->B);
    dst->alpha = gl::float16ToFloat32(src->A);
}

void A16B16G16R16F::writeColor(A16B16G16R16F *dst, const gl::ColorF *src)
{
    dst->R = gl::float32ToFloat16(src->red);
    dst->G = gl::float32ToFloat16(src->green);
    dst->B = gl::float32ToFloat16(src->blue);
    dst->A = gl::float32ToFloat16(src->alpha);
}

void A16B16G16R16F::average(A16B16G16R16F *dst,
                            const A16B16G16R16F *src1,
                            const A16B16G16R16F *src2)
{
    dst->R = gl::averageHalfFloat(src1->R, src2->R);
    dst->G = gl::averageHalfFloat(src1->G, src2->G);
    dst->B = gl::averageHalfFloat(src1->B, src2->B);
    dst->A = gl::averageHalfFloat(src1->A, src2->A);
}

void R16G16B16A16F::readColor(gl::ColorF *dst, const R16G16B16A16F *src)
{
    dst->red   = gl::float16ToFloat32(src->R);
    dst->green = gl::float16ToFloat32(src->G);
    dst->blue  = gl::float16ToFloat32(src->B);
    dst->alpha = gl::float16ToFloat32(src->A);
}

void R16G16B16A16F::writeColor(R16G16B16A16F *dst, const gl::ColorF *src)
{
    dst->R = gl::float32ToFloat16(src->red);
    dst->G = gl::float32ToFloat16(src->green);
    dst->B = gl::float32ToFloat16(src->blue);
    dst->A = gl::float32ToFloat16(src->alpha);
}

void R16G16B16A16F::average(R16G16B16A16F *dst,
                            const R16G16B16A16F *src1,
                            const R16G16B16A16F *src2)
{
    dst->R = gl::averageHalfFloat(src1->R, src2->R);
    dst->G = gl::averageHalfFloat(src1->G, src2->G);
    dst->B = gl::averageHalfFloat(src1->B, src2->B);
    dst->A = gl::averageHalfFloat(src1->A, src2->A);
}

void R16F::readColor(gl::ColorF *dst, const R16F *src)
{
    dst->red   = gl::float16ToFloat32(src->R);
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R16F::writeColor(R16F *dst, const gl::ColorF *src)
{
    dst->R = gl::float32ToFloat16(src->red);
}

void R16F::average(R16F *dst, const R16F *src1, const R16F *src2)
{
    dst->R = gl::averageHalfFloat(src1->R, src2->R);
}

void A16F::readColor(gl::ColorF *dst, const A16F *src)
{
    dst->red   = 0.0f;
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = gl::float16ToFloat32(src->A);
}

void A16F::writeColor(A16F *dst, const gl::ColorF *src)
{
    dst->A = gl::float32ToFloat16(src->alpha);
}

void A16F::average(A16F *dst, const A16F *src1, const A16F *src2)
{
    dst->A = gl::averageHalfFloat(src1->A, src2->A);
}

void L16F::readColor(gl::ColorF *dst, const L16F *src)
{
    float lum  = gl::float16ToFloat32(src->L);
    dst->red   = lum;
    dst->green = lum;
    dst->blue  = lum;
    dst->alpha = 1.0f;
}

void L16F::writeColor(L16F *dst, const gl::ColorF *src)
{
    dst->L = gl::float32ToFloat16(src->red);
}

void L16F::average(L16F *dst, const L16F *src1, const L16F *src2)
{
    dst->L = gl::averageHalfFloat(src1->L, src2->L);
}

void L16A16F::readColor(gl::ColorF *dst, const L16A16F *src)
{
    float lum  = gl::float16ToFloat32(src->L);
    dst->red   = lum;
    dst->green = lum;
    dst->blue  = lum;
    dst->alpha = gl::float16ToFloat32(src->A);
}

void L16A16F::writeColor(L16A16F *dst, const gl::ColorF *src)
{
    dst->L = gl::float32ToFloat16(src->red);
    dst->A = gl::float32ToFloat16(src->alpha);
}

void L16A16F::average(L16A16F *dst, const L16A16F *src1, const L16A16F *src2)
{
    dst->L = gl::averageHalfFloat(src1->L, src2->L);
    dst->A = gl::averageHalfFloat(src1->A, src2->A);
}

void R16G16F::readColor(gl::ColorF *dst, const R16G16F *src)
{
    dst->red   = gl::float16ToFloat32(src->R);
    dst->green = gl::float16ToFloat32(src->G);
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R16G16F::writeColor(R16G16F *dst, const gl::ColorF *src)
{
    dst->R = gl::float32ToFloat16(src->red);
    dst->G = gl::float32ToFloat16(src->green);
}

void R16G16F::average(R16G16F *dst, const R16G16F *src1, const R16G16F *src2)
{
    dst->R = gl::averageHalfFloat(src1->R, src2->R);
    dst->G = gl::averageHalfFloat(src1->G, src2->G);
}

void R16G16B16F::readColor(gl::ColorF *dst, const R16G16B16F *src)
{
    dst->red   = gl::float16ToFloat32(src->R);
    dst->green = gl::float16ToFloat32(src->G);
    dst->blue  = gl::float16ToFloat32(src->B);
    dst->alpha = 1.0f;
}

void R16G16B16F::writeColor(R16G16B16F *dst, const gl::ColorF *src)
{
    dst->R = gl::float32ToFloat16(src->red);
    dst->G = gl::float32ToFloat16(src->green);
    dst->B = gl::float32ToFloat16(src->blue);
}

void R16G16B16F::average(R16G16B16F *dst, const R16G16B16F *src1, const R16G16B16F *src2)
{
    dst->R = gl::averageHalfFloat(src1->R, src2->R);
    dst->G = gl::averageHalfFloat(src1->G, src2->G);
    dst->B = gl::averageHalfFloat(src1->B, src2->B);
}

void A32B32G32R32F::readColor(gl::ColorF *dst, const A32B32G32R32F *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void A32B32G32R32F::writeColor(A32B32G32R32F *dst, const gl::ColorF *src)
{
    dst->R = src->red;
    dst->G = src->green;
    dst->B = src->blue;
    dst->A = src->alpha;
}

void A32B32G32R32F::average(A32B32G32R32F *dst,
                            const A32B32G32R32F *src1,
                            const A32B32G32R32F *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
    dst->A = gl::average(src1->A, src2->A);
}

void R32G32B32A32F::readColor(gl::ColorF *dst, const R32G32B32A32F *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R32G32B32A32F::writeColor(R32G32B32A32F *dst, const gl::ColorF *src)
{
    dst->R = src->red;
    dst->G = src->green;
    dst->B = src->blue;
    dst->A = src->alpha;
}

void R32G32B32A32F::average(R32G32B32A32F *dst,
                            const R32G32B32A32F *src1,
                            const R32G32B32A32F *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
    dst->A = gl::average(src1->A, src2->A);
}

void R32F::readColor(gl::ColorF *dst, const R32F *src)
{
    dst->red   = src->R;
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R32F::writeColor(R32F *dst, const gl::ColorF *src)
{
    dst->R = src->red;
}

void R32F::average(R32F *dst, const R32F *src1, const R32F *src2)
{
    dst->R = gl::average(src1->R, src2->R);
}

void A32F::readColor(gl::ColorF *dst, const A32F *src)
{
    dst->red   = 0.0f;
    dst->green = 0.0f;
    dst->blue  = 0.0f;
    dst->alpha = src->A;
}

void A32F::writeColor(A32F *dst, const gl::ColorF *src)
{
    dst->A = src->alpha;
}

void A32F::average(A32F *dst, const A32F *src1, const A32F *src2)
{
    dst->A = gl::average(src1->A, src2->A);
}

void L32F::readColor(gl::ColorF *dst, const L32F *src)
{
    dst->red   = src->L;
    dst->green = src->L;
    dst->blue  = src->L;
    dst->alpha = 1.0f;
}

void L32F::writeColor(L32F *dst, const gl::ColorF *src)
{
    dst->L = src->red;
}

void L32F::average(L32F *dst, const L32F *src1, const L32F *src2)
{
    dst->L = gl::average(src1->L, src2->L);
}

void L32A32F::readColor(gl::ColorF *dst, const L32A32F *src)
{
    dst->red   = src->L;
    dst->green = src->L;
    dst->blue  = src->L;
    dst->alpha = src->A;
}

void L32A32F::writeColor(L32A32F *dst, const gl::ColorF *src)
{
    dst->L = src->red;
    dst->A = src->alpha;
}

void L32A32F::average(L32A32F *dst, const L32A32F *src1, const L32A32F *src2)
{
    dst->L = gl::average(src1->L, src2->L);
    dst->A = gl::average(src1->A, src2->A);
}

void R32G32F::readColor(gl::ColorF *dst, const R32G32F *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = 0.0f;
    dst->alpha = 1.0f;
}

void R32G32F::writeColor(R32G32F *dst, const gl::ColorF *src)
{
    dst->R = src->red;
    dst->G = src->green;
}

void R32G32F::average(R32G32F *dst, const R32G32F *src1, const R32G32F *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
}

void R32G32B32F::readColor(gl::ColorF *dst, const R32G32B32F *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 1.0f;
}

void R32G32B32F::writeColor(R32G32B32F *dst, const gl::ColorF *src)
{
    dst->R = src->red;
    dst->G = src->green;
    dst->B = src->blue;
}

void R32G32B32F::average(R32G32B32F *dst, const R32G32B32F *src1, const R32G32B32F *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
}

void R10G10B10A2::readColor(gl::ColorUI *dst, const R10G10B10A2 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R10G10B10A2::readColor(gl::ColorF *dst, const R10G10B10A2 *src)
{
    dst->red   = gl::normalizedToFloat<10>(src->R);
    dst->green = gl::normalizedToFloat<10>(src->G);
    dst->blue  = gl::normalizedToFloat<10>(src->B);
    dst->alpha = gl::normalizedToFloat<2>(src->A);
}

void R10G10B10A2::writeColor(R10G10B10A2 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint32_t>(src->red);
    dst->G = static_cast<uint32_t>(src->green);
    dst->B = static_cast<uint32_t>(src->blue);
    dst->A = static_cast<uint32_t>(src->alpha);
}

void R10G10B10A2::writeColor(R10G10B10A2 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<10, uint32_t>(src->red);
    dst->G = gl::floatToNormalized<10, uint32_t>(src->green);
    dst->B = gl::floatToNormalized<10, uint32_t>(src->blue);
    dst->A = gl::floatToNormalized<2, uint32_t>(src->alpha);
}

void R10G10B10A2::average(R10G10B10A2 *dst, const R10G10B10A2 *src1, const R10G10B10A2 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
    dst->A = gl::average(src1->A, src2->A);
}

void R10G10B10A2S::readColor(gl::ColorI *dst, const R10G10B10A2S *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = src->A;
}

void R10G10B10A2S::readColor(gl::ColorF *dst, const R10G10B10A2S *src)
{
    dst->red   = gl::normalizedToFloat<10>(src->R);
    dst->green = gl::normalizedToFloat<10>(src->G);
    dst->blue  = gl::normalizedToFloat<10>(src->B);
    dst->alpha = gl::normalizedToFloat<2>(src->A);
}

void R10G10B10A2S::writeColor(R10G10B10A2S *dst, const gl::ColorI *src)
{
    dst->R = static_cast<int32_t>(src->red);
    dst->G = static_cast<int32_t>(src->green);
    dst->B = static_cast<int32_t>(src->blue);
    dst->A = static_cast<int32_t>(src->alpha);
}

void R10G10B10A2S::writeColor(R10G10B10A2S *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<10, int32_t>(src->red);
    dst->G = gl::floatToNormalized<10, int32_t>(src->green);
    dst->B = gl::floatToNormalized<10, int32_t>(src->blue);
    dst->A = gl::floatToNormalized<2, int32_t>(src->alpha);
}

void R10G10B10A2S::average(R10G10B10A2S *dst, const R10G10B10A2S *src1, const R10G10B10A2S *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
    dst->A = gl::average(src1->A, src2->A);
}

void R10G10B10X2::readColor(gl::ColorUI *dst, const R10G10B10X2 *src)
{
    dst->red   = src->R;
    dst->green = src->G;
    dst->blue  = src->B;
    dst->alpha = 0x3;
}

void R10G10B10X2::readColor(gl::ColorF *dst, const R10G10B10X2 *src)
{
    dst->red   = gl::normalizedToFloat<10>(src->R);
    dst->green = gl::normalizedToFloat<10>(src->G);
    dst->blue  = gl::normalizedToFloat<10>(src->B);
    dst->alpha = 1.0f;
}

void R10G10B10X2::writeColor(R10G10B10X2 *dst, const gl::ColorUI *src)
{
    dst->R = static_cast<uint32_t>(src->red);
    dst->G = static_cast<uint32_t>(src->green);
    dst->B = static_cast<uint32_t>(src->blue);
}

void R10G10B10X2::writeColor(R10G10B10X2 *dst, const gl::ColorF *src)
{
    dst->R = gl::floatToNormalized<10, uint32_t>(src->red);
    dst->G = gl::floatToNormalized<10, uint32_t>(src->green);
    dst->B = gl::floatToNormalized<10, uint32_t>(src->blue);
}

void R10G10B10X2::average(R10G10B10X2 *dst, const R10G10B10X2 *src1, const R10G10B10X2 *src2)
{
    dst->R = gl::average(src1->R, src2->R);
    dst->G = gl::average(src1->G, src2->G);
    dst->B = gl::average(src1->B, src2->B);
}

void R9G9B9E5::readColor(gl::ColorF *dst, const R9G9B9E5 *src)
{
    gl::convert999E5toRGBFloats(gl::bitCast<uint32_t>(*src), &dst->red, &dst->green, &dst->blue);
    dst->alpha = 1.0f;
}

void R9G9B9E5::writeColor(R9G9B9E5 *dst, const gl::ColorF *src)
{
    *reinterpret_cast<uint32_t *>(dst) =
        gl::convertRGBFloatsTo999E5(src->red, src->green, src->blue);
}

void R9G9B9E5::average(R9G9B9E5 *dst, const R9G9B9E5 *src1, const R9G9B9E5 *src2)
{
    float r1, g1, b1;
    gl::convert999E5toRGBFloats(*reinterpret_cast<const uint32_t *>(src1), &r1, &g1, &b1);

    float r2, g2, b2;
    gl::convert999E5toRGBFloats(*reinterpret_cast<const uint32_t *>(src2), &r2, &g2, &b2);

    *reinterpret_cast<uint32_t *>(dst) =
        gl::convertRGBFloatsTo999E5(gl::average(r1, r2), gl::average(g1, g2), gl::average(b1, b2));
}

void R11G11B10F::readColor(gl::ColorF *dst, const R11G11B10F *src)
{
    dst->red   = gl::float11ToFloat32(src->R);
    dst->green = gl::float11ToFloat32(src->G);
    dst->blue  = gl::float10ToFloat32(src->B);
    dst->alpha = 1.0f;
}

void R11G11B10F::writeColor(R11G11B10F *dst, const gl::ColorF *src)
{
    dst->R = gl::float32ToFloat11(src->red);
    dst->G = gl::float32ToFloat11(src->green);
    dst->B = gl::float32ToFloat10(src->blue);
}

void R11G11B10F::average(R11G11B10F *dst, const R11G11B10F *src1, const R11G11B10F *src2)
{
    dst->R = gl::averageFloat11(src1->R, src2->R);
    dst->G = gl::averageFloat11(src1->G, src2->G);
    dst->B = gl::averageFloat10(src1->B, src2->B);
}

void D24S8::ReadDepthStencil(DepthStencil *dst, const D24S8 *src)
{
    dst->depth   = gl::normalizedToFloat<24>(src->D);
    dst->stencil = src->S;
}

void D24S8::WriteDepthStencil(D24S8 *dst, const DepthStencil *src)
{
    dst->D = gl::floatToNormalized<24, uint32_t>(static_cast<float>(src->depth));
    dst->S = src->stencil & 0xFF;
}

void S8::ReadDepthStencil(DepthStencil *dst, const S8 *src)
{
    dst->depth   = 0;
    dst->stencil = src->S;
}

void S8::WriteDepthStencil(S8 *dst, const DepthStencil *src)
{
    dst->S = src->stencil & 0xFF;
}

void D16::ReadDepthStencil(DepthStencil *dst, const D16 *src)
{
    dst->depth   = gl::normalizedToFloat(src->D);
    dst->stencil = 0;
}

void D16::WriteDepthStencil(D16 *dst, const DepthStencil *src)
{
    dst->D = gl::floatToNormalized<uint16_t>(static_cast<float>(src->depth));
}

void D24X8::ReadDepthStencil(DepthStencil *dst, const D24X8 *src)
{
    dst->depth = gl::normalizedToFloat<24>(gl::getShiftedData<24, 8>(src->D));
}

void D24X8::WriteDepthStencil(D24X8 *dst, const DepthStencil *src)
{
    dst->D =
        gl::shiftData<24, 8>(gl::floatToNormalized<24, uint32_t>(static_cast<float>(src->depth)));
}

void D32F::ReadDepthStencil(DepthStencil *dst, const D32F *src)
{
    dst->depth = src->D;
}

void D32F::WriteDepthStencil(D32F *dst, const DepthStencil *src)
{
    dst->D = static_cast<float>(src->depth);
}

void D32::ReadDepthStencil(DepthStencil *dst, const D32 *src)
{
    dst->depth   = gl::normalizedToFloat(src->D);
    dst->stencil = 0;
}

void D32::WriteDepthStencil(D32 *dst, const DepthStencil *src)
{
    dst->D = gl::floatToNormalized<uint32_t>(static_cast<float>(src->depth));
}

void D32FS8X24::ReadDepthStencil(DepthStencil *dst, const D32FS8X24 *src)
{
    dst->depth   = src->D;
    dst->stencil = src->S;
}

void D32FS8X24::WriteDepthStencil(D32FS8X24 *dst, const DepthStencil *src)
{
    dst->D = static_cast<float>(src->depth);
    dst->S = src->stencil & 0xFF;
}
}  // namespace angle
