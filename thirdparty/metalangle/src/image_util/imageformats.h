//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// imageformats.h: Defines image format types with functions for mip generation
// and copying.

#ifndef IMAGEUTIL_IMAGEFORMATS_H_
#define IMAGEUTIL_IMAGEFORMATS_H_

#include "common/Color.h"

#include <cstdint>

namespace angle
{

// Several structures share functionality for reading, writing or mipmapping but the layout
// must match the texture format which the structure represents. If collapsing or typedefing
// structs in this header, make sure the functionality and memory layout is exactly the same.

struct L8
{
    uint8_t L;

    static void readColor(gl::ColorF *dst, const L8 *src);
    static void writeColor(L8 *dst, const gl::ColorF *src);
    static void average(L8 *dst, const L8 *src1, const L8 *src2);
};

struct R8
{
    uint8_t R;

    static void readColor(gl::ColorF *dst, const R8 *src);
    static void readColor(gl::ColorUI *dst, const R8 *src);
    static void writeColor(R8 *dst, const gl::ColorF *src);
    static void writeColor(R8 *dst, const gl::ColorUI *src);
    static void average(R8 *dst, const R8 *src1, const R8 *src2);
};

struct A8
{
    uint8_t A;

    static void readColor(gl::ColorF *dst, const A8 *src);
    static void writeColor(A8 *dst, const gl::ColorF *src);
    static void average(A8 *dst, const A8 *src1, const A8 *src2);
};

struct L8A8
{
    uint8_t L;
    uint8_t A;

    static void readColor(gl::ColorF *dst, const L8A8 *src);
    static void writeColor(L8A8 *dst, const gl::ColorF *src);
    static void average(L8A8 *dst, const L8A8 *src1, const L8A8 *src2);
};

struct A8L8
{
    uint8_t A;
    uint8_t L;

    static void readColor(gl::ColorF *dst, const A8L8 *src);
    static void writeColor(A8L8 *dst, const gl::ColorF *src);
    static void average(A8L8 *dst, const A8L8 *src1, const A8L8 *src2);
};

struct R8G8
{
    uint8_t R;
    uint8_t G;

    static void readColor(gl::ColorF *dst, const R8G8 *src);
    static void readColor(gl::ColorUI *dst, const R8G8 *src);
    static void writeColor(R8G8 *dst, const gl::ColorF *src);
    static void writeColor(R8G8 *dst, const gl::ColorUI *src);
    static void average(R8G8 *dst, const R8G8 *src1, const R8G8 *src2);
};

struct R8G8B8
{
    uint8_t R;
    uint8_t G;
    uint8_t B;

    static void readColor(gl::ColorF *dst, const R8G8B8 *src);
    static void readColor(gl::ColorUI *dst, const R8G8B8 *src);
    static void writeColor(R8G8B8 *dst, const gl::ColorF *src);
    static void writeColor(R8G8B8 *dst, const gl::ColorUI *src);
    static void average(R8G8B8 *dst, const R8G8B8 *src1, const R8G8B8 *src2);
};

struct B8G8R8
{
    uint8_t B;
    uint8_t G;
    uint8_t R;

    static void readColor(gl::ColorF *dst, const B8G8R8 *src);
    static void readColor(gl::ColorUI *dst, const B8G8R8 *src);
    static void writeColor(B8G8R8 *dst, const gl::ColorF *src);
    static void writeColor(B8G8R8 *dst, const gl::ColorUI *src);
    static void average(B8G8R8 *dst, const B8G8R8 *src1, const B8G8R8 *src2);
};

struct R5G6B5
{
    // OpenGL ES 2.0.25 spec Section 3.6.2: "Components are packed with the first component in the
    // most significant bits of the bitfield, and successive component occupying progressively less
    // significant locations"
    uint16_t RGB;

    static void readColor(gl::ColorF *dst, const R5G6B5 *src);
    static void writeColor(R5G6B5 *dst, const gl::ColorF *src);
    static void average(R5G6B5 *dst, const R5G6B5 *src1, const R5G6B5 *src2);
};

struct B5G6R5
{
    uint16_t BGR;

    static void readColor(gl::ColorF *dst, const B5G6R5 *src);
    static void writeColor(B5G6R5 *dst, const gl::ColorF *src);
    static void average(B5G6R5 *dst, const B5G6R5 *src1, const B5G6R5 *src2);
};

struct A8R8G8B8
{
    uint8_t A;
    uint8_t R;
    uint8_t G;
    uint8_t B;

    static void readColor(gl::ColorF *dst, const A8R8G8B8 *src);
    static void readColor(gl::ColorUI *dst, const A8R8G8B8 *src);
    static void writeColor(A8R8G8B8 *dst, const gl::ColorF *src);
    static void writeColor(A8R8G8B8 *dst, const gl::ColorUI *src);
    static void average(A8R8G8B8 *dst, const A8R8G8B8 *src1, const A8R8G8B8 *src2);
};

struct R8G8B8A8
{
    uint8_t R;
    uint8_t G;
    uint8_t B;
    uint8_t A;

    static void readColor(gl::ColorF *dst, const R8G8B8A8 *src);
    static void readColor(gl::ColorUI *dst, const R8G8B8A8 *src);
    static void writeColor(R8G8B8A8 *dst, const gl::ColorF *src);
    static void writeColor(R8G8B8A8 *dst, const gl::ColorUI *src);
    static void average(R8G8B8A8 *dst, const R8G8B8A8 *src1, const R8G8B8A8 *src2);
};

struct R8G8B8A8SRGB
{
    uint8_t R;
    uint8_t G;
    uint8_t B;
    uint8_t A;

    static void readColor(gl::ColorF *dst, const R8G8B8A8SRGB *src);
    static void writeColor(R8G8B8A8SRGB *dst, const gl::ColorF *src);
    static void average(R8G8B8A8SRGB *dst, const R8G8B8A8SRGB *src1, const R8G8B8A8SRGB *src2);
};

struct B8G8R8A8
{
    uint8_t B;
    uint8_t G;
    uint8_t R;
    uint8_t A;

    static void readColor(gl::ColorF *dst, const B8G8R8A8 *src);
    static void readColor(gl::ColorUI *dst, const B8G8R8A8 *src);
    static void writeColor(B8G8R8A8 *dst, const gl::ColorF *src);
    static void writeColor(B8G8R8A8 *dst, const gl::ColorUI *src);
    static void average(B8G8R8A8 *dst, const B8G8R8A8 *src1, const B8G8R8A8 *src2);
};

struct B8G8R8X8
{
    uint8_t B;
    uint8_t G;
    uint8_t R;
    uint8_t X;

    static void readColor(gl::ColorF *dst, const B8G8R8X8 *src);
    static void readColor(gl::ColorUI *dst, const B8G8R8X8 *src);
    static void writeColor(B8G8R8X8 *dst, const gl::ColorF *src);
    static void writeColor(B8G8R8X8 *dst, const gl::ColorUI *src);
    static void average(B8G8R8X8 *dst, const B8G8R8X8 *src1, const B8G8R8X8 *src2);
};

struct A1R5G5B5
{
    uint16_t ARGB;

    static void readColor(gl::ColorF *dst, const A1R5G5B5 *src);
    static void writeColor(A1R5G5B5 *dst, const gl::ColorF *src);
    static void average(A1R5G5B5 *dst, const A1R5G5B5 *src1, const A1R5G5B5 *src2);
};

struct R5G5B5A1
{
    // OpenGL ES 2.0.25 spec Section 3.6.2: "Components are packed with the first component in the
    // most significant
    // bits of the bitfield, and successive component occupying progressively less significant
    // locations"
    uint16_t RGBA;

    static void readColor(gl::ColorF *dst, const R5G5B5A1 *src);
    static void writeColor(R5G5B5A1 *dst, const gl::ColorF *src);
    static void average(R5G5B5A1 *dst, const R5G5B5A1 *src1, const R5G5B5A1 *src2);
};

struct R4G4B4A4
{
    // OpenGL ES 2.0.25 spec Section 3.6.2: "Components are packed with the first component in the
    // most significant
    // bits of the bitfield, and successive component occupying progressively less significant
    // locations"
    uint16_t RGBA;

    static void readColor(gl::ColorF *dst, const R4G4B4A4 *src);
    static void writeColor(R4G4B4A4 *dst, const gl::ColorF *src);
    static void average(R4G4B4A4 *dst, const R4G4B4A4 *src1, const R4G4B4A4 *src2);
};

struct A4R4G4B4
{
    uint16_t ARGB;

    static void readColor(gl::ColorF *dst, const A4R4G4B4 *src);
    static void writeColor(A4R4G4B4 *dst, const gl::ColorF *src);
    static void average(A4R4G4B4 *dst, const A4R4G4B4 *src1, const A4R4G4B4 *src2);
};

struct R16
{
    uint16_t R;

    static void readColor(gl::ColorF *dst, const R16 *src);
    static void readColor(gl::ColorUI *dst, const R16 *src);
    static void writeColor(R16 *dst, const gl::ColorF *src);
    static void writeColor(R16 *dst, const gl::ColorUI *src);
    static void average(R16 *dst, const R16 *src1, const R16 *src2);
};

struct R16G16
{
    uint16_t R;
    uint16_t G;

    static void readColor(gl::ColorF *dst, const R16G16 *src);
    static void readColor(gl::ColorUI *dst, const R16G16 *src);
    static void writeColor(R16G16 *dst, const gl::ColorF *src);
    static void writeColor(R16G16 *dst, const gl::ColorUI *src);
    static void average(R16G16 *dst, const R16G16 *src1, const R16G16 *src2);
};

struct R16G16B16
{
    uint16_t R;
    uint16_t G;
    uint16_t B;

    static void readColor(gl::ColorF *dst, const R16G16B16 *src);
    static void readColor(gl::ColorUI *dst, const R16G16B16 *src);
    static void writeColor(R16G16B16 *dst, const gl::ColorF *src);
    static void writeColor(R16G16B16 *dst, const gl::ColorUI *src);
    static void average(R16G16B16 *dst, const R16G16B16 *src1, const R16G16B16 *src2);
};

struct R16G16B16A16
{
    uint16_t R;
    uint16_t G;
    uint16_t B;
    uint16_t A;

    static void readColor(gl::ColorF *dst, const R16G16B16A16 *src);
    static void readColor(gl::ColorUI *dst, const R16G16B16A16 *src);
    static void writeColor(R16G16B16A16 *dst, const gl::ColorF *src);
    static void writeColor(R16G16B16A16 *dst, const gl::ColorUI *src);
    static void average(R16G16B16A16 *dst, const R16G16B16A16 *src1, const R16G16B16A16 *src2);
};

struct R32
{
    uint32_t R;

    static void readColor(gl::ColorF *dst, const R32 *src);
    static void readColor(gl::ColorUI *dst, const R32 *src);
    static void writeColor(R32 *dst, const gl::ColorF *src);
    static void writeColor(R32 *dst, const gl::ColorUI *src);
    static void average(R32 *dst, const R32 *src1, const R32 *src2);
};

struct R32G32
{
    uint32_t R;
    uint32_t G;

    static void readColor(gl::ColorF *dst, const R32G32 *src);
    static void readColor(gl::ColorUI *dst, const R32G32 *src);
    static void writeColor(R32G32 *dst, const gl::ColorF *src);
    static void writeColor(R32G32 *dst, const gl::ColorUI *src);
    static void average(R32G32 *dst, const R32G32 *src1, const R32G32 *src2);
};

struct R32G32B32
{
    uint32_t R;
    uint32_t G;
    uint32_t B;

    static void readColor(gl::ColorF *dst, const R32G32B32 *src);
    static void readColor(gl::ColorUI *dst, const R32G32B32 *src);
    static void writeColor(R32G32B32 *dst, const gl::ColorF *src);
    static void writeColor(R32G32B32 *dst, const gl::ColorUI *src);
    static void average(R32G32B32 *dst, const R32G32B32 *src1, const R32G32B32 *src2);
};

struct R32G32B32A32
{
    uint32_t R;
    uint32_t G;
    uint32_t B;
    uint32_t A;

    static void readColor(gl::ColorF *dst, const R32G32B32A32 *src);
    static void readColor(gl::ColorUI *dst, const R32G32B32A32 *src);
    static void writeColor(R32G32B32A32 *dst, const gl::ColorF *src);
    static void writeColor(R32G32B32A32 *dst, const gl::ColorUI *src);
    static void average(R32G32B32A32 *dst, const R32G32B32A32 *src1, const R32G32B32A32 *src2);
};

struct R8S
{
    int8_t R;

    static void readColor(gl::ColorF *dst, const R8S *src);
    static void readColor(gl::ColorI *dst, const R8S *src);
    static void writeColor(R8S *dst, const gl::ColorF *src);
    static void writeColor(R8S *dst, const gl::ColorI *src);
    static void average(R8S *dst, const R8S *src1, const R8S *src2);
};

struct R8G8S
{
    int8_t R;
    int8_t G;

    static void readColor(gl::ColorF *dst, const R8G8S *src);
    static void readColor(gl::ColorI *dst, const R8G8S *src);
    static void writeColor(R8G8S *dst, const gl::ColorF *src);
    static void writeColor(R8G8S *dst, const gl::ColorI *src);
    static void average(R8G8S *dst, const R8G8S *src1, const R8G8S *src2);
};

struct R8G8B8S
{
    int8_t R;
    int8_t G;
    int8_t B;

    static void readColor(gl::ColorF *dst, const R8G8B8S *src);
    static void readColor(gl::ColorI *dst, const R8G8B8S *src);
    static void writeColor(R8G8B8S *dst, const gl::ColorF *src);
    static void writeColor(R8G8B8S *dst, const gl::ColorI *src);
    static void average(R8G8B8S *dst, const R8G8B8S *src1, const R8G8B8S *src2);
};

struct R8G8B8A8S
{
    int8_t R;
    int8_t G;
    int8_t B;
    int8_t A;

    static void readColor(gl::ColorF *dst, const R8G8B8A8S *src);
    static void readColor(gl::ColorI *dst, const R8G8B8A8S *src);
    static void writeColor(R8G8B8A8S *dst, const gl::ColorF *src);
    static void writeColor(R8G8B8A8S *dst, const gl::ColorI *src);
    static void average(R8G8B8A8S *dst, const R8G8B8A8S *src1, const R8G8B8A8S *src2);
};

struct R16S
{
    int16_t R;

    static void readColor(gl::ColorF *dst, const R16S *src);
    static void readColor(gl::ColorI *dst, const R16S *src);
    static void writeColor(R16S *dst, const gl::ColorF *src);
    static void writeColor(R16S *dst, const gl::ColorI *src);
    static void average(R16S *dst, const R16S *src1, const R16S *src2);
};

struct R16G16S
{
    int16_t R;
    int16_t G;

    static void readColor(gl::ColorF *dst, const R16G16S *src);
    static void readColor(gl::ColorI *dst, const R16G16S *src);
    static void writeColor(R16G16S *dst, const gl::ColorF *src);
    static void writeColor(R16G16S *dst, const gl::ColorI *src);
    static void average(R16G16S *dst, const R16G16S *src1, const R16G16S *src2);
};

struct R16G16B16S
{
    int16_t R;
    int16_t G;
    int16_t B;

    static void readColor(gl::ColorF *dst, const R16G16B16S *src);
    static void readColor(gl::ColorI *dst, const R16G16B16S *src);
    static void writeColor(R16G16B16S *dst, const gl::ColorF *src);
    static void writeColor(R16G16B16S *dst, const gl::ColorI *src);
    static void average(R16G16B16S *dst, const R16G16B16S *src1, const R16G16B16S *src2);
};

struct R16G16B16A16S
{
    int16_t R;
    int16_t G;
    int16_t B;
    int16_t A;

    static void readColor(gl::ColorF *dst, const R16G16B16A16S *src);
    static void readColor(gl::ColorI *dst, const R16G16B16A16S *src);
    static void writeColor(R16G16B16A16S *dst, const gl::ColorF *src);
    static void writeColor(R16G16B16A16S *dst, const gl::ColorI *src);
    static void average(R16G16B16A16S *dst, const R16G16B16A16S *src1, const R16G16B16A16S *src2);
};

struct R32S
{
    int32_t R;

    static void readColor(gl::ColorF *dst, const R32S *src);
    static void readColor(gl::ColorI *dst, const R32S *src);
    static void writeColor(R32S *dst, const gl::ColorF *src);
    static void writeColor(R32S *dst, const gl::ColorI *src);
    static void average(R32S *dst, const R32S *src1, const R32S *src2);
};

struct R32G32S
{
    int32_t R;
    int32_t G;

    static void readColor(gl::ColorF *dst, const R32G32S *src);
    static void readColor(gl::ColorI *dst, const R32G32S *src);
    static void writeColor(R32G32S *dst, const gl::ColorF *src);
    static void writeColor(R32G32S *dst, const gl::ColorI *src);
    static void average(R32G32S *dst, const R32G32S *src1, const R32G32S *src2);
};

struct R32G32B32S
{
    int32_t R;
    int32_t G;
    int32_t B;

    static void readColor(gl::ColorF *dst, const R32G32B32S *src);
    static void readColor(gl::ColorI *dst, const R32G32B32S *src);
    static void writeColor(R32G32B32S *dst, const gl::ColorF *src);
    static void writeColor(R32G32B32S *dst, const gl::ColorI *src);
    static void average(R32G32B32S *dst, const R32G32B32S *src1, const R32G32B32S *src2);
};

struct R32G32B32A32S
{
    int32_t R;
    int32_t G;
    int32_t B;
    int32_t A;

    static void readColor(gl::ColorF *dst, const R32G32B32A32S *src);
    static void readColor(gl::ColorI *dst, const R32G32B32A32S *src);
    static void writeColor(R32G32B32A32S *dst, const gl::ColorF *src);
    static void writeColor(R32G32B32A32S *dst, const gl::ColorI *src);
    static void average(R32G32B32A32S *dst, const R32G32B32A32S *src1, const R32G32B32A32S *src2);
};

struct A16B16G16R16F
{
    uint16_t A;
    uint16_t R;
    uint16_t G;
    uint16_t B;

    static void readColor(gl::ColorF *dst, const A16B16G16R16F *src);
    static void writeColor(A16B16G16R16F *dst, const gl::ColorF *src);
    static void average(A16B16G16R16F *dst, const A16B16G16R16F *src1, const A16B16G16R16F *src2);
};

struct R16G16B16A16F
{
    uint16_t R;
    uint16_t G;
    uint16_t B;
    uint16_t A;

    static void readColor(gl::ColorF *dst, const R16G16B16A16F *src);
    static void writeColor(R16G16B16A16F *dst, const gl::ColorF *src);
    static void average(R16G16B16A16F *dst, const R16G16B16A16F *src1, const R16G16B16A16F *src2);
};

struct R16F
{
    uint16_t R;

    static void readColor(gl::ColorF *dst, const R16F *src);
    static void writeColor(R16F *dst, const gl::ColorF *src);
    static void average(R16F *dst, const R16F *src1, const R16F *src2);
};

struct A16F
{
    uint16_t A;

    static void readColor(gl::ColorF *dst, const A16F *src);
    static void writeColor(A16F *dst, const gl::ColorF *src);
    static void average(A16F *dst, const A16F *src1, const A16F *src2);
};

struct L16F
{
    uint16_t L;

    static void readColor(gl::ColorF *dst, const L16F *src);
    static void writeColor(L16F *dst, const gl::ColorF *src);
    static void average(L16F *dst, const L16F *src1, const L16F *src2);
};

struct L16A16F
{
    uint16_t L;
    uint16_t A;

    static void readColor(gl::ColorF *dst, const L16A16F *src);
    static void writeColor(L16A16F *dst, const gl::ColorF *src);
    static void average(L16A16F *dst, const L16A16F *src1, const L16A16F *src2);
};

struct R16G16F
{
    uint16_t R;
    uint16_t G;

    static void readColor(gl::ColorF *dst, const R16G16F *src);
    static void writeColor(R16G16F *dst, const gl::ColorF *src);
    static void average(R16G16F *dst, const R16G16F *src1, const R16G16F *src2);
};

struct R16G16B16F
{
    uint16_t R;
    uint16_t G;
    uint16_t B;

    static void readColor(gl::ColorF *dst, const R16G16B16F *src);
    static void writeColor(R16G16B16F *dst, const gl::ColorF *src);
    static void average(R16G16B16F *dst, const R16G16B16F *src1, const R16G16B16F *src2);
};

struct A32B32G32R32F
{
    float A;
    float R;
    float G;
    float B;

    static void readColor(gl::ColorF *dst, const A32B32G32R32F *src);
    static void writeColor(A32B32G32R32F *dst, const gl::ColorF *src);
    static void average(A32B32G32R32F *dst, const A32B32G32R32F *src1, const A32B32G32R32F *src2);
};

struct R32G32B32A32F
{
    float R;
    float G;
    float B;
    float A;

    static void readColor(gl::ColorF *dst, const R32G32B32A32F *src);
    static void writeColor(R32G32B32A32F *dst, const gl::ColorF *src);
    static void average(R32G32B32A32F *dst, const R32G32B32A32F *src1, const R32G32B32A32F *src2);
};

struct R32F
{
    float R;

    static void readColor(gl::ColorF *dst, const R32F *src);
    static void writeColor(R32F *dst, const gl::ColorF *src);
    static void average(R32F *dst, const R32F *src1, const R32F *src2);
};

struct A32F
{
    float A;

    static void readColor(gl::ColorF *dst, const A32F *src);
    static void writeColor(A32F *dst, const gl::ColorF *src);
    static void average(A32F *dst, const A32F *src1, const A32F *src2);
};

struct L32F
{
    float L;

    static void readColor(gl::ColorF *dst, const L32F *src);
    static void writeColor(L32F *dst, const gl::ColorF *src);
    static void average(L32F *dst, const L32F *src1, const L32F *src2);
};

struct L32A32F
{
    float L;
    float A;

    static void readColor(gl::ColorF *dst, const L32A32F *src);
    static void writeColor(L32A32F *dst, const gl::ColorF *src);
    static void average(L32A32F *dst, const L32A32F *src1, const L32A32F *src2);
};

struct R32G32F
{
    float R;
    float G;

    static void readColor(gl::ColorF *dst, const R32G32F *src);
    static void writeColor(R32G32F *dst, const gl::ColorF *src);
    static void average(R32G32F *dst, const R32G32F *src1, const R32G32F *src2);
};

struct R32G32B32F
{
    float R;
    float G;
    float B;

    static void readColor(gl::ColorF *dst, const R32G32B32F *src);
    static void writeColor(R32G32B32F *dst, const gl::ColorF *src);
    static void average(R32G32B32F *dst, const R32G32B32F *src1, const R32G32B32F *src2);
};

struct R10G10B10A2
{
    uint32_t R : 10;
    uint32_t G : 10;
    uint32_t B : 10;
    uint32_t A : 2;

    static void readColor(gl::ColorF *dst, const R10G10B10A2 *src);
    static void readColor(gl::ColorUI *dst, const R10G10B10A2 *src);
    static void writeColor(R10G10B10A2 *dst, const gl::ColorF *src);
    static void writeColor(R10G10B10A2 *dst, const gl::ColorUI *src);
    static void average(R10G10B10A2 *dst, const R10G10B10A2 *src1, const R10G10B10A2 *src2);
};
static_assert(sizeof(R10G10B10A2) == 4, "R10G10B10A2 struct not 32-bits.");

struct R10G10B10A2S
{
    int32_t R : 10;
    int32_t G : 10;
    int32_t B : 10;
    int32_t A : 2;

    static void readColor(gl::ColorF *dst, const R10G10B10A2S *src);
    static void readColor(gl::ColorI *dst, const R10G10B10A2S *src);
    static void writeColor(R10G10B10A2S *dst, const gl::ColorF *src);
    static void writeColor(R10G10B10A2S *dst, const gl::ColorI *src);
    static void average(R10G10B10A2S *dst, const R10G10B10A2S *src1, const R10G10B10A2S *src2);
};
static_assert(sizeof(R10G10B10A2S) == 4, "R10G10B10A2S struct not 32-bits.");

struct R10G10B10X2
{
    uint32_t R : 10;
    uint32_t G : 10;
    uint32_t B : 10;

    static void readColor(gl::ColorF *dst, const R10G10B10X2 *src);
    static void readColor(gl::ColorUI *dst, const R10G10B10X2 *src);
    static void writeColor(R10G10B10X2 *dst, const gl::ColorF *src);
    static void writeColor(R10G10B10X2 *dst, const gl::ColorUI *src);
    static void average(R10G10B10X2 *dst, const R10G10B10X2 *src1, const R10G10B10X2 *src2);
};
static_assert(sizeof(R10G10B10X2) == 4, "R10G10B10X2 struct not 32-bits.");

struct R9G9B9E5
{
    uint32_t R : 9;
    uint32_t G : 9;
    uint32_t B : 9;
    uint32_t E : 5;

    static void readColor(gl::ColorF *dst, const R9G9B9E5 *src);
    static void writeColor(R9G9B9E5 *dst, const gl::ColorF *src);
    static void average(R9G9B9E5 *dst, const R9G9B9E5 *src1, const R9G9B9E5 *src2);
};
static_assert(sizeof(R9G9B9E5) == 4, "R9G9B9E5 struct not 32-bits.");

struct R11G11B10F
{
    uint32_t R : 11;
    uint32_t G : 11;
    uint32_t B : 10;

    static void readColor(gl::ColorF *dst, const R11G11B10F *src);
    static void writeColor(R11G11B10F *dst, const gl::ColorF *src);
    static void average(R11G11B10F *dst, const R11G11B10F *src1, const R11G11B10F *src2);
};
static_assert(sizeof(R11G11B10F) == 4, "R11G11B10F struct not 32-bits.");

struct D24S8
{
    uint32_t S : 8;
    uint32_t D : 24;

    static void ReadDepthStencil(DepthStencil *dst, const D24S8 *src);
    static void WriteDepthStencil(D24S8 *dst, const DepthStencil *src);
};

struct S8
{
    uint8_t S;

    static void ReadDepthStencil(DepthStencil *dst, const S8 *src);
    static void WriteDepthStencil(S8 *dst, const DepthStencil *src);
};

struct D16
{
    uint16_t D;

    static void ReadDepthStencil(DepthStencil *dst, const D16 *src);
    static void WriteDepthStencil(D16 *dst, const DepthStencil *src);
};

struct D24X8
{
    uint32_t D;

    static void ReadDepthStencil(DepthStencil *dst, const D24X8 *src);
    static void WriteDepthStencil(D24X8 *dst, const DepthStencil *src);
};

struct D32F
{
    float D;

    static void ReadDepthStencil(DepthStencil *dst, const D32F *src);
    static void WriteDepthStencil(D32F *dst, const DepthStencil *src);
};

struct D32
{
    uint32_t D;

    static void ReadDepthStencil(DepthStencil *dst, const D32 *src);
    static void WriteDepthStencil(D32 *dst, const DepthStencil *src);
};

struct D32FS8X24
{
    float D;
    uint32_t S;

    static void ReadDepthStencil(DepthStencil *dst, const D32FS8X24 *src);
    static void WriteDepthStencil(D32FS8X24 *dst, const DepthStencil *src);
};
}  // namespace angle

#endif  // IMAGEUTIL_IMAGEFORMATS_H_
