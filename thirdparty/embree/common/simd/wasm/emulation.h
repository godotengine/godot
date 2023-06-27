// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// According to https://emscripten.org/docs/porting/simd.html, _MM_SET_EXCEPTION_MASK and
// _mm_setcsr are unavailable in WebAssembly.

#define _MM_SET_EXCEPTION_MASK(x)

__forceinline void _mm_setcsr(unsigned int)
{
}
