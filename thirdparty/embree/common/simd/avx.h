// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sse.h"

#if defined(__AVX512VL__)
#include "vboolf8_avx512.h"
#include "vboold4_avx512.h"
#else
#include "vboolf8_avx.h"
#include "vboold4_avx.h"
#endif

#if defined(__AVX2__)
#include "vint8_avx2.h"
#include "vuint8_avx2.h"
#if defined(__X86_64__)
#include "vllong4_avx2.h"
#endif
#else
#include "vint8_avx.h"
#include "vuint8_avx.h"
#endif
#include "vfloat8_avx.h"
#if defined(__X86_64__)
#include "vdouble4_avx.h"
#endif

#if defined(__AVX512F__)
#include "avx512.h"
#endif
