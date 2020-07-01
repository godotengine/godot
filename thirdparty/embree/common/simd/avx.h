// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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

