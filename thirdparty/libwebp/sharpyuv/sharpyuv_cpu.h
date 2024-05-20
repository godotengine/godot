// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
#ifndef WEBP_SHARPYUV_SHARPYUV_CPU_H_
#define WEBP_SHARPYUV_SHARPYUV_CPU_H_

#include "sharpyuv/sharpyuv.h"

// Avoid exporting SharpYuvGetCPUInfo in shared object / DLL builds.
// SharpYuvInit() replaces the use of the function pointer.
#undef WEBP_EXTERN
#define WEBP_EXTERN extern
#define VP8GetCPUInfo SharpYuvGetCPUInfo
#include "src/dsp/cpu.h"

#endif  // WEBP_SHARPYUV_SHARPYUV_CPU_H_
