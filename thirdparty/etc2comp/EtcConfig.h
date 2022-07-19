/*
 * Copyright 2015 The Etc2Comp Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef _WIN32
#define ETC_WINDOWS (1)
#else
#define ETC_WINDOWS (0)
#endif

#if __APPLE__
#define ETC_OSX (1)
#else
#define ETC_OSX (0)
#endif

#if __unix__
#define ETC_UNIX (1)
#else
#define ETC_UNIX (0)
#endif


// short names for common types
#include <stdint.h>
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float	f32;
typedef double	f64;

// Keep asserts enabled in release builds during development
#undef NDEBUG

// 0=disable. stb_image can be used if you need to compress
//other image formats like jpg
#define USE_STB_IMAGE_LOAD 0

#if ETC_WINDOWS
#include <sdkddkver.h>
#define _CRT_SECURE_NO_WARNINGS (1)
#include <tchar.h>
#endif

#include <stdio.h>

