/**************************************************************************/
/*  godot_d3dx12.h                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/typedefs.h"

#if !defined(_MSC_VER) && !defined(__REQUIRED_RPCNDR_H_VERSION__)
// Match current version used by MinGW, MSVC and Direct3D 12 headers use 500.
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif // !defined(_MSC_VER) && !defined(__REQUIRED_RPCNDR_H_VERSION__)

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wnon-virtual-dtor")
GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wnon-virtual-dtor")

#include <thirdparty/directx_headers/include/directx/d3dx12.h> // IWYU pragma: export.

GODOT_GCC_WARNING_POP
GODOT_CLANG_WARNING_POP
