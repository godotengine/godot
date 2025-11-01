/**************************************************************************/
/*  d3d12ma.cpp                                                           */
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

// Wrapper needed to set the required rpcndr version for MinGW compatibility.
// Since we're compiling thirdparty code in a Godot SCons environment with
// warnings enabled, we also need to silence them manually.

#include "rendering_device_driver_d3d12.h" // For __REQUIRED_RPCNDR_H_VERSION__.

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wduplicated-branches")
GODOT_GCC_WARNING_IGNORE("-Wimplicit-fallthrough")
GODOT_GCC_WARNING_IGNORE("-Wmaybe-uninitialized")
GODOT_GCC_WARNING_IGNORE("-Wmissing-field-initializers")
GODOT_GCC_WARNING_IGNORE("-Wnon-virtual-dtor")
GODOT_GCC_WARNING_IGNORE("-Wnonnull-compare")
GODOT_GCC_WARNING_IGNORE("-Wshadow")
GODOT_GCC_WARNING_IGNORE("-Wsign-compare")
GODOT_GCC_WARNING_IGNORE("-Wswitch")
GODOT_GCC_WARNING_IGNORE("-Wunused-function")
GODOT_GCC_WARNING_IGNORE("-Wunused-variable")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wimplicit-fallthrough")
GODOT_CLANG_WARNING_IGNORE("-Wmissing-field-initializers")
GODOT_CLANG_WARNING_IGNORE("-Wnon-virtual-dtor")
GODOT_CLANG_WARNING_IGNORE("-Wstring-plus-int")
GODOT_CLANG_WARNING_IGNORE("-Wswitch")
GODOT_CLANG_WARNING_IGNORE("-Wtautological-undefined-compare")
GODOT_CLANG_WARNING_IGNORE("-Wunused-but-set-variable")
GODOT_CLANG_WARNING_IGNORE("-Wunused-function")
GODOT_CLANG_WARNING_IGNORE("-Wunused-private-field")
GODOT_CLANG_WARNING_IGNORE("-Wunused-variable")
GODOT_MSVC_WARNING_PUSH
GODOT_MSVC_WARNING_IGNORE(4189) // "Local variable is initialized but not referenced".
GODOT_MSVC_WARNING_IGNORE(4505) // "Unreferenced local function has been removed".

#include <D3D12MemAlloc.cpp>

GODOT_GCC_WARNING_POP
GODOT_CLANG_WARNING_POP
GODOT_MSVC_WARNING_POP
