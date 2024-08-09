/**************************************************************************/
/*  core_globals.cpp                                                      */
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

#include "core_globals.h"

#ifdef SANITIZERS_ENABLED
#ifdef __has_feature
#if __has_feature(address_sanitizer)
#define ASAN_ENABLED
#endif
#elif defined(__SANITIZE_ADDRESS__)
#define ASAN_ENABLED
#endif
#endif

#ifdef ASAN_ENABLED
#include "core/string/print_string.h"
#include "os/os.h"
#endif

bool CoreGlobals::leak_reporting_enabled = true;
bool CoreGlobals::print_line_enabled = true;
bool CoreGlobals::print_error_enabled = true;

#ifdef ASAN_ENABLED
static bool init_allocators_use_asan_malloc() {
	bool use_malloc = !OS::get_singleton()->has_environment("ALLOCATORS_DO_NOT_USE_ASAN_MALLOC");
	if (use_malloc) {
		print_line("Allocators: Using ASan malloc. (Set env ALLOCATORS_DO_NOT_USE_ASAN_MALLOC=1 to disable)");
	} else {
		print_line("Allocators: Not using ASan malloc. (Unset env ALLOCATORS_DO_NOT_USE_ASAN_MALLOC to enable)");
	}
	return use_malloc;
}

bool CoreGlobals::allocators_use_asan_malloc() {
	static bool use_malloc = init_allocators_use_asan_malloc();
	return use_malloc;
}
#endif
