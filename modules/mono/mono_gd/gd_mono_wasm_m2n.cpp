/*************************************************************************/
/*  gd_mono_wasm_m2n.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "gd_mono_wasm_m2n.h"

#ifdef JAVASCRIPT_ENABLED

#include "core/templates/oa_hash_map.h"

typedef mono_bool (*GodotMonoM2nIcallTrampolineDispatch)(const char *cookie, void *target_func, Mono_InterpMethodArguments *margs);

// This extern function is implemented in our patched version of Mono
MONO_API void godot_mono_register_m2n_icall_trampoline_dispatch_hook(GodotMonoM2nIcallTrampolineDispatch hook);

namespace GDMonoWasmM2n {

struct HashMapCookieComparator {
	static bool compare(const char *p_lhs, const char *p_rhs) {
		return strcmp(p_lhs, p_rhs) == 0;
	}
};

// The default hasher supports 'const char *' C Strings, but we need a custom comparator
OAHashMap<const char *, TrampolineFunc, HashMapHasherDefault, HashMapCookieComparator> trampolines;

void set_trampoline(const char *cookies, GDMonoWasmM2n::TrampolineFunc trampoline_func) {
	trampolines.set(cookies, trampoline_func);
}

mono_bool trampoline_dispatch_hook(const char *cookie, void *target_func, Mono_InterpMethodArguments *margs) {
	TrampolineFunc *trampoline_func = trampolines.lookup_ptr(cookie);

	if (!trampoline_func) {
		return false;
	}

	(*trampoline_func)(target_func, margs);
	return true;
}

bool initialized = false;

void lazy_initialize() {
	// Doesn't need to be thread safe
	if (!initialized) {
		initialized = true;
		godot_mono_register_m2n_icall_trampoline_dispatch_hook(&trampoline_dispatch_hook);
	}
}
} // namespace GDMonoWasmM2n

#endif
