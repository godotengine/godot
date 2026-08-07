/**************************************************************************/
/*  inline_cache.h                                                        */
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

#include "core/object/script_language.h"
#include "core/variant/variant.h"

class InlineCache {
	// Used as the size of polymorphic allocations.
	static const uint8_t MAX_SIZE = 4u;
	// Indicates that the cache is megamorphic
	static const uint8_t SIZE_MEGA = 255u;

	union {
		const Script *script;
		const Script **scripts;
	};
	union {
		void *method;
		void **methods;
	};
	// Size determines type of cache used:
	// 0 - uninitialized
	// 1 - monomorphic, check script/method
	// 2-4 - polymorphic, search scripts/methods
	// 255 - megamorphic - fallback to normal lookup
	uint8_t size = 0;

	void add_entry(const Script *p_script, void *p_method);
	void set_mega();

public:
	void call(Variant &p_base, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Callable::CallError &r_error);

	InlineCache() = default;
	InlineCache(const InlineCache &) = delete;
	InlineCache operator=(const InlineCache &) = delete;
	~InlineCache();
};
