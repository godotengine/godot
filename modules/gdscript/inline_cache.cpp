/**************************************************************************/
/*  inline_cache.cpp                                                      */
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

#include "inline_cache.h"

#include "core/debugger/engine_debugger.h"
#include "core/object/object.h"
#include "core/os/memory.h"
#include "core/variant/variant_internal.h"

void FunctionInlineCache::load(Variant &p_base, const StringName &p_method) {
	Callable::CallError::Error err;
	VariantCallCache found = p_base.lookup_function_call(p_method, err);
	if (found.type != VariantCallCache::Type::INVALID) {
		// There is a chance another thread already updated while we looked up the function.
		if (state == CacheState::UNINITIALIZED) {
			state = CacheState::INITIALIZING;
			fn = std::move(found);
			type = p_base.get_type();
			gdtype = get_gdtype(p_base);
			if (p_base.get_type() == Variant::OBJECT) {
				Object *obj = *VariantInternal::get_object(&p_base);

				const ScriptInstance *si = obj->get_script_instance();
				if (si) {
					script = si->get_script();
					is_static = false;
				} else {
					script = Object::cast_to<Script>(obj);
					DEV_ASSERT(script.is_valid());
					is_static = true;
				}
			}
			state = CacheState::MONOMORPHIC;
		}
	} else {
		state = CacheState::DISABLED;
	}
}

void FunctionInlineCache::reset() {
	script = nullptr;
	fn = {};
	type = Variant::NIL;
	state = CacheState::UNINITIALIZED;
}
