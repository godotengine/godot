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

void InlineCache::add_entry(const Script *p_script, void *p_method) {
	DEV_ASSERT(size < MAX_SIZE);

	if (size == 0) {
		script = p_script;
		method = p_method;
	} else {
		if (size == 1) {
			// convert to polymorphic
			const Script **new_scripts = reinterpret_cast<const Script **>(memalloc(sizeof(const Script *) * MAX_SIZE));
			void **new_methods = reinterpret_cast<void **>(memalloc(sizeof(void *) * MAX_SIZE));

			new_scripts[0] = script;
			new_methods[0] = method;

			scripts = new_scripts;
			methods = new_methods;
		}
		scripts[size] = p_script;
		methods[size] = p_method;
	}
	size++;
}

void InlineCache::set_mega() {
	if (size > 1 && size <= MAX_SIZE) {
		memfree(scripts);
		memfree(methods);
	}
	size = SIZE_MEGA;
}

void InlineCache::call(Variant &p_base, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Callable::CallError &r_error) {
	if (size == SIZE_MEGA || p_base.get_type() != Variant::OBJECT) {
		goto fallback;
	}
	{
		const Object *obj = *VariantInternal::get_object(&p_base);

		if (!obj) {
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		}

#ifdef DEBUG_ENABLED
		if (EngineDebugger::is_active() && !VariantInternal::get_object_id(&p_base).is_ref_counted() && ObjectDB::get_instance(VariantInternal::get_object_id(&p_base)) == nullptr) {
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		}

#endif // DEBUG_ENABLED

		const ScriptInstance *si = obj->get_script_instance();
		// For static method calls, there is no ScriptInstance instead the object is the Script.
		const Script *lookup_script = si != nullptr ? si->get_script_raw() : Object::cast_to<Script>(obj);

		if (unlikely(lookup_script == nullptr)) {
			set_mega();
			goto fallback;
		}

		void *found_method = nullptr;
		// Lookup section
		if (size == 1) {
			if (likely(lookup_script == script)) {
				found_method = method;
			}
		} else if (size > 1) {
			// Search entries
			for (uint8_t i = 0; i < size; i++) {
				if (lookup_script == scripts[i]) {
					found_method = methods[i];
					break;
				}
			}
		}
		if (likely(found_method != nullptr)) {
			lookup_script->call_method(p_base, found_method, p_args, p_argcount, r_ret, r_error);
			return;
		}
		// Lookup failed, add new cache entry
		if (size != MAX_SIZE) {
			found_method = lookup_script->lookup_method(p_method);
			if (likely(found_method != nullptr)) {
				add_entry(lookup_script, found_method);
				lookup_script->call_method(p_base, found_method, p_args, p_argcount, r_ret, r_error);
				return;
			}
		}
		set_mega();
	}

	// Bad case
fallback:
	if (r_ret != nullptr) {
		p_base.callp(p_method, p_args, p_argcount, *r_ret, r_error);
	} else {
		Variant v;
		p_base.callp(p_method, p_args, p_argcount, v, r_error);
	}
}

InlineCache::~InlineCache() {
	if (size > 1 && size <= MAX_SIZE) {
		memfree(scripts);
		memfree(methods);
	}
}
