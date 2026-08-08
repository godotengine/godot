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

struct FunctionInlineCache {
	enum class CacheState : uint8_t {
		UNINITIALIZED,
		INITIALIZING,
		MONOMORPHIC,
		DISABLED,
	};

	CacheState state;
	bool is_static;

	GDType *type;
	Ref<Script> script;
	Variant::VariantCacheFunctionCall fn;

private:
	void load(Variant &p_base, const StringName &p_method);

	_FORCE_INLINE_ static GDType *get_type(const Variant &v) {
		Variant::Type vtype = v.get_type();
		if (vtype == Variant::OBJECT) {
			return const_cast<GDType *>(&(*VariantInternal::get_object(&v))->get_gdtype());
		}
		// Others not supported currently.
		return nullptr;
	}

	_FORCE_INLINE_ bool hit(Variant &p_base) const {
		return get_type(p_base) == type && (p_base.get_type() != Variant::OBJECT || script_matches(*VariantInternal::get_object(&p_base)));
	}

	_FORCE_INLINE_ bool script_matches(Object *obj) const {
		if (is_static) {
			return *script == Object::cast_to<Script>(obj);
		}
		const ScriptInstance *si = obj->get_script_instance();
		return si->script_eq(script);
	}

public:
	_FORCE_INLINE_ void callp(Variant &p_base, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *p_ret, Callable::CallError &p_error) {
		if (unlikely(state == CacheState::UNINITIALIZED)) {
			load(p_base, p_method);
		}
		if (state == CacheState::MONOMORPHIC) {
			if (likely(hit(p_base))) {
				if (p_ret != nullptr) {
					*p_ret = fn(&p_base, p_args, p_argcount, p_error);
				} else {
					fn(&p_base, p_args, p_argcount, p_error);
				}
				return;
			} else {
				state = CacheState::DISABLED;
				// TODO: cleanup?
			}
		}
		// Fallback to variant call
		if (p_ret != nullptr) {
			p_base.callp(p_method, p_args, p_argcount, *p_ret, p_error);
		} else {
			Variant v;
			p_base.callp(p_method, p_args, p_argcount, v, p_error);
		}
	}

	void reset();
};

constexpr size_t FunctionInlineCacheIntSize = (sizeof(FunctionInlineCache) + sizeof(int) - 1) / sizeof(int);
