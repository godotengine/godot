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

#include "gdscript_function.h"

#include "core/object/method_bind.h"
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

	Variant::Type type;
	GDType *gdtype;
	Ref<Script> script;
	VariantCallCache fn;

private:
	void load(Variant &p_base, const StringName &p_method);

	_FORCE_INLINE_ static GDType *get_gdtype(const Variant &v) {
		Variant::Type vtype = v.get_type();
		if (vtype == Variant::OBJECT) {
			return const_cast<GDType *>(&(*VariantInternal::get_object(&v))->get_gdtype());
		}
		// Others not supported currently.
		return nullptr;
	}

	_FORCE_INLINE_ bool hit(Variant &p_base) const {
		return p_base.get_type() == type && (p_base.get_type() != Variant::OBJECT || (get_gdtype(p_base) == gdtype && script_matches(*VariantInternal::get_object(&p_base))));
	}

	_FORCE_INLINE_ bool script_matches(Object *obj) const {
		if (is_static) {
			return *script == Object::cast_to<Script>(obj);
		}
		const ScriptInstance *si = obj->get_script_instance();
		return si->script_eq(script);
	}

public:
	_FORCE_INLINE_ Variant callp(Variant &p_base, const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &p_error) {
		if (unlikely(state == CacheState::UNINITIALIZED)) {
			load(p_base, p_method);
		}
		if (state == CacheState::MONOMORPHIC) {
			if (likely(hit(p_base))) {
				switch (fn.type) {
					case VariantCallCache::Type::GDSCRIPT_FUNCTION: {
						DEV_ASSERT(p_base.get_type() == Variant::OBJECT);
						Object *obj = *VariantInternal::get_object(&p_base);
						return fn.gdscript_function->call(reinterpret_cast<GDScriptInstance *>(obj->get_script_instance()), p_args, p_argcount, p_error);
					} break;
					case VariantCallCache::Type::METHOD_BIND: {
						return fn.method_bind->call(*VariantInternal::get_object(&p_base), p_args, p_argcount, p_error);
					} break;
					case VariantCallCache::Type::VARIANT_BUILTIN_METHOD: {
						Variant ret;
						fn.variant_builtin_method.call(&p_base, p_args, p_argcount, ret, *fn.variant_builtin_method.default_values, p_error);
						return ret;
					}
					default:
						WARN_PRINT("Unhandled call cache type");
						state = CacheState::DISABLED;
				}
			} else {
				state = CacheState::DISABLED;
				// TODO: cleanup?
			}
		}
		// Fallback to variant call
		Variant v;
		p_base.callp(p_method, p_args, p_argcount, v, p_error);
		return v;
	}

	void reset();

	static _FORCE_INLINE_ FunctionInlineCache *get_ptr(uintptr_t start_idx) {
		// Use next aligned location
		return reinterpret_cast<FunctionInlineCache *>((start_idx + alignof(FunctionInlineCache) - 1) & -static_cast<intptr_t>(alignof(FunctionInlineCache)));
	}
};

// Rounds up and add space for alignment
constexpr size_t FunctionInlineCacheIntSize = (sizeof(FunctionInlineCache) + alignof(FunctionInlineCache) - 1) / sizeof(int);
