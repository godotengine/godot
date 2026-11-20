/**************************************************************************/
/*  method_bind.h                                                         */
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

#include "core/variant/type_info.h"
#include "core/variant/variant.h"

// some helpers

class MethodBind {
	int method_id;
	uint32_t hint_flags = METHOD_FLAGS_DEFAULT;
	StringName name;
	StringName instance_class;
	Vector<Variant> default_arguments;
	int default_argument_count = 0;
	int argument_count = 0;

	bool _static = false;
	bool _const = false;
	bool _returns = false;
	bool _returns_raw_obj_ptr = false;

protected:
	Variant::Type *argument_types = nullptr;
#ifdef DEBUG_ENABLED
	Vector<StringName> arg_names;
#endif // DEBUG_ENABLED
	void _set_const(bool p_const);
	void _set_static(bool p_static);
	void _set_returns(bool p_returns);
	virtual Variant::Type _gen_argument_type(int p_arg) const = 0;
	virtual PropertyInfo _gen_argument_type_info(int p_arg) const = 0;
	void _generate_argument_types(int p_count);

	void set_argument_count(int p_count) { argument_count = p_count; }

public:
	_FORCE_INLINE_ const Vector<Variant> &get_default_arguments() const { return default_arguments; }
	_FORCE_INLINE_ int get_default_argument_count() const { return default_argument_count; }

	_FORCE_INLINE_ Variant has_default_argument(int p_arg) const {
		int idx = p_arg - (argument_count - default_arguments.size());

		if (idx < 0 || idx >= default_arguments.size()) {
			return false;
		} else {
			return true;
		}
	}

	_FORCE_INLINE_ Variant get_default_argument(int p_arg) const {
		int idx = p_arg - (argument_count - default_arguments.size());

		if (idx < 0 || idx >= default_arguments.size()) {
			return Variant();
		} else {
			return default_arguments[idx];
		}
	}

	_FORCE_INLINE_ Variant::Type get_argument_type(int p_argument) const {
		ERR_FAIL_COND_V(p_argument < -1 || p_argument >= argument_count, Variant::NIL);
		return argument_types[p_argument + 1];
	}

	PropertyInfo get_argument_info(int p_argument) const;
	PropertyInfo get_return_info() const;

#ifdef DEBUG_ENABLED
	void set_argument_names(const Vector<StringName> &p_names); // Set by ClassDB, can't be inferred otherwise.
	Vector<StringName> get_argument_names() const;

	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const = 0;
#endif // DEBUG_ENABLED

	void set_hint_flags(uint32_t p_hint) { hint_flags = p_hint; }
	uint32_t get_hint_flags() const { return hint_flags | (is_const() ? METHOD_FLAG_CONST : 0) | (is_vararg() ? METHOD_FLAG_VARARG : 0) | (is_static() ? METHOD_FLAG_STATIC : 0); }
	_FORCE_INLINE_ StringName get_instance_class() const { return instance_class; }
	_FORCE_INLINE_ void set_instance_class(const StringName &p_class) { instance_class = p_class; }

	_FORCE_INLINE_ int get_argument_count() const { return argument_count; }

#ifdef TOOLS_ENABLED
	virtual bool is_valid() const { return true; }
#endif

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const = 0;
	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const = 0;

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const = 0;

	StringName get_name() const;
	void set_name(const StringName &p_name);
	_FORCE_INLINE_ int get_method_id() const { return method_id; }
	_FORCE_INLINE_ bool is_const() const { return _const; }
	_FORCE_INLINE_ bool is_static() const { return _static; }
	_FORCE_INLINE_ bool has_return() const { return _returns; }
	virtual bool is_vararg() const { return false; }

	_FORCE_INLINE_ bool is_return_type_raw_object_ptr() { return _returns_raw_obj_ptr; }
	_FORCE_INLINE_ void set_return_type_is_raw_object_ptr(bool p_returns_raw_obj) { _returns_raw_obj_ptr = p_returns_raw_obj; }

	void set_default_arguments(const Vector<Variant> &p_defargs);

	uint32_t get_hash() const;

	// No-inline to save on binary space with LTO.
	_NO_INLINE_ MethodBind();
	virtual ~MethodBind();
};

// MethodBindVarArg.
template <typename T, typename R>
class MethodBindVarArg : public MethodBind {
	R (T::*method)(const Variant **, int, Callable::CallError &) = nullptr;
	LocalVector<PropertyInfo> arguments;

protected:
	virtual PropertyInfo _gen_argument_type_info(int p_arg) const override {
		if (p_arg < 0) {
			if constexpr (std::is_void_v<R>) {
				return PropertyInfo();
			} else {
				return GetTypeInfo<R>::get_class_info();
			}
		} else if (p_arg < (int)arguments.size()) {
			return arguments[p_arg];
		} else {
			return PropertyInfo(Variant::NIL, "arg_" + itos(p_arg), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
		}
	}

	virtual Variant::Type _gen_argument_type(int p_arg) const override {
		return _gen_argument_type_info(p_arg).type;
	}

public:
#ifdef DEBUG_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int) const override {
		return GodotTypeInfo::METADATA_NONE;
	}
#endif // DEBUG_ENABLED

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_V_MSG(p_object && p_object->is_extension_placeholder() && p_object->get_class_name() == MethodBind::get_instance_class(), Variant(), vformat("Cannot call method bind '%s' on placeholder instance.", MethodBind::get_name()));
#endif
		if constexpr (std::is_void_v<R>) {
			(static_cast<T *>(p_object)->*method)(p_args, p_arg_count, r_error);
			return Variant();
		} else {
			return (static_cast<T *>(p_object)->*method)(p_args, p_arg_count, r_error);
		}
	}

	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const override {
		ERR_FAIL_MSG("Validated call can't be used with vararg methods. This is a bug.");
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
		ERR_FAIL_MSG("ptrcall can't be used with vararg methods. This is a bug.");
	}

	virtual bool is_vararg() const override { return true; }

	MethodBindVarArg(R (T::*p_method)(const Variant **, int, Callable::CallError &), const MethodInfo &p_info) :
			method(p_method), arguments(p_info.arguments) {
		set_argument_count(arguments.size());
		_generate_argument_types(arguments.size());
		_set_returns(!std::is_void_v<R>);

#ifdef DEBUG_ENABLED
		if (arguments.size()) {
			Vector<StringName> names;
			names.resize(arguments.size());
			for (uint32_t i = 0; i < arguments.size(); ++i) {
				names.write[i] = arguments[i].name;
			}
			set_argument_names(names);
		}
#endif // DEBUG_ENABLED
	}
};

template <typename T, typename R>
MethodBind *create_vararg_method_bind(R (T::*p_method)(const Variant **, int, Callable::CallError &), const MethodInfo &p_info) {
	MethodBind *a = memnew((MethodBindVarArg<T, R>)(p_method, p_info));
	a->set_instance_class(T::get_class_static());
	return a;
}
