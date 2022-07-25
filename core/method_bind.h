/*************************************************************************/
/*  method_bind.h                                                        */
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

#ifndef METHOD_BIND_H
#define METHOD_BIND_H

#include "core/list.h"
#include "core/method_ptrcall.h"
#include "core/object.h"
#include "core/variant.h"

#include <stdio.h>

#ifdef DEBUG_ENABLED
#define DEBUG_METHODS_ENABLED
#endif

#include "core/type_info.h"

enum MethodFlags {

	METHOD_FLAG_NORMAL = 1,
	METHOD_FLAG_EDITOR = 2,
	METHOD_FLAG_NOSCRIPT = 4,
	METHOD_FLAG_CONST = 8,
	METHOD_FLAG_REVERSE = 16, // used for events
	METHOD_FLAG_VIRTUAL = 32,
	METHOD_FLAG_FROM_SCRIPT = 64,
	METHOD_FLAG_VARARG = 128,
	METHOD_FLAGS_DEFAULT = METHOD_FLAG_NORMAL,
};

template <class T>
struct VariantCaster {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		return p_variant;
	}
};

template <class T>
struct VariantCaster<T &> {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		return p_variant;
	}
};

template <class T>
struct VariantCaster<const T &> {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		return p_variant;
	}
};

#define _VC(m_idx) \
	(VariantCaster<P##m_idx>::cast((m_idx - 1) >= p_arg_count ? get_default_argument(m_idx - 1) : *p_args[m_idx - 1]))

#ifdef PTRCALL_ENABLED

#define VARIANT_ENUM_CAST(m_enum)                                            \
	MAKE_ENUM_TYPE_INFO(m_enum)                                              \
	template <>                                                              \
	struct VariantCaster<m_enum> {                                           \
		static _FORCE_INLINE_ m_enum cast(const Variant &p_variant) {        \
			return (m_enum)p_variant.operator int();                         \
		}                                                                    \
	};                                                                       \
	template <>                                                              \
	struct PtrToArg<m_enum> {                                                \
		_FORCE_INLINE_ static m_enum convert(const void *p_ptr) {            \
			return m_enum(*reinterpret_cast<const int *>(p_ptr));            \
		}                                                                    \
		_FORCE_INLINE_ static void encode(m_enum p_val, const void *p_ptr) { \
			*(int *)p_ptr = p_val;                                           \
		}                                                                    \
	};

#else

#define VARIANT_ENUM_CAST(m_enum)                                     \
	MAKE_ENUM_TYPE_INFO(m_enum)                                       \
	template <>                                                       \
	struct VariantCaster<m_enum> {                                    \
		static _FORCE_INLINE_ m_enum cast(const Variant &p_variant) { \
			return (m_enum)p_variant.operator int();                  \
		}                                                             \
	};

#endif

// Object enum casts must go here
VARIANT_ENUM_CAST(Object::ConnectFlags);

template <typename T>
struct VariantObjectClassChecker {
	static _FORCE_INLINE_ bool check(const Variant &p_variant) {
		return true;
	}
};

template <>
struct VariantObjectClassChecker<Node *> {
	static _FORCE_INLINE_ bool check(const Variant &p_variant) {
		Object *obj = p_variant;
		Node *node = p_variant;
		return node || !obj;
	}
};

template <>
struct VariantObjectClassChecker<Control *> {
	static _FORCE_INLINE_ bool check(const Variant &p_variant) {
		Object *obj = p_variant;
		Control *control = p_variant;
		return control || !obj;
	}
};

#define CHECK_ARG(m_arg)                                                            \
	if ((m_arg - 1) < p_arg_count) {                                                \
		Variant::Type argtype = get_argument_type(m_arg - 1);                       \
		if (!Variant::can_convert_strict(p_args[m_arg - 1]->get_type(), argtype) || \
				!VariantObjectClassChecker<P##m_arg>::check(*p_args[m_arg - 1])) {  \
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;        \
			r_error.argument = m_arg - 1;                                           \
			r_error.expected = argtype;                                             \
			return Variant();                                                       \
		}                                                                           \
	}

#define CHECK_NOARG(m_arg)                             \
	{                                                  \
		if (p_arg##m_arg.get_type() != Variant::NIL) { \
			if (r_argerror)                            \
				*r_argerror = (m_arg - 1);             \
			return CALL_ERROR_EXTRA_ARGUMENT;          \
		}                                              \
	}

// some helpers

VARIANT_ENUM_CAST(Vector3::Axis);

VARIANT_ENUM_CAST(Error);
VARIANT_ENUM_CAST(Margin);
VARIANT_ENUM_CAST(Corner);
VARIANT_ENUM_CAST(Orientation);
VARIANT_ENUM_CAST(HAlign);
VARIANT_ENUM_CAST(VAlign);
VARIANT_ENUM_CAST(PropertyHint);
VARIANT_ENUM_CAST(PropertyUsageFlags);
VARIANT_ENUM_CAST(MethodFlags);
VARIANT_ENUM_CAST(Variant::Type);
VARIANT_ENUM_CAST(Variant::Operator);

template <>
struct VariantCaster<wchar_t> {
	static _FORCE_INLINE_ wchar_t cast(const Variant &p_variant) {
		return (wchar_t)p_variant.operator int();
	}
};
#ifdef PTRCALL_ENABLED
template <>
struct PtrToArg<wchar_t> {
	_FORCE_INLINE_ static wchar_t convert(const void *p_ptr) {
		return wchar_t(*reinterpret_cast<const int *>(p_ptr));
	}
	_FORCE_INLINE_ static void encode(wchar_t p_val, const void *p_ptr) {
		*(int *)p_ptr = p_val;
	}
};
#endif

class MethodBind {
	int method_id;
	uint32_t hint_flags;
	StringName name;
	Vector<Variant> default_arguments;
	int default_argument_count;
	int argument_count;

	bool _const;
	bool _returns;

protected:
	Variant::Type *argument_types;
#ifdef DEBUG_METHODS_ENABLED
	Vector<StringName> arg_names;
#endif
	void _set_const(bool p_const);
	void _set_returns(bool p_returns);
	virtual Variant::Type _gen_argument_type(int p_arg) const = 0;
	virtual PropertyInfo _gen_argument_type_info(int p_arg) const = 0;
	void _generate_argument_types(int p_count);

	void set_argument_count(int p_count) { argument_count = p_count; }

public:
	Vector<Variant> get_default_arguments() const { return default_arguments; }
	_FORCE_INLINE_ int get_default_argument_count() const { return default_argument_count; }

	_FORCE_INLINE_ Variant has_default_argument(int p_arg) const {
		int idx = argument_count - p_arg - 1;

		if (idx < 0 || idx >= default_arguments.size()) {
			return false;
		} else {
			return true;
		}
	}

	_FORCE_INLINE_ Variant get_default_argument(int p_arg) const {
		int idx = argument_count - p_arg - 1;

		if (idx < 0 || idx >= default_arguments.size()) {
			return Variant();
		} else {
			return default_arguments[idx];
		}
	}

	_FORCE_INLINE_ Variant::Type get_argument_type(int p_argument) const {
		ERR_FAIL_COND_V(p_argument < -1 || p_argument > argument_count, Variant::NIL);
		return argument_types[p_argument + 1];
	}

	PropertyInfo get_argument_info(int p_argument) const;
	PropertyInfo get_return_info() const;

#ifdef DEBUG_METHODS_ENABLED
	void set_argument_names(const Vector<StringName> &p_names); //set by class, db, can't be inferred otherwise
	Vector<StringName> get_argument_names() const;

	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const = 0;

#endif
	void set_hint_flags(uint32_t p_hint) { hint_flags = p_hint; }
	uint32_t get_hint_flags() const { return hint_flags | (is_const() ? METHOD_FLAG_CONST : 0) | (is_vararg() ? METHOD_FLAG_VARARG : 0); }
	virtual String get_instance_class() const = 0;

	_FORCE_INLINE_ int get_argument_count() const { return argument_count; };

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Variant::CallError &r_error) = 0;

#ifdef PTRCALL_ENABLED
	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) = 0;
#endif

	StringName get_name() const;
	void set_name(const StringName &p_name);
	_FORCE_INLINE_ int get_method_id() const { return method_id; }
	_FORCE_INLINE_ bool is_const() const { return _const; }
	_FORCE_INLINE_ bool has_return() const { return _returns; }
	virtual bool is_vararg() const { return false; }

	void set_default_arguments(const Vector<Variant> &p_defargs);

	MethodBind();
	virtual ~MethodBind();
};

template <class T>
class MethodBindVarArg : public MethodBind {
public:
	typedef Variant (T::*NativeCall)(const Variant **, int, Variant::CallError &);

protected:
	NativeCall call_method;
	MethodInfo arguments;

public:
	virtual PropertyInfo _gen_argument_type_info(int p_arg) const {
		if (p_arg < 0) {
			return arguments.return_val;
		} else if (p_arg < arguments.arguments.size()) {
			return arguments.arguments[p_arg];
		} else {
			return PropertyInfo(Variant::NIL, "arg_" + itos(p_arg), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
		}
	}

	virtual Variant::Type _gen_argument_type(int p_arg) const {
		return _gen_argument_type_info(p_arg).type;
	}

#ifdef DEBUG_METHODS_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int) const {
		return GodotTypeInfo::METADATA_NONE;
	}
#endif

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Variant::CallError &r_error) {
		T *instance = static_cast<T *>(p_object);
		return (instance->*call_method)(p_args, p_arg_count, r_error);
	}

	void set_method_info(const MethodInfo &p_info, bool p_return_nil_is_variant) {
		set_argument_count(p_info.arguments.size());
		Variant::Type *at = memnew_arr(Variant::Type, p_info.arguments.size() + 1);
		at[0] = p_info.return_val.type;
		if (p_info.arguments.size()) {
#ifdef DEBUG_METHODS_ENABLED
			Vector<StringName> names;
			names.resize(p_info.arguments.size());
#endif
			for (int i = 0; i < p_info.arguments.size(); i++) {
				at[i + 1] = p_info.arguments[i].type;
#ifdef DEBUG_METHODS_ENABLED
				names.write[i] = p_info.arguments[i].name;
#endif
			}

#ifdef DEBUG_METHODS_ENABLED
			set_argument_names(names);
#endif
		}
		argument_types = at;
		arguments = p_info;
		if (p_return_nil_is_variant) {
			arguments.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		}
	}

#ifdef PTRCALL_ENABLED
	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) {
		ERR_FAIL(); //can't call
	} //todo
#endif

	void set_method(NativeCall p_method) { call_method = p_method; }
	virtual bool is_const() const { return false; }
	virtual String get_instance_class() const { return T::get_class_static(); }

	virtual bool is_vararg() const { return true; }

	MethodBindVarArg() {
		call_method = nullptr;
		_set_returns(true);
	}
};

template <class T>
MethodBind *create_vararg_method_bind(Variant (T::*p_method)(const Variant **, int, Variant::CallError &), const MethodInfo &p_info, bool p_return_nil_is_variant) {
	MethodBindVarArg<T> *a = memnew((MethodBindVarArg<T>));
	a->set_method(p_method);
	a->set_method_info(p_info, p_return_nil_is_variant);
	return a;
}

/** This amazing hack is based on the FastDelegates theory */

// tale of an amazing hack.. //

// if you declare a nonexistent class..
class __UnexistingClass;

#include "method_bind.gen.inc"

#endif // METHOD_BIND_H
