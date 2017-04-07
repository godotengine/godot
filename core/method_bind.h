/*************************************************************************/
/*  method_bind.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "list.h"
#include "method_ptrcall.h"
#include "object.h"
#include "variant.h"
#include <stdio.h>

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#ifdef DEBUG_ENABLED
#define DEBUG_METHODS_ENABLED
#endif

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

//SIMPLE_NUMERIC_TYPE is used to avoid a warning on Variant::get_type_for

#ifdef PTRCALL_ENABLED

#define VARIANT_ENUM_CAST(m_enum)                                            \
	SIMPLE_NUMERIC_TYPE(m_enum);                                             \
	template <>                                                              \
	struct VariantCaster<m_enum> {                                           \
                                                                             \
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
	SIMPLE_NUMERIC_TYPE(m_enum);                                      \
	template <>                                                       \
	struct VariantCaster<m_enum> {                                    \
                                                                      \
		static _FORCE_INLINE_ m_enum cast(const Variant &p_variant) { \
			return (m_enum)p_variant.operator int();                  \
		}                                                             \
	};

#endif

#define CHECK_ARG(m_arg)                                                            \
	if ((m_arg - 1) < p_arg_count) {                                                \
		Variant::Type argtype = get_argument_type(m_arg - 1);                       \
		if (!Variant::can_convert_strict(p_args[m_arg - 1]->get_type(), argtype)) { \
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;        \
			r_error.argument = m_arg - 1;                                           \
			r_error.expected = argtype;                                             \
			return Variant();                                                       \
		}                                                                           \
	}

#define CHECK_NOARG(m_arg)                             \
	{                                                  \
		if (p_arg##m_arg.get_type() != Variant::NIL) { \
			if (r_argerror) *r_argerror = (m_arg - 1); \
			return CALL_ERROR_EXTRA_ARGUMENT;          \
		}                                              \
	}

// some helpers

VARIANT_ENUM_CAST(Vector3::Axis);
VARIANT_ENUM_CAST(Image::Format);
VARIANT_ENUM_CAST(Error);
VARIANT_ENUM_CAST(wchar_t);
VARIANT_ENUM_CAST(Margin);
VARIANT_ENUM_CAST(Orientation);
VARIANT_ENUM_CAST(HAlign);
VARIANT_ENUM_CAST(Variant::Type);
VARIANT_ENUM_CAST(Variant::Operator);
VARIANT_ENUM_CAST(InputEvent::Type);

class MethodBind {

	int method_id;
	uint32_t hint_flags;
	StringName name;
	Vector<Variant> default_arguments;
	int default_argument_count;
	int argument_count;
#ifdef DEBUG_METHODS_ENABLED
	Vector<StringName> arg_names;
	Variant::Type *argument_types;
	StringName ret_type;
#endif
	bool _const;
	bool _returns;

protected:
	void _set_const(bool p_const);
	void _set_returns(bool p_returns);
#ifdef DEBUG_METHODS_ENABLED
	virtual Variant::Type _gen_argument_type(int p_arg) const = 0;
	void _generate_argument_types(int p_count);
	void set_argument_types(Variant::Type *p_types) { argument_types = p_types; }
#endif
	void set_argument_count(int p_count) { argument_count = p_count; }

public:
	Vector<Variant> get_default_arguments() const { return default_arguments; }
	_FORCE_INLINE_ int get_default_argument_count() const { return default_argument_count; }

	_FORCE_INLINE_ Variant has_default_argument(int p_arg) const {

		int idx = argument_count - p_arg - 1;

		if (idx < 0 || idx >= default_arguments.size())
			return false;
		else
			return true;
	}

	_FORCE_INLINE_ Variant get_default_argument(int p_arg) const {

		int idx = argument_count - p_arg - 1;

		if (idx < 0 || idx >= default_arguments.size())
			return Variant();
		else
			return default_arguments[idx];
	}

#ifdef DEBUG_METHODS_ENABLED

	_FORCE_INLINE_ void set_return_type(const StringName &p_type) { ret_type = p_type; }
	_FORCE_INLINE_ StringName get_return_type() const { return ret_type; }

	_FORCE_INLINE_ Variant::Type get_argument_type(int p_argument) const {

		ERR_FAIL_COND_V(p_argument < -1 || p_argument > argument_count, Variant::NIL);
		return argument_types[p_argument + 1];
	}

	PropertyInfo get_argument_info(int p_argument) const;

	void set_argument_names(const Vector<StringName> &p_names);
	Vector<StringName> get_argument_names() const;
#endif
	void set_hint_flags(uint32_t p_hint) { hint_flags = p_hint; }
	uint32_t get_hint_flags() const { return hint_flags | (is_const() ? METHOD_FLAG_CONST : 0) | (is_vararg() ? METHOD_FLAG_VARARG : 0); }
	virtual String get_instance_class() const = 0;

	_FORCE_INLINE_ int get_argument_count() const { return argument_count; };

#if 0
	_FORCE_INLINE_ Variant call_safe(const Variant** p_args,int p_arg_count, Variant::CallError& r_error) {

		r_error.error=Variant::CallError::CALL_OK;
		check_call( p_args, &errorarg );
		if (!err)
			return call(p_object, VARIANT_ARG_PASS );

		VARIANT_ARGPTRS
		String errstr;
		String methodname = get_instance_type()+"::"+name;
		if (err==CALL_ERROR_ARGUMENT_TYPE) {
			errstr="Invalid Argument to call: '"+methodname+"'. Cannot convert argument "+itos(errorarg+1)+" from "+Variant::get_type_name(get_argument_type(errorarg))+" to "+Variant::get_type_name(argptr[errorarg]->get_type())+".";
		}
		if (err==CALL_ERROR_EXTRA_ARGUMENT) {
			errstr="Invalid call. Member function '"+methodname+"' takes "+itos(get_argument_count())+" argument, but argument "+itos(errorarg+1)+" was received.";
		}

		ERR_PRINT(errstr.ascii().get_data());
		return Variant();
	}
#endif
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

public:
	virtual Variant::Type _gen_argument_type(int p_arg) const {

		return Variant::NIL;
	}

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Variant::CallError &r_error) {

		T *instance = static_cast<T *>(p_object);
		return (instance->*call_method)(p_args, p_arg_count, r_error);
	}
	void set_method_info(const MethodInfo &p_info) {

		set_argument_count(p_info.arguments.size());
#ifdef DEBUG_METHODS_ENABLED
		Variant::Type *at = memnew_arr(Variant::Type, p_info.arguments.size() + 1);
		at[0] = p_info.return_val.type;
		if (p_info.arguments.size()) {

			Vector<StringName> names;
			names.resize(p_info.arguments.size());
			for (int i = 0; i < p_info.arguments.size(); i++) {

				at[i + 1] = p_info.arguments[i].type;
				names[i] = p_info.arguments[i].name;
			}

			set_argument_names(names);
		}
		set_argument_types(at);
#endif
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
		call_method = NULL;
		_set_returns(true);
	}
};

template <class T>
MethodBind *create_vararg_method_bind(Variant (T::*p_method)(const Variant **, int, Variant::CallError &), const MethodInfo &p_info) {

	MethodBindVarArg<T> *a = memnew((MethodBindVarArg<T>));
	a->set_method(p_method);
	a->set_method_info(p_info);
	return a;
}

/** This amazing hack is based on the FastDelegates theory */

// tale of an amazing hack.. //

// if you declare an nonexistent class..
class __UnexistingClass;

#include "method_bind.inc"

#endif
