/*************************************************************************/
/*  variant_call.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "variant.h"

#include "core/class_db.h"
#include "core/color_names.inc"
#include "core/core_string_names.h"
#include "core/crypto/crypto_core.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/compression.h"
#include "core/oa_hash_map.h"
#include "core/os/os.h"

_FORCE_INLINE_ void sarray_add_str(Vector<String> &arr) {
}

_FORCE_INLINE_ void sarray_add_str(Vector<String> &arr, const String &p_str) {
	arr.push_back(p_str);
}

template <class... P>
_FORCE_INLINE_ void sarray_add_str(Vector<String> &arr, const String &p_str, P... p_args) {
	arr.push_back(p_str);
	sarray_add_str(arr, p_args...);
}

template <class... P>
_FORCE_INLINE_ Vector<String> sarray(P... p_args) {
	Vector<String> arr;
	sarray_add_str(arr, p_args...);
	return arr;
}

typedef void (*VariantFunc)(Variant &r_ret, Variant &p_self, const Variant **p_args);
typedef void (*VariantConstructFunc)(Variant &r_ret, const Variant **p_args);

struct _VariantCall {
	template <class T, class... P>
	class InternalMethod : public Variant::InternalMethod {
	public:
		void (T::*method)(P...);
		Vector<Variant> default_values;
#ifdef DEBUG_ENABLED
		Vector<String> argument_names;
#endif

		virtual int get_argument_count() const {
			return sizeof...(P);
		}
		virtual Variant::Type get_argument_type(int p_arg) const {
			return call_get_argument_type<P...>(p_arg);
		}
#ifdef DEBUG_ENABLED
		virtual String get_argument_name(int p_arg) const {
			ERR_FAIL_INDEX_V(p_arg, argument_names.size(), String());
			return argument_names[p_arg];
		}
#endif
		virtual Vector<Variant> get_default_arguments() const {
			return default_values;
		}

		virtual Variant::Type get_return_type() const {
			return Variant::NIL;
		}
		virtual uint32_t get_flags() const {
			return 0;
		}

		virtual void call(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
			call_with_variant_args_dv(VariantGetInternalPtr<T>::get_ptr(base), method, p_args, p_argcount, r_error, default_values);
		}

		virtual void validated_call(Variant *base, const Variant **p_args, Variant *r_ret) {
			call_with_validated_variant_args(base, method, p_args);
		}

#ifdef PTRCALL_ENABLED
		virtual void ptrcall(void *p_base, const void **p_args, void *r_ret) {
			call_with_ptr_args<T, P...>(reinterpret_cast<T *>(p_base), method, p_args);
		}
#endif
		InternalMethod(void (T::*p_method)(P...), const Vector<Variant> &p_default_args
#ifdef DEBUG_ENABLED
				,
				const Vector<String> &p_arg_names, const StringName &p_method_name, Variant::Type p_base_type
#endif
		) {
			method = p_method;
			default_values = p_default_args;
#ifdef DEBUG_ENABLED
			argument_names = p_arg_names;
			method_name = p_method_name;
			base_type = p_base_type;
#endif
		}
	};

	template <class T, class R, class... P>
	class InternalMethodR : public Variant::InternalMethod {
	public:
		R(T::*method)
		(P...);
		Vector<Variant> default_values;
#ifdef DEBUG_ENABLED
		Vector<String> argument_names;
#endif

		virtual int get_argument_count() const {
			return sizeof...(P);
		}
		virtual Variant::Type get_argument_type(int p_arg) const {
			return call_get_argument_type<P...>(p_arg);
			return Variant::NIL;
		}
#ifdef DEBUG_ENABLED
		virtual String get_argument_name(int p_arg) const {
			ERR_FAIL_INDEX_V(p_arg, argument_names.size(), String());
			return argument_names[p_arg];
		}
#endif
		virtual Vector<Variant> get_default_arguments() const {
			return default_values;
		}

		virtual Variant::Type get_return_type() const {
			return GetTypeInfo<R>::VARIANT_TYPE;
		}
		virtual uint32_t get_flags() const {
			uint32_t f = 0;
			if (get_return_type() == Variant::NIL) {
				f |= FLAG_RETURNS_VARIANT;
			}
			return f;
		}

		virtual void call(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
			call_with_variant_args_ret_dv(VariantGetInternalPtr<T>::get_ptr(base), method, p_args, p_argcount, r_ret, r_error, default_values);
		}

		virtual void validated_call(Variant *base, const Variant **p_args, Variant *r_ret) {
			call_with_validated_variant_args_ret(base, method, p_args, r_ret);
		}
#ifdef PTRCALL_ENABLED
		virtual void ptrcall(void *p_base, const void **p_args, void *r_ret) {
			call_with_ptr_args_ret<T, R, P...>(reinterpret_cast<T *>(p_base), method, p_args, r_ret);
		}
#endif
		InternalMethodR(R (T::*p_method)(P...), const Vector<Variant> &p_default_args
#ifdef DEBUG_ENABLED
				,
				const Vector<String> &p_arg_names, const StringName &p_method_name, Variant::Type p_base_type
#endif
		) {
			method = p_method;
			default_values = p_default_args;
#ifdef DEBUG_ENABLED
			argument_names = p_arg_names;
			method_name = p_method_name;
			base_type = p_base_type;
#endif
		}
	};

	template <class T, class R, class... P>
	class InternalMethodRC : public Variant::InternalMethod {
	public:
		R(T::*method)
		(P...) const;
		Vector<Variant> default_values;
#ifdef DEBUG_ENABLED
		Vector<String> argument_names;
#endif

		virtual int get_argument_count() const {
			return sizeof...(P);
		}
		virtual Variant::Type get_argument_type(int p_arg) const {
			return call_get_argument_type<P...>(p_arg);
		}
#ifdef DEBUG_ENABLED
		virtual String get_argument_name(int p_arg) const {
			ERR_FAIL_INDEX_V(p_arg, argument_names.size(), String());
			return argument_names[p_arg];
		}
#endif
		virtual Vector<Variant> get_default_arguments() const {
			return default_values;
		}

		virtual Variant::Type get_return_type() const {
			return GetTypeInfo<R>::VARIANT_TYPE;
		}
		virtual uint32_t get_flags() const {
			uint32_t f = FLAG_IS_CONST;
			if (get_return_type() == Variant::NIL) {
				f |= FLAG_RETURNS_VARIANT;
			}
			return f;
		}

		virtual void call(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
			call_with_variant_args_retc_dv(VariantGetInternalPtr<T>::get_ptr(base), method, p_args, p_argcount, r_ret, r_error, default_values);
		}

		virtual void validated_call(Variant *base, const Variant **p_args, Variant *r_ret) {
			call_with_validated_variant_args_retc(base, method, p_args, r_ret);
		}
#ifdef PTRCALL_ENABLED
		virtual void ptrcall(void *p_base, const void **p_args, void *r_ret) {
			call_with_ptr_args_retc<T, R, P...>(reinterpret_cast<T *>(p_base), method, p_args, r_ret);
		}
#endif
		InternalMethodRC(R (T::*p_method)(P...) const, const Vector<Variant> &p_default_args
#ifdef DEBUG_ENABLED
				,
				const Vector<String> &p_arg_names, const StringName &p_method_name, Variant::Type p_base_type
#endif
		) {
			method = p_method;
			default_values = p_default_args;
#ifdef DEBUG_ENABLED
			argument_names = p_arg_names;
			method_name = p_method_name;
			base_type = p_base_type;
#endif
		}
	};

	template <class T, class R, class... P>
	class InternalMethodRS : public Variant::InternalMethod {
	public:
		R(*method)
		(T *, P...);
		Vector<Variant> default_values;
#ifdef DEBUG_ENABLED
		Vector<String> argument_names;
#endif

		virtual int get_argument_count() const {
			return sizeof...(P);
		}
		virtual Variant::Type get_argument_type(int p_arg) const {
			return call_get_argument_type<P...>(p_arg);
		}
#ifdef DEBUG_ENABLED
		virtual String get_argument_name(int p_arg) const {
			ERR_FAIL_INDEX_V(p_arg, argument_names.size(), String());
			return argument_names[p_arg];
		}
#endif
		virtual Vector<Variant> get_default_arguments() const {
			return default_values;
		}

		virtual Variant::Type get_return_type() const {
			return GetTypeInfo<R>::VARIANT_TYPE;
		}
		virtual uint32_t get_flags() const {
			uint32_t f = 0;
			if (get_return_type() == Variant::NIL) {
				f |= FLAG_RETURNS_VARIANT;
			}
			return f;
		}

		virtual void call(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
			const Variant **args = p_args;
#ifdef DEBUG_ENABLED
			if ((size_t)p_argcount > sizeof...(P)) {
				r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
				r_error.argument = sizeof...(P);
				return;
			}
#endif
			if ((size_t)p_argcount < sizeof...(P)) {
				size_t missing = sizeof...(P) - (size_t)p_argcount;
				if (missing <= (size_t)default_values.size()) {
					args = (const Variant **)alloca(sizeof...(P) * sizeof(const Variant *));
					// GCC fails to see that `sizeof...(P)` cannot be 0 here given the previous
					// conditions, so it raises a warning on the potential use of `i < 0` as the
					// execution condition.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
					for (size_t i = 0; i < sizeof...(P); i++) {
						if (i < (size_t)p_argcount) {
							args[i] = p_args[i];
						} else {
							args[i] = &default_values[i - p_argcount + (default_values.size() - missing)];
						}
					}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
				} else {
#ifdef DEBUG_ENABLED
					r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
					r_error.argument = sizeof...(P);
#endif
					return;
				}
			}
			call_with_variant_args_retc_static_helper(VariantGetInternalPtr<T>::get_ptr(base), method, args, r_ret, r_error, BuildIndexSequence<sizeof...(P)>{});
		}

		virtual void validated_call(Variant *base, const Variant **p_args, Variant *r_ret) {
			call_with_validated_variant_args_static_retc(base, method, p_args, r_ret);
		}
#ifdef PTRCALL_ENABLED
		virtual void ptrcall(void *p_base, const void **p_args, void *r_ret) {
			call_with_ptr_args_static_retc<T, R, P...>(reinterpret_cast<T *>(p_base), method, p_args, r_ret);
		}
#endif
		InternalMethodRS(R (*p_method)(T *, P...), const Vector<Variant> &p_default_args
#ifdef DEBUG_ENABLED
				,
				const Vector<String> &p_arg_names, const StringName &p_method_name, Variant::Type p_base_type
#endif
		) {
			method = p_method;
			default_values = p_default_args;
#ifdef DEBUG_ENABLED
			argument_names = p_arg_names;
			method_name = p_method_name;
			base_type = p_base_type;
#endif
		}
	};

	class InternalMethodVC : public Variant::InternalMethod {
	public:
		typedef void (*MethodVC)(Variant *, const Variant **, int, Variant &r_ret, Callable::CallError &);
		MethodVC methodvc = nullptr;
		uint32_t base_flags = 0;
		Vector<String> argument_names;
		Vector<Variant::Type> argument_types;
		Variant::Type return_type = Variant::NIL;

		virtual int get_argument_count() const {
			return argument_names.size();
		}
		virtual Variant::Type get_argument_type(int p_arg) const {
			ERR_FAIL_INDEX_V(p_arg, argument_types.size(), Variant::NIL);
			return argument_types[p_arg];
		}
#ifdef DEBUG_ENABLED
		virtual String get_argument_name(int p_arg) const {
			ERR_FAIL_INDEX_V(p_arg, argument_names.size(), String());
			return argument_names[p_arg];
		}
#endif
		virtual Vector<Variant> get_default_arguments() const {
			return Vector<Variant>();
		}

		virtual Variant::Type get_return_type() const {
			return return_type;
		}
		virtual uint32_t get_flags() const {
			return base_flags | FLAG_NO_PTRCALL;
		}

		virtual void call(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
			methodvc(base, p_args, p_argcount, r_ret, r_error);
		}

		virtual void validated_call(Variant *base, const Variant **p_args, Variant *r_ret) {
			ERR_FAIL_MSG("No support for validated call");
		}
#ifdef PTRCALL_ENABLED
		virtual void ptrcall(void *p_base, const void **p_args, void *r_ret) {
			ERR_FAIL_MSG("No support for ptrcall call");
		}
#endif
		InternalMethodVC(MethodVC p_method, uint32_t p_flags, const Vector<Variant::Type> &p_argument_types, const Variant::Type &p_return_type
#ifdef DEBUG_ENABLED
				,
				const Vector<String> &p_arg_names, const StringName &p_method_name, Variant::Type p_base_type
#endif
		) {
			methodvc = p_method;
			argument_types = p_argument_types;
			return_type = p_return_type;
			base_flags = p_flags;
#ifdef DEBUG_ENABLED
			argument_names = p_arg_names;
			method_name = p_method_name;
			base_type = p_base_type;
#endif
		}
	};

	typedef OAHashMap<StringName, Variant::InternalMethod *> MethodMap;
	static MethodMap *type_internal_methods;
	static List<StringName> *type_internal_method_names;

	template <class T, class... P>
	static void _bind_method(const StringName &p_name, void (T::*p_method)(P...), const Vector<Variant> &p_default_args = Vector<Variant>()
#ifdef DEBUG_ENABLED
																						  ,
			const Vector<String> &p_argument_names = Vector<String>()
#endif
	) {

#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_MSG(p_argument_names.size() != sizeof...(P), "Wrong argument name count supplied for method:  " + Variant::get_type_name(GetTypeInfo<T>::VARIANT_TYPE) + "::" + String(p_name));
		ERR_FAIL_COND(type_internal_methods[GetTypeInfo<T>::VARIANT_TYPE].has(p_name));
#endif
#ifdef DEBUG_ENABLED
		Variant::InternalMethod *m = memnew((InternalMethod<T, P...>)(p_method, p_default_args, p_argument_names, p_name, GetTypeInfo<T>::VARIANT_TYPE));
#else
		Variant::InternalMethod *m = memnew((InternalMethod<T, P...>)(p_method, p_default_args));
#endif

		type_internal_methods[GetTypeInfo<T>::VARIANT_TYPE].insert(p_name, m);
		type_internal_method_names[GetTypeInfo<T>::VARIANT_TYPE].push_back(p_name);
	}

	template <class T, class R, class... P>
	static void _bind_method(const StringName &p_name, R (T::*p_method)(P...) const, const Vector<Variant> &p_default_args = Vector<Variant>()
#ifdef DEBUG_ENABLED
																							 ,
			const Vector<String> &p_argument_names = Vector<String>()
#endif
	) {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_MSG(p_argument_names.size() != sizeof...(P), "Wrong argument name count supplied for method:  " + Variant::get_type_name(GetTypeInfo<T>::VARIANT_TYPE) + "::" + String(p_name));
		ERR_FAIL_COND_MSG(type_internal_methods[GetTypeInfo<T>::VARIANT_TYPE].has(p_name), " Method already registered: " + Variant::get_type_name(GetTypeInfo<T>::VARIANT_TYPE) + "::" + String(p_name));

#endif
#ifdef DEBUG_ENABLED
		Variant::InternalMethod *m = memnew((InternalMethodRC<T, R, P...>)(p_method, p_default_args, p_argument_names, p_name, GetTypeInfo<T>::VARIANT_TYPE));
#else
		Variant::InternalMethod *m = memnew((InternalMethodRC<T, R, P...>)(p_method, p_default_args));
#endif

		type_internal_methods[GetTypeInfo<T>::VARIANT_TYPE].insert(p_name, m);
		type_internal_method_names[GetTypeInfo<T>::VARIANT_TYPE].push_back(p_name);
	}

	template <class T, class R, class... P>
	static void _bind_method(const StringName &p_name, R (T::*p_method)(P...), const Vector<Variant> &p_default_args = Vector<Variant>()
#ifdef DEBUG_ENABLED
																					   ,
			const Vector<String> &p_argument_names = Vector<String>()
#endif
	) {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_MSG(p_argument_names.size() != sizeof...(P), "Wrong argument name count supplied for method:  " + Variant::get_type_name(GetTypeInfo<T>::VARIANT_TYPE) + "::" + String(p_name));
		ERR_FAIL_COND_MSG(type_internal_methods[GetTypeInfo<T>::VARIANT_TYPE].has(p_name), " Method already registered: " + Variant::get_type_name(GetTypeInfo<T>::VARIANT_TYPE) + "::" + String(p_name));
#endif

#ifdef DEBUG_ENABLED
		Variant::InternalMethod *m = memnew((InternalMethodR<T, R, P...>)(p_method, p_default_args, p_argument_names, p_name, GetTypeInfo<T>::VARIANT_TYPE));
#else
		Variant::InternalMethod *m = memnew((InternalMethodR<T, R, P...>)(p_method, p_default_args));
#endif
		type_internal_methods[GetTypeInfo<T>::VARIANT_TYPE].insert(p_name, m);
		type_internal_method_names[GetTypeInfo<T>::VARIANT_TYPE].push_back(p_name);
	}

#ifdef DEBUG_ENABLED
#define bind_method(m_type, m_method, m_arg_names, m_default_args) _VariantCall::_bind_method(#m_method, &m_type ::m_method, m_default_args, m_arg_names)
#else
#define bind_method(m_type, m_method, m_arg_names, m_default_args) _VariantCall::_bind_method(#m_method, &m_type ::m_method, m_default_args)
#endif

#ifdef DEBUG_ENABLED
#define bind_methodv(m_name, m_method, m_arg_names, m_default_args) _VariantCall::_bind_method(#m_name, m_method, m_default_args, m_arg_names)
#else
#define bind_methodv(m_name, m_method, m_arg_names, m_default_args) _VariantCall::_bind_method(#m_name, m_method, m_default_args)
#endif

	template <class T, class R, class... P>
	static void _bind_function(const StringName &p_name, R (*p_method)(T *, P...), const Vector<Variant> &p_default_args = Vector<Variant>()
#ifdef DEBUG_ENABLED
																						   ,
			const Vector<String> &p_argument_names = Vector<String>()
#endif
	) {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_MSG(p_argument_names.size() != sizeof...(P), "Wrong argument name count supplied for method:  " + Variant::get_type_name(GetTypeInfo<T>::VARIANT_TYPE) + "::" + String(p_name));
		ERR_FAIL_COND_MSG(type_internal_methods[GetTypeInfo<T>::VARIANT_TYPE].has(p_name), " Method already registered: " + Variant::get_type_name(GetTypeInfo<T>::VARIANT_TYPE) + "::" + String(p_name));
#endif

#ifdef DEBUG_ENABLED
		Variant::InternalMethod *m = memnew((InternalMethodRS<T, R, P...>)(p_method, p_default_args, p_argument_names, p_name, GetTypeInfo<T>::VARIANT_TYPE));
#else
		Variant::InternalMethod *m = memnew((InternalMethodRS<T, R, P...>)(p_method, p_default_args));
#endif

		type_internal_methods[GetTypeInfo<T>::VARIANT_TYPE].insert(p_name, m);
		type_internal_method_names[GetTypeInfo<T>::VARIANT_TYPE].push_back(p_name);
	}

#ifdef DEBUG_ENABLED
#define bind_function(m_name, m_method, m_arg_names, m_default_args) _VariantCall::_bind_function(m_name, m_method, m_default_args, m_arg_names)
#else
#define bind_function(m_name, m_method, m_arg_names, m_default_args) _VariantCall::_bind_function(m_name, m_method, m_default_args)
#endif

	static void _bind_custom(Variant::Type p_type, const StringName &p_name, InternalMethodVC::MethodVC p_method, uint32_t p_flags, const Vector<Variant::Type> &p_argument_types, const Variant::Type &p_return_type
#ifdef DEBUG_ENABLED
			,
			const Vector<String> &p_argument_names = Vector<String>()
#endif
	) {

#ifdef DEBUG_ENABLED
		Variant::InternalMethod *m = memnew(InternalMethodVC(p_method, p_flags, p_argument_types, p_return_type, p_argument_names, p_name, p_type));
#else
		Variant::InternalMethod *m = memnew(InternalMethodVC(p_method, p_flags, p_argument_types, p_return_type));
#endif

		type_internal_methods[p_type].insert(p_name, m);
		type_internal_method_names[p_type].push_back(p_name);
	}

#ifdef DEBUG_ENABLED
#define bind_custom(m_type, m_name, m_method, m_flags, m_arg_types, m_ret_type, m_arg_names) _VariantCall::_bind_custom(m_type, m_name, m_method, m_flags, m_arg_types, m_ret_type, m_arg_names)
#else
#define bind_custom(m_type, m_name, m_method, m_flags, m_arg_types, m_ret_type, m_arg_names) _VariantCall::_bind_custom(m_type, m_name, m_method, m_flags, m_arg_types, m_ret_type)
#endif

	static String func_PackedByteArray_get_string_from_ascii(PackedByteArray *p_instance) {
		String s;
		if (p_instance->size() > 0) {
			const uint8_t *r = p_instance->ptr();
			CharString cs;
			cs.resize(p_instance->size() + 1);
			copymem(cs.ptrw(), r, p_instance->size());
			cs[p_instance->size()] = 0;

			s = cs.get_data();
		}
		return s;
	}

	static String func_PackedByteArray_get_string_from_utf8(PackedByteArray *p_instance) {
		String s;
		if (p_instance->size() > 0) {
			const uint8_t *r = p_instance->ptr();
			s.parse_utf8((const char *)r, p_instance->size());
		}
		return s;
	}

	static String func_PackedByteArray_get_string_from_utf16(PackedByteArray *p_instance) {
		String s;
		if (p_instance->size() > 0) {
			const uint8_t *r = p_instance->ptr();
			s.parse_utf16((const char16_t *)r, p_instance->size() / 2);
		}
		return s;
	}

	static String func_PackedByteArray_get_string_from_utf32(PackedByteArray *p_instance) {
		String s;
		if (p_instance->size() > 0) {
			const uint8_t *r = p_instance->ptr();
			s = String((const char32_t *)r, p_instance->size() / 4);
		}
		return s;
	}

	static PackedByteArray func_PackedByteArray_compress(PackedByteArray *p_instance, int p_mode) {
		PackedByteArray compressed;

		if (p_instance->size() > 0) {
			Compression::Mode mode = (Compression::Mode)(p_mode);
			compressed.resize(Compression::get_max_compressed_buffer_size(p_instance->size(), mode));
			int result = Compression::compress(compressed.ptrw(), p_instance->ptr(), p_instance->size(), mode);

			result = result >= 0 ? result : 0;
			compressed.resize(result);
		}

		return compressed;
	}

	static PackedByteArray func_PackedByteArray_decompress(PackedByteArray *p_instance, int64_t p_buffer_size, int p_mode) {
		PackedByteArray decompressed;
		Compression::Mode mode = (Compression::Mode)(p_mode);

		int64_t buffer_size = p_buffer_size;

		if (buffer_size <= 0) {
			ERR_FAIL_V_MSG(decompressed, "Decompression buffer size must be greater than zero.");
		}

		decompressed.resize(buffer_size);
		int result = Compression::decompress(decompressed.ptrw(), buffer_size, p_instance->ptr(), p_instance->size(), mode);

		result = result >= 0 ? result : 0;
		decompressed.resize(result);

		return decompressed;
	}

	static PackedByteArray func_PackedByteArray_decompress_dynamic(PackedByteArray *p_instance, int64_t p_buffer_size, int p_mode) {
		PackedByteArray decompressed;
		int64_t max_output_size = p_buffer_size;
		Compression::Mode mode = (Compression::Mode)(p_mode);

		int result = Compression::decompress_dynamic(&decompressed, max_output_size, p_instance->ptr(), p_instance->size(), mode);

		if (result == OK) {
			return decompressed;
		} else {
			decompressed.clear();
			ERR_FAIL_V_MSG(decompressed, "Decompression failed.");
		}
	}

	static String func_PackedByteArray_hex_encode(PackedByteArray *p_instance) {
		if (p_instance->size() == 0) {
			return String();
		}
		const uint8_t *r = p_instance->ptr();
		String s = String::hex_encode_buffer(&r[0], p_instance->size());
		return s;
	}

	static void func_Callable_call(Variant *v, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
		Callable *callable = VariantGetInternalPtr<Callable>::get_ptr(v);
		callable->call(p_args, p_argcount, r_ret, r_error);
	}

	static void func_Callable_call_deferred(Variant *v, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
		Callable *callable = VariantGetInternalPtr<Callable>::get_ptr(v);
		callable->call_deferred(p_args, p_argcount);
	}

	static void func_Callable_bind(Variant *v, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
		Callable *callable = VariantGetInternalPtr<Callable>::get_ptr(v);
		r_ret = callable->bind(p_args, p_argcount);
	}

	static void func_Signal_emit(Variant *v, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
		Signal *signal = VariantGetInternalPtr<Signal>::get_ptr(v);
		signal->emit(p_args, p_argcount);
	}

	struct ConstructData {
		int arg_count;
		Vector<Variant::Type> arg_types;
		Vector<String> arg_names;
		VariantConstructFunc func;
	};

	struct ConstructFunc {
		List<ConstructData> constructors;
	};

	static ConstructFunc *construct_funcs;

	static void Vector2_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Vector2(*p_args[0], *p_args[1]);
	}

	static void Vector2i_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Vector2i(*p_args[0], *p_args[1]);
	}

	static void Rect2_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Rect2(*p_args[0], *p_args[1]);
	}

	static void Rect2_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Rect2(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Rect2i_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Rect2i(*p_args[0], *p_args[1]);
	}

	static void Rect2i_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Rect2i(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Transform2D_init2(Variant &r_ret, const Variant **p_args) {
		Transform2D m(*p_args[0], *p_args[1]);
		r_ret = m;
	}

	static void Transform2D_init3(Variant &r_ret, const Variant **p_args) {
		Transform2D m;
		m[0] = *p_args[0];
		m[1] = *p_args[1];
		m[2] = *p_args[2];
		r_ret = m;
	}

	static void Vector3_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Vector3(*p_args[0], *p_args[1], *p_args[2]);
	}

	static void Vector3i_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Vector3i(*p_args[0], *p_args[1], *p_args[2]);
	}

	static void Plane_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Plane(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Plane_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Plane(*p_args[0], *p_args[1], *p_args[2]);
	}

	static void Plane_init3(Variant &r_ret, const Variant **p_args) {
		r_ret = Plane(p_args[0]->operator Vector3(), p_args[1]->operator real_t());
	}
	static void Plane_init4(Variant &r_ret, const Variant **p_args) {
		r_ret = Plane(p_args[0]->operator Vector3(), p_args[1]->operator Vector3());
	}

	static void Quat_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Quat(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Quat_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Quat(((Vector3)(*p_args[0])), ((real_t)(*p_args[1])));
	}

	static void Quat_init3(Variant &r_ret, const Variant **p_args) {
		r_ret = Quat(((Vector3)(*p_args[0])));
	}

	static void Color_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = Color(*p_args[0], *p_args[1], *p_args[2], *p_args[3]);
	}

	static void Color_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Color(*p_args[0], *p_args[1], *p_args[2]);
	}

	static void Color_init3(Variant &r_ret, const Variant **p_args) {
		r_ret = Color::html(*p_args[0]);
	}

	static void Color_init4(Variant &r_ret, const Variant **p_args) {
		r_ret = Color::hex(*p_args[0]);
	}

	static void Color_init5(Variant &r_ret, const Variant **p_args) {
		r_ret = Color(((Color)(*p_args[0])), *p_args[1]);
	}

	static void AABB_init1(Variant &r_ret, const Variant **p_args) {
		r_ret = ::AABB(*p_args[0], *p_args[1]);
	}

	static void Basis_init1(Variant &r_ret, const Variant **p_args) {
		Basis m;
		m.set_axis(0, *p_args[0]);
		m.set_axis(1, *p_args[1]);
		m.set_axis(2, *p_args[2]);
		r_ret = m;
	}

	static void Basis_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Basis(p_args[0]->operator Vector3(), p_args[1]->operator real_t());
	}

	static void Transform_init1(Variant &r_ret, const Variant **p_args) {
		Transform t;
		t.basis.set_axis(0, *p_args[0]);
		t.basis.set_axis(1, *p_args[1]);
		t.basis.set_axis(2, *p_args[2]);
		t.origin = *p_args[3];
		r_ret = t;
	}

	static void Transform_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Transform(p_args[0]->operator Basis(), p_args[1]->operator Vector3());
	}

	static void Callable_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Callable(p_args[0]->operator ObjectID(), p_args[1]->operator String());
	}

	static void Signal_init2(Variant &r_ret, const Variant **p_args) {
		r_ret = Signal(p_args[0]->operator ObjectID(), p_args[1]->operator String());
	}

	static void add_constructor(VariantConstructFunc p_func, const Variant::Type p_type,
			const String &p_name1 = "", const Variant::Type p_type1 = Variant::NIL,
			const String &p_name2 = "", const Variant::Type p_type2 = Variant::NIL,
			const String &p_name3 = "", const Variant::Type p_type3 = Variant::NIL,
			const String &p_name4 = "", const Variant::Type p_type4 = Variant::NIL) {
		ConstructData cd;
		cd.func = p_func;
		cd.arg_count = 0;

		if (p_name1 == "") {
			goto end;
		}
		cd.arg_count++;
		cd.arg_names.push_back(p_name1);
		cd.arg_types.push_back(p_type1);

		if (p_name2 == "") {
			goto end;
		}
		cd.arg_count++;
		cd.arg_names.push_back(p_name2);
		cd.arg_types.push_back(p_type2);

		if (p_name3 == "") {
			goto end;
		}
		cd.arg_count++;
		cd.arg_names.push_back(p_name3);
		cd.arg_types.push_back(p_type3);

		if (p_name4 == "") {
			goto end;
		}
		cd.arg_count++;
		cd.arg_names.push_back(p_name4);
		cd.arg_types.push_back(p_type4);

	end:

		construct_funcs[p_type].constructors.push_back(cd);
	}

	struct ConstantData {
		Map<StringName, int> value;
#ifdef DEBUG_ENABLED
		List<StringName> value_ordered;
#endif
		Map<StringName, Variant> variant_value;
#ifdef DEBUG_ENABLED
		List<StringName> variant_value_ordered;
#endif
	};

	static ConstantData *constant_data;

	static void add_constant(int p_type, StringName p_constant_name, int p_constant_value) {
		constant_data[p_type].value[p_constant_name] = p_constant_value;
#ifdef DEBUG_ENABLED
		constant_data[p_type].value_ordered.push_back(p_constant_name);
#endif
	}

	static void add_variant_constant(int p_type, StringName p_constant_name, const Variant &p_constant_value) {
		constant_data[p_type].variant_value[p_constant_name] = p_constant_value;
#ifdef DEBUG_ENABLED
		constant_data[p_type].variant_value_ordered.push_back(p_constant_name);
#endif
	}
};

_VariantCall::ConstructFunc *_VariantCall::construct_funcs = nullptr;
_VariantCall::ConstantData *_VariantCall::constant_data = nullptr;
_VariantCall::MethodMap *_VariantCall::type_internal_methods = nullptr;
List<StringName> *_VariantCall::type_internal_method_names = nullptr;

Variant Variant::call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	Variant ret;
	call_ptr(p_method, p_args, p_argcount, &ret, r_error);
	return ret;
}

void Variant::call_ptr(const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Callable::CallError &r_error) {
	Variant ret;

	if (type == Variant::OBJECT) {
		//call object
		Object *obj = _get_obj().obj;
		if (!obj) {
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		}
#ifdef DEBUG_ENABLED
		if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		}

#endif
		ret = _get_obj().obj->call(p_method, p_args, p_argcount, r_error);

		//else if (type==Variant::METHOD) {

	} else {
		r_error.error = Callable::CallError::CALL_OK;

		Variant::InternalMethod **m = _VariantCall::type_internal_methods[type].lookup_ptr(p_method);

		if (m) {
			(*m)->call((Variant *)this, p_args, p_argcount, ret, r_error);
		} else {
			//ok fail because not found
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
			return;
		}
	}

	if (r_error.error == Callable::CallError::CALL_OK && r_ret) {
		*r_ret = ret;
	}
}

#define VCALL(m_type, m_method) _VariantCall::_call_##m_type##_##m_method

Variant Variant::construct(const Variant::Type p_type, const Variant **p_args, int p_argcount, Callable::CallError &r_error, bool p_strict) {
	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	ERR_FAIL_INDEX_V(p_type, VARIANT_MAX, Variant());

	r_error.error = Callable::CallError::CALL_OK;
	if (p_argcount == 0) { //generic construct

		switch (p_type) {
			case NIL:
				return Variant();

			// atomic types
			case BOOL:
				return Variant(false);
			case INT:
				return 0;
			case FLOAT:
				return 0.0f;
			case STRING:
				return String();

			// math types
			case VECTOR2:
				return Vector2();
			case VECTOR2I:
				return Vector2i();
			case RECT2:
				return Rect2();
			case RECT2I:
				return Rect2i();
			case VECTOR3:
				return Vector3();
			case VECTOR3I:
				return Vector3i();
			case TRANSFORM2D:
				return Transform2D();
			case PLANE:
				return Plane();
			case QUAT:
				return Quat();
			case AABB:
				return ::AABB();
			case BASIS:
				return Basis();
			case TRANSFORM:
				return Transform();

			// misc types
			case COLOR:
				return Color();
			case STRING_NAME:
				return StringName();
			case NODE_PATH:
				return NodePath();
			case _RID:
				return RID();
			case OBJECT:
				return (Object *)nullptr;
			case CALLABLE:
				return Callable();
			case SIGNAL:
				return Signal();
			case DICTIONARY:
				return Dictionary();
			case ARRAY:
				return Array();
			case PACKED_BYTE_ARRAY:
				return PackedByteArray();
			case PACKED_INT32_ARRAY:
				return PackedInt32Array();
			case PACKED_INT64_ARRAY:
				return PackedInt64Array();
			case PACKED_FLOAT32_ARRAY:
				return PackedFloat32Array();
			case PACKED_FLOAT64_ARRAY:
				return PackedFloat64Array();
			case PACKED_STRING_ARRAY:
				return PackedStringArray();
			case PACKED_VECTOR2_ARRAY:
				return PackedVector2Array();
			case PACKED_VECTOR3_ARRAY:
				return PackedVector3Array();
			case PACKED_COLOR_ARRAY:
				return PackedColorArray();
			default:
				return Variant();
		}

	} else if (p_argcount == 1 && p_args[0]->type == p_type) {
		return *p_args[0]; //copy construct
	} else if (p_argcount == 1 && (!p_strict || Variant::can_convert(p_args[0]->type, p_type))) {
		//near match construct

		switch (p_type) {
			case NIL: {
				return Variant();
			} break;
			case BOOL: {
				return Variant(bool(*p_args[0]));
			}
			case INT: {
				return (int64_t(*p_args[0]));
			}
			case FLOAT: {
				return double(*p_args[0]);
			}
			case STRING: {
				return String(*p_args[0]);
			}
			case VECTOR2: {
				return Vector2(*p_args[0]);
			}
			case VECTOR2I: {
				return Vector2i(*p_args[0]);
			}
			case RECT2:
				return (Rect2(*p_args[0]));
			case RECT2I:
				return (Rect2i(*p_args[0]));
			case VECTOR3:
				return (Vector3(*p_args[0]));
			case VECTOR3I:
				return (Vector3i(*p_args[0]));
			case TRANSFORM2D:
				return (Transform2D(p_args[0]->operator Transform2D()));
			case PLANE:
				return (Plane(*p_args[0]));
			case QUAT:
				return (p_args[0]->operator Quat());
			case AABB:
				return (::AABB(*p_args[0]));
			case BASIS:
				return (Basis(p_args[0]->operator Basis()));
			case TRANSFORM:
				return (Transform(p_args[0]->operator Transform()));

			// misc types
			case COLOR:
				return p_args[0]->type == Variant::STRING ? Color::html(*p_args[0]) : Color::hex(*p_args[0]);
			case STRING_NAME:
				return (StringName(p_args[0]->operator StringName()));
			case NODE_PATH:
				return (NodePath(p_args[0]->operator NodePath()));
			case _RID:
				return (RID(*p_args[0]));
			case OBJECT:
				return ((Object *)(p_args[0]->operator Object *()));
			case CALLABLE:
				return ((Callable)(p_args[0]->operator Callable()));
			case SIGNAL:
				return ((Signal)(p_args[0]->operator Signal()));
			case DICTIONARY:
				return p_args[0]->operator Dictionary();
			case ARRAY:
				return p_args[0]->operator Array();

			// arrays
			case PACKED_BYTE_ARRAY:
				return (PackedByteArray(*p_args[0]));
			case PACKED_INT32_ARRAY:
				return (PackedInt32Array(*p_args[0]));
			case PACKED_INT64_ARRAY:
				return (PackedInt64Array(*p_args[0]));
			case PACKED_FLOAT32_ARRAY:
				return (PackedFloat32Array(*p_args[0]));
			case PACKED_FLOAT64_ARRAY:
				return (PackedFloat64Array(*p_args[0]));
			case PACKED_STRING_ARRAY:
				return (PackedStringArray(*p_args[0]));
			case PACKED_VECTOR2_ARRAY:
				return (PackedVector2Array(*p_args[0]));
			case PACKED_VECTOR3_ARRAY:
				return (PackedVector3Array(*p_args[0]));
			case PACKED_COLOR_ARRAY:
				return (PackedColorArray(*p_args[0]));
			default:
				return Variant();
		}
	} else if (p_argcount >= 1) {
		_VariantCall::ConstructFunc &c = _VariantCall::construct_funcs[p_type];

		for (List<_VariantCall::ConstructData>::Element *E = c.constructors.front(); E; E = E->next()) {
			const _VariantCall::ConstructData &cd = E->get();

			if (cd.arg_count != p_argcount) {
				continue;
			}

			//validate parameters
			for (int i = 0; i < cd.arg_count; i++) {
				if (!Variant::can_convert(p_args[i]->type, cd.arg_types[i])) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT; //no such constructor
					r_error.argument = i;
					r_error.expected = cd.arg_types[i];
					return Variant();
				}
			}

			Variant v;
			cd.func(v, p_args);
			return v;
		}
	}
	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD; //no such constructor
	return Variant();
}

bool Variant::has_method(const StringName &p_method) const {
	if (type == OBJECT) {
		Object *obj = get_validated_object();
		if (!obj) {
			return false;
		}

		return obj->has_method(p_method);
	}

	return _VariantCall::type_internal_methods[type].has(p_method);
}

Vector<Variant::Type> Variant::get_method_argument_types(Variant::Type p_type, const StringName &p_method) {
	Vector<Variant::Type> types;

	Variant::InternalMethod **m = _VariantCall::type_internal_methods[p_type].lookup_ptr(p_method);
	if (*m) {
		types.resize((*m)->get_argument_count());
		for (int i = 0; i < (*m)->get_argument_count(); i++) {
			types.write[i] = (*m)->get_argument_type(i);
		}
	}

	return types;
}

bool Variant::is_method_const(Variant::Type p_type, const StringName &p_method) {
	Variant::InternalMethod **m = _VariantCall::type_internal_methods[p_type].lookup_ptr(p_method);
	if (*m) {
		return (*m)->get_flags() & Variant::InternalMethod::FLAG_IS_CONST;
	}
	return false;
}

Vector<StringName> Variant::get_method_argument_names(Variant::Type p_type, const StringName &p_method) {
	Vector<StringName> argnames;

#ifdef DEBUG_ENABLED
	Variant::InternalMethod **m = _VariantCall::type_internal_methods[p_type].lookup_ptr(p_method);
	if (*m) {
		argnames.resize((*m)->get_argument_count());
		for (int i = 0; i < (*m)->get_argument_count(); i++) {
			argnames.write[i] = (*m)->get_argument_name(i);
		}
	}
#endif
	return argnames;
}

Variant::Type Variant::get_method_return_type(Variant::Type p_type, const StringName &p_method, bool *r_has_return) {
	Variant::Type rt = Variant::NIL;
	Variant::InternalMethod **m = _VariantCall::type_internal_methods[p_type].lookup_ptr(p_method);
	if (*m) {
		rt = (*m)->get_return_type();
		if (r_has_return) {
			*r_has_return = ((*m)->get_flags() & Variant::InternalMethod::FLAG_RETURNS_VARIANT) || rt != Variant::NIL;
		}
	}
	return rt;
}

Vector<Variant> Variant::get_method_default_arguments(Variant::Type p_type, const StringName &p_method) {
	Variant::InternalMethod **m = _VariantCall::type_internal_methods[p_type].lookup_ptr(p_method);
	if (*m) {
		return (*m)->get_default_arguments();
	}
	return Vector<Variant>();
}

void Variant::get_method_list(List<MethodInfo> *p_list) const {
	for (List<StringName>::Element *E = _VariantCall::type_internal_method_names[type].front(); E; E = E->next()) {
		Variant::InternalMethod **m = _VariantCall::type_internal_methods[type].lookup_ptr(E->get());
		ERR_CONTINUE(!*m);

		MethodInfo mi;
		mi.name = E->get();
		mi.return_val.type = (*m)->get_return_type();
		if ((*m)->get_flags() & Variant::InternalMethod::FLAG_RETURNS_VARIANT) {
			mi.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		}
		if ((*m)->get_flags() & Variant::InternalMethod::FLAG_IS_CONST) {
			mi.flags |= METHOD_FLAG_CONST;
		}
		if ((*m)->get_flags() & Variant::InternalMethod::FLAG_VARARGS) {
			mi.flags |= METHOD_FLAG_VARARG;
		}

		for (int i = 0; i < (*m)->get_argument_count(); i++) {
			PropertyInfo arg;
#ifdef DEBUG_ENABLED
			arg.name = (*m)->get_argument_name(i);
#else
			arg.name = "arg" + itos(i + 1);
#endif
			arg.type = (*m)->get_argument_type(i);
			mi.arguments.push_back(arg);
		}

		mi.default_arguments = (*m)->get_default_arguments();
		p_list->push_back(mi);
	}
}

void Variant::get_constructor_list(Variant::Type p_type, List<MethodInfo> *p_list) {
	ERR_FAIL_INDEX(p_type, VARIANT_MAX);

	//custom constructors
	for (const List<_VariantCall::ConstructData>::Element *E = _VariantCall::construct_funcs[p_type].constructors.front(); E; E = E->next()) {
		const _VariantCall::ConstructData &cd = E->get();
		MethodInfo mi;
		mi.name = Variant::get_type_name(p_type);
		mi.return_val.type = p_type;
		for (int i = 0; i < cd.arg_count; i++) {
			PropertyInfo pi;
			pi.name = cd.arg_names[i];
			pi.type = cd.arg_types[i];
			mi.arguments.push_back(pi);
		}
		p_list->push_back(mi);
	}
	//default constructors
	for (int i = 0; i < VARIANT_MAX; i++) {
		if (i == p_type) {
			continue;
		}
		if (!Variant::can_convert(Variant::Type(i), p_type)) {
			continue;
		}

		MethodInfo mi;
		mi.name = Variant::get_type_name(p_type);
		PropertyInfo pi;
		pi.name = "from";
		pi.type = Variant::Type(i);
		mi.arguments.push_back(pi);
		mi.return_val.type = p_type;
		p_list->push_back(mi);
	}
}

void Variant::get_constants_for_type(Variant::Type p_type, List<StringName> *p_constants) {
	ERR_FAIL_INDEX(p_type, Variant::VARIANT_MAX);

	_VariantCall::ConstantData &cd = _VariantCall::constant_data[p_type];

#ifdef DEBUG_ENABLED
	for (List<StringName>::Element *E = cd.value_ordered.front(); E; E = E->next()) {
		p_constants->push_back(E->get());
#else
	for (Map<StringName, int>::Element *E = cd.value.front(); E; E = E->next()) {
		p_constants->push_back(E->key());
#endif
	}

#ifdef DEBUG_ENABLED
	for (List<StringName>::Element *E = cd.variant_value_ordered.front(); E; E = E->next()) {
		p_constants->push_back(E->get());
#else
	for (Map<StringName, Variant>::Element *E = cd.variant_value.front(); E; E = E->next()) {
		p_constants->push_back(E->key());
#endif
	}
}

bool Variant::has_constant(Variant::Type p_type, const StringName &p_value) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, false);
	_VariantCall::ConstantData &cd = _VariantCall::constant_data[p_type];
	return cd.value.has(p_value) || cd.variant_value.has(p_value);
}

Variant Variant::get_constant_value(Variant::Type p_type, const StringName &p_value, bool *r_valid) {
	if (r_valid) {
		*r_valid = false;
	}

	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, 0);
	_VariantCall::ConstantData &cd = _VariantCall::constant_data[p_type];

	Map<StringName, int>::Element *E = cd.value.find(p_value);
	if (!E) {
		Map<StringName, Variant>::Element *F = cd.variant_value.find(p_value);
		if (F) {
			if (r_valid) {
				*r_valid = true;
			}
			return F->get();
		} else {
			return -1;
		}
	}
	if (r_valid) {
		*r_valid = true;
	}

	return E->get();
}

Variant::InternalMethod *Variant::get_internal_method(Type p_type, const StringName &p_method_name) {
	ERR_FAIL_INDEX_V(p_type, VARIANT_MAX, nullptr);

	Variant::InternalMethod **m = _VariantCall::type_internal_methods[p_type].lookup_ptr(p_method_name);
	if (*m) {
		return *m;
	}
	return nullptr;
}

void register_variant_methods() {
	_VariantCall::type_internal_methods = memnew_arr(_VariantCall::MethodMap, Variant::VARIANT_MAX);
	_VariantCall::type_internal_method_names = memnew_arr(List<StringName>, Variant::VARIANT_MAX);
	_VariantCall::construct_funcs = memnew_arr(_VariantCall::ConstructFunc, Variant::VARIANT_MAX);
	_VariantCall::constant_data = memnew_arr(_VariantCall::ConstantData, Variant::VARIANT_MAX);

	/* String */

	bind_method(String, casecmp_to, sarray("to"), varray());
	bind_method(String, nocasecmp_to, sarray("to"), varray());
	bind_method(String, naturalnocasecmp_to, sarray("to"), varray());
	bind_method(String, length, sarray(), varray());
	bind_method(String, substr, sarray("from", "len"), varray(-1));
	bind_methodv(find, static_cast<int (String::*)(const String &, int) const>(&String::find), sarray("what", "from"), varray(0));
	bind_method(String, count, sarray("what", "from", "to"), varray(0, 0));
	bind_method(String, countn, sarray("what", "from", "to"), varray(0, 0));
	bind_method(String, findn, sarray("what", "from"), varray(0));
	bind_method(String, rfind, sarray("what", "from"), varray(-1));
	bind_method(String, rfindn, sarray("what", "from"), varray(-1));
	bind_method(String, match, sarray("expr"), varray());
	bind_method(String, matchn, sarray("expr"), varray());
	bind_methodv(begins_with, static_cast<bool (String::*)(const String &) const>(&String::begins_with), sarray("text"), varray());
	bind_method(String, ends_with, sarray("text"), varray());
	bind_method(String, is_subsequence_of, sarray("text"), varray());
	bind_method(String, is_subsequence_ofi, sarray("text"), varray());
	bind_method(String, bigrams, sarray(), varray());
	bind_method(String, similarity, sarray("text"), varray());

	bind_method(String, format, sarray("values", "placeholder"), varray("{_}"));
	bind_methodv(replace, static_cast<String (String::*)(const String &, const String &) const>(&String::replace), sarray("what", "forwhat"), varray());
	bind_method(String, replacen, sarray("what", "forwhat"), varray());
	bind_method(String, repeat, sarray("count"), varray());
	bind_method(String, insert, sarray("position", "what"), varray());
	bind_method(String, capitalize, sarray(), varray());
	bind_method(String, split, sarray("delimiter", "allow_empty", "maxsplit"), varray(true, 0));
	bind_method(String, rsplit, sarray("delimiter", "allow_empty", "maxsplit"), varray(true, 0));
	bind_method(String, split_floats, sarray("delimiter", "allow_empty"), varray(true));
	bind_method(String, join, sarray("parts"), varray());

	bind_method(String, to_upper, sarray(), varray());
	bind_method(String, to_lower, sarray(), varray());

	bind_method(String, left, sarray("position"), varray());
	bind_method(String, right, sarray("position"), varray());

	bind_method(String, strip_edges, sarray("left", "right"), varray(true, true));
	bind_method(String, strip_escapes, sarray(), varray());
	bind_method(String, lstrip, sarray("chars"), varray());
	bind_method(String, rstrip, sarray("chars"), varray());
	bind_method(String, get_extension, sarray(), varray());
	bind_method(String, get_basename, sarray(), varray());
	bind_method(String, plus_file, sarray("file"), varray());
	bind_method(String, ord_at, sarray("at"), varray());
	bind_method(String, dedent, sarray(), varray());
	// FIXME: String needs to be immutable when binding
	//bind_method(String, erase, sarray("position", "chars"), varray());
	bind_method(String, hash, sarray(), varray());
	bind_method(String, md5_text, sarray(), varray());
	bind_method(String, sha1_text, sarray(), varray());
	bind_method(String, sha256_text, sarray(), varray());
	bind_method(String, md5_buffer, sarray(), varray());
	bind_method(String, sha1_buffer, sarray(), varray());
	bind_method(String, sha256_buffer, sarray(), varray());
	bind_method(String, empty, sarray(), varray());
	// FIXME: Static function, not sure how to bind
	//bind_method(String, humanize_size, sarray("size"), varray());

	bind_method(String, is_abs_path, sarray(), varray());
	bind_method(String, is_rel_path, sarray(), varray());
	bind_method(String, get_base_dir, sarray(), varray());
	bind_method(String, get_file, sarray(), varray());
	bind_method(String, xml_escape, sarray("escape_quotes"), varray(false));
	bind_method(String, xml_unescape, sarray(), varray());
	bind_method(String, http_escape, sarray(), varray());
	bind_method(String, http_unescape, sarray(), varray());
	bind_method(String, c_escape, sarray(), varray());
	bind_method(String, c_unescape, sarray(), varray());
	bind_method(String, json_escape, sarray(), varray());
	bind_method(String, percent_encode, sarray(), varray());
	bind_method(String, percent_decode, sarray(), varray());

	bind_method(String, is_valid_identifier, sarray(), varray());
	bind_method(String, is_valid_integer, sarray(), varray());
	bind_method(String, is_valid_float, sarray(), varray());
	bind_method(String, is_valid_hex_number, sarray("with_prefix"), varray(false));
	bind_method(String, is_valid_html_color, sarray(), varray());
	bind_method(String, is_valid_ip_address, sarray(), varray());
	bind_method(String, is_valid_filename, sarray(), varray());

	bind_method(String, to_int, sarray(), varray());
	bind_method(String, to_float, sarray(), varray());
	bind_method(String, hex_to_int, sarray("with_prefix"), varray(true));
	bind_method(String, bin_to_int, sarray("with_prefix"), varray(true));

	bind_method(String, lpad, sarray("min_length", "character"), varray(" "));
	bind_method(String, rpad, sarray("min_length", "character"), varray(" "));
	bind_method(String, pad_decimals, sarray("digits"), varray());
	bind_method(String, pad_zeros, sarray("digits"), varray());
	bind_method(String, trim_prefix, sarray("prefix"), varray());
	bind_method(String, trim_suffix, sarray("suffix"), varray());

	bind_method(String, to_ascii_buffer, sarray(), varray());
	bind_method(String, to_utf8_buffer, sarray(), varray());
	bind_method(String, to_utf16_buffer, sarray(), varray());
	bind_method(String, to_utf32_buffer, sarray(), varray());

	/* Vector2 */

	bind_method(Vector2, angle, sarray(), varray());
	bind_method(Vector2, angle_to, sarray("to"), varray());
	bind_method(Vector2, angle_to_point, sarray("to"), varray());
	bind_method(Vector2, direction_to, sarray("b"), varray());
	bind_method(Vector2, distance_to, sarray("to"), varray());
	bind_method(Vector2, distance_squared_to, sarray("to"), varray());
	bind_method(Vector2, length, sarray(), varray());
	bind_method(Vector2, length_squared, sarray(), varray());
	bind_method(Vector2, normalized, sarray(), varray());
	bind_method(Vector2, is_normalized, sarray(), varray());
	bind_method(Vector2, is_equal_approx, sarray("to"), varray());
	bind_method(Vector2, posmod, sarray("mod"), varray());
	bind_method(Vector2, posmodv, sarray("modv"), varray());
	bind_method(Vector2, project, sarray("b"), varray());
	bind_method(Vector2, lerp, sarray("with", "t"), varray());
	bind_method(Vector2, slerp, sarray("with", "t"), varray());
	bind_method(Vector2, cubic_interpolate, sarray("b", "pre_a", "post_b", "t"), varray());
	bind_method(Vector2, move_toward, sarray("to", "delta"), varray());
	bind_method(Vector2, rotated, sarray("phi"), varray());
	bind_method(Vector2, tangent, sarray(), varray());
	bind_method(Vector2, floor, sarray(), varray());
	bind_method(Vector2, ceil, sarray(), varray());
	bind_method(Vector2, round, sarray(), varray());
	bind_method(Vector2, aspect, sarray(), varray());
	bind_method(Vector2, dot, sarray("with"), varray());
	bind_method(Vector2, slide, sarray("n"), varray());
	bind_method(Vector2, bounce, sarray("n"), varray());
	bind_method(Vector2, reflect, sarray("n"), varray());
	bind_method(Vector2, cross, sarray("with"), varray());
	bind_method(Vector2, abs, sarray(), varray());
	bind_method(Vector2, sign, sarray(), varray());
	bind_method(Vector2, snapped, sarray("by"), varray());
	bind_method(Vector2, clamped, sarray("length"), varray());

	/* Vector2i */

	bind_method(Vector2i, aspect, sarray(), varray());
	bind_method(Vector2i, sign, sarray(), varray());
	bind_method(Vector2i, abs, sarray(), varray());

	/* Rect2 */

	bind_method(Rect2, get_area, sarray(), varray());
	bind_method(Rect2, has_no_area, sarray(), varray());
	bind_method(Rect2, has_point, sarray("point"), varray());
	bind_method(Rect2, is_equal_approx, sarray("rect"), varray());
	bind_method(Rect2, intersects, sarray("b", "include_borders"), varray(false));
	bind_method(Rect2, encloses, sarray("b"), varray());
	bind_method(Rect2, clip, sarray("b"), varray());
	bind_method(Rect2, merge, sarray("b"), varray());
	bind_method(Rect2, expand, sarray("to"), varray());
	bind_method(Rect2, grow, sarray("by"), varray());
	bind_methodv(grow_margin, &Rect2::grow_margin_bind, sarray("margin", "by"), varray());
	bind_method(Rect2, grow_individual, sarray("left", "top", "right", "bottom"), varray());
	bind_method(Rect2, abs, sarray(), varray());

	/* Rect2i */

	bind_method(Rect2i, get_area, sarray(), varray());
	bind_method(Rect2i, has_no_area, sarray(), varray());
	bind_method(Rect2i, has_point, sarray("point"), varray());
	bind_method(Rect2i, intersects, sarray("b"), varray());
	bind_method(Rect2i, encloses, sarray("b"), varray());
	bind_method(Rect2i, clip, sarray("b"), varray());
	bind_method(Rect2i, merge, sarray("b"), varray());
	bind_method(Rect2i, expand, sarray("to"), varray());
	bind_method(Rect2i, grow, sarray("by"), varray());
	bind_methodv(grow_margin, &Rect2i::grow_margin_bind, sarray("margin", "by"), varray());
	bind_method(Rect2i, grow_individual, sarray("left", "top", "right", "bottom"), varray());
	bind_method(Rect2i, abs, sarray(), varray());

	/* Vector3 */

	bind_method(Vector3, min_axis, sarray(), varray());
	bind_method(Vector3, max_axis, sarray(), varray());
	bind_method(Vector3, angle_to, sarray("to"), varray());
	bind_method(Vector3, direction_to, sarray("b"), varray());
	bind_method(Vector3, distance_to, sarray("b"), varray());
	bind_method(Vector3, distance_squared_to, sarray("b"), varray());
	bind_method(Vector3, length, sarray(), varray());
	bind_method(Vector3, length_squared, sarray(), varray());
	bind_method(Vector3, normalized, sarray(), varray());
	bind_method(Vector3, is_normalized, sarray(), varray());
	bind_method(Vector3, is_equal_approx, sarray("to"), varray());
	bind_method(Vector3, inverse, sarray(), varray());
	bind_method(Vector3, snapped, sarray("by"), varray());
	bind_method(Vector3, rotated, sarray("by_axis", "phi"), varray());
	bind_method(Vector3, lerp, sarray("b", "t"), varray());
	bind_method(Vector3, slerp, sarray("b", "t"), varray());
	bind_method(Vector3, cubic_interpolate, sarray("b", "pre_a", "post_b", "t"), varray());
	bind_method(Vector3, move_toward, sarray("to", "delta"), varray());
	bind_method(Vector3, dot, sarray("with"), varray());
	bind_method(Vector3, cross, sarray("with"), varray());
	bind_method(Vector3, outer, sarray("with"), varray());
	bind_method(Vector3, to_diagonal_matrix, sarray(), varray());
	bind_method(Vector3, abs, sarray(), varray());
	bind_method(Vector3, floor, sarray(), varray());
	bind_method(Vector3, ceil, sarray(), varray());
	bind_method(Vector3, round, sarray(), varray());
	bind_method(Vector3, posmod, sarray("mod"), varray());
	bind_method(Vector3, posmodv, sarray("modv"), varray());
	bind_method(Vector3, project, sarray("b"), varray());
	bind_method(Vector3, slide, sarray("n"), varray());
	bind_method(Vector3, bounce, sarray("n"), varray());
	bind_method(Vector3, reflect, sarray("n"), varray());
	bind_method(Vector3, sign, sarray(), varray());

	/* Vector3i */

	bind_method(Vector3i, min_axis, sarray(), varray());
	bind_method(Vector3i, max_axis, sarray(), varray());
	bind_method(Vector3i, sign, sarray(), varray());
	bind_method(Vector3i, abs, sarray(), varray());

	/* Plane */

	bind_method(Plane, normalized, sarray(), varray());
	bind_method(Plane, center, sarray(), varray());
	bind_method(Plane, is_equal_approx, sarray("to_plane"), varray());
	bind_method(Plane, is_point_over, sarray("plane"), varray());
	bind_method(Plane, distance_to, sarray("point"), varray());
	bind_method(Plane, has_point, sarray("point", "epsilon"), varray(CMP_EPSILON));
	bind_method(Plane, project, sarray("point"), varray());
	bind_methodv(intersect_3, &Plane::intersect_3_bind, sarray("b", "c"), varray());
	bind_methodv(intersects_ray, &Plane::intersects_ray_bind, sarray("from", "dir"), varray());
	bind_methodv(intersects_segment, &Plane::intersects_segment_bind, sarray("from", "to"), varray());

	/* Quat */

	bind_method(Quat, length, sarray(), varray());
	bind_method(Quat, length_squared, sarray(), varray());
	bind_method(Quat, normalized, sarray(), varray());
	bind_method(Quat, is_normalized, sarray(), varray());
	bind_method(Quat, is_equal_approx, sarray("to"), varray());
	bind_method(Quat, inverse, sarray(), varray());
	bind_method(Quat, dot, sarray("with"), varray());
	bind_method(Quat, slerp, sarray("b", "t"), varray());
	bind_method(Quat, slerpni, sarray("b", "t"), varray());
	bind_method(Quat, cubic_slerp, sarray("b", "pre_a", "post_b", "t"), varray());
	bind_method(Quat, get_euler, sarray(), varray());

	// FIXME: Quat is atomic, this should be done via construcror
	//ADDFUNC1(QUAT, NIL, Quat, set_euler, VECTOR3, "euler", varray());
	//ADDFUNC2(QUAT, NIL, Quat, set_axis_angle, VECTOR3, "axis", FLOAT, "angle", varray());

	/* Color */

	bind_method(Color, to_argb32, sarray(), varray());
	bind_method(Color, to_abgr32, sarray(), varray());
	bind_method(Color, to_rgba32, sarray(), varray());
	bind_method(Color, to_argb64, sarray(), varray());
	bind_method(Color, to_abgr64, sarray(), varray());
	bind_method(Color, to_rgba64, sarray(), varray());

	bind_method(Color, inverted, sarray(), varray());
	bind_method(Color, contrasted, sarray(), varray());
	bind_method(Color, lerp, sarray("b", "t"), varray());
	bind_method(Color, lightened, sarray("amount"), varray());
	bind_method(Color, darkened, sarray("amount"), varray());
	bind_method(Color, to_html, sarray("with_alpha"), varray(true));
	bind_method(Color, blend, sarray("over"), varray());

	// FIXME: Color is immutable, need to probably find a way to do this via constructor
	//ADDFUNC4R(COLOR, COLOR, Color, from_hsv, FLOAT, "h", FLOAT, "s", FLOAT, "v", FLOAT, "a", varray(1.0));
	bind_method(Color, is_equal_approx, sarray("to"), varray());

	/* RID */

	bind_method(RID, get_id, sarray(), varray());

	/* NodePath */

	bind_method(NodePath, is_absolute, sarray(), varray());
	bind_method(NodePath, get_name_count, sarray(), varray());
	bind_method(NodePath, get_name, sarray("idx"), varray());
	bind_method(NodePath, get_subname_count, sarray(), varray());
	bind_method(NodePath, get_subname, sarray("idx"), varray());
	bind_method(NodePath, get_concatenated_subnames, sarray(), varray());
	bind_method(NodePath, get_as_property_path, sarray(), varray());
	bind_method(NodePath, is_empty, sarray(), varray());

	/* Callable */

	bind_method(Callable, is_null, sarray(), varray());
	bind_method(Callable, is_custom, sarray(), varray());
	bind_method(Callable, is_standard, sarray(), varray());
	bind_method(Callable, get_object, sarray(), varray());
	bind_method(Callable, get_object_id, sarray(), varray());
	bind_method(Callable, get_method, sarray(), varray());
	bind_method(Callable, hash, sarray(), varray());
	bind_method(Callable, unbind, sarray("argcount"), varray());

	bind_custom(Variant::CALLABLE, "call", _VariantCall::func_Callable_call, Variant::InternalMethod::FLAG_VARARGS | Variant::InternalMethod::FLAG_RETURNS_VARIANT, Vector<Variant::Type>(), Variant::NIL, sarray());
	bind_custom(Variant::CALLABLE, "call_deferred", _VariantCall::func_Callable_call_deferred, Variant::InternalMethod::FLAG_VARARGS, Vector<Variant::Type>(), Variant::NIL, sarray());
	bind_custom(Variant::CALLABLE, "bind", _VariantCall::func_Callable_bind, Variant::InternalMethod::FLAG_VARARGS, Vector<Variant::Type>(), Variant::CALLABLE, sarray());

	/* Signal */

	bind_method(Signal, is_null, sarray(), varray());
	bind_method(Signal, get_object, sarray(), varray());
	bind_method(Signal, get_object_id, sarray(), varray());
	bind_method(Signal, get_name, sarray(), varray());

	bind_method(Signal, connect, sarray("callable", "binds", "flags"), varray(Array(), 0));
	bind_method(Signal, disconnect, sarray("callable"), varray());
	bind_method(Signal, is_connected, sarray("callable"), varray());
	bind_method(Signal, get_connections, sarray(), varray());

	bind_custom(Variant::SIGNAL, "emit", _VariantCall::func_Signal_emit, Variant::InternalMethod::FLAG_VARARGS, Vector<Variant::Type>(), Variant::NIL, sarray());

	/* Transform2D */

	bind_method(Transform2D, inverse, sarray(), varray());
	bind_method(Transform2D, affine_inverse, sarray(), varray());
	bind_method(Transform2D, get_rotation, sarray(), varray());
	bind_method(Transform2D, get_origin, sarray(), varray());
	bind_method(Transform2D, get_scale, sarray(), varray());
	bind_method(Transform2D, orthonormalized, sarray(), varray());
	bind_method(Transform2D, rotated, sarray("phi"), varray());
	bind_method(Transform2D, scaled, sarray("scale"), varray());
	bind_method(Transform2D, translated, sarray("offset"), varray());
	bind_method(Transform2D, basis_xform, sarray("v"), varray());
	bind_method(Transform2D, basis_xform_inv, sarray("v"), varray());
	bind_method(Transform2D, interpolate_with, sarray("xform", "t"), varray());
	bind_method(Transform2D, is_equal_approx, sarray("xform"), varray());

	/* Basis */

	bind_method(Basis, inverse, sarray(), varray());
	bind_method(Basis, transposed, sarray(), varray());
	bind_method(Basis, orthonormalized, sarray(), varray());
	bind_method(Basis, determinant, sarray(), varray());
	bind_methodv(rotated, static_cast<Basis (Basis::*)(const Vector3 &, float) const>(&Basis::rotated), sarray("axis", "phi"), varray());
	bind_method(Basis, scaled, sarray("scale"), varray());
	bind_method(Basis, get_scale, sarray(), varray());
	bind_method(Basis, get_euler, sarray(), varray());
	bind_method(Basis, tdotx, sarray("with"), varray());
	bind_method(Basis, tdoty, sarray("with"), varray());
	bind_method(Basis, tdotz, sarray("with"), varray());
	bind_method(Basis, get_orthogonal_index, sarray(), varray());
	bind_method(Basis, slerp, sarray("b", "t"), varray());
	bind_method(Basis, is_equal_approx, sarray("b"), varray());
	bind_method(Basis, get_rotation_quat, sarray(), varray());

	/* AABB */

	bind_method(::AABB, abs, sarray(), varray());
	bind_method(::AABB, get_area, sarray(), varray());
	bind_method(::AABB, has_no_area, sarray(), varray());
	bind_method(::AABB, has_no_surface, sarray(), varray());
	bind_method(::AABB, has_point, sarray("point"), varray());
	bind_method(::AABB, is_equal_approx, sarray("aabb"), varray());
	bind_method(::AABB, intersects, sarray("with"), varray());
	bind_method(::AABB, encloses, sarray("with"), varray());
	bind_method(::AABB, intersects_plane, sarray("plane"), varray());
	bind_method(::AABB, intersection, sarray("with"), varray());
	bind_method(::AABB, merge, sarray("with"), varray());
	bind_method(::AABB, expand, sarray("to_point"), varray());
	bind_method(::AABB, grow, sarray("by"), varray());
	bind_method(::AABB, get_support, sarray("dir"), varray());
	bind_method(::AABB, get_longest_axis, sarray(), varray());
	bind_method(::AABB, get_longest_axis_index, sarray(), varray());
	bind_method(::AABB, get_longest_axis_size, sarray(), varray());
	bind_method(::AABB, get_shortest_axis, sarray(), varray());
	bind_method(::AABB, get_shortest_axis_index, sarray(), varray());
	bind_method(::AABB, get_shortest_axis_size, sarray(), varray());
	bind_method(::AABB, get_endpoint, sarray("idx"), varray());
	bind_methodv(intersects_segment, &AABB::intersects_segment_bind, sarray("from", "to"), varray());
	bind_methodv(intersects_ray, &AABB::intersects_ray_bind, sarray("from", "dir"), varray());

	/* Transform */

	bind_method(Transform, inverse, sarray(), varray());
	bind_method(Transform, affine_inverse, sarray(), varray());
	bind_method(Transform, orthonormalized, sarray(), varray());
	bind_method(Transform, rotated, sarray("axis", "phi"), varray());
	bind_method(Transform, scaled, sarray("scale"), varray());
	bind_method(Transform, translated, sarray("offset"), varray());
	bind_method(Transform, looking_at, sarray("target", "up"), varray());
	bind_method(Transform, interpolate_with, sarray("xform", "weight"), varray());
	bind_method(Transform, is_equal_approx, sarray("xform"), varray());

	/* Dictionary */

	bind_method(Dictionary, size, sarray(), varray());
	bind_method(Dictionary, empty, sarray(), varray());
	bind_method(Dictionary, clear, sarray(), varray());
	bind_method(Dictionary, has, sarray("key"), varray());
	bind_method(Dictionary, has_all, sarray("keys"), varray());
	bind_method(Dictionary, erase, sarray("key"), varray());
	bind_method(Dictionary, hash, sarray(), varray());
	bind_method(Dictionary, keys, sarray(), varray());
	bind_method(Dictionary, values, sarray(), varray());
	bind_method(Dictionary, duplicate, sarray("deep"), varray(false));
	bind_method(Dictionary, get, sarray("key", "default"), varray(Variant()));

	/* Array */

	bind_method(Array, size, sarray(), varray());
	bind_method(Array, empty, sarray(), varray());
	bind_method(Array, clear, sarray(), varray());
	bind_method(Array, hash, sarray(), varray());
	bind_method(Array, push_back, sarray("value"), varray());
	bind_method(Array, push_front, sarray("value"), varray());
	bind_method(Array, append, sarray("value"), varray());
	bind_method(Array, resize, sarray("size"), varray());
	bind_method(Array, insert, sarray("position", "value"), varray());
	bind_method(Array, remove, sarray("position"), varray());
	bind_method(Array, erase, sarray("value"), varray());
	bind_method(Array, front, sarray(), varray());
	bind_method(Array, back, sarray(), varray());
	bind_method(Array, find, sarray("what", "from"), varray(0));
	bind_method(Array, rfind, sarray("what", "from"), varray(-1));
	bind_method(Array, find_last, sarray("value"), varray());
	bind_method(Array, count, sarray("value"), varray());
	bind_method(Array, has, sarray("value"), varray());
	bind_method(Array, pop_back, sarray(), varray());
	bind_method(Array, pop_front, sarray(), varray());
	bind_method(Array, sort, sarray(), varray());
	bind_method(Array, sort_custom, sarray("obj", "func"), varray());
	bind_method(Array, shuffle, sarray(), varray());
	bind_method(Array, bsearch, sarray("value", "before"), varray(true));
	bind_method(Array, bsearch_custom, sarray("value", "obj", "func", "before"), varray(true));
	bind_method(Array, invert, sarray(), varray());
	bind_method(Array, duplicate, sarray("deep"), varray(false));
	bind_method(Array, slice, sarray("begin", "end", "step", "deep"), varray(1, false));
	bind_method(Array, max, sarray(), varray());
	bind_method(Array, min, sarray(), varray());

	/* Byte Array */
	bind_method(PackedByteArray, size, sarray(), varray());
	bind_method(PackedByteArray, empty, sarray(), varray());
	bind_method(PackedByteArray, set, sarray("index", "value"), varray());
	bind_method(PackedByteArray, push_back, sarray("value"), varray());
	bind_method(PackedByteArray, append, sarray("value"), varray());
	bind_method(PackedByteArray, append_array, sarray("array"), varray());
	bind_method(PackedByteArray, remove, sarray("index"), varray());
	bind_method(PackedByteArray, insert, sarray("at_index", "value"), varray());
	bind_method(PackedByteArray, resize, sarray("new_size"), varray());
	bind_method(PackedByteArray, has, sarray("value"), varray());
	bind_method(PackedByteArray, invert, sarray(), varray());
	bind_method(PackedByteArray, subarray, sarray("from", "to"), varray());
	bind_method(PackedByteArray, sort, sarray(), varray());

	bind_function("get_string_from_ascii", _VariantCall::func_PackedByteArray_get_string_from_ascii, sarray(), varray());
	bind_function("get_string_from_utf8", _VariantCall::func_PackedByteArray_get_string_from_utf8, sarray(), varray());
	bind_function("get_string_from_utf16", _VariantCall::func_PackedByteArray_get_string_from_utf16, sarray(), varray());
	bind_function("get_string_from_utf32", _VariantCall::func_PackedByteArray_get_string_from_utf32, sarray(), varray());
	bind_function("hex_encode", _VariantCall::func_PackedByteArray_hex_encode, sarray(), varray());
	bind_function("compress", _VariantCall::func_PackedByteArray_compress, sarray("compression_mode"), varray(0));
	bind_function("decompress", _VariantCall::func_PackedByteArray_decompress, sarray("buffer_size", "compression_mode"), varray(0));
	bind_function("decompress_dynamic", _VariantCall::func_PackedByteArray_decompress_dynamic, sarray("max_output_size", "compression_mode"), varray(0));

	/* Int32 Array */

	bind_method(PackedInt32Array, size, sarray(), varray());
	bind_method(PackedInt32Array, empty, sarray(), varray());
	bind_method(PackedInt32Array, set, sarray("index", "value"), varray());
	bind_method(PackedInt32Array, push_back, sarray("value"), varray());
	bind_method(PackedInt32Array, append, sarray("value"), varray());
	bind_method(PackedInt32Array, append_array, sarray("array"), varray());
	bind_method(PackedInt32Array, remove, sarray("index"), varray());
	bind_method(PackedInt32Array, insert, sarray("at_index", "value"), varray());
	bind_method(PackedInt32Array, resize, sarray("new_size"), varray());
	bind_method(PackedInt32Array, has, sarray("value"), varray());
	bind_method(PackedInt32Array, invert, sarray(), varray());
	bind_method(PackedInt32Array, subarray, sarray("from", "to"), varray());
	bind_method(PackedInt32Array, to_byte_array, sarray(), varray());
	bind_method(PackedInt32Array, sort, sarray(), varray());

	/* Int64 Array */

	bind_method(PackedInt64Array, size, sarray(), varray());
	bind_method(PackedInt64Array, empty, sarray(), varray());
	bind_method(PackedInt64Array, set, sarray("index", "value"), varray());
	bind_method(PackedInt64Array, push_back, sarray("value"), varray());
	bind_method(PackedInt64Array, append, sarray("value"), varray());
	bind_method(PackedInt64Array, append_array, sarray("array"), varray());
	bind_method(PackedInt64Array, remove, sarray("index"), varray());
	bind_method(PackedInt64Array, insert, sarray("at_index", "value"), varray());
	bind_method(PackedInt64Array, resize, sarray("new_size"), varray());
	bind_method(PackedInt64Array, has, sarray("value"), varray());
	bind_method(PackedInt64Array, invert, sarray(), varray());
	bind_method(PackedInt64Array, subarray, sarray("from", "to"), varray());
	bind_method(PackedInt64Array, to_byte_array, sarray(), varray());
	bind_method(PackedInt64Array, sort, sarray(), varray());

	/* Float32 Array */

	bind_method(PackedFloat32Array, size, sarray(), varray());
	bind_method(PackedFloat32Array, empty, sarray(), varray());
	bind_method(PackedFloat32Array, set, sarray("index", "value"), varray());
	bind_method(PackedFloat32Array, push_back, sarray("value"), varray());
	bind_method(PackedFloat32Array, append, sarray("value"), varray());
	bind_method(PackedFloat32Array, append_array, sarray("array"), varray());
	bind_method(PackedFloat32Array, remove, sarray("index"), varray());
	bind_method(PackedFloat32Array, insert, sarray("at_index", "value"), varray());
	bind_method(PackedFloat32Array, resize, sarray("new_size"), varray());
	bind_method(PackedFloat32Array, has, sarray("value"), varray());
	bind_method(PackedFloat32Array, invert, sarray(), varray());
	bind_method(PackedFloat32Array, subarray, sarray("from", "to"), varray());
	bind_method(PackedFloat32Array, to_byte_array, sarray(), varray());
	bind_method(PackedFloat32Array, sort, sarray(), varray());

	/* Float64 Array */

	bind_method(PackedFloat64Array, size, sarray(), varray());
	bind_method(PackedFloat64Array, empty, sarray(), varray());
	bind_method(PackedFloat64Array, set, sarray("index", "value"), varray());
	bind_method(PackedFloat64Array, push_back, sarray("value"), varray());
	bind_method(PackedFloat64Array, append, sarray("value"), varray());
	bind_method(PackedFloat64Array, append_array, sarray("array"), varray());
	bind_method(PackedFloat64Array, remove, sarray("index"), varray());
	bind_method(PackedFloat64Array, insert, sarray("at_index", "value"), varray());
	bind_method(PackedFloat64Array, resize, sarray("new_size"), varray());
	bind_method(PackedFloat64Array, has, sarray("value"), varray());
	bind_method(PackedFloat64Array, invert, sarray(), varray());
	bind_method(PackedFloat64Array, subarray, sarray("from", "to"), varray());
	bind_method(PackedFloat64Array, to_byte_array, sarray(), varray());
	bind_method(PackedFloat64Array, sort, sarray(), varray());

	/* String Array */

	bind_method(PackedStringArray, size, sarray(), varray());
	bind_method(PackedStringArray, empty, sarray(), varray());
	bind_method(PackedStringArray, set, sarray("index", "value"), varray());
	bind_method(PackedStringArray, push_back, sarray("value"), varray());
	bind_method(PackedStringArray, append, sarray("value"), varray());
	bind_method(PackedStringArray, append_array, sarray("array"), varray());
	bind_method(PackedStringArray, remove, sarray("index"), varray());
	bind_method(PackedStringArray, insert, sarray("at_index", "value"), varray());
	bind_method(PackedStringArray, resize, sarray("new_size"), varray());
	bind_method(PackedStringArray, has, sarray("value"), varray());
	bind_method(PackedStringArray, invert, sarray(), varray());
	bind_method(PackedStringArray, subarray, sarray("from", "to"), varray());
	bind_method(PackedStringArray, to_byte_array, sarray(), varray());
	bind_method(PackedStringArray, sort, sarray(), varray());

	/* Vector2 Array */

	bind_method(PackedVector2Array, size, sarray(), varray());
	bind_method(PackedVector2Array, empty, sarray(), varray());
	bind_method(PackedVector2Array, set, sarray("index", "value"), varray());
	bind_method(PackedVector2Array, push_back, sarray("value"), varray());
	bind_method(PackedVector2Array, append, sarray("value"), varray());
	bind_method(PackedVector2Array, append_array, sarray("array"), varray());
	bind_method(PackedVector2Array, remove, sarray("index"), varray());
	bind_method(PackedVector2Array, insert, sarray("at_index", "value"), varray());
	bind_method(PackedVector2Array, resize, sarray("new_size"), varray());
	bind_method(PackedVector2Array, has, sarray("value"), varray());
	bind_method(PackedVector2Array, invert, sarray(), varray());
	bind_method(PackedVector2Array, subarray, sarray("from", "to"), varray());
	bind_method(PackedVector2Array, to_byte_array, sarray(), varray());
	bind_method(PackedVector2Array, sort, sarray(), varray());

	/* Vector3 Array */

	bind_method(PackedVector3Array, size, sarray(), varray());
	bind_method(PackedVector3Array, empty, sarray(), varray());
	bind_method(PackedVector3Array, set, sarray("index", "value"), varray());
	bind_method(PackedVector3Array, push_back, sarray("value"), varray());
	bind_method(PackedVector3Array, append, sarray("value"), varray());
	bind_method(PackedVector3Array, append_array, sarray("array"), varray());
	bind_method(PackedVector3Array, remove, sarray("index"), varray());
	bind_method(PackedVector3Array, insert, sarray("at_index", "value"), varray());
	bind_method(PackedVector3Array, resize, sarray("new_size"), varray());
	bind_method(PackedVector3Array, has, sarray("value"), varray());
	bind_method(PackedVector3Array, invert, sarray(), varray());
	bind_method(PackedVector3Array, subarray, sarray("from", "to"), varray());
	bind_method(PackedVector3Array, to_byte_array, sarray(), varray());
	bind_method(PackedVector3Array, sort, sarray(), varray());

	/* Color Array */

	bind_method(PackedColorArray, size, sarray(), varray());
	bind_method(PackedColorArray, empty, sarray(), varray());
	bind_method(PackedColorArray, set, sarray("index", "value"), varray());
	bind_method(PackedColorArray, push_back, sarray("value"), varray());
	bind_method(PackedColorArray, append, sarray("value"), varray());
	bind_method(PackedColorArray, append_array, sarray("array"), varray());
	bind_method(PackedColorArray, remove, sarray("index"), varray());
	bind_method(PackedColorArray, insert, sarray("at_index", "value"), varray());
	bind_method(PackedColorArray, resize, sarray("new_size"), varray());
	bind_method(PackedColorArray, has, sarray("value"), varray());
	bind_method(PackedColorArray, invert, sarray(), varray());
	bind_method(PackedColorArray, subarray, sarray("from", "to"), varray());
	bind_method(PackedColorArray, to_byte_array, sarray(), varray());
	bind_method(PackedColorArray, sort, sarray(), varray());

	/* Register constructors */

	_VariantCall::add_constructor(_VariantCall::Vector2_init1, Variant::VECTOR2, "x", Variant::FLOAT, "y", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Vector2i_init1, Variant::VECTOR2I, "x", Variant::INT, "y", Variant::INT);

	_VariantCall::add_constructor(_VariantCall::Rect2_init1, Variant::RECT2, "position", Variant::VECTOR2, "size", Variant::VECTOR2);
	_VariantCall::add_constructor(_VariantCall::Rect2_init2, Variant::RECT2, "x", Variant::FLOAT, "y", Variant::FLOAT, "width", Variant::FLOAT, "height", Variant::FLOAT);

	_VariantCall::add_constructor(_VariantCall::Rect2i_init1, Variant::RECT2I, "position", Variant::VECTOR2I, "size", Variant::VECTOR2I);
	_VariantCall::add_constructor(_VariantCall::Rect2i_init2, Variant::RECT2I, "x", Variant::INT, "y", Variant::INT, "width", Variant::INT, "height", Variant::INT);

	_VariantCall::add_constructor(_VariantCall::Transform2D_init2, Variant::TRANSFORM2D, "rotation", Variant::FLOAT, "position", Variant::VECTOR2);
	_VariantCall::add_constructor(_VariantCall::Transform2D_init3, Variant::TRANSFORM2D, "x_axis", Variant::VECTOR2, "y_axis", Variant::VECTOR2, "origin", Variant::VECTOR2);

	_VariantCall::add_constructor(_VariantCall::Vector3_init1, Variant::VECTOR3, "x", Variant::FLOAT, "y", Variant::FLOAT, "z", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Vector3i_init1, Variant::VECTOR3I, "x", Variant::INT, "y", Variant::INT, "z", Variant::INT);

	_VariantCall::add_constructor(_VariantCall::Plane_init1, Variant::PLANE, "a", Variant::FLOAT, "b", Variant::FLOAT, "c", Variant::FLOAT, "d", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Plane_init2, Variant::PLANE, "v1", Variant::VECTOR3, "v2", Variant::VECTOR3, "v3", Variant::VECTOR3);
	_VariantCall::add_constructor(_VariantCall::Plane_init3, Variant::PLANE, "normal", Variant::VECTOR3, "d", Variant::FLOAT);

	_VariantCall::add_constructor(_VariantCall::Quat_init1, Variant::QUAT, "x", Variant::FLOAT, "y", Variant::FLOAT, "z", Variant::FLOAT, "w", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Quat_init2, Variant::QUAT, "axis", Variant::VECTOR3, "angle", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Quat_init3, Variant::QUAT, "euler", Variant::VECTOR3);

	_VariantCall::add_constructor(_VariantCall::Color_init1, Variant::COLOR, "r", Variant::FLOAT, "g", Variant::FLOAT, "b", Variant::FLOAT, "a", Variant::FLOAT);
	_VariantCall::add_constructor(_VariantCall::Color_init2, Variant::COLOR, "r", Variant::FLOAT, "g", Variant::FLOAT, "b", Variant::FLOAT);
	// init3 and init4 are the constructors for HTML hex strings and integers respectively which don't need binding here, so we skip to init5.
	_VariantCall::add_constructor(_VariantCall::Color_init5, Variant::COLOR, "c", Variant::COLOR, "a", Variant::FLOAT);

	_VariantCall::add_constructor(_VariantCall::AABB_init1, Variant::AABB, "position", Variant::VECTOR3, "size", Variant::VECTOR3);

	_VariantCall::add_constructor(_VariantCall::Basis_init1, Variant::BASIS, "x_axis", Variant::VECTOR3, "y_axis", Variant::VECTOR3, "z_axis", Variant::VECTOR3);
	_VariantCall::add_constructor(_VariantCall::Basis_init2, Variant::BASIS, "axis", Variant::VECTOR3, "phi", Variant::FLOAT);

	_VariantCall::add_constructor(_VariantCall::Transform_init1, Variant::TRANSFORM, "x_axis", Variant::VECTOR3, "y_axis", Variant::VECTOR3, "z_axis", Variant::VECTOR3, "origin", Variant::VECTOR3);
	_VariantCall::add_constructor(_VariantCall::Transform_init2, Variant::TRANSFORM, "basis", Variant::BASIS, "origin", Variant::VECTOR3);

	_VariantCall::add_constructor(_VariantCall::Callable_init2, Variant::CALLABLE, "object", Variant::OBJECT, "method_name", Variant::STRING_NAME);
	_VariantCall::add_constructor(_VariantCall::Signal_init2, Variant::SIGNAL, "object", Variant::OBJECT, "signal_name", Variant::STRING_NAME);

	/* Register constants */

	_populate_named_colors();
	for (Map<String, Color>::Element *color = _named_colors.front(); color; color = color->next()) {
		_VariantCall::add_variant_constant(Variant::COLOR, color->key(), color->value());
	}

	_VariantCall::add_constant(Variant::VECTOR3, "AXIS_X", Vector3::AXIS_X);
	_VariantCall::add_constant(Variant::VECTOR3, "AXIS_Y", Vector3::AXIS_Y);
	_VariantCall::add_constant(Variant::VECTOR3, "AXIS_Z", Vector3::AXIS_Z);

	_VariantCall::add_variant_constant(Variant::VECTOR3, "ZERO", Vector3(0, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "ONE", Vector3(1, 1, 1));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "INF", Vector3(Math_INF, Math_INF, Math_INF));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "LEFT", Vector3(-1, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "RIGHT", Vector3(1, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "UP", Vector3(0, 1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "DOWN", Vector3(0, -1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "FORWARD", Vector3(0, 0, -1));
	_VariantCall::add_variant_constant(Variant::VECTOR3, "BACK", Vector3(0, 0, 1));

	_VariantCall::add_constant(Variant::VECTOR3I, "AXIS_X", Vector3i::AXIS_X);
	_VariantCall::add_constant(Variant::VECTOR3I, "AXIS_Y", Vector3i::AXIS_Y);
	_VariantCall::add_constant(Variant::VECTOR3I, "AXIS_Z", Vector3i::AXIS_Z);

	_VariantCall::add_variant_constant(Variant::VECTOR3I, "ZERO", Vector3i(0, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "ONE", Vector3i(1, 1, 1));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "LEFT", Vector3i(-1, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "RIGHT", Vector3i(1, 0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "UP", Vector3i(0, 1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "DOWN", Vector3i(0, -1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "FORWARD", Vector3i(0, 0, -1));
	_VariantCall::add_variant_constant(Variant::VECTOR3I, "BACK", Vector3i(0, 0, 1));

	_VariantCall::add_constant(Variant::VECTOR2, "AXIS_X", Vector2::AXIS_X);
	_VariantCall::add_constant(Variant::VECTOR2, "AXIS_Y", Vector2::AXIS_Y);

	_VariantCall::add_constant(Variant::VECTOR2I, "AXIS_X", Vector2i::AXIS_X);
	_VariantCall::add_constant(Variant::VECTOR2I, "AXIS_Y", Vector2i::AXIS_Y);

	_VariantCall::add_variant_constant(Variant::VECTOR2, "ZERO", Vector2(0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "ONE", Vector2(1, 1));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "INF", Vector2(Math_INF, Math_INF));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "LEFT", Vector2(-1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "RIGHT", Vector2(1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "UP", Vector2(0, -1));
	_VariantCall::add_variant_constant(Variant::VECTOR2, "DOWN", Vector2(0, 1));

	_VariantCall::add_variant_constant(Variant::VECTOR2I, "ZERO", Vector2i(0, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "ONE", Vector2i(1, 1));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "LEFT", Vector2i(-1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "RIGHT", Vector2i(1, 0));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "UP", Vector2i(0, -1));
	_VariantCall::add_variant_constant(Variant::VECTOR2I, "DOWN", Vector2i(0, 1));

	_VariantCall::add_variant_constant(Variant::TRANSFORM2D, "IDENTITY", Transform2D());
	_VariantCall::add_variant_constant(Variant::TRANSFORM2D, "FLIP_X", Transform2D(-1, 0, 0, 1, 0, 0));
	_VariantCall::add_variant_constant(Variant::TRANSFORM2D, "FLIP_Y", Transform2D(1, 0, 0, -1, 0, 0));

	Transform identity_transform = Transform();
	Transform flip_x_transform = Transform(-1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
	Transform flip_y_transform = Transform(1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0);
	Transform flip_z_transform = Transform(1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0);
	_VariantCall::add_variant_constant(Variant::TRANSFORM, "IDENTITY", identity_transform);
	_VariantCall::add_variant_constant(Variant::TRANSFORM, "FLIP_X", flip_x_transform);
	_VariantCall::add_variant_constant(Variant::TRANSFORM, "FLIP_Y", flip_y_transform);
	_VariantCall::add_variant_constant(Variant::TRANSFORM, "FLIP_Z", flip_z_transform);

	Basis identity_basis = Basis();
	Basis flip_x_basis = Basis(-1, 0, 0, 0, 1, 0, 0, 0, 1);
	Basis flip_y_basis = Basis(1, 0, 0, 0, -1, 0, 0, 0, 1);
	Basis flip_z_basis = Basis(1, 0, 0, 0, 1, 0, 0, 0, -1);
	_VariantCall::add_variant_constant(Variant::BASIS, "IDENTITY", identity_basis);
	_VariantCall::add_variant_constant(Variant::BASIS, "FLIP_X", flip_x_basis);
	_VariantCall::add_variant_constant(Variant::BASIS, "FLIP_Y", flip_y_basis);
	_VariantCall::add_variant_constant(Variant::BASIS, "FLIP_Z", flip_z_basis);

	_VariantCall::add_variant_constant(Variant::PLANE, "PLANE_YZ", Plane(Vector3(1, 0, 0), 0));
	_VariantCall::add_variant_constant(Variant::PLANE, "PLANE_XZ", Plane(Vector3(0, 1, 0), 0));
	_VariantCall::add_variant_constant(Variant::PLANE, "PLANE_XY", Plane(Vector3(0, 0, 1), 0));

	_VariantCall::add_variant_constant(Variant::QUAT, "IDENTITY", Quat(0, 0, 0, 1));
}

void unregister_variant_methods() {
	//clear methods
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		for (List<StringName>::Element *E = _VariantCall::type_internal_method_names[i].front(); E; E = E->next()) {
			Variant::InternalMethod **m = _VariantCall::type_internal_methods[i].lookup_ptr(E->get());
			if (*m) {
				memdelete(*m);
			}
		}
	}

	memdelete_arr(_VariantCall::type_internal_methods);
	memdelete_arr(_VariantCall::type_internal_method_names);
	memdelete_arr(_VariantCall::construct_funcs);
	memdelete_arr(_VariantCall::constant_data);
}
