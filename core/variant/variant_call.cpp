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

#include "core/core_string_names.h"
#include "core/crypto/crypto_core.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/compression.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/templates/local_vector.h"
#include "core/templates/oa_hash_map.h"

typedef void (*VariantFunc)(Variant &r_ret, Variant &p_self, const Variant **p_args);
typedef void (*VariantConstructFunc)(Variant &r_ret, const Variant **p_args);

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_method_call(R (T::*method)(P...), Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, const Vector<Variant> &p_defvals, Callable::CallError &r_error) {
	call_with_variant_args_ret_dv(VariantGetInternalPtr<T>::get_ptr(base), method, p_args, p_argcount, r_ret, r_error, p_defvals);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_method_call(R (T::*method)(P...) const, Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, const Vector<Variant> &p_defvals, Callable::CallError &r_error) {
	call_with_variant_args_retc_dv(VariantGetInternalPtr<T>::get_ptr(base), method, p_args, p_argcount, r_ret, r_error, p_defvals);
}

template <class T, class... P>
static _FORCE_INLINE_ void vc_method_call(void (T::*method)(P...), Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, const Vector<Variant> &p_defvals, Callable::CallError &r_error) {
	call_with_variant_args_dv(VariantGetInternalPtr<T>::get_ptr(base), method, p_args, p_argcount, r_error, p_defvals);
}

template <class T, class... P>
static _FORCE_INLINE_ void vc_method_call(void (T::*method)(P...) const, Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, const Vector<Variant> &p_defvals, Callable::CallError &r_error) {
	call_with_variant_argsc_dv(VariantGetInternalPtr<T>::get_ptr(base), method, p_args, p_argcount, r_error, p_defvals);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_validated_call(R (T::*method)(P...), Variant *base, const Variant **p_args, Variant *r_ret) {
	call_with_validated_variant_args_ret(base, method, p_args, r_ret);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_validated_call(R (T::*method)(P...) const, Variant *base, const Variant **p_args, Variant *r_ret) {
	call_with_validated_variant_args_retc(base, method, p_args, r_ret);
}
template <class T, class... P>
static _FORCE_INLINE_ void vc_validated_call(void (T::*method)(P...), Variant *base, const Variant **p_args, Variant *r_ret) {
	call_with_validated_variant_args(base, method, p_args);
}

template <class T, class... P>
static _FORCE_INLINE_ void vc_validated_call(void (T::*method)(P...) const, Variant *base, const Variant **p_args, Variant *r_ret) {
	call_with_validated_variant_argsc(base, method, p_args);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_ptrcall(R (T::*method)(P...), void *p_base, const void **p_args, void *r_ret) {
	call_with_ptr_args_ret(reinterpret_cast<T *>(p_base), method, p_args, r_ret);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_ptrcall(R (T::*method)(P...) const, void *p_base, const void **p_args, void *r_ret) {
	call_with_ptr_args_retc(reinterpret_cast<T *>(p_base), method, p_args, r_ret);
}

template <class T, class... P>
static _FORCE_INLINE_ void vc_ptrcall(void (T::*method)(P...), void *p_base, const void **p_args, void *r_ret) {
	call_with_ptr_args(reinterpret_cast<T *>(p_base), method, p_args);
}

template <class T, class... P>
static _FORCE_INLINE_ void vc_ptrcall(void (T::*method)(P...) const, void *p_base, const void **p_args, void *r_ret) {
	call_with_ptr_argsc(reinterpret_cast<T *>(p_base), method, p_args);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_change_return_type(R (T::*method)(P...), Variant *v) {
	VariantTypeAdjust<R>::adjust(v);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_change_return_type(R (T::*method)(P...) const, Variant *v) {
	VariantTypeAdjust<R>::adjust(v);
}

template <class T, class... P>
static _FORCE_INLINE_ void vc_change_return_type(void (T::*method)(P...), Variant *v) {
	VariantInternal::clear(v);
}

template <class T, class... P>
static _FORCE_INLINE_ void vc_change_return_type(void (T::*method)(P...) const, Variant *v) {
	VariantInternal::clear(v);
}

template <class R, class... P>
static _FORCE_INLINE_ void vc_change_return_type(R (*method)(P...), Variant *v) {
	VariantTypeAdjust<R>::adjust(v);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ int vc_get_argument_count(R (T::*method)(P...)) {
	return sizeof...(P);
}
template <class R, class T, class... P>
static _FORCE_INLINE_ int vc_get_argument_count(R (T::*method)(P...) const) {
	return sizeof...(P);
}

template <class T, class... P>
static _FORCE_INLINE_ int vc_get_argument_count(void (T::*method)(P...)) {
	return sizeof...(P);
}

template <class T, class... P>
static _FORCE_INLINE_ int vc_get_argument_count(void (T::*method)(P...) const) {
	return sizeof...(P);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ int vc_get_argument_count(R (*method)(T *, P...)) {
	return sizeof...(P);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_argument_type(R (T::*method)(P...), int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}
template <class R, class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_argument_type(R (T::*method)(P...) const, int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}

template <class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_argument_type(void (T::*method)(P...), int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}

template <class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_argument_type(void (T::*method)(P...) const, int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_argument_type(R (*method)(T *, P...), int p_arg) {
	return call_get_argument_type<P...>(p_arg);
}

template <class R, class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_return_type(R (T::*method)(P...)) {
	return GetTypeInfo<R>::VARIANT_TYPE;
}

template <class R, class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_return_type(R (T::*method)(P...) const) {
	return GetTypeInfo<R>::VARIANT_TYPE;
}

template <class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_return_type(void (T::*method)(P...)) {
	return Variant::NIL;
}

template <class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_return_type(void (T::*method)(P...) const) {
	return Variant::NIL;
}

template <class R, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_return_type(R (*method)(P...)) {
	return GetTypeInfo<R>::VARIANT_TYPE;
}

template <class R, class T, class... P>
static _FORCE_INLINE_ bool vc_has_return_type(R (T::*method)(P...)) {
	return true;
}
template <class R, class T, class... P>
static _FORCE_INLINE_ bool vc_has_return_type(R (T::*method)(P...) const) {
	return true;
}

template <class T, class... P>
static _FORCE_INLINE_ bool vc_has_return_type(void (T::*method)(P...)) {
	return false;
}

template <class T, class... P>
static _FORCE_INLINE_ bool vc_has_return_type(void (T::*method)(P...) const) {
	return false;
}

template <class R, class T, class... P>
static _FORCE_INLINE_ bool vc_is_const(R (T::*method)(P...)) {
	return false;
}
template <class R, class T, class... P>
static _FORCE_INLINE_ bool vc_is_const(R (T::*method)(P...) const) {
	return true;
}

template <class T, class... P>
static _FORCE_INLINE_ bool vc_is_const(void (T::*method)(P...)) {
	return false;
}

template <class T, class... P>
static _FORCE_INLINE_ bool vc_is_const(void (T::*method)(P...) const) {
	return true;
}

template <class R, class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_base_type(R (T::*method)(P...)) {
	return GetTypeInfo<T>::VARIANT_TYPE;
}
template <class R, class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_base_type(R (T::*method)(P...) const) {
	return GetTypeInfo<T>::VARIANT_TYPE;
}

template <class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_base_type(void (T::*method)(P...)) {
	return GetTypeInfo<T>::VARIANT_TYPE;
}

template <class T, class... P>
static _FORCE_INLINE_ Variant::Type vc_get_base_type(void (T::*method)(P...) const) {
	return GetTypeInfo<T>::VARIANT_TYPE;
}

#define METHOD_CLASS(m_class, m_method_name, m_method_ptr)                                                                                                        \
	struct Method_##m_class##_##m_method_name {                                                                                                                   \
		static void call(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, const Vector<Variant> &p_defvals, Callable::CallError &r_error) { \
			vc_method_call(m_method_ptr, base, p_args, p_argcount, r_ret, p_defvals, r_error);                                                                    \
		}                                                                                                                                                         \
		static void validated_call(Variant *base, const Variant **p_args, int p_argcount, Variant *r_ret) {                                                       \
			vc_change_return_type(m_method_ptr, r_ret);                                                                                                           \
			vc_validated_call(m_method_ptr, base, p_args, r_ret);                                                                                                 \
		}                                                                                                                                                         \
		static void ptrcall(void *p_base, const void **p_args, void *r_ret, int p_argcount) {                                                                     \
			vc_ptrcall(m_method_ptr, p_base, p_args, r_ret);                                                                                                      \
		}                                                                                                                                                         \
		static int get_argument_count() {                                                                                                                         \
			return vc_get_argument_count(m_method_ptr);                                                                                                           \
		}                                                                                                                                                         \
		static Variant::Type get_argument_type(int p_arg) {                                                                                                       \
			return vc_get_argument_type(m_method_ptr, p_arg);                                                                                                     \
		}                                                                                                                                                         \
		static Variant::Type get_return_type() {                                                                                                                  \
			return vc_get_return_type(m_method_ptr);                                                                                                              \
		}                                                                                                                                                         \
		static bool has_return_type() {                                                                                                                           \
			return vc_has_return_type(m_method_ptr);                                                                                                              \
		}                                                                                                                                                         \
		static bool is_const() {                                                                                                                                  \
			return vc_is_const(m_method_ptr);                                                                                                                     \
		}                                                                                                                                                         \
		static bool is_vararg() {                                                                                                                                 \
			return false;                                                                                                                                         \
		}                                                                                                                                                         \
		static Variant::Type get_base_type() {                                                                                                                    \
			return vc_get_base_type(m_method_ptr);                                                                                                                \
		}                                                                                                                                                         \
		static StringName get_name() {                                                                                                                            \
			return #m_method_name;                                                                                                                                \
		}                                                                                                                                                         \
	};

template <class R, class T, class... P>
static _FORCE_INLINE_ void vc_ptrcall(R (*method)(T *, P...), void *p_base, const void **p_args, void *r_ret) {
	call_with_ptr_args_static_retc<T, R, P...>(reinterpret_cast<T *>(p_base), method, p_args, r_ret);
}

#define FUNCTION_CLASS(m_class, m_method_name, m_method_ptr)                                                                                                          \
	struct Method_##m_class##_##m_method_name {                                                                                                                       \
		static void call(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, const Vector<Variant> &p_defvals, Callable::CallError &r_error) {     \
			call_with_variant_args_retc_static_helper_dv(VariantGetInternalPtr<m_class>::get_ptr(base), m_method_ptr, p_args, p_argcount, r_ret, p_defvals, r_error); \
		}                                                                                                                                                             \
		static void validated_call(Variant *base, const Variant **p_args, int p_argcount, Variant *r_ret) {                                                           \
			vc_change_return_type(m_method_ptr, r_ret);                                                                                                               \
			call_with_validated_variant_args_static_retc(base, m_method_ptr, p_args, r_ret);                                                                          \
		}                                                                                                                                                             \
		static void ptrcall(void *p_base, const void **p_args, void *r_ret, int p_argcount) {                                                                         \
			vc_ptrcall(m_method_ptr, p_base, p_args, r_ret);                                                                                                          \
		}                                                                                                                                                             \
		static int get_argument_count() {                                                                                                                             \
			return vc_get_argument_count(m_method_ptr);                                                                                                               \
		}                                                                                                                                                             \
		static Variant::Type get_argument_type(int p_arg) {                                                                                                           \
			return vc_get_argument_type(m_method_ptr, p_arg);                                                                                                         \
		}                                                                                                                                                             \
		static Variant::Type get_return_type() {                                                                                                                      \
			return vc_get_return_type(m_method_ptr);                                                                                                                  \
		}                                                                                                                                                             \
		static bool has_return_type() {                                                                                                                               \
			return true;                                                                                                                                              \
		}                                                                                                                                                             \
		static bool is_const() {                                                                                                                                      \
			return true;                                                                                                                                              \
		}                                                                                                                                                             \
		static bool is_vararg() {                                                                                                                                     \
			return false;                                                                                                                                             \
		}                                                                                                                                                             \
		static Variant::Type get_base_type() {                                                                                                                        \
			return GetTypeInfo<m_class>::VARIANT_TYPE;                                                                                                                \
		}                                                                                                                                                             \
		static StringName get_name() {                                                                                                                                \
			return #m_method_name;                                                                                                                                    \
		}                                                                                                                                                             \
	};

#define VARARG_CLASS(m_class, m_method_name, m_method_ptr, m_has_return, m_return_type)                                                                           \
	struct Method_##m_class##_##m_method_name {                                                                                                                   \
		static void call(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, const Vector<Variant> &p_defvals, Callable::CallError &r_error) { \
			m_method_ptr(base, p_args, p_argcount, r_ret, r_error);                                                                                               \
		}                                                                                                                                                         \
		static void validated_call(Variant *base, const Variant **p_args, int p_argcount, Variant *r_ret) {                                                       \
			Callable::CallError ce;                                                                                                                               \
			m_method_ptr(base, p_args, p_argcount, *r_ret, ce);                                                                                                   \
		}                                                                                                                                                         \
		static void ptrcall(void *p_base, const void **p_args, void *r_ret, int p_argcount) {                                                                     \
			LocalVector<Variant> vars;                                                                                                                            \
			vars.resize(p_argcount);                                                                                                                              \
			LocalVector<const Variant *> vars_ptrs;                                                                                                               \
			vars_ptrs.resize(p_argcount);                                                                                                                         \
			for (int i = 0; i < p_argcount; i++) {                                                                                                                \
				vars[i] = PtrToArg<Variant>::convert(p_args[i]);                                                                                                  \
				vars_ptrs[i] = &vars[i];                                                                                                                          \
			}                                                                                                                                                     \
			Variant base = PtrToArg<m_class>::convert(p_base);                                                                                                    \
			Variant ret;                                                                                                                                          \
			Callable::CallError ce;                                                                                                                               \
			m_method_ptr(&base, (const Variant **)&vars_ptrs[0], p_argcount, ret, ce);                                                                            \
			if (m_has_return) {                                                                                                                                   \
				m_return_type r = ret;                                                                                                                            \
				PtrToArg<m_return_type>::encode(ret, r_ret);                                                                                                      \
			}                                                                                                                                                     \
		}                                                                                                                                                         \
		static int get_argument_count() {                                                                                                                         \
			return 0;                                                                                                                                             \
		}                                                                                                                                                         \
		static Variant::Type get_argument_type(int p_arg) {                                                                                                       \
			return Variant::NIL;                                                                                                                                  \
		}                                                                                                                                                         \
		static Variant::Type get_return_type() {                                                                                                                  \
			return GetTypeInfo<m_return_type>::VARIANT_TYPE;                                                                                                      \
		}                                                                                                                                                         \
		static bool has_return_type() {                                                                                                                           \
			return m_has_return;                                                                                                                                  \
		}                                                                                                                                                         \
		static bool is_const() {                                                                                                                                  \
			return true;                                                                                                                                          \
		}                                                                                                                                                         \
		static bool is_vararg() {                                                                                                                                 \
			return true;                                                                                                                                          \
		}                                                                                                                                                         \
		static Variant::Type get_base_type() {                                                                                                                    \
			return GetTypeInfo<m_class>::VARIANT_TYPE;                                                                                                            \
		}                                                                                                                                                         \
		static StringName get_name() {                                                                                                                            \
			return #m_method_name;                                                                                                                                \
		}                                                                                                                                                         \
	};

struct _VariantCall {
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
			s.parse_utf16((const char16_t *)r, floor((double)p_instance->size() / (double)sizeof(char16_t)));
		}
		return s;
	}

	static String func_PackedByteArray_get_string_from_utf32(PackedByteArray *p_instance) {
		String s;
		if (p_instance->size() > 0) {
			const uint8_t *r = p_instance->ptr();
			s = String((const char32_t *)r, floor((double)p_instance->size() / (double)sizeof(char32_t)));
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

_VariantCall::ConstantData *_VariantCall::constant_data = nullptr;

struct VariantBuiltInMethodInfo {
	void (*call)(Variant *base, const Variant **p_args, int p_argcount, Variant &r_ret, const Vector<Variant> &p_defvals, Callable::CallError &r_error);
	Variant::ValidatedBuiltInMethod validated_call;
	Variant::PTRBuiltInMethod ptrcall;

	Vector<Variant> default_arguments;
	Vector<String> argument_names;

	bool is_const;
	bool has_return_type;
	bool is_vararg;
	Variant::Type return_type;
	int argument_count;
	Variant::Type (*get_argument_type)(int p_arg);
};

typedef OAHashMap<StringName, VariantBuiltInMethodInfo> BuiltinMethodMap;
static BuiltinMethodMap *builtin_method_info;
static List<StringName> *builtin_method_names;

template <class T>
static void register_builtin_method(const Vector<String> &p_argnames, const Vector<Variant> &p_def_args) {
	StringName name = T::get_name();

	ERR_FAIL_COND(builtin_method_info[T::get_base_type()].has(name));

	VariantBuiltInMethodInfo imi;

	imi.call = T::call;
	imi.validated_call = T::validated_call;
	if (T::is_vararg()) {
		imi.ptrcall = nullptr;
	} else {
		imi.ptrcall = T::ptrcall;
	}

	imi.default_arguments = p_def_args;
	imi.argument_names = p_argnames;

	imi.is_const = T::is_const();
	imi.is_vararg = T::is_vararg();
	imi.has_return_type = T::has_return_type();
	imi.return_type = T::get_return_type();
	imi.argument_count = T::get_argument_count();
	imi.get_argument_type = T::get_argument_type;
#ifdef DEBUG_METHODS_ENABLED
	ERR_FAIL_COND(!imi.is_vararg && imi.argument_count != imi.argument_names.size());
#endif

	builtin_method_info[T::get_base_type()].insert(name, imi);
	builtin_method_names[T::get_base_type()].push_back(name);
}

void Variant::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
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
		r_ret = _get_obj().obj->call(p_method, p_args, p_argcount, r_error);

		//else if (type==Variant::METHOD) {
	} else {
		r_error.error = Callable::CallError::CALL_OK;

		const VariantBuiltInMethodInfo *imf = builtin_method_info[type].lookup_ptr(p_method);

		if (!imf) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
			return;
		}

		imf->call(this, p_args, p_argcount, r_ret, imf->default_arguments, r_error);
	}
}

bool Variant::has_method(const StringName &p_method) const {
	if (type == OBJECT) {
		Object *obj = get_validated_object();
		if (!obj) {
			return false;
		}

		return obj->has_method(p_method);
	}

	return builtin_method_info[type].has(p_method);
}

bool Variant::has_builtin_method(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, false);
	return builtin_method_info[p_type].has(p_method);
}

Variant::ValidatedBuiltInMethod Variant::get_validated_builtin_method(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, nullptr);
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, nullptr);
	return method->validated_call;
}

Variant::PTRBuiltInMethod Variant::get_ptr_builtin_method(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, nullptr);
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, nullptr);
	return method->ptrcall;
}

int Variant::get_builtin_method_argument_count(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, 0);
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, 0);
	return method->argument_count;
}

Variant::Type Variant::get_builtin_method_argument_type(Variant::Type p_type, const StringName &p_method, int p_argument) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, Variant::NIL);
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, Variant::NIL);
	ERR_FAIL_INDEX_V(p_argument, method->argument_count, Variant::NIL);
	return method->get_argument_type(p_argument);
}

String Variant::get_builtin_method_argument_name(Variant::Type p_type, const StringName &p_method, int p_argument) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, String());
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, String());
#ifdef DEBUG_METHODS_ENABLED
	ERR_FAIL_INDEX_V(p_argument, method->argument_count, String());
	return method->argument_names[p_argument];
#else
	return "arg" + itos(p_argument + 1);
#endif
}

Vector<Variant> Variant::get_builtin_method_default_arguments(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, Vector<Variant>());
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, Vector<Variant>());
	return method->default_arguments;
}

bool Variant::has_builtin_method_return_value(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, false);
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, false);
	return method->has_return_type;
}

void Variant::get_builtin_method_list(Variant::Type p_type, List<StringName> *p_list) {
	ERR_FAIL_INDEX(p_type, Variant::VARIANT_MAX);
	for (List<StringName>::Element *E = builtin_method_names[p_type].front(); E; E = E->next()) {
		p_list->push_back(E->get());
	}
}

Variant::Type Variant::get_builtin_method_return_type(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, Variant::NIL);
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, Variant::NIL);
	return method->return_type;
}

bool Variant::is_builtin_method_const(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, false);
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, false);
	return method->is_const;
}

bool Variant::is_builtin_method_vararg(Variant::Type p_type, const StringName &p_method) {
	ERR_FAIL_INDEX_V(p_type, Variant::VARIANT_MAX, false);
	const VariantBuiltInMethodInfo *method = builtin_method_info[p_type].lookup_ptr(p_method);
	ERR_FAIL_COND_V(!method, false);
	return method->is_vararg;
}

void Variant::get_method_list(List<MethodInfo> *p_list) const {
	if (type == OBJECT) {
		Object *obj = get_validated_object();
		if (obj) {
			obj->get_method_list(p_list);
		}
	} else {
		for (List<StringName>::Element *E = builtin_method_names[type].front(); E; E = E->next()) {
			const VariantBuiltInMethodInfo *method = builtin_method_info[type].lookup_ptr(E->get());
			ERR_CONTINUE(!method);

			MethodInfo mi;
			mi.name = E->get();

			//return type
			if (method->has_return_type) {
				mi.return_val.type = method->return_type;
				if (mi.return_val.type == Variant::NIL) {
					mi.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
				}
			}

			if (method->is_const) {
				mi.flags |= METHOD_FLAG_CONST;
			}
			if (method->is_vararg) {
				mi.flags |= METHOD_FLAG_VARARG;
			}

			for (int i = 0; i < method->argument_count; i++) {
				PropertyInfo pi;
#ifdef DEBUG_METHODS_ENABLED
				pi.name = method->argument_names[i];
#else
				pi.name = "arg" + itos(i + 1);
#endif
				pi.type = method->get_argument_type(i);
				if (pi.type == Variant::NIL) {
					pi.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
				}
				mi.arguments.push_back(pi);
			}

			mi.default_arguments = method->default_arguments;
			p_list->push_back(mi);
		}
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

#ifdef DEBUG_METHODS_ENABLED
#define bind_method(m_type, m_method, m_arg_names, m_default_args) \
	METHOD_CLASS(m_type, m_method, &m_type::m_method);             \
	register_builtin_method<Method_##m_type##_##m_method>(m_arg_names, m_default_args);
#else
#define bind_method(m_type, m_method, m_arg_names, m_default_args) \
	METHOD_CLASS(m_type, m_method, &m_type ::m_method);            \
	register_builtin_method<Method_##m_type##_##m_method>(sarray(), m_default_args);
#endif

#ifdef DEBUG_METHODS_ENABLED
#define bind_methodv(m_type, m_name, m_method, m_arg_names, m_default_args) \
	METHOD_CLASS(m_type, m_name, m_method);                                 \
	register_builtin_method<Method_##m_type##_##m_name>(m_arg_names, m_default_args);
#else
#define bind_methodv(m_type, m_name, m_method, m_arg_names, m_default_args) \
	METHOD_CLASS(m_type, m_name, m_method);                                 \
	register_builtin_method<Method_##m_type##_##m_name>(sarray(), m_default_args);
#endif

#ifdef DEBUG_METHODS_ENABLED
#define bind_function(m_type, m_name, m_method, m_arg_names, m_default_args) \
	FUNCTION_CLASS(m_type, m_name, m_method);                                \
	register_builtin_method<Method_##m_type##_##m_name>(m_arg_names, m_default_args);
#else
#define bind_function(m_type, m_name, m_method, m_arg_names, m_default_args) \
	FUNCTION_CLASS(m_type, m_name, m_method);                                \
	register_builtin_method<Method_##m_type##_##m_name>(sarray(), m_default_args);
#endif

#define bind_custom(m_type, m_name, m_method, m_has_return, m_ret_type) \
	VARARG_CLASS(m_type, m_name, m_method, m_has_return, m_ret_type)    \
	register_builtin_method<Method_##m_type##_##m_name>(sarray(), Vector<Variant>());

static void _register_variant_builtin_methods() {
	_VariantCall::constant_data = memnew_arr(_VariantCall::ConstantData, Variant::VARIANT_MAX);
	builtin_method_info = memnew_arr(BuiltinMethodMap, Variant::VARIANT_MAX);
	builtin_method_names = memnew_arr(List<StringName>, Variant::VARIANT_MAX);

	/* String */

	bind_method(String, casecmp_to, sarray("to"), varray());
	bind_method(String, nocasecmp_to, sarray("to"), varray());
	bind_method(String, naturalnocasecmp_to, sarray("to"), varray());
	bind_method(String, length, sarray(), varray());
	bind_method(String, substr, sarray("from", "len"), varray(-1));
	bind_methodv(String, find, static_cast<int (String::*)(const String &, int) const>(&String::find), sarray("what", "from"), varray(0));
	bind_method(String, count, sarray("what", "from", "to"), varray(0, 0));
	bind_method(String, countn, sarray("what", "from", "to"), varray(0, 0));
	bind_method(String, findn, sarray("what", "from"), varray(0));
	bind_method(String, rfind, sarray("what", "from"), varray(-1));
	bind_method(String, rfindn, sarray("what", "from"), varray(-1));
	bind_method(String, match, sarray("expr"), varray());
	bind_method(String, matchn, sarray("expr"), varray());
	bind_methodv(String, begins_with, static_cast<bool (String::*)(const String &) const>(&String::begins_with), sarray("text"), varray());
	bind_method(String, ends_with, sarray("text"), varray());
	bind_method(String, is_subsequence_of, sarray("text"), varray());
	bind_method(String, is_subsequence_ofi, sarray("text"), varray());
	bind_method(String, bigrams, sarray(), varray());
	bind_method(String, similarity, sarray("text"), varray());

	bind_method(String, format, sarray("values", "placeholder"), varray("{_}"));
	bind_methodv(String, replace, static_cast<String (String::*)(const String &, const String &) const>(&String::replace), sarray("what", "forwhat"), varray());
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
	bind_methodv(Rect2, grow_margin, &Rect2::grow_margin_bind, sarray("margin", "by"), varray());
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
	bind_methodv(Rect2i, grow_margin, &Rect2i::grow_margin_bind, sarray("margin", "by"), varray());
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
	bind_methodv(Plane, intersect_3, &Plane::intersect_3_bind, sarray("b", "c"), varray());
	bind_methodv(Plane, intersects_ray, &Plane::intersects_ray_bind, sarray("from", "dir"), varray());
	bind_methodv(Plane, intersects_segment, &Plane::intersects_segment_bind, sarray("from", "to"), varray());

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

	bind_custom(Callable, call, _VariantCall::func_Callable_call, true, Variant);
	bind_custom(Callable, call_deferred, _VariantCall::func_Callable_call_deferred, false, Variant);
	bind_custom(Callable, bind, _VariantCall::func_Callable_bind, true, Callable);

	/* Signal */

	bind_method(Signal, is_null, sarray(), varray());
	bind_method(Signal, get_object, sarray(), varray());
	bind_method(Signal, get_object_id, sarray(), varray());
	bind_method(Signal, get_name, sarray(), varray());

	bind_method(Signal, connect, sarray("callable", "binds", "flags"), varray(Array(), 0));
	bind_method(Signal, disconnect, sarray("callable"), varray());
	bind_method(Signal, is_connected, sarray("callable"), varray());
	bind_method(Signal, get_connections, sarray(), varray());

	bind_custom(Signal, emit, _VariantCall::func_Signal_emit, false, Variant);

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
	bind_methodv(Basis, rotated, static_cast<Basis (Basis::*)(const Vector3 &, float) const>(&Basis::rotated), sarray("axis", "phi"), varray());
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

	bind_method(AABB, abs, sarray(), varray());
	bind_method(AABB, get_area, sarray(), varray());
	bind_method(AABB, has_no_area, sarray(), varray());
	bind_method(AABB, has_no_surface, sarray(), varray());
	bind_method(AABB, has_point, sarray("point"), varray());
	bind_method(AABB, is_equal_approx, sarray("aabb"), varray());
	bind_method(AABB, intersects, sarray("with"), varray());
	bind_method(AABB, encloses, sarray("with"), varray());
	bind_method(AABB, intersects_plane, sarray("plane"), varray());
	bind_method(AABB, intersection, sarray("with"), varray());
	bind_method(AABB, merge, sarray("with"), varray());
	bind_method(AABB, expand, sarray("to_point"), varray());
	bind_method(AABB, grow, sarray("by"), varray());
	bind_method(AABB, get_support, sarray("dir"), varray());
	bind_method(AABB, get_longest_axis, sarray(), varray());
	bind_method(AABB, get_longest_axis_index, sarray(), varray());
	bind_method(AABB, get_longest_axis_size, sarray(), varray());
	bind_method(AABB, get_shortest_axis, sarray(), varray());
	bind_method(AABB, get_shortest_axis_index, sarray(), varray());
	bind_method(AABB, get_shortest_axis_size, sarray(), varray());
	bind_method(AABB, get_endpoint, sarray("idx"), varray());
	bind_methodv(AABB, intersects_segment, &AABB::intersects_segment_bind, sarray("from", "to"), varray());
	bind_methodv(AABB, intersects_ray, &AABB::intersects_ray_bind, sarray("from", "dir"), varray());

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
	bind_method(Array, append_array, sarray("array"), varray());
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

	bind_function(PackedByteArray, get_string_from_ascii, _VariantCall::func_PackedByteArray_get_string_from_ascii, sarray(), varray());
	bind_function(PackedByteArray, get_string_from_utf8, _VariantCall::func_PackedByteArray_get_string_from_utf8, sarray(), varray());
	bind_function(PackedByteArray, get_string_from_utf16, _VariantCall::func_PackedByteArray_get_string_from_utf16, sarray(), varray());
	bind_function(PackedByteArray, get_string_from_utf32, _VariantCall::func_PackedByteArray_get_string_from_utf32, sarray(), varray());
	bind_function(PackedByteArray, hex_encode, _VariantCall::func_PackedByteArray_hex_encode, sarray(), varray());
	bind_function(PackedByteArray, compress, _VariantCall::func_PackedByteArray_compress, sarray("compression_mode"), varray(0));
	bind_function(PackedByteArray, decompress, _VariantCall::func_PackedByteArray_decompress, sarray("buffer_size", "compression_mode"), varray(0));
	bind_function(PackedByteArray, decompress_dynamic, _VariantCall::func_PackedByteArray_decompress_dynamic, sarray("max_output_size", "compression_mode"), varray(0));

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

	/* Register constants */

	int ncc = Color::get_named_color_count();
	for (int i = 0; i < ncc; i++) {
		_VariantCall::add_variant_constant(Variant::COLOR, Color::get_named_color_name(i), Color::get_named_color(i));
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

void Variant::_register_variant_methods() {
	_register_variant_builtin_methods(); //needs to be out due to namespace
}

void Variant::_unregister_variant_methods() {
	//clear methods
	memdelete_arr(builtin_method_names);
	memdelete_arr(builtin_method_info);
	memdelete_arr(_VariantCall::constant_data);
}
