/**************************************************************************/
/*  ffi_library.h                                                         */
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

#ifndef FFI_LIBRARY_H
#define FFI_LIBRARY_H

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/os/os.h"
#include "core/variant/callable.h"

#include <ffi.h>

class FFILibrary : public RefCounted {
	GDCLASS(FFILibrary, RefCounted);

public:
	enum FFIABI {
		ABI_DEFAULT,
		ABI_SYSV,
		ABI_VFP,
		ABI_WIN64,
		ABI_GNUW64,
		ABI_UNIX64,
		ABI_STDCALL,
		ABI_THISCALL,
		ABI_FASTCALL,
		ABI_MS_CDECL,
		ABI_PASCAL,
		ABI_REGISTER,
	};

	enum FFIType {
		TYPE_VOID,
		TYPE_UINT8,
		TYPE_SINT8,
		TYPE_UINT16,
		TYPE_SINT16,
		TYPE_UINT32,
		TYPE_SINT32,
		TYPE_UINT64,
		TYPE_SINT64,
		TYPE_FLOAT,
		TYPE_DOUBLE,
		TYPE_UCHAR,
		TYPE_SCHAR,
		TYPE_USHORT,
		TYPE_SSHORT,
		TYPE_UINT,
		TYPE_SINT,
		TYPE_ULONG,
		TYPE_SLONG,
		TYPE_POINTER,
		TYPE_MAX,
	};

	struct FFIStructInfo {
		ffi_type struct_type;
		Vector<Variant> gd_element_types;
		Vector<ffi_type *> element_types;
		Vector<size_t> offsets;
	};

	struct FFIMethodInfo {
		ffi_cif cif;

		Variant gd_ret_type;
		ffi_type *ret_type = nullptr;

		Vector<Variant> gd_arg_types;
		Vector<ffi_type *> arg_types;

		void *handle = nullptr;
	};

private:
	ffi_abi abi = FFI_DEFAULT_ABI;
	String library_path;
	void *library_handle = nullptr;
	mutable HashMap<StringName, FFIStructInfo *> structs;
	mutable HashMap<StringName, FFIMethodInfo *> methods;

	bool _get_type(const Variant &p_arg, ffi_type **r_type, bool p_allow_struct, bool p_allow_void) const;

	bool _encode(const Variant &p_arg_type, const Variant &p_arg_value, uint8_t *r_out) const;
	bool _decode(const Variant &p_arg_type, uint8_t *p_value, Variant &r_out) const;

protected:
	static void _bind_methods();

public:
	Error open(const String &p_path, FFILibrary::FFIABI p_abi = ABI_DEFAULT);

	Variant _bind_struct(const Variant **p_args, int p_argcount, Callable::CallError &r_call_error);
	Variant _bind_method(const Variant **p_args, int p_argcount, Callable::CallError &r_call_error);
	void _callptr(FFILibrary::FFIMethodInfo *p_method, const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const;

	uint32_t _struct_hash(const StringName &p_name) const;
	uint32_t _method_hash(const StringName &p_name) const;

	String get_path() const { return library_path; }

	int get_struct_size(const StringName &p_name) const;
	PackedInt64Array get_struct_offsets(const StringName &p_name) const;
	Callable get_method(const StringName &p_name) const;

	bool is_opened() const { return library_handle != nullptr; }

	FFILibrary();
	~FFILibrary();
};

class FFICallable : public CallableCustom {
	Ref<FFILibrary> library;
	FFILibrary::FFIMethodInfo *method = nullptr;
	StringName name;

	static bool _equal_func(const CallableCustom *p_a, const CallableCustom *p_b);
	static bool _less_func(const CallableCustom *p_a, const CallableCustom *p_b);

public:
	bool is_valid() const override;
	uint32_t hash() const override;
	String get_as_text() const override;
	CallableCustom::CompareEqualFunc get_compare_equal_func() const override;
	CallableCustom::CompareLessFunc get_compare_less_func() const override;
	ObjectID get_object() const override;
	StringName get_method() const override;
	void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const override;

	FFICallable(Ref<FFILibrary> p_library, FFILibrary::FFIMethodInfo *p_method, const StringName &p_name);
	virtual ~FFICallable() = default;
};

VARIANT_ENUM_CAST(FFILibrary::FFIType);
VARIANT_ENUM_CAST(FFILibrary::FFIABI);

#endif // FFI_LIBRARY_H
