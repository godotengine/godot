/**************************************************************************/
/*  ffi_library.cpp                                                       */
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

#include "ffi_library.h"

void FFILibrary::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path", "abi"), &FFILibrary::open, DEFVAL(ABI_DEFAULT));
	ClassDB::bind_method(D_METHOD("is_opened"), &FFILibrary::is_opened);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "bind_struct", &FFILibrary::_bind_struct);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "bind_method", &FFILibrary::_bind_method);

	ClassDB::bind_method(D_METHOD("get_struct_size", "name"), &FFILibrary::get_struct_size);
	ClassDB::bind_method(D_METHOD("get_struct_offsets", "name"), &FFILibrary::get_struct_offsets);

	ClassDB::bind_method(D_METHOD("get_method", "name"), &FFILibrary::get_method);

	BIND_ENUM_CONSTANT(ABI_DEFAULT);
	BIND_ENUM_CONSTANT(ABI_SYSV);
	BIND_ENUM_CONSTANT(ABI_VFP);
	BIND_ENUM_CONSTANT(ABI_WIN64);
	BIND_ENUM_CONSTANT(ABI_GNUW64);
	BIND_ENUM_CONSTANT(ABI_UNIX64);
	BIND_ENUM_CONSTANT(ABI_STDCALL);
	BIND_ENUM_CONSTANT(ABI_THISCALL);
	BIND_ENUM_CONSTANT(ABI_FASTCALL);
	BIND_ENUM_CONSTANT(ABI_MS_CDECL);
	BIND_ENUM_CONSTANT(ABI_PASCAL);
	BIND_ENUM_CONSTANT(ABI_REGISTER);

	BIND_ENUM_CONSTANT(TYPE_VOID);
	BIND_ENUM_CONSTANT(TYPE_UINT8);
	BIND_ENUM_CONSTANT(TYPE_SINT8);
	BIND_ENUM_CONSTANT(TYPE_UINT16);
	BIND_ENUM_CONSTANT(TYPE_SINT16);
	BIND_ENUM_CONSTANT(TYPE_UINT32);
	BIND_ENUM_CONSTANT(TYPE_SINT32);
	BIND_ENUM_CONSTANT(TYPE_UINT64);
	BIND_ENUM_CONSTANT(TYPE_SINT64);
	BIND_ENUM_CONSTANT(TYPE_FLOAT);
	BIND_ENUM_CONSTANT(TYPE_DOUBLE);
	BIND_ENUM_CONSTANT(TYPE_UCHAR);
	BIND_ENUM_CONSTANT(TYPE_SCHAR);
	BIND_ENUM_CONSTANT(TYPE_USHORT);
	BIND_ENUM_CONSTANT(TYPE_SSHORT);
	BIND_ENUM_CONSTANT(TYPE_UINT);
	BIND_ENUM_CONSTANT(TYPE_SINT);
	BIND_ENUM_CONSTANT(TYPE_ULONG);
	BIND_ENUM_CONSTANT(TYPE_SLONG);
	BIND_ENUM_CONSTANT(TYPE_POINTER);
}

Error FFILibrary::open(const String &p_path, FFILibrary::FFIABI p_abi) {
	ERR_FAIL_COND_V_MSG(library_handle, ERR_CANT_OPEN, "Library is already opened.");

	Error err = OS::get_singleton()->open_dynamic_library(p_path, library_handle, nullptr);
	ERR_FAIL_COND_V_MSG(err != OK, ERR_CANT_OPEN, "Unable to open library.");

	switch (p_abi) {
		case ABI_DEFAULT: {
			abi = FFI_DEFAULT_ABI;
		} break;
		case ABI_SYSV: {
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64) || defined(__i386) || defined(__i386__) || defined(_M_IX86) || defined(__riscv) || defined(__powerpc__)
			abi = FFI_SYSV;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
		case ABI_VFP: {
#if defined(__arm__) || defined(_M_ARM)
			abi = FFI_VFP;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
		case ABI_WIN64: {
#if defined(X86_WIN64)
			abi = FFI_WIN64;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI");
#endif
		} break;
		case ABI_GNUW64: {
#if defined(X86_WIN64)
			abi = FFI_GNUW64;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI");
#endif
		} break;
		case ABI_UNIX64: {
#if (defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)) && !defined(X86_WIN64)
			abi = FFI_UNIX64;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
		case ABI_STDCALL: {
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
			abi = FFI_STDCALL;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
		case ABI_THISCALL: {
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
			abi = FFI_THISCALL;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
		case ABI_FASTCALL: {
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
			abi = FFI_FASTCALL;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
		case ABI_MS_CDECL: {
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
			abi = FFI_MS_CDECL;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
		case ABI_PASCAL: {
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
			abi = FFI_PASCAL;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
		case ABI_REGISTER: {
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
			abi = FFI_REGISTER;
#else
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Unsupported ABI.");
#endif
		} break;
	}

	library_path = p_path;

	return OK;
}

bool FFILibrary::_get_type(const Variant &p_arg, ffi_type **r_type, bool p_allow_struct, bool p_allow_void) const {
	if (p_allow_struct && (p_arg.get_type() == Variant::STRING || p_arg.get_type() == Variant::STRING_NAME)) {
		const StringName &type_name = p_arg;
		if (structs.has(type_name)) {
			*r_type = &(structs[type_name]->struct_type);
			return true;
		}
	} else {
		int type_value = p_arg;
		if (type_value >= 0 && type_value < TYPE_MAX) {
			switch (type_value) {
				case TYPE_VOID: {
					if (!p_allow_void) {
						return false;
					}
					*r_type = &ffi_type_void;
				} break;
				case TYPE_UINT8: {
					*r_type = &ffi_type_uint8;
				} break;
				case TYPE_SINT8: {
					*r_type = &ffi_type_sint8;
				} break;
				case TYPE_UINT16: {
					*r_type = &ffi_type_uint16;
				} break;
				case TYPE_SINT16: {
					*r_type = &ffi_type_sint16;
				} break;
				case TYPE_UINT32: {
					*r_type = &ffi_type_uint32;
				} break;
				case TYPE_SINT32: {
					*r_type = &ffi_type_sint32;
				} break;
				case TYPE_UINT64: {
					*r_type = &ffi_type_uint64;
				} break;
				case TYPE_SINT64: {
					*r_type = &ffi_type_sint64;
				} break;
				case TYPE_FLOAT: {
					*r_type = &ffi_type_float;
				} break;
				case TYPE_DOUBLE: {
					*r_type = &ffi_type_double;
				} break;
				case TYPE_UCHAR: {
					*r_type = &ffi_type_uchar;
				} break;
				case TYPE_SCHAR: {
					*r_type = &ffi_type_schar;
				} break;
				case TYPE_USHORT: {
					*r_type = &ffi_type_ushort;
				} break;
				case TYPE_SSHORT: {
					*r_type = &ffi_type_sshort;
				} break;
				case TYPE_UINT: {
					*r_type = &ffi_type_uint;
				} break;
				case TYPE_SINT: {
					*r_type = &ffi_type_sint;
				} break;
				case TYPE_ULONG: {
					*r_type = &ffi_type_ulong;
				} break;
				case TYPE_SLONG: {
					*r_type = &ffi_type_slong;
				} break;
				case TYPE_POINTER: {
					*r_type = &ffi_type_pointer;
				} break;
			}
			return true;
		}
	}
	return false;
}

bool FFILibrary::_encode(const Variant &p_arg_type, const Variant &p_arg_value, uint8_t *r_out) const {
#define ENCODE_VAR(FT, CT, VT)            \
	case FT: {                            \
		*((CT *)r_out) = (VT)p_arg_value; \
	} break;

	if (p_arg_type.get_type() == Variant::STRING || p_arg_type.get_type() == Variant::STRING_NAME) {
		if (p_arg_value.get_type() == Variant::ARRAY) {
			const Array &arr = p_arg_value;
			const StringName &name = p_arg_type;
			ERR_FAIL_COND_V(!structs.has(name), false);

			const FFIStructInfo *info = structs[name];
			ERR_FAIL_COND_V(info->gd_element_types.size() != arr.size(), false);

			for (int i = 0; i < arr.size(); i++) {
				ERR_FAIL_COND_V(!_encode(info->gd_element_types[i], arr[i], r_out + info->offsets[i]), false);
			}
		} else {
			return false;
		}
	} else {
		int type_value = p_arg_type;
		switch (type_value) {
			ENCODE_VAR(TYPE_UINT8, uint8_t, uint64_t);
			ENCODE_VAR(TYPE_SINT8, int8_t, int64_t);
			ENCODE_VAR(TYPE_UINT16, uint16_t, uint64_t);
			ENCODE_VAR(TYPE_SINT16, int16_t, int64_t);
			ENCODE_VAR(TYPE_UINT32, uint32_t, uint64_t);
			ENCODE_VAR(TYPE_SINT32, int32_t, int64_t);
			ENCODE_VAR(TYPE_UINT64, uint64_t, uint64_t);
			ENCODE_VAR(TYPE_SINT64, int64_t, int64_t);
			ENCODE_VAR(TYPE_FLOAT, float, double);
			ENCODE_VAR(TYPE_DOUBLE, double, double);
			ENCODE_VAR(TYPE_UCHAR, unsigned char, uint64_t);
			ENCODE_VAR(TYPE_SCHAR, signed char, int64_t);
			ENCODE_VAR(TYPE_USHORT, unsigned short, uint64_t);
			ENCODE_VAR(TYPE_SSHORT, signed short, int64_t);
			ENCODE_VAR(TYPE_UINT, unsigned int, uint64_t);
			ENCODE_VAR(TYPE_SINT, signed int, int64_t);
			ENCODE_VAR(TYPE_ULONG, unsigned long, uint64_t);
			ENCODE_VAR(TYPE_SLONG, signed long, int64_t);
			case TYPE_POINTER: {
				*((void **)r_out) = (void *)(uint64_t)p_arg_value;
			} break;
			default:
				return false;
		}
	}
#undef ENCODE_VAR
	return true;
}

bool FFILibrary::_decode(const Variant &p_arg_type, uint8_t *p_value, Variant &r_out) const {
#define DECODE_VAR(FT, VT, CT)          \
	case FT: {                          \
		r_out = (VT)(*((CT *)p_value)); \
	} break;

	if (p_arg_type.get_type() == Variant::STRING || p_arg_type.get_type() == Variant::STRING_NAME) {
		const StringName &name = p_arg_type;
		ERR_FAIL_COND_V(!structs.has(name), false);

		const FFIStructInfo *info = structs[name];

		Array arr;
		arr.resize(info->gd_element_types.size());

		for (int i = 0; i < arr.size(); i++) {
			ERR_FAIL_COND_V(!_decode(info->gd_element_types[i], p_value + info->offsets[i], arr[i]), false);
		}

		r_out = arr;
	} else {
		int type_value = p_arg_type;
		switch (type_value) {
			DECODE_VAR(TYPE_UINT8, uint64_t, uint8_t);
			DECODE_VAR(TYPE_SINT8, int64_t, int8_t);
			DECODE_VAR(TYPE_UINT16, uint64_t, uint16_t);
			DECODE_VAR(TYPE_SINT16, int64_t, int16_t);
			DECODE_VAR(TYPE_UINT32, uint64_t, uint32_t);
			DECODE_VAR(TYPE_SINT32, int64_t, int32_t);
			DECODE_VAR(TYPE_UINT64, uint64_t, uint64_t);
			DECODE_VAR(TYPE_SINT64, int64_t, int64_t);
			DECODE_VAR(TYPE_FLOAT, double, float);
			DECODE_VAR(TYPE_DOUBLE, double, double);
			DECODE_VAR(TYPE_UCHAR, uint64_t, unsigned char);
			DECODE_VAR(TYPE_SCHAR, int64_t, signed char);
			DECODE_VAR(TYPE_USHORT, uint64_t, unsigned short);
			DECODE_VAR(TYPE_SSHORT, int64_t, signed short);
			DECODE_VAR(TYPE_UINT, uint64_t, unsigned int);
			DECODE_VAR(TYPE_SINT, int64_t, signed int);
			DECODE_VAR(TYPE_ULONG, uint64_t, unsigned long);
			DECODE_VAR(TYPE_SLONG, int64_t, signed long);
			case TYPE_POINTER: {
				r_out = (uint64_t)(*(void **)p_value);
			} break;
			case TYPE_VOID: {
				// NOP
			} break;
			default:
				return false;
		}
	}
#undef DECODE_VAR
	return true;
}

Variant FFILibrary::_bind_struct(const Variant **p_args, int p_argcount, Callable::CallError &r_call_error) {
	if (p_argcount < 2) {
		r_call_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_call_error.expected = 2;
		return false;
	}
	r_call_error.error = Callable::CallError::CALL_OK;

	ERR_FAIL_COND_V_MSG(!library_handle, false, "Library is not opened.");

	StringName name = *p_args[0];
	ERR_FAIL_COND_V_MSG(structs.has(name), false, vformat("FFI structure \"%s\" is already registered.", name));

	FFIStructInfo *info = memnew(FFIStructInfo);
	info->element_types.resize(p_argcount);
	info->gd_element_types.resize(p_argcount - 1);
	info->offsets.resize(p_argcount - 1);
	ffi_type **element_types_r = info->element_types.ptrw();
	Variant *gd_element_types_r = info->gd_element_types.ptrw();
	for (int i = 1; i < p_argcount; i++) {
		gd_element_types_r[i - 1] = *p_args[i];
		if (!_get_type(*p_args[i], &element_types_r[i - 1], true, false)) {
			memdelete(info);
			ERR_FAIL_V_MSG(false, vformat("Invalid element %d type, should be FFIType or registered structure name.", i - 1));
		}
	}
	element_types_r[p_argcount - 1] = nullptr;
	info->struct_type.size = 0;
	info->struct_type.alignment = 0;
	info->struct_type.type = FFI_TYPE_STRUCT;
	info->struct_type.elements = info->element_types.ptrw();

	if (ffi_get_struct_offsets(abi, &(info->struct_type), info->offsets.ptrw()) != FFI_OK) {
		memdelete(info);
		ERR_FAIL_V_MSG(false, "Unabe to get structure offsets.");
	}

	structs[name] = info;

	return true;
}

int FFILibrary::get_struct_size(const StringName &p_name) const {
	ERR_FAIL_COND_V(!structs.has(p_name), 0);
	const FFIStructInfo *info = structs[p_name];

	return info->struct_type.size;
}

PackedInt64Array FFILibrary::get_struct_offsets(const StringName &p_name) const {
	ERR_FAIL_COND_V(!structs.has(p_name), PackedInt64Array());
	const FFIStructInfo *info = structs[p_name];

	PackedInt64Array arr;
	for (int i = 0; i < info->offsets.size(); i++) {
		arr.push_back(info->offsets[i]);
	}
	return arr;
}

Callable FFILibrary::get_method(const StringName &p_name) const {
	ERR_FAIL_COND_V_MSG(!library_handle, Callable(), "Library is not opened.");
	ERR_FAIL_COND_V_MSG(!methods.has(p_name), Callable(), vformat("FFI method \"%s\" is not registered.", p_name));

	FFIMethodInfo *info = methods[p_name];
	return Callable(memnew(FFICallable(Ref<FFILibrary>(this), info, p_name)));
}

Variant FFILibrary::_bind_method(const Variant **p_args, int p_argcount, Callable::CallError &r_call_error) {
	if (p_argcount < 2) {
		r_call_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_call_error.expected = 2;
		return Callable();
	}
	r_call_error.error = Callable::CallError::CALL_OK;

	ERR_FAIL_COND_V_MSG(!library_handle, Callable(), "Library is not opened.");

	StringName name = *p_args[1];
	ERR_FAIL_COND_V_MSG(methods.has(name), Callable(), vformat("FFI method \"%s\" is already registered.", name));

	FFIMethodInfo *info = memnew(FFIMethodInfo);
	info->gd_ret_type = *p_args[0];
	if (!_get_type(*p_args[0], &(info->ret_type), true, true)) {
		memdelete(info);
		ERR_FAIL_V_MSG(Callable(), "Invalid return value type, should be FFIType or registered structure name.");
	}

	info->arg_types.resize(p_argcount - 2);
	info->gd_arg_types.resize(p_argcount - 2);
	ffi_type **arg_types_r = info->arg_types.ptrw();
	Variant *gd_arg_types_r = info->gd_arg_types.ptrw();
	for (int i = 2; i < p_argcount; i++) {
		gd_arg_types_r[i - 2] = *p_args[i];
		if (!_get_type(*p_args[i], &arg_types_r[i - 2], true, false)) {
			memdelete(info);
			ERR_FAIL_V_MSG(Callable(), vformat("Invalid argument %d type, should be FFIType or registered structure name.", i - 2));
		}
	}

	if (OS::get_singleton()->get_dynamic_library_symbol_handle(library_handle, name, info->handle, false) != OK) {
		memdelete(info);
		ERR_FAIL_V_MSG(Callable(), vformat("Unable to get method \"%s\" address.", name));
	}
	if (ffi_prep_cif(&(info->cif), abi, info->arg_types.size(), info->ret_type, info->arg_types.ptrw()) != FFI_OK) {
		memdelete(info);
		ERR_FAIL_V_MSG(Callable(), "Unable to bind FFI wrapper.");
	}

	methods[name] = info;

	return Callable(memnew(FFICallable(Ref<FFILibrary>(this), info, name)));
}

void FFILibrary::_callptr(FFILibrary::FFIMethodInfo *p_method, const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	if (!library_handle || !p_method) {
		r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		return;
	}

	if (p_argcount < p_method->arg_types.size()) {
		r_call_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_call_error.expected = p_method->arg_types.size();
		return;
	} else if (p_argcount > p_method->arg_types.size()) {
		r_call_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_call_error.expected = p_method->arg_types.size();
		return;
	}

#define ALLOC_VAR(FT, CT)                 \
	case FT: {                            \
		values_r[i] = alloca(sizeof(CT)); \
	} break;

	Vector<void *> values;
	values.resize(p_argcount);
	void **values_r = values.ptrw();
	for (int i = 0; i < p_argcount; i++) {
		if (p_method->gd_arg_types[i].get_type() == Variant::STRING || p_method->gd_arg_types[i].get_type() == Variant::STRING_NAME) {
			const StringName &name = p_method->gd_arg_types[i];
			if (!structs.has(name)) {
				r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_call_error.argument = i;
				return;
			}

			const FFIStructInfo *st_info = structs[name];
			values_r[i] = alloca(st_info->struct_type.size);
		} else {
			int type_value = p_method->gd_arg_types[i];
			switch (type_value) {
				ALLOC_VAR(TYPE_UINT8, uint8_t);
				ALLOC_VAR(TYPE_SINT8, int8_t);
				ALLOC_VAR(TYPE_UINT16, uint16_t);
				ALLOC_VAR(TYPE_SINT16, int16_t);
				ALLOC_VAR(TYPE_UINT32, uint32_t);
				ALLOC_VAR(TYPE_SINT32, int32_t);
				ALLOC_VAR(TYPE_UINT64, uint64_t);
				ALLOC_VAR(TYPE_SINT64, int64_t);
				ALLOC_VAR(TYPE_FLOAT, float);
				ALLOC_VAR(TYPE_DOUBLE, double);
				ALLOC_VAR(TYPE_UCHAR, unsigned char);
				ALLOC_VAR(TYPE_SCHAR, signed char);
				ALLOC_VAR(TYPE_USHORT, unsigned short);
				ALLOC_VAR(TYPE_SSHORT, signed short);
				ALLOC_VAR(TYPE_UINT, unsigned int);
				ALLOC_VAR(TYPE_SINT, signed int);
				ALLOC_VAR(TYPE_ULONG, unsigned long);
				ALLOC_VAR(TYPE_SLONG, signed long);
				ALLOC_VAR(TYPE_POINTER, void *);
				default: {
					r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_call_error.argument = i;
					return;
				}
			}
		}
		if (!_encode(p_method->gd_arg_types[i], *p_arguments[i], (uint8_t *)values_r[i])) {
			r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_call_error.argument = i;
			return;
		}
	}
#undef ALLOC_VAR

	void *rc = nullptr;
	if (p_method->gd_ret_type.get_type() == Variant::STRING || p_method->gd_ret_type.get_type() == Variant::STRING_NAME) {
		const StringName &name = p_method->gd_ret_type;
		ERR_FAIL_COND(!structs.has(name));

		const FFIStructInfo *st_info = structs[name];
		rc = alloca(st_info->struct_type.size);
	} else {
		rc = alloca(sizeof(ffi_arg));
	}

	ffi_call(&(p_method->cif), (void (*)())p_method->handle, rc, values.ptrw());
	if (!_decode(p_method->gd_ret_type, (uint8_t *)rc, r_return_value)) {
		r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		return;
	}

	r_call_error.error = Callable::CallError::CALL_OK;
}

uint32_t FFILibrary::_struct_hash(const StringName &p_name) const {
	if (!structs.has(p_name)) {
		return 0;
	}
	const FFIStructInfo *info = structs[p_name];
	uint32_t hash = p_name.hash();
	for (int i = 0; i < info->gd_element_types.size(); i++) {
		hash = hash_murmur3_one_64((int64_t)info->gd_element_types[i], hash);
	}

	return hash;
}

uint32_t FFILibrary::_method_hash(const StringName &p_name) const {
	if (!methods.has(p_name)) {
		return 0;
	}
	const FFIMethodInfo *info = methods[p_name];
	uint32_t hash = p_name.hash();
	for (int i = 0; i < info->arg_types.size(); i++) {
		hash = hash_murmur3_one_64((int64_t)info->arg_types[i], hash);
	}
	hash = hash_murmur3_one_64((int64_t)info->ret_type, hash);
	hash = hash_murmur3_one_64((int64_t)info->handle, hash);

	return hash;
}

FFILibrary::FFILibrary() {
}

FFILibrary::~FFILibrary() {
	if (library_handle) {
		OS::get_singleton()->close_dynamic_library(library_handle);
		library_handle = nullptr;
	}
	for (HashMap<StringName, FFIMethodInfo *>::Iterator E = methods.begin(); E; ++E) {
		memdelete(E->value);
	}
	methods.clear();
	for (HashMap<StringName, FFIStructInfo *>::Iterator E = structs.begin(); E; ++E) {
		memdelete(E->value);
	}
	structs.clear();
}

/**************************************************************************/

bool FFICallable::_equal_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	return p_a->hash() == p_b->hash();
}

bool FFICallable::_less_func(const CallableCustom *p_a, const CallableCustom *p_b) {
	return p_a->hash() < p_b->hash();
}

bool FFICallable::is_valid() const {
	return library.is_valid();
}

uint32_t FFICallable::hash() const {
	if (library.is_null()) {
		return 0;
	}
	return library->_method_hash(name);
}

String FFICallable::get_as_text() const {
	if (library.is_null()) {
		return "Invalid (FFI)";
	}
	return library->get_path().get_file() + "::" + String(name) + " (FFI)";
}

CallableCustom::CompareEqualFunc FFICallable::get_compare_equal_func() const {
	return _equal_func;
}

CallableCustom::CompareLessFunc FFICallable::get_compare_less_func() const {
	return _less_func;
}

ObjectID FFICallable::get_object() const {
	return ObjectID();
}

StringName FFICallable::get_method() const {
	return name;
}

void FFICallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	if (library.is_valid()) {
		library->_callptr(method, p_arguments, p_argcount, r_return_value, r_call_error);
	}
}

FFICallable::FFICallable(Ref<FFILibrary> p_library, FFILibrary::FFIMethodInfo *p_method, const StringName &p_name) {
	library = p_library;
	method = p_method;
	name = p_name;
}
