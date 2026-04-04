/**************************************************************************/
/*  pck_packer.cpp                                                        */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/pck_packer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Error PCKPacker::pck_start(const String &p_pck_path, int32_t p_alignment, const String &p_key, bool p_encrypt_directory) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PCKPacker::get_class_static()._native_ptr(), StringName("pck_start")._native_ptr(), 508410629);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	int8_t p_encrypt_directory_encoded;
	PtrToArg<bool>::encode(p_encrypt_directory, &p_encrypt_directory_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_pck_path, &p_alignment_encoded, &p_key, &p_encrypt_directory_encoded);
}

Error PCKPacker::add_file(const String &p_target_path, const String &p_source_path, bool p_encrypt) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PCKPacker::get_class_static()._native_ptr(), StringName("add_file")._native_ptr(), 2215643711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_encrypt_encoded;
	PtrToArg<bool>::encode(p_encrypt, &p_encrypt_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_target_path, &p_source_path, &p_encrypt_encoded);
}

Error PCKPacker::add_file_removal(const String &p_target_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PCKPacker::get_class_static()._native_ptr(), StringName("add_file_removal")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_target_path);
}

Error PCKPacker::flush(bool p_verbose) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PCKPacker::get_class_static()._native_ptr(), StringName("flush")._native_ptr(), 1633102583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_verbose_encoded;
	PtrToArg<bool>::encode(p_verbose, &p_verbose_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_verbose_encoded);
}

} // namespace godot
