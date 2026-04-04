/**************************************************************************/
/*  sky.cpp                                                               */
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

#include <godot_cpp/classes/sky.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/material.hpp>

namespace godot {

void Sky::set_radiance_size(Sky::RadianceSize p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Sky::get_class_static()._native_ptr(), StringName("set_radiance_size")._native_ptr(), 1512957179);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

Sky::RadianceSize Sky::get_radiance_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Sky::get_class_static()._native_ptr(), StringName("get_radiance_size")._native_ptr(), 2708733976);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Sky::RadianceSize(0)));
	return (Sky::RadianceSize)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Sky::set_process_mode(Sky::ProcessMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Sky::get_class_static()._native_ptr(), StringName("set_process_mode")._native_ptr(), 875986769);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Sky::ProcessMode Sky::get_process_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Sky::get_class_static()._native_ptr(), StringName("get_process_mode")._native_ptr(), 731245043);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Sky::ProcessMode(0)));
	return (Sky::ProcessMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Sky::set_material(const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Sky::get_class_static()._native_ptr(), StringName("set_material")._native_ptr(), 2757459619);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<Material> Sky::get_material() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Sky::get_class_static()._native_ptr(), StringName("get_material")._native_ptr(), 5934680);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner));
}

} // namespace godot
