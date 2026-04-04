/**************************************************************************/
/*  open_xr_spatial_capability_configuration_april_tag.cpp                */
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

#include <godot_cpp/classes/open_xr_spatial_capability_configuration_april_tag.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

PackedInt64Array OpenXRSpatialCapabilityConfigurationAprilTag::get_enabled_components() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialCapabilityConfigurationAprilTag::get_class_static()._native_ptr(), StringName("get_enabled_components")._native_ptr(), 235988956);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt64Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt64Array>(_gde_method_bind, _owner);
}

void OpenXRSpatialCapabilityConfigurationAprilTag::set_april_dict(OpenXRSpatialCapabilityConfigurationAprilTag::AprilTagDict p_april_dict) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialCapabilityConfigurationAprilTag::get_class_static()._native_ptr(), StringName("set_april_dict")._native_ptr(), 3902905799);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_april_dict_encoded;
	PtrToArg<int64_t>::encode(p_april_dict, &p_april_dict_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_april_dict_encoded);
}

OpenXRSpatialCapabilityConfigurationAprilTag::AprilTagDict OpenXRSpatialCapabilityConfigurationAprilTag::get_april_dict() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialCapabilityConfigurationAprilTag::get_class_static()._native_ptr(), StringName("get_april_dict")._native_ptr(), 440273016);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRSpatialCapabilityConfigurationAprilTag::AprilTagDict(0)));
	return (OpenXRSpatialCapabilityConfigurationAprilTag::AprilTagDict)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
