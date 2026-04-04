/**************************************************************************/
/*  open_xr_spatial_capability_configuration_plane_tracking.cpp           */
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

#include <godot_cpp/classes/open_xr_spatial_capability_configuration_plane_tracking.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

bool OpenXRSpatialCapabilityConfigurationPlaneTracking::supports_mesh_2d() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialCapabilityConfigurationPlaneTracking::get_class_static()._native_ptr(), StringName("supports_mesh_2d")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OpenXRSpatialCapabilityConfigurationPlaneTracking::supports_polygons() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialCapabilityConfigurationPlaneTracking::get_class_static()._native_ptr(), StringName("supports_polygons")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OpenXRSpatialCapabilityConfigurationPlaneTracking::supports_labels() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialCapabilityConfigurationPlaneTracking::get_class_static()._native_ptr(), StringName("supports_labels")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

PackedInt64Array OpenXRSpatialCapabilityConfigurationPlaneTracking::get_enabled_components() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialCapabilityConfigurationPlaneTracking::get_class_static()._native_ptr(), StringName("get_enabled_components")._native_ptr(), 235988956);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt64Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt64Array>(_gde_method_bind, _owner);
}

} // namespace godot
