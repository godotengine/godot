/**************************************************************************/
/*  open_xr_spatial_entity_tracker.cpp                                    */
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

#include <godot_cpp/classes/open_xr_spatial_entity_tracker.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void OpenXRSpatialEntityTracker::set_entity(const RID &p_entity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityTracker::get_class_static()._native_ptr(), StringName("set_entity")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_entity);
}

RID OpenXRSpatialEntityTracker::get_entity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityTracker::get_class_static()._native_ptr(), StringName("get_entity")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void OpenXRSpatialEntityTracker::set_spatial_tracking_state(OpenXRSpatialEntityTracker::EntityTrackingState p_spatial_tracking_state) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityTracker::get_class_static()._native_ptr(), StringName("set_spatial_tracking_state")._native_ptr(), 2170234447);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_spatial_tracking_state_encoded;
	PtrToArg<int64_t>::encode(p_spatial_tracking_state, &p_spatial_tracking_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_spatial_tracking_state_encoded);
}

OpenXRSpatialEntityTracker::EntityTrackingState OpenXRSpatialEntityTracker::get_spatial_tracking_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityTracker::get_class_static()._native_ptr(), StringName("get_spatial_tracking_state")._native_ptr(), 3351876560);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRSpatialEntityTracker::EntityTrackingState(0)));
	return (OpenXRSpatialEntityTracker::EntityTrackingState)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
