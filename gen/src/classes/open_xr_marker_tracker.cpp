/**************************************************************************/
/*  open_xr_marker_tracker.cpp                                            */
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

#include <godot_cpp/classes/open_xr_marker_tracker.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void OpenXRMarkerTracker::set_bounds_size(const Vector2 &p_bounds_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRMarkerTracker::get_class_static()._native_ptr(), StringName("set_bounds_size")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bounds_size);
}

Vector2 OpenXRMarkerTracker::get_bounds_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRMarkerTracker::get_class_static()._native_ptr(), StringName("get_bounds_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void OpenXRMarkerTracker::set_marker_type(OpenXRSpatialComponentMarkerList::MarkerType p_marker_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRMarkerTracker::get_class_static()._native_ptr(), StringName("set_marker_type")._native_ptr(), 2156241362);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_marker_type_encoded;
	PtrToArg<int64_t>::encode(p_marker_type, &p_marker_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_marker_type_encoded);
}

OpenXRSpatialComponentMarkerList::MarkerType OpenXRMarkerTracker::get_marker_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRMarkerTracker::get_class_static()._native_ptr(), StringName("get_marker_type")._native_ptr(), 612702862);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRSpatialComponentMarkerList::MarkerType(0)));
	return (OpenXRSpatialComponentMarkerList::MarkerType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRMarkerTracker::set_marker_id(uint32_t p_marker_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRMarkerTracker::get_class_static()._native_ptr(), StringName("set_marker_id")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_marker_id_encoded;
	PtrToArg<int64_t>::encode(p_marker_id, &p_marker_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_marker_id_encoded);
}

uint32_t OpenXRMarkerTracker::get_marker_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRMarkerTracker::get_class_static()._native_ptr(), StringName("get_marker_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRMarkerTracker::set_marker_data(const Variant &p_marker_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRMarkerTracker::get_class_static()._native_ptr(), StringName("set_marker_data")._native_ptr(), 1114965689);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_marker_data);
}

Variant OpenXRMarkerTracker::get_marker_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRMarkerTracker::get_class_static()._native_ptr(), StringName("get_marker_data")._native_ptr(), 1214101251);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner);
}

} // namespace godot
