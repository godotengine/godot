/**************************************************************************/
/*  xr_server.cpp                                                         */
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

#include <godot_cpp/classes/xr_server.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/xr_interface.hpp>
#include <godot_cpp/classes/xr_tracker.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

XRServer *XRServer::singleton = nullptr;

XRServer *XRServer::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(XRServer::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<XRServer *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &XRServer::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(XRServer::get_class_static(), singleton);
		}
	}
	return singleton;
}

XRServer::~XRServer() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(XRServer::get_class_static());
		singleton = nullptr;
	}
}

double XRServer::get_world_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_world_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void XRServer::set_world_scale(double p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("set_world_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

Transform3D XRServer::get_world_origin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_world_origin")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

void XRServer::set_world_origin(const Transform3D &p_world_origin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("set_world_origin")._native_ptr(), 2952846383);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_world_origin);
}

Transform3D XRServer::get_reference_frame() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_reference_frame")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

void XRServer::clear_reference_frame() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("clear_reference_frame")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void XRServer::center_on_hmd(XRServer::RotationMode p_rotation_mode, bool p_keep_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("center_on_hmd")._native_ptr(), 1450904707);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_rotation_mode_encoded;
	PtrToArg<int64_t>::encode(p_rotation_mode, &p_rotation_mode_encoded);
	int8_t p_keep_height_encoded;
	PtrToArg<bool>::encode(p_keep_height, &p_keep_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rotation_mode_encoded, &p_keep_height_encoded);
}

Transform3D XRServer::get_hmd_transform() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_hmd_transform")._native_ptr(), 4183770049);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

void XRServer::set_camera_locked_to_origin(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("set_camera_locked_to_origin")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool XRServer::is_camera_locked_to_origin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("is_camera_locked_to_origin")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void XRServer::add_interface(const Ref<XRInterface> &p_interface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("add_interface")._native_ptr(), 1898711491);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_interface != nullptr ? &p_interface->_owner : nullptr));
}

int32_t XRServer::get_interface_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_interface_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void XRServer::remove_interface(const Ref<XRInterface> &p_interface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("remove_interface")._native_ptr(), 1898711491);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_interface != nullptr ? &p_interface->_owner : nullptr));
}

Ref<XRInterface> XRServer::get_interface(int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_interface")._native_ptr(), 4237347919);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<XRInterface>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<XRInterface>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<XRInterface>(_gde_method_bind, _owner, &p_idx_encoded));
}

TypedArray<Dictionary> XRServer::get_interfaces() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_interfaces")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner);
}

Ref<XRInterface> XRServer::find_interface(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("find_interface")._native_ptr(), 1395192955);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<XRInterface>()));
	return Ref<XRInterface>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<XRInterface>(_gde_method_bind, _owner, &p_name));
}

void XRServer::add_tracker(const Ref<XRTracker> &p_tracker) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("add_tracker")._native_ptr(), 684804553);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_tracker != nullptr ? &p_tracker->_owner : nullptr));
}

void XRServer::remove_tracker(const Ref<XRTracker> &p_tracker) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("remove_tracker")._native_ptr(), 684804553);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_tracker != nullptr ? &p_tracker->_owner : nullptr));
}

Dictionary XRServer::get_trackers(int32_t p_tracker_types) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_trackers")._native_ptr(), 3554694381);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_tracker_types_encoded;
	PtrToArg<int64_t>::encode(p_tracker_types, &p_tracker_types_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_tracker_types_encoded);
}

Ref<XRTracker> XRServer::get_tracker(const StringName &p_tracker_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_tracker")._native_ptr(), 147382240);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<XRTracker>()));
	return Ref<XRTracker>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<XRTracker>(_gde_method_bind, _owner, &p_tracker_name));
}

Ref<XRInterface> XRServer::get_primary_interface() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("get_primary_interface")._native_ptr(), 2143545064);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<XRInterface>()));
	return Ref<XRInterface>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<XRInterface>(_gde_method_bind, _owner));
}

void XRServer::set_primary_interface(const Ref<XRInterface> &p_interface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(XRServer::get_class_static()._native_ptr(), StringName("set_primary_interface")._native_ptr(), 1898711491);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_interface != nullptr ? &p_interface->_owner : nullptr));
}

} // namespace godot
