/**************************************************************************/
/*  open_xr_spatial_anchor_capability.cpp                                 */
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

#include <godot_cpp/classes/open_xr_spatial_anchor_capability.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/open_xr_anchor_tracker.hpp>
#include <godot_cpp/classes/open_xr_future_result.hpp>
#include <godot_cpp/variant/transform3d.hpp>

namespace godot {

bool OpenXRSpatialAnchorCapability::is_spatial_anchor_supported() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("is_spatial_anchor_supported")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OpenXRSpatialAnchorCapability::is_spatial_persistence_supported() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("is_spatial_persistence_supported")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OpenXRSpatialAnchorCapability::is_persistence_scope_supported(OpenXRSpatialAnchorCapability::PersistenceScope p_scope) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("is_persistence_scope_supported")._native_ptr(), 3651771626);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_scope_encoded;
	PtrToArg<int64_t>::encode(p_scope, &p_scope_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_scope_encoded);
}

Ref<OpenXRFutureResult> OpenXRSpatialAnchorCapability::create_persistence_context(OpenXRSpatialAnchorCapability::PersistenceScope p_scope, const Callable &p_user_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("create_persistence_context")._native_ptr(), 856276630);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRFutureResult>()));
	int64_t p_scope_encoded;
	PtrToArg<int64_t>::encode(p_scope, &p_scope_encoded);
	return Ref<OpenXRFutureResult>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRFutureResult>(_gde_method_bind, _owner, &p_scope_encoded, &p_user_callback));
}

uint64_t OpenXRSpatialAnchorCapability::get_persistence_context_handle(const RID &p_persistence_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("get_persistence_context_handle")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_persistence_context);
}

void OpenXRSpatialAnchorCapability::free_persistence_context(const RID &p_persistence_context) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("free_persistence_context")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_persistence_context);
}

Ref<OpenXRAnchorTracker> OpenXRSpatialAnchorCapability::create_new_anchor(const Transform3D &p_transform, const RID &p_spatial_context) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("create_new_anchor")._native_ptr(), 607100373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRAnchorTracker>()));
	return Ref<OpenXRAnchorTracker>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRAnchorTracker>(_gde_method_bind, _owner, &p_transform, &p_spatial_context));
}

void OpenXRSpatialAnchorCapability::remove_anchor(const Ref<OpenXRAnchorTracker> &p_anchor_tracker) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("remove_anchor")._native_ptr(), 3579451518);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_anchor_tracker != nullptr ? &p_anchor_tracker->_owner : nullptr));
}

Ref<OpenXRFutureResult> OpenXRSpatialAnchorCapability::persist_anchor(const Ref<OpenXRAnchorTracker> &p_anchor_tracker, const RID &p_persistence_context, const Callable &p_user_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("persist_anchor")._native_ptr(), 4244202513);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRFutureResult>()));
	return Ref<OpenXRFutureResult>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRFutureResult>(_gde_method_bind, _owner, (p_anchor_tracker != nullptr ? &p_anchor_tracker->_owner : nullptr), &p_persistence_context, &p_user_callback));
}

Ref<OpenXRFutureResult> OpenXRSpatialAnchorCapability::unpersist_anchor(const Ref<OpenXRAnchorTracker> &p_anchor_tracker, const RID &p_persistence_context, const Callable &p_user_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialAnchorCapability::get_class_static()._native_ptr(), StringName("unpersist_anchor")._native_ptr(), 4244202513);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRFutureResult>()));
	return Ref<OpenXRFutureResult>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRFutureResult>(_gde_method_bind, _owner, (p_anchor_tracker != nullptr ? &p_anchor_tracker->_owner : nullptr), &p_persistence_context, &p_user_callback));
}

} // namespace godot
