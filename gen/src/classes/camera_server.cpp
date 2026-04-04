/**************************************************************************/
/*  camera_server.cpp                                                     */
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

#include <godot_cpp/classes/camera_server.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera_feed.hpp>

namespace godot {

CameraServer *CameraServer::singleton = nullptr;

CameraServer *CameraServer::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(CameraServer::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<CameraServer *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &CameraServer::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(CameraServer::get_class_static(), singleton);
		}
	}
	return singleton;
}

CameraServer::~CameraServer() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(CameraServer::get_class_static());
		singleton = nullptr;
	}
}

void CameraServer::set_monitoring_feeds(bool p_is_monitoring_feeds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraServer::get_class_static()._native_ptr(), StringName("set_monitoring_feeds")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_is_monitoring_feeds_encoded;
	PtrToArg<bool>::encode(p_is_monitoring_feeds, &p_is_monitoring_feeds_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_is_monitoring_feeds_encoded);
}

bool CameraServer::is_monitoring_feeds() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraServer::get_class_static()._native_ptr(), StringName("is_monitoring_feeds")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<CameraFeed> CameraServer::get_feed(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraServer::get_class_static()._native_ptr(), StringName("get_feed")._native_ptr(), 361927068);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<CameraFeed>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<CameraFeed>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<CameraFeed>(_gde_method_bind, _owner, &p_index_encoded));
}

int32_t CameraServer::get_feed_count() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraServer::get_class_static()._native_ptr(), StringName("get_feed_count")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

TypedArray<Ref<CameraFeed>> CameraServer::feeds() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraServer::get_class_static()._native_ptr(), StringName("feeds")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<CameraFeed>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<CameraFeed>>>(_gde_method_bind, _owner);
}

void CameraServer::add_feed(const Ref<CameraFeed> &p_feed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraServer::get_class_static()._native_ptr(), StringName("add_feed")._native_ptr(), 3204782488);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_feed != nullptr ? &p_feed->_owner : nullptr));
}

void CameraServer::remove_feed(const Ref<CameraFeed> &p_feed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraServer::get_class_static()._native_ptr(), StringName("remove_feed")._native_ptr(), 3204782488);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_feed != nullptr ? &p_feed->_owner : nullptr));
}

} // namespace godot
