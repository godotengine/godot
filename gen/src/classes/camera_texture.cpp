/**************************************************************************/
/*  camera_texture.cpp                                                    */
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

#include <godot_cpp/classes/camera_texture.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void CameraTexture::set_camera_feed_id(int32_t p_feed_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraTexture::get_class_static()._native_ptr(), StringName("set_camera_feed_id")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_feed_id_encoded;
	PtrToArg<int64_t>::encode(p_feed_id, &p_feed_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_feed_id_encoded);
}

int32_t CameraTexture::get_camera_feed_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraTexture::get_class_static()._native_ptr(), StringName("get_camera_feed_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CameraTexture::set_which_feed(CameraServer::FeedImage p_which_feed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraTexture::get_class_static()._native_ptr(), StringName("set_which_feed")._native_ptr(), 1595299230);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_which_feed_encoded;
	PtrToArg<int64_t>::encode(p_which_feed, &p_which_feed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_which_feed_encoded);
}

CameraServer::FeedImage CameraTexture::get_which_feed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraTexture::get_class_static()._native_ptr(), StringName("get_which_feed")._native_ptr(), 91039457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CameraServer::FeedImage(0)));
	return (CameraServer::FeedImage)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CameraTexture::set_camera_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraTexture::get_class_static()._native_ptr(), StringName("set_camera_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

bool CameraTexture::get_camera_active() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CameraTexture::get_class_static()._native_ptr(), StringName("get_camera_active")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
