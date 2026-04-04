/**************************************************************************/
/*  camera3d.cpp                                                          */
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

#include <godot_cpp/classes/camera3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera_attributes.hpp>
#include <godot_cpp/classes/compositor.hpp>
#include <godot_cpp/classes/environment.hpp>

namespace godot {

Vector3 Camera3D::project_ray_normal(const Vector2 &p_screen_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("project_ray_normal")._native_ptr(), 1718073306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_screen_point);
}

Vector3 Camera3D::project_local_ray_normal(const Vector2 &p_screen_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("project_local_ray_normal")._native_ptr(), 1718073306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_screen_point);
}

Vector3 Camera3D::project_ray_origin(const Vector2 &p_screen_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("project_ray_origin")._native_ptr(), 1718073306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_screen_point);
}

Vector2 Camera3D::unproject_position(const Vector3 &p_world_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("unproject_position")._native_ptr(), 3758901831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_world_point);
}

bool Camera3D::is_position_behind(const Vector3 &p_world_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("is_position_behind")._native_ptr(), 3108956480);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_world_point);
}

Vector3 Camera3D::project_position(const Vector2 &p_screen_point, float p_z_depth) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("project_position")._native_ptr(), 2171975744);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	double p_z_depth_encoded;
	PtrToArg<double>::encode(p_z_depth, &p_z_depth_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_screen_point, &p_z_depth_encoded);
}

void Camera3D::set_perspective(float p_fov, float p_z_near, float p_z_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_perspective")._native_ptr(), 2385087082);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fov_encoded;
	PtrToArg<double>::encode(p_fov, &p_fov_encoded);
	double p_z_near_encoded;
	PtrToArg<double>::encode(p_z_near, &p_z_near_encoded);
	double p_z_far_encoded;
	PtrToArg<double>::encode(p_z_far, &p_z_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fov_encoded, &p_z_near_encoded, &p_z_far_encoded);
}

void Camera3D::set_orthogonal(float p_size, float p_z_near, float p_z_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_orthogonal")._native_ptr(), 2385087082);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	double p_z_near_encoded;
	PtrToArg<double>::encode(p_z_near, &p_z_near_encoded);
	double p_z_far_encoded;
	PtrToArg<double>::encode(p_z_far, &p_z_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded, &p_z_near_encoded, &p_z_far_encoded);
}

void Camera3D::set_frustum(float p_size, const Vector2 &p_offset, float p_z_near, float p_z_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_frustum")._native_ptr(), 354890663);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	double p_z_near_encoded;
	PtrToArg<double>::encode(p_z_near, &p_z_near_encoded);
	double p_z_far_encoded;
	PtrToArg<double>::encode(p_z_far, &p_z_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded, &p_offset, &p_z_near_encoded, &p_z_far_encoded);
}

void Camera3D::make_current() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("make_current")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Camera3D::clear_current(bool p_enable_next) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("clear_current")._native_ptr(), 3216645846);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_next_encoded;
	PtrToArg<bool>::encode(p_enable_next, &p_enable_next_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_next_encoded);
}

void Camera3D::set_current(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_current")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Camera3D::is_current() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("is_current")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Transform3D Camera3D::get_camera_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_camera_transform")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

Projection Camera3D::get_camera_projection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_camera_projection")._native_ptr(), 2910717950);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Projection()));
	return ::godot::internal::_call_native_mb_ret<Projection>(_gde_method_bind, _owner);
}

float Camera3D::get_fov() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_fov")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Vector2 Camera3D::get_frustum_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_frustum_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

float Camera3D::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float Camera3D::get_far() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_far")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float Camera3D::get_near() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_near")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Camera3D::set_fov(float p_fov) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_fov")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fov_encoded;
	PtrToArg<double>::encode(p_fov, &p_fov_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fov_encoded);
}

void Camera3D::set_frustum_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_frustum_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

void Camera3D::set_size(float p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

void Camera3D::set_far(float p_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_far")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_far_encoded;
	PtrToArg<double>::encode(p_far, &p_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_far_encoded);
}

void Camera3D::set_near(float p_near) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_near")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_near_encoded;
	PtrToArg<double>::encode(p_near, &p_near_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_near_encoded);
}

Camera3D::ProjectionType Camera3D::get_projection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_projection")._native_ptr(), 2624185235);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Camera3D::ProjectionType(0)));
	return (Camera3D::ProjectionType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Camera3D::set_projection(Camera3D::ProjectionType p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_projection")._native_ptr(), 4218540108);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

void Camera3D::set_h_offset(float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_h_offset")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset_encoded);
}

float Camera3D::get_h_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_h_offset")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Camera3D::set_v_offset(float p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_v_offset")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset_encoded);
}

float Camera3D::get_v_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_v_offset")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Camera3D::set_cull_mask(uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_cull_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask_encoded);
}

uint32_t Camera3D::get_cull_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_cull_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Camera3D::set_environment(const Ref<Environment> &p_env) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_environment")._native_ptr(), 4143518816);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_env != nullptr ? &p_env->_owner : nullptr));
}

Ref<Environment> Camera3D::get_environment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_environment")._native_ptr(), 3082064660);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Environment>()));
	return Ref<Environment>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Environment>(_gde_method_bind, _owner));
}

void Camera3D::set_attributes(const Ref<CameraAttributes> &p_env) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_attributes")._native_ptr(), 2817810567);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_env != nullptr ? &p_env->_owner : nullptr));
}

Ref<CameraAttributes> Camera3D::get_attributes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_attributes")._native_ptr(), 3921283215);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<CameraAttributes>()));
	return Ref<CameraAttributes>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<CameraAttributes>(_gde_method_bind, _owner));
}

void Camera3D::set_compositor(const Ref<Compositor> &p_compositor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_compositor")._native_ptr(), 1586754307);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_compositor != nullptr ? &p_compositor->_owner : nullptr));
}

Ref<Compositor> Camera3D::get_compositor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_compositor")._native_ptr(), 3647707413);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Compositor>()));
	return Ref<Compositor>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Compositor>(_gde_method_bind, _owner));
}

void Camera3D::set_keep_aspect_mode(Camera3D::KeepAspect p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_keep_aspect_mode")._native_ptr(), 1740651252);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Camera3D::KeepAspect Camera3D::get_keep_aspect_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_keep_aspect_mode")._native_ptr(), 2790278316);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Camera3D::KeepAspect(0)));
	return (Camera3D::KeepAspect)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Camera3D::set_doppler_tracking(Camera3D::DopplerTracking p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_doppler_tracking")._native_ptr(), 3109431270);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Camera3D::DopplerTracking Camera3D::get_doppler_tracking() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_doppler_tracking")._native_ptr(), 1584483649);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Camera3D::DopplerTracking(0)));
	return (Camera3D::DopplerTracking)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

TypedArray<Plane> Camera3D::get_frustum() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_frustum")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Plane>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Plane>>(_gde_method_bind, _owner);
}

bool Camera3D::is_position_in_frustum(const Vector3 &p_world_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("is_position_in_frustum")._native_ptr(), 3108956480);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_world_point);
}

RID Camera3D::get_camera_rid() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_camera_rid")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID Camera3D::get_pyramid_shape_rid() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_pyramid_shape_rid")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void Camera3D::set_cull_mask_value(int32_t p_layer_number, bool p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("set_cull_mask_value")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	int8_t p_value_encoded;
	PtrToArg<bool>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_number_encoded, &p_value_encoded);
}

bool Camera3D::get_cull_mask_value(int32_t p_layer_number) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Camera3D::get_class_static()._native_ptr(), StringName("get_cull_mask_value")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_number_encoded);
}

} // namespace godot
