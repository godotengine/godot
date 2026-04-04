/**************************************************************************/
/*  input.cpp                                                             */
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

#include <godot_cpp/classes/input.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

Input *Input::singleton = nullptr;

Input *Input::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(Input::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<Input *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &Input::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(Input::get_class_static(), singleton);
		}
	}
	return singleton;
}

Input::~Input() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(Input::get_class_static());
		singleton = nullptr;
	}
}

bool Input::is_anything_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_anything_pressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Input::is_key_pressed(Key p_keycode) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_key_pressed")._native_ptr(), 1938909964);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_keycode_encoded;
	PtrToArg<int64_t>::encode(p_keycode, &p_keycode_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_keycode_encoded);
}

bool Input::is_physical_key_pressed(Key p_keycode) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_physical_key_pressed")._native_ptr(), 1938909964);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_keycode_encoded;
	PtrToArg<int64_t>::encode(p_keycode, &p_keycode_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_keycode_encoded);
}

bool Input::is_key_label_pressed(Key p_keycode) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_key_label_pressed")._native_ptr(), 1938909964);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_keycode_encoded;
	PtrToArg<int64_t>::encode(p_keycode, &p_keycode_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_keycode_encoded);
}

bool Input::is_mouse_button_pressed(MouseButton p_button) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_mouse_button_pressed")._native_ptr(), 1821097125);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_button_encoded;
	PtrToArg<int64_t>::encode(p_button, &p_button_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_button_encoded);
}

bool Input::is_joy_button_pressed(int32_t p_device, JoyButton p_button) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_joy_button_pressed")._native_ptr(), 787208542);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	int64_t p_button_encoded;
	PtrToArg<int64_t>::encode(p_button, &p_button_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_device_encoded, &p_button_encoded);
}

bool Input::is_action_pressed(const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_action_pressed")._native_ptr(), 1558498928);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, &p_exact_match_encoded);
}

bool Input::is_action_just_pressed(const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_action_just_pressed")._native_ptr(), 1558498928);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, &p_exact_match_encoded);
}

bool Input::is_action_just_released(const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_action_just_released")._native_ptr(), 1558498928);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, &p_exact_match_encoded);
}

bool Input::is_action_just_pressed_by_event(const StringName &p_action, const Ref<InputEvent> &p_event, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_action_just_pressed_by_event")._native_ptr(), 551972873);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, (p_event != nullptr ? &p_event->_owner : nullptr), &p_exact_match_encoded);
}

bool Input::is_action_just_released_by_event(const StringName &p_action, const Ref<InputEvent> &p_event, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_action_just_released_by_event")._native_ptr(), 551972873);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, (p_event != nullptr ? &p_event->_owner : nullptr), &p_exact_match_encoded);
}

float Input::get_action_strength(const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_action_strength")._native_ptr(), 801543509);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_action, &p_exact_match_encoded);
}

float Input::get_action_raw_strength(const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_action_raw_strength")._native_ptr(), 801543509);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_action, &p_exact_match_encoded);
}

float Input::get_axis(const StringName &p_negative_action, const StringName &p_positive_action) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_axis")._native_ptr(), 1958752504);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_negative_action, &p_positive_action);
}

Vector2 Input::get_vector(const StringName &p_negative_x, const StringName &p_positive_x, const StringName &p_negative_y, const StringName &p_positive_y, float p_deadzone) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_vector")._native_ptr(), 2479607902);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	double p_deadzone_encoded;
	PtrToArg<double>::encode(p_deadzone, &p_deadzone_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_negative_x, &p_positive_x, &p_negative_y, &p_positive_y, &p_deadzone_encoded);
}

void Input::add_joy_mapping(const String &p_mapping, bool p_update_existing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("add_joy_mapping")._native_ptr(), 1168363258);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_update_existing_encoded;
	PtrToArg<bool>::encode(p_update_existing, &p_update_existing_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mapping, &p_update_existing_encoded);
}

void Input::remove_joy_mapping(const String &p_guid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("remove_joy_mapping")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_guid);
}

bool Input::is_joy_known(int32_t p_device) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_joy_known")._native_ptr(), 3067735520);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_device_encoded);
}

float Input::get_joy_axis(int32_t p_device, JoyAxis p_axis) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_joy_axis")._native_ptr(), 4063175957);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_device_encoded, &p_axis_encoded);
}

String Input::get_joy_name(int32_t p_device) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_joy_name")._native_ptr(), 990163283);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_device_encoded);
}

String Input::get_joy_guid(int32_t p_device) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_joy_guid")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_device_encoded);
}

Dictionary Input::get_joy_info(int32_t p_device) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_joy_info")._native_ptr(), 3485342025);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_device_encoded);
}

bool Input::should_ignore_device(int32_t p_vendor_id, int32_t p_product_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("should_ignore_device")._native_ptr(), 2522259332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_vendor_id_encoded;
	PtrToArg<int64_t>::encode(p_vendor_id, &p_vendor_id_encoded);
	int64_t p_product_id_encoded;
	PtrToArg<int64_t>::encode(p_product_id, &p_product_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_vendor_id_encoded, &p_product_id_encoded);
}

TypedArray<int> Input::get_connected_joypads() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_connected_joypads")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<int>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<int>>(_gde_method_bind, _owner);
}

Vector2 Input::get_joy_vibration_strength(int32_t p_device) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_joy_vibration_strength")._native_ptr(), 3114997196);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_device_encoded);
}

float Input::get_joy_vibration_duration(int32_t p_device) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_joy_vibration_duration")._native_ptr(), 4025615559);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_device_encoded);
}

void Input::start_joy_vibration(int32_t p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("start_joy_vibration")._native_ptr(), 2576575033);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	double p_weak_magnitude_encoded;
	PtrToArg<double>::encode(p_weak_magnitude, &p_weak_magnitude_encoded);
	double p_strong_magnitude_encoded;
	PtrToArg<double>::encode(p_strong_magnitude, &p_strong_magnitude_encoded);
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_device_encoded, &p_weak_magnitude_encoded, &p_strong_magnitude_encoded, &p_duration_encoded);
}

void Input::stop_joy_vibration(int32_t p_device) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("stop_joy_vibration")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_device_encoded);
}

void Input::vibrate_handheld(int32_t p_duration_ms, float p_amplitude) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("vibrate_handheld")._native_ptr(), 544894297);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_duration_ms_encoded;
	PtrToArg<int64_t>::encode(p_duration_ms, &p_duration_ms_encoded);
	double p_amplitude_encoded;
	PtrToArg<double>::encode(p_amplitude, &p_amplitude_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_duration_ms_encoded, &p_amplitude_encoded);
}

Vector3 Input::get_gravity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_gravity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 Input::get_accelerometer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_accelerometer")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 Input::get_magnetometer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_magnetometer")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 Input::get_gyroscope() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_gyroscope")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void Input::set_gravity(const Vector3 &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_gravity")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value);
}

void Input::set_accelerometer(const Vector3 &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_accelerometer")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value);
}

void Input::set_magnetometer(const Vector3 &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_magnetometer")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value);
}

void Input::set_gyroscope(const Vector3 &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_gyroscope")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value);
}

void Input::set_joy_light(int32_t p_device, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_joy_light")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_device_encoded, &p_color);
}

bool Input::has_joy_light(int32_t p_device) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("has_joy_light")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_device_encoded);
}

Vector2 Input::get_last_mouse_velocity() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_last_mouse_velocity")._native_ptr(), 1497962370);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 Input::get_last_mouse_screen_velocity() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_last_mouse_screen_velocity")._native_ptr(), 1497962370);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

BitField<MouseButtonMask> Input::get_mouse_button_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_mouse_button_mask")._native_ptr(), 2512161324);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<MouseButtonMask>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Input::set_mouse_mode(Input::MouseMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_mouse_mode")._native_ptr(), 2228490894);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Input::MouseMode Input::get_mouse_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_mouse_mode")._native_ptr(), 965286182);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Input::MouseMode(0)));
	return (Input::MouseMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Input::warp_mouse(const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("warp_mouse")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

void Input::action_press(const StringName &p_action, float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("action_press")._native_ptr(), 1713091165);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action, &p_strength_encoded);
}

void Input::action_release(const StringName &p_action) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("action_release")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_action);
}

void Input::set_default_cursor_shape(Input::CursorShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_default_cursor_shape")._native_ptr(), 2124816902);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

Input::CursorShape Input::get_current_cursor_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("get_current_cursor_shape")._native_ptr(), 3455658929);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Input::CursorShape(0)));
	return (Input::CursorShape)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Input::set_custom_mouse_cursor(const Ref<Resource> &p_image, Input::CursorShape p_shape, const Vector2 &p_hotspot) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_custom_mouse_cursor")._native_ptr(), 703945977);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr), &p_shape_encoded, &p_hotspot);
}

void Input::parse_input_event(const Ref<InputEvent> &p_event) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("parse_input_event")._native_ptr(), 3754044979);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_event != nullptr ? &p_event->_owner : nullptr));
}

void Input::set_use_accumulated_input(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_use_accumulated_input")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Input::is_using_accumulated_input() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_using_accumulated_input")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Input::flush_buffered_events() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("flush_buffered_events")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Input::set_emulate_mouse_from_touch(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_emulate_mouse_from_touch")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Input::is_emulating_mouse_from_touch() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_emulating_mouse_from_touch")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Input::set_emulate_touch_from_mouse(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("set_emulate_touch_from_mouse")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Input::is_emulating_touch_from_mouse() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Input::get_class_static()._native_ptr(), StringName("is_emulating_touch_from_mouse")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
