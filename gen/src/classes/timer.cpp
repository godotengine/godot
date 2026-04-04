/**************************************************************************/
/*  timer.cpp                                                             */
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

#include <godot_cpp/classes/timer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void Timer::set_wait_time(double p_time_sec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("set_wait_time")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_time_sec_encoded);
}

double Timer::get_wait_time() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("get_wait_time")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Timer::set_one_shot(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("set_one_shot")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Timer::is_one_shot() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("is_one_shot")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Timer::set_autostart(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("set_autostart")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Timer::has_autostart() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("has_autostart")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Timer::start(double p_time_sec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("start")._native_ptr(), 1392008558);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_sec_encoded;
	PtrToArg<double>::encode(p_time_sec, &p_time_sec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_time_sec_encoded);
}

void Timer::stop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("stop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Timer::set_paused(bool p_paused) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("set_paused")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_paused_encoded;
	PtrToArg<bool>::encode(p_paused, &p_paused_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_paused_encoded);
}

bool Timer::is_paused() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("is_paused")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Timer::set_ignore_time_scale(bool p_ignore) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("set_ignore_time_scale")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_ignore_encoded;
	PtrToArg<bool>::encode(p_ignore, &p_ignore_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ignore_encoded);
}

bool Timer::is_ignoring_time_scale() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("is_ignoring_time_scale")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Timer::is_stopped() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("is_stopped")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

double Timer::get_time_left() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("get_time_left")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Timer::set_timer_process_callback(Timer::TimerProcessCallback p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("set_timer_process_callback")._native_ptr(), 3469495063);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_callback_encoded;
	PtrToArg<int64_t>::encode(p_callback, &p_callback_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callback_encoded);
}

Timer::TimerProcessCallback Timer::get_timer_process_callback() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Timer::get_class_static()._native_ptr(), StringName("get_timer_process_callback")._native_ptr(), 2672570227);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Timer::TimerProcessCallback(0)));
	return (Timer::TimerProcessCallback)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
