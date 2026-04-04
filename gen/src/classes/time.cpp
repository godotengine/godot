/**************************************************************************/
/*  time.cpp                                                              */
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

#include <godot_cpp/classes/time.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Time *Time::singleton = nullptr;

Time *Time::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(Time::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<Time *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &Time::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(Time::get_class_static(), singleton);
		}
	}
	return singleton;
}

Time::~Time() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(Time::get_class_static());
		singleton = nullptr;
	}
}

Dictionary Time::get_datetime_dict_from_unix_time(int64_t p_unix_time_val) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_datetime_dict_from_unix_time")._native_ptr(), 3485342025);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_unix_time_val_encoded;
	PtrToArg<int64_t>::encode(p_unix_time_val, &p_unix_time_val_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_unix_time_val_encoded);
}

Dictionary Time::get_date_dict_from_unix_time(int64_t p_unix_time_val) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_date_dict_from_unix_time")._native_ptr(), 3485342025);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_unix_time_val_encoded;
	PtrToArg<int64_t>::encode(p_unix_time_val, &p_unix_time_val_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_unix_time_val_encoded);
}

Dictionary Time::get_time_dict_from_unix_time(int64_t p_unix_time_val) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_time_dict_from_unix_time")._native_ptr(), 3485342025);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_unix_time_val_encoded;
	PtrToArg<int64_t>::encode(p_unix_time_val, &p_unix_time_val_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_unix_time_val_encoded);
}

String Time::get_datetime_string_from_unix_time(int64_t p_unix_time_val, bool p_use_space) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_datetime_string_from_unix_time")._native_ptr(), 2311239925);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_unix_time_val_encoded;
	PtrToArg<int64_t>::encode(p_unix_time_val, &p_unix_time_val_encoded);
	int8_t p_use_space_encoded;
	PtrToArg<bool>::encode(p_use_space, &p_use_space_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_unix_time_val_encoded, &p_use_space_encoded);
}

String Time::get_date_string_from_unix_time(int64_t p_unix_time_val) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_date_string_from_unix_time")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_unix_time_val_encoded;
	PtrToArg<int64_t>::encode(p_unix_time_val, &p_unix_time_val_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_unix_time_val_encoded);
}

String Time::get_time_string_from_unix_time(int64_t p_unix_time_val) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_time_string_from_unix_time")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_unix_time_val_encoded;
	PtrToArg<int64_t>::encode(p_unix_time_val, &p_unix_time_val_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_unix_time_val_encoded);
}

Dictionary Time::get_datetime_dict_from_datetime_string(const String &p_datetime, bool p_weekday) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_datetime_dict_from_datetime_string")._native_ptr(), 3253569256);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_weekday_encoded;
	PtrToArg<bool>::encode(p_weekday, &p_weekday_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_datetime, &p_weekday_encoded);
}

String Time::get_datetime_string_from_datetime_dict(const Dictionary &p_datetime, bool p_use_space) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_datetime_string_from_datetime_dict")._native_ptr(), 1898123706);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_use_space_encoded;
	PtrToArg<bool>::encode(p_use_space, &p_use_space_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_datetime, &p_use_space_encoded);
}

int64_t Time::get_unix_time_from_datetime_dict(const Dictionary &p_datetime) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_unix_time_from_datetime_dict")._native_ptr(), 3021115443);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_datetime);
}

int64_t Time::get_unix_time_from_datetime_string(const String &p_datetime) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_unix_time_from_datetime_string")._native_ptr(), 1321353865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_datetime);
}

String Time::get_offset_string_from_offset_minutes(int64_t p_offset_minutes) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_offset_string_from_offset_minutes")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_offset_minutes_encoded;
	PtrToArg<int64_t>::encode(p_offset_minutes, &p_offset_minutes_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_offset_minutes_encoded);
}

Dictionary Time::get_datetime_dict_from_system(bool p_utc) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_datetime_dict_from_system")._native_ptr(), 205769976);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_utc_encoded;
	PtrToArg<bool>::encode(p_utc, &p_utc_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_utc_encoded);
}

Dictionary Time::get_date_dict_from_system(bool p_utc) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_date_dict_from_system")._native_ptr(), 205769976);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_utc_encoded;
	PtrToArg<bool>::encode(p_utc, &p_utc_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_utc_encoded);
}

Dictionary Time::get_time_dict_from_system(bool p_utc) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_time_dict_from_system")._native_ptr(), 205769976);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_utc_encoded;
	PtrToArg<bool>::encode(p_utc, &p_utc_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_utc_encoded);
}

String Time::get_datetime_string_from_system(bool p_utc, bool p_use_space) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_datetime_string_from_system")._native_ptr(), 1136425492);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_utc_encoded;
	PtrToArg<bool>::encode(p_utc, &p_utc_encoded);
	int8_t p_use_space_encoded;
	PtrToArg<bool>::encode(p_use_space, &p_use_space_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_utc_encoded, &p_use_space_encoded);
}

String Time::get_date_string_from_system(bool p_utc) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_date_string_from_system")._native_ptr(), 1162154673);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_utc_encoded;
	PtrToArg<bool>::encode(p_utc, &p_utc_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_utc_encoded);
}

String Time::get_time_string_from_system(bool p_utc) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_time_string_from_system")._native_ptr(), 1162154673);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_utc_encoded;
	PtrToArg<bool>::encode(p_utc, &p_utc_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_utc_encoded);
}

Dictionary Time::get_time_zone_from_system() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_time_zone_from_system")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

double Time::get_unix_time_from_system() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_unix_time_from_system")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

uint64_t Time::get_ticks_msec() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_ticks_msec")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t Time::get_ticks_usec() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Time::get_class_static()._native_ptr(), StringName("get_ticks_usec")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

} // namespace godot
