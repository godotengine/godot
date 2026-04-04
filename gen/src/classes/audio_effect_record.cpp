/**************************************************************************/
/*  audio_effect_record.cpp                                               */
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

#include <godot_cpp/classes/audio_effect_record.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AudioEffectRecord::set_recording_active(bool p_record) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectRecord::get_class_static()._native_ptr(), StringName("set_recording_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_record_encoded;
	PtrToArg<bool>::encode(p_record, &p_record_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_record_encoded);
}

bool AudioEffectRecord::is_recording_active() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectRecord::get_class_static()._native_ptr(), StringName("is_recording_active")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AudioEffectRecord::set_format(AudioStreamWAV::Format p_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectRecord::get_class_static()._native_ptr(), StringName("set_format")._native_ptr(), 60648488);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_format_encoded);
}

AudioStreamWAV::Format AudioEffectRecord::get_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectRecord::get_class_static()._native_ptr(), StringName("get_format")._native_ptr(), 3151724922);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AudioStreamWAV::Format(0)));
	return (AudioStreamWAV::Format)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<AudioStreamWAV> AudioEffectRecord::get_recording() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioEffectRecord::get_class_static()._native_ptr(), StringName("get_recording")._native_ptr(), 2964110865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamWAV>()));
	return Ref<AudioStreamWAV>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamWAV>(_gde_method_bind, _owner));
}

} // namespace godot
