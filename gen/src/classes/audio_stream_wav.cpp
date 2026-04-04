/**************************************************************************/
/*  audio_stream_wav.cpp                                                  */
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

#include <godot_cpp/classes/audio_stream_wav.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string.hpp>

namespace godot {

Ref<AudioStreamWAV> AudioStreamWAV::load_from_buffer(const PackedByteArray &p_stream_data, const Dictionary &p_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("load_from_buffer")._native_ptr(), 4266838938);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamWAV>()));
	return Ref<AudioStreamWAV>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamWAV>(_gde_method_bind, nullptr, &p_stream_data, &p_options));
}

Ref<AudioStreamWAV> AudioStreamWAV::load_from_file(const String &p_path, const Dictionary &p_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("load_from_file")._native_ptr(), 4015802384);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamWAV>()));
	return Ref<AudioStreamWAV>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamWAV>(_gde_method_bind, nullptr, &p_path, &p_options));
}

void AudioStreamWAV::set_data(const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("set_data")._native_ptr(), 2971499966);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data);
}

PackedByteArray AudioStreamWAV::get_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("get_data")._native_ptr(), 2362200018);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner);
}

void AudioStreamWAV::set_format(AudioStreamWAV::Format p_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("set_format")._native_ptr(), 60648488);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_format_encoded);
}

AudioStreamWAV::Format AudioStreamWAV::get_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("get_format")._native_ptr(), 3151724922);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AudioStreamWAV::Format(0)));
	return (AudioStreamWAV::Format)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamWAV::set_loop_mode(AudioStreamWAV::LoopMode p_loop_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("set_loop_mode")._native_ptr(), 2444882972);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_loop_mode_encoded;
	PtrToArg<int64_t>::encode(p_loop_mode, &p_loop_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_loop_mode_encoded);
}

AudioStreamWAV::LoopMode AudioStreamWAV::get_loop_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("get_loop_mode")._native_ptr(), 393560655);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AudioStreamWAV::LoopMode(0)));
	return (AudioStreamWAV::LoopMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamWAV::set_loop_begin(int32_t p_loop_begin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("set_loop_begin")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_loop_begin_encoded;
	PtrToArg<int64_t>::encode(p_loop_begin, &p_loop_begin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_loop_begin_encoded);
}

int32_t AudioStreamWAV::get_loop_begin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("get_loop_begin")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamWAV::set_loop_end(int32_t p_loop_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("set_loop_end")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_loop_end_encoded;
	PtrToArg<int64_t>::encode(p_loop_end, &p_loop_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_loop_end_encoded);
}

int32_t AudioStreamWAV::get_loop_end() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("get_loop_end")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamWAV::set_mix_rate(int32_t p_mix_rate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("set_mix_rate")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mix_rate_encoded;
	PtrToArg<int64_t>::encode(p_mix_rate, &p_mix_rate_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mix_rate_encoded);
}

int32_t AudioStreamWAV::get_mix_rate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("get_mix_rate")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamWAV::set_stereo(bool p_stereo) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("set_stereo")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_stereo_encoded;
	PtrToArg<bool>::encode(p_stereo, &p_stereo_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stereo_encoded);
}

bool AudioStreamWAV::is_stereo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("is_stereo")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AudioStreamWAV::set_tags(const Dictionary &p_tags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("set_tags")._native_ptr(), 4155329257);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tags);
}

Dictionary AudioStreamWAV::get_tags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("get_tags")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

Error AudioStreamWAV::save_to_wav(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamWAV::get_class_static()._native_ptr(), StringName("save_to_wav")._native_ptr(), 166001499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path);
}

} // namespace godot
