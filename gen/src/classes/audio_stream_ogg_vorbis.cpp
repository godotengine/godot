/**************************************************************************/
/*  audio_stream_ogg_vorbis.cpp                                           */
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

#include <godot_cpp/classes/audio_stream_ogg_vorbis.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/ogg_packet_sequence.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

Ref<AudioStreamOggVorbis> AudioStreamOggVorbis::load_from_buffer(const PackedByteArray &p_stream_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("load_from_buffer")._native_ptr(), 354904730);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamOggVorbis>()));
	return Ref<AudioStreamOggVorbis>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamOggVorbis>(_gde_method_bind, nullptr, &p_stream_data));
}

Ref<AudioStreamOggVorbis> AudioStreamOggVorbis::load_from_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("load_from_file")._native_ptr(), 797568536);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamOggVorbis>()));
	return Ref<AudioStreamOggVorbis>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamOggVorbis>(_gde_method_bind, nullptr, &p_path));
}

void AudioStreamOggVorbis::set_packet_sequence(const Ref<OggPacketSequence> &p_packet_sequence) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("set_packet_sequence")._native_ptr(), 438882457);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_packet_sequence != nullptr ? &p_packet_sequence->_owner : nullptr));
}

Ref<OggPacketSequence> AudioStreamOggVorbis::get_packet_sequence() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("get_packet_sequence")._native_ptr(), 2801636033);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OggPacketSequence>()));
	return Ref<OggPacketSequence>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OggPacketSequence>(_gde_method_bind, _owner));
}

void AudioStreamOggVorbis::set_loop(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("set_loop")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AudioStreamOggVorbis::has_loop() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("has_loop")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AudioStreamOggVorbis::set_loop_offset(double p_seconds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("set_loop_offset")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_seconds_encoded;
	PtrToArg<double>::encode(p_seconds, &p_seconds_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seconds_encoded);
}

double AudioStreamOggVorbis::get_loop_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("get_loop_offset")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamOggVorbis::set_bpm(double p_bpm) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("set_bpm")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bpm_encoded;
	PtrToArg<double>::encode(p_bpm, &p_bpm_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bpm_encoded);
}

double AudioStreamOggVorbis::get_bpm() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("get_bpm")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamOggVorbis::set_beat_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("set_beat_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t AudioStreamOggVorbis::get_beat_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("get_beat_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamOggVorbis::set_bar_beats(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("set_bar_beats")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t AudioStreamOggVorbis::get_bar_beats() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("get_bar_beats")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamOggVorbis::set_tags(const Dictionary &p_tags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("set_tags")._native_ptr(), 4155329257);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tags);
}

Dictionary AudioStreamOggVorbis::get_tags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamOggVorbis::get_class_static()._native_ptr(), StringName("get_tags")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

} // namespace godot
