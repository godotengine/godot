/**************************************************************************/
/*  audio_stream_player2d.cpp                                             */
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

#include <godot_cpp/classes/audio_stream_player2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/audio_stream.hpp>
#include <godot_cpp/classes/audio_stream_playback.hpp>

namespace godot {

void AudioStreamPlayer2D::set_stream(const Ref<AudioStream> &p_stream) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_stream")._native_ptr(), 2210767741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_stream != nullptr ? &p_stream->_owner : nullptr));
}

Ref<AudioStream> AudioStreamPlayer2D::get_stream() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_stream")._native_ptr(), 160907539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStream>()));
	return Ref<AudioStream>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStream>(_gde_method_bind, _owner));
}

void AudioStreamPlayer2D::set_volume_db(float p_volume_db) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_volume_db")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_volume_db_encoded;
	PtrToArg<double>::encode(p_volume_db, &p_volume_db_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_volume_db_encoded);
}

float AudioStreamPlayer2D::get_volume_db() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_volume_db")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_volume_linear(float p_volume_linear) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_volume_linear")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_volume_linear_encoded;
	PtrToArg<double>::encode(p_volume_linear, &p_volume_linear_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_volume_linear_encoded);
}

float AudioStreamPlayer2D::get_volume_linear() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_volume_linear")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_pitch_scale(float p_pitch_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_pitch_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pitch_scale_encoded;
	PtrToArg<double>::encode(p_pitch_scale, &p_pitch_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pitch_scale_encoded);
}

float AudioStreamPlayer2D::get_pitch_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_pitch_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::play(float p_from_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("play")._native_ptr(), 1958160172);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_from_position_encoded;
	PtrToArg<double>::encode(p_from_position, &p_from_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_position_encoded);
}

void AudioStreamPlayer2D::seek(float p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("seek")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_to_position_encoded;
	PtrToArg<double>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_to_position_encoded);
}

void AudioStreamPlayer2D::stop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("stop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool AudioStreamPlayer2D::is_playing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("is_playing")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

float AudioStreamPlayer2D::get_playback_position() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_playback_position")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_bus(const StringName &p_bus) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_bus")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bus);
}

StringName AudioStreamPlayer2D::get_bus() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_bus")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_autoplay(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_autoplay")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AudioStreamPlayer2D::is_autoplay_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("is_autoplay_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_playing(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_playing")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void AudioStreamPlayer2D::set_max_distance(float p_pixels) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_max_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pixels_encoded;
	PtrToArg<double>::encode(p_pixels, &p_pixels_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pixels_encoded);
}

float AudioStreamPlayer2D::get_max_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_max_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_attenuation(float p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_attenuation")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_curve_encoded;
	PtrToArg<double>::encode(p_curve, &p_curve_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_curve_encoded);
}

float AudioStreamPlayer2D::get_attenuation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_attenuation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_area_mask(uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_area_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask_encoded);
}

uint32_t AudioStreamPlayer2D::get_area_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_area_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_stream_paused(bool p_pause) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_stream_paused")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pause_encoded;
	PtrToArg<bool>::encode(p_pause, &p_pause_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pause_encoded);
}

bool AudioStreamPlayer2D::get_stream_paused() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_stream_paused")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_max_polyphony(int32_t p_max_polyphony) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_max_polyphony")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_polyphony_encoded;
	PtrToArg<int64_t>::encode(p_max_polyphony, &p_max_polyphony_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_polyphony_encoded);
}

int32_t AudioStreamPlayer2D::get_max_polyphony() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_max_polyphony")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AudioStreamPlayer2D::set_panning_strength(float p_panning_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_panning_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_panning_strength_encoded;
	PtrToArg<double>::encode(p_panning_strength, &p_panning_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_panning_strength_encoded);
}

float AudioStreamPlayer2D::get_panning_strength() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_panning_strength")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool AudioStreamPlayer2D::has_stream_playback() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("has_stream_playback")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<AudioStreamPlayback> AudioStreamPlayer2D::get_stream_playback() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_stream_playback")._native_ptr(), 210135309);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AudioStreamPlayback>()));
	return Ref<AudioStreamPlayback>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AudioStreamPlayback>(_gde_method_bind, _owner));
}

void AudioStreamPlayer2D::set_playback_type(AudioServer::PlaybackType p_playback_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("set_playback_type")._native_ptr(), 725473817);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_playback_type_encoded;
	PtrToArg<int64_t>::encode(p_playback_type, &p_playback_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_playback_type_encoded);
}

AudioServer::PlaybackType AudioStreamPlayer2D::get_playback_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AudioStreamPlayer2D::get_class_static()._native_ptr(), StringName("get_playback_type")._native_ptr(), 4011264623);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AudioServer::PlaybackType(0)));
	return (AudioServer::PlaybackType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
