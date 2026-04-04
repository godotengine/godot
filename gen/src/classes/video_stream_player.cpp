/**************************************************************************/
/*  video_stream_player.cpp                                               */
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

#include <godot_cpp/classes/video_stream_player.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/video_stream.hpp>

namespace godot {

void VideoStreamPlayer::set_stream(const Ref<VideoStream> &p_stream) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_stream")._native_ptr(), 2317102564);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_stream != nullptr ? &p_stream->_owner : nullptr));
}

Ref<VideoStream> VideoStreamPlayer::get_stream() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_stream")._native_ptr(), 438621487);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<VideoStream>()));
	return Ref<VideoStream>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<VideoStream>(_gde_method_bind, _owner));
}

void VideoStreamPlayer::play() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("play")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void VideoStreamPlayer::stop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("stop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool VideoStreamPlayer::is_playing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("is_playing")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_paused(bool p_paused) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_paused")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_paused_encoded;
	PtrToArg<bool>::encode(p_paused, &p_paused_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_paused_encoded);
}

bool VideoStreamPlayer::is_paused() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("is_paused")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_loop(bool p_loop) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_loop")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_loop_encoded;
	PtrToArg<bool>::encode(p_loop, &p_loop_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_loop_encoded);
}

bool VideoStreamPlayer::has_loop() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("has_loop")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_volume(float p_volume) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_volume")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_volume_encoded;
	PtrToArg<double>::encode(p_volume, &p_volume_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_volume_encoded);
}

float VideoStreamPlayer::get_volume() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_volume")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_volume_db(float p_db) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_volume_db")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_db_encoded;
	PtrToArg<double>::encode(p_db, &p_db_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_db_encoded);
}

float VideoStreamPlayer::get_volume_db() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_volume_db")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_speed_scale(float p_speed_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_speed_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_speed_scale_encoded;
	PtrToArg<double>::encode(p_speed_scale, &p_speed_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_speed_scale_encoded);
}

float VideoStreamPlayer::get_speed_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_speed_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_audio_track(int32_t p_track) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_audio_track")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_track_encoded;
	PtrToArg<int64_t>::encode(p_track, &p_track_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_track_encoded);
}

int32_t VideoStreamPlayer::get_audio_track() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_audio_track")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String VideoStreamPlayer::get_stream_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_stream_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

double VideoStreamPlayer::get_stream_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_stream_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_stream_position(double p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_stream_position")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_position_encoded;
	PtrToArg<double>::encode(p_position, &p_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position_encoded);
}

double VideoStreamPlayer::get_stream_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_stream_position")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_autoplay(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_autoplay")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool VideoStreamPlayer::has_autoplay() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("has_autoplay")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_expand(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_expand")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool VideoStreamPlayer::has_expand() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("has_expand")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_buffering_msec(int32_t p_msec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_buffering_msec")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msec_encoded;
	PtrToArg<int64_t>::encode(p_msec, &p_msec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msec_encoded);
}

int32_t VideoStreamPlayer::get_buffering_msec() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_buffering_msec")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VideoStreamPlayer::set_bus(const StringName &p_bus) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("set_bus")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bus);
}

StringName VideoStreamPlayer::get_bus() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_bus")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

Ref<Texture2D> VideoStreamPlayer::get_video_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VideoStreamPlayer::get_class_static()._native_ptr(), StringName("get_video_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

} // namespace godot
