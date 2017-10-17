/*************************************************************************/
/*  audio_player.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "audio_player.h"

#include "engine.h"

void AudioStreamPlayer::_mix_audio() {

	if (!stream_playback.is_valid()) {
		return;
	}

	if (!active) {
		return;
	}

	if (setseek >= 0.0) {
		stream_playback->start(setseek);
		setseek = -1.0; //reset seek
	}

	int bus_index = AudioServer::get_singleton()->thread_find_bus_index(bus);

	//get data
	AudioFrame *buffer = mix_buffer.ptr();
	int buffer_size = mix_buffer.size();

	//mix
	stream_playback->mix(buffer, 1.0, buffer_size);

	//multiply volume interpolating to avoid clicks if this changes
	float vol = Math::db2linear(mix_volume_db);
	float vol_inc = (Math::db2linear(volume_db) - vol) / float(buffer_size);

	for (int i = 0; i < buffer_size; i++) {
		buffer[i] *= vol;
		vol += vol_inc;
	}
	//set volume for next mix
	mix_volume_db = volume_db;

	AudioFrame *targets[4] = { NULL, NULL, NULL, NULL };

	if (AudioServer::get_singleton()->get_speaker_mode() == AudioServer::SPEAKER_MODE_STEREO) {
		targets[0] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 0);
	} else {
		switch (mix_target) {
			case MIX_TARGET_STEREO: {
				targets[0] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 0);
			} break;
			case MIX_TARGET_SURROUND: {
				for (int i = 0; i < AudioServer::get_singleton()->get_channel_count(); i++) {
					targets[i] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, i);
				}
			} break;
			case MIX_TARGET_CENTER: {
				targets[0] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 1);
			} break;
		}
	}

	for (int c = 0; c < 4; c++) {
		if (!targets[c])
			break;
		for (int i = 0; i < buffer_size; i++) {
			targets[c][i] += buffer[i];
		}
	}
}

void AudioStreamPlayer::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		AudioServer::get_singleton()->add_callback(_mix_audios, this);
		if (autoplay && !Engine::get_singleton()->is_editor_hint()) {
			play();
		}
	}

	if (p_what == NOTIFICATION_INTERNAL_PROCESS) {

		if (!active || (setseek < 0 && !stream_playback->is_playing())) {
			active = false;
			emit_signal("finished");
			set_process_internal(false);
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		AudioServer::get_singleton()->remove_callback(_mix_audios, this);
	}
}

void AudioStreamPlayer::set_stream(Ref<AudioStream> p_stream) {

	AudioServer::get_singleton()->lock();

	mix_buffer.resize(AudioServer::get_singleton()->thread_get_mix_buffer_size());

	if (stream_playback.is_valid()) {
		stream_playback.unref();
		stream.unref();
		active = false;
		setseek = -1;
	}

	if (p_stream.is_valid()) {
		stream = p_stream;
		stream_playback = p_stream->instance_playback();
	}

	AudioServer::get_singleton()->unlock();

	if (p_stream.is_valid() && stream_playback.is_null()) {
		stream.unref();
		ERR_FAIL_COND(stream_playback.is_null());
	}
}

Ref<AudioStream> AudioStreamPlayer::get_stream() const {

	return stream;
}

void AudioStreamPlayer::set_volume_db(float p_volume) {

	volume_db = p_volume;
}
float AudioStreamPlayer::get_volume_db() const {

	return volume_db;
}

void AudioStreamPlayer::play(float p_from_pos) {

	if (stream_playback.is_valid()) {
		mix_volume_db = volume_db; //reset volume ramp
		setseek = p_from_pos;
		active = true;
		set_process_internal(true);
	}
}

void AudioStreamPlayer::seek(float p_seconds) {

	if (stream_playback.is_valid()) {
		setseek = p_seconds;
	}
}

void AudioStreamPlayer::stop() {

	if (stream_playback.is_valid()) {
		active = false;
		set_process_internal(false);
	}
}

bool AudioStreamPlayer::is_playing() const {

	if (stream_playback.is_valid()) {
		return active; //&& stream_playback->is_playing();
	}

	return false;
}

float AudioStreamPlayer::get_playback_position() {

	if (stream_playback.is_valid()) {
		return stream_playback->get_playback_position();
	}

	return 0;
}

void AudioStreamPlayer::set_bus(const StringName &p_bus) {

	//if audio is active, must lock this
	AudioServer::get_singleton()->lock();
	bus = p_bus;
	AudioServer::get_singleton()->unlock();
}
StringName AudioStreamPlayer::get_bus() const {

	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == bus) {
			return bus;
		}
	}
	return "Master";
}

void AudioStreamPlayer::set_autoplay(bool p_enable) {

	autoplay = p_enable;
}
bool AudioStreamPlayer::is_autoplay_enabled() {

	return autoplay;
}

void AudioStreamPlayer::set_mix_target(MixTarget p_target) {

	mix_target = p_target;
}

AudioStreamPlayer::MixTarget AudioStreamPlayer::get_mix_target() const {

	return mix_target;
}

void AudioStreamPlayer::_set_playing(bool p_enable) {

	if (p_enable)
		play();
	else
		stop();
}
bool AudioStreamPlayer::_is_active() const {

	return active;
}

void AudioStreamPlayer::_validate_property(PropertyInfo &property) const {

	if (property.name == "bus") {

		String options;
		for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
			if (i > 0)
				options += ",";
			String name = AudioServer::get_singleton()->get_bus_name(i);
			options += name;
		}

		property.hint_string = options;
	}
}

void AudioStreamPlayer::_bus_layout_changed() {

	_change_notify();
}

void AudioStreamPlayer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_stream", "stream"), &AudioStreamPlayer::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream"), &AudioStreamPlayer::get_stream);

	ClassDB::bind_method(D_METHOD("set_volume_db", "volume_db"), &AudioStreamPlayer::set_volume_db);
	ClassDB::bind_method(D_METHOD("get_volume_db"), &AudioStreamPlayer::get_volume_db);

	ClassDB::bind_method(D_METHOD("play", "from_position"), &AudioStreamPlayer::play, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("seek", "to_position"), &AudioStreamPlayer::seek);
	ClassDB::bind_method(D_METHOD("stop"), &AudioStreamPlayer::stop);

	ClassDB::bind_method(D_METHOD("is_playing"), &AudioStreamPlayer::is_playing);
	ClassDB::bind_method(D_METHOD("get_playback_position"), &AudioStreamPlayer::get_playback_position);

	ClassDB::bind_method(D_METHOD("set_bus", "bus"), &AudioStreamPlayer::set_bus);
	ClassDB::bind_method(D_METHOD("get_bus"), &AudioStreamPlayer::get_bus);

	ClassDB::bind_method(D_METHOD("set_autoplay", "enable"), &AudioStreamPlayer::set_autoplay);
	ClassDB::bind_method(D_METHOD("is_autoplay_enabled"), &AudioStreamPlayer::is_autoplay_enabled);

	ClassDB::bind_method(D_METHOD("set_mix_target", "mix_target"), &AudioStreamPlayer::set_mix_target);
	ClassDB::bind_method(D_METHOD("get_mix_target"), &AudioStreamPlayer::get_mix_target);

	ClassDB::bind_method(D_METHOD("_set_playing", "enable"), &AudioStreamPlayer::_set_playing);
	ClassDB::bind_method(D_METHOD("_is_active"), &AudioStreamPlayer::_is_active);

	ClassDB::bind_method(D_METHOD("_bus_layout_changed"), &AudioStreamPlayer::_bus_layout_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_stream", "get_stream");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_db", PROPERTY_HINT_RANGE, "-80,24"), "set_volume_db", "get_volume_db");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_playing", "is_playing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autoplay"), "set_autoplay", "is_autoplay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mix_target", PROPERTY_HINT_ENUM, "Stereo,Surround,Center"), "set_mix_target", "get_mix_target");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bus", PROPERTY_HINT_ENUM, ""), "set_bus", "get_bus");

	ADD_SIGNAL(MethodInfo("finished"));

	BIND_ENUM_CONSTANT(MIX_TARGET_STEREO);
	BIND_ENUM_CONSTANT(MIX_TARGET_SURROUND);
	BIND_ENUM_CONSTANT(MIX_TARGET_CENTER);
}

AudioStreamPlayer::AudioStreamPlayer() {

	mix_volume_db = 0;
	volume_db = 0;
	autoplay = false;
	setseek = -1;
	active = false;
	mix_target = MIX_TARGET_STEREO;

	AudioServer::get_singleton()->connect("bus_layout_changed", this, "_bus_layout_changed");
}

AudioStreamPlayer::~AudioStreamPlayer() {
}
