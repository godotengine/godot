/*************************************************************************/
/*  audio_player.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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

void AudioPlayer::_mix_audio() {

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

	AudioFrame *targets[3] = { NULL, NULL, NULL };

	if (AudioServer::get_singleton()->get_speaker_mode() == AudioServer::SPEAKER_MODE_STEREO) {
		targets[0] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 0);
	} else {
		switch (mix_target) {
			case MIX_TARGET_STEREO: {
				targets[0] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 1);
			} break;
			case MIX_TARGET_SURROUND: {
				targets[0] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 1);
				targets[1] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 2);
				if (AudioServer::get_singleton()->get_speaker_mode() == AudioServer::SPEAKER_SURROUND_71) {
					targets[2] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 3);
				}
			} break;
			case MIX_TARGET_CENTER: {
				targets[0] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(bus_index, 0);
			} break;
		}
	}

	for (int c = 0; c < 3; c++) {
		if (!targets[c])
			break;
		for (int i = 0; i < buffer_size; i++) {
			targets[c][i] += buffer[i];
		}
	}
}

void AudioPlayer::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		AudioServer::get_singleton()->add_callback(_mix_audios, this);
		if (autoplay && !get_tree()->is_editor_hint()) {
			play();
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		AudioServer::get_singleton()->remove_callback(_mix_audios, this);
	}
}

void AudioPlayer::set_stream(Ref<AudioStream> p_stream) {

	AudioServer::get_singleton()->lock();

	mix_buffer.resize(AudioServer::get_singleton()->thread_get_mix_buffer_size());

	if (stream_playback.is_valid()) {
		stream_playback.unref();
		stream.unref();
		active = false;
		setseek = -1;
	}

	stream = p_stream;
	stream_playback = p_stream->instance_playback();

	if (stream_playback.is_null()) {
		stream.unref();
		ERR_FAIL_COND(stream_playback.is_null());
	}

	AudioServer::get_singleton()->unlock();
}

Ref<AudioStream> AudioPlayer::get_stream() const {

	return stream;
}

void AudioPlayer::set_volume_db(float p_volume) {

	volume_db = p_volume;
}
float AudioPlayer::get_volume_db() const {

	return volume_db;
}

void AudioPlayer::play(float p_from_pos) {

	if (stream_playback.is_valid()) {
		mix_volume_db = volume_db; //reset volume ramp
		setseek = p_from_pos;
		active = true;
	}
}

void AudioPlayer::seek(float p_seconds) {

	if (stream_playback.is_valid()) {
		setseek = p_seconds;
	}
}

void AudioPlayer::stop() {

	if (stream_playback.is_valid()) {
		active = false;
	}
}

bool AudioPlayer::is_playing() const {

	if (stream_playback.is_valid()) {
		return active && stream_playback->is_playing();
	}

	return false;
}

float AudioPlayer::get_pos() {

	if (stream_playback.is_valid()) {
		return stream_playback->get_pos();
	}

	return 0;
}

void AudioPlayer::set_bus(const StringName &p_bus) {

	//if audio is active, must lock this
	AudioServer::get_singleton()->lock();
	bus = p_bus;
	AudioServer::get_singleton()->unlock();
}
StringName AudioPlayer::get_bus() const {

	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == bus) {
			return bus;
		}
	}
	return "Master";
}

void AudioPlayer::set_autoplay(bool p_enable) {

	autoplay = p_enable;
}
bool AudioPlayer::is_autoplay_enabled() {

	return autoplay;
}

void AudioPlayer::set_mix_target(MixTarget p_target) {

	mix_target = p_target;
}

AudioPlayer::MixTarget AudioPlayer::get_mix_target() const {

	return mix_target;
}

void AudioPlayer::_set_playing(bool p_enable) {

	if (p_enable)
		play();
	else
		stop();
}
bool AudioPlayer::_is_active() const {

	return active;
}

void AudioPlayer::_validate_property(PropertyInfo &property) const {

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

void AudioPlayer::_bus_layout_changed() {

	_change_notify();
}

void AudioPlayer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_stream", "stream:AudioStream"), &AudioPlayer::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream"), &AudioPlayer::get_stream);

	ClassDB::bind_method(D_METHOD("set_volume_db", "volume_db"), &AudioPlayer::set_volume_db);
	ClassDB::bind_method(D_METHOD("get_volume_db"), &AudioPlayer::get_volume_db);

	ClassDB::bind_method(D_METHOD("play", "from_pos"), &AudioPlayer::play, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("seek", "to_pos"), &AudioPlayer::seek);
	ClassDB::bind_method(D_METHOD("stop"), &AudioPlayer::stop);

	ClassDB::bind_method(D_METHOD("is_playing"), &AudioPlayer::is_playing);
	ClassDB::bind_method(D_METHOD("get_pos"), &AudioPlayer::get_pos);

	ClassDB::bind_method(D_METHOD("set_bus", "bus"), &AudioPlayer::set_bus);
	ClassDB::bind_method(D_METHOD("get_bus"), &AudioPlayer::get_bus);

	ClassDB::bind_method(D_METHOD("set_autoplay", "enable"), &AudioPlayer::set_autoplay);
	ClassDB::bind_method(D_METHOD("is_autoplay_enabled"), &AudioPlayer::is_autoplay_enabled);

	ClassDB::bind_method(D_METHOD("set_mix_target", "mix_target"), &AudioPlayer::set_mix_target);
	ClassDB::bind_method(D_METHOD("get_mix_target"), &AudioPlayer::get_mix_target);

	ClassDB::bind_method(D_METHOD("_set_playing", "enable"), &AudioPlayer::_set_playing);
	ClassDB::bind_method(D_METHOD("_is_active"), &AudioPlayer::_is_active);

	ClassDB::bind_method(D_METHOD("_bus_layout_changed"), &AudioPlayer::_bus_layout_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_stream", "get_stream");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_db", PROPERTY_HINT_RANGE, "-80,24"), "set_volume_db", "get_volume_db");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_playing", "_is_active");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autoplay"), "set_autoplay", "is_autoplay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mix_target", PROPERTY_HINT_ENUM, "Stereo,Surround,Center"), "set_mix_target", "get_mix_target");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bus", PROPERTY_HINT_ENUM, ""), "set_bus", "get_bus");
}

AudioPlayer::AudioPlayer() {

	mix_volume_db = 0;
	volume_db = 0;
	autoplay = false;
	setseek = -1;
	active = false;
	mix_target = MIX_TARGET_STEREO;

	AudioServer::get_singleton()->connect("bus_layout_changed", this, "_bus_layout_changed");
}

AudioPlayer::~AudioPlayer() {
}
