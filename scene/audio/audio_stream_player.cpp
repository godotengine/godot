/*************************************************************************/
/*  audio_stream_player.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_stream_player.h"

#include "core/config/engine.h"
#include "core/math/audio_frame.h"
#include "servers/audio_server.h"

void AudioStreamPlayer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (autoplay && !Engine::get_singleton()->is_editor_hint()) {
				play();
			}
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			Vector<Ref<AudioStreamPlayback>> playbacks_to_remove;
			for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
				if (playback.is_valid() && !AudioServer::get_singleton()->is_playback_active(playback) && !AudioServer::get_singleton()->is_playback_paused(playback)) {
					playbacks_to_remove.push_back(playback);
				}
			}
			// Now go through and remove playbacks that have finished. Removing elements from a Vector in a range based for is asking for trouble.
			for (Ref<AudioStreamPlayback> &playback : playbacks_to_remove) {
				stream_playbacks.erase(playback);
			}
			if (!playbacks_to_remove.is_empty() && stream_playbacks.is_empty()) {
				// This node is no longer actively playing audio.
				active.clear();
				set_process_internal(false);
			}
			if (!playbacks_to_remove.is_empty()) {
				emit_signal(SNAME("finished"));
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
				AudioServer::get_singleton()->stop_playback_stream(playback);
			}
			stream_playbacks.clear();
		} break;

		case NOTIFICATION_PAUSED: {
			if (!can_process()) {
				// Node can't process so we start fading out to silence
				set_stream_paused(true);
			}
		} break;

		case NOTIFICATION_UNPAUSED: {
			set_stream_paused(false);
		} break;
	}
}

void AudioStreamPlayer::set_stream(Ref<AudioStream> p_stream) {
	stop();
	stream = p_stream;
}

Ref<AudioStream> AudioStreamPlayer::get_stream() const {
	return stream;
}

void AudioStreamPlayer::set_volume_db(float p_volume) {
	volume_db = p_volume;

	update_sends_internal();
}

float AudioStreamPlayer::get_volume_db() const {
	return volume_db;
}

void AudioStreamPlayer::set_pitch_scale(float p_pitch_scale) {
	ERR_FAIL_COND(p_pitch_scale <= 0.0);
	pitch_scale = p_pitch_scale;

	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->set_playback_pitch_scale(playback, pitch_scale);
	}
}

float AudioStreamPlayer::get_pitch_scale() const {
	return pitch_scale;
}

void AudioStreamPlayer::set_max_polyphony(int p_max_polyphony) {
	if (p_max_polyphony > 0) {
		max_polyphony = p_max_polyphony;
	}
}

int AudioStreamPlayer::get_max_polyphony() const {
	return max_polyphony;
}

void AudioStreamPlayer::play(float p_from_pos) {
	if (stream.is_null()) {
		return;
	}
	ERR_FAIL_COND_MSG(!is_inside_tree(), "Playback can only happen when a node is inside the scene tree");
	if (stream->is_monophonic() && is_playing()) {
		stop();
	}
	Ref<AudioStreamPlayback> stream_playback = stream->instance_playback();
	ERR_FAIL_COND_MSG(stream_playback.is_null(), "Failed to instantiate playback.");

	AudioServer::get_singleton()->start_playback_stream(stream_playback, get_sends_internal(), p_from_pos, pitch_scale);
	stream_playbacks.push_back(stream_playback);
	active.set();
	set_process_internal(true);
	while (stream_playbacks.size() > max_polyphony) {
		AudioServer::get_singleton()->stop_playback_stream(stream_playbacks[0]);
		stream_playbacks.remove_at(0);
	}
}

void AudioStreamPlayer::seek(float p_seconds) {
	if (is_playing()) {
		stop();
		play(p_seconds);
	}
}

void AudioStreamPlayer::stop() {
	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->stop_playback_stream(playback);
	}
	stream_playbacks.clear();
	active.clear();
	set_process_internal(false);
}

bool AudioStreamPlayer::is_playing() const {
	for (const Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		if (AudioServer::get_singleton()->is_playback_active(playback)) {
			return true;
		}
	}
	return false;
}

float AudioStreamPlayer::get_playback_position() {
	// Return the playback position of the most recently started playback stream.
	if (!stream_playbacks.is_empty()) {
		return AudioServer::get_singleton()->get_playback_position(stream_playbacks[stream_playbacks.size() - 1]);
	}
	return 0;
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
	if (p_enable) {
		play();
	} else {
		stop();
	}
}

bool AudioStreamPlayer::_is_active() const {
	for (const Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		if (AudioServer::get_singleton()->is_playback_active(playback)) {
			return true;
		}
	}
	return false;
}

void AudioStreamPlayer::set_stream_paused(bool p_pause) {
	// TODO this does not have perfect recall, fix that maybe? If there are zero playbacks registered with the AudioServer, this bool isn't persisted.
	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->set_playback_paused(playback, p_pause);
	}
}

bool AudioStreamPlayer::get_stream_paused() const {
	// There's currently no way to pause some playback streams but not others. Check the first and don't bother looking at the rest.
	if (!stream_playbacks.is_empty()) {
		return AudioServer::get_singleton()->is_playback_paused(stream_playbacks[0]);
	}
	return false;
}

Vector<AudioFrame> AudioStreamPlayer::_get_volume_vector() const {
	Vector<AudioFrame> volume_vector;
	// We need at most four stereo pairs (for 7.1 systems).
	volume_vector.resize(4);

	// Initialize the volume vector to zero.
	for (AudioFrame &channel_volume_db : volume_vector) {
		channel_volume_db = AudioFrame(0, 0);
	}

	float volume_linear = Math::db2linear(volume_db);

	// Set the volume vector up according to the speaker mode and mix target.
	// TODO do we need to scale the volume down when we output to more channels?
	if (AudioServer::get_singleton()->get_speaker_mode() == AudioServer::SPEAKER_MODE_STEREO) {
		volume_vector.write[0] = AudioFrame(volume_linear, volume_linear);
	} else {
		switch (mix_target) {
			case MIX_TARGET_STEREO: {
				volume_vector.write[0] = AudioFrame(volume_linear, volume_linear);
			} break;
			case MIX_TARGET_SURROUND: {
				// TODO Make sure this is right.
				volume_vector.write[0] = AudioFrame(volume_linear, volume_linear);
				volume_vector.write[1] = AudioFrame(volume_linear, /* LFE= */ 1.0f);
				volume_vector.write[2] = AudioFrame(volume_linear, volume_linear);
				volume_vector.write[3] = AudioFrame(volume_linear, volume_linear);
			} break;
			case MIX_TARGET_CENTER: {
				// TODO Make sure this is right.
				volume_vector.write[1] = AudioFrame(volume_linear, /* LFE= */ 1.0f);
			} break;
		}
	}
	return volume_vector;
}

void AudioStreamPlayer::_validate_property(PropertyInfo &property) const {
	if (property.name.ends_with("/send")) {
		String options;
		for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
			if (i > 0) {
				options += ",";
			}
			String name = AudioServer::get_singleton()->get_bus_name(i);
			options += name;
		}

		property.hint_string = options;
	}

	Node::_validate_property(property);
}

void AudioStreamPlayer::_bus_layout_changed() {
	notify_property_list_changed();
}

void AudioStreamPlayer::add_send(int p_index) {
	if (p_index < 0) {
		p_index = sends.size();
	}
	ERR_FAIL_COND(p_index > sends.size());
	SendEntry entry{ SNAME("Master"), 1.0f };
	sends.insert(p_index, entry);
	emit_signal(SNAME("changed"));
	update_sends_internal();
	notify_property_list_changed();
}

void AudioStreamPlayer::move_send(int p_index_from, int p_index_to) {
	ERR_FAIL_COND(p_index_from < 0);
	ERR_FAIL_COND(p_index_from >= sends.size());
	ERR_FAIL_COND(p_index_to < 0);
	ERR_FAIL_COND(p_index_to > sends.size());
	sends.insert(p_index_to, sends[p_index_from]);
	// If 'from' is strictly after 'to' we need to increment the index by one because of the insertion.
	if (p_index_from > p_index_to) {
		p_index_from++;
	}
	sends.remove_at(p_index_from);
	emit_signal(SNAME("changed"));
	update_sends_internal();
	notify_property_list_changed();
}

void AudioStreamPlayer::remove_send(int p_index) {
	ERR_FAIL_COND(p_index < 0);
	ERR_FAIL_COND(p_index >= sends.size());
	sends.remove_at(p_index);
	if (sends.size() == 0) {
		add_send(0); // Don't let the user remove all sends.
	} else {
		emit_signal(SNAME("changed"));
		update_sends_internal();
		notify_property_list_changed();
	}
}

void AudioStreamPlayer::set_send(int p_index, StringName p_send_name) {
	ERR_FAIL_COND(p_index < 0);
	ERR_FAIL_COND(p_index >= sends.size());
	sends.write[p_index].send = p_send_name;
	update_sends_internal();
}

StringName AudioStreamPlayer::get_send(int p_index) const {
	ERR_FAIL_COND_V(p_index < 0, SNAME(""));
	ERR_FAIL_COND_V(p_index >= sends.size(), SNAME(""));
	return sends[p_index].send;
}

void AudioStreamPlayer::set_send_loudness_scale(int p_index, float p_loudness_scale) {
	ERR_FAIL_COND(p_index < 0);
	ERR_FAIL_COND(p_index >= sends.size());
	if (p_loudness_scale < 0) {
		p_loudness_scale = 0;
	}
	sends.write[p_index].loudness_scale = p_loudness_scale;
	update_sends_internal();
}

float AudioStreamPlayer::get_send_loudness_scale(int p_index) const {
	ERR_FAIL_COND_V(p_index < 0, 0.0f);
	ERR_FAIL_COND_V(p_index >= sends.size(), 0.0f);
	return sends[p_index].loudness_scale;
}

int AudioStreamPlayer::find_send_index_by_name(StringName p_send_name) const {
	for (int send_idx = 0; send_idx < sends.size(); send_idx++) {
		if (sends[send_idx].send == p_send_name) {
			return send_idx;
		}
	}
	return -1;
}

void AudioStreamPlayer::set_sends_count(int p_count) {
	if (p_count >= 1) {
		sends.resize(p_count);
	} else {
		sends.resize(0);
		add_send(0);
	}
	emit_signal(SNAME("changed"));
	update_sends_internal();
	notify_property_list_changed();
}

int AudioStreamPlayer::get_sends_count() const {
	return sends.size();
}

Map<StringName, Vector<AudioFrame>> AudioStreamPlayer::get_sends_internal() const {
	// If the user set up this player to send multiple times to the same bus, add the loudness together.
	Map<StringName, float> per_send_loudness_scale_total;
	for (int send_idx = 0; send_idx < sends.size(); send_idx++) {
		per_send_loudness_scale_total[sends[send_idx].send] += sends[send_idx].loudness_scale;
	}
	// Set up each volume vector according to the totals.
	Map<StringName, Vector<AudioFrame>> per_send_volume_vector;
	for (KeyValue<StringName, float> pair : per_send_loudness_scale_total) {
		Vector<AudioFrame> volume_vector = _get_volume_vector();
		for (AudioFrame &volume : volume_vector) {
			volume *= pair.value;
		}
		per_send_volume_vector[pair.key] = volume_vector;
	}
	return per_send_volume_vector;
}

void AudioStreamPlayer::update_sends_internal() {
#ifdef TOOLS_ENABLED
	update_configuration_warnings();
#endif
	Map<StringName, Vector<AudioFrame>> sends_volume_linear = get_sends_internal();
	for (Ref<AudioStreamPlayback> playback : stream_playbacks) {
		AudioServer::get_singleton()->set_playback_bus_volumes_linear(playback, sends_volume_linear);
	}
}

Ref<AudioStreamPlayback> AudioStreamPlayer::get_stream_playback() {
	if (!stream_playbacks.is_empty()) {
		return stream_playbacks[stream_playbacks.size() - 1];
	}
	return nullptr;
}

TypedArray<String> AudioStreamPlayer::get_configuration_warnings() const {
	float total_loudness = 0;
	for (int send_idx = 0; send_idx < sends.size(); send_idx++) {
		total_loudness += sends[send_idx].loudness_scale;
	}
	TypedArray<String> warnings;
	if (total_loudness == 0) {
		warnings.append("This player node is not configured to send audio to a bus (or the sends are configured to be silent).");
	}
	return warnings;
}

bool AudioStreamPlayer::_get(const StringName &p_name, Variant &r_ret) const {
	if (Node::_get(p_name, r_ret)) {
		return true;
	}
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() == 2 && components[0].begins_with("send_") && components[0].trim_prefix("send_").is_valid_int()) {
		int index = components[0].trim_prefix("send_").to_int();
		if (index < 0 || index >= (int)sends.size()) {
			return false;
		}

		if (components[1] == "send") {
			r_ret = get_send(index);
			return true;
		} else if (components[1] == "loudness_scale") {
			r_ret = get_send_loudness_scale(index);
			return true;
		} else {
			return false;
		}
	}
	return false;
}

bool AudioStreamPlayer::_set(const StringName &p_name, const Variant &p_value) {
	if (Node::_set(p_name, p_value)) {
		return true;
	}
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() == 2 && components[0].begins_with("send_") && components[0].trim_prefix("send_").is_valid_int()) {
		int index = components[0].trim_prefix("send_").to_int();
		if (index < 0 || index >= (int)sends.size()) {
			return false;
		}

		if (components[1] == "send") {
			set_send(index, p_value);
			return true;
		} else if (components[1] == "loudness_scale") {
			set_send_loudness_scale(index, p_value);
			return true;
		} else {
			return false;
		}
	}
	return false;
}

void AudioStreamPlayer::_get_property_list(List<PropertyInfo> *p_list) const {
	Node::_get_property_list(p_list);

	String bus_hint_string;
	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (i > 0) {
			bus_hint_string += ",";
		}
		String name = AudioServer::get_singleton()->get_bus_name(i);
		bus_hint_string += name;
	}

	p_list->push_back(PropertyInfo(Variant::NIL, "Sends", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
	for (int i = 0; i < sends.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, vformat("send_%d/send", i), PROPERTY_HINT_ENUM, bus_hint_string));
		p_list->push_back(PropertyInfo(Variant::FLOAT, vformat("send_%d/loudness_scale", i), PROPERTY_HINT_RANGE, "0,2,0.001,or_greater"));
	}
}

void AudioStreamPlayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_stream", "stream"), &AudioStreamPlayer::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream"), &AudioStreamPlayer::get_stream);

	ClassDB::bind_method(D_METHOD("set_volume_db", "volume_db"), &AudioStreamPlayer::set_volume_db);
	ClassDB::bind_method(D_METHOD("get_volume_db"), &AudioStreamPlayer::get_volume_db);

	ClassDB::bind_method(D_METHOD("set_pitch_scale", "pitch_scale"), &AudioStreamPlayer::set_pitch_scale);
	ClassDB::bind_method(D_METHOD("get_pitch_scale"), &AudioStreamPlayer::get_pitch_scale);

	ClassDB::bind_method(D_METHOD("play", "from_position"), &AudioStreamPlayer::play, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("seek", "to_position"), &AudioStreamPlayer::seek);
	ClassDB::bind_method(D_METHOD("stop"), &AudioStreamPlayer::stop);

	ClassDB::bind_method(D_METHOD("is_playing"), &AudioStreamPlayer::is_playing);
	ClassDB::bind_method(D_METHOD("get_playback_position"), &AudioStreamPlayer::get_playback_position);

	ClassDB::bind_method(D_METHOD("set_autoplay", "enable"), &AudioStreamPlayer::set_autoplay);
	ClassDB::bind_method(D_METHOD("is_autoplay_enabled"), &AudioStreamPlayer::is_autoplay_enabled);

	ClassDB::bind_method(D_METHOD("set_mix_target", "mix_target"), &AudioStreamPlayer::set_mix_target);
	ClassDB::bind_method(D_METHOD("get_mix_target"), &AudioStreamPlayer::get_mix_target);

	ClassDB::bind_method(D_METHOD("_set_playing", "enable"), &AudioStreamPlayer::_set_playing);
	ClassDB::bind_method(D_METHOD("_is_active"), &AudioStreamPlayer::_is_active);

	ClassDB::bind_method(D_METHOD("set_stream_paused", "pause"), &AudioStreamPlayer::set_stream_paused);
	ClassDB::bind_method(D_METHOD("get_stream_paused"), &AudioStreamPlayer::get_stream_paused);

	ClassDB::bind_method(D_METHOD("set_max_polyphony", "max_polyphony"), &AudioStreamPlayer::set_max_polyphony);
	ClassDB::bind_method(D_METHOD("get_max_polyphony"), &AudioStreamPlayer::get_max_polyphony);

	ClassDB::bind_method(D_METHOD("get_stream_playback"), &AudioStreamPlayer::get_stream_playback);

	ClassDB::bind_method(D_METHOD("add_send", "index"), &AudioStreamPlayer::add_send);
	ClassDB::bind_method(D_METHOD("move_send", "index_from", "index_to"), &AudioStreamPlayer::move_send);
	ClassDB::bind_method(D_METHOD("remove_send", "index"), &AudioStreamPlayer::remove_send);

	ClassDB::bind_method(D_METHOD("set_send", "index", "send_name"), &AudioStreamPlayer::set_send);
	ClassDB::bind_method(D_METHOD("get_send", "index"), &AudioStreamPlayer::get_send);
	ClassDB::bind_method(D_METHOD("set_send_loudness_scale", "index", "loudness_scale"), &AudioStreamPlayer::set_send_loudness_scale);
	ClassDB::bind_method(D_METHOD("get_send_loudness_scale", "index"), &AudioStreamPlayer::get_send_loudness_scale);
	ClassDB::bind_method(D_METHOD("find_send_index_by_name", "send_name"), &AudioStreamPlayer::find_send_index_by_name);
	ClassDB::bind_method(D_METHOD("set_sends_count", "count"), &AudioStreamPlayer::set_sends_count);
	ClassDB::bind_method(D_METHOD("get_sends_count"), &AudioStreamPlayer::get_sends_count);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "sends_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_sends_count", "get_sends_count");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_stream", "get_stream");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "volume_db", PROPERTY_HINT_RANGE, "-80,24"), "set_volume_db", "get_volume_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pitch_scale", PROPERTY_HINT_RANGE, "0.01,4,0.01,or_greater"), "set_pitch_scale", "get_pitch_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_playing", "is_playing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autoplay"), "set_autoplay", "is_autoplay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stream_paused", PROPERTY_HINT_NONE, ""), "set_stream_paused", "get_stream_paused");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mix_target", PROPERTY_HINT_ENUM, "Stereo,Surround,Center"), "set_mix_target", "get_mix_target");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_polyphony", PROPERTY_HINT_NONE, ""), "set_max_polyphony", "get_max_polyphony");

	ADD_ARRAY("sends", "send_");

	ADD_SIGNAL(MethodInfo("finished"));

	BIND_ENUM_CONSTANT(MIX_TARGET_STEREO);
	BIND_ENUM_CONSTANT(MIX_TARGET_SURROUND);
	BIND_ENUM_CONSTANT(MIX_TARGET_CENTER);
}

AudioStreamPlayer::AudioStreamPlayer() {
	add_send(0);
	AudioServer::get_singleton()->connect("bus_layout_changed", callable_mp(this, &AudioStreamPlayer::_bus_layout_changed));
}

AudioStreamPlayer::~AudioStreamPlayer() {
}
