/**************************************************************************/
/*  audio_stream_randomizer.cpp                                           */
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

#include "audio_stream_randomizer.h"

#include "core/object/class_db.h"

void AudioStreamRandomizer::add_stream(int p_index, Ref<AudioStream> p_stream, float p_weight) {
	if (p_index < 0) {
		p_index = audio_stream_pool.size();
	}
	ERR_FAIL_COND(p_index > audio_stream_pool.size());
	PoolEntry entry{ p_stream, p_weight };
	audio_stream_pool.insert(p_index, entry);
	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

// p_index_to is relative to the array prior to the removal of from.
// Example: [0, 1, 2, 3], move(1, 3) => [0, 2, 1, 3]
void AudioStreamRandomizer::move_stream(int p_index_from, int p_index_to) {
	ERR_FAIL_INDEX(p_index_from, audio_stream_pool.size());
	// p_index_to == audio_stream_pool.size() is valid (move to end).
	ERR_FAIL_COND(p_index_to < 0);
	ERR_FAIL_COND(p_index_to > audio_stream_pool.size());
	audio_stream_pool.insert(p_index_to, audio_stream_pool[p_index_from]);
	// If 'from' is strictly after 'to' we need to increment the index by one because of the insertion.
	if (p_index_from > p_index_to) {
		p_index_from++;
	}
	audio_stream_pool.remove_at(p_index_from);
	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

void AudioStreamRandomizer::remove_stream(int p_index) {
	ERR_FAIL_INDEX(p_index, audio_stream_pool.size());
	audio_stream_pool.remove_at(p_index);
	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

void AudioStreamRandomizer::set_stream(int p_index, Ref<AudioStream> p_stream) {
	ERR_FAIL_INDEX(p_index, audio_stream_pool.size());
	audio_stream_pool.write[p_index].stream = p_stream;
	emit_signal(CoreStringName(changed));
}

Ref<AudioStream> AudioStreamRandomizer::get_stream(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, audio_stream_pool.size(), nullptr);
	return audio_stream_pool[p_index].stream;
}

void AudioStreamRandomizer::set_stream_probability_weight(int p_index, float p_weight) {
	ERR_FAIL_INDEX(p_index, audio_stream_pool.size());
	audio_stream_pool.write[p_index].weight = p_weight;
	emit_signal(CoreStringName(changed));
}

float AudioStreamRandomizer::get_stream_probability_weight(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, audio_stream_pool.size(), 0);
	return audio_stream_pool[p_index].weight;
}

void AudioStreamRandomizer::set_streams_count(int p_count) {
	audio_stream_pool.resize(p_count);
}

int AudioStreamRandomizer::get_streams_count() const {
	return audio_stream_pool.size();
}

void AudioStreamRandomizer::set_random_pitch(float p_pitch) {
	if (p_pitch < 1) {
		p_pitch = 1;
	}
	random_pitch_scale = p_pitch;
}

float AudioStreamRandomizer::get_random_pitch() const {
	return random_pitch_scale;
}

void AudioStreamRandomizer::set_random_pitch_semitones(float p_semitones) {
	random_pitch_scale = powf(2, p_semitones * 0.08333333f);
}

float AudioStreamRandomizer::get_random_pitch_semitones() const {
	return 12.0f * log2f(MAX(1.0f, random_pitch_scale));
}

void AudioStreamRandomizer::set_random_volume_offset_db(float p_volume_offset_db) {
	if (p_volume_offset_db < 0) {
		p_volume_offset_db = 0;
	}
	random_volume_offset_db = p_volume_offset_db;
}

float AudioStreamRandomizer::get_random_volume_offset_db() const {
	return random_volume_offset_db;
}

void AudioStreamRandomizer::set_playback_mode(PlaybackMode p_playback_mode) {
	playback_mode = p_playback_mode;
}

AudioStreamRandomizer::PlaybackMode AudioStreamRandomizer::get_playback_mode() const {
	return playback_mode;
}

Ref<AudioStreamPlayback> AudioStreamRandomizer::instance_playback_random() {
	Ref<AudioStreamPlaybackRandomizer> playback;
	playback.instantiate();
	playbacks.insert(playback.ptr());
	playback->randomizer = Ref<AudioStreamRandomizer>((AudioStreamRandomizer *)this);

	double total_weight = 0;
	Vector<PoolEntry> local_pool;
	for (const PoolEntry &entry : audio_stream_pool) {
		if (entry.stream.is_valid() && entry.weight > 0) {
			local_pool.push_back(entry);
			total_weight += entry.weight;
		}
	}
	if (local_pool.is_empty()) {
		return playback;
	}
	double chosen_cumulative_weight = Math::random(0.0, total_weight);
	double cumulative_weight = 0;
	for (PoolEntry &entry : local_pool) {
		cumulative_weight += entry.weight;
		if (cumulative_weight > chosen_cumulative_weight) {
			playback->playback = entry.stream->instantiate_playback();
			last_playback = entry.stream;
			break;
		}
	}
	if (playback->playback.is_null()) {
		// This indicates a floating point error. Take the last element.
		last_playback = local_pool[local_pool.size() - 1].stream;
		playback->playback = local_pool.write[local_pool.size() - 1].stream->instantiate_playback();
	}
	return playback;
}

Ref<AudioStreamPlayback> AudioStreamRandomizer::instance_playback_no_repeats() {
	Ref<AudioStreamPlaybackRandomizer> playback;

	double total_weight = 0;
	Vector<PoolEntry> local_pool;
	for (const PoolEntry &entry : audio_stream_pool) {
		if (entry.stream == last_playback) {
			continue;
		}
		if (entry.stream.is_valid() && entry.weight > 0) {
			local_pool.push_back(entry);
			total_weight += entry.weight;
		}
	}
	if (local_pool.is_empty()) {
		// There is only one sound to choose from.
		// Always play a random sound while allowing repeats (which always plays the same sound).
		playback = instance_playback_random();
		return playback;
	}

	playback.instantiate();
	playbacks.insert(playback.ptr());
	playback->randomizer = Ref<AudioStreamRandomizer>((AudioStreamRandomizer *)this);
	double chosen_cumulative_weight = Math::random(0.0, total_weight);
	double cumulative_weight = 0;
	for (PoolEntry &entry : local_pool) {
		cumulative_weight += entry.weight;
		if (cumulative_weight > chosen_cumulative_weight) {
			last_playback = entry.stream;
			playback->playback = entry.stream->instantiate_playback();
			break;
		}
	}
	if (playback->playback.is_null()) {
		// This indicates a floating point error. Take the last element.
		last_playback = local_pool[local_pool.size() - 1].stream;
		playback->playback = local_pool.write[local_pool.size() - 1].stream->instantiate_playback();
	}
	return playback;
}

Ref<AudioStreamPlayback> AudioStreamRandomizer::instance_playback_sequential() {
	Ref<AudioStreamPlaybackRandomizer> playback;
	playback.instantiate();
	playbacks.insert(playback.ptr());
	playback->randomizer = Ref<AudioStreamRandomizer>((AudioStreamRandomizer *)this);

	Vector<Ref<AudioStream>> local_pool;
	for (const PoolEntry &entry : audio_stream_pool) {
		if (entry.stream.is_null()) {
			continue;
		}
		if (local_pool.has(entry.stream)) {
			WARN_PRINT("Duplicate stream in sequential playback pool");
			continue;
		}
		local_pool.push_back(entry.stream);
	}
	if (local_pool.is_empty()) {
		return playback;
	}
	bool found_last_stream = false;
	for (Ref<AudioStream> &entry : local_pool) {
		if (found_last_stream) {
			last_playback = entry;
			playback->playback = entry->instantiate_playback();
			break;
		}
		if (entry == last_playback) {
			found_last_stream = true;
		}
	}
	if (playback->playback.is_null()) {
		// Wrap around
		last_playback = local_pool[0];
		playback->playback = local_pool.write[0]->instantiate_playback();
	}
	return playback;
}

Ref<AudioStreamPlayback> AudioStreamRandomizer::instantiate_playback() {
	switch (playback_mode) {
		case PLAYBACK_RANDOM:
			return instance_playback_random();
		case PLAYBACK_RANDOM_NO_REPEATS:
			return instance_playback_no_repeats();
		case PLAYBACK_SEQUENTIAL:
			return instance_playback_sequential();
		default:
			ERR_FAIL_V_MSG(nullptr, "Unhandled playback mode.");
	}
}

double AudioStreamRandomizer::get_length() const {
	if (!last_playback.is_valid()) {
		return 0;
	}
	return last_playback->get_length();
}

bool AudioStreamRandomizer::is_monophonic() const {
	for (const PoolEntry &entry : audio_stream_pool) {
		if (entry.stream.is_valid() && entry.stream->is_monophonic()) {
			return true;
		}
	}
	return false;
}

void AudioStreamRandomizer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_stream", "index", "stream", "weight"), &AudioStreamRandomizer::add_stream, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("move_stream", "index_from", "index_to"), &AudioStreamRandomizer::move_stream);
	ClassDB::bind_method(D_METHOD("remove_stream", "index"), &AudioStreamRandomizer::remove_stream);

	ClassDB::bind_method(D_METHOD("set_stream", "index", "stream"), &AudioStreamRandomizer::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream", "index"), &AudioStreamRandomizer::get_stream);
	ClassDB::bind_method(D_METHOD("set_stream_probability_weight", "index", "weight"), &AudioStreamRandomizer::set_stream_probability_weight);
	ClassDB::bind_method(D_METHOD("get_stream_probability_weight", "index"), &AudioStreamRandomizer::get_stream_probability_weight);

	ClassDB::bind_method(D_METHOD("set_streams_count", "count"), &AudioStreamRandomizer::set_streams_count);
	ClassDB::bind_method(D_METHOD("get_streams_count"), &AudioStreamRandomizer::get_streams_count);

	ClassDB::bind_method(D_METHOD("set_random_pitch", "scale"), &AudioStreamRandomizer::set_random_pitch);
	ClassDB::bind_method(D_METHOD("get_random_pitch"), &AudioStreamRandomizer::get_random_pitch);

	ClassDB::bind_method(D_METHOD("set_random_pitch_semitones", "semitones"), &AudioStreamRandomizer::set_random_pitch_semitones);
	ClassDB::bind_method(D_METHOD("get_random_pitch_semitones"), &AudioStreamRandomizer::get_random_pitch_semitones);

	ClassDB::bind_method(D_METHOD("set_random_volume_offset_db", "db_offset"), &AudioStreamRandomizer::set_random_volume_offset_db);
	ClassDB::bind_method(D_METHOD("get_random_volume_offset_db"), &AudioStreamRandomizer::get_random_volume_offset_db);

	ClassDB::bind_method(D_METHOD("set_playback_mode", "mode"), &AudioStreamRandomizer::set_playback_mode);
	ClassDB::bind_method(D_METHOD("get_playback_mode"), &AudioStreamRandomizer::get_playback_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "playback_mode", PROPERTY_HINT_ENUM, "Random (Avoid Repeats),Random,Sequential"), "set_playback_mode", "get_playback_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "random_pitch", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_random_pitch", "get_random_pitch");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "random_pitch_semitones", PROPERTY_HINT_RANGE, "0,24,0.001,or_greater,suffix:Semitones", PROPERTY_USAGE_EDITOR), "set_random_pitch_semitones", "get_random_pitch_semitones");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "random_volume_offset_db", PROPERTY_HINT_RANGE, "0,40,0.01,suffix:dB"), "set_random_volume_offset_db", "get_random_volume_offset_db");
	ADD_ARRAY("streams", "stream_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "streams_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_streams_count", "get_streams_count");

	BIND_ENUM_CONSTANT(PLAYBACK_RANDOM_NO_REPEATS);
	BIND_ENUM_CONSTANT(PLAYBACK_RANDOM);
	BIND_ENUM_CONSTANT(PLAYBACK_SEQUENTIAL);

	PoolEntry defaults;

	base_property_helper.set_prefix("stream_");
	base_property_helper.set_array_length_getter(&AudioStreamRandomizer::get_streams_count);
	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, AudioStream::get_class_static()), defaults.stream, &AudioStreamRandomizer::set_stream, &AudioStreamRandomizer::get_stream);
	base_property_helper.register_property(PropertyInfo(Variant::FLOAT, "weight", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), defaults.weight, &AudioStreamRandomizer::set_stream_probability_weight, &AudioStreamRandomizer::get_stream_probability_weight);
	PropertyListHelper::register_base_helper(get_class_static(), &base_property_helper);
}

AudioStreamRandomizer::AudioStreamRandomizer() {
	property_helper.setup_for_instance(base_property_helper, this);
}

void AudioStreamPlaybackRandomizer::start(double p_from_pos) {
	playing = playback;
	{
		// GH-10238 : Pitch_scale is multiplicative, so picking a random number for it without log
		// conversion will bias it towards higher pitches (0.5 is down one octave, 2.0 is up one octave).
		// See: https://pressbooks.pub/sound/chapter/pitch-and-frequency-in-music/
		float range_from = Math::log(1.0f / randomizer->random_pitch_scale);
		float range_to = Math::log(randomizer->random_pitch_scale);

		pitch_scale = Math::exp(range_from + Math::randf() * (range_to - range_from));
	}
	{
		float range_from = -randomizer->random_volume_offset_db;
		float range_to = randomizer->random_volume_offset_db;

		float volume_offset_db = range_from + Math::randf() * (range_to - range_from);
		volume_scale = Math::db_to_linear(volume_offset_db);
	}

	if (playing.is_valid()) {
		playing->start(p_from_pos);
	}
}

void AudioStreamPlaybackRandomizer::stop() {
	if (playing.is_valid()) {
		playing->stop();
	}
}

bool AudioStreamPlaybackRandomizer::is_playing() const {
	if (playing.is_valid()) {
		return playing->is_playing();
	}

	return false;
}

int AudioStreamPlaybackRandomizer::get_loop_count() const {
	if (playing.is_valid()) {
		return playing->get_loop_count();
	}

	return 0;
}

double AudioStreamPlaybackRandomizer::get_playback_position() const {
	if (playing.is_valid()) {
		return playing->get_playback_position();
	}

	return 0;
}

void AudioStreamPlaybackRandomizer::seek(double p_time) {
	if (playing.is_valid()) {
		playing->seek(p_time);
	}
}

void AudioStreamPlaybackRandomizer::tag_used_streams() {
	Ref<AudioStreamPlayback> p = playing; // Thread safety
	if (p.is_valid()) {
		p->tag_used_streams();
	}
	randomizer->tag_used(0);
}

int AudioStreamPlaybackRandomizer::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	if (playing.is_valid()) {
		int mixed_samples = playing->mix(p_buffer, p_rate_scale * pitch_scale, p_frames);
		for (int samp = 0; samp < mixed_samples; samp++) {
			p_buffer[samp] *= volume_scale;
		}
		return mixed_samples;
	} else {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0, 0);
		}
		return p_frames;
	}
}

AudioStreamPlaybackRandomizer::~AudioStreamPlaybackRandomizer() {
	randomizer->playbacks.erase(this);
}
