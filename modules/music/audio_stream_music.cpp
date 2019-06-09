/*************************************************************************/
/*  audio_stream_sample.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "audio_stream_music.h"
#include "core/io/marshalls.h"
#include "core/os/file_access.h"
#include "jar_xm.h"

void AudioStreamPlaybackMusic::start(float p_from_pos) {

	//seek(p_from_pos);
	jar_xm_reset(music_state_copy);
	active = true;
}

void AudioStreamPlaybackMusic::stop() {

	active = false;
}

bool AudioStreamPlaybackMusic::is_playing() const {

	return active;
}
int AudioStreamPlaybackMusic::get_loop_count() const {
	return jar_xm_get_loop_count(music_state_copy);
}

float AudioStreamPlaybackMusic::get_playback_position() const {
	uint8_t pattern_index;
	uint8_t pattern;
	uint8_t row;
	uint64_t samples;

	jar_xm_get_position(music_state_copy, &pattern_index, &pattern, &row, &samples);
	return samples / base->mix_rate;
}
void AudioStreamPlaybackMusic::seek(float p_time) {
	//Figure out how to convert time to pattern position.
	//jar_xm_seek(music_state_copy, x, y, z);
}

void AudioStreamPlaybackMusic::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {

	if (!base->is_loaded || !active) {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0, 0);
		}
		return;
	}

	// I guess when godot draws the waveform ::mix gets called with larger p_frames... not implemented for xm.
	if (p_frames > PCM_BUFFER_SIZE) {
		printf("avoiding crash! p_frames=%d\n", p_frames);
		return;
	}

	jar_xm_set_playback_rate(music_state_copy, p_rate_scale);
	float *buf = (float *)pcm_buffer;
	jar_xm_generate_samples(music_state_copy, buf, p_frames);

	for (int i = 0; i < p_frames; i++) {
		p_buffer[i] = AudioFrame(buf[i * 2], buf[i * 2 + 1]);
	}
}

AudioStreamPlaybackMusic::AudioStreamPlaybackMusic() {
	music_state_copy = NULL;
	active = false;
	pcm_buffer = AudioServer::get_singleton()->audio_data_alloc(PCM_BUFFER_SIZE * 2);
}

AudioStreamPlaybackMusic::~AudioStreamPlaybackMusic() {
	if (pcm_buffer) {
		AudioServer::get_singleton()->audio_data_free(pcm_buffer);
		pcm_buffer = NULL;
	}
	if (music_state_copy) {
		free(music_state_copy);
	}
}
/////////////////////

void AudioStreamMusic::set_mix_rate(int p_hz) {

	mix_rate = p_hz;
}
int AudioStreamMusic::get_mix_rate() const {

	return mix_rate;
}

float AudioStreamMusic::get_length() const {

	return 0;
}

Ref<AudioStreamPlayback> AudioStreamMusic::instance_playback() {

	Ref<AudioStreamPlaybackMusic> music;
	music.instance();
	music->base = Ref<AudioStreamMusic>(this);
	if (music->music_state_copy) {
		free(music->music_state_copy);
	}

	music->music_state_copy = jar_xm_context_copy(musicptr);
	is_loaded = true;
	return music;
}

String AudioStreamMusic::get_stream_name() const {

	return "";
}

void AudioStreamMusic::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_mix_rate", "mix_rate"), &AudioStreamMusic::set_mix_rate);
	ClassDB::bind_method(D_METHOD("get_mix_rate"), &AudioStreamMusic::get_mix_rate);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mix_rate"), "set_mix_rate", "get_mix_rate");
}

AudioStreamMusic::AudioStreamMusic() {
	mix_rate = 44100;
	is_loaded = false;
}
AudioStreamMusic::~AudioStreamMusic() {
}

Error AudioStreamMusic::set_file(const String &p_path) {
	Error error_file;
	FileAccess *fh = FileAccess::open(p_path, FileAccess::READ, &error_file);
	if (error_file) {
		printf("Error opening file: %s\n", p_path.ascii());
		return FAILED;
	}

	mod_data.resize(fh->get_len());

	PoolVector<uint8_t>::Write w = mod_data.write();

	int i = 0;
	while (!fh->eof_reached()) {
		w[i++] = fh->get_8();
	}

	printf("Loaded: %d bytes\n", mod_data.size());

	fh->close();

	int ret = jar_xm_create_context(&musicptr, (char *)w.ptr(), mix_rate);
	printf("Return val: %d\n", ret);
	return OK;
}

RES ResourceFormatLoaderAudioStreamMusic::load(const String &p_path, const String &p_original_path, Error *r_error) {
	if (r_error)
		*r_error = OK;

	AudioStreamMusic *music_stream = memnew(AudioStreamMusic);
	Error file_err = music_stream->set_file(p_path);
	if (file_err)
		return NULL; //dont know what to return here really...

	return Ref<AudioStreamMusic>(music_stream);
}

void ResourceFormatLoaderAudioStreamMusic::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("xm");
}
String ResourceFormatLoaderAudioStreamMusic::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower() == "xm")
		return "AudioStreamMusic";
	return "";
}

bool ResourceFormatLoaderAudioStreamMusic::handles_type(const String &p_type) const {
	return (p_type == "AudioStream" || p_type == "AudioStreamMusic");
}
