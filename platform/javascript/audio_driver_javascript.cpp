/*************************************************************************/
/*  audio_driver_javascript.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_driver_javascript.h"

#include "core/project_settings.h"

#include <emscripten.h>

#include "godot_audio.h"

AudioDriverJavaScript *AudioDriverJavaScript::singleton = NULL;

bool AudioDriverJavaScript::is_available() {
	return godot_audio_is_available() != 0;
}

const char *AudioDriverJavaScript::get_name() const {
	return "JavaScript";
}

#ifndef NO_THREADS
void AudioDriverJavaScript::_audio_thread_func(void *p_data) {
	AudioDriverJavaScript *obj = static_cast<AudioDriverJavaScript *>(p_data);
	while (!obj->quit) {
		obj->lock();
		if (!obj->needs_process) {
			obj->unlock();
			OS::get_singleton()->delay_usec(1000); // Give the browser some slack.
			continue;
		}
		obj->_js_driver_process();
		obj->needs_process = false;
		obj->unlock();
	}
}
#endif

extern "C" EMSCRIPTEN_KEEPALIVE void audio_driver_process_start() {
#ifndef NO_THREADS
	AudioDriverJavaScript::singleton->lock();
#else
	AudioDriverJavaScript::singleton->_js_driver_process();
#endif
}

extern "C" EMSCRIPTEN_KEEPALIVE void audio_driver_process_end() {
#ifndef NO_THREADS
	AudioDriverJavaScript::singleton->needs_process = true;
	AudioDriverJavaScript::singleton->unlock();
#endif
}

extern "C" EMSCRIPTEN_KEEPALIVE void audio_driver_process_capture(float sample) {
	AudioDriverJavaScript::singleton->process_capture(sample);
}

void AudioDriverJavaScript::_js_driver_process() {
	int sample_count = memarr_len(internal_buffer) / channel_count;
	int32_t *stream_buffer = reinterpret_cast<int32_t *>(internal_buffer);
	audio_server_process(sample_count, stream_buffer);
	for (int i = 0; i < sample_count * channel_count; i++) {
		internal_buffer[i] = float(stream_buffer[i] >> 16) / 32768.f;
	}
}

void AudioDriverJavaScript::process_capture(float sample) {
	int32_t sample32 = int32_t(sample * 32768.f) * (1U << 16);
	input_buffer_write(sample32);
}

Error AudioDriverJavaScript::init() {
	mix_rate = GLOBAL_GET("audio/mix_rate");
	int latency = GLOBAL_GET("audio/output_latency");

	channel_count = godot_audio_init(mix_rate, latency);
	buffer_length = closest_power_of_2(latency * mix_rate / 1000);
	buffer_length = godot_audio_create_processor(buffer_length, channel_count);
	if (!buffer_length) {
		return FAILED;
	}

	if (!internal_buffer || (int)memarr_len(internal_buffer) != buffer_length * channel_count) {
		if (internal_buffer)
			memdelete_arr(internal_buffer);
		internal_buffer = memnew_arr(float, buffer_length *channel_count);
	}

	if (!internal_buffer) {
		return ERR_OUT_OF_MEMORY;
	}
	return OK;
}

void AudioDriverJavaScript::start() {
#ifndef NO_THREADS
	mutex = Mutex::create();
	thread = Thread::create(_audio_thread_func, this);
#endif
	godot_audio_start(internal_buffer);
}

void AudioDriverJavaScript::resume() {
	godot_audio_resume();
}

float AudioDriverJavaScript::get_latency() {
	return godot_audio_get_latency();
}

int AudioDriverJavaScript::get_mix_rate() const {
	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverJavaScript::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(channel_count);
}

void AudioDriverJavaScript::lock() {
#ifndef NO_THREADS
	if (mutex) {
		mutex->lock();
	}
#endif
}

void AudioDriverJavaScript::unlock() {
#ifndef NO_THREADS
	if (mutex) {
		mutex->unlock();
	}
#endif
}

void AudioDriverJavaScript::finish_async() {
#ifndef NO_THREADS
	quit = true; // Ask thread to quit.
#endif
	godot_audio_finish_async();
}

void AudioDriverJavaScript::finish() {
#ifndef NO_THREADS
	Thread::wait_to_finish(thread);
	memdelete(thread);
	thread = NULL;
	memdelete(mutex);
	mutex = NULL;
#endif
	if (internal_buffer) {
		memdelete_arr(internal_buffer);
		internal_buffer = NULL;
	}
}

Error AudioDriverJavaScript::capture_start() {
	input_buffer_init(buffer_length);
	godot_audio_capture_start();
	return OK;
}

Error AudioDriverJavaScript::capture_stop() {
	godot_audio_capture_stop();
	input_buffer.clear();
	return OK;
}

AudioDriverJavaScript::AudioDriverJavaScript() {
	internal_buffer = NULL;
	buffer_length = 0;
	mix_rate = 0;
	channel_count = 0;

#ifndef NO_THREADS
	mutex = NULL;
	thread = NULL;
	quit = false;
	needs_process = true;
#endif

	singleton = this;
}
