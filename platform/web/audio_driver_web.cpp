/**************************************************************************/
/*  audio_driver_web.cpp                                                  */
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

#include "audio_driver_web.h"

#include "godot_audio.h"

#include "core/config/project_settings.h"

#include <emscripten.h>

AudioDriverWeb::AudioContext AudioDriverWeb::audio_context;

bool AudioDriverWeb::is_available() {
	return godot_audio_is_available() != 0;
}

void AudioDriverWeb::_state_change_callback(int p_state) {
	AudioDriverWeb::audio_context.state = p_state;
}

void AudioDriverWeb::_latency_update_callback(float p_latency) {
	AudioDriverWeb::audio_context.output_latency = p_latency;
}

void AudioDriverWeb::_audio_driver_process(int p_from, int p_samples) {
	const int max_samples = output_buffer.size();

	int write_pos = p_from;
	int to_write = p_samples;
	if (to_write == 0) {
		to_write = max_samples;
	}

	// High part.
	if (write_pos + to_write > max_samples) {
		const int samples_high = max_samples - write_pos;
		audio_server_process(samples_high / audio_context.output_channels, &output_buffer[write_pos]);
		to_write -= samples_high;
		write_pos = 0;
	}

	// Leftover.
	audio_server_process(to_write / audio_context.output_channels, &output_buffer[write_pos]);
}

void AudioDriverWeb::_audio_driver_capture(int p_from, int p_samples) {
	if (unlikely(!audio_context.is_input_active)) {
		return; // Input capture stopped.
	}

	if (unlikely(audio_context.input_channels == 0)) {
		audio_context.input_channels = godot_audio_get_input_channels();
	}

	const int max_samples = input_buffer.size();

	int read_pos = p_from;
	int to_read = p_samples;
	if (to_read == 0) {
		to_read = max_samples;
	}

	// High part.
	if (read_pos + to_read > max_samples) {
		const int samples_high = max_samples - read_pos;
		input_process(samples_high / audio_context.input_channels, &input_buffer[read_pos]);
		to_read -= samples_high;
		read_pos = 0;
	}

	// Leftover.
	input_process(to_read / audio_context.input_channels, &input_buffer[read_pos]);
}

Error AudioDriverWeb::init() {
	int latency = Engine::get_singleton()->get_audio_output_latency();

	if (!audio_context.inited) {
		audio_context.mix_rate = _get_configured_mix_rate();
		audio_context.output_channels = godot_audio_init(&audio_context.mix_rate, latency, &_state_change_callback, &_latency_update_callback);
		audio_context.inited = true;
	}

	buffer_frames = closest_power_of_2(latency * audio_context.mix_rate / 1000);
	Error err = create(buffer_frames, audio_context.output_channels);
	if (err != OK) {
		return err;
	}

	const size_t buffer_size = buffer_frames * (size_t)audio_context.output_channels;
	output_buffer.resize(buffer_size);
	input_buffer.resize(buffer_size);
	return OK;
}

void AudioDriverWeb::start() {
	start(output_buffer.ptr(), output_buffer.size(), input_buffer.ptr(), input_buffer.size());
}

void AudioDriverWeb::resume() {
	if (audio_context.state == 0) { // 'suspended'.
		godot_audio_resume();
	}
}

float AudioDriverWeb::get_latency() {
	return audio_context.output_latency + (float)buffer_frames / audio_context.mix_rate;
}

int AudioDriverWeb::get_mix_rate() const {
	return audio_context.mix_rate;
}

int AudioDriverWeb::get_output_channels() const {
	return audio_context.output_channels;
}

AudioDriver::BufferFormat AudioDriverWeb::get_output_buffer_format() const {
	return output_buffer_format;
}

void AudioDriverWeb::finish() {
	finish_driver();
}

Error AudioDriverWeb::input_start() {
	lock();
	audio_context.is_input_active = true;
	audio_context.input_channels = 0;
	input_buffer_init(buffer_frames);
	unlock();

	if (godot_audio_input_start()) {
		return FAILED;
	}
	return OK;
}

Error AudioDriverWeb::input_stop() {
	godot_audio_input_stop();

	lock();
	audio_context.is_input_active = false;
	unlock();
	return OK;
}

int AudioDriverWeb::get_input_channels() const {
	return audio_context.input_channels;
}

AudioDriver::BufferFormat AudioDriverWeb::get_input_buffer_format() const {
	return input_buffer_format;
}

#ifdef THREADS_ENABLED

/// AudioWorkletNode implementation (threads).
void AudioDriverWorklet::_audio_thread_func(void *p_data) {
	AudioDriverWorklet *ad = static_cast<AudioDriverWorklet *>(p_data);

	const int out_samples = ad->get_output_buffer_size();
	const int in_samples = ad->get_input_buffer_size();

	int wpos = 0;
	int to_write = out_samples;
	int rpos = 0;
	int to_read = 0;
	int32_t step = 0;

	while (!ad->quit) {
		if (to_read) {
			ad->lock();
			ad->_audio_driver_capture(rpos, to_read);
			godot_audio_worklet_state_add(ad->state, STATE_SAMPLES_IN, -to_read);
			ad->unlock();

			rpos += to_read;
			if (rpos >= in_samples) {
				rpos -= in_samples;
			}
		}

		if (to_write) {
			ad->lock();
			ad->_audio_driver_process(wpos, to_write);
			godot_audio_worklet_state_add(ad->state, STATE_SAMPLES_OUT, to_write);
			ad->unlock();

			wpos += to_write;
			if (wpos >= out_samples) {
				wpos -= out_samples;
			}
		}

		step = godot_audio_worklet_state_wait(ad->state, STATE_PROCESS, step, 1);
		to_write = out_samples - godot_audio_worklet_state_get(ad->state, STATE_SAMPLES_OUT);
		to_read = godot_audio_worklet_state_get(ad->state, STATE_SAMPLES_IN);
	}
}

Error AudioDriverWorklet::create(int &p_buffer_frames, int p_channels) {
	if (!godot_audio_has_worklet()) {
		return ERR_UNAVAILABLE;
	}
	return (Error)godot_audio_worklet_create(p_channels);
}

void AudioDriverWorklet::start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) {
	godot_audio_worklet_start(p_in_buf, p_in_buf_size, p_out_buf, p_out_buf_size, state);
	thread.start(_audio_thread_func, this);
}

void AudioDriverWorklet::lock() {
	mutex.lock();
}

void AudioDriverWorklet::unlock() {
	mutex.unlock();
}

void AudioDriverWorklet::finish_driver() {
	quit = true; // Ask thread to quit.
	thread.wait_to_finish();
}

#else // No threads.

/// AudioWorkletNode implementation (no threads).
AudioDriverWorklet *AudioDriverWorklet::singleton = nullptr;

Error AudioDriverWorklet::create(int &p_buffer_frames, int p_channels) {
	if (!godot_audio_has_worklet()) {
		return ERR_UNAVAILABLE;
	}
	return (Error)godot_audio_worklet_create(p_channels);
}

void AudioDriverWorklet::start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) {
	_audio_driver_process();
	godot_audio_worklet_start_no_threads(p_out_buf, p_out_buf_size, &_process_callback, p_in_buf, p_in_buf_size, &_capture_callback);
}

void AudioDriverWorklet::_process_callback(int p_pos, int p_samples) {
	AudioDriverWorklet *ad = AudioDriverWorklet::get_singleton();
	ad->_audio_driver_process(p_pos, p_samples);
}

void AudioDriverWorklet::_capture_callback(int p_pos, int p_samples) {
	AudioDriverWorklet *ad = AudioDriverWorklet::get_singleton();
	ad->_audio_driver_capture(p_pos, p_samples);
}

/// ScriptProcessorNode implementation.
AudioDriverScriptProcessor *AudioDriverScriptProcessor::singleton = nullptr;

void AudioDriverScriptProcessor::_process_callback() {
	AudioDriverScriptProcessor::get_singleton()->_audio_driver_capture();
	AudioDriverScriptProcessor::get_singleton()->_audio_driver_process();
}

Error AudioDriverScriptProcessor::create(int &p_buffer_frames, int p_channels) {
	if (!godot_audio_has_script_processor()) {
		return ERR_UNAVAILABLE;
	}
	return (Error)godot_audio_script_create(&p_buffer_frames, p_channels);
}

void AudioDriverScriptProcessor::start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) {
	godot_audio_script_start(p_in_buf, p_in_buf_size, p_out_buf, p_out_buf_size, &_process_callback);
}

#endif // THREADS_ENABLED
