/*************************************************************************/
/*  audio_driver_javascript.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/config/project_settings.h"

#include <emscripten.h>

AudioDriverJavaScript *AudioDriverJavaScript::singleton = nullptr;

bool AudioDriverJavaScript::is_available() {
	return godot_audio_is_available() != 0;
}

const char *AudioDriverJavaScript::get_name() const {
	return "JavaScript";
}

void AudioDriverJavaScript::_state_change_callback(int p_state) {
	singleton->state = p_state;
}

void AudioDriverJavaScript::_latency_update_callback(float p_latency) {
	singleton->output_latency = p_latency;
}

void AudioDriverJavaScript::_audio_driver_process(int p_from, int p_samples) {
	int32_t *stream_buffer = reinterpret_cast<int32_t *>(output_rb);
	const int max_samples = memarr_len(output_rb);

	int write_pos = p_from;
	int to_write = p_samples;
	if (to_write == 0) {
		to_write = max_samples;
	}
	// High part
	if (write_pos + to_write > max_samples) {
		const int samples_high = max_samples - write_pos;
		audio_server_process(samples_high / channel_count, &stream_buffer[write_pos]);
		for (int i = write_pos; i < max_samples; i++) {
			output_rb[i] = float(stream_buffer[i] >> 16) / 32768.f;
		}
		to_write -= samples_high;
		write_pos = 0;
	}
	// Leftover
	audio_server_process(to_write / channel_count, &stream_buffer[write_pos]);
	for (int i = write_pos; i < write_pos + to_write; i++) {
		output_rb[i] = float(stream_buffer[i] >> 16) / 32768.f;
	}
}

void AudioDriverJavaScript::_audio_driver_capture(int p_from, int p_samples) {
	if (get_input_buffer().size() == 0) {
		return; // Input capture stopped.
	}
	const int max_samples = memarr_len(input_rb);

	int read_pos = p_from;
	int to_read = p_samples;
	if (to_read == 0) {
		to_read = max_samples;
	}
	// High part
	if (read_pos + to_read > max_samples) {
		const int samples_high = max_samples - read_pos;
		for (int i = read_pos; i < max_samples; i++) {
			input_buffer_write(int32_t(input_rb[i] * 32768.f) * (1U << 16));
		}
		to_read -= samples_high;
		read_pos = 0;
	}
	// Leftover
	for (int i = read_pos; i < read_pos + to_read; i++) {
		input_buffer_write(int32_t(input_rb[i] * 32768.f) * (1U << 16));
	}
}

Error AudioDriverJavaScript::init() {
	mix_rate = GLOBAL_GET("audio/driver/mix_rate");
	int latency = GLOBAL_GET("audio/driver/output_latency");

	channel_count = godot_audio_init(mix_rate, latency, &_state_change_callback, &_latency_update_callback);
	buffer_length = closest_power_of_2((latency * mix_rate / 1000));
#ifndef NO_THREADS
	node = memnew(WorkletNode);
#else
	node = memnew(ScriptProcessorNode);
#endif
	buffer_length = node->create(buffer_length, channel_count);
	if (output_rb) {
		memdelete_arr(output_rb);
	}
	output_rb = memnew_arr(float, buffer_length *channel_count);
	if (!output_rb) {
		return ERR_OUT_OF_MEMORY;
	}
	if (input_rb) {
		memdelete_arr(input_rb);
	}
	input_rb = memnew_arr(float, buffer_length *channel_count);
	if (!input_rb) {
		return ERR_OUT_OF_MEMORY;
	}
	return OK;
}

void AudioDriverJavaScript::start() {
	if (node) {
		node->start(output_rb, memarr_len(output_rb), input_rb, memarr_len(input_rb));
	}
}

void AudioDriverJavaScript::resume() {
	if (state == 0) { // 'suspended'
		godot_audio_resume();
	}
}

float AudioDriverJavaScript::get_latency() {
	return output_latency + (float(buffer_length) / mix_rate);
}

int AudioDriverJavaScript::get_mix_rate() const {
	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverJavaScript::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(channel_count);
}

void AudioDriverJavaScript::lock() {
	if (node) {
		node->unlock();
	}
}

void AudioDriverJavaScript::unlock() {
	if (node) {
		node->unlock();
	}
}

void AudioDriverJavaScript::finish() {
	if (node) {
		node->finish();
		memdelete(node);
		node = nullptr;
	}
	if (output_rb) {
		memdelete_arr(output_rb);
		output_rb = nullptr;
	}
	if (input_rb) {
		memdelete_arr(input_rb);
		input_rb = nullptr;
	}
}

Error AudioDriverJavaScript::capture_start() {
	lock();
	input_buffer_init(buffer_length);
	unlock();
	if (godot_audio_capture_start()) {
		return FAILED;
	}
	return OK;
}

Error AudioDriverJavaScript::capture_stop() {
	godot_audio_capture_stop();
	lock();
	input_buffer.clear();
	unlock();
	return OK;
}

AudioDriverJavaScript::AudioDriverJavaScript() {
	singleton = this;
}

#ifdef NO_THREADS
/// ScriptProcessorNode implementation
void AudioDriverJavaScript::ScriptProcessorNode::_process_callback() {
	AudioDriverJavaScript::singleton->_audio_driver_capture();
	AudioDriverJavaScript::singleton->_audio_driver_process();
}

int AudioDriverJavaScript::ScriptProcessorNode::create(int p_buffer_samples, int p_channels) {
	return godot_audio_script_create(p_buffer_samples, p_channels);
}

void AudioDriverJavaScript::ScriptProcessorNode::start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) {
	godot_audio_script_start(p_in_buf, p_in_buf_size, p_out_buf, p_out_buf_size, &_process_callback);
}
#else
/// AudioWorkletNode implementation
void AudioDriverJavaScript::WorkletNode::_audio_thread_func(void *p_data) {
	AudioDriverJavaScript::WorkletNode *obj = static_cast<AudioDriverJavaScript::WorkletNode *>(p_data);
	AudioDriverJavaScript *driver = AudioDriverJavaScript::singleton;
	const int out_samples = memarr_len(driver->output_rb);
	const int in_samples = memarr_len(driver->input_rb);
	int wpos = 0;
	int to_write = out_samples;
	int rpos = 0;
	int to_read = 0;
	int32_t step = 0;
	while (!obj->quit) {
		if (to_read) {
			driver->lock();
			driver->_audio_driver_capture(rpos, to_read);
			godot_audio_worklet_state_add(obj->state, STATE_SAMPLES_IN, -to_read);
			driver->unlock();
			rpos += to_read;
			if (rpos >= in_samples) {
				rpos -= in_samples;
			}
		}
		if (to_write) {
			driver->lock();
			driver->_audio_driver_process(wpos, to_write);
			godot_audio_worklet_state_add(obj->state, STATE_SAMPLES_OUT, to_write);
			driver->unlock();
			wpos += to_write;
			if (wpos >= out_samples) {
				wpos -= out_samples;
			}
		}
		step = godot_audio_worklet_state_wait(obj->state, STATE_PROCESS, step, 1);
		to_write = out_samples - godot_audio_worklet_state_get(obj->state, STATE_SAMPLES_OUT);
		to_read = godot_audio_worklet_state_get(obj->state, STATE_SAMPLES_IN);
	}
}

int AudioDriverJavaScript::WorkletNode::create(int p_buffer_size, int p_channels) {
	godot_audio_worklet_create(p_channels);
	return p_buffer_size;
}

void AudioDriverJavaScript::WorkletNode::start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) {
	godot_audio_worklet_start(p_in_buf, p_in_buf_size, p_out_buf, p_out_buf_size, state);
	thread.start(_audio_thread_func, this);
}

void AudioDriverJavaScript::WorkletNode::lock() {
	mutex.lock();
}

void AudioDriverJavaScript::WorkletNode::unlock() {
	mutex.unlock();
}

void AudioDriverJavaScript::WorkletNode::finish() {
	quit = true; // Ask thread to quit.
	thread.wait_to_finish();
}
#endif
