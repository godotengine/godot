/**************************************************************************/
/*  audio_driver_openharmony.cpp                                          */
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

#include "audio_driver_openharmony.h"

AudioDriverOpenHarmony::AudioDriverOpenHarmony() {
}

OH_AudioData_Callback_Result AudioDriverOpenHarmony::_buffer_callback(OH_AudioRenderer *renderer, void *userData, void *audioData, int32_t audioDataSize) {
	bool mix = true;

	if (pause) {
		mix = false;
	} else {
		mix = mutex.try_lock();
	}

	if (mix) {
		audio_server_process(buffer_size, mixdown_buffer);
	} else {
		int32_t *src_buff = mixdown_buffer;
		for (unsigned int i = 0; i < buffer_size * 2; i++) {
			src_buff[i] = 0;
		}
	}

	if (mix) {
		mutex.unlock();
	}

	const int32_t *src_buff = mixdown_buffer;
	int16_t *ptr = static_cast<int16_t *>(audioData);

	for (unsigned int i = 0; i < buffer_size * 2; i++) {
		ptr[i] = src_buff[i] >> 16;
	}
	return AUDIO_DATA_CALLBACK_RESULT_VALID;
}

OH_AudioData_Callback_Result AudioDriverOpenHarmony::_buffer_callbacks(OH_AudioRenderer *renderer, void *userData, void *audioData, int32_t audioDataSize) {
	AudioDriverOpenHarmony *ad = static_cast<AudioDriverOpenHarmony *>(userData);
	return ad->_buffer_callback(renderer, userData, audioData, audioDataSize);
}

int32_t AudioDriverOpenHarmony::_capturer_read_data(OH_AudioCapturer *capturer, void *buffer, int32_t length) {
	int16_t *input_data = static_cast<int16_t *>(buffer);
	int32_t samples = length / sizeof(int16_t);

	for (int32_t i = 0; i < samples; i += 2) {
		int32_t left_sample = input_data[i] << 16;
		int32_t right_sample = (i + 1 < samples) ? input_data[i + 1] << 16 : left_sample;

		input_buffer_write(left_sample);
		input_buffer_write(right_sample);
	}

	return length;
}

int32_t AudioDriverOpenHarmony::_on_capturer_read_data(OH_AudioCapturer *capturer, void *userData, void *buffer, int32_t length) {
	AudioDriverOpenHarmony *ad = static_cast<AudioDriverOpenHarmony *>(userData);
	return ad->_capturer_read_data(capturer, buffer, length);
}

int32_t AudioDriverOpenHarmony::_on_capturer_error(OH_AudioCapturer *capturer, void *userData, OH_AudioStream_Result error) {
	return 0;
}

int32_t AudioDriverOpenHarmony::_on_capturer_interrupt_event(OH_AudioCapturer *capturer, void *userData, OH_AudioInterrupt_ForceType type, OH_AudioInterrupt_Hint hint) {
	return 0;
}

int32_t AudioDriverOpenHarmony::_on_capturer_stream_event(OH_AudioCapturer *capturer, void *userData, OH_AudioStream_Event event) {
	return 0;
}

Error AudioDriverOpenHarmony::init() {
	return OK;
}

void AudioDriverOpenHarmony::start() {
	if (active) {
		return;
	}
	active = false;

	if (!mixdown_buffer) {
		buffer_size = 960;
		mixdown_buffer = memnew_arr(int32_t, buffer_size * 2);
	}

	if (!audio_stream_builder) {
		OH_AudioStreamBuilder_Create(&audio_stream_builder, AUDIOSTREAM_TYPE_RENDERER);
		OH_AudioStreamBuilder_SetSamplingRate(audio_stream_builder, get_mix_rate());
		OH_AudioStreamBuilder_SetChannelCount(audio_stream_builder, 2);
		OH_AudioStreamBuilder_SetSampleFormat(audio_stream_builder, AUDIOSTREAM_SAMPLE_S16LE);
		OH_AudioStreamBuilder_SetEncodingType(audio_stream_builder, AUDIOSTREAM_ENCODING_TYPE_RAW);
		OH_AudioStreamBuilder_SetRendererInfo(audio_stream_builder, AUDIOSTREAM_USAGE_MUSIC);
		OH_AudioStreamBuilder_SetRendererWriteDataCallback(audio_stream_builder, _buffer_callbacks, this);
		OH_AudioStreamBuilder_SetLatencyMode(audio_stream_builder, AUDIOSTREAM_LATENCY_MODE_FAST);
		OH_AudioStreamBuilder_SetFrameSizeInCallback(audio_stream_builder, buffer_size);
	}

	if (!audio_renderer) {
		OH_AudioStreamBuilder_GenerateRenderer(audio_stream_builder, &audio_renderer);
		OH_AudioRenderer_Start(audio_renderer);
	}

	active = true;
}

int AudioDriverOpenHarmony::get_mix_rate() const {
	return 48000;
}

AudioDriver::SpeakerMode AudioDriverOpenHarmony::get_speaker_mode() const {
	return SPEAKER_MODE_STEREO;
}

void AudioDriverOpenHarmony::lock() {
	if (active) {
		mutex.lock();
	}
}

void AudioDriverOpenHarmony::unlock() {
	if (active) {
		mutex.unlock();
	}
}

void AudioDriverOpenHarmony::finish() {
	if (audio_renderer) {
		OH_AudioRenderer_Stop(audio_renderer);
		OH_AudioRenderer_Flush(audio_renderer);
		OH_AudioRenderer_Release(audio_renderer);
		audio_renderer = nullptr;
	}

	if (audio_stream_builder) {
		OH_AudioStreamBuilder_Destroy(audio_stream_builder);
		audio_stream_builder = nullptr;
	}

	if (!mixdown_buffer) {
		buffer_size = 1024;
		memdelete_arr(mixdown_buffer);
		mixdown_buffer = nullptr;
	}
	active = false;
}

Error AudioDriverOpenHarmony::input_start() {
	if (!OS::get_singleton()->request_permission("ohos.permission.MICROPHONE")) {
		ERR_FAIL_V_MSG(FAILED, "Microphone permission not granted.");
	}
	if (!audio_stream_capture_builder) {
		OH_AudioStreamBuilder_Create(&audio_stream_capture_builder, AUDIOSTREAM_TYPE_CAPTURER);
		OH_AudioStreamBuilder_SetSamplingRate(audio_stream_capture_builder, get_mix_rate());
		OH_AudioStreamBuilder_SetChannelCount(audio_stream_capture_builder, 2);
		OH_AudioStreamBuilder_SetSampleFormat(audio_stream_capture_builder, AUDIOSTREAM_SAMPLE_S16LE);
		OH_AudioStreamBuilder_SetEncodingType(audio_stream_capture_builder, AUDIOSTREAM_ENCODING_TYPE_RAW);
		OH_AudioStreamBuilder_SetCapturerInfo(audio_stream_capture_builder, AUDIOSTREAM_SOURCE_TYPE_MIC);
		OH_AudioStreamBuilder_SetLatencyMode(audio_stream_capture_builder, AUDIOSTREAM_LATENCY_MODE_FAST);

		OH_AudioCapturer_Callbacks callbacks;
		callbacks.OH_AudioCapturer_OnReadData = _on_capturer_read_data;
		callbacks.OH_AudioCapturer_OnError = _on_capturer_error;
		callbacks.OH_AudioCapturer_OnInterruptEvent = _on_capturer_interrupt_event;
		callbacks.OH_AudioCapturer_OnStreamEvent = _on_capturer_stream_event;

		OH_AudioStreamBuilder_SetCapturerCallback(audio_stream_capture_builder, callbacks, this);
	}

	if (!audio_capturer) {
		OH_AudioStream_Result r = OH_AudioStreamBuilder_GenerateCapturer(audio_stream_capture_builder, &audio_capturer);
		ERR_FAIL_COND_V_MSG(r != AUDIOSTREAM_SUCCESS, FAILED, vformat("Failed to generate capturer: %d.", r));

		input_buffer_init(2048);
		r = OH_AudioCapturer_Start(audio_capturer);
		ERR_FAIL_COND_V_MSG(r != AUDIOSTREAM_SUCCESS, FAILED, vformat("Failed to start capturer: %d.", r));
	}
	return OK;
}

Error AudioDriverOpenHarmony::input_stop() {
	// release audio_capturer
	if (audio_capturer) {
		OH_AudioCapturer_Stop(audio_capturer);
		OH_AudioCapturer_Flush(audio_capturer);
		OH_AudioCapturer_Release(audio_capturer);
		audio_capturer = nullptr;
	}

	if (audio_stream_capture_builder) {
		OH_AudioStreamBuilder_Destroy(audio_stream_capture_builder);
		audio_stream_capture_builder = nullptr;
	}
	return OK;
}

void AudioDriverOpenHarmony::set_pause(bool p_pause) {
	pause = p_pause;

	if (active && audio_renderer) {
		if (pause) {
			OH_AudioRenderer_Pause(audio_renderer);
		} else {
			OH_AudioRenderer_Start(audio_renderer);
		}
	}
}
