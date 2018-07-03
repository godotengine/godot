/*************************************************************************/
/*  audio_driver_javascript.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include <emscripten.h>

AudioDriverJavaScript *AudioDriverJavaScript::singleton = NULL;

const char *AudioDriverJavaScript::get_name() const {

	return "JavaScript";
}

extern "C" EMSCRIPTEN_KEEPALIVE void audio_driver_js_mix() {

	AudioDriverJavaScript::singleton->mix_to_js();
}

void AudioDriverJavaScript::mix_to_js() {

	int channel_count = get_total_channels_by_speaker_mode(get_speaker_mode());
	int sample_count = memarr_len(internal_buffer) / channel_count;
	int32_t *stream_buffer = reinterpret_cast<int32_t *>(internal_buffer);
	audio_server_process(sample_count, stream_buffer);
	for (int i = 0; i < sample_count * channel_count; i++) {
		internal_buffer[i] = float(stream_buffer[i] >> 16) / 32768.0;
	}
}

Error AudioDriverJavaScript::init() {

	/* clang-format off */
	EM_ASM({
		_audioDriver_audioContext = new (window.AudioContext || window.webkitAudioContext);
		_audioDriver_scriptNode = null;
	});
	/* clang-format on */

	int channel_count = get_total_channels_by_speaker_mode(get_speaker_mode());
	/* clang-format off */
	int buffer_length = EM_ASM_INT({
		var CHANNEL_COUNT = $0;

		var channelCount = _audioDriver_audioContext.destination.channelCount;
		try {
			// Try letting the browser recommend a buffer length.
			_audioDriver_scriptNode = _audioDriver_audioContext.createScriptProcessor(0, 0, channelCount);
		} catch (e) {
			// ...otherwise, default to 4096.
			_audioDriver_scriptNode = _audioDriver_audioContext.createScriptProcessor(4096, 0, channelCount);
		}
		_audioDriver_scriptNode.connect(_audioDriver_audioContext.destination);

		return _audioDriver_scriptNode.bufferSize;
	}, channel_count);
	/* clang-format on */
	if (!buffer_length) {
		return FAILED;
	}

	if (!internal_buffer || memarr_len(internal_buffer) != buffer_length * channel_count) {
		if (internal_buffer)
			memdelete_arr(internal_buffer);
		internal_buffer = memnew_arr(float, buffer_length *channel_count);
	}
	return internal_buffer ? OK : ERR_OUT_OF_MEMORY;
}

void AudioDriverJavaScript::start() {

	/* clang-format off */
	EM_ASM({
		var INTERNAL_BUFFER_PTR = $0;

		var audioDriverMixFunction = cwrap('audio_driver_js_mix');
		_audioDriver_scriptNode.onaudioprocess = function(audioProcessingEvent) {
			audioDriverMixFunction();
			// The output buffer contains the samples that will be modified and played.
			var output = audioProcessingEvent.outputBuffer;
			var input = HEAPF32.subarray(
					INTERNAL_BUFFER_PTR / HEAPF32.BYTES_PER_ELEMENT,
					INTERNAL_BUFFER_PTR / HEAPF32.BYTES_PER_ELEMENT + output.length * output.numberOfChannels);

			for (var channel = 0; channel < output.numberOfChannels; channel++) {
				var outputData = output.getChannelData(channel);
				// Loop through samples.
				for (var sample = 0; sample < outputData.length; sample++) {
					// Set output equal to input.
					outputData[sample] = input[sample * output.numberOfChannels + channel];
				}
			}
		};
	}, internal_buffer);
	/* clang-format on */
}

int AudioDriverJavaScript::get_mix_rate() const {

	/* clang-format off */
	return EM_ASM_INT_V({
		return _audioDriver_audioContext.sampleRate;
	});
	/* clang-format on */
}

AudioDriver::SpeakerMode AudioDriverJavaScript::get_speaker_mode() const {

	/* clang-format off */
	return get_speaker_mode_by_total_channels(EM_ASM_INT_V({
		return _audioDriver_audioContext.destination.channelCount;
	}));
	/* clang-format on */
}

// No locking, as threads are not supported.
void AudioDriverJavaScript::lock() {
}

void AudioDriverJavaScript::unlock() {
}

void AudioDriverJavaScript::finish() {

	/* clang-format off */
	EM_ASM({
		_audioDriver_audioContext = null;
		_audioDriver_scriptNode = null;
	});
	/* clang-format on */
	memdelete_arr(internal_buffer);
	internal_buffer = NULL;
}

AudioDriverJavaScript::AudioDriverJavaScript() {

	singleton = this;
}
