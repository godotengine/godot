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

#include <emscripten.h>

AudioDriverJavaScript *AudioDriverJavaScript::singleton = nullptr;

const char *AudioDriverJavaScript::get_name() const {

	return "JavaScript";
}

extern "C" EMSCRIPTEN_KEEPALIVE void audio_driver_js_mix() {

	AudioDriverJavaScript::singleton->mix_to_js();
}

extern "C" EMSCRIPTEN_KEEPALIVE void audio_driver_process_capture(float sample) {

	AudioDriverJavaScript::singleton->process_capture(sample);
}

void AudioDriverJavaScript::mix_to_js() {

	int channel_count = get_total_channels_by_speaker_mode(get_speaker_mode());
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

	/* clang-format off */
	_driver_id = EM_ASM_INT({
		return Module.IDHandler.add({
			'context': new (window.AudioContext || window.webkitAudioContext),
			'input': null,
			'stream': null,
			'script': null
		});
	});
	/* clang-format on */

	int channel_count = get_total_channels_by_speaker_mode(get_speaker_mode());
	/* clang-format off */
	buffer_length = EM_ASM_INT({
		var ref = Module.IDHandler.get($0);
		var ctx = ref['context'];
		var CHANNEL_COUNT = $1;

		var channelCount = ctx.destination.channelCount;
		var script = null;
		try {
			// Try letting the browser recommend a buffer length.
			script = ctx.createScriptProcessor(0, 2, channelCount);
		} catch (e) {
			// ...otherwise, default to 4096.
			script = ctx.createScriptProcessor(4096, 2, channelCount);
		}
		script.connect(ctx.destination);
		ref['script'] = script;

		return script.bufferSize;
	}, _driver_id, channel_count);
	/* clang-format on */
	if (!buffer_length) {
		return FAILED;
	}

	if (!internal_buffer || (int)memarr_len(internal_buffer) != buffer_length * channel_count) {
		if (internal_buffer)
			memdelete_arr(internal_buffer);
		internal_buffer = memnew_arr(float, buffer_length *channel_count);
	}

	return internal_buffer ? OK : ERR_OUT_OF_MEMORY;
}

void AudioDriverJavaScript::start() {

	/* clang-format off */
	EM_ASM({
		const ref = Module.IDHandler.get($0);
		var INTERNAL_BUFFER_PTR = $1;

		var audioDriverMixFunction = cwrap('audio_driver_js_mix');
		var audioDriverProcessCapture = cwrap('audio_driver_process_capture', null, ['number']);
		ref['script'].onaudioprocess = function(audioProcessingEvent) {
			audioDriverMixFunction();

			var input = audioProcessingEvent.inputBuffer;
			var output = audioProcessingEvent.outputBuffer;
			var internalBuffer = HEAPF32.subarray(
					INTERNAL_BUFFER_PTR / HEAPF32.BYTES_PER_ELEMENT,
					INTERNAL_BUFFER_PTR / HEAPF32.BYTES_PER_ELEMENT + output.length * output.numberOfChannels);

			for (var channel = 0; channel < output.numberOfChannels; channel++) {
				var outputData = output.getChannelData(channel);
				// Loop through samples.
				for (var sample = 0; sample < outputData.length; sample++) {
					outputData[sample] = internalBuffer[sample * output.numberOfChannels + channel];
				}
			}

			if (ref['input']) {
				var inputDataL = input.getChannelData(0);
				var inputDataR = input.getChannelData(1);
				for (var i = 0; i < inputDataL.length; i++) {
					audioDriverProcessCapture(inputDataL[i]);
					audioDriverProcessCapture(inputDataR[i]);
				}
			}
		};
	}, _driver_id, internal_buffer);
	/* clang-format on */
}

void AudioDriverJavaScript::resume() {
	/* clang-format off */
	EM_ASM({
		const ref = Module.IDHandler.get($0);
		if (ref && ref['context'] && ref['context'].resume)
			ref['context'].resume();
	}, _driver_id);
	/* clang-format on */
}

int AudioDriverJavaScript::get_mix_rate() const {

	/* clang-format off */
	return EM_ASM_INT({
		const ref = Module.IDHandler.get($0);
		return ref && ref['context'] ? ref['context'].sampleRate : 0;
	}, _driver_id);
	/* clang-format on */
}

AudioDriver::SpeakerMode AudioDriverJavaScript::get_speaker_mode() const {

	/* clang-format off */
	return get_speaker_mode_by_total_channels(EM_ASM_INT({
		const ref = Module.IDHandler.get($0);
		return ref && ref['context'] ? ref['context'].destination.channelCount : 0;
	}, _driver_id));
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
		Module.IDHandler.remove($0);
	}, _driver_id);
	/* clang-format on */

	if (internal_buffer) {
		memdelete_arr(internal_buffer);
		internal_buffer = nullptr;
	}
	_driver_id = 0;
}

Error AudioDriverJavaScript::capture_start() {

	input_buffer_init(buffer_length);

	/* clang-format off */
	EM_ASM({
		function gotMediaInput(stream) {
			var ref = Module.IDHandler.get($0);
			ref['stream'] = stream;
			ref['input'] = ref['context'].createMediaStreamSource(stream);
			ref['input'].connect(ref['script']);
		}

		function gotMediaInputError(e) {
			out(e);
		}

		if (navigator.mediaDevices.getUserMedia) {
			navigator.mediaDevices.getUserMedia({"audio": true}).then(gotMediaInput, gotMediaInputError);
		} else {
			if (!navigator.getUserMedia)
				navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
			navigator.getUserMedia({"audio": true}, gotMediaInput, gotMediaInputError);
		}
	}, _driver_id);
	/* clang-format on */

	return OK;
}

Error AudioDriverJavaScript::capture_stop() {

	/* clang-format off */
	EM_ASM({
		var ref = Module.IDHandler.get($0);
		if (ref['stream']) {
			const tracks = ref['stream'].getTracks();
			for (var i = 0; i < tracks.length; i++) {
				tracks[i].stop();
			}
			ref['stream'] = null;
		}

		if (ref['input']) {
			ref['input'].disconnect();
			ref['input'] = null;
		}

	}, _driver_id);
	/* clang-format on */

	input_buffer.clear();

	return OK;
}

AudioDriverJavaScript::AudioDriverJavaScript() {

	_driver_id = 0;
	internal_buffer = nullptr;
	buffer_length = 0;

	singleton = this;
}
