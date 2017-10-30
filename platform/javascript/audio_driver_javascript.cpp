/*************************************************************************/
/*  audio_driver_javascript.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "audio_driver_javascript.h"

#include <emscripten.h>
#include <string.h>

#define MAX_NUMBER_INTERFACES 3
#define MAX_NUMBER_OUTPUT_DEVICES 6

/* Structure for passing information to callback function */

//AudioDriverJavaScript* AudioDriverJavaScript::s_ad=NULL;

AudioDriverJavaScript *AudioDriverJavaScript::singleton_js = NULL;
const char *AudioDriverJavaScript::get_name() const {

	return "JavaScript";
}

extern "C" {

void js_audio_driver_mix_function(int p_frames) {

	//print_line("MIXI! "+itos(p_frames));
	AudioDriverJavaScript::singleton_js->mix_to_js(p_frames);
}
}

void AudioDriverJavaScript::mix_to_js(int p_frames) {

	int todo = p_frames;
	int offset = 0;

	while (todo) {

		int tomix = MIN(todo, INTERNAL_BUFFER_SIZE);

		audio_server_process(p_frames, stream_buffer);
		for (int i = 0; i < tomix * internal_buffer_channels; i++) {
			internal_buffer[i] = float(stream_buffer[i] >> 16) * 32768.0;
		}

		/* clang-format off */
		EM_ASM_({
			var data = HEAPF32.subarray($0 / 4, $0 / 4 + $2 * 2);

			for (var channel = 0; channel < _as_output_buffer.numberOfChannels; channel++) {
				var outputData = _as_output_buffer.getChannelData(channel);
				// Loop through samples
				for (var sample = 0; sample < $2; sample++) {
					// make output equal to the same as the input
					outputData[sample + $1] = data[sample * 2 + channel];
				}
			}
		}, internal_buffer, offset, tomix);
		/* clang-format on */

		todo -= tomix;
		offset += tomix;
	}
}

Error AudioDriverJavaScript::init() {

	return OK;
}

void AudioDriverJavaScript::start() {

	internal_buffer_channels = 2;
	internal_buffer = memnew_arr(float, INTERNAL_BUFFER_SIZE *internal_buffer_channels);
	stream_buffer = memnew_arr(int32_t, INTERNAL_BUFFER_SIZE * 4); //max 4 channels

	/* clang-format off */
	EM_ASM(
			_as_audioctx = new (window.AudioContext || window.webkitAudioContext)();

			audio_server_mix_function = Module.cwrap('js_audio_driver_mix_function', 'void', ['number']);
		);

	int buffer_latency = 16384;
	EM_ASM_( {
		_as_script_node = _as_audioctx.createScriptProcessor($0, 0, 2);
		_as_script_node.connect(_as_audioctx.destination);
		console.log(_as_script_node.bufferSize);

		_as_script_node.onaudioprocess = function(audioProcessingEvent) {
		// The output buffer contains the samples that will be modified and played
			_as_output_buffer = audioProcessingEvent.outputBuffer;
			audio_server_mix_function(_as_output_buffer.getChannelData(0).length);
		}
	}, buffer_latency);
	/* clang-format on */
}

int AudioDriverJavaScript::get_mix_rate() const {

	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverJavaScript::get_speaker_mode() const {

	return SPEAKER_MODE_STEREO;
}

void AudioDriverJavaScript::lock() {

	/*no locking, as threads are not supported
	if (active && mutex)
		mutex->lock();
	*/
}

void AudioDriverJavaScript::unlock() {

	/*no locking, as threads are not supported
	if (active && mutex)
		mutex->unlock();
	*/
}

void AudioDriverJavaScript::finish() {
}

AudioDriverJavaScript::AudioDriverJavaScript() {

	mix_rate = 44100;
	singleton_js = this;
}
