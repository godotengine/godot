/*************************************************************************/
/*  library_godot_audio.js                                               */
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
var GodotAudio = {

	$GodotAudio: {

		ctx: null,
		input: null,
		script: null,
	},

	godot_audio_is_available__proxy: 'sync',
	godot_audio_is_available: function () {
		if (!(window.AudioContext || window.webkitAudioContext)) {
			return 0;
		}
		return 1;
	},

	godot_audio_init: function(mix_rate, latency) {
		GodotAudio.ctx = new (window.AudioContext || window.webkitAudioContext)({
			sampleRate: mix_rate,
			// latencyHint: latency / 1000 // Do not specify, leave 'interactive' for good performance.
		});
		return GodotAudio.ctx.destination.channelCount;
	},

	godot_audio_create_processor: function(buffer_length, channel_count) {
		GodotAudio.script = GodotAudio.ctx.createScriptProcessor(buffer_length, 2, channel_count);
		GodotAudio.script.connect(GodotAudio.ctx.destination);
		return GodotAudio.script.bufferSize;
	},

	godot_audio_start: function(buffer_ptr) {
		var audioDriverProcessStart = cwrap('audio_driver_process_start');
		var audioDriverProcessEnd = cwrap('audio_driver_process_end');
		var audioDriverProcessCapture = cwrap('audio_driver_process_capture', null, ['number']);
		GodotAudio.script.onaudioprocess = function(audioProcessingEvent) {
			audioDriverProcessStart();

			var input = audioProcessingEvent.inputBuffer;
			var output = audioProcessingEvent.outputBuffer;
			var internalBuffer = HEAPF32.subarray(
					buffer_ptr / HEAPF32.BYTES_PER_ELEMENT,
					buffer_ptr / HEAPF32.BYTES_PER_ELEMENT + output.length * output.numberOfChannels);
			for (var channel = 0; channel < output.numberOfChannels; channel++) {
				var outputData = output.getChannelData(channel);
				// Loop through samples.
				for (var sample = 0; sample < outputData.length; sample++) {
					outputData[sample] = internalBuffer[sample * output.numberOfChannels + channel];
				}
			}

			if (GodotAudio.input) {
				var inputDataL = input.getChannelData(0);
				var inputDataR = input.getChannelData(1);
				for (var i = 0; i < inputDataL.length; i++) {
					audioDriverProcessCapture(inputDataL[i]);
					audioDriverProcessCapture(inputDataR[i]);
				}
			}
			audioDriverProcessEnd();
		};
	},

	godot_audio_resume: function() {
		if (GodotAudio.ctx && GodotAudio.ctx.state != 'running') {
			GodotAudio.ctx.resume();
		}
	},

	godot_audio_finish_async: function() {
		Module.async_finish.push(new Promise(function(accept, reject) {
			if (!GodotAudio.ctx) {
				setTimeout(accept, 0);
			} else {
				if (GodotAudio.script) {
					GodotAudio.script.disconnect();
					GodotAudio.script = null;
				}
				if (GodotAudio.input) {
					GodotAudio.input.disconnect();
					GodotAudio.input = null;
				}
				GodotAudio.ctx.close().then(function() {
					accept();
				}).catch(function(e) {
					accept();
				});
				GodotAudio.ctx = null;
			}
		}));
	},

	godot_audio_get_latency__proxy: 'sync',
	godot_audio_get_latency: function() {
		var latency = 0;
		if (GodotAudio.ctx) {
			if (GodotAudio.ctx.baseLatency) {
				latency += GodotAudio.ctx.baseLatency;
			}
			if (GodotAudio.ctx.outputLatency) {
				latency += GodotAudio.ctx.outputLatency;
			}
		}
		return latency;
	},

	godot_audio_capture_start__proxy: 'sync',
	godot_audio_capture_start: function() {
		if (GodotAudio.input) {
			return; // Already started.
		}
		function gotMediaInput(stream) {
			GodotAudio.input = GodotAudio.ctx.createMediaStreamSource(stream);
			GodotAudio.input.connect(GodotAudio.script);
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
	},

	godot_audio_capture_stop__proxy: 'sync',
	godot_audio_capture_stop: function() {
		if (GodotAudio.input) {
			const tracks = GodotAudio.input.mediaStream.getTracks();
			for (var i = 0; i < tracks.length; i++) {
				tracks[i].stop();
			}
			GodotAudio.input.disconnect();
			GodotAudio.input = null;
		}
	},
};

autoAddDeps(GodotAudio, "$GodotAudio");
mergeInto(LibraryManager.library, GodotAudio);
