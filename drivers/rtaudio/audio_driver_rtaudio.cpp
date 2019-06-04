/*************************************************************************/
/*  audio_driver_rtaudio.cpp                                             */
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
#include "audio_driver_rtaudio.h"

#include "globals.h"
#include "os/os.h"

#ifdef RTAUDIO_ENABLED

const char *AudioDriverRtAudio::get_name() const {

#ifdef OSX_ENABLED
	return "RtAudio-OSX";
#elif defined(UNIX_ENABLED)
	return "RtAudio-ALSA";
#elif defined(WINDOWS_ENABLED)
	return "RtAudio-DirectSound";
#else
	return "RtAudio-None";
#endif
}

int AudioDriverRtAudio::callback(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames, double streamTime, RtAudioStreamStatus status, void *userData) {

	if (status) {
		if (status & RTAUDIO_INPUT_OVERFLOW) {
			WARN_PRINT("RtAudio input overflow!");
		}
		if (status & RTAUDIO_OUTPUT_UNDERFLOW) {
			WARN_PRINT("RtAudio output underflow!");
		}
	}
	int32_t *buffer = (int32_t *)outputBuffer;

	AudioDriverRtAudio *self = (AudioDriverRtAudio *)userData;

	if (self->mutex->try_lock() != OK) {
		// what should i do..
		for (unsigned int i = 0; i < nBufferFrames; i++)
			buffer[i] = 0;

		return 0;
	}

	self->audio_server_process(nBufferFrames, buffer);

	self->mutex->unlock();

	return 0;
}

Error AudioDriverRtAudio::init() {

	active = false;
	mutex = NULL;
	dac = memnew(RtAudio);

	ERR_EXPLAIN("Cannot initialize RtAudio audio driver: No devices present.")
	ERR_FAIL_COND_V(dac->getDeviceCount() < 1, ERR_UNAVAILABLE);

	String channels = GLOBAL_DEF("audio/output", "stereo");

	if (channels == "5.1")
		output_format = OUTPUT_5_1;
	else if (channels == "quad")
		output_format = OUTPUT_QUAD;
	else if (channels == "mono")
		output_format = OUTPUT_MONO;
	else
		output_format = OUTPUT_STEREO;

	RtAudio::StreamParameters parameters;
	parameters.deviceId = dac->getDefaultOutputDevice();
	RtAudio::StreamOptions options;

	// set the desired numberOfBuffers
	unsigned int target_number_of_buffers = 4;
	options.numberOfBuffers = target_number_of_buffers;

	//	options.
	//	RtAudioStreamFlags flags;      /*!< A bit-mask of stream flags (RTAUDIO_NONINTERLEAVED, RTAUDIO_MINIMIZE_LATENCY, RTAUDIO_HOG_DEVICE). *///
	//	unsigned int numberOfBuffers;  /*!< Number of stream buffers. */
	//	std::string streamName;        /*!< A stream name (currently used only in Jack). */
	//	int priority;                  /*!< Scheduling priority of callback thread (only used with flag RTAUDIO_SCHEDULE_REALTIME). */

	parameters.firstChannel = 0;
	mix_rate = GLOBAL_DEF("audio/mix_rate", 44100);

	int latency = GLOBAL_DEF("audio/output_latency", 25);
	// calculate desired buffer_size, taking the desired numberOfBuffers into account (latency depends on numberOfBuffers*buffer_size)
	unsigned int buffer_size = next_power_of_2(latency * mix_rate / 1000 / target_number_of_buffers);

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("audio buffer size: " + itos(buffer_size));
	}

	short int tries = 2;

	while (true) {
		while (true) {
			switch (output_format) {
				case OUTPUT_MONO: parameters.nChannels = 1; break;
				case OUTPUT_STEREO: parameters.nChannels = 2; break;
				case OUTPUT_QUAD: parameters.nChannels = 4; break;
				case OUTPUT_5_1: parameters.nChannels = 6; break;
			};

			try {
				dac->openStream(&parameters, NULL, RTAUDIO_SINT32, mix_rate, &buffer_size, &callback, this, &options);
				mutex = Mutex::create(true);
				active = true;

				break;
			} catch (RtAudioError &e) {
				// try with less channels
				ERR_PRINT("Unable to open audio, retrying with fewer channels..");

				switch (output_format) {
					case OUTPUT_MONO:
						ERR_EXPLAIN("Unable to open audio.");
						ERR_FAIL_V(ERR_UNAVAILABLE);
						break;
					case OUTPUT_STEREO: output_format = OUTPUT_MONO; break;
					case OUTPUT_QUAD: output_format = OUTPUT_STEREO; break;
					case OUTPUT_5_1: output_format = OUTPUT_QUAD; break;
				};
			}
		}

		// compare actual numberOfBuffers with the desired one. If not equal, close and reopen the stream with adjusted buffer size, so the desired output_latency is still correct
		if (target_number_of_buffers != options.numberOfBuffers) {
			if (tries <= 0) {
				ERR_EXPLAIN("RtAudio: Unable to set correct number of buffers.");
				ERR_FAIL_V(ERR_UNAVAILABLE);
				break;
			}

			try {
				dac->closeStream();
			} catch (RtAudioError &e) {
				ERR_PRINT(e.what());
				ERR_FAIL_V(ERR_UNAVAILABLE);
				break;
			}
			if (OS::get_singleton()->is_stdout_verbose())
				print_line("RtAudio: Desired number of buffers (" + itos(target_number_of_buffers) + ") not available. Using " + itos(options.numberOfBuffers) + " instead. Reopening stream with adjusted buffer_size.");

			// new buffer size dependent on the ratio between set and actual numberOfBuffers
			buffer_size = buffer_size / (options.numberOfBuffers / target_number_of_buffers);
			target_number_of_buffers = options.numberOfBuffers;
			tries--;
		} else {
			break;
		}
	}

	return OK;
}

int AudioDriverRtAudio::get_mix_rate() const {

	return mix_rate;
}

AudioDriverSW::OutputFormat AudioDriverRtAudio::get_output_format() const {

	return output_format;
}

void AudioDriverRtAudio::start() {

	if (active)
		dac->startStream();
}

void AudioDriverRtAudio::lock() {

	if (mutex)
		mutex->lock();
}

void AudioDriverRtAudio::unlock() {

	if (mutex)
		mutex->unlock();
}

void AudioDriverRtAudio::finish() {

	if (active && dac->isStreamOpen())
		dac->closeStream();
	if (mutex)
		memdelete(mutex);
	if (dac)
		memdelete(dac);
}

AudioDriverRtAudio::AudioDriverRtAudio() {

	active = false;
	mutex = NULL;
	mix_rate = 44100;
	output_format = OUTPUT_STEREO;
}

#endif
