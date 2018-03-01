/*************************************************************************/
/*  audio_driver_pulseaudio.cpp                                          */
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

#include "audio_driver_pulseaudio.h"

#ifdef PULSEAUDIO_ENABLED

#include <pulse/pulseaudio.h>

#include "os/os.h"
#include "project_settings.h"

void pa_state_cb(pa_context *c, void *userdata) {
	pa_context_state_t state;
	int *pa_ready = (int *)userdata;

	state = pa_context_get_state(c);
	switch (state) {
		case PA_CONTEXT_FAILED:
		case PA_CONTEXT_TERMINATED:
			*pa_ready = 2;
			break;

		case PA_CONTEXT_READY:
			*pa_ready = 1;
			break;
	}
}

void sink_info_cb(pa_context *c, const pa_sink_info *l, int eol, void *userdata) {
	unsigned int *channels = (unsigned int *)userdata;

	// If eol is set to a positive number, you're at the end of the list
	if (eol > 0) {
		return;
	}

	*channels = l->channel_map.channels;
}

void server_info_cb(pa_context *c, const pa_server_info *i, void *userdata) {
	char *default_output = (char *)userdata;

	strncpy(default_output, i->default_sink_name, 1024);
}

static unsigned int detect_channels() {

	pa_mainloop *pa_ml;
	pa_mainloop_api *pa_mlapi;
	pa_operation *pa_op;
	pa_context *pa_ctx;

	int state = 0;
	int pa_ready = 0;

	char default_output[1024];
	unsigned int channels = 2;

	pa_ml = pa_mainloop_new();
	pa_mlapi = pa_mainloop_get_api(pa_ml);
	pa_ctx = pa_context_new(pa_mlapi, "Godot");

	int ret = pa_context_connect(pa_ctx, NULL, PA_CONTEXT_NOFLAGS, NULL);
	if (ret < 0) {
		pa_context_unref(pa_ctx);
		pa_mainloop_free(pa_ml);

		return 2;
	}

	pa_context_set_state_callback(pa_ctx, pa_state_cb, &pa_ready);

	// Wait until the pa server is ready
	while (pa_ready == 0) {
		pa_mainloop_iterate(pa_ml, 1, NULL);
	}

	// Check if there was an error connecting to the pa server
	if (pa_ready == 2) {
		pa_context_disconnect(pa_ctx);
		pa_context_unref(pa_ctx);
		pa_mainloop_free(pa_ml);

		return 2;
	}

	// Get the default output device name
	pa_op = pa_context_get_server_info(pa_ctx, &server_info_cb, (void *)default_output);
	if (pa_op) {
		while (pa_operation_get_state(pa_op) == PA_OPERATION_RUNNING) {
			ret = pa_mainloop_iterate(pa_ml, 1, NULL);
			if (ret < 0) {
				ERR_PRINT("pa_mainloop_iterate error");
			}
		}

		pa_operation_unref(pa_op);

		// Now using the device name get the amount of channels
		pa_op = pa_context_get_sink_info_by_name(pa_ctx, default_output, &sink_info_cb, (void *)&channels);
		if (pa_op) {
			while (pa_operation_get_state(pa_op) == PA_OPERATION_RUNNING) {
				ret = pa_mainloop_iterate(pa_ml, 1, NULL);
				if (ret < 0) {
					ERR_PRINT("pa_mainloop_iterate error");
				}
			}

			pa_operation_unref(pa_op);
		} else {
			ERR_PRINT("pa_context_get_sink_info_by_name error");
		}
	} else {
		ERR_PRINT("pa_context_get_server_info error");
	}

	pa_context_disconnect(pa_ctx);
	pa_context_unref(pa_ctx);
	pa_mainloop_free(pa_ml);

	return channels;
}

Error AudioDriverPulseAudio::init() {

	active = false;
	thread_exited = false;
	exit_thread = false;

	mix_rate = GLOBAL_DEF("audio/mix_rate", DEFAULT_MIX_RATE);

	// Detect the amount of channels PulseAudio is using
	// Note: If using an even amount of channels (2, 4, etc) channels and pa_channels will be equal,
	// if not then pa_channels will have the real amount of channels PulseAudio is using and channels
	// will have the amount of channels Godot is using (in this case it's pa_channels + 1)
	pa_channels = detect_channels();
	switch (pa_channels) {
		case 1: // Mono
		case 3: // Surround 2.1
		case 5: // Surround 5.0
		case 7: // Surround 7.0
			channels = pa_channels + 1;
			break;

		case 2: // Stereo
		case 4: // Surround 4.0
		case 6: // Surround 5.1
		case 8: // Surround 7.1
			channels = pa_channels;
			break;

		default:
			ERR_PRINTS("PulseAudio: Unsupported number of channels: " + itos(pa_channels));
			ERR_FAIL_V(ERR_CANT_OPEN);
			break;
	}

	pa_sample_spec spec;
	spec.format = PA_SAMPLE_S16LE;
	spec.channels = pa_channels;
	spec.rate = mix_rate;

	int latency = GLOBAL_DEF("audio/output_latency", DEFAULT_OUTPUT_LATENCY);
	buffer_frames = closest_power_of_2(latency * mix_rate / 1000);
	pa_buffer_size = buffer_frames * pa_channels;

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("PulseAudio: detected " + itos(pa_channels) + " channels");
		print_line("PulseAudio: audio buffer frames: " + itos(buffer_frames) + " calculated latency: " + itos(buffer_frames * 1000 / mix_rate) + "ms");
	}

	pa_buffer_attr attr;
	// set to appropriate buffer length (in bytes) from global settings
	attr.tlength = pa_buffer_size * sizeof(int16_t);
	// set them to be automatically chosen
	attr.prebuf = (uint32_t)-1;
	attr.maxlength = (uint32_t)-1;
	attr.minreq = (uint32_t)-1;

	int error_code;
	pulse = pa_simple_new(NULL, // default server
			"Godot", // application name
			PA_STREAM_PLAYBACK,
			NULL, // default device
			"Sound", // stream description
			&spec,
			NULL, // use default channel map
			&attr, // use buffering attributes from above
			&error_code);

	if (pulse == NULL) {
		fprintf(stderr, "PulseAudio ERR: %s\n", pa_strerror(error_code));
		ERR_FAIL_COND_V(pulse == NULL, ERR_CANT_OPEN);
	}

	samples_in.resize(buffer_frames * channels);
	samples_out.resize(pa_buffer_size);

	mutex = Mutex::create();
	thread = Thread::create(AudioDriverPulseAudio::thread_func, this);

	return OK;
}

float AudioDriverPulseAudio::get_latency() {

	if (latency == 0) { //only do this once since it's approximate anyway
		int error_code;
		pa_usec_t palat = pa_simple_get_latency(pulse, &error_code);
		latency = double(palat) / 1000000.0;
	}

	return latency;
}

void AudioDriverPulseAudio::thread_func(void *p_udata) {

	AudioDriverPulseAudio *ad = (AudioDriverPulseAudio *)p_udata;

	while (!ad->exit_thread) {
		if (!ad->active) {
			for (unsigned int i = 0; i < ad->pa_buffer_size; i++) {
				ad->samples_out[i] = 0;
			}

		} else {
			ad->lock();

			ad->audio_server_process(ad->buffer_frames, ad->samples_in.ptrw());

			ad->unlock();

			if (ad->channels == ad->pa_channels) {
				for (unsigned int i = 0; i < ad->pa_buffer_size; i++) {
					ad->samples_out[i] = ad->samples_in[i] >> 16;
				}
			} else {
				// Uneven amount of channels
				unsigned int in_idx = 0;
				unsigned int out_idx = 0;

				for (unsigned int i = 0; i < ad->buffer_frames; i++) {
					for (unsigned int j = 0; j < ad->pa_channels - 1; j++) {
						ad->samples_out[out_idx++] = ad->samples_in[in_idx++] >> 16;
					}
					uint32_t l = ad->samples_in[in_idx++];
					uint32_t r = ad->samples_in[in_idx++];
					ad->samples_out[out_idx++] = (l >> 1 + r >> 1) >> 16;
				}
			}
		}

		// pa_simple_write always consumes the entire buffer

		int error_code;
		int byte_size = ad->pa_buffer_size * sizeof(int16_t);
		if (pa_simple_write(ad->pulse, ad->samples_out.ptr(), byte_size, &error_code) < 0) {
			// can't recover here
			fprintf(stderr, "PulseAudio failed and can't recover: %s\n", pa_strerror(error_code));
			ad->active = false;
			ad->exit_thread = true;
			break;
		}
	}

	ad->thread_exited = true;
}

void AudioDriverPulseAudio::start() {

	active = true;
}

int AudioDriverPulseAudio::get_mix_rate() const {

	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverPulseAudio::get_speaker_mode() const {

	return get_speaker_mode_by_total_channels(channels);
}

void AudioDriverPulseAudio::lock() {

	if (!thread || !mutex)
		return;
	mutex->lock();
}

void AudioDriverPulseAudio::unlock() {

	if (!thread || !mutex)
		return;
	mutex->unlock();
}

void AudioDriverPulseAudio::finish() {

	if (!thread)
		return;

	exit_thread = true;
	Thread::wait_to_finish(thread);

	if (pulse) {
		pa_simple_free(pulse);
		pulse = NULL;
	}

	memdelete(thread);
	if (mutex) {
		memdelete(mutex);
		mutex = NULL;
	}

	thread = NULL;
}

AudioDriverPulseAudio::AudioDriverPulseAudio() {

	mutex = NULL;
	thread = NULL;
	pulse = NULL;

	samples_in.clear();
	samples_out.clear();

	mix_rate = 0;
	buffer_frames = 0;
	pa_buffer_size = 0;
	channels = 0;
	pa_channels = 0;

	active = false;
	thread_exited = false;
	exit_thread = false;

	latency = 0;
}

AudioDriverPulseAudio::~AudioDriverPulseAudio() {
}

#endif
