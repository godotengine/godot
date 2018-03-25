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

void AudioDriverPulseAudio::pa_state_cb(pa_context *c, void *userdata) {
	AudioDriverPulseAudio *ad = (AudioDriverPulseAudio *)userdata;

	switch (pa_context_get_state(c)) {
		case PA_CONTEXT_TERMINATED:
		case PA_CONTEXT_FAILED:
			ad->pa_ready = -1;
			break;

		case PA_CONTEXT_READY:
			ad->pa_ready = 1;
			break;
	}
}

void AudioDriverPulseAudio::pa_sink_info_cb(pa_context *c, const pa_sink_info *l, int eol, void *userdata) {
	AudioDriverPulseAudio *ad = (AudioDriverPulseAudio *)userdata;

	// If eol is set to a positive number, you're at the end of the list
	if (eol > 0) {
		return;
	}

	ad->pa_channels = l->channel_map.channels;
	ad->pa_status++;
}

void AudioDriverPulseAudio::pa_server_info_cb(pa_context *c, const pa_server_info *i, void *userdata) {
	AudioDriverPulseAudio *ad = (AudioDriverPulseAudio *)userdata;

	ad->default_device = i->default_sink_name;
	ad->pa_status++;
}

void AudioDriverPulseAudio::detect_channels() {

	pa_channels = 2;

	if (device_name == "Default") {
		// Get the default output device name
		pa_status = 0;
		pa_operation *pa_op = pa_context_get_server_info(pa_ctx, &AudioDriverPulseAudio::pa_server_info_cb, (void *)this);
		if (pa_op) {
			while (pa_status == 0) {
				int ret = pa_mainloop_iterate(pa_ml, 1, NULL);
				if (ret < 0) {
					ERR_PRINT("pa_mainloop_iterate error");
				}
			}

			pa_operation_unref(pa_op);
		} else {
			ERR_PRINT("pa_context_get_server_info error");
		}
	}

	char device[1024];
	if (device_name == "Default") {
		strcpy(device, default_device.utf8().get_data());
	} else {
		strcpy(device, device_name.utf8().get_data());
	}

	// Now using the device name get the amount of channels
	pa_status = 0;
	pa_operation *pa_op = pa_context_get_sink_info_by_name(pa_ctx, device, &AudioDriverPulseAudio::pa_sink_info_cb, (void *)this);
	if (pa_op) {
		while (pa_status == 0) {
			int ret = pa_mainloop_iterate(pa_ml, 1, NULL);
			if (ret < 0) {
				ERR_PRINT("pa_mainloop_iterate error");
			}
		}

		pa_operation_unref(pa_op);
	} else {
		ERR_PRINT("pa_context_get_sink_info_by_name error");
	}
}

Error AudioDriverPulseAudio::init_device() {

	// If there is a specified device check that it is really present
	if (device_name != "Default") {
		Array list = get_device_list();
		if (list.find(device_name) == -1) {
			device_name = "Default";
			new_device = "Default";
		}
	}

	// Detect the amount of channels PulseAudio is using
	// Note: If using an even amount of channels (2, 4, etc) channels and pa_channels will be equal,
	// if not then pa_channels will have the real amount of channels PulseAudio is using and channels
	// will have the amount of channels Godot is using (in this case it's pa_channels + 1)
	detect_channels();
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

	pa_str = pa_stream_new(pa_ctx, "Sound", &spec, NULL);
	ERR_FAIL_COND_V(pa_ctx == NULL, ERR_CANT_OPEN);

	const char *dev = device_name == "Default" ? NULL : device_name.utf8().get_data();
	pa_stream_flags flags = pa_stream_flags(PA_STREAM_INTERPOLATE_TIMING | PA_STREAM_ADJUST_LATENCY | PA_STREAM_AUTO_TIMING_UPDATE);
	int error_code = pa_stream_connect_playback(pa_str, dev, &attr, flags, NULL, NULL);
	ERR_FAIL_COND_V(error_code < 0, ERR_CANT_OPEN);

	samples_in.resize(buffer_frames * channels);
	samples_out.resize(pa_buffer_size);

	return OK;
}

Error AudioDriverPulseAudio::init() {

	active = false;
	thread_exited = false;
	exit_thread = false;

	mix_rate = GLOBAL_DEF("audio/mix_rate", DEFAULT_MIX_RATE);

	pa_ml = pa_mainloop_new();
	ERR_FAIL_COND_V(pa_ml == NULL, ERR_CANT_OPEN);

	pa_ctx = pa_context_new(pa_mainloop_get_api(pa_ml), "Godot");
	ERR_FAIL_COND_V(pa_ctx == NULL, ERR_CANT_OPEN);

	pa_ready = 0;
	pa_context_set_state_callback(pa_ctx, pa_state_cb, (void *)this);

	int ret = pa_context_connect(pa_ctx, NULL, PA_CONTEXT_NOFLAGS, NULL);
	if (ret < 0) {
		if (pa_ctx) {
			pa_context_unref(pa_ctx);
			pa_ctx = NULL;
		}

		if (pa_ml) {
			pa_mainloop_free(pa_ml);
			pa_ml = NULL;
		}

		return ERR_CANT_OPEN;
	}

	while (pa_ready == 0) {
		pa_mainloop_iterate(pa_ml, 1, NULL);
	}

	if (pa_ready < 0) {
		if (pa_ctx) {
			pa_context_disconnect(pa_ctx);
			pa_context_unref(pa_ctx);
			pa_ctx = NULL;
		}

		if (pa_ml) {
			pa_mainloop_free(pa_ml);
			pa_ml = NULL;
		}

		return ERR_CANT_OPEN;
	}

	Error err = init_device();
	if (err == OK) {
		mutex = Mutex::create();
		thread = Thread::create(AudioDriverPulseAudio::thread_func, this);
	}

	return OK;
}

float AudioDriverPulseAudio::get_latency() {

	if (latency == 0) { //only do this once since it's approximate anyway
		lock();

		pa_usec_t palat = 0;
		if (pa_stream_get_state(pa_str) == PA_STREAM_READY) {
			int negative = 0;

			if (pa_stream_get_latency(pa_str, &palat, &negative) >= 0) {
				if (negative) {
					palat = 0;
				}
			}
		}

		if (palat > 0) {
			latency = double(palat) / 1000000.0;
		}

		unlock();
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

		int error_code;
		int byte_size = ad->pa_buffer_size * sizeof(int16_t);

		ad->lock();

		int ret;
		do {
			ret = pa_mainloop_iterate(ad->pa_ml, 0, NULL);
		} while (ret > 0);

		if (pa_stream_get_state(ad->pa_str) == PA_STREAM_READY) {
			const void *ptr = ad->samples_out.ptr();
			while (byte_size > 0) {
				size_t bytes = pa_stream_writable_size(ad->pa_str);
				if (bytes > 0) {
					if (bytes > byte_size) {
						bytes = byte_size;
					}

					int ret = pa_stream_write(ad->pa_str, ptr, bytes, NULL, 0LL, PA_SEEK_RELATIVE);
					if (ret >= 0) {
						byte_size -= bytes;
						ptr = (const char *)ptr + bytes;
					}
				} else {
					pa_mainloop_iterate(ad->pa_ml, 1, NULL);
				}
			}
		}

		// User selected a new device, finish the current one so we'll init the new device
		if (ad->device_name != ad->new_device) {
			ad->device_name = ad->new_device;
			ad->finish_device();

			Error err = ad->init_device();
			if (err != OK) {
				ERR_PRINT("PulseAudio: init_device error");
				ad->device_name = "Default";
				ad->new_device = "Default";

				err = ad->init_device();
				if (err != OK) {
					ad->active = false;
					ad->exit_thread = true;
					break;
				}
			}
		}

		ad->unlock();
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

void AudioDriverPulseAudio::pa_sinklist_cb(pa_context *c, const pa_sink_info *l, int eol, void *userdata) {
	AudioDriverPulseAudio *ad = (AudioDriverPulseAudio *)userdata;
	int ctr = 0;

	// If eol is set to a positive number, you're at the end of the list
	if (eol > 0) {
		return;
	}

	ad->pa_devices.push_back(l->name);
	ad->pa_status++;
}

Array AudioDriverPulseAudio::get_device_list() {

	pa_devices.clear();
	pa_devices.push_back("Default");

	if (pa_ctx == NULL) {
		return pa_devices;
	}

	lock();

	// Get the device list
	pa_status = 0;
	pa_operation *pa_op = pa_context_get_sink_info_list(pa_ctx, pa_sinklist_cb, (void *)this);
	if (pa_op) {
		while (pa_status == 0) {
			int ret = pa_mainloop_iterate(pa_ml, 1, NULL);
			if (ret < 0) {
				ERR_PRINT("pa_mainloop_iterate error");
			}
		}

		pa_operation_unref(pa_op);
	} else {
		ERR_PRINT("pa_context_get_server_info error");
	}

	unlock();

	return pa_devices;
}

String AudioDriverPulseAudio::get_device() {

	return device_name;
}

void AudioDriverPulseAudio::set_device(String device) {

	new_device = device;
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

void AudioDriverPulseAudio::finish_device() {

	if (pa_str) {
		pa_stream_disconnect(pa_str);
		pa_stream_unref(pa_str);
		pa_str = NULL;
	}
}

void AudioDriverPulseAudio::finish() {

	if (!thread)
		return;

	exit_thread = true;
	Thread::wait_to_finish(thread);

	finish_device();

	if (pa_ctx) {
		pa_context_disconnect(pa_ctx);
		pa_context_unref(pa_ctx);
		pa_ctx = NULL;
	}

	if (pa_ml) {
		pa_mainloop_free(pa_ml);
		pa_ml = NULL;
	}

	memdelete(thread);
	if (mutex) {
		memdelete(mutex);
		mutex = NULL;
	}

	thread = NULL;
}

AudioDriverPulseAudio::AudioDriverPulseAudio() {

	pa_ml = NULL;
	pa_ctx = NULL;
	pa_str = NULL;

	mutex = NULL;
	thread = NULL;

	device_name = "Default";
	new_device = "Default";
	default_device = "";

	samples_in.clear();
	samples_out.clear();

	mix_rate = 0;
	buffer_frames = 0;
	pa_buffer_size = 0;
	channels = 0;
	pa_channels = 0;
	pa_ready = 0;
	pa_status = 0;

	active = false;
	thread_exited = false;
	exit_thread = false;

	latency = 0;
}

AudioDriverPulseAudio::~AudioDriverPulseAudio() {
}

#endif
