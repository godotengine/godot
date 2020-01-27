/*************************************************************************/
/*  audio_driver_jack.cpp                                                */
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

#ifdef JACK_ENABLED

#include "audio_driver_jack.h"
#include "core/project_settings.h"
#include "core/version.h"

#include <stdio.h>
#include <string.h>

#define PORT_NAME_FORMAT "Out %u"
#define CAPTURE_PORT_NAME_FORMAT "In %u"

#define PORT_NAME_BUFFER_SIZE 32

const AudioDriverJACK::DeviceJACK AudioDriverJACK::devices[] = {

	// default device outputs to sound card in stereo
	{ "Sound card playback", SPEAKER_MODE_STEREO },

	// others are just virtual software clients
	{ "Stereo playback", SPEAKER_MODE_STEREO },
	{ "Surround 3.1 playback", SPEAKER_SURROUND_31 },
	{ "Surround 5.1 playback", SPEAKER_SURROUND_51 },
	{ "Surround 7.1 playback", SPEAKER_SURROUND_71 },
};

const unsigned AudioDriverJACK::num_devices =
		sizeof(devices) / sizeof(devices[0]);

const AudioDriverJACK::DeviceJACK AudioDriverJACK::capture_devices[] = {

	// default device inputs from sound card in stereo
	{ "Sound card capture", SPEAKER_MODE_STEREO },

	// others are just virtual software clients
	{ "Stereo capture", SPEAKER_MODE_STEREO },
};

const unsigned AudioDriverJACK::num_capture_devices =
		sizeof(capture_devices) / sizeof(capture_devices[0]);

int AudioDriverJACK::DeviceJACK::channels() const {

	switch (speaker_mode) {
		default:
		case SPEAKER_MODE_STEREO: return 2;
		case SPEAKER_SURROUND_31: return 4;
		case SPEAKER_SURROUND_51: return 6;
		case SPEAKER_SURROUND_71: return 8;
	}
}

Error AudioDriverJACK::init_device() {

	String client_name = GLOBAL_GET("application/config/name");

	if (client_name.length() == 0)
		client_name = VERSION_NAME;

	client = jack_client_open(client_name.utf8(), JackNoStartServer, NULL);

	if (!client)
		ERR_FAIL_COND_V(!client, ERR_CANT_OPEN);

	jack_set_process_callback(client, &process_func, this);

	unsigned mix_rate = jack_get_sample_rate(client);

	// set up playback
	const DeviceJACK &jdev = devices[device_index];
	unsigned channels = jdev.channels();

#define CHECK_FAIL(m_cond, msg)                 \
	if (m_cond) {                               \
		fprintf(stderr, "JACK ERR: %s\n", msg); \
		finish_device();                        \
		ERR_FAIL_COND_V(m_cond, ERR_CANT_OPEN); \
	}

	for (unsigned ch = 0; ch < channels; ++ch) {
		char port_name[PORT_NAME_BUFFER_SIZE];
		sprintf(port_name, PORT_NAME_FORMAT, ch + 1);

		jack_port_t *port = jack_port_register(
				client, port_name, JACK_DEFAULT_AUDIO_TYPE,
				JackPortIsOutput, 0);

		CHECK_FAIL(!port, "cannot register playback port");

		ports.push_back(port);
	}

	jack_nframes_t buffer_size = jack_get_buffer_size(client);
	samples_in.resize(buffer_size * channels);

	// set up capture
	const DeviceJACK &cdev = capture_devices[device_index];
	unsigned capture_channels = cdev.channels();

	for (unsigned ch = 0; ch < capture_channels; ++ch) {
		char port_name[PORT_NAME_BUFFER_SIZE];
		sprintf(port_name, CAPTURE_PORT_NAME_FORMAT, ch + 1);

		jack_port_t *port = jack_port_register(
				client, port_name, JACK_DEFAULT_AUDIO_TYPE,
				JackPortIsInput, 0);

		CHECK_FAIL(!port, "cannot register capture port");

		capture_ports.push_back(port);
	}

	// use the JACK buffer size or 30ms, whichever is larger
	unsigned input_latency = 30;
	unsigned input_buffer_frames = closest_power_of_2(input_latency * mix_rate / 1000);
	if (buffer_size > input_buffer_frames)
		input_buffer_frames = buffer_size;

	input_buffer_init(input_buffer_frames);

	return OK;
}

Error AudioDriverJACK::init() {

	Error err = init_device();
	if (err == OK)
		mutex = Mutex::create();

	return err;
}

void AudioDriverJACK::start() {

	active = true;

	if (!client)
		return;

	jack_activate(client);

	// if the sound card output was picked,
	// identify the physical client and connect
	// note: must always go after activate
	if (device_index == 0)
		connect_physical_ports();
	if (capture_device_index == 0)
		connect_physical_capture_ports();
}

int AudioDriverJACK::get_mix_rate() const {

	if (!client)
		return 0;

	return jack_get_sample_rate(client);
}

AudioDriver::SpeakerMode AudioDriverJACK::get_speaker_mode() const {

	const DeviceJACK &jdev = devices[device_index];
	return jdev.speaker_mode;
}

Array AudioDriverJACK::get_device_list() {

	Array names;

	for (unsigned i = 0; i < num_devices; ++i) {
		const DeviceJACK &jdev = devices[i];
		names.push_back(jdev.name);
	}

	return names;
}

String AudioDriverJACK::get_device() {

	const DeviceJACK &jdev = devices[device_index];
	return jdev.name;
}

void AudioDriverJACK::set_device(String device) {

	int new_index = -1;

	for (unsigned i = 0; i < num_devices && new_index == -1; ++i) {
		const DeviceJACK &jdev = devices[i];
		if (device == jdev.name)
			new_index = i;
	}

	// fallback to default
	if (new_index == -1)
		new_index = 0;

	// close current client, and reopen
	if (device_index != new_index) {
		finish_device();
		device_index = new_index;
		init_device();

		// if it was started before recreating, restart it
		if (active)
			start();
	}
}

void AudioDriverJACK::lock() {

	if (!mutex)
		return;
	mutex->lock();
}

void AudioDriverJACK::unlock() {

	if (!mutex)
		return;
	mutex->unlock();
}

void AudioDriverJACK::finish_device() {

	if (client) {
		jack_client_close(client);
		client = NULL;
	}

	ports.clear();
	capture_ports.clear();
}

void AudioDriverJACK::finish() {

	finish_device();

	if (mutex) {
		memdelete(mutex);
		mutex = NULL;
	}
}

static inline int32_t saturate16bit(int32_t sample) {
	sample = (sample < +32767) ? sample : +32767;
	sample = (sample > -32767) ? sample : -32767;
	return sample;
}

int AudioDriverJACK::process_func(jack_nframes_t total_frames, void *p_udata) {

	AudioDriverJACK *jd = (AudioDriverJACK *)p_udata;

	jack_port_t *const *ports = jd->ports.ptr();
	unsigned channels = jd->ports.size();

	jack_port_t *const *capture_ports = jd->capture_ports.ptr();
	unsigned capture_channels = jd->capture_ports.size();

	bool capture_active = jd->capture_active;

	Mutex *mutex = jd->mutex;

	// invoke try-lock; if it's a failure, fill channels with zero
	// never xrun the JACK server!
	if (mutex && mutex->try_lock() != OK) {
		for (unsigned ch = 0; ch < channels; ++ch) {
			float *ch_out = (float *)jack_port_get_buffer(ports[ch], total_frames);
			memset(ch_out, 0, total_frames * sizeof(float));
		}
		return 0;
	}

	int32_t *frames_in = jd->samples_in.ptrw();
	jack_nframes_t max_frames_in = jd->samples_in.size() / channels;

	jack_nframes_t frame_index = 0;

	while (frame_index < total_frames) {
		// the buffer size can change dynamically, and grow bigger than
		// our initial allocation. process the data in segments.

		jack_nframes_t current_frames = total_frames - frame_index;

		if (current_frames > max_frames_in)
			current_frames = max_frames_in;

		jd->audio_server_process(current_frames, frames_in);

		// deinterleave what we obtained, convert to +-1 floats
		for (unsigned ch = 0; ch < channels; ++ch) {
			float *ch_out = (float *)jack_port_get_buffer(ports[ch], total_frames) + frame_index;
			for (jack_nframes_t i = 0; i < current_frames; ++i) {
				int32_t sample = frames_in[ch + i * channels];
				ch_out[i] = (sample >> 16) * (1.0f / (1 << 16));
			}
		}

		// convert captured samples, and write to buffer
		if (capture_active && capture_channels >= 1) {
			const float *ch_in1 = (float *)jack_port_get_buffer(capture_ports[0], total_frames) + frame_index;
			const float *ch_in2;

			if (capture_channels >= 2)
				ch_in2 = (float *)jack_port_get_buffer(capture_ports[1], total_frames) + frame_index;
			else
				ch_in2 = ch_in1; // mono device

			for (jack_nframes_t i = 0; i < current_frames; ++i) {
				int32_t sample1 = saturate16bit((int32_t)(ch_in1[i] * 32767)) << 16;
				int32_t sample2 = saturate16bit((int32_t)(ch_in2[i] * 32767)) << 16;
				jd->input_buffer_write(sample1);
				jd->input_buffer_write(sample2);
			}
		}

		frame_index += current_frames;
	}

	if (mutex)
		mutex->unlock();

	return 0;
}

void AudioDriverJACK::connect_physical_ports() {

	const DeviceJACK &jdev = devices[device_index];
	unsigned channels = jdev.channels();

	// we need the effective name of our client
	const char *src_client = jack_get_client_name(client);
	if (!src_client)
		return;

	// list physical output ports
	const char **ports = jack_get_ports(
			client, NULL, JACK_DEFAULT_AUDIO_TYPE,
			JackPortIsInput | JackPortIsPhysical);

	if (!ports)
		return;

	// first port belongs to our wanted client
	const char *first_port = ports[0];
	if (!first_port) {
		jack_free(ports);
		return;
	}

	// port name must be "client:port"
	const char *dst_client_start = first_port;
	const char *dst_client_end = strchr(dst_client_start, ':');
	size_t dst_client_length = dst_client_end - dst_client_start;
	if (!dst_client_end) {
		jack_free(ports);
		return;
	}

	// allocate some space to construct the source port name
	Vector<char> src_port_buffer;
	src_port_buffer.resize(strlen(src_client) + 1 + PORT_NAME_BUFFER_SIZE);

	// connect up to `channels` ports, as long as it belongs to the client
	for (unsigned ch = 0; ch < channels; ++ch) {
		const char *dst_port = ports[ch];
		if (!dst_port)
			break;

		if (strncmp(dst_port, dst_client_start, dst_client_length + 1))
			break;

		char *src_port = src_port_buffer.ptrw();
		sprintf(src_port, "%s:" PORT_NAME_FORMAT, src_client, ch + 1);

		jack_connect(client, src_port, dst_port);
	}

	jack_free(ports);
}

void AudioDriverJACK::connect_physical_capture_ports() {

	const DeviceJACK &cdev = capture_devices[capture_device_index];
	unsigned capture_channels = cdev.channels();

	// we need the effective name of our client
	const char *dst_client = jack_get_client_name(client);
	if (!dst_client)
		return;

	// list physical output ports
	const char **ports = jack_get_ports(
			client, NULL, JACK_DEFAULT_AUDIO_TYPE,
			JackPortIsOutput | JackPortIsPhysical);

	if (!ports)
		return;

	// first port belongs to our wanted client
	const char *first_port = ports[0];
	if (!first_port) {
		jack_free(ports);
		return;
	}

	// port name must be "client:port"
	const char *src_client_start = first_port;
	const char *src_client_end = strchr(src_client_start, ':');
	size_t src_client_length = src_client_end - src_client_start;
	if (!src_client_end) {
		jack_free(ports);
		return;
	}

	// allocate some space to construct the destination port name
	Vector<char> dst_port_buffer;
	dst_port_buffer.resize(strlen(dst_client) + 1 + PORT_NAME_BUFFER_SIZE);

	// connect up to `capture_channels` ports, as long as it belongs to the client
	for (unsigned ch = 0; ch < capture_channels; ++ch) {
		const char *src_port = ports[ch];
		if (!src_port)
			break;

		if (strncmp(src_port, src_client_start, src_client_length + 1))
			break;

		char *dst_port = dst_port_buffer.ptrw();
		sprintf(dst_port, "%s:" CAPTURE_PORT_NAME_FORMAT, dst_client, ch + 1);

		jack_connect(client, src_port, dst_port);
	}

	jack_free(ports);
}

Error AudioDriverJACK::capture_start() {

	capture_active = true;
	return OK;
}

Error AudioDriverJACK::capture_stop() {

	capture_active = false;
	return OK;
}

void AudioDriverJACK::capture_set_device(const String &device) {

	int new_index = -1;

	for (unsigned i = 0; i < num_capture_devices && new_index == -1; ++i) {
		const DeviceJACK &cdev = capture_devices[i];
		if (device == cdev.name)
			new_index = i;
	}

	// fallback to default
	if (new_index == -1)
		new_index = 0;

	// close current client, and reopen
	if (capture_device_index != new_index) {
		finish_device();
		capture_device_index = new_index;
		init_device();

		// if it was started before recreating, restart it
		if (active)
			start();
	}
}

String AudioDriverJACK::capture_get_device() {

	const DeviceJACK &cdev = capture_devices[device_index];
	return cdev.name;
}

Array AudioDriverJACK::capture_get_device_list() {

	Array names;

	for (unsigned i = 0; i < num_capture_devices; ++i) {
		const DeviceJACK &cdev = capture_devices[i];
		names.push_back(cdev.name);
	}

	return names;
}

AudioDriverJACK::AudioDriverJACK() :
		mutex(NULL),
		client(NULL),
		device_index(0),
		capture_device_index(0),
		active(false),
		capture_active(false) {
}

AudioDriverJACK::~AudioDriverJACK() {
}

#endif
