/*************************************************************************/
/*  audio_driver_pulseaudio.h                                            */
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

#ifndef AUDIO_DRIVER_PULSEAUDIO_H
#define AUDIO_DRIVER_PULSEAUDIO_H

#ifdef PULSEAUDIO_ENABLED

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "servers/audio_server.h"

#include <pulse/pulseaudio.h>

class AudioDriverPulseAudio : public AudioDriver {
	Thread *thread = nullptr;
	Mutex mutex;

	pa_mainloop *pa_ml = nullptr;
	pa_context *pa_ctx = nullptr;
	pa_stream *pa_str = nullptr;
	pa_stream *pa_rec_str = nullptr;
	pa_channel_map pa_map;
	pa_channel_map pa_rec_map;

	String device_name = "Default";
	String new_device = "Default";
	String default_device;

	String capture_device_name;
	String capture_new_device;
	String capture_default_device;

	Vector<int32_t> samples_in;
	Vector<int16_t> samples_out;

	unsigned int mix_rate = 0;
	unsigned int buffer_frames = 0;
	unsigned int pa_buffer_size = 0;
	int channels = 0;
	int pa_ready = 0;
	int pa_status = 0;
	Array pa_devices;
	Array pa_rec_devices;

	bool active = false;
	bool thread_exited = false;
	mutable bool exit_thread = false;

	float latency = 0;

	static void pa_state_cb(pa_context *c, void *userdata);
	static void pa_sink_info_cb(pa_context *c, const pa_sink_info *l, int eol, void *userdata);
	static void pa_source_info_cb(pa_context *c, const pa_source_info *l, int eol, void *userdata);
	static void pa_server_info_cb(pa_context *c, const pa_server_info *i, void *userdata);
	static void pa_sinklist_cb(pa_context *c, const pa_sink_info *l, int eol, void *userdata);
	static void pa_sourcelist_cb(pa_context *c, const pa_source_info *l, int eol, void *userdata);

	Error init_device();
	void finish_device();

	Error capture_init_device();
	void capture_finish_device();

	void detect_channels(bool capture = false);

	static void thread_func(void *p_udata);

public:
	const char *get_name() const {
		return "PulseAudio";
	};

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const;
	virtual SpeakerMode get_speaker_mode() const;

	virtual Array get_device_list();
	virtual String get_device();
	virtual void set_device(String device);

	virtual Array capture_get_device_list();
	virtual void capture_set_device(const String &p_name);
	virtual String capture_get_device();

	virtual void lock();
	virtual void unlock();
	virtual void finish();

	virtual float get_latency();

	virtual Error capture_start();
	virtual Error capture_stop();

	AudioDriverPulseAudio();
	~AudioDriverPulseAudio() {}
};

#endif // PULSEAUDIO_ENABLED

#endif // AUDIO_DRIVER_PULSEAUDIO_H
