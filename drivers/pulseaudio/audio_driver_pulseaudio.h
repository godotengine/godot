/*************************************************************************/
/*  audio_driver_pulseaudio.h                                            */
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

#ifdef PULSEAUDIO_ENABLED

#ifndef AUDIO_DRIVER_PULSEAUDIO_H
#define AUDIO_DRIVER_PULSEAUDIO_H

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "servers/audio_server.h"

#include <pulse/pulseaudio.h>

class AudioDriverPulseAudio : public AudioDriver {

	typedef pa_channel_map *(*pa_channel_map_init_stereo_t)(pa_channel_map *);
	typedef int (*pa_context_connect_t)(pa_context *, const char *, pa_context_flags_t, const pa_spawn_api *);
	typedef void (*pa_context_disconnect_t)(pa_context *);
	typedef int (*pa_context_errno_t)(pa_context *);
	typedef pa_operation *(*pa_context_get_server_info_t)(pa_context *, pa_server_info_cb_t, void *);
	typedef pa_operation *(*pa_context_get_sink_info_list_t)(pa_context *, pa_sink_info_cb_t, void *);
	typedef pa_operation *(*pa_context_get_sink_info_by_name_t)(pa_context *, const char *, pa_sink_info_cb_t cb, void *);
	typedef pa_operation *(*pa_context_get_source_info_list_t)(pa_context *, pa_source_info_cb_t, void *);
	typedef pa_operation *(*pa_context_get_source_info_by_name_t)(pa_context *, const char *, pa_source_info_cb_t, void *);
	typedef pa_context_state_t (*pa_context_get_state_t)(pa_context *);
	typedef pa_context *(*pa_context_new_t)(pa_mainloop_api *, const char *);
	typedef void (*pa_context_set_state_callback_t)(pa_context *, pa_context_notify_cb_t, void *);
	typedef void (*pa_context_unref_t)(pa_context *);
	typedef void (*pa_mainloop_free_t)(pa_mainloop *);
	typedef int (*pa_mainloop_iterate_t)(pa_mainloop *, int, int *);
	typedef pa_mainloop *(*pa_mainloop_new_t)(void);
	typedef pa_mainloop_api *(*pa_mainloop_get_api_t)(pa_mainloop *);
	typedef void (*pa_operation_unref_t)(pa_operation *);
	typedef const char *(*pa_strerror_t)(int);
	typedef int (*pa_stream_connect_playback_t)(pa_stream *, const char *, const pa_buffer_attr *, pa_stream_flags_t, const pa_cvolume *, pa_stream *);
	typedef int (*pa_stream_connect_record_t)(pa_stream *, const char *, const pa_buffer_attr *, pa_stream_flags_t);
	typedef int (*pa_stream_disconnect_t)(pa_stream *);
	typedef int (*pa_stream_drop_t)(pa_stream *);
	typedef int (*pa_stream_get_latency_t)(pa_stream *, pa_usec_t *, int *);
	typedef pa_stream_state_t (*pa_stream_get_state_t)(pa_stream *);
	typedef pa_stream *(*pa_stream_new_t)(pa_context *, const char *, const pa_sample_spec *, const pa_channel_map *);
	typedef int (*pa_stream_peek_t)(pa_stream *, const void **, size_t *);
	typedef size_t (*pa_stream_readable_size_t)(pa_stream *);
	typedef void (*pa_stream_unref_t)(pa_stream *);
	typedef int (*pa_stream_write_t)(pa_stream *, const void *, size_t, pa_free_cb_t, int64_t, pa_seek_mode_t);
	typedef size_t (*pa_stream_writable_size_t)(pa_stream *);

	pa_channel_map_init_stereo_t libpulse_pa_channel_map_init_stereo;
	pa_context_connect_t libpulse_pa_context_connect;
	pa_context_disconnect_t libpulse_pa_context_disconnect;
	pa_context_errno_t libpulse_pa_context_errno;
	pa_context_get_server_info_t libpulse_pa_context_get_server_info;
	pa_context_get_sink_info_list_t libpulse_pa_context_get_sink_info_list;
	pa_context_get_sink_info_by_name_t libpulse_pa_context_get_sink_info_by_name;
	pa_context_get_source_info_list_t libpulse_pa_context_get_source_info_list;
	pa_context_get_source_info_by_name_t libpulse_pa_context_get_source_info_by_name;
	pa_context_get_state_t libpulse_pa_context_get_state;
	pa_context_new_t libpulse_pa_context_new;
	pa_context_set_state_callback_t libpulse_pa_context_set_state_callback;
	pa_context_unref_t libpulse_pa_context_unref;
	pa_mainloop_free_t libpulse_pa_mainloop_free;
	pa_mainloop_iterate_t libpulse_pa_mainloop_iterate;
	pa_mainloop_new_t libpulse_pa_mainloop_new;
	pa_mainloop_get_api_t libpulse_pa_mainloop_get_api;
	pa_operation_unref_t libpulse_pa_operation_unref;
	pa_strerror_t libpulse_pa_strerror;
	pa_stream_connect_playback_t libpulse_pa_stream_connect_playback;
	pa_stream_connect_record_t libpulse_pa_stream_connect_record;
	pa_stream_disconnect_t libpulse_pa_stream_disconnect;
	pa_stream_drop_t libpulse_pa_stream_drop;
	pa_stream_get_latency_t libpulse_pa_stream_get_latency;
	pa_stream_get_state_t libpulse_pa_stream_get_state;
	pa_stream_new_t libpulse_pa_stream_new;
	pa_stream_peek_t libpulse_pa_stream_peek;
	pa_stream_readable_size_t libpulse_pa_stream_readable_size;
	pa_stream_write_t libpulse_pa_stream_write;
	pa_stream_writable_size_t libpulse_pa_stream_writable_size;
	pa_stream_unref_t libpulse_pa_stream_unref;

	void *libpulse;

	Thread *thread;
	Mutex *mutex;

	pa_mainloop *pa_ml;
	pa_context *pa_ctx;
	pa_stream *pa_str;
	pa_stream *pa_rec_str;
	pa_channel_map pa_map;
	pa_channel_map pa_rec_map;

	String device_name;
	String new_device;
	String default_device;

	String capture_device_name;
	String capture_new_device;
	String capture_default_device;

	Vector<int32_t> samples_in;
	Vector<int16_t> samples_out;

	unsigned int mix_rate;
	unsigned int buffer_frames;
	unsigned int input_buffer_frames;
	unsigned int pa_buffer_size;
	int channels;
	int pa_ready;
	int pa_status;
	Array pa_devices;
	Array pa_rec_devices;

	bool active;
	bool thread_exited;
	mutable bool exit_thread;

	float latency;

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
	~AudioDriverPulseAudio();
};

#endif // AUDIO_DRIVER_PULSEAUDIO_H

#endif // PULSEAUDIO_ENABLED
