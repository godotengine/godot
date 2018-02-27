/*************************************************************************/
/*  audio_driver_wasapi.h                                                */
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

#ifndef AUDIO_DRIVER_WASAPI_H
#define AUDIO_DRIVER_WASAPI_H

#ifdef WASAPI_ENABLED

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "servers/audio_server.h"

#include <audioclient.h>
#include <mmdeviceapi.h>
#include <windows.h>

class AudioDriverWASAPI : public AudioDriver {

	HANDLE event;
	// Audio out
	IAudioClient *audio_client;
	IAudioRenderClient *render_client;
	// Microphone
	class MicrophoneDeviceOutputDirectWASAPI : public MicrophoneDeviceOutputDirect {
	public:
		IAudioClient *audio_client;
		IAudioCaptureClient *capture_client;
	};
	//
	Mutex *mutex;
	Thread *thread;

	String device_name;
	String new_device;
	String capture_device_default_name;

	WORD format_tag;
	WORD bits_per_sample;

	Vector<int32_t> samples_in;

	Map<StringName, StringName> capture_device_id_map;

	unsigned int buffer_size;
	unsigned int channels;
	unsigned int wasapi_channels;
	int mix_rate;
	int buffer_frames;

	bool thread_exited;
	mutable bool exit_thread;
	bool active;

	_FORCE_INLINE_ void write_sample(AudioDriverWASAPI *ad, BYTE *buffer, int i, int32_t sample);
	static void thread_func(void *p_udata);

	StringName get_default_capture_device_name(IMMDeviceEnumerator *p_enumerator);

	Error init_render_device(bool reinit = false);
	Error init_capture_devices(bool reinit = false);

	Error finish_render_device();
	Error finish_capture_devices();

public:
	virtual const char *get_name() const {
		return "WASAPI";
	}

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const;
	virtual SpeakerMode get_speaker_mode() const;
	virtual Array get_device_list();
	virtual String get_device();
	virtual void set_device(String device);
	virtual void lock();
	virtual void unlock();
	virtual void finish();

	virtual bool capture_device_start(StringName p_name);
	virtual bool capture_device_stop(StringName p_name);
	virtual PoolStringArray capture_device_get_names();
	virtual StringName capture_device_get_default_name();

	AudioDriverWASAPI();
};

#endif // AUDIO_DRIVER_WASAPI_H
#endif
