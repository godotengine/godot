/**************************************************************************/
/*  audio_driver_sndio.h                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#ifdef SNDIO_ENABLED

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "core/templates/safe_refcount.h"
#include "servers/audio/audio_server.h"

#ifdef SOWRAP_ENABLED
#include "drivers/sndio/sndio-so_wrap.h"
#else
#include <sndio.h>
#endif

class AudioDriverSndio : public AudioDriver {
	class AudioDeviceSndio {
	public:
		struct sio_hdl *handle = nullptr;
		struct sio_par parameters;
		unsigned int channels;
		SafeFlag exit_thread;

		Error start(unsigned int p_mode);
		void close();
	};

	Thread thread;
	Thread input_thread;

	Mutex mutex;
	Mutex input_thread_mutex;

	AudioDeviceSndio device;
	AudioDeviceSndio input_device;

	static void thread_func(void *p_udata);
	static void input_thread_func(void *p_udata);

	SafeFlag active;

public:
	virtual const char *get_name() const override { return "sndio"; }

	virtual Error init() override;
	virtual void start() override { active.set(); }
	virtual int get_mix_rate() const override;
	virtual int get_input_mix_rate() const override;
	virtual SpeakerMode get_speaker_mode() const override;
	virtual float get_latency() override;

	virtual void lock() override { mutex.lock(); }
	virtual void unlock() override { mutex.unlock(); }
	virtual void finish() override;

	virtual Error input_start() override;
	virtual Error input_stop() override;
};

#endif /* SNDIO_ENABLED */
