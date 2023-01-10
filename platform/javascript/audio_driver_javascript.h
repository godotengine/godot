/**************************************************************************/
/*  audio_driver_javascript.h                                             */
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

#ifndef AUDIO_DRIVER_JAVASCRIPT_H
#define AUDIO_DRIVER_JAVASCRIPT_H

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "servers/audio_server.h"

#include "godot_audio.h"

class AudioDriverJavaScript : public AudioDriver {
private:
	struct AudioContext {
		bool inited = false;
		float output_latency = 0.0;
		int state = -1;
		int channel_count = 0;
		int mix_rate = 0;
	};
	static AudioContext audio_context;

	float *output_rb = nullptr;
	float *input_rb = nullptr;

	int buffer_length = 0;
	int mix_rate = 0;
	int channel_count = 0;

	static void _state_change_callback(int p_state);
	static void _latency_update_callback(float p_latency);

	static AudioDriverJavaScript *singleton;

protected:
	void _audio_driver_process(int p_from = 0, int p_samples = 0);
	void _audio_driver_capture(int p_from = 0, int p_samples = 0);
	float *get_output_rb() const { return output_rb; }
	float *get_input_rb() const { return input_rb; }

	virtual Error create(int &p_buffer_samples, int p_channels) = 0;
	virtual void start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) = 0;
	virtual void finish_driver() {}

public:
	static bool is_available();

	virtual Error init() final;
	virtual void start() final;
	virtual void finish() final;

	virtual float get_latency() override;
	virtual int get_mix_rate() const override;
	virtual SpeakerMode get_speaker_mode() const override;

	virtual Error capture_start() override;
	virtual Error capture_stop() override;

	static void resume();

	AudioDriverJavaScript() {}
};

#ifdef NO_THREADS
class AudioDriverScriptProcessor : public AudioDriverJavaScript {
private:
	static void _process_callback();

	static AudioDriverScriptProcessor *singleton;

protected:
	Error create(int &p_buffer_samples, int p_channels) override;
	void start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) override;

public:
	virtual const char *get_name() const override { return "ScriptProcessor"; }

	virtual void lock() override {}
	virtual void unlock() override {}

	AudioDriverScriptProcessor() { singleton = this; }
};

class AudioDriverWorklet : public AudioDriverJavaScript {
private:
	static void _process_callback(int p_pos, int p_samples);
	static void _capture_callback(int p_pos, int p_samples);

	static AudioDriverWorklet *singleton;

protected:
	virtual Error create(int &p_buffer_size, int p_output_channels) override;
	virtual void start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) override;

public:
	virtual const char *get_name() const override { return "AudioWorklet"; }

	virtual void lock() override {}
	virtual void unlock() override {}

	AudioDriverWorklet() { singleton = this; }
};
#else
class AudioDriverWorklet : public AudioDriverJavaScript {
private:
	enum {
		STATE_LOCK,
		STATE_PROCESS,
		STATE_SAMPLES_IN,
		STATE_SAMPLES_OUT,
		STATE_MAX,
	};
	Mutex mutex;
	Thread thread;
	bool quit = false;
	int32_t state[STATE_MAX] = { 0 };

	static void _audio_thread_func(void *p_data);

protected:
	virtual Error create(int &p_buffer_size, int p_output_channels) override;
	virtual void start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) override;
	virtual void finish_driver() override;

public:
	virtual const char *get_name() const override { return "AudioWorklet"; }

	void lock() override;
	void unlock() override;
};
#endif

#endif // AUDIO_DRIVER_JAVASCRIPT_H
