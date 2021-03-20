/*************************************************************************/
/*  audio_driver_javascript.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef AUDIO_DRIVER_JAVASCRIPT_H
#define AUDIO_DRIVER_JAVASCRIPT_H

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "servers/audio_server.h"

#include "godot_audio.h"

class AudioDriverJavaScript : public AudioDriver {
public:
	class AudioNode {
	public:
		virtual int create(int p_buffer_size, int p_output_channels) = 0;
		virtual void start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) = 0;
		virtual void finish() {}
		virtual void lock() {}
		virtual void unlock() {}
		virtual ~AudioNode() {}
	};

	class WorkletNode : public AudioNode {
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

	public:
		int create(int p_buffer_size, int p_output_channels) override;
		void start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) override;
		void finish() override;
		void lock() override;
		void unlock() override;
	};

	class ScriptProcessorNode : public AudioNode {
	private:
		static void _process_callback();

	public:
		int create(int p_buffer_samples, int p_channels) override;
		void start(float *p_out_buf, int p_out_buf_size, float *p_in_buf, int p_in_buf_size) override;
	};

private:
	AudioNode *node = nullptr;

	float *output_rb = nullptr;
	float *input_rb = nullptr;

	int buffer_length = 0;
	int mix_rate = 0;
	int channel_count = 0;
	int state = 0;
	float output_latency = 0.0;

	static void _state_change_callback(int p_state);
	static void _latency_update_callback(float p_latency);

protected:
	void _audio_driver_process(int p_from = 0, int p_samples = 0);
	void _audio_driver_capture(int p_from = 0, int p_samples = 0);

public:
	static bool is_available();

	static AudioDriverJavaScript *singleton;

	virtual const char *get_name() const;

	virtual Error init();
	virtual void start();
	void resume();
	virtual float get_latency();
	virtual int get_mix_rate() const;
	virtual SpeakerMode get_speaker_mode() const;
	virtual void lock();
	virtual void unlock();
	virtual void finish();

	virtual Error capture_start();
	virtual Error capture_stop();

	AudioDriverJavaScript();
};
#endif
