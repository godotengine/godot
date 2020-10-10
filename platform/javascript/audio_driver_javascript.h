/*************************************************************************/
/*  audio_driver_javascript.h                                            */
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

#ifndef AUDIO_DRIVER_JAVASCRIPT_H
#define AUDIO_DRIVER_JAVASCRIPT_H

#include "servers/audio_server.h"

#include "core/os/mutex.h"
#include "core/os/thread.h"

class AudioDriverJavaScript : public AudioDriver {
private:
	float *internal_buffer = nullptr;

	int buffer_length = 0;
	int mix_rate = 0;
	int channel_count = 0;

public:
#ifndef NO_THREADS
	Mutex mutex;
	Thread *thread = nullptr;
	bool quit = false;
	bool needs_process = true;

	static void _audio_thread_func(void *p_data);
#endif

	void _js_driver_process();

	static bool is_available();
	void process_capture(float sample);

	static AudioDriverJavaScript *singleton;

	const char *get_name() const override;

	Error init() override;
	void start() override;
	void resume();
	float get_latency() override;
	int get_mix_rate() const override;
	SpeakerMode get_speaker_mode() const override;
	void lock() override;
	void unlock() override;
	void finish() override;
	void finish_async();

	Error capture_start() override;
	Error capture_stop() override;

	AudioDriverJavaScript();
};

#endif
