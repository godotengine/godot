/**************************************************************************/
/*  audio_driver_openharmony.h                                            */
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

#include "servers/audio_server.h"

#include <ohaudio/native_audiocapturer.h>
#include <ohaudio/native_audiorenderer.h>
#include <ohaudio/native_audiostreambuilder.h>

class AudioDriverOpenHarmony : public AudioDriver {
	bool active = false;
	Mutex mutex;
	bool pause = false;

	uint32_t buffer_size = 0;
	int32_t *mixdown_buffer = nullptr;

	OH_AudioStreamBuilder *audio_stream_builder = nullptr;
	OH_AudioRenderer *audio_renderer = nullptr;

	OH_AudioStreamBuilder *audio_stream_capture_builder = nullptr;
	OH_AudioCapturer *audio_capturer = nullptr;

	OH_AudioData_Callback_Result _buffer_callback(OH_AudioRenderer *renderer, void *userData, void *audioData, int32_t audioDataSize);
	static OH_AudioData_Callback_Result _buffer_callbacks(OH_AudioRenderer *renderer, void *userData, void *audioData, int32_t audioDataSize);

	// Capturer callback functions
	int32_t _capturer_read_data(OH_AudioCapturer *capturer, void *buffer, int32_t length);
	static int32_t _on_capturer_read_data(OH_AudioCapturer *capturer, void *userData, void *buffer, int32_t length);
	static int32_t _on_capturer_error(OH_AudioCapturer *capturer, void *userData, OH_AudioStream_Result error);
	static int32_t _on_capturer_interrupt_event(OH_AudioCapturer *capturer, void *userData, OH_AudioInterrupt_ForceType type, OH_AudioInterrupt_Hint hint);
	static int32_t _on_capturer_stream_event(OH_AudioCapturer *capturer, void *userData, OH_AudioStream_Event event);

public:
	virtual const char *get_name() const override {
		return "OpenHarmony";
	}

	virtual Error init() override;
	virtual void start() override;
	virtual int get_mix_rate() const override;
	virtual SpeakerMode get_speaker_mode() const override;

	virtual void lock() override;
	virtual void unlock() override;
	virtual void finish() override;

	virtual Error input_start() override;
	virtual Error input_stop() override;

	void set_pause(bool p_pause);

	AudioDriverOpenHarmony();
};
