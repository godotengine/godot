/**************************************************************************/
/*  audio_driver_miniaudio.h                                              */
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

#include "core/math/math_defs.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "servers/audio/audio_driver.h"

// Enable/disable debug printing underruns and overruns
#define GDT_MA_AUDIO_DRIVER_DEBUG 0

struct ma_device;

class AudioDriverMiniaudio final : public AudioDriver {
	static void output_device_cb(ma_device *pDevice, void *pOutput, const void *pInput, uint32_t frameCount);
	static void input_device_cb(ma_device *pDevice, void *pOutput, const void *pInput, uint32_t frameCount);

	template <class T>
	static void output_device_notification_cb(T p_notification);
	template <class T>
	static void input_device_notification_cb(T p_notification);

public:
	AudioDriverMiniaudio();
	~AudioDriverMiniaudio();

	const char *get_name() const final { return "miniaudio"; }

	Error init() final;
	void start() final;
	int get_mix_rate() const final;
	SpeakerMode get_speaker_mode() const final;
	float get_latency() final;

	void lock() final;
	void unlock() final;
	void finish() final;

	PackedStringArray get_output_device_list() final;
	String get_output_device() final;
	void set_output_device(const String &p_name) final;

	Error input_start() final;
	Error input_stop() final;

	PackedStringArray get_input_device_list() final;
	String get_input_device() final;
	void set_input_device(const String &p_name) final;

private:
	// This must not be called while input device is running
	bool _init_input_ring_buffer();
	// These must be called from output audio device thread
	void _drain_input_buffer();
	void _commit_input_buffer();
	void _reset_input_buffers();

	static void _audio_server_update(void *p_userdata);
	void _process_main_thread_update();

private:
	Thread::ID _owner_therad_id;
	_FORCE_INLINE_ bool _is_owner_thread() const { return Thread::get_caller_id() == _owner_therad_id; }

	bool _update_cb_registered;
	bool _out_should_be_started;
	bool _in_should_be_started;

	int _sample_rate;
	int _num_channels;

	// If either input or output was reinitialized, we need to let
	// audio callback know to resynchronize buffers.
	// Note: we might get a bit of garbage in the buffesr before this is processed,
	// but that's acceptable for something that happens once in a lifetime.
	SafeFlag _device_was_reinitialized;

	// Currently AudioServer relies on mutex in AudioDriver
	Mutex _mix_mutex;

	struct MAData;
	MAData *_ma_data;

	// At the moment Godot only uses 2 channel input,
	// miniaudio doesn't have this limitation,
	// in the future this can be project config,
	// or just whatever device gives us
	static constexpr uint32_t NUM_IN_CHANNELS = 2;
	static constexpr uint32_t PERIOD_FRAME_COUNT = 512; // TODO: get this constant from AudioServer (?)

	// Must be initialized on main thread, never reset()
	LocalVector<int32_t> _input_stage_buffer;

#if GDT_MA_AUDIO_DRIVER_DEBUG
	SafeNumeric<uint64_t> _input_rb_overrun{ 0 };
	SafeNumeric<uint64_t> _input_sb_overrun{ 0 };
	SafeNumeric<uint64_t> _underrun{ 0 };
#endif
};
