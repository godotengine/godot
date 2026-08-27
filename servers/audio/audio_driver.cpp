/**************************************************************************/
/*  audio_driver.cpp                                                      */
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

#include "audio_driver.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "scene/resources/audio/audio_stream.h"
#include "servers/audio/audio_driver_dummy.h"
#include "servers/audio/audio_server.h"

AudioDriver *AudioDriver::singleton = nullptr;
AudioDriver *AudioDriver::get_singleton() {
	return singleton;
}

void AudioDriver::set_singleton() {
	singleton = this;
}

#ifdef DEBUG_ENABLED
void AudioDriver::start_counting_ticks() {
	prof_ticks.set(OS::get_singleton()->get_ticks_usec());
}

void AudioDriver::stop_counting_ticks() {
	prof_time.add(OS::get_singleton()->get_ticks_usec() - prof_ticks.get());
}
#endif // DEBUG_ENABLED

void AudioDriver::audio_server_process(int p_frames, int32_t *p_buffer, bool p_update_mix_time) {
	if (p_update_mix_time) {
		update_mix_time(p_frames);
	}

	if (AudioServer::get_singleton()) {
		AudioServer::get_singleton()->_driver_process(p_frames, p_buffer);
	}
}

void AudioDriver::update_mix_time(int p_frames) {
	_last_mix_frames = p_frames;
	if (OS::get_singleton()) {
		_last_mix_time = OS::get_singleton()->get_ticks_usec();
	}
}

double AudioDriver::get_time_since_last_mix() {
	lock();
	uint64_t last_mix_time = _last_mix_time;
	unlock();
	return (OS::get_singleton()->get_ticks_usec() - last_mix_time) / 1000000.0;
}

double AudioDriver::get_time_to_next_mix() {
	lock();
	uint64_t last_mix_time = _last_mix_time;
	uint64_t last_mix_frames = _last_mix_frames;
	unlock();
	double total = (OS::get_singleton()->get_ticks_usec() - last_mix_time) / 1000000.0;
	double mix_buffer = last_mix_frames / (double)get_mix_rate();
	return mix_buffer - total;
}

void AudioDriver::input_buffer_init(int driver_buffer_frames) {
	const int input_buffer_channels = 2;
	input_buffer.resize(driver_buffer_frames * input_buffer_channels * 4);
	input_position = 0;
	input_size = 0;
}

void AudioDriver::input_buffer_write(int32_t sample) {
	if ((int)input_position < input_buffer.size()) {
		input_buffer.write[input_position++] = sample;
		if ((int)input_position >= input_buffer.size()) {
			input_position = 0;
		}
		if ((int)input_size < input_buffer.size()) {
			input_size++;
		}
	} else {
		// This protection was added in GH-26505 due to a "possible crash".
		// This cannot have happened unless two non-locked threads entered function simultaneously, which was possible when multiple calls to
		// `AudioDriver::input_start()` did not raise an error condition.
		WARN_PRINT("input_buffer_write: Invalid input_position=" + itos(input_position) + " input_buffer.size()=" + itos(input_buffer.size()));
	}
}

int AudioDriver::_get_configured_mix_rate() {
	StringName audio_driver_setting = "audio/driver/mix_rate";
	int mix_rate = GLOBAL_GET(audio_driver_setting);

#ifdef WEB_ENABLED
	// `0` is an acceptable value (resorts to the browser's default).
	return MAX(0, mix_rate);
#else // !WEB_ENABLED
	// In the case of invalid mix rate, let's default to a sensible value..
	if (mix_rate <= 0) {
		WARN_PRINT(vformat("Invalid mix rate of %d, consider reassigning setting \'%s\'. \nDefaulting mix rate to value %d.",
				mix_rate, audio_driver_setting, AudioDriverManager::DEFAULT_MIX_RATE));
		mix_rate = AudioDriverManager::DEFAULT_MIX_RATE;
	}
	return mix_rate;
#endif
}

AudioDriver::SpeakerMode AudioDriver::get_speaker_mode_by_total_channels(int p_channels) const {
	switch (p_channels) {
		case 4:
			return SPEAKER_SURROUND_31;
		case 6:
			return SPEAKER_SURROUND_51;
		case 8:
			return SPEAKER_SURROUND_71;
	}

	// Default to STEREO
	return SPEAKER_MODE_STEREO;
}

int AudioDriver::get_total_channels_by_speaker_mode(AudioDriver::SpeakerMode p_mode) const {
	switch (p_mode) {
		case SPEAKER_MODE_STEREO:
			return 2;
		case SPEAKER_SURROUND_31:
			return 4;
		case SPEAKER_SURROUND_51:
			return 6;
		case SPEAKER_SURROUND_71:
			return 8;
	}

	ERR_FAIL_V(2);
}

PackedStringArray AudioDriver::get_output_device_list() {
	PackedStringArray list;

	list.push_back("Default");

	return list;
}

String AudioDriver::get_output_device() {
	return "Default";
}

PackedStringArray AudioDriver::get_input_device_list() {
	PackedStringArray list;

	list.push_back("Default");

	return list;
}

void AudioDriver::start_sample_playback(const Ref<AudioSamplePlayback> &p_playback) {
	if (p_playback.is_valid()) {
		if (p_playback->stream.is_valid()) {
			WARN_PRINT_ED(vformat(R"(Trying to play stream (%s) as a sample (%s), but the driver doesn't support sample playback.)", p_playback->get_instance_id(), p_playback->stream->get_instance_id()));
		} else {
			WARN_PRINT_ED(vformat(R"(Trying to play stream (%s) as a null sample, but the driver doesn't support sample playback.)", p_playback->get_instance_id()));
		}
	} else {
		WARN_PRINT_ED("Trying to play a null sample playback from a driver that doesn't support sample playback.");
	}
}

AudioDriverDummy AudioDriverManager::dummy_driver;
AudioDriver *AudioDriverManager::drivers[MAX_DRIVERS] = {
	&AudioDriverManager::dummy_driver,
};
int AudioDriverManager::driver_count = 1;

void AudioDriverManager::add_driver(AudioDriver *p_driver) {
	ERR_FAIL_COND(driver_count >= MAX_DRIVERS);
	drivers[driver_count - 1] = p_driver;

	// Last driver is always our dummy driver
	drivers[driver_count++] = &AudioDriverManager::dummy_driver;
}

int AudioDriverManager::get_driver_count() {
	return driver_count;
}

void AudioDriverManager::initialize(int p_driver) {
	GLOBAL_DEF_RST("audio/driver/enable_input", false);
	GLOBAL_DEF_RST(PropertyInfo(Variant::INT, "audio/driver/mix_rate", PROPERTY_HINT_RANGE, "11025,192000,1,or_greater,suffix:Hz"), DEFAULT_MIX_RATE);
	GLOBAL_DEF_RST(PropertyInfo(Variant::INT, "audio/driver/mix_rate.web", PROPERTY_HINT_RANGE, "0,192000,1,or_greater,suffix:Hz"), 0); // Safer default output_latency for web (use browser default).

	int failed_driver = -1;

	// Check if there is a selected driver
	if (p_driver >= 0 && p_driver < driver_count) {
		if (drivers[p_driver]->init() == OK) {
			drivers[p_driver]->set_singleton();
			return;
		} else {
			failed_driver = p_driver;
		}
	}

	// No selected driver, try them all in order
	for (int i = 0; i < driver_count; i++) {
		// Don't re-init the driver if it failed above
		if (i == failed_driver) {
			continue;
		}

		if (drivers[i]->init() == OK) {
			drivers[i]->set_singleton();
			break;
		}
	}

	if (driver_count > 1 && String(AudioDriver::get_singleton()->get_name()) == "Dummy") {
		WARN_PRINT("All audio drivers failed, falling back to the dummy driver.");
	}
}

AudioDriver *AudioDriverManager::get_driver(int p_driver) {
	ERR_FAIL_INDEX_V(p_driver, driver_count, nullptr);
	return drivers[p_driver];
}
