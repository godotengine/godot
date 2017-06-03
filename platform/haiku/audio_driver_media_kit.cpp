/*************************************************************************/
/*  audio_driver_media_kit.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "audio_driver_media_kit.h"

#ifdef MEDIA_KIT_ENABLED

#include "global_config.h"

int32_t *AudioDriverMediaKit::samples_in = NULL;

Error AudioDriverMediaKit::init() {
	active = false;

	mix_rate = 44100;
	speaker_mode = SPEAKER_MODE_STEREO;
	channels = 2;

	int latency = GLOBAL_DEF("audio/output_latency", 25);
	buffer_size = nearest_power_of_2(latency * mix_rate / 1000);
	samples_in = memnew_arr(int32_t, buffer_size * channels);

	media_raw_audio_format format;
	format = media_raw_audio_format::wildcard;
	format.frame_rate = mix_rate;
	format.channel_count = channels;
	format.format = media_raw_audio_format::B_AUDIO_INT;
	format.byte_order = B_MEDIA_LITTLE_ENDIAN;
	format.buffer_size = buffer_size * sizeof(int32_t) * channels;

	player = new BSoundPlayer(
			&format,
			"godot_sound_server",
			AudioDriverMediaKit::PlayBuffer,
			NULL,
			this);

	if (player->InitCheck() != B_OK) {
		fprintf(stderr, "MediaKit ERR: can not create a BSoundPlayer instance\n");
		ERR_FAIL_COND_V(player == NULL, ERR_CANT_OPEN);
	}

	mutex = Mutex::create();
	player->Start();

	return OK;
}

void AudioDriverMediaKit::PlayBuffer(void *cookie, void *buffer, size_t size, const media_raw_audio_format &format) {
	AudioDriverMediaKit *ad = (AudioDriverMediaKit *)cookie;
	int32_t *buf = (int32_t *)buffer;

	if (!ad->active) {
		for (unsigned int i = 0; i < ad->buffer_size * ad->channels; i++) {
			AudioDriverMediaKit::samples_in[i] = 0;
		}
	} else {
		ad->lock();
		ad->audio_server_process(ad->buffer_size, AudioDriverMediaKit::samples_in);
		ad->unlock();
	}

	for (unsigned int i = 0; i < ad->buffer_size * ad->channels; i++) {
		buf[i] = AudioDriverMediaKit::samples_in[i];
	}
}

void AudioDriverMediaKit::start() {
	active = true;
}

int AudioDriverMediaKit::get_mix_rate() const {
	return mix_rate;
}

AudioDriverSW::SpeakerMode AudioDriverMediaKit::get_speaker_mode() const {
	return speaker_mode;
}

void AudioDriverMediaKit::lock() {
	if (!mutex)
		return;

	mutex->lock();
}

void AudioDriverMediaKit::unlock() {
	if (!mutex)
		return;

	mutex->unlock();
}

void AudioDriverMediaKit::finish() {
	delete player;

	if (samples_in) {
		memdelete_arr(samples_in);
	};

	if (mutex) {
		memdelete(mutex);
		mutex = NULL;
	}
}

AudioDriverMediaKit::AudioDriverMediaKit() {
	mutex = NULL;
	player = NULL;
}

AudioDriverMediaKit::~AudioDriverMediaKit() {
}

#endif
