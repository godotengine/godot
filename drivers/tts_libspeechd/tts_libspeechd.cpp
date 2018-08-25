/*************************************************************************/
/*  tts_libspeechd.cpp                                                   */
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

#include "tts_libspeechd.h"

/*
API Documentation:
https://freebsoft.org/doc/speechd/speech-dispatcher.html
*/

#ifdef SPDTTS_ENABLED

List<int> TTSDriverSPD::messages;

void TTSDriverSPD::end_of_speech(size_t msg_id, size_t client_id, SPDNotificationType type) {

	messages.erase(msg_id);
};

void TTSDriverSPD::speak(const String &p_text, bool p_interrupt) {

	if (synth) {
		if (p_interrupt) {
			spd_cancel(synth);
		}
		int id = spd_say(synth, SPD_MESSAGE, p_text.utf8().get_data());
		if (id != -1)
			messages.push_back(id);
	}
};

void TTSDriverSPD::stop() {

	if (synth) {
		spd_cancel(synth);
	}
};

Array TTSDriverSPD::get_voices() {

	Array list;
	if (synth) {
		SPDVoice **voices = spd_list_synthesis_voices(synth);
		if (voices != NULL) {
			SPDVoice **voices_ptr = voices;
			while (*voices_ptr != NULL) {
				Dictionary voice_d;
				voice_d["name"] = String::utf8((*voices_ptr)->name);
				voice_d["language"] = String::utf8((*voices_ptr)->language) + "_" + String::utf8((*voices_ptr)->variant);
				list.push_back(voice_d);

				voices_ptr++;
			}
			free_spd_voices(voices);
		}
	}
	return list;
};

void TTSDriverSPD::set_voice(const String &p_voice) {

	if (synth) {
		spd_set_synthesis_voice(synth, p_voice.utf8().get_data());
	}
};

bool TTSDriverSPD::is_speaking() {

	return !messages.empty();
};

void TTSDriverSPD::set_volume(int p_volume) {

	if (synth) {
		spd_set_volume(synth, p_volume * 2 - 100);
	}
};

int TTSDriverSPD::get_volume() {

	if (synth) {
		return spd_get_volume(synth) / 2 + 100;
	} else {
		return 0;
	}
};

void TTSDriverSPD::set_rate(int p_rate) {

	if (synth) {
		spd_set_voice_rate(synth, p_rate);
	}
};

int TTSDriverSPD::get_rate() {

	if (synth) {
		return spd_get_voice_rate(synth);
	} else {
		return 0;
	}
};

TTSDriverSPD::TTSDriverSPD() {

	synth = spd_open("Godot", NULL, NULL, SPD_MODE_THREADED);
	ERR_FAIL_COND(!synth);
	if (synth) {
		synth->callback_end = synth->callback_cancel = end_of_speech;

		spd_set_notification_on(synth, SPD_END);
		spd_set_notification_on(synth, SPD_CANCEL);
	}
};

TTSDriverSPD::~TTSDriverSPD() {

	if (synth) {
		spd_close(synth);
		synth = NULL;
	}
};

#endif
