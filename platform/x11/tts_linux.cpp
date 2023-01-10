/**************************************************************************/
/*  tts_linux.cpp                                                         */
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

#include "tts_linux.h"

#include "core/project_settings.h"

TTS_Linux *TTS_Linux::singleton = nullptr;

static bool _is_whitespace(CharType c) {
	return c == '\t' || c == ' ';
}

void TTS_Linux::speech_init_thread_func(void *p_userdata) {
	TTS_Linux *tts = (TTS_Linux *)p_userdata;
	if (tts) {
		MutexLock thread_safe_method(tts->_thread_safe_);
#ifdef DEBUG_ENABLED
		int dylibloader_verbose = 1;
#else
		int dylibloader_verbose = 0;
#endif
		if (initialize_speechd(dylibloader_verbose) == 0) {
			CharString class_str;
			String config_name = GLOBAL_GET("application/config/name");
			if (config_name.length() == 0) {
				class_str = "Godot_Engine";
			} else {
				class_str = config_name.utf8();
			}
			tts->synth = spd_open(class_str, "Godot_Engine_Speech_API", "Godot_Engine", SPD_MODE_THREADED);
			if (tts->synth) {
				tts->synth->callback_end = &speech_event_callback;
				tts->synth->callback_cancel = &speech_event_callback;
				tts->synth->callback_im = &speech_event_index_mark;
				spd_set_notification_on(tts->synth, SPD_END);
				spd_set_notification_on(tts->synth, SPD_CANCEL);

				print_verbose("Text-to-Speech: Speech Dispatcher initialized.");
			} else {
				print_verbose("Text-to-Speech: Cannot initialize Speech Dispatcher synthesizer!");
			}
		} else {
			print_verbose("Text-to-Speech: Cannot load Speech Dispatcher library!");
		}
	}
}

void TTS_Linux::speech_event_index_mark(size_t p_msg_id, size_t p_client_id, SPDNotificationType p_type, char *p_index_mark) {
	TTS_Linux *tts = TTS_Linux::get_singleton();
	if (tts && tts->ids.has(p_msg_id)) {
		MutexLock thread_safe_method(tts->_thread_safe_);
		// Get word offset from the index mark injected to the text stream.
		String mark = String::utf8(p_index_mark);
		OS::get_singleton()->tts_post_utterance_event(OS::TTS_UTTERANCE_BOUNDARY, tts->ids[p_msg_id], mark.to_int());
	}
}

void TTS_Linux::speech_event_callback(size_t p_msg_id, size_t p_client_id, SPDNotificationType p_type) {
	TTS_Linux *tts = TTS_Linux::get_singleton();
	if (tts) {
		MutexLock thread_safe_method(tts->_thread_safe_);
		List<OS::TTSUtterance> &queue = tts->queue;
		if (!tts->paused && tts->ids.has(p_msg_id)) {
			if (p_type == SPD_EVENT_END) {
				OS::get_singleton()->tts_post_utterance_event(OS::TTS_UTTERANCE_ENDED, tts->ids[p_msg_id]);
				tts->ids.erase(p_msg_id);
				tts->last_msg_id = -1;
				tts->speaking = false;
			} else if (p_type == SPD_EVENT_CANCEL) {
				OS::get_singleton()->tts_post_utterance_event(OS::TTS_UTTERANCE_CANCELED, tts->ids[p_msg_id]);
				tts->ids.erase(p_msg_id);
				tts->last_msg_id = -1;
				tts->speaking = false;
			}
		}
		if (!tts->speaking && queue.size() > 0) {
			OS::TTSUtterance &message = queue.front()->get();

			// Inject index mark after each word.
			String text;
			String language;
			SPDVoice **voices = spd_list_synthesis_voices(tts->synth);
			if (voices != nullptr) {
				SPDVoice **voices_ptr = voices;
				while (*voices_ptr != nullptr) {
					if (String::utf8((*voices_ptr)->name) == message.voice) {
						language = String::utf8((*voices_ptr)->language);
						break;
					}
					voices_ptr++;
				}
				free_spd_voices(voices);
			}
			PoolIntArray breaks;
			for (int i = 0; i < message.text.size(); i++) {
				if (_is_whitespace(message.text[i])) {
					breaks.push_back(i);
				}
			}
			int prev = 0;
			for (int i = 0; i < breaks.size(); i++) {
				text += message.text.substr(prev, breaks[i] - prev);
				text += "<mark name=\"" + String::num_int64(breaks[i], 10) + "\"/>";
				prev = breaks[i];
			}
			text += message.text.substr(prev, -1);

			spd_set_synthesis_voice(tts->synth, message.voice.utf8().get_data());
			spd_set_volume(tts->synth, message.volume * 2 - 100);
			spd_set_voice_pitch(tts->synth, (message.pitch - 1) * 100);
			float rate = 0;
			if (message.rate > 1.f) {
				rate = log10(MIN(message.rate, 2.5f)) / log10(2.5f) * 100;
			} else if (message.rate < 1.f) {
				rate = log10(MAX(message.rate, 0.5f)) / log10(0.5f) * -100;
			}
			spd_set_voice_rate(tts->synth, rate);
			spd_set_data_mode(tts->synth, SPD_DATA_SSML);
			tts->last_msg_id = spd_say(tts->synth, SPD_TEXT, text.utf8().get_data());
			tts->ids[tts->last_msg_id] = message.id;
			OS::get_singleton()->tts_post_utterance_event(OS::TTS_UTTERANCE_STARTED, message.id);

			queue.pop_front();
			tts->speaking = true;
		}
	}
}

bool TTS_Linux::is_speaking() const {
	return speaking;
}

bool TTS_Linux::is_paused() const {
	return paused;
}

Array TTS_Linux::get_voices() const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!synth, Array());
	Array list;
	SPDVoice **voices = spd_list_synthesis_voices(synth);
	if (voices != nullptr) {
		SPDVoice **voices_ptr = voices;
		while (*voices_ptr != nullptr) {
			Dictionary voice_d;
			voice_d["name"] = String::utf8((*voices_ptr)->name);
			voice_d["id"] = String::utf8((*voices_ptr)->name);
			voice_d["language"] = String::utf8((*voices_ptr)->language) + "_" + String::utf8((*voices_ptr)->variant);
			list.push_back(voice_d);

			voices_ptr++;
		}
		free_spd_voices(voices);
	}
	return list;
}

void TTS_Linux::speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!synth);
	if (p_interrupt) {
		stop();
	}

	if (p_text.empty()) {
		OS::get_singleton()->tts_post_utterance_event(OS::TTS_UTTERANCE_CANCELED, p_utterance_id);
		return;
	}

	OS::TTSUtterance message;
	message.text = p_text;
	message.voice = p_voice;
	message.volume = CLAMP(p_volume, 0, 100);
	message.pitch = CLAMP(p_pitch, 0.f, 2.f);
	message.rate = CLAMP(p_rate, 0.1f, 10.f);
	message.id = p_utterance_id;
	queue.push_back(message);

	if (is_paused()) {
		resume();
	} else {
		speech_event_callback(0, 0, SPD_EVENT_BEGIN);
	}
}

void TTS_Linux::pause() {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!synth);
	if (spd_pause(synth) == 0) {
		paused = true;
	}
}

void TTS_Linux::resume() {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!synth);
	spd_resume(synth);
	paused = false;
}

void TTS_Linux::stop() {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!synth);
	for (List<OS::TTSUtterance>::Element *E = queue.front(); E; E = E->next()) {
		OS::TTSUtterance &message = E->get();
		OS::get_singleton()->tts_post_utterance_event(OS::TTS_UTTERANCE_CANCELED, message.id);
	}
	if ((last_msg_id != -1) && ids.has(last_msg_id)) {
		OS::get_singleton()->tts_post_utterance_event(OS::TTS_UTTERANCE_CANCELED, ids[last_msg_id]);
	}
	queue.clear();
	ids.clear();
	last_msg_id = -1;
	spd_cancel(synth);
	spd_resume(synth);
	speaking = false;
	paused = false;
}

TTS_Linux *TTS_Linux::get_singleton() {
	return singleton;
}

TTS_Linux::TTS_Linux() {
	singleton = this;
	// Speech Dispatcher init can be slow, it might wait for helper process to start on background, so run it in the thread.
	init_thread.start(speech_init_thread_func, this);
}

TTS_Linux::~TTS_Linux() {
	init_thread.wait_to_finish();
	if (synth) {
		spd_close(synth);
	}

	singleton = nullptr;
}
