/**************************************************************************/
/*  tts_windows.cpp                                                       */
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

#include "tts_windows.h"

#include "tts_driver_sapi.h"

#ifdef WINRT_ENABLED
#include "tts_driver_onecore.h"
#endif

TTS_Windows *TTS_Windows::singleton = nullptr;

TTS_Windows *TTS_Windows::get_singleton() {
	return singleton;
}

bool TTS_Windows::is_speaking() const {
	if (driver) {
		return driver->is_speaking();
	}
	return false;
}

bool TTS_Windows::is_paused() const {
	if (driver) {
		return driver->is_paused();
	}
	return false;
}

Array TTS_Windows::get_voices() const {
	if (driver) {
		return driver->get_voices();
	}
	return Array();
}

void TTS_Windows::speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int64_t p_utterance_id, bool p_interrupt) {
	if (driver) {
		driver->speak(p_text, p_voice, p_volume, p_pitch, p_rate, p_utterance_id, p_interrupt);
	}
}

void TTS_Windows::pause() {
	if (driver) {
		driver->pause();
	}
}

void TTS_Windows::resume() {
	if (driver) {
		driver->resume();
	}
}

void TTS_Windows::stop() {
	if (driver) {
		driver->stop();
	}
}

void TTS_Windows::process_events() {
	if (driver) {
		driver->process_events();
	}
}

TTS_Windows::TTS_Windows() {
#ifdef WINRT_ENABLED
	// Try OneCore driver.
	if (!driver) {
		driver = memnew(TTSDriverOneCore);
		if (!driver->init()) {
			memdelete(driver);
			driver = nullptr;
		}
	}
#endif
	// Try SAPI driver.
	if (!driver) {
		driver = memnew(TTSDriverSAPI);
		if (!driver->init()) {
			memdelete(driver);
			driver = nullptr;
		}
	}
}

TTS_Windows::~TTS_Windows() {
	if (driver) {
		memdelete(driver);
	}
}
