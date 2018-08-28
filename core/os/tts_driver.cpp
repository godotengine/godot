/*************************************************************************/
/*  tts_driver.cpp                                                       */
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

#include "tts_driver.h"

TTSDriver *TTSDriver::singleton = NULL;
TTSDriver *TTSDriver::get_singleton() {

	return singleton;
};

void TTSDriver::set_singleton() {

	singleton = this;
};

void TTSDriver::speak(const String &p_text, bool p_interrupt) {

	WARN_PRINT("Text-to-speech is not implemented on this platform");
};

void TTSDriver::stop() {

	WARN_PRINT("Text-to-speech is not implemented on this platform");
};

bool TTSDriver::is_speaking() {

	WARN_PRINT("Text-to-speech is not implemented on this platform");
	return false;
};

Array TTSDriver::get_voices() {

	Array list;
	WARN_PRINT("Text-to-speech is not implemented on this platform");
	return list;
};

void TTSDriver::set_voice(const String &p_voice) {

	WARN_PRINT("Text-to-speech is not implemented on this platform");
};

void TTSDriver::set_volume(int p_volume) {

	WARN_PRINT("Text-to-speech is not implemented on this platform");
};

int TTSDriver::get_volume() {

	WARN_PRINT("Text-to-speech is not implemented on this platform");
	return 0;
};

void TTSDriver::set_rate(int p_rate) {

	WARN_PRINT("Text-to-speech is not implemented on this platform");
};

int TTSDriver::get_rate() {

	WARN_PRINT("Text-to-speech is not implemented on this platform");
	return 0;
};

TTSDriver::TTSDriver() {

	set_singleton();
};
