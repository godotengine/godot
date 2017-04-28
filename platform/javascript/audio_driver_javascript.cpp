/*************************************************************************/
/*  audio_driver_javascript.cpp                                          */
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
#include "audio_driver_javascript.h"

#include <string.h>

#define MAX_NUMBER_INTERFACES 3
#define MAX_NUMBER_OUTPUT_DEVICES 6

/* Structure for passing information to callback function */

//AudioDriverJavaScript* AudioDriverJavaScript::s_ad=NULL;

const char *AudioDriverJavaScript::get_name() const {

	return "JavaScript";
}

Error AudioDriverJavaScript::init() {

	return OK;
}

void AudioDriverJavaScript::start() {
}

int AudioDriverJavaScript::get_mix_rate() const {

	return 44100;
}

AudioDriver::SpeakerMode AudioDriverJavaScript::get_speaker_mode() const {

	return SPEAKER_MODE_STEREO;
}

void AudioDriverJavaScript::lock() {

	/*
	if (active && mutex)
		mutex->lock();
	*/
}

void AudioDriverJavaScript::unlock() {

	/*
	if (active && mutex)
		mutex->unlock();
	*/
}

void AudioDriverJavaScript::finish() {
}

AudioDriverJavaScript::AudioDriverJavaScript() {
}
