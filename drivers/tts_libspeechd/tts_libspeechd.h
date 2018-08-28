/*************************************************************************/
/*  tts_libspeechd.h                                                     */
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

#ifndef TTS_SPD_H
#define TTS_SPD_H

#ifdef SPDTTS_ENABLED

#include "os/tts_driver.h"

#include <speech-dispatcher/libspeechd.h>

#include "core/list.h"

class TTSDriverSPD : public TTSDriver {

	static List<int> messages;
	SPDConnection *synth;

protected:
	static void end_of_speech(size_t msg_id, size_t client_id, SPDNotificationType type);

public:
	void speak(const String &p_text, bool p_interrupt);
	void stop();

	bool is_speaking();

	Array get_voices();
	void set_voice(const String &p_voice);

	void set_volume(int p_volume);
	int get_volume();

	void set_rate(int p_rate);
	int get_rate();

	TTSDriverSPD();
	~TTSDriverSPD();
};

#endif
#endif
