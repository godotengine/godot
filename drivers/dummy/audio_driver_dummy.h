/*************************************************************************/
/*  audio_driver_dummy.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef AUDIO_DRIVER_DUMMY_H
#define AUDIO_DRIVER_DUMMY_H

#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "servers/audio_server.h"

class AudioDriverDummy : public AudioDriver {
public:
	const char *get_name() const {
		return "Dummy";
	};

	virtual Error init() { return OK; }
	virtual void start(){};
	virtual int get_mix_rate() const { return DEFAULT_MIX_RATE; };
	virtual SpeakerMode get_speaker_mode() const { return SPEAKER_MODE_STEREO; };
	virtual void lock(){};
	virtual void unlock(){};
	virtual void finish(){};

	virtual float get_latency() { return 0; };

	AudioDriverDummy(){};
	~AudioDriverDummy(){};
};

#endif // AUDIO_DRIVER_DUMMY_H
