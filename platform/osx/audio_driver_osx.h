/*************************************************************************/
/*  audio_driver_osx.h                                                   */
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
#ifdef OSX_ENABLED

#ifndef AUDIO_DRIVER_OSX_H
#define AUDIO_DRIVER_OSX_H

#include "servers/audio_server.h"

#include <AudioUnit/AudioUnit.h>

class AudioDriverOSX : public AudioDriver {

	AudioComponentInstance audio_unit;
	bool active;
	Mutex *mutex;

	int channels;
	int32_t *samples_in;
	int buffer_frames;

	static OSStatus output_callback(void *inRefCon,
			AudioUnitRenderActionFlags *ioActionFlags,
			const AudioTimeStamp *inTimeStamp,
			UInt32 inBusNumber, UInt32 inNumberFrames,
			AudioBufferList *ioData);

public:
	const char *get_name() const {
		return "AudioUnit";
	};

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const;
	virtual SpeakerMode get_speaker_mode() const;
	virtual void lock();
	virtual void unlock();
	virtual void finish();

	AudioDriverOSX();
	~AudioDriverOSX();
};

#endif

#endif
