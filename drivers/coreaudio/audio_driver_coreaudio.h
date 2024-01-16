/**************************************************************************/
/*  audio_driver_coreaudio.h                                              */
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

#ifndef AUDIO_DRIVER_COREAUDIO_H
#define AUDIO_DRIVER_COREAUDIO_H

#ifdef COREAUDIO_ENABLED

#include "servers/audio_server.h"

#include <AudioUnit/AudioUnit.h>
#ifdef OSX_ENABLED
#include <CoreAudio/AudioHardware.h>
#endif

class AudioDriverCoreAudio : public AudioDriver {
	AudioComponentInstance audio_unit;
	AudioComponentInstance input_unit;

	bool active;
	Mutex mutex;

	String device_name;
	String capture_device_name;

	int mix_rate;
	unsigned int channels;
	unsigned int capture_channels;
	unsigned int buffer_frames;

	Vector<int32_t> samples_in;
	Vector<int16_t> input_buf;

#ifdef OSX_ENABLED
	Array _get_device_list(bool capture = false);
	void _set_device(const String &device, bool capture = false);

	static OSStatus input_device_address_cb(AudioObjectID inObjectID,
			UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses,
			void *inClientData);

	static OSStatus output_device_address_cb(AudioObjectID inObjectID,
			UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses,
			void *inClientData);
#endif

	static OSStatus output_callback(void *inRefCon,
			AudioUnitRenderActionFlags *ioActionFlags,
			const AudioTimeStamp *inTimeStamp,
			UInt32 inBusNumber, UInt32 inNumberFrames,
			AudioBufferList *ioData);

	static OSStatus input_callback(void *inRefCon,
			AudioUnitRenderActionFlags *ioActionFlags,
			const AudioTimeStamp *inTimeStamp,
			UInt32 inBusNumber, UInt32 inNumberFrames,
			AudioBufferList *ioData);

	Error capture_init();
	void capture_finish();

public:
	const char *get_name() const {
		return "CoreAudio";
	};

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const;
	virtual SpeakerMode get_speaker_mode() const;

	virtual void lock();
	virtual void unlock();
	virtual void finish();

	virtual Error capture_start();
	virtual Error capture_stop();

	bool try_lock();
	void stop();

#ifdef OSX_ENABLED
	virtual Array get_device_list();
	virtual String get_device();
	virtual void set_device(String device);

	virtual Array capture_get_device_list();
	virtual void capture_set_device(const String &p_name);
	virtual String capture_get_device();
#endif

	AudioDriverCoreAudio();
	~AudioDriverCoreAudio();
};

#endif // COREAUDIO_ENABLED

#endif // AUDIO_DRIVER_COREAUDIO_H
