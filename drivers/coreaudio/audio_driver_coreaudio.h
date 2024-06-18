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

#import <AudioUnit/AudioUnit.h>
#ifdef MACOS_ENABLED
#import <CoreAudio/AudioHardware.h>
#endif

class AudioDriverCoreAudio : public AudioDriver {
	AudioComponentInstance audio_unit = nullptr;
	AudioComponentInstance input_unit = nullptr;

	bool active = false;
	Mutex mutex;

	String output_device_name = "Default";
	String input_device_name = "Default";

	int mix_rate = 0;
	unsigned int buffer_frames = 0;

	unsigned int output_channels = 2;
	unsigned int input_channels = 2;

	BufferFormat output_buffer_format = NO_BUFFER;
	BufferFormat input_buffer_format = NO_BUFFER;

	LocalVector<int8_t> input_buffer;

#ifdef MACOS_ENABLED
	PackedStringArray _get_device_list(bool capture = false);
	void _set_device(const String &output_device, bool capture = false);

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

	Error init_input_device();
	void finish_input_device();

public:
	virtual const char *get_name() const override {
		return "CoreAudio";
	};

	virtual Error init() override;
	virtual void start() override;
	virtual int get_mix_rate() const override;

	virtual void lock() override;
	virtual void unlock() override;
	virtual void finish() override;

	virtual int get_output_channels() const override;
	virtual BufferFormat get_output_buffer_format() const override;

	virtual int get_input_channels() const override;
	virtual BufferFormat get_input_buffer_format() const override;

#ifdef MACOS_ENABLED
	virtual PackedStringArray get_output_device_list() override;
	virtual String get_output_device() override;
	virtual void set_output_device(const String &p_name) override;

	virtual PackedStringArray get_input_device_list() override;
	virtual String get_input_device() override;
	virtual void set_input_device(const String &p_name) override;
#endif

	virtual Error input_start() override;
	virtual Error input_stop() override;

	bool try_lock();
	void stop();
};

#endif // COREAUDIO_ENABLED

#endif // AUDIO_DRIVER_COREAUDIO_H
