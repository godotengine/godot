/*************************************************************************/
/*  audio_driver_coreaudio.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef COREAUDIO_ENABLED

#include "audio_driver_coreaudio.h"

#include "core/os/os.h"
#include "core/project_settings.h"

#define kOutputBus 0
#define kInputBus 1

#ifdef OSX_ENABLED
OSStatus AudioDriverCoreAudio::input_device_address_cb(AudioObjectID inObjectID,
		UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses,
		void *inClientData) {
	AudioDriverCoreAudio *driver = (AudioDriverCoreAudio *)inClientData;

	// If our selected device is the Default call set_device to update the
	// kAudioOutputUnitProperty_CurrentDevice property
	if (driver->capture_device_name == "Default") {
		driver->capture_set_device("Default");
	}

	return noErr;
}

OSStatus AudioDriverCoreAudio::output_device_address_cb(AudioObjectID inObjectID,
		UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses,
		void *inClientData) {
	AudioDriverCoreAudio *driver = (AudioDriverCoreAudio *)inClientData;

	// If our selected device is the Default call set_device to update the
	// kAudioOutputUnitProperty_CurrentDevice property
	if (driver->device_name == "Default") {
		driver->set_device("Default");
	}

	return noErr;
}
#endif

Error AudioDriverCoreAudio::init() {
	AudioComponentDescription desc;
	memset(&desc, 0, sizeof(desc));
	desc.componentType = kAudioUnitType_Output;
#ifdef OSX_ENABLED
	desc.componentSubType = kAudioUnitSubType_HALOutput;
#else
	desc.componentSubType = kAudioUnitSubType_RemoteIO;
#endif
	desc.componentManufacturer = kAudioUnitManufacturer_Apple;

	AudioComponent comp = AudioComponentFindNext(NULL, &desc);
	ERR_FAIL_COND_V(comp == NULL, FAILED);

	OSStatus result = AudioComponentInstanceNew(comp, &audio_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);

#ifdef OSX_ENABLED
	AudioObjectPropertyAddress prop;
	prop.mSelector = kAudioHardwarePropertyDefaultOutputDevice;
	prop.mScope = kAudioObjectPropertyScopeGlobal;
	prop.mElement = kAudioObjectPropertyElementMaster;

	result = AudioObjectAddPropertyListener(kAudioObjectSystemObject, &prop, &output_device_address_cb, this);
	ERR_FAIL_COND_V(result != noErr, FAILED);
#endif

	AudioStreamBasicDescription strdesc;

	memset(&strdesc, 0, sizeof(strdesc));
	UInt32 size = sizeof(strdesc);
	result = AudioUnitGetProperty(audio_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, kOutputBus, &strdesc, &size);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	switch (strdesc.mChannelsPerFrame) {
		case 2: // Stereo
		case 4: // Surround 3.1
		case 6: // Surround 5.1
		case 8: // Surround 7.1
			channels = strdesc.mChannelsPerFrame;
			break;

		default:
			// Unknown number of channels, default to stereo
			channels = 2;
			break;
	}

	mix_rate = GLOBAL_GET("audio/mix_rate");

	memset(&strdesc, 0, sizeof(strdesc));
	strdesc.mFormatID = kAudioFormatLinearPCM;
	strdesc.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
	strdesc.mChannelsPerFrame = channels;
	strdesc.mSampleRate = mix_rate;
	strdesc.mFramesPerPacket = 1;
	strdesc.mBitsPerChannel = 16;
	strdesc.mBytesPerFrame = strdesc.mBitsPerChannel * strdesc.mChannelsPerFrame / 8;
	strdesc.mBytesPerPacket = strdesc.mBytesPerFrame * strdesc.mFramesPerPacket;

	result = AudioUnitSetProperty(audio_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, kOutputBus, &strdesc, sizeof(strdesc));
	ERR_FAIL_COND_V(result != noErr, FAILED);

	int latency = GLOBAL_GET("audio/output_latency");
	// Sample rate is independent of channels (ref: https://stackoverflow.com/questions/11048825/audio-sample-frequency-rely-on-channels)
	buffer_frames = closest_power_of_2(latency * mix_rate / 1000);

#ifdef OSX_ENABLED
	result = AudioUnitSetProperty(audio_unit, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, kOutputBus, &buffer_frames, sizeof(UInt32));
	ERR_FAIL_COND_V(result != noErr, FAILED);
#endif

	unsigned int buffer_size = buffer_frames * channels;
	samples_in.resize(buffer_size);
	input_buf.resize(buffer_size);

	print_verbose("CoreAudio: detected " + itos(channels) + " channels");
	print_verbose("CoreAudio: audio buffer frames: " + itos(buffer_frames) + " calculated latency: " + itos(buffer_frames * 1000 / mix_rate) + "ms");

	AURenderCallbackStruct callback;
	memset(&callback, 0, sizeof(AURenderCallbackStruct));
	callback.inputProc = &AudioDriverCoreAudio::output_callback;
	callback.inputProcRefCon = this;
	result = AudioUnitSetProperty(audio_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, kOutputBus, &callback, sizeof(callback));
	ERR_FAIL_COND_V(result != noErr, FAILED);

	result = AudioUnitInitialize(audio_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	if (GLOBAL_GET("audio/enable_audio_input")) {
		return capture_init();
	}
	return OK;
}

OSStatus AudioDriverCoreAudio::output_callback(void *inRefCon,
		AudioUnitRenderActionFlags *ioActionFlags,
		const AudioTimeStamp *inTimeStamp,
		UInt32 inBusNumber, UInt32 inNumberFrames,
		AudioBufferList *ioData) {
	AudioDriverCoreAudio *ad = (AudioDriverCoreAudio *)inRefCon;

	if (!ad->active || !ad->try_lock()) {
		for (unsigned int i = 0; i < ioData->mNumberBuffers; i++) {
			AudioBuffer *abuf = &ioData->mBuffers[i];
			memset(abuf->mData, 0, abuf->mDataByteSize);
		};
		return 0;
	};

	ad->start_counting_ticks();

	for (unsigned int i = 0; i < ioData->mNumberBuffers; i++) {
		AudioBuffer *abuf = &ioData->mBuffers[i];
		unsigned int frames_left = inNumberFrames;
		int16_t *out = (int16_t *)abuf->mData;

		while (frames_left) {
			unsigned int frames = MIN(frames_left, ad->buffer_frames);
			ad->audio_server_process(frames, ad->samples_in.ptrw());

			for (unsigned int j = 0; j < frames * ad->channels; j++) {
				out[j] = ad->samples_in[j] >> 16;
			}

			frames_left -= frames;
			out += frames * ad->channels;
		};
	};

	ad->stop_counting_ticks();
	ad->unlock();

	return 0;
};

OSStatus AudioDriverCoreAudio::input_callback(void *inRefCon,
		AudioUnitRenderActionFlags *ioActionFlags,
		const AudioTimeStamp *inTimeStamp,
		UInt32 inBusNumber, UInt32 inNumberFrames,
		AudioBufferList *ioData) {
	AudioDriverCoreAudio *ad = (AudioDriverCoreAudio *)inRefCon;
	if (!ad->active) {
		return 0;
	}

	ad->lock();

	AudioBufferList bufferList;
	bufferList.mNumberBuffers = 1;
	bufferList.mBuffers[0].mData = ad->input_buf.ptrw();
	bufferList.mBuffers[0].mNumberChannels = ad->capture_channels;
	bufferList.mBuffers[0].mDataByteSize = ad->input_buf.size() * sizeof(int16_t);

	OSStatus result = AudioUnitRender(ad->input_unit, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, &bufferList);
	if (result == noErr) {
		for (unsigned int i = 0; i < inNumberFrames * ad->capture_channels; i++) {
			int32_t sample = ad->input_buf[i] << 16;
			ad->input_buffer_write(sample);

			if (ad->capture_channels == 1) {
				// In case input device is single channel convert it to Stereo
				ad->input_buffer_write(sample);
			}
		}
	} else {
		ERR_PRINT("AudioUnitRender failed, code: " + itos(result));
	}

	ad->unlock();

	return result;
}

void AudioDriverCoreAudio::start() {
	if (!active) {
		OSStatus result = AudioOutputUnitStart(audio_unit);
		if (result != noErr) {
			ERR_PRINT("AudioOutputUnitStart failed, code: " + itos(result));
		} else {
			active = true;
		}
	}
};

void AudioDriverCoreAudio::stop() {
	if (active) {
		OSStatus result = AudioOutputUnitStop(audio_unit);
		if (result != noErr) {
			ERR_PRINT("AudioOutputUnitStop failed, code: " + itos(result));
		} else {
			active = false;
		}
	}
}

int AudioDriverCoreAudio::get_mix_rate() const {
	return mix_rate;
};

AudioDriver::SpeakerMode AudioDriverCoreAudio::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(channels);
};

void AudioDriverCoreAudio::lock() {
	mutex.lock();
};

void AudioDriverCoreAudio::unlock() {
	mutex.unlock();
};

bool AudioDriverCoreAudio::try_lock() {
	return mutex.try_lock() == OK;
}

void AudioDriverCoreAudio::finish() {
	capture_finish();

	if (audio_unit) {
		OSStatus result;

		lock();

		AURenderCallbackStruct callback;
		memset(&callback, 0, sizeof(AURenderCallbackStruct));
		result = AudioUnitSetProperty(audio_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, kOutputBus, &callback, sizeof(callback));
		if (result != noErr) {
			ERR_PRINT("AudioUnitSetProperty failed");
		}

		if (active) {
			result = AudioOutputUnitStop(audio_unit);
			if (result != noErr) {
				ERR_PRINT("AudioOutputUnitStop failed");
			}

			active = false;
		}

		result = AudioUnitUninitialize(audio_unit);
		if (result != noErr) {
			ERR_PRINT("AudioUnitUninitialize failed");
		}

#ifdef OSX_ENABLED
		AudioObjectPropertyAddress prop;
		prop.mSelector = kAudioHardwarePropertyDefaultOutputDevice;
		prop.mScope = kAudioObjectPropertyScopeGlobal;
		prop.mElement = kAudioObjectPropertyElementMaster;

		result = AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &prop, &output_device_address_cb, this);
		if (result != noErr) {
			ERR_PRINT("AudioObjectRemovePropertyListener failed");
		}
#endif

		result = AudioComponentInstanceDispose(audio_unit);
		if (result != noErr) {
			ERR_PRINT("AudioComponentInstanceDispose failed");
		}

		audio_unit = NULL;
		unlock();
	}
}

Error AudioDriverCoreAudio::capture_init() {
	AudioComponentDescription desc;
	memset(&desc, 0, sizeof(desc));
	desc.componentType = kAudioUnitType_Output;
#ifdef OSX_ENABLED
	desc.componentSubType = kAudioUnitSubType_HALOutput;
#else
	desc.componentSubType = kAudioUnitSubType_RemoteIO;
#endif
	desc.componentManufacturer = kAudioUnitManufacturer_Apple;

	AudioComponent comp = AudioComponentFindNext(NULL, &desc);
	ERR_FAIL_COND_V(comp == NULL, FAILED);

	OSStatus result = AudioComponentInstanceNew(comp, &input_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);

#ifdef OSX_ENABLED
	AudioObjectPropertyAddress prop;
	prop.mSelector = kAudioHardwarePropertyDefaultInputDevice;
	prop.mScope = kAudioObjectPropertyScopeGlobal;
	prop.mElement = kAudioObjectPropertyElementMaster;

	result = AudioObjectAddPropertyListener(kAudioObjectSystemObject, &prop, &input_device_address_cb, this);
	ERR_FAIL_COND_V(result != noErr, FAILED);
#endif

	UInt32 flag = 1;
	result = AudioUnitSetProperty(input_unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, kInputBus, &flag, sizeof(flag));
	ERR_FAIL_COND_V(result != noErr, FAILED);
	flag = 0;
	result = AudioUnitSetProperty(input_unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, kOutputBus, &flag, sizeof(flag));
	ERR_FAIL_COND_V(result != noErr, FAILED);

	UInt32 size;
#ifdef OSX_ENABLED
	AudioDeviceID deviceId;
	size = sizeof(AudioDeviceID);
	AudioObjectPropertyAddress property = { kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMaster };

	result = AudioObjectGetPropertyData(kAudioObjectSystemObject, &property, 0, NULL, &size, &deviceId);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	result = AudioUnitSetProperty(input_unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &deviceId, sizeof(AudioDeviceID));
	ERR_FAIL_COND_V(result != noErr, FAILED);
#endif

	AudioStreamBasicDescription strdesc;
	memset(&strdesc, 0, sizeof(strdesc));
	size = sizeof(strdesc);
	result = AudioUnitGetProperty(input_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, kInputBus, &strdesc, &size);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	switch (strdesc.mChannelsPerFrame) {
		case 1: // Mono
			capture_channels = 1;
			break;

		case 2: // Stereo
			capture_channels = 2;
			break;

		default:
			// Unknown number of channels, default to stereo
			capture_channels = 2;
			break;
	}

	mix_rate = GLOBAL_GET("audio/mix_rate");

	memset(&strdesc, 0, sizeof(strdesc));
	strdesc.mFormatID = kAudioFormatLinearPCM;
	strdesc.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
	strdesc.mChannelsPerFrame = capture_channels;
	strdesc.mSampleRate = mix_rate;
	strdesc.mFramesPerPacket = 1;
	strdesc.mBitsPerChannel = 16;
	strdesc.mBytesPerFrame = strdesc.mBitsPerChannel * strdesc.mChannelsPerFrame / 8;
	strdesc.mBytesPerPacket = strdesc.mBytesPerFrame * strdesc.mFramesPerPacket;

	result = AudioUnitSetProperty(input_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, kInputBus, &strdesc, sizeof(strdesc));
	ERR_FAIL_COND_V(result != noErr, FAILED);

	AURenderCallbackStruct callback;
	memset(&callback, 0, sizeof(AURenderCallbackStruct));
	callback.inputProc = &AudioDriverCoreAudio::input_callback;
	callback.inputProcRefCon = this;
	result = AudioUnitSetProperty(input_unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, kInputBus, &callback, sizeof(callback));
	ERR_FAIL_COND_V(result != noErr, FAILED);

	result = AudioUnitInitialize(input_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	return OK;
}

void AudioDriverCoreAudio::capture_finish() {
	if (input_unit) {
		lock();

		AURenderCallbackStruct callback;
		memset(&callback, 0, sizeof(AURenderCallbackStruct));
		OSStatus result = AudioUnitSetProperty(input_unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0, &callback, sizeof(callback));
		if (result != noErr) {
			ERR_PRINT("AudioUnitSetProperty failed");
		}

		result = AudioUnitUninitialize(input_unit);
		if (result != noErr) {
			ERR_PRINT("AudioUnitUninitialize failed");
		}

#ifdef OSX_ENABLED
		AudioObjectPropertyAddress prop;
		prop.mSelector = kAudioHardwarePropertyDefaultInputDevice;
		prop.mScope = kAudioObjectPropertyScopeGlobal;
		prop.mElement = kAudioObjectPropertyElementMaster;

		result = AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &prop, &input_device_address_cb, this);
		if (result != noErr) {
			ERR_PRINT("AudioObjectRemovePropertyListener failed");
		}
#endif

		result = AudioComponentInstanceDispose(input_unit);
		if (result != noErr) {
			ERR_PRINT("AudioComponentInstanceDispose failed");
		}

		input_unit = NULL;
		unlock();
	}
}

Error AudioDriverCoreAudio::capture_start() {
	input_buffer_init(buffer_frames);

	OSStatus result = AudioOutputUnitStart(input_unit);
	if (result != noErr) {
		ERR_PRINT("AudioOutputUnitStart failed, code: " + itos(result));
	}

	return OK;
}

Error AudioDriverCoreAudio::capture_stop() {
	if (input_unit) {
		OSStatus result = AudioOutputUnitStop(input_unit);
		if (result != noErr) {
			ERR_PRINT("AudioOutputUnitStop failed, code: " + itos(result));
		}
	}

	return OK;
}

#ifdef OSX_ENABLED

Array AudioDriverCoreAudio::_get_device_list(bool capture) {
	Array list;

	list.push_back("Default");

	AudioObjectPropertyAddress prop;

	prop.mSelector = kAudioHardwarePropertyDevices;
	prop.mScope = kAudioObjectPropertyScopeGlobal;
	prop.mElement = kAudioObjectPropertyElementMaster;

	UInt32 size = 0;
	AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &prop, 0, NULL, &size);
	AudioDeviceID *audioDevices = (AudioDeviceID *)memalloc(size);
	ERR_FAIL_NULL_V_MSG(audioDevices, list, "Out of memory.");
	AudioObjectGetPropertyData(kAudioObjectSystemObject, &prop, 0, NULL, &size, audioDevices);

	UInt32 deviceCount = size / sizeof(AudioDeviceID);
	for (UInt32 i = 0; i < deviceCount; i++) {
		prop.mScope = capture ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput;
		prop.mSelector = kAudioDevicePropertyStreamConfiguration;

		AudioObjectGetPropertyDataSize(audioDevices[i], &prop, 0, NULL, &size);
		AudioBufferList *bufferList = (AudioBufferList *)memalloc(size);
		ERR_FAIL_NULL_V_MSG(bufferList, list, "Out of memory.");
		AudioObjectGetPropertyData(audioDevices[i], &prop, 0, NULL, &size, bufferList);

		UInt32 channelCount = 0;
		for (UInt32 j = 0; j < bufferList->mNumberBuffers; j++)
			channelCount += bufferList->mBuffers[j].mNumberChannels;

		memfree(bufferList);

		if (channelCount >= 1) {
			CFStringRef cfname;

			size = sizeof(CFStringRef);
			prop.mSelector = kAudioObjectPropertyName;

			AudioObjectGetPropertyData(audioDevices[i], &prop, 0, NULL, &size, &cfname);

			CFIndex length = CFStringGetLength(cfname);
			CFIndex maxSize = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
			char *buffer = (char *)memalloc(maxSize);
			ERR_FAIL_NULL_V_MSG(buffer, list, "Out of memory.");
			if (CFStringGetCString(cfname, buffer, maxSize, kCFStringEncodingUTF8)) {
				// Append the ID to the name in case we have devices with duplicate name
				list.push_back(String::utf8(buffer) + " (" + itos(audioDevices[i]) + ")");
			}

			memfree(buffer);
		}
	}

	memfree(audioDevices);

	return list;
}

void AudioDriverCoreAudio::_set_device(const String &device, bool capture) {
	AudioDeviceID deviceId;
	bool found = false;
	if (device != "Default") {
		AudioObjectPropertyAddress prop;

		prop.mSelector = kAudioHardwarePropertyDevices;
		prop.mScope = kAudioObjectPropertyScopeGlobal;
		prop.mElement = kAudioObjectPropertyElementMaster;

		UInt32 size = 0;
		AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &prop, 0, NULL, &size);
		AudioDeviceID *audioDevices = (AudioDeviceID *)memalloc(size);
		ERR_FAIL_NULL_MSG(audioDevices, "Out of memory.");
		AudioObjectGetPropertyData(kAudioObjectSystemObject, &prop, 0, NULL, &size, audioDevices);

		UInt32 deviceCount = size / sizeof(AudioDeviceID);
		for (UInt32 i = 0; i < deviceCount && !found; i++) {
			prop.mScope = capture ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput;
			prop.mSelector = kAudioDevicePropertyStreamConfiguration;

			AudioObjectGetPropertyDataSize(audioDevices[i], &prop, 0, NULL, &size);
			AudioBufferList *bufferList = (AudioBufferList *)memalloc(size);
			ERR_FAIL_NULL_MSG(bufferList, "Out of memory.");
			AudioObjectGetPropertyData(audioDevices[i], &prop, 0, NULL, &size, bufferList);

			UInt32 channelCount = 0;
			for (UInt32 j = 0; j < bufferList->mNumberBuffers; j++)
				channelCount += bufferList->mBuffers[j].mNumberChannels;

			memfree(bufferList);

			if (channelCount >= 1) {
				CFStringRef cfname;

				size = sizeof(CFStringRef);
				prop.mSelector = kAudioObjectPropertyName;

				AudioObjectGetPropertyData(audioDevices[i], &prop, 0, NULL, &size, &cfname);

				CFIndex length = CFStringGetLength(cfname);
				CFIndex maxSize = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
				char *buffer = (char *)memalloc(maxSize);
				ERR_FAIL_NULL_MSG(buffer, "Out of memory.");
				if (CFStringGetCString(cfname, buffer, maxSize, kCFStringEncodingUTF8)) {
					String name = String::utf8(buffer) + " (" + itos(audioDevices[i]) + ")";
					if (name == device) {
						deviceId = audioDevices[i];
						found = true;
					}
				}

				memfree(buffer);
			}
		}

		memfree(audioDevices);
	}

	if (!found) {
		// If we haven't found the desired device get the system default one
		UInt32 size = sizeof(AudioDeviceID);
		UInt32 elem = capture ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice;
		AudioObjectPropertyAddress property = { elem, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMaster };

		OSStatus result = AudioObjectGetPropertyData(kAudioObjectSystemObject, &property, 0, NULL, &size, &deviceId);
		ERR_FAIL_COND(result != noErr);

		found = true;
	}

	if (found) {
		OSStatus result = AudioUnitSetProperty(capture ? input_unit : audio_unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &deviceId, sizeof(AudioDeviceID));
		ERR_FAIL_COND(result != noErr);

		if (capture) {
			// Reset audio input to keep synchronisation.
			input_position = 0;
			input_size = 0;
		}
	}
}

Array AudioDriverCoreAudio::get_device_list() {
	return _get_device_list();
}

String AudioDriverCoreAudio::get_device() {
	return device_name;
}

void AudioDriverCoreAudio::set_device(String device) {
	device_name = device;
	if (active) {
		_set_device(device_name);
	}
}

void AudioDriverCoreAudio::capture_set_device(const String &p_name) {
	capture_device_name = p_name;
	if (active) {
		_set_device(capture_device_name, true);
	}
}

Array AudioDriverCoreAudio::capture_get_device_list() {
	return _get_device_list(true);
}

String AudioDriverCoreAudio::capture_get_device() {
	return capture_device_name;
}

#endif

AudioDriverCoreAudio::AudioDriverCoreAudio() :
		audio_unit(NULL),
		input_unit(NULL),
		active(false),
		device_name("Default"),
		capture_device_name("Default"),
		mix_rate(0),
		channels(2),
		capture_channels(2),
		buffer_frames(0) {
	samples_in.clear();
}

AudioDriverCoreAudio::~AudioDriverCoreAudio(){};

#endif
