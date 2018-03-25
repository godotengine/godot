/*************************************************************************/
/*  audio_driver_coreaudio.cpp                                           */
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

#ifdef COREAUDIO_ENABLED

#include "audio_driver_coreaudio.h"
#include "core/project_settings.h"
#include "os/os.h"

#define kOutputBus 0

#ifdef OSX_ENABLED
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

Error AudioDriverCoreAudio::init_device() {
	AudioComponentDescription desc;
	zeromem(&desc, sizeof(desc));
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

	AudioStreamBasicDescription strdesc;

	zeromem(&strdesc, sizeof(strdesc));
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

	mix_rate = GLOBAL_DEF("audio/mix_rate", DEFAULT_MIX_RATE);

	zeromem(&strdesc, sizeof(strdesc));
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

	int latency = GLOBAL_DEF("audio/output_latency", DEFAULT_OUTPUT_LATENCY);
	// Sample rate is independent of channels (ref: https://stackoverflow.com/questions/11048825/audio-sample-frequency-rely-on-channels)
	buffer_frames = closest_power_of_2(latency * mix_rate / 1000);

#ifdef OSX_ENABLED
	result = AudioUnitSetProperty(audio_unit, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, kOutputBus, &buffer_frames, sizeof(UInt32));
	ERR_FAIL_COND_V(result != noErr, FAILED);
#endif

	buffer_size = buffer_frames * channels;
	samples_in.resize(buffer_size);

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("CoreAudio: detected " + itos(channels) + " channels");
		print_line("CoreAudio: audio buffer frames: " + itos(buffer_frames) + " calculated latency: " + itos(buffer_frames * 1000 / mix_rate) + "ms");
	}

	AURenderCallbackStruct callback;
	zeromem(&callback, sizeof(AURenderCallbackStruct));
	callback.inputProc = &AudioDriverCoreAudio::output_callback;
	callback.inputProcRefCon = this;
	result = AudioUnitSetProperty(audio_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, kOutputBus, &callback, sizeof(callback));
	ERR_FAIL_COND_V(result != noErr, FAILED);

	result = AudioUnitInitialize(audio_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	return OK;
}

Error AudioDriverCoreAudio::finish_device() {
	OSStatus result;

	if (active) {
		result = AudioOutputUnitStop(audio_unit);
		ERR_FAIL_COND_V(result != noErr, FAILED);

		active = false;
	}

	result = AudioUnitUninitialize(audio_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	return OK;
}

Error AudioDriverCoreAudio::init() {
	OSStatus result;

	mutex = Mutex::create();
	active = false;
	channels = 2;

#ifdef OSX_ENABLED
	AudioObjectPropertyAddress prop;
	prop.mSelector = kAudioHardwarePropertyDefaultOutputDevice;
	prop.mScope = kAudioObjectPropertyScopeGlobal;
	prop.mElement = kAudioObjectPropertyElementMaster;

	result = AudioObjectAddPropertyListener(kAudioObjectSystemObject, &prop, &output_device_address_cb, this);
	ERR_FAIL_COND_V(result != noErr, FAILED);
#endif

	return init_device();
};

OSStatus AudioDriverCoreAudio::output_callback(void *inRefCon,
		AudioUnitRenderActionFlags *ioActionFlags,
		const AudioTimeStamp *inTimeStamp,
		UInt32 inBusNumber, UInt32 inNumberFrames,
		AudioBufferList *ioData) {

	AudioDriverCoreAudio *ad = (AudioDriverCoreAudio *)inRefCon;

	if (!ad->active || !ad->try_lock()) {
		for (unsigned int i = 0; i < ioData->mNumberBuffers; i++) {
			AudioBuffer *abuf = &ioData->mBuffers[i];
			zeromem(abuf->mData, abuf->mDataByteSize);
		};
		return 0;
	};

	for (unsigned int i = 0; i < ioData->mNumberBuffers; i++) {

		AudioBuffer *abuf = &ioData->mBuffers[i];
		int frames_left = inNumberFrames;
		int16_t *out = (int16_t *)abuf->mData;

		while (frames_left) {

			int frames = MIN(frames_left, ad->buffer_frames);
			ad->audio_server_process(frames, ad->samples_in.ptrw());

			for (int j = 0; j < frames * ad->channels; j++) {

				out[j] = ad->samples_in[j] >> 16;
			}

			frames_left -= frames;
			out += frames * ad->channels;
		};
	};

	ad->unlock();

	return 0;
};

void AudioDriverCoreAudio::start() {
	if (!active) {
		OSStatus result = AudioOutputUnitStart(audio_unit);
		if (result != noErr) {
			ERR_PRINT(("AudioOutputUnitStart failed, code: " + itos(result)).utf8().get_data());
		} else {
			active = true;
		}
	}
};

void AudioDriverCoreAudio::stop() {
	if (active) {
		OSStatus result = AudioOutputUnitStop(audio_unit);
		if (result != noErr) {
			ERR_PRINT(("AudioOutputUnitStop failed, code: " + itos(result)).utf8().get_data());
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

#ifdef OSX_ENABLED

Array AudioDriverCoreAudio::get_device_list() {

	Array list;

	list.push_back("Default");

	AudioObjectPropertyAddress prop;

	prop.mSelector = kAudioHardwarePropertyDevices;
	prop.mScope = kAudioObjectPropertyScopeGlobal;
	prop.mElement = kAudioObjectPropertyElementMaster;

	UInt32 size = 0;
	AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &prop, 0, NULL, &size);
	AudioDeviceID *audioDevices = (AudioDeviceID *)malloc(size);
	AudioObjectGetPropertyData(kAudioObjectSystemObject, &prop, 0, NULL, &size, audioDevices);

	UInt32 deviceCount = size / sizeof(AudioDeviceID);
	for (UInt32 i = 0; i < deviceCount; i++) {
		prop.mScope = kAudioDevicePropertyScopeOutput;
		prop.mSelector = kAudioDevicePropertyStreamConfiguration;

		AudioObjectGetPropertyDataSize(audioDevices[i], &prop, 0, NULL, &size);
		AudioBufferList *bufferList = (AudioBufferList *)malloc(size);
		AudioObjectGetPropertyData(audioDevices[i], &prop, 0, NULL, &size, bufferList);

		UInt32 outputChannelCount = 0;
		for (UInt32 j = 0; j < bufferList->mNumberBuffers; j++)
			outputChannelCount += bufferList->mBuffers[j].mNumberChannels;

		free(bufferList);

		if (outputChannelCount >= 1) {
			CFStringRef cfname;

			size = sizeof(CFStringRef);
			prop.mSelector = kAudioObjectPropertyName;

			AudioObjectGetPropertyData(audioDevices[i], &prop, 0, NULL, &size, &cfname);

			CFIndex length = CFStringGetLength(cfname);
			CFIndex maxSize = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
			char *buffer = (char *)malloc(maxSize);
			if (CFStringGetCString(cfname, buffer, maxSize, kCFStringEncodingUTF8)) {
				// Append the ID to the name in case we have devices with duplicate name
				list.push_back(String(buffer) + " (" + itos(audioDevices[i]) + ")");
			}

			free(buffer);
		}
	}

	free(audioDevices);

	return list;
}

String AudioDriverCoreAudio::get_device() {

	return device_name;
}

void AudioDriverCoreAudio::set_device(String device) {

	device_name = device;
	if (!active) {
		return;
	}

	AudioDeviceID deviceId;
	bool found = false;
	if (device_name != "Default") {
		AudioObjectPropertyAddress prop;

		prop.mSelector = kAudioHardwarePropertyDevices;
		prop.mScope = kAudioObjectPropertyScopeGlobal;
		prop.mElement = kAudioObjectPropertyElementMaster;

		UInt32 size = 0;
		AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &prop, 0, NULL, &size);
		AudioDeviceID *audioDevices = (AudioDeviceID *)malloc(size);
		AudioObjectGetPropertyData(kAudioObjectSystemObject, &prop, 0, NULL, &size, audioDevices);

		UInt32 deviceCount = size / sizeof(AudioDeviceID);
		for (UInt32 i = 0; i < deviceCount && !found; i++) {
			prop.mScope = kAudioDevicePropertyScopeOutput;
			prop.mSelector = kAudioDevicePropertyStreamConfiguration;

			AudioObjectGetPropertyDataSize(audioDevices[i], &prop, 0, NULL, &size);
			AudioBufferList *bufferList = (AudioBufferList *)malloc(size);
			AudioObjectGetPropertyData(audioDevices[i], &prop, 0, NULL, &size, bufferList);

			UInt32 outputChannelCount = 0;
			for (UInt32 j = 0; j < bufferList->mNumberBuffers; j++)
				outputChannelCount += bufferList->mBuffers[j].mNumberChannels;

			free(bufferList);

			if (outputChannelCount >= 1) {
				CFStringRef cfname;

				size = sizeof(CFStringRef);
				prop.mSelector = kAudioObjectPropertyName;

				AudioObjectGetPropertyData(audioDevices[i], &prop, 0, NULL, &size, &cfname);

				CFIndex length = CFStringGetLength(cfname);
				CFIndex maxSize = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
				char *buffer = (char *)malloc(maxSize);
				if (CFStringGetCString(cfname, buffer, maxSize, kCFStringEncodingUTF8)) {
					String name = String(buffer) + " (" + itos(audioDevices[i]) + ")";
					if (name == device_name) {
						deviceId = audioDevices[i];
						found = true;
					}
				}

				free(buffer);
			}
		}

		free(audioDevices);
	}

	if (!found) {
		UInt32 size = sizeof(AudioDeviceID);
		AudioObjectPropertyAddress property = { kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMaster };

		OSStatus result = AudioObjectGetPropertyData(kAudioObjectSystemObject, &property, 0, NULL, &size, &deviceId);
		ERR_FAIL_COND(result != noErr);

		found = true;
	}

	if (found) {
		OSStatus result = AudioUnitSetProperty(audio_unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &deviceId, sizeof(AudioDeviceID));
		ERR_FAIL_COND(result != noErr);
	}
}

#endif

void AudioDriverCoreAudio::lock() {
	if (mutex)
		mutex->lock();
};

void AudioDriverCoreAudio::unlock() {
	if (mutex)
		mutex->unlock();
};

bool AudioDriverCoreAudio::try_lock() {
	if (mutex)
		return mutex->try_lock() == OK;
	return true;
}

void AudioDriverCoreAudio::finish() {
	OSStatus result;

	finish_device();

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

	AURenderCallbackStruct callback;
	zeromem(&callback, sizeof(AURenderCallbackStruct));
	result = AudioUnitSetProperty(audio_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, kOutputBus, &callback, sizeof(callback));
	if (result != noErr) {
		ERR_PRINT("AudioUnitSetProperty failed");
	}

	if (mutex) {
		memdelete(mutex);
		mutex = NULL;
	}
};

AudioDriverCoreAudio::AudioDriverCoreAudio() {
	active = false;
	mutex = NULL;

	mix_rate = 0;
	channels = 2;

	buffer_size = 0;
	buffer_frames = 0;

	samples_in.clear();

	device_name = "Default";
};

AudioDriverCoreAudio::~AudioDriverCoreAudio(){};

#endif
