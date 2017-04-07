/*************************************************************************/
/*  audio_driver_iphone.cpp                                              */
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
#include "audio_driver_iphone.h"

Error AudioDriverIphone::init() {

	active = false;
	channels = 2;

	AudioStreamBasicDescription strdesc;
	strdesc.mFormatID = kAudioFormatLinearPCM;
	strdesc.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
	strdesc.mChannelsPerFrame = channels;
	strdesc.mSampleRate = 44100;
	strdesc.mFramesPerPacket = 1;
	strdesc.mBitsPerChannel = 16;
	strdesc.mBytesPerFrame =
			strdesc.mBitsPerChannel * strdesc.mChannelsPerFrame / 8;
	strdesc.mBytesPerPacket =
			strdesc.mBytesPerFrame * strdesc.mFramesPerPacket;

	OSStatus result = noErr;
	AURenderCallbackStruct callback;
	AudioComponentDescription desc;
	AudioComponent comp = NULL;
	const AudioUnitElement output_bus = 0;
	const AudioUnitElement bus = output_bus;
	const AudioUnitScope scope = kAudioUnitScope_Input;

	zeromem(&desc, sizeof(desc));
	desc.componentType = kAudioUnitType_Output;
	desc.componentSubType = kAudioUnitSubType_RemoteIO; /* !!! FIXME: ? */
	comp = AudioComponentFindNext(NULL, &desc);
	desc.componentManufacturer = kAudioUnitManufacturer_Apple;

	result = AudioComponentInstanceNew(comp, &audio_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);
	ERR_FAIL_COND_V(comp == NULL, FAILED);

	result = AudioUnitSetProperty(audio_unit,
			kAudioUnitProperty_StreamFormat,
			scope, bus, &strdesc, sizeof(strdesc));
	ERR_FAIL_COND_V(result != noErr, FAILED);

	zeromem(&callback, sizeof(AURenderCallbackStruct));
	callback.inputProc = &AudioDriverIphone::output_callback;
	callback.inputProcRefCon = this;
	result = AudioUnitSetProperty(audio_unit,
			kAudioUnitProperty_SetRenderCallback,
			scope, bus, &callback, sizeof(callback));
	ERR_FAIL_COND_V(result != noErr, FAILED);

	result = AudioUnitInitialize(audio_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	result = AudioOutputUnitStart(audio_unit);
	ERR_FAIL_COND_V(result != noErr, FAILED);

	const int samples = 1024;
	samples_in = memnew_arr(int32_t, samples); // whatever
	buffer_frames = samples / channels;

	return FAILED;
};

OSStatus AudioDriverIphone::output_callback(void *inRefCon,
		AudioUnitRenderActionFlags *ioActionFlags,
		const AudioTimeStamp *inTimeStamp,
		UInt32 inBusNumber, UInt32 inNumberFrames,
		AudioBufferList *ioData) {

	AudioBuffer *abuf;
	AudioDriverIphone *ad = (AudioDriverIphone *)inRefCon;

	bool mix = true;

	if (!ad->active)
		mix = false;
	else if (ad->mutex) {
		mix = ad->mutex->try_lock() == OK;
	};

	if (!mix) {
		for (unsigned int i = 0; i < ioData->mNumberBuffers; i++) {
			abuf = &ioData->mBuffers[i];
			zeromem(abuf->mData, abuf->mDataByteSize);
		};
		return 0;
	};

	int frames_left;

	for (unsigned int i = 0; i < ioData->mNumberBuffers; i++) {

		abuf = &ioData->mBuffers[i];
		frames_left = inNumberFrames;
		int16_t *out = (int16_t *)abuf->mData;

		while (frames_left) {

			int frames = MIN(frames_left, ad->buffer_frames);
			//ad->lock();
			ad->audio_server_process(frames, ad->samples_in);
			//ad->unlock();

			for (int i = 0; i < frames * ad->channels; i++) {

				out[i] = ad->samples_in[i] >> 16;
			}

			frames_left -= frames;
			out += frames * ad->channels;
		};
	};

	if (ad->mutex)
		ad->mutex->unlock();

	return 0;
};

void AudioDriverIphone::start() {
	active = true;
};

int AudioDriverIphone::get_mix_rate() const {
	return 44100;
};

AudioDriver::SpeakerMode AudioDriverIphone::get_speaker_mode() const {
	return SPEAKER_MODE_STEREO;
};

void AudioDriverIphone::lock() {

	if (active && mutex)
		mutex->lock();
};

void AudioDriverIphone::unlock() {
	if (active && mutex)
		mutex->unlock();
};

void AudioDriverIphone::finish() {

	memdelete_arr(samples_in);
};

AudioDriverIphone::AudioDriverIphone() {

	mutex = Mutex::create(); //NULL;
};

AudioDriverIphone::~AudioDriverIphone(){

};
