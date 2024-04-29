/**************************************************************************/
/*  audio_driver_opensl.cpp                                               */
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

#include "audio_driver_opensl.h"

void AudioDriverOpenSL::_buffer_callback(
		SLAndroidSimpleBufferQueueItf queueItf) {
	bool mix = true;

	if (pause) {
		mix = false;
	} else {
		mix = mutex.try_lock();
	}

	audio_server_process(output_buffer_frames, buffers[last_free].ptr(), mix);

	if (mix) {
		mutex.unlock();
	}

	(*queueItf)->Enqueue(queueItf, buffers[last_free].ptr(), buffers[last_free].size());
	last_free = (last_free + 1) % BUFFER_COUNT;
}

void AudioDriverOpenSL::_buffer_callbacks(
		SLAndroidSimpleBufferQueueItf queueItf,
		void *pContext) {
	AudioDriverOpenSL *ad = static_cast<AudioDriverOpenSL *>(pContext);

	ad->_buffer_callback(queueItf);
}

Error AudioDriverOpenSL::init() {
	SLresult res;
	SLEngineOption EngineOption[] = {
		{ (SLuint32)SL_ENGINEOPTION_THREADSAFE, (SLuint32)SL_BOOLEAN_TRUE }
	};
	res = slCreateEngine(&sl, 1, EngineOption, 0, nullptr, nullptr);
	ERR_FAIL_COND_V_MSG(res != SL_RESULT_SUCCESS, ERR_INVALID_PARAMETER, "Could not initialize OpenSL.");

	res = (*sl)->Realize(sl, SL_BOOLEAN_FALSE);
	ERR_FAIL_COND_V_MSG(res != SL_RESULT_SUCCESS, ERR_INVALID_PARAMETER, "Could not realize OpenSL.");

	return OK;
}

void AudioDriverOpenSL::start() {
	active = false;

	SLresult res;

	// TODO: `mix_rate`, `output_channels` and `output_buffer_format` are hardcoded.
	mix_rate = 44100;
	output_channels = 2;
	output_buffer_format = BUFFER_FORMAT_INTEGER_16;

	output_buffer_frames = 1024;
	for (int i = 0; i < BUFFER_COUNT; i++) {
		buffers[i].resize(output_buffer_frames * output_channels * get_size_of_sample(output_buffer_format));
	}

	// Callback context for the buffer queue callback function.

	// Get the SL Engine Interface which is implicit.
	res = (*sl)->GetInterface(sl, SL_IID_ENGINE, (void *)&EngineItf);

	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);

	{
		const SLInterfaceID ids[1] = { SL_IID_ENVIRONMENTALREVERB };
		const SLboolean req[1] = { SL_BOOLEAN_FALSE };
		res = (*EngineItf)->CreateOutputMix(EngineItf, &OutputMix, 0, ids, req);
	}

	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);
	// Realizing the Output Mix object in synchronous mode.
	res = (*OutputMix)->Realize(OutputMix, SL_BOOLEAN_FALSE);
	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);

	SLDataLocator_AndroidSimpleBufferQueue loc_bufq = { SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE, BUFFER_COUNT };
	// Setup the format of the content in the buffer queue.
	pcm.formatType = SL_DATAFORMAT_PCM;
	pcm.numChannels = output_channels;
	pcm.samplesPerSec = SL_SAMPLINGRATE_44_1;
	pcm.bitsPerSample = SL_PCMSAMPLEFORMAT_FIXED_16;
	pcm.containerSize = SL_PCMSAMPLEFORMAT_FIXED_16;
	pcm.channelMask = SL_SPEAKER_FRONT_LEFT | SL_SPEAKER_FRONT_RIGHT;
#ifdef BIG_ENDIAN_ENABLED
	pcm.endianness = SL_BYTEORDER_BIGENDIAN;
#else
	pcm.endianness = SL_BYTEORDER_LITTLEENDIAN;
#endif
	audioSource.pFormat = (void *)&pcm;
	audioSource.pLocator = (void *)&loc_bufq;

	// Setup the data sink structure.
	locator_outputmix.locatorType = SL_DATALOCATOR_OUTPUTMIX;
	locator_outputmix.outputMix = OutputMix;
	audioSink.pLocator = (void *)&locator_outputmix;
	audioSink.pFormat = nullptr;

	// Create the music player.
	{
		const SLInterfaceID ids[2] = { SL_IID_BUFFERQUEUE, SL_IID_EFFECTSEND };
		const SLboolean req[2] = { SL_BOOLEAN_TRUE, SL_BOOLEAN_TRUE };

		res = (*EngineItf)->CreateAudioPlayer(EngineItf, &player, &audioSource, &audioSink, 1, ids, req);
		ERR_FAIL_COND(res != SL_RESULT_SUCCESS);
	}
	// Realizing the player in synchronous mode.
	res = (*player)->Realize(player, SL_BOOLEAN_FALSE);
	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);
	// Get seek and play interfaces.
	res = (*player)->GetInterface(player, SL_IID_PLAY, (void *)&playItf);
	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);
	res = (*player)->GetInterface(player, SL_IID_BUFFERQUEUE,
			(void *)&bufferQueueItf);
	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);
	// Setup to receive buffer queue event callbacks.
	res = (*bufferQueueItf)->RegisterCallback(bufferQueueItf, _buffer_callbacks, this);
	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);

	last_free = 0;

	// Fill up buffers.
	for (int i = 0; i < BUFFER_COUNT; i++) {
		// Enqueue a few buffers to get the ball rolling.
		res = (*bufferQueueItf)->Enqueue(bufferQueueItf, buffers[i].ptr(), buffers[i].size()); // Size given in.
	}

	res = (*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PLAYING);
	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);

	active = true;
}

void AudioDriverOpenSL::_record_buffer_callback(SLAndroidSimpleBufferQueueItf queueItf) {
	input_process(input_buffer_frames, input_buffer.ptr());
	SLresult res = (*recordBufferQueueItf)->Enqueue(recordBufferQueueItf, input_buffer.ptr(), input_buffer.size());
	ERR_FAIL_COND(res != SL_RESULT_SUCCESS);
}

void AudioDriverOpenSL::_record_buffer_callbacks(SLAndroidSimpleBufferQueueItf queueItf, void *pContext) {
	AudioDriverOpenSL *ad = static_cast<AudioDriverOpenSL *>(pContext);

	ad->_record_buffer_callback(queueItf);
}

Error AudioDriverOpenSL::init_input_device() {
	// TODO: `input_channels` and `input_buffer_format` are hardcoded.
	input_channels = 1;
	input_buffer_format = BUFFER_FORMAT_INTEGER_16;

	SLDataLocator_IODevice loc_dev = {
		SL_DATALOCATOR_IODEVICE,
		SL_IODEVICE_AUDIOINPUT,
		SL_DEFAULTDEVICEID_AUDIOINPUT,
		nullptr
	};
	SLDataSource recSource = { &loc_dev, nullptr };

	SLDataLocator_AndroidSimpleBufferQueue loc_bq = {
		SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE,
		2
	};
	SLDataFormat_PCM format_pcm = {
		SL_DATAFORMAT_PCM,
		input_channels,
		SL_SAMPLINGRATE_44_1,
		SL_PCMSAMPLEFORMAT_FIXED_16,
		SL_PCMSAMPLEFORMAT_FIXED_16,
		SL_SPEAKER_FRONT_CENTER,
		SL_BYTEORDER_LITTLEENDIAN
	};
	SLDataSink recSnk = { &loc_bq, &format_pcm };

	const SLInterfaceID ids[2] = { SL_IID_ANDROIDSIMPLEBUFFERQUEUE, SL_IID_ANDROIDCONFIGURATION };
	const SLboolean req[2] = { SL_BOOLEAN_TRUE, SL_BOOLEAN_TRUE };

	SLresult res = (*EngineItf)->CreateAudioRecorder(EngineItf, &recorder, &recSource, &recSnk, 2, ids, req);
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	res = (*recorder)->Realize(recorder, SL_BOOLEAN_FALSE);
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	res = (*recorder)->GetInterface(recorder, SL_IID_RECORD, (void *)&recordItf);
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	res = (*recorder)->GetInterface(recorder, SL_IID_ANDROIDSIMPLEBUFFERQUEUE, (void *)&recordBufferQueueItf);
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	res = (*recordBufferQueueItf)->RegisterCallback(recordBufferQueueItf, _record_buffer_callbacks, this);
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	SLuint32 state;
	res = (*recordItf)->GetRecordState(recordItf, &state);
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	if (state != SL_RECORDSTATE_STOPPED) {
		res = (*recordItf)->SetRecordState(recordItf, SL_RECORDSTATE_STOPPED);
		ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

		res = (*recordBufferQueueItf)->Clear(recordBufferQueueItf);
		ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);
	}

	input_buffer_frames = 2048;
	input_buffer.resize(input_buffer_frames * get_size_of_sample(input_buffer_format));
	input_buffer_init(input_buffer_frames);

	res = (*recordBufferQueueItf)->Enqueue(recordBufferQueueItf, input_buffer.ptr(), input_buffer.size());
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	res = (*recordItf)->SetRecordState(recordItf, SL_RECORDSTATE_RECORDING);
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	return OK;
}

Error AudioDriverOpenSL::input_start() {
	if (OS::get_singleton()->request_permission("RECORD_AUDIO")) {
		return init_input_device();
	}

	WARN_PRINT("Unable to start audio capture - No RECORD_AUDIO permission");
	return ERR_UNAUTHORIZED;
}

Error AudioDriverOpenSL::input_stop() {
	SLuint32 state;
	SLresult res = (*recordItf)->GetRecordState(recordItf, &state);
	ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	if (state != SL_RECORDSTATE_STOPPED) {
		res = (*recordItf)->SetRecordState(recordItf, SL_RECORDSTATE_STOPPED);
		ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

		res = (*recordBufferQueueItf)->Clear(recordBufferQueueItf);
		ERR_FAIL_COND_V(res != SL_RESULT_SUCCESS, ERR_CANT_OPEN);
	}

	return OK;
}

int AudioDriverOpenSL::get_mix_rate() const {
	return mix_rate;
}

int AudioDriverOpenSL::get_output_channels() const {
	return output_channels;
}

AudioDriver::BufferFormat AudioDriverOpenSL::get_output_buffer_format() const {
	return output_buffer_format;
}

int AudioDriverOpenSL::get_input_channels() const {
	return input_channels;
}

AudioDriver::BufferFormat AudioDriverOpenSL::get_input_buffer_format() const {
	return input_buffer_format;
}

void AudioDriverOpenSL::lock() {
	if (active) {
		mutex.lock();
	}
}

void AudioDriverOpenSL::unlock() {
	if (active) {
		mutex.unlock();
	}
}

void AudioDriverOpenSL::finish() {
	(*sl)->Destroy(sl);
}

void AudioDriverOpenSL::set_pause(bool p_pause) {
	pause = p_pause;

	if (active) {
		if (pause) {
			(*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PAUSED);
		} else {
			(*playItf)->SetPlayState(playItf, SL_PLAYSTATE_PLAYING);
		}
	}
}
