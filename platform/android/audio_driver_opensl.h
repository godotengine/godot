/*************************************************************************/
/*  audio_driver_opensl.h                                                */
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
#ifndef AUDIO_DRIVER_OPENSL_H
#define AUDIO_DRIVER_OPENSL_H

#include "os/mutex.h"
#include "servers/audio_server.h"

#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>

class AudioDriverOpenSL : public AudioDriver {

	bool active;
	Mutex *mutex;

	enum {

		BUFFER_COUNT = 2
	};

	bool pause;

	uint32_t buffer_size;
	int16_t *buffers[BUFFER_COUNT];
	int32_t *mixdown_buffer;
	int last_free;

	SLPlayItf playItf;
	SLObjectItf sl;
	SLEngineItf EngineItf;
	SLObjectItf OutputMix;
	SLVolumeItf volumeItf;
	SLObjectItf player;
	SLAndroidSimpleBufferQueueItf bufferQueueItf;
	SLDataSource audioSource;
	SLDataFormat_PCM pcm;
	SLDataSink audioSink;
	SLDataLocator_OutputMix locator_outputmix;
	SLBufferQueueState state;

	static AudioDriverOpenSL *s_ad;

	void _buffer_callback(
			SLAndroidSimpleBufferQueueItf queueItf
			/*   SLuint32 eventFlags,
	    const void * pBuffer,
	    SLuint32 bufferSize,
	    SLuint32 dataUsed*/);

	static void _buffer_callbacks(
			SLAndroidSimpleBufferQueueItf queueItf,
			/*SLuint32 eventFlags,
	    const void * pBuffer,
	    SLuint32 bufferSize,
	    SLuint32 dataUsed,*/
			void *pContext);

public:
	void set_singleton();

	virtual const char *get_name() const;

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const;
	virtual SpeakerMode get_speaker_mode() const;
	virtual void lock();
	virtual void unlock();
	virtual void finish();

	virtual void set_pause(bool p_pause);

	AudioDriverOpenSL();
};

#endif // AUDIO_DRIVER_ANDROID_H
