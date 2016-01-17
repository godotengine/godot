/*************************************************/
/*  audio_driver_rtaudio.h                       */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef AUDIO_DRIVER_RTAUDIO_H
#define AUDIO_DRIVER_RTAUDIO_H

#ifdef RTAUDIO_ENABLED

#include "servers/audio/audio_server_sw.h"
#include "drivers/rtaudio/RtAudio.h"

class AudioDriverRtAudio : public AudioDriverSW {


	static int callback( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
	 double streamTime, RtAudioStreamStatus status, void *userData );
	OutputFormat output_format;
	Mutex *mutex;
	RtAudio *dac;
	int mix_rate;
	bool active;
public:


	virtual const char* get_name() const;

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const ;
	virtual OutputFormat get_output_format() const;
	virtual void lock();
	virtual void unlock();
	virtual void finish();

	AudioDriverRtAudio();

};

#endif // AUDIO_DRIVER_RTAUDIO_H
#endif
