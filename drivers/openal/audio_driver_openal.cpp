/*************************************************************************/
/*  audio_driver_openal.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#include "audio_driver_openal.h"

#ifdef OPENAL_ENABLED

#include <errno.h>
#include "globals.h"




Error AudioDriverOPENAL::init() {

	active=false;
	thread_exited=false;
	exit_thread=false;
	pcm_open = false;
	samples_in = NULL;
	samples_out = NULL;

	mix_rate = 44100;
	output_format = OUTPUT_STEREO;
	channels = 2;


	int                  status;
	//snd_pcm_hw_params_t *hwparams;
	//snd_pcm_sw_params_t *swparams;

#define CHECK_FAIL(m_cond)\
	if (m_cond) {\
		printf("OPENAL ERR");\
		ERR_FAIL_COND_V(m_cond,ERR_CANT_OPEN);\
	}
//fprintf(stderr,"OPENAL ERR: %s\n",snd_strerror(status));\
	//todo, add
	//6 chans - "plug:surround51"
	//4 chans - "plug:surround40";

	int major, minor;
  alcGetIntegerv(NULL, ALC_MAJOR_VERSION, 1, &major);
  alcGetIntegerv(NULL, ALC_MAJOR_VERSION, 1, &minor);

	CHECK_FAIL(major != 1);

  printf("ALC version: %i.%i\n", major, minor);
  printf("Default device: %s\n", alcGetString(NULL, ALC_DEFAULT_DEVICE_SPECIFIER));

  ALCdevice* device = alcOpenDevice(NULL);
	CHECK_FAIL(device == NULL);
	ALCcontext* context = alcCreateContext(device, NULL);
	CHECK_FAIL(context == NULL);
  alcMakeContextCurrent(context);

	CHECK_FAIL(!alGetString(AL_VERSION));

	printf("OpenAL version: %s\n", alGetString(AL_VERSION));
  printf("OpenAL vendor: %s\n", alGetString(AL_VENDOR));
  printf("OpenAL renderer: %s\n", alGetString(AL_RENDERER));

  ALfloat listenerPos[] = {0.0, 0.0, 0.0};
  ALfloat listenerVel[] = {0.0, 0.0, 0.0};
  ALfloat listenerOri[] = {0.0, 0.0, -1.0, 0.0, 1.0, 0.0};

  alListenerfv(AL_POSITION, listenerPos);
  alListenerfv(AL_VELOCITY, listenerVel);
  alListenerfv(AL_ORIENTATION, listenerOri);

  alGenBuffers(1, buffers);
  alGenSources(1, sources);

  CHECK_FAIL(!alIsSource(sources[0]));

/*
	status = snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);

	ERR_FAIL_COND_V( status<0, ERR_CANT_OPEN );

	snd_pcm_hw_params_alloca(&hwparams);
	status = snd_pcm_hw_params_any(pcm_handle, hwparams);

	CHECK_FAIL( status<0 );

	status = snd_pcm_hw_params_set_access(pcm_handle, hwparams, SND_PCM_ACCESS_RW_INTERLEAVED);

	CHECK_FAIL( status<0 );

	//not interested in anything else
	status = snd_pcm_hw_params_set_format(pcm_handle, hwparams, SND_PCM_FORMAT_S16_LE);

	CHECK_FAIL( status<0 );

	//todo: support 4 and 6
	status = snd_pcm_hw_params_set_channels(pcm_handle, hwparams, 2);

	CHECK_FAIL( status<0 );

	status = snd_pcm_hw_params_set_rate_near(pcm_handle, hwparams, &mix_rate, NULL);


	CHECK_FAIL( status<0 );

	int latency = GLOBAL_DEF("audio/output_latency",25);
	buffer_size = nearest_power_of_2( latency * mix_rate / 1000 );

	status = snd_pcm_hw_params_set_period_size_near(pcm_handle, hwparams, &buffer_size, NULL);


	CHECK_FAIL( status<0 );

	unsigned int periods=2;
	status = snd_pcm_hw_params_set_periods_near(pcm_handle, hwparams, &periods, NULL);

	CHECK_FAIL( status<0 );

	status = snd_pcm_hw_params(pcm_handle,hwparams);

	CHECK_FAIL( status<0 );

	//snd_pcm_hw_params_free(&hwparams);


	snd_pcm_sw_params_alloca(&swparams);
	status = snd_pcm_sw_params_current(pcm_handle, swparams);
	CHECK_FAIL( status<0 );

	status = snd_pcm_sw_params_set_avail_min(pcm_handle, swparams, buffer_size);

	CHECK_FAIL( status<0 );

	status = snd_pcm_sw_params_set_start_threshold(pcm_handle, swparams, 1);

	CHECK_FAIL( status<0 );

	status = snd_pcm_sw_params(pcm_handle, swparams);

	CHECK_FAIL( status<0 );
*/
	int latency = GLOBAL_DEF("audio/output_latency",25);
	buffer_size = nearest_power_of_2( latency * mix_rate / 1000 );

	samples_in = memnew_arr(int32_t, buffer_size*channels);
	samples_out = memnew_arr(int16_t, buffer_size*channels);

	//snd_pcm_nonblock(pcm_handle, 0);

	mutex=Mutex::create();
	//thread = Thread::create(AudioDriverOPENAL::thread_func, this);
	emscripten_async_call(AudioDriverOPENAL::thread_func, reinterpret_cast<void*>(this), 0);

	return OK;
};

void AudioDriverOPENAL::thread_func(void* p_udata) {

	//WARN_PRINT("called");
	AudioDriverOPENAL* ad = (AudioDriverOPENAL*)p_udata;
	ALenum error;

	//while (!ad->exit_thread) {
		if (!ad->active) {

			for (unsigned int i=0; i < ad->buffer_size*ad->channels; i++) {

				ad->samples_out[i] = 0;
			};
		} else {
			//WARN_PRINT("active");
			ad->lock();

			ad->audio_server_process(ad->buffer_size, ad->samples_in);

			ad->unlock();

			for(unsigned int i=0;i<ad->buffer_size*ad->channels;i++) {

				ad->samples_out[i]=ad->samples_in[i]>>16;
			}
		};

		alBufferData(ad->buffers[0], AL_FORMAT_STEREO16, &ad->samples_out[0], ad->buffer_size, ad->mix_rate);

		error = alGetError();
		if ( error != AL_NO_ERROR) {
				fprintf(stderr, "OPENAL after alBufferData error: %d %x\n", (int)error, (unsigned int)error);
		}

		alSourcei(ad->sources[0], AL_BUFFER, ad->buffers[0]);

		error = alGetError();
		if ( error != AL_NO_ERROR) {
				fprintf(stderr, "OPENAL after alSourcei error: %d %x\n", (int)error, (unsigned int)error);
		}

		/*ALint state;
	  alGetSourcei(ad->sources[0], AL_SOURCE_STATE, &state);
		if(state != AL_INITIAL) {
			ERR_PRINT("No AL_INITIAL, OpenAL en cualquiera");
		}*/

	  alSourcePlay(ad->sources[0]);

		error = alGetError();
		if ( error != AL_NO_ERROR) {
				fprintf(stderr, "OPENAL after alSourcePlay error: %d %x\n", (int)error, (unsigned int)error);
		}

		/*ALint state;
	  alGetSourcei(ad->sources[0], AL_SOURCE_STATE, &state);
		if(state != AL_PLAYING) {
			ERR_PRINT("No AL_PLAYING, OpenAL en cualquiera");
			fprintf(stderr, "OPENAL failed and can't recover: %d\n", state);
			ad->active=false;
			ad->exit_thread=true;
		}*/


/*
		int todo = ad->buffer_size; // * ad->channels * 2;
		int total = 0;

		while (todo) {

			if (ad->exit_thread)
				break;
			uint8_t* src = (uint8_t*)ad->samples_out;
			int wrote = snd_pcm_writei(ad->pcm_handle, (void*)(src + (total*ad->channels)), todo);

			if (wrote < 0) {
				if (ad->exit_thread)
					break;

				if ( wrote == -EAGAIN ) {
					usleep(1000); //can't write yet (though this is blocking..)
					continue;
				}
				wrote = snd_pcm_recover(ad->pcm_handle, wrote, 0);
				if ( wrote < 0 ) {
					//absolute fail
					fprintf(stderr, "OPENAL failed and can't recover: %s\n", snd_strerror(wrote));
					ad->active=false;
					ad->exit_thread=true;
					break;
				}
				continue;
			};
			total += wrote;
			todo -= wrote;

		};*/
	//};
	if ( (!ad->exit_thread) ) {
		emscripten_async_call(AudioDriverOPENAL::thread_func, reinterpret_cast<void*>(ad), 0);
	} else {
		ad->thread_exited=true;
	}
};

void AudioDriverOPENAL::start() {

	active = true;
};

int AudioDriverOPENAL::get_mix_rate() const {

	return mix_rate;
};

AudioDriverSW::OutputFormat AudioDriverOPENAL::get_output_format() const {

	return output_format;
};
void AudioDriverOPENAL::lock() {

	if (!thread || !mutex)
		return;
	mutex->lock();
};
void AudioDriverOPENAL::unlock() {

	if (!thread || !mutex)
		return;
	mutex->unlock();
};

void AudioDriverOPENAL::finish() {

	if (!thread)
		return;

	exit_thread = true;
	//Thread::wait_to_finish(thread);

	/*if (pcm_open)
		snd_pcm_close(pcm_handle);*/

	if (samples_in) {
		memdelete_arr(samples_in);
		memdelete_arr(samples_out);
	};

	memdelete(thread);
	if (mutex)
		memdelete(mutex);
	thread = NULL;
};

AudioDriverOPENAL::AudioDriverOPENAL() {

	mutex = NULL;
	thread=NULL;
	//pcm_handle=NULL;
};

AudioDriverOPENAL::~AudioDriverOPENAL() {
	WARN_PRINT("OPENAL dying");
};

#endif
