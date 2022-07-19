/*************************************************************************/
/*  audio_driver_jandroid.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_driver_jandroid.h"

#include "core/os/os.h"
#include "core/project_settings.h"
#include "thread_jandroid.h"

AudioDriverAndroid *AudioDriverAndroid::s_ad = NULL;

jobject AudioDriverAndroid::io;
jmethodID AudioDriverAndroid::_init_audio;
jmethodID AudioDriverAndroid::_write_buffer;
jmethodID AudioDriverAndroid::_quit;
jmethodID AudioDriverAndroid::_pause;
bool AudioDriverAndroid::active = false;
jclass AudioDriverAndroid::cls;
int AudioDriverAndroid::audioBufferFrames = 0;
int AudioDriverAndroid::mix_rate = 44100;
bool AudioDriverAndroid::quit = false;
jobject AudioDriverAndroid::audioBuffer = NULL;
void *AudioDriverAndroid::audioBufferPinned = NULL;
Mutex AudioDriverAndroid::mutex;
int32_t *AudioDriverAndroid::audioBuffer32 = NULL;

const char *AudioDriverAndroid::get_name() const {

	return "Android";
}

Error AudioDriverAndroid::init() {

	/*
	// TODO: pass in/return a (Java) device ID, also whether we're opening for input or output
	   this->spec.samples = Android_JNI_OpenAudioDevice(this->spec.freq, this->spec.format == AUDIO_U8 ? 0 : 1, this->spec.channels, this->spec.samples);
	   SDL_CalculateAudioSpec(&this->spec);

	   if (this->spec.samples == 0) {
	       // Init failed?
	       SDL_SetError("Java-side initialization failed!");
	       return 0;
	   }
*/

	//Android_JNI_SetupThread();

	//        __android_log_print(ANDROID_LOG_VERBOSE, "SDL", "SDL audio: opening device");

	JNIEnv *env = get_jni_env();
	int mix_rate = GLOBAL_GET("audio/mix_rate");

	int latency = GLOBAL_GET("audio/output_latency");
	unsigned int buffer_size = next_power_of_2(latency * mix_rate / 1000);
	print_verbose("Audio buffer size: " + itos(buffer_size));

	audioBuffer = env->CallObjectMethod(io, _init_audio, mix_rate, buffer_size);

	ERR_FAIL_COND_V(audioBuffer == NULL, ERR_INVALID_PARAMETER);

	audioBuffer = env->NewGlobalRef(audioBuffer);

	jboolean isCopy = JNI_FALSE;
	audioBufferPinned = env->GetShortArrayElements((jshortArray)audioBuffer, &isCopy);
	audioBufferFrames = env->GetArrayLength((jshortArray)audioBuffer);
	audioBuffer32 = memnew_arr(int32_t, audioBufferFrames);

	return OK;
}

void AudioDriverAndroid::start() {
	active = true;
}

void AudioDriverAndroid::setup(jobject p_io) {

	JNIEnv *env = get_jni_env();
	io = p_io;

	jclass c = env->GetObjectClass(io);
	cls = (jclass)env->NewGlobalRef(c);

	_init_audio = env->GetMethodID(cls, "audioInit", "(II)Ljava/lang/Object;");
	_write_buffer = env->GetMethodID(cls, "audioWriteShortBuffer", "([S)V");
	_quit = env->GetMethodID(cls, "audioQuit", "()V");
	_pause = env->GetMethodID(cls, "audioPause", "(Z)V");
}

void AudioDriverAndroid::thread_func(JNIEnv *env) {

	jclass cls = env->FindClass("org/godotengine/godot/Godot");
	if (cls) {

		cls = (jclass)env->NewGlobalRef(cls);
	}
	jfieldID fid = env->GetStaticFieldID(cls, "io", "Lorg/godotengine/godot/GodotIO;");
	jobject ob = env->GetStaticObjectField(cls, fid);
	jobject gob = env->NewGlobalRef(ob);
	jclass c = env->GetObjectClass(gob);
	jclass lcls = (jclass)env->NewGlobalRef(c);
	_write_buffer = env->GetMethodID(lcls, "audioWriteShortBuffer", "([S)V");

	while (!quit) {

		int16_t *ptr = (int16_t *)audioBufferPinned;
		int fc = audioBufferFrames;

		if (!s_ad->active || mutex.try_lock() != OK) {

			for (int i = 0; i < fc; i++) {
				ptr[i] = 0;
			}

		} else {

			s_ad->audio_server_process(fc / 2, audioBuffer32);

			mutex.unlock();

			for (int i = 0; i < fc; i++) {

				ptr[i] = audioBuffer32[i] >> 16;
			}
		}
		env->ReleaseShortArrayElements((jshortArray)audioBuffer, (jshort *)ptr, JNI_COMMIT);
		env->CallVoidMethod(gob, _write_buffer, (jshortArray)audioBuffer);
	}
}

int AudioDriverAndroid::get_mix_rate() const {

	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverAndroid::get_speaker_mode() const {

	return SPEAKER_MODE_STEREO;
}

void AudioDriverAndroid::lock() {

	mutex.lock();
}

void AudioDriverAndroid::unlock() {

	mutex.unlock();
}

void AudioDriverAndroid::finish() {

	JNIEnv *env = get_jni_env();
	env->CallVoidMethod(io, _quit);

	if (audioBuffer) {
		env->DeleteGlobalRef(audioBuffer);
		audioBuffer = NULL;
		audioBufferPinned = NULL;
	}

	active = false;
}

void AudioDriverAndroid::set_pause(bool p_pause) {

	JNIEnv *env = get_jni_env();
	env->CallVoidMethod(io, _pause, p_pause);
}

AudioDriverAndroid::AudioDriverAndroid() {

	s_ad = this;
	active = false;
}
