/*************************************************************************/
/*  audio_driver_nacl.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "audio_driver_nacl.h"

#include "ppapi/cpp/instance.h"

extern pp::Instance* godot_instance;

const char* AudioDriverNacl::get_name() const {

	return "Nacl";
}

void AudioDriverNacl::output_callback(void* samples, uint32_t buffer_size, void* data) {

	AudioDriverNacl* ad = (AudioDriverNacl*)data;
	int16_t* out = (int16_t*)samples;

	ad->lock();
	ad->audio_server_process(ad->sample_frame_count_, ad->samples_in);
	ad->unlock();

	for (int i=0; i<ad->sample_count; i++) {

		out[i] = ad->samples_in[i]>>16;
	};

};

Error AudioDriverNacl::init(){

	int frame_size = 4096;
	sample_frame_count_ = pp::AudioConfig::RecommendSampleFrameCount(godot_instance,PP_AUDIOSAMPLERATE_44100, frame_size);
	sample_count = sample_frame_count_ * 2;

	audio_ = pp::Audio(godot_instance,
					   pp::AudioConfig(godot_instance,
									   PP_AUDIOSAMPLERATE_44100,
									   sample_frame_count_),
					   &AudioDriverNacl::output_callback,
					   this);

	samples_in = memnew_arr(int32_t, sample_frame_count_ * 2);

	return OK;
}
void AudioDriverNacl::start(){

	audio_.StartPlayback();
}
int AudioDriverNacl::get_mix_rate() const {

	return 44100;
}
AudioDriverSW::OutputFormat AudioDriverNacl::get_output_format() const{

	return OUTPUT_STEREO;
}
void AudioDriverNacl::lock(){

}
void AudioDriverNacl::unlock() {


}
void AudioDriverNacl::finish(){

	audio_.StopPlayback();
}


AudioDriverNacl::AudioDriverNacl() {
}

AudioDriverNacl::~AudioDriverNacl() {

	memdelete_arr(samples_in);
}


