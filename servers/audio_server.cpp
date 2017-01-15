/*************************************************************************/
/*  audio_server.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "audio_server.h"
#include "globals.h"
#include "os/os.h"
AudioDriver *AudioDriver::singleton=NULL;
AudioDriver *AudioDriver::get_singleton() {

	return singleton;
}

void AudioDriver::set_singleton() {

	singleton=this;
}

void AudioDriver::audio_server_process(int p_frames,int32_t *p_buffer,bool p_update_mix_time) {

	AudioServer * audio_server = static_cast<AudioServer*>(AudioServer::get_singleton());
	if (p_update_mix_time)
		update_mix_time(p_frames);
//	audio_server->driver_process(p_frames,p_buffer);
}

void AudioDriver::update_mix_time(int p_frames) {

	_mix_amount+=p_frames;
	_last_mix_time=OS::get_singleton()->get_ticks_usec();
}

double AudioDriver::get_mix_time() const {

	double total = (OS::get_singleton()->get_ticks_usec() - _last_mix_time) / 1000000.0;
	total+=_mix_amount/(double)get_mix_rate();
	return total;

}


AudioDriver::AudioDriver() {

	_last_mix_time=0;
	_mix_amount=0;
}


AudioDriver *AudioDriverManager::drivers[MAX_DRIVERS];
int AudioDriverManager::driver_count=0;



void AudioDriverManager::add_driver(AudioDriver *p_driver) {

	ERR_FAIL_COND(driver_count>=MAX_DRIVERS);
	drivers[driver_count++]=p_driver;
}

int AudioDriverManager::get_driver_count() {

	return driver_count;
}
AudioDriver *AudioDriverManager::get_driver(int p_driver) {

	ERR_FAIL_INDEX_V(p_driver,driver_count,NULL);
	return drivers[p_driver];
}


//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////

void AudioServer::set_bus_count(int p_count) {

	ERR_FAIL_COND(p_count<1);
	ERR_FAIL_INDEX(p_count,256);
	lock();
	buses.resize(p_count);
	unlock();
}

int AudioServer::get_bus_count() const {

	return buses.size();
}

void AudioServer::set_bus_mode(int p_bus,BusMode p_mode) {

	ERR_FAIL_INDEX(p_bus,buses.size());

}
AudioServer::BusMode AudioServer::get_bus_mode(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus,buses.size(),BUS_MODE_STEREO);

	return buses[p_bus].mode;
}

void AudioServer::set_bus_name(int p_bus,const String& p_name) {

	ERR_FAIL_INDEX(p_bus,buses.size());
	buses[p_bus].name=p_name;

}
String AudioServer::get_bus_name(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus,buses.size(),String());
	return buses[p_bus].name;
}

void AudioServer::set_bus_volume_db(int p_bus,float p_volume_db) {

	ERR_FAIL_INDEX(p_bus,buses.size());
	buses[p_bus].volume_db=p_volume_db;

}
float AudioServer::get_bus_volume_db(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus,buses.size(),0);
	return buses[p_bus].volume_db;

}

void AudioServer::add_bus_effect(int p_bus,const Ref<AudioEffect>& p_effect,int p_at_pos) {

	ERR_FAIL_COND(p_effect.is_null());
	ERR_FAIL_INDEX(p_bus,buses.size());

	lock();

	Bus::Effect fx;
	fx.effect=p_effect;
	//fx.instance=p_effect->instance();
	fx.enabled=true;

	if (p_at_pos>=buses[p_bus].effects.size() || p_at_pos<0) {
		buses[p_bus].effects.push_back(fx);
	} else {
		buses[p_bus].effects.insert(p_at_pos,fx);
	}

	unlock();
}


void AudioServer::remove_bus_effect(int p_bus,int p_effect) {

	ERR_FAIL_INDEX(p_bus,buses.size());

	lock();

	buses[p_bus].effects.remove(p_effect);

	unlock();
}

int AudioServer::get_bus_effect_count(int p_bus) {

	ERR_FAIL_INDEX_V(p_bus,buses.size(),0);

	return buses[p_bus].effects.size();

}

Ref<AudioEffect> AudioServer::get_bus_effect(int p_bus,int p_effect) {

	ERR_FAIL_INDEX_V(p_bus,buses.size(),Ref<AudioEffect>());
	ERR_FAIL_INDEX_V(p_effect,buses[p_bus].effects.size(),Ref<AudioEffect>());

	return buses[p_bus].effects[p_effect].effect;

}

void AudioServer::swap_bus_effects(int p_bus,int p_effect,int p_by_effect) {

	ERR_FAIL_INDEX(p_bus,buses.size());
	ERR_FAIL_INDEX(p_effect,buses[p_bus].effects.size());
	ERR_FAIL_INDEX(p_by_effect,buses[p_bus].effects.size());

	lock();
	SWAP( buses[p_bus].effects[p_effect], buses[p_bus].effects[p_by_effect] );
	unlock();
}

void AudioServer::set_bus_effect_enabled(int p_bus,int p_effect,bool p_enabled) {

	ERR_FAIL_INDEX(p_bus,buses.size());
	ERR_FAIL_INDEX(p_effect,buses[p_bus].effects.size());
	buses[p_bus].effects[p_effect].enabled=p_enabled;

}
bool AudioServer::is_bus_effect_enabled(int p_bus,int p_effect) const {

	ERR_FAIL_INDEX_V(p_bus,buses.size(),false);
	ERR_FAIL_INDEX_V(p_effect,buses[p_bus].effects.size(),false);
	return buses[p_bus].effects[p_effect].enabled;

}

void AudioServer::init() {

	set_bus_count(1);;
	set_bus_name(0,"Master");
}
void AudioServer::finish() {

	buses.clear();
}
void AudioServer::update() {


}

/* MISC config */

void AudioServer::lock() {

	AudioDriver::get_singleton()->lock();
}
void AudioServer::unlock() {

	AudioDriver::get_singleton()->unlock();

}


AudioServer::SpeakerMode AudioServer::get_speaker_mode() const  {

	return (AudioServer::SpeakerMode)AudioDriver::get_singleton()->get_speaker_mode();
}
float AudioServer::get_mix_rate() const {

	return AudioDriver::get_singleton()->get_mix_rate();
}

float AudioServer::read_output_peak_db() const {

	return 0;
}

AudioServer *AudioServer::get_singleton() {

	return singleton;
}

double AudioServer::get_mix_time() const {

	return 0;
}
double AudioServer::get_output_delay() const {

	return 0;
}

AudioServer* AudioServer::singleton=NULL;


void AudioServer::_bind_methods() {

}


AudioServer::AudioServer() {

	singleton=this;
}

AudioServer::~AudioServer() {


}

