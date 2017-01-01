/*************************************************************************/
/*  cp_sample.cpp                                                        */
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
#include "cp_sample.h"

const char * CPSample::get_name() const {

	return name;
}
void CPSample::set_name(const char *p_name) {

	if (p_name==NULL) {
		name[0]=0;
		return;
	}
	
	
	bool done=false;
	for (int i=0;i<NAME_MAX_LEN;i++) {
		
		
		name[i]=done?0:p_name[i];
		if (!done && p_name[i]==0)
			done=true;
	}
	
	name[NAME_MAX_LEN-1]=0; /* just in case */
	
}

void CPSample::set_default_volume(uint8_t p_vol) {

	default_volume=p_vol;
}
uint8_t CPSample::get_default_volume() const{

	return default_volume;
}

void CPSample::set_global_volume(uint8_t p_vol) {

	global_volume=p_vol;
}
uint8_t CPSample::get_global_volume() const{

	return global_volume;
}

void CPSample::set_pan_enabled(bool p_vol) {

	pan_enabled=p_vol;
}
bool CPSample::is_pan_enabled() const{

	return pan_enabled;
}

void CPSample::set_pan(uint8_t p_pan) {

	pan=p_pan;

}
uint8_t CPSample::get_pan() const{

	return pan;
}


void CPSample::set_vibrato_type(VibratoType p_vibrato_type) {

	vibrato_type=p_vibrato_type;
}
CPSample::VibratoType CPSample::get_vibrato_type()  const{

	return vibrato_type;
}

void CPSample::set_vibrato_speed(uint8_t p_vibrato_speed) {

	vibrato_speed=p_vibrato_speed;
}
uint8_t CPSample::get_vibrato_speed() const {

	return vibrato_speed;
}

void CPSample::set_vibrato_depth(uint8_t p_vibrato_depth) {

	vibrato_depth=p_vibrato_depth;
}
uint8_t CPSample::get_vibrato_depth() const{

	return vibrato_depth;
}

void CPSample::set_vibrato_rate(uint8_t p_vibrato_rate) {

	vibrato_rate=p_vibrato_rate;
}
uint8_t CPSample::get_vibrato_rate() const{

	return vibrato_rate;
}

void CPSample::set_sample_data(CPSample_ID p_ID) {
	
	id=p_ID;
}
CPSample_ID CPSample::get_sample_data() const{
	
	return id;
}

void CPSample::operator=(const CPSample &p_sample) {
	
	copy_from(p_sample);
}
void CPSample::copy_from(const CPSample &p_sample) {
	
	reset();
	set_name(p_sample.get_name());
	
	default_volume=p_sample.default_volume;
	global_volume=p_sample.global_volume;

	pan_enabled=p_sample.pan_enabled;
	pan=p_sample.pan;

	vibrato_type=p_sample.vibrato_type;
	vibrato_speed=p_sample.vibrato_speed;
	vibrato_depth=p_sample.vibrato_depth;
	vibrato_rate=p_sample.vibrato_rate;
	
	if (CPSampleManager::get_singleton() && !p_sample.id.is_null())
		CPSampleManager::get_singleton()->copy_to( p_sample.id, id );
}


	


void CPSample::reset() {

	
	name[0]=0;

	default_volume=64;
	global_volume=64;

	pan_enabled=false;
	pan=32;

	vibrato_type=VIBRATO_SINE;
	vibrato_speed=0;
	vibrato_depth=0;
	vibrato_rate=0;

	if (!id.is_null() && CPSampleManager::get_singleton())
		CPSampleManager::get_singleton()->destroy( id );
	
	id=CPSample_ID();
	
}

CPSample::CPSample(const CPSample&p_from) {
	
	reset();
	copy_from(p_from);
}
CPSample::CPSample() {

	reset();
}

CPSample::~CPSample() {

	reset();
}
