/*************************************************************************/
/*  sample.cpp                                                           */
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
#include "sample.h"


void Sample::_set_data(const Dictionary& p_data) {

	ERR_FAIL_COND(!p_data.has("packing"));
	String packing = p_data["packing"];

	if (packing=="raw") {

		ERR_FAIL_COND( !p_data.has("stereo"));
		ERR_FAIL_COND( !p_data.has("format"));
		ERR_FAIL_COND( !p_data.has("length"));
		bool stereo=p_data["stereo"];
		int length=p_data["length"];
		Format fmt;
		String fmtstr=p_data["format"];
		if (fmtstr=="pcm8")
			fmt=FORMAT_PCM8;
		else if (fmtstr=="pcm16")
			fmt=FORMAT_PCM16;
		else if (fmtstr=="ima_adpcm")
			fmt=FORMAT_IMA_ADPCM;
		else {
			ERR_EXPLAIN("Invalid format for sample: "+fmtstr);
			ERR_FAIL();
		}

		ERR_FAIL_COND(!p_data.has("data"));

		create(fmt,stereo,length);
		set_data(p_data["data"]);
	} else {

		ERR_EXPLAIN("Invalid packing for sample data: "+packing);
		ERR_FAIL();
	}
}

Dictionary Sample::_get_data() const {

	Dictionary d;
	switch(get_format()) {

		case FORMAT_PCM8: d["format"]="pcm8"; break;
		case FORMAT_PCM16: d["format"]="pcm16"; break;
		case FORMAT_IMA_ADPCM: d["format"]="ima_adpcm"; break;
	}

	d["stereo"]=is_stereo();
	d["length"]=get_length();
	d["packing"]="raw";
	d["data"]=get_data();
	return d;

}

void Sample::create(Format p_format, bool p_stereo, int p_length) {

	if (p_length<1)
		return;

	if (sample.is_valid())
		AudioServer::get_singleton()->free(sample);


	mix_rate=44100;
	stereo=p_stereo;
	length=p_length;
	format=p_format;
	loop_format=LOOP_NONE;
	loop_begin=0;
	loop_end=0;

	sample=AudioServer::get_singleton()->sample_create((AudioServer::SampleFormat)p_format,p_stereo,p_length);
}


Sample::Format Sample::get_format() const {

	return format;
}
bool Sample::is_stereo() const {


	return stereo;
}
int Sample::get_length() const {


	return length;
}

void Sample::set_data(const DVector<uint8_t>& p_buffer) {

	if (sample.is_valid())
		AudioServer::get_singleton()->sample_set_data(sample,p_buffer);

}
DVector<uint8_t> Sample::get_data() const {

	if (sample.is_valid())
		return AudioServer::get_singleton()->sample_get_data(sample);

	return DVector<uint8_t>();

}

void Sample::set_mix_rate(int p_rate) {

	mix_rate=p_rate;
	if (sample.is_valid())
		return AudioServer::get_singleton()->sample_set_mix_rate(sample,mix_rate);

}
int Sample::get_mix_rate() const {

	return mix_rate;
}

void Sample::set_loop_format(LoopFormat p_format) {

	if (sample.is_valid())
		AudioServer::get_singleton()->sample_set_loop_format(sample,(AudioServer::SampleLoopFormat)p_format);
	loop_format=p_format;
}

Sample::LoopFormat Sample::get_loop_format() const {

	return loop_format;
}

void Sample::set_loop_begin(int p_pos) {

	if (sample.is_valid())
		AudioServer::get_singleton()->sample_set_loop_begin(sample,p_pos);
	loop_begin=p_pos;

}
int Sample::get_loop_begin() const {

	return loop_begin;
}

void Sample::set_loop_end(int p_pos) {

	if (sample.is_valid())
		AudioServer::get_singleton()->sample_set_loop_end(sample,p_pos);
	loop_end=p_pos;
}

int Sample::get_loop_end() const {

	return loop_end;
}

RID Sample::get_rid() const {

	return sample;
}



void Sample::_bind_methods(){


	ClassDB::bind_method(_MD("create","format","stereo","length"),&Sample::create);
	ClassDB::bind_method(_MD("get_format"),&Sample::get_format);
	ClassDB::bind_method(_MD("is_stereo"),&Sample::is_stereo);
	ClassDB::bind_method(_MD("get_length"),&Sample::get_length);
	ClassDB::bind_method(_MD("set_data","data"),&Sample::set_data);
	ClassDB::bind_method(_MD("get_data"),&Sample::get_data);
	ClassDB::bind_method(_MD("set_mix_rate","hz"),&Sample::set_mix_rate);
	ClassDB::bind_method(_MD("get_mix_rate"),&Sample::get_mix_rate);
	ClassDB::bind_method(_MD("set_loop_format","format"),&Sample::set_loop_format);
	ClassDB::bind_method(_MD("get_loop_format"),&Sample::get_loop_format);
	ClassDB::bind_method(_MD("set_loop_begin","pos"),&Sample::set_loop_begin);
	ClassDB::bind_method(_MD("get_loop_begin"),&Sample::get_loop_begin);
	ClassDB::bind_method(_MD("set_loop_end","pos"),&Sample::set_loop_end);
	ClassDB::bind_method(_MD("get_loop_end"),&Sample::get_loop_end);

	ClassDB::bind_method(_MD("_set_data"),&Sample::_set_data);
	ClassDB::bind_method(_MD("_get_data"),&Sample::_get_data);

	ADD_PROPERTY( PropertyInfo( Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), _SCS("_set_data"), _SCS("_get_data") );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "stereo"), _SCS(""), _SCS("is_stereo") );
	ADD_PROPERTY( PropertyInfo( Variant::INT, "length",PROPERTY_HINT_RANGE,"0,999999999"), _SCS(""), _SCS("get_length") );
	ADD_PROPERTY( PropertyInfo( Variant::INT, "mix_rate", PROPERTY_HINT_RANGE,"1,192000,1" ), _SCS("set_mix_rate"), _SCS("get_mix_rate") );
	ADD_PROPERTY( PropertyInfo( Variant::INT, "loop_format", PROPERTY_HINT_ENUM,"None,Forward,PingPong" ), _SCS("set_loop_format"), _SCS("get_loop_format") );
	ADD_PROPERTY( PropertyInfo( Variant::INT, "loop_begin", PROPERTY_HINT_RANGE,"0,"+itos(999999999)+",1"), _SCS("set_loop_begin"), _SCS("get_loop_begin") );
	ADD_PROPERTY( PropertyInfo( Variant::INT, "loop_end", PROPERTY_HINT_RANGE,"0,"+itos(999999999)+",1"), _SCS("set_loop_end"), _SCS("get_loop_end") );

	BIND_CONSTANT( FORMAT_PCM8 );
	BIND_CONSTANT( FORMAT_PCM16 );
	BIND_CONSTANT( FORMAT_IMA_ADPCM );

	BIND_CONSTANT( LOOP_NONE );
	BIND_CONSTANT( LOOP_FORWARD );
	BIND_CONSTANT( LOOP_PING_PONG );

}

Sample::Sample() {

	format=FORMAT_PCM8;
	length=0;
	stereo=false;

	loop_format=LOOP_NONE;
	loop_begin=0;
	loop_end=0;
	mix_rate=44100;

}

Sample::~Sample() {

	if (sample.is_valid())
		AudioServer::get_singleton()->free(sample);
}
