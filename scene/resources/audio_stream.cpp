/*************************************************************************/
/*  audio_stream.cpp                                                     */
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
#include "audio_stream.h"



int AudioStream::InternalAudioStream::get_channel_count() const {

	return owner->get_channel_count();

}

void AudioStream::InternalAudioStream::set_mix_rate(int p_rate) {

	owner->_mix_rate=p_rate;
}

bool AudioStream::InternalAudioStream::mix(int32_t *p_buffer,int p_frames) {

	return owner->mix(p_buffer,p_frames);
}

bool AudioStream::InternalAudioStream::can_update_mt() const {

	return owner->get_update_mode()==UPDATE_THREAD;
}

void AudioStream::InternalAudioStream::update() {

	owner->update();
}

AudioServer::AudioStream *AudioStream::get_audio_stream() {

	return internal_audio_stream;
}


void AudioStream::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("play"),&AudioStream::play);
	ObjectTypeDB::bind_method(_MD("stop"),&AudioStream::stop);
	ObjectTypeDB::bind_method(_MD("is_playing"),&AudioStream::is_playing);

	ObjectTypeDB::bind_method(_MD("set_loop","enabled"),&AudioStream::set_loop);
	ObjectTypeDB::bind_method(_MD("has_loop"),&AudioStream::has_loop);

	ObjectTypeDB::bind_method(_MD("get_stream_name"),&AudioStream::get_stream_name);
	ObjectTypeDB::bind_method(_MD("get_loop_count"),&AudioStream::get_loop_count);

	ObjectTypeDB::bind_method(_MD("seek_pos","pos"),&AudioStream::seek_pos);
	ObjectTypeDB::bind_method(_MD("get_pos"),&AudioStream::get_pos);

	ObjectTypeDB::bind_method(_MD("get_length"),&AudioStream::get_length);

	ObjectTypeDB::bind_method(_MD("get_update_mode"),&AudioStream::get_update_mode);

	ObjectTypeDB::bind_method(_MD("update"),&AudioStream::update);

	BIND_CONSTANT( UPDATE_NONE );
	BIND_CONSTANT( UPDATE_IDLE );
	BIND_CONSTANT( UPDATE_THREAD );

}

AudioStream::AudioStream() {

	_mix_rate=44100;
	internal_audio_stream = memnew( InternalAudioStream );
	internal_audio_stream->owner=this;
}


AudioStream::~AudioStream()  {

	memdelete(internal_audio_stream);
}
