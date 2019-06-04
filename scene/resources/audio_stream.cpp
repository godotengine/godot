/*************************************************************************/
/*  audio_stream.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

//////////////////////////////

void AudioStreamPlayback::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("play", "from_pos_sec"), &AudioStreamPlayback::play, DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("stop"), &AudioStreamPlayback::stop);
	ObjectTypeDB::bind_method(_MD("is_playing"), &AudioStreamPlayback::is_playing);

	ObjectTypeDB::bind_method(_MD("set_loop", "enabled"), &AudioStreamPlayback::set_loop);
	ObjectTypeDB::bind_method(_MD("has_loop"), &AudioStreamPlayback::has_loop);

	ObjectTypeDB::bind_method(_MD("get_loop_count"), &AudioStreamPlayback::get_loop_count);

	ObjectTypeDB::bind_method(_MD("seek_pos", "pos"), &AudioStreamPlayback::seek_pos);
	ObjectTypeDB::bind_method(_MD("get_pos"), &AudioStreamPlayback::get_pos);

	ObjectTypeDB::bind_method(_MD("get_length"), &AudioStreamPlayback::get_length);
	ObjectTypeDB::bind_method(_MD("get_channels"), &AudioStreamPlayback::get_channels);
	ObjectTypeDB::bind_method(_MD("get_mix_rate"), &AudioStreamPlayback::get_mix_rate);
	ObjectTypeDB::bind_method(_MD("get_minimum_buffer_size"), &AudioStreamPlayback::get_minimum_buffer_size);
}

void AudioStream::_bind_methods() {
}
