/**************************************************************************/
/*  microphone_feed.cpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "microphone_feed.h"
#include "core/config/project_settings.h"
#include "servers/audio_server.h"

void MicrophoneFeed::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &MicrophoneFeed::get_name);
	ClassDB::bind_method(D_METHOD("set_name", "name"), &MicrophoneFeed::set_name);

	ClassDB::bind_method(D_METHOD("is_active"), &MicrophoneFeed::is_active);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &MicrophoneFeed::set_active);

	ClassDB::bind_method(D_METHOD("get_frames_available"), &MicrophoneFeed::get_frames_available);
	ClassDB::bind_method(D_METHOD("get_frames", "frames"), &MicrophoneFeed::get_frames);
	ClassDB::bind_method(D_METHOD("get_buffer_length_frames"), &MicrophoneFeed::get_buffer_length_frames);

	ADD_GROUP("Feed", "feed_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "feed_is_active"), "set_active", "is_active");
}

String MicrophoneFeed::get_name() const {
	return name;
}

void MicrophoneFeed::set_name(String p_name) {
	name = p_name;
}

MicrophoneFeed::MicrophoneFeed() {
	// initialize our feed
	name = "???";
}

MicrophoneFeed::MicrophoneFeed(String p_name) {
	// initialize our feed
	name = p_name;
}

MicrophoneFeed::~MicrophoneFeed() {
}

bool MicrophoneFeed::is_active() const {
	return active;
}

Error MicrophoneFeed::start_microphone() {
	if (!GLOBAL_GET("audio/driver/enable_input")) {
		WARN_PRINT("You must enable the project setting \"audio/driver/enable_input\" to use audio capture.");
		return FAILED;
	}

	microphone_buffer_ofs = 0;
	return AudioDriver::get_singleton()->input_start();
}

Error MicrophoneFeed::stop_microphone() {
	return AudioDriver::get_singleton()->input_stop();
}

void MicrophoneFeed::set_active(bool p_is_active) {
	Error e = OK;
	if (p_is_active == active) {
		// all good
	} else if (p_is_active) {
		// attempt to activate this feed
		e = start_microphone();
		if (e == OK) {
			active = true;
		}
	} else {
		// just deactivate it
		e = stop_microphone();
		active = false;
	}
	if (e != OK) {
		WARN_PRINT("MicrophoneFeed.set_active(" + itos(p_is_active) + ") encountered error " + itos(e) + ".");
	}
}

int MicrophoneFeed::get_frames_available() {
	AudioDriver *ad = AudioDriver::get_singleton();
	ad->lock();
	int input_position = ad->get_input_position();
	if (input_position < microphone_buffer_ofs) {
		int buffsize = ad->get_input_buffer().size();
		input_position += buffsize;
	}
	ad->unlock();
	return (input_position - microphone_buffer_ofs) / 2;
}

int MicrophoneFeed::get_buffer_length_frames() {
	AudioDriver *ad = AudioDriver::get_singleton();
	ad->lock();
	int buffsize = ad->get_input_buffer().size();
	ad->unlock();
	return buffsize / 2;
}

PackedVector2Array MicrophoneFeed::get_frames(int p_frames) {
	PackedVector2Array ret;
	AudioDriver *ad = AudioDriver::get_singleton();
	ad->lock();
	int input_position = ad->get_input_position();
	Vector<int32_t> buf = ad->get_input_buffer();
	if (input_position < microphone_buffer_ofs) {
		input_position += buf.size();
	}
	if ((microphone_buffer_ofs + p_frames * 2 <= input_position) && (p_frames >= 0)) {
		ret.resize(p_frames);
		for (int i = 0; i < p_frames; i++) {
			float l = (buf[microphone_buffer_ofs++] >> 16) / 32768.f;
			if (microphone_buffer_ofs >= buf.size()) {
				microphone_buffer_ofs = 0;
			}
			float r = (buf[microphone_buffer_ofs++] >> 16) / 32768.f;
			if (microphone_buffer_ofs >= buf.size()) {
				microphone_buffer_ofs = 0;
			}
			ret.write[i] = Vector2(l, r);
		}
	}
	ad->unlock();
	return ret;
}
