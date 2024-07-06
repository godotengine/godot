/**************************************************************************/
/*  webrtc_data_channel.cpp                                               */
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

#include "webrtc_data_channel.h"

#include "core/config/project_settings.h"

void WebRTCDataChannel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("poll"), &WebRTCDataChannel::poll);
	ClassDB::bind_method(D_METHOD("close"), &WebRTCDataChannel::close);

	ClassDB::bind_method(D_METHOD("was_string_packet"), &WebRTCDataChannel::was_string_packet);
	ClassDB::bind_method(D_METHOD("set_write_mode", "write_mode"), &WebRTCDataChannel::set_write_mode);
	ClassDB::bind_method(D_METHOD("get_write_mode"), &WebRTCDataChannel::get_write_mode);
	ClassDB::bind_method(D_METHOD("get_ready_state"), &WebRTCDataChannel::get_ready_state);
	ClassDB::bind_method(D_METHOD("get_label"), &WebRTCDataChannel::get_label);
	ClassDB::bind_method(D_METHOD("is_ordered"), &WebRTCDataChannel::is_ordered);
	ClassDB::bind_method(D_METHOD("get_id"), &WebRTCDataChannel::get_id);
	ClassDB::bind_method(D_METHOD("get_max_packet_life_time"), &WebRTCDataChannel::get_max_packet_life_time);
	ClassDB::bind_method(D_METHOD("get_max_retransmits"), &WebRTCDataChannel::get_max_retransmits);
	ClassDB::bind_method(D_METHOD("get_protocol"), &WebRTCDataChannel::get_protocol);
	ClassDB::bind_method(D_METHOD("is_negotiated"), &WebRTCDataChannel::is_negotiated);
	ClassDB::bind_method(D_METHOD("get_buffered_amount"), &WebRTCDataChannel::get_buffered_amount);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "write_mode", PROPERTY_HINT_ENUM), "set_write_mode", "get_write_mode");

	BIND_ENUM_CONSTANT(WRITE_MODE_TEXT);
	BIND_ENUM_CONSTANT(WRITE_MODE_BINARY);

	BIND_ENUM_CONSTANT(STATE_CONNECTING);
	BIND_ENUM_CONSTANT(STATE_OPEN);
	BIND_ENUM_CONSTANT(STATE_CLOSING);
	BIND_ENUM_CONSTANT(STATE_CLOSED);
}

WebRTCDataChannel::WebRTCDataChannel() {
	_in_buffer_shift = nearest_shift((int)GLOBAL_GET("network/limits/webrtc/max_channel_in_buffer_kb") - 1) + 10;
}

WebRTCDataChannel::~WebRTCDataChannel() {
}
