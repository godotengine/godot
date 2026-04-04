/**************************************************************************/
/*  web_rtc_data_channel_extension.cpp                                    */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/web_rtc_data_channel_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Error WebRTCDataChannelExtension::_get_packet(const uint8_t **r_buffer, int32_t *r_buffer_size) {
	return Error(0);
}

Error WebRTCDataChannelExtension::_put_packet(const uint8_t *p_buffer, int32_t p_buffer_size) {
	return Error(0);
}

int32_t WebRTCDataChannelExtension::_get_available_packet_count() const {
	return 0;
}

int32_t WebRTCDataChannelExtension::_get_max_packet_size() const {
	return 0;
}

Error WebRTCDataChannelExtension::_poll() {
	return Error(0);
}

void WebRTCDataChannelExtension::_close() {}

void WebRTCDataChannelExtension::_set_write_mode(WebRTCDataChannel::WriteMode p_write_mode) {}

WebRTCDataChannel::WriteMode WebRTCDataChannelExtension::_get_write_mode() const {
	return WebRTCDataChannel::WriteMode(0);
}

bool WebRTCDataChannelExtension::_was_string_packet() const {
	return false;
}

WebRTCDataChannel::ChannelState WebRTCDataChannelExtension::_get_ready_state() const {
	return WebRTCDataChannel::ChannelState(0);
}

String WebRTCDataChannelExtension::_get_label() const {
	return String();
}

bool WebRTCDataChannelExtension::_is_ordered() const {
	return false;
}

int32_t WebRTCDataChannelExtension::_get_id() const {
	return 0;
}

int32_t WebRTCDataChannelExtension::_get_max_packet_life_time() const {
	return 0;
}

int32_t WebRTCDataChannelExtension::_get_max_retransmits() const {
	return 0;
}

String WebRTCDataChannelExtension::_get_protocol() const {
	return String();
}

bool WebRTCDataChannelExtension::_is_negotiated() const {
	return false;
}

int32_t WebRTCDataChannelExtension::_get_buffered_amount() const {
	return 0;
}

} // namespace godot
