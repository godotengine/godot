/*************************************************************************/
/*  webrtc_data_channel_extension.cpp                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "webrtc_data_channel_extension.h"

void WebRTCDataChannelExtension::_bind_methods() {
	ADD_PROPERTY_DEFAULT("write_mode", WRITE_MODE_BINARY);

	GDVIRTUAL_BIND(_get_packet, "r_buffer", "r_buffer_size");
	GDVIRTUAL_BIND(_put_packet, "p_buffer", "p_buffer_size");
	GDVIRTUAL_BIND(_get_available_packet_count);
	GDVIRTUAL_BIND(_get_max_packet_size);

	GDVIRTUAL_BIND(_poll);
	GDVIRTUAL_BIND(_close);

	GDVIRTUAL_BIND(_set_write_mode, "p_write_mode");
	GDVIRTUAL_BIND(_get_write_mode);

	GDVIRTUAL_BIND(_was_string_packet);
	GDVIRTUAL_BIND(_get_ready_state);
	GDVIRTUAL_BIND(_get_label);
	GDVIRTUAL_BIND(_is_ordered);
	GDVIRTUAL_BIND(_get_id);
	GDVIRTUAL_BIND(_get_max_packet_life_time);
	GDVIRTUAL_BIND(_get_max_retransmits);
	GDVIRTUAL_BIND(_get_protocol);
	GDVIRTUAL_BIND(_is_negotiated);
	GDVIRTUAL_BIND(_get_buffered_amount);
}

int WebRTCDataChannelExtension::get_available_packet_count() const {
	int count;
	if (GDVIRTUAL_CALL(_get_available_packet_count, count)) {
		return count;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_available_packet_count is unimplemented!");
	return -1;
}

Error WebRTCDataChannelExtension::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	int err;
	if (GDVIRTUAL_CALL(_get_packet, r_buffer, &r_buffer_size, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_packet_native is unimplemented!");
	return FAILED;
}

Error WebRTCDataChannelExtension::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	int err;
	if (GDVIRTUAL_CALL(_put_packet, p_buffer, p_buffer_size, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_put_packet_native is unimplemented!");
	return FAILED;
}

int WebRTCDataChannelExtension::get_max_packet_size() const {
	int size;
	if (GDVIRTUAL_CALL(_get_max_packet_size, size)) {
		return size;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_max_packet_size is unimplemented!");
	return 0;
}

Error WebRTCDataChannelExtension::poll() {
	int err;
	if (GDVIRTUAL_CALL(_poll, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_poll is unimplemented!");
	return ERR_UNCONFIGURED;
}

void WebRTCDataChannelExtension::close() {
	if (GDVIRTUAL_CALL(_close)) {
		return;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_close is unimplemented!");
}

void WebRTCDataChannelExtension::set_write_mode(WriteMode p_mode) {
	if (GDVIRTUAL_CALL(_set_write_mode, p_mode)) {
		return;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_set_write_mode is unimplemented!");
}

WebRTCDataChannel::WriteMode WebRTCDataChannelExtension::get_write_mode() const {
	int mode;
	if (GDVIRTUAL_CALL(_get_write_mode, mode)) {
		return (WriteMode)mode;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_write_mode is unimplemented!");
	return WRITE_MODE_BINARY;
}

bool WebRTCDataChannelExtension::was_string_packet() const {
	bool was_string;
	if (GDVIRTUAL_CALL(_was_string_packet, was_string)) {
		return was_string;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_was_string_packet is unimplemented!");
	return false;
}

WebRTCDataChannel::ChannelState WebRTCDataChannelExtension::get_ready_state() const {
	int state;
	if (GDVIRTUAL_CALL(_get_ready_state, state)) {
		return (ChannelState)state;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_ready_state is unimplemented!");
	return STATE_CLOSED;
}

String WebRTCDataChannelExtension::get_label() const {
	String label;
	if (GDVIRTUAL_CALL(_get_label, label)) {
		return label;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_label is unimplemented!");
	return label;
}

bool WebRTCDataChannelExtension::is_ordered() const {
	bool ordered;
	if (GDVIRTUAL_CALL(_is_ordered, ordered)) {
		return ordered;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_is_ordered is unimplemented!");
	return false;
}

int WebRTCDataChannelExtension::get_id() const {
	int id;
	if (GDVIRTUAL_CALL(_get_id, id)) {
		return id;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_id is unimplemented!");
	return -1;
}

int WebRTCDataChannelExtension::get_max_packet_life_time() const {
	int lifetime;
	if (GDVIRTUAL_CALL(_get_max_packet_life_time, lifetime)) {
		return lifetime;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_max_packet_life_time is unimplemented!");
	return -1;
}

int WebRTCDataChannelExtension::get_max_retransmits() const {
	int retransmits;
	if (GDVIRTUAL_CALL(_get_max_retransmits, retransmits)) {
		return retransmits;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_max_retransmits is unimplemented!");
	return -1;
}

String WebRTCDataChannelExtension::get_protocol() const {
	String protocol;
	if (GDVIRTUAL_CALL(_get_protocol, protocol)) {
		return protocol;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_protocol is unimplemented!");
	return protocol;
}

bool WebRTCDataChannelExtension::is_negotiated() const {
	bool negotiated;
	if (GDVIRTUAL_CALL(_is_negotiated, negotiated)) {
		return negotiated;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_is_negotiated is unimplemented!");
	return false;
}

int WebRTCDataChannelExtension::get_buffered_amount() const {
	int amount;
	if (GDVIRTUAL_CALL(_get_buffered_amount, amount)) {
		return amount;
	}
	WARN_PRINT_ONCE("WebRTCDataChannelExtension::_get_buffered_amount is unimplemented!");
	return -1;
}
