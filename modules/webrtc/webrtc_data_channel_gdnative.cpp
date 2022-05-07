/*************************************************************************/
/*  webrtc_data_channel_gdnative.cpp                                     */
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

#ifdef WEBRTC_GDNATIVE_ENABLED

#include "webrtc_data_channel_gdnative.h"
#include "core/io/resource_loader.h"
#include "modules/gdnative/nativescript/nativescript.h"

void WebRTCDataChannelGDNative::_bind_methods() {
	ADD_PROPERTY_DEFAULT("write_mode", WRITE_MODE_BINARY);
}

WebRTCDataChannelGDNative::WebRTCDataChannelGDNative() {
	interface = nullptr;
}

WebRTCDataChannelGDNative::~WebRTCDataChannelGDNative() {
}

Error WebRTCDataChannelGDNative::poll() {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)interface->poll(interface->data);
}

void WebRTCDataChannelGDNative::close() {
	ERR_FAIL_COND(interface == nullptr);
	interface->close(interface->data);
}

void WebRTCDataChannelGDNative::set_write_mode(WriteMode p_mode) {
	ERR_FAIL_COND(interface == nullptr);
	interface->set_write_mode(interface->data, p_mode);
}

WebRTCDataChannel::WriteMode WebRTCDataChannelGDNative::get_write_mode() const {
	ERR_FAIL_COND_V(interface == nullptr, WRITE_MODE_BINARY);
	return (WriteMode)interface->get_write_mode(interface->data);
}

bool WebRTCDataChannelGDNative::was_string_packet() const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->was_string_packet(interface->data);
}

WebRTCDataChannel::ChannelState WebRTCDataChannelGDNative::get_ready_state() const {
	ERR_FAIL_COND_V(interface == nullptr, STATE_CLOSED);
	return (ChannelState)interface->get_ready_state(interface->data);
}

String WebRTCDataChannelGDNative::get_label() const {
	ERR_FAIL_COND_V(interface == nullptr, "");
	return String(interface->get_label(interface->data));
}

bool WebRTCDataChannelGDNative::is_ordered() const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->is_ordered(interface->data);
}

int WebRTCDataChannelGDNative::get_id() const {
	ERR_FAIL_COND_V(interface == nullptr, -1);
	return interface->get_id(interface->data);
}

int WebRTCDataChannelGDNative::get_max_packet_life_time() const {
	ERR_FAIL_COND_V(interface == nullptr, -1);
	return interface->get_max_packet_life_time(interface->data);
}

int WebRTCDataChannelGDNative::get_max_retransmits() const {
	ERR_FAIL_COND_V(interface == nullptr, -1);
	return interface->get_max_retransmits(interface->data);
}

String WebRTCDataChannelGDNative::get_protocol() const {
	ERR_FAIL_COND_V(interface == nullptr, "");
	return String(interface->get_protocol(interface->data));
}

bool WebRTCDataChannelGDNative::is_negotiated() const {
	ERR_FAIL_COND_V(interface == nullptr, false);
	return interface->is_negotiated(interface->data);
}

int WebRTCDataChannelGDNative::get_buffered_amount() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	ERR_FAIL_COND_V(interface->next == nullptr, 0);

	return ((godot_net_webrtc_data_channel_ext *)interface->next)->get_buffered_amount(interface->data);
}

Error WebRTCDataChannelGDNative::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)interface->get_packet(interface->data, r_buffer, &r_buffer_size);
}

Error WebRTCDataChannelGDNative::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(interface == nullptr, ERR_UNCONFIGURED);
	return (Error)interface->put_packet(interface->data, p_buffer, p_buffer_size);
}

int WebRTCDataChannelGDNative::get_max_packet_size() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->get_max_packet_size(interface->data);
}

int WebRTCDataChannelGDNative::get_available_packet_count() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->get_available_packet_count(interface->data);
}

void WebRTCDataChannelGDNative::set_native_webrtc_data_channel(const godot_net_webrtc_data_channel *p_impl) {
	interface = p_impl;
}

#endif // WEBRTC_GDNATIVE_ENABLED
