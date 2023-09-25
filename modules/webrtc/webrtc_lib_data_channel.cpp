/**************************************************************************/
/*  webrtc_lib_data_channel.cpp                                           */
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

#ifdef ENABLE_LIBDATACHANNEL

#include "webrtc_lib_data_channel.h"
#include "rtc/exception_wrapper_godot.hpp"

#include <stdio.h>
#include <string.h>
#include <cstring>

// DataChannel
void WebRTCLibDataChannel::bind_channel(std::shared_ptr<rtc::DataChannel> p_channel, bool p_negotiated) {
	ERR_FAIL_COND(!p_channel);

	channel = p_channel;
	negotiated = p_negotiated;

	// Binding this should be fine as long as we call close when going out of scope.
	p_channel->onMessage([this](auto message) {
		if (std::holds_alternative<rtc::string>(message)) {
			rtc::string str = std::get<rtc::string>(message);
			queue_packet(reinterpret_cast<const uint8_t *>(str.c_str()), str.size(), WRITE_MODE_TEXT);
		} else if (std::holds_alternative<rtc::binary>(message)) {
			rtc::binary bin = std::get<rtc::binary>(message);
			queue_packet(reinterpret_cast<const uint8_t *>(&bin[0]), bin.size(), WRITE_MODE_BINARY);
		} else {
			ERR_PRINT("Message parsing bug. Unknown message type.");
		}
	});
	p_channel->onOpen([this]() {
		channel_state = STATE_OPEN;
	});
	p_channel->onClosed([this]() {
		channel_state = STATE_CLOSED;
	});
	p_channel->onError([](auto error) {
		ERR_PRINT("Channel Error: " + String(std::string(error).c_str()));
	});
}

void WebRTCLibDataChannel::queue_packet(const uint8_t *data, uint32_t size, WriteMode p_message_type) {
	MutexLock lock(mutex);

	Vector<uint8_t> packet;
	packet.resize(size + 1);
	packet.ptrw()[0] = (uint8_t)p_message_type;
	memcpy(packet.ptrw() + 1, data, size);

	if (packet_queue_head != 0) {
		for (int i = packet_queue_head; i < packet_queue.size(); i++) {
			packet_queue.set(i, packet_queue[i - packet_queue_head]);
		}
		packet_queue.resize(packet_queue.size() - packet_queue_head + 1);
		packet_queue.set(packet_queue.size() - 1, packet);
		packet_queue_head = 0;
	} else {
		packet_queue.push_back(packet);
	}
}

void WebRTCLibDataChannel::set_write_mode(WriteMode p_mode) {
	ERR_FAIL_COND(p_mode != WRITE_MODE_TEXT && p_mode != WRITE_MODE_BINARY);
	write_mode = p_mode;
}

WebRTCDataChannel::WriteMode WebRTCLibDataChannel::get_write_mode() const {
	return write_mode;
}

bool WebRTCLibDataChannel::was_string_packet() const {
	return current_packet.ptr()[0] == (uint8_t)WRITE_MODE_TEXT;
}

WebRTCDataChannel::ChannelState WebRTCLibDataChannel::get_ready_state() const {
	ERR_FAIL_COND_V(!channel, STATE_CLOSED);
	return channel_state;
}

String WebRTCLibDataChannel::get_label() const {
	ERR_FAIL_COND_V(!channel, "");
	return channel->label().c_str();
}

bool WebRTCLibDataChannel::is_ordered() const {
	ERR_FAIL_COND_V(!channel, false);
	return channel->reliability().unordered == false;
}

int32_t WebRTCLibDataChannel::get_id() const {
	ERR_FAIL_COND_V(!channel, -1);
	return channel->id().value_or(-1);
}

int32_t WebRTCLibDataChannel::get_max_packet_life_time() const {
	ERR_FAIL_COND_V(!channel, 0);
	return channel->reliability().type == rtc::Reliability::Type::Timed ? std::get<std::chrono::milliseconds>(channel->reliability().rexmit).count() : -1;
}

int32_t WebRTCLibDataChannel::get_max_retransmits() const {
	ERR_FAIL_COND_V(!channel, 0);
	return channel->reliability().type == rtc::Reliability::Type::Rexmit ? std::get<int>(channel->reliability().rexmit) : -1;
}

String WebRTCLibDataChannel::get_protocol() const {
	ERR_FAIL_COND_V(!channel, "");
	return channel->protocol().c_str();
}

bool WebRTCLibDataChannel::is_negotiated() const {
	ERR_FAIL_COND_V(!channel, false);
	return negotiated;
}

int32_t WebRTCLibDataChannel::get_buffered_amount() const {
	ERR_FAIL_COND_V(!channel, 0);
	return channel->bufferedAmount();
}

Error WebRTCLibDataChannel::poll() {
	return OK;
}

void WebRTCLibDataChannel::close() {
	LibDataChannelExceptionWrapper::close_data_channel(channel);
}

Error WebRTCLibDataChannel::get_packet(const uint8_t **r_buffer, int &r_len) {
	MutexLock lock(mutex);

	ERR_FAIL_COND_V(packet_queue_head >= packet_queue.size(), ERR_UNAVAILABLE);

	// Update current packet and pop queue
	current_packet = packet_queue[packet_queue_head];
	packet_queue.set(packet_queue_head, Vector<uint8_t>());
	packet_queue_head++;

	// Set out buffer and size (buffer will be gone at next get_packet or close)
	*r_buffer = current_packet.ptrw() + 1;
	r_len = current_packet.size() - 1;

	return OK;
}

Error WebRTCLibDataChannel::put_packet(const uint8_t *p_buffer, int p_len) {
	ERR_FAIL_COND_V(!channel, FAILED);
	ERR_FAIL_COND_V(channel->isClosed(), FAILED);
	std::string error;
	// Only text and binary modes exist for now, so we made it a boolean.
	if (!LibDataChannelExceptionWrapper::put_packet(channel, p_buffer, p_len, write_mode == WRITE_MODE_TEXT, error)) {
		ERR_FAIL_V_MSG(FAILED, error.c_str());
	}
	return OK;
}

int32_t WebRTCLibDataChannel::get_available_packet_count() const {
	return packet_queue.size() - packet_queue_head;
}

int32_t WebRTCLibDataChannel::get_max_packet_size() const {
	return 16384; // See RFC-8831 section 6.6: https://datatracker.ietf.org/doc/rfc8831/
}

WebRTCLibDataChannel::WebRTCLibDataChannel() {
}

WebRTCLibDataChannel::~WebRTCLibDataChannel() {
	close();
	channel = nullptr;
}

#endif // ENABLE_LIBDATACHANNEL
