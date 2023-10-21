/**************************************************************************/
/*  webrtc_lib_data_channel.h                                             */
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

#ifndef WEBRTC_LIB_DATA_CHANNEL_H
#define WEBRTC_LIB_DATA_CHANNEL_H

#ifdef ENABLE_LIBDATACHANNEL
#include <utility>

#include "core/os/mutex.h"
#include "core/templates/vector.h"
#include "webrtc_data_channel.h"

#include "rtc/rtc.hpp"

class WebRTCLibDataChannel : public WebRTCDataChannel {
	GDCLASS(WebRTCLibDataChannel, WebRTCDataChannel);

private:
	Mutex mutex;
	int packet_queue_head = 0;
	Vector<Vector<uint8_t>> packet_queue;
	Vector<uint8_t> current_packet;
	std::shared_ptr<rtc::DataChannel> channel = nullptr;

	WriteMode write_mode = WRITE_MODE_BINARY;
	ChannelState channel_state = STATE_CONNECTING;
	bool negotiated = false;

	void queue_packet(const uint8_t *data, uint32_t size, WriteMode p_message_type);

protected:
	static void _bind_methods() {}

public:
	/* PacketPeer */
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) override;
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) override;
	virtual int32_t get_available_packet_count() const override;
	virtual int32_t get_max_packet_size() const override;

	/* WebRTCDataChannel */
	Error poll() override;
	void close() override;

	void set_write_mode(WriteMode p_mode) override;
	WriteMode get_write_mode() const override;
	bool was_string_packet() const override;

	ChannelState get_ready_state() const override;
	String get_label() const override;
	bool is_ordered() const override;
	int32_t get_id() const override;
	int32_t get_max_packet_life_time() const override;
	int32_t get_max_retransmits() const override;
	String get_protocol() const override;
	bool is_negotiated() const override;
	int32_t get_buffered_amount() const override;

	void bind_channel(std::shared_ptr<rtc::DataChannel> p_channel, bool p_negotiated);

	WebRTCLibDataChannel();
	~WebRTCLibDataChannel();
};

#endif // ENABLE_LIBDATACHANNEL

#endif // WEBRTC_LIB_DATA_CHANNEL_H
