/**************************************************************************/
/*  webrtc_data_channel.h                                                 */
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

#ifndef WEBRTC_DATA_CHANNEL_H
#define WEBRTC_DATA_CHANNEL_H

#include "core/io/packet_peer.h"

#define WRTC_IN_BUF PNAME("network/limits/webrtc/max_channel_in_buffer_kb")

class WebRTCDataChannel : public PacketPeer {
	GDCLASS(WebRTCDataChannel, PacketPeer);

public:
	enum WriteMode {
		WRITE_MODE_TEXT,
		WRITE_MODE_BINARY,
	};

	enum ChannelState {
		STATE_CONNECTING,
		STATE_OPEN,
		STATE_CLOSING,
		STATE_CLOSED
	};

protected:
	unsigned int _in_buffer_shift;

	static void _bind_methods();

public:
	virtual void set_write_mode(WriteMode mode) = 0;
	virtual WriteMode get_write_mode() const = 0;
	virtual bool was_string_packet() const = 0;

	virtual ChannelState get_ready_state() const = 0;
	virtual String get_label() const = 0;
	virtual bool is_ordered() const = 0;
	virtual int get_id() const = 0;
	virtual int get_max_packet_life_time() const = 0;
	virtual int get_max_retransmits() const = 0;
	virtual String get_protocol() const = 0;
	virtual bool is_negotiated() const = 0;

	virtual int get_buffered_amount() const = 0;

	virtual Error poll() = 0;
	virtual void close() = 0;

	/** Inherited from PacketPeer: **/
	virtual int get_available_packet_count() const = 0;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) = 0; ///< buffer is GONE after next get_packet
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) = 0;

	virtual int get_max_packet_size() const = 0;

	WebRTCDataChannel();
	~WebRTCDataChannel();
};

VARIANT_ENUM_CAST(WebRTCDataChannel::WriteMode);
VARIANT_ENUM_CAST(WebRTCDataChannel::ChannelState);

#endif // WEBRTC_DATA_CHANNEL_H
