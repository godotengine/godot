/*************************************************************************/
/*  webrtc_peer.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef WEBRTC_PEER_H
#define WEBRTC_PEER_H

#include "core/io/packet_peer.h"

class WebRTCPeer : public PacketPeer {
	GDCLASS(WebRTCPeer, PacketPeer);

public:
	enum WriteMode {
		WRITE_MODE_TEXT,
		WRITE_MODE_BINARY,
	};

	enum ConnectionState {
		STATE_NEW,
		STATE_CONNECTING,
		STATE_CONNECTED,
		STATE_DISCONNECTED,
		STATE_FAILED,
		STATE_CLOSED
	};

protected:
	static void _bind_methods();
	static WebRTCPeer *(*_create)();

public:
	virtual void set_write_mode(WriteMode mode) = 0;
	virtual WriteMode get_write_mode() const = 0;
	virtual bool was_string_packet() const = 0;
	virtual ConnectionState get_connection_state() const = 0;

	virtual Error create_offer() = 0;
	virtual Error set_remote_description(String type, String sdp) = 0;
	virtual Error set_local_description(String type, String sdp) = 0;
	virtual Error add_ice_candidate(String sdpMidName, int sdpMlineIndexName, String sdpName) = 0;
	virtual Error poll() = 0;

	/** Inherited from PacketPeer: **/
	virtual int get_available_packet_count() const = 0;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) = 0; ///< buffer is GONE after next get_packet
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) = 0;

	virtual int get_max_packet_size() const = 0;

	static Ref<WebRTCPeer> create_ref();
	static WebRTCPeer *create();

	WebRTCPeer();
	~WebRTCPeer();
};

VARIANT_ENUM_CAST(WebRTCPeer::WriteMode);
VARIANT_ENUM_CAST(WebRTCPeer::ConnectionState);
#endif // WEBRTC_PEER_H
