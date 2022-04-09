/*************************************************************************/
/*  websocket_peer.h                                                     */
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

#ifndef WEBSOCKETPEER_H
#define WEBSOCKETPEER_H

#include "core/error/error_list.h"
#include "core/io/packet_peer.h"
#include "websocket_macros.h"

class WebSocketPeer : public PacketPeer {
	GDCLASS(WebSocketPeer, PacketPeer);
	GDCICLASS(WebSocketPeer);

public:
	enum WriteMode {
		WRITE_MODE_TEXT,
		WRITE_MODE_BINARY,
	};

protected:
	static void _bind_methods();

public:
	virtual WriteMode get_write_mode() const = 0;
	virtual void set_write_mode(WriteMode p_mode) = 0;

	virtual void close(int p_code = 1000, String p_reason = "") = 0;

	virtual bool is_connected_to_host() const = 0;
	virtual IPAddress get_connected_host() const = 0;
	virtual uint16_t get_connected_port() const = 0;
	virtual bool was_string_packet() const = 0;
	virtual void set_no_delay(bool p_enabled) = 0;
	virtual int get_current_outbound_buffered_amount() const = 0;

	WebSocketPeer();
	~WebSocketPeer();
};

VARIANT_ENUM_CAST(WebSocketPeer::WriteMode);
#endif // WEBSOCKETPEER_H
