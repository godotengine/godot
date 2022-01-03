/*************************************************************************/
/*  emws_peer.h                                                          */
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

#ifndef EMWSPEER_H
#define EMWSPEER_H

#ifdef JAVASCRIPT_ENABLED

#include "core/error/error_list.h"
#include "core/io/packet_peer.h"
#include "core/templates/ring_buffer.h"
#include "emscripten.h"
#include "packet_buffer.h"
#include "websocket_peer.h"

extern "C" {
typedef void (*WSOnOpen)(void *p_ref, char *p_protocol);
typedef void (*WSOnMessage)(void *p_ref, const uint8_t *p_buf, int p_buf_len, int p_is_string);
typedef void (*WSOnClose)(void *p_ref, int p_code, const char *p_reason, int p_is_clean);
typedef void (*WSOnError)(void *p_ref);

extern int godot_js_websocket_create(void *p_ref, const char *p_url, const char *p_proto, WSOnOpen p_on_open, WSOnMessage p_on_message, WSOnError p_on_error, WSOnClose p_on_close);
extern int godot_js_websocket_send(int p_id, const uint8_t *p_buf, int p_buf_len, int p_raw);
extern int godot_js_websocket_buffered_amount(int p_id);
extern void godot_js_websocket_close(int p_id, int p_code, const char *p_reason);
extern void godot_js_websocket_destroy(int p_id);
}

class EMWSPeer : public WebSocketPeer {
	GDCIIMPL(EMWSPeer, WebSocketPeer);

private:
	int peer_sock = -1;
	WriteMode write_mode = WRITE_MODE_BINARY;

	Vector<uint8_t> _packet_buffer;
	PacketBuffer<uint8_t> _in_buffer;
	uint8_t _is_string = 0;
	int _out_buf_size = 0;

public:
	Error read_msg(const uint8_t *p_data, uint32_t p_size, bool p_is_string);
	void set_sock(int p_sock, unsigned int p_in_buf_size, unsigned int p_in_pkt_size, unsigned int p_out_buf_size);
	virtual int get_available_packet_count() const;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size);
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size);
	virtual int get_max_packet_size() const { return _packet_buffer.size(); };
	virtual int get_current_outbound_buffered_amount() const;

	virtual void close(int p_code = 1000, String p_reason = "");
	virtual bool is_connected_to_host() const;
	virtual IPAddress get_connected_host() const;
	virtual uint16_t get_connected_port() const;

	virtual WriteMode get_write_mode() const;
	virtual void set_write_mode(WriteMode p_mode);
	virtual bool was_string_packet() const;
	virtual void set_no_delay(bool p_enabled);

	EMWSPeer();
	~EMWSPeer();
};

#endif // JAVASCRIPT_ENABLED

#endif // LSWPEER_H
