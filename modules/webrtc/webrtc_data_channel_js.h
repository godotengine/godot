/*************************************************************************/
/*  webrtc_data_channel_js.h                                             */
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

#ifndef WEBRTC_DATA_CHANNEL_JS_H
#define WEBRTC_DATA_CHANNEL_JS_H

#ifdef JAVASCRIPT_ENABLED

#include "webrtc_data_channel.h"

class WebRTCDataChannelJS : public WebRTCDataChannel {
	GDCLASS(WebRTCDataChannelJS, WebRTCDataChannel);

private:
	String _label;
	String _protocol;

	bool _was_string;
	WriteMode _write_mode;

	enum {
		PACKET_BUFFER_SIZE = 65536 - 5 // 4 bytes for the size, 1 for for type
	};

	int _js_id;
	RingBuffer<uint8_t> in_buffer;
	int queue_count;
	uint8_t packet_buffer[PACKET_BUFFER_SIZE];

	static void _on_open(void *p_obj);
	static void _on_close(void *p_obj);
	static void _on_error(void *p_obj);
	static void _on_message(void *p_obj, const uint8_t *p_data, int p_size, int p_is_string);

public:
	virtual void set_write_mode(WriteMode mode);
	virtual WriteMode get_write_mode() const;
	virtual bool was_string_packet() const;

	virtual ChannelState get_ready_state() const;
	virtual String get_label() const;
	virtual bool is_ordered() const;
	virtual int get_id() const;
	virtual int get_max_packet_life_time() const;
	virtual int get_max_retransmits() const;
	virtual String get_protocol() const;
	virtual bool is_negotiated() const;
	virtual int get_buffered_amount() const;

	virtual Error poll();
	virtual void close();

	/** Inherited from PacketPeer: **/
	virtual int get_available_packet_count() const;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size); ///< buffer is GONE after next get_packet
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size);

	virtual int get_max_packet_size() const;

	WebRTCDataChannelJS();
	WebRTCDataChannelJS(int js_id);
	~WebRTCDataChannelJS();
};

#endif // JAVASCRIPT_ENABLED

#endif // WEBRTC_DATA_CHANNEL_JS_H
