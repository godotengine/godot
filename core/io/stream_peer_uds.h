/**************************************************************************/
/*  stream_peer_uds.h                                                     */
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

#pragma once

#include "core/io/stream_peer.h"
#include "core/io/uds_socket.h"

class StreamPeerUDS : public StreamPeer {
	GDCLASS(StreamPeerUDS, StreamPeer);

public:
	enum Status {
		STATUS_NONE,
		STATUS_CONNECTING,
		STATUS_CONNECTED,
		STATUS_ERROR,
	};

protected:
	Ref<UDSSocket> _sock;
	uint64_t timeout = 0;
	Status status = STATUS_NONE;
	String path;

	Error write(const uint8_t *p_data, int p_bytes, int &r_sent, bool p_block);
	Error read(uint8_t *p_buffer, int p_bytes, int &r_received, bool p_block);

	static void _bind_methods();

public:
	void accept_socket(Ref<UDSSocket> p_sock);

	Error bind(const String &p_path);
	Error connect_to_host(const String &p_path);
	const String &get_connected_path() const { return path; }
	void disconnect_from_host();

	virtual int get_available_bytes() const override;
	Status get_status() const;

	// Poll socket updating its state.
	Error poll();

	// Wait or check for writable, readable.
	Error wait(UDSSocket::PollType p_type, int p_timeout = 0);

	// Read/Write from StreamPeer
	virtual Error put_data(const uint8_t *p_data, int p_bytes) override;
	virtual Error put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) override;
	virtual Error get_data(uint8_t *p_buffer, int p_bytes) override;
	virtual Error get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) override;

	StreamPeerUDS();
	~StreamPeerUDS();
};

VARIANT_ENUM_CAST(StreamPeerUDS::Status);
