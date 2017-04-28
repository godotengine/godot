/*************************************************************************/
/*  packet_peer.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "packet_peer.h"

#include "global_config.h"
#include "io/marshalls.h"
/* helpers / binders */

PacketPeer::PacketPeer() {

	last_get_error = OK;
}

Error PacketPeer::get_packet_buffer(PoolVector<uint8_t> &r_buffer) const {

	const uint8_t *buffer;
	int buffer_size;
	Error err = get_packet(&buffer, buffer_size);
	if (err)
		return err;

	r_buffer.resize(buffer_size);
	if (buffer_size == 0)
		return OK;

	PoolVector<uint8_t>::Write w = r_buffer.write();
	for (int i = 0; i < buffer_size; i++)
		w[i] = buffer[i];

	return OK;
}

Error PacketPeer::put_packet_buffer(const PoolVector<uint8_t> &p_buffer) {

	int len = p_buffer.size();
	if (len == 0)
		return OK;

	PoolVector<uint8_t>::Read r = p_buffer.read();
	return put_packet(&r[0], len);
}

Error PacketPeer::get_var(Variant &r_variant) const {

	const uint8_t *buffer;
	int buffer_size;
	Error err = get_packet(&buffer, buffer_size);
	if (err)
		return err;

	return decode_variant(r_variant, buffer, buffer_size);
}

Error PacketPeer::put_var(const Variant &p_packet) {

	int len;
	Error err = encode_variant(p_packet, NULL, len); // compute len first
	if (err)
		return err;

	if (len == 0)
		return OK;

	uint8_t *buf = (uint8_t *)alloca(len);
	ERR_FAIL_COND_V(!buf, ERR_OUT_OF_MEMORY);
	err = encode_variant(p_packet, buf, len);
	ERR_FAIL_COND_V(err, err);

	return put_packet(buf, len);
}

Variant PacketPeer::_bnd_get_var() const {
	Variant var;
	get_var(var);

	return var;
};

Error PacketPeer::_put_packet(const PoolVector<uint8_t> &p_buffer) {
	return put_packet_buffer(p_buffer);
}
PoolVector<uint8_t> PacketPeer::_get_packet() const {

	PoolVector<uint8_t> raw;
	last_get_error = get_packet_buffer(raw);
	return raw;
}

Error PacketPeer::_get_packet_error() const {

	return last_get_error;
}

void PacketPeer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_var:Variant"), &PacketPeer::_bnd_get_var);
	ClassDB::bind_method(D_METHOD("put_var", "var:Variant"), &PacketPeer::put_var);
	ClassDB::bind_method(D_METHOD("get_packet"), &PacketPeer::_get_packet);
	ClassDB::bind_method(D_METHOD("put_packet:Error", "buffer"), &PacketPeer::_put_packet);
	ClassDB::bind_method(D_METHOD("get_packet_error:Error"), &PacketPeer::_get_packet_error);
	ClassDB::bind_method(D_METHOD("get_available_packet_count"), &PacketPeer::get_available_packet_count);
};

/***************/

void PacketPeerStream::_set_stream_peer(REF p_peer) {

	ERR_FAIL_COND(p_peer.is_null());
	set_stream_peer(p_peer);
}

void PacketPeerStream::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_stream_peer", "peer:StreamPeer"), &PacketPeerStream::_set_stream_peer);
}

Error PacketPeerStream::_poll_buffer() const {

	ERR_FAIL_COND_V(peer.is_null(), ERR_UNCONFIGURED);

	int read = 0;
	Error err = peer->get_partial_data(&temp_buffer[0], ring_buffer.space_left(), read);
	if (err)
		return err;
	if (read == 0)
		return OK;

	int w = ring_buffer.write(&temp_buffer[0], read);
	ERR_FAIL_COND_V(w != read, ERR_BUG);

	return OK;
}

int PacketPeerStream::get_available_packet_count() const {

	_poll_buffer();

	uint32_t remaining = ring_buffer.data_left();

	int ofs = 0;
	int count = 0;

	while (remaining >= 4) {

		uint8_t lbuf[4];
		ring_buffer.copy(lbuf, ofs, 4);
		uint32_t len = decode_uint32(lbuf);
		remaining -= 4;
		ofs += 4;
		if (len > remaining)
			break;
		remaining -= len;
		ofs += len;
		count++;
	}

	return count;
}

Error PacketPeerStream::get_packet(const uint8_t **r_buffer, int &r_buffer_size) const {

	ERR_FAIL_COND_V(peer.is_null(), ERR_UNCONFIGURED);
	_poll_buffer();

	int remaining = ring_buffer.data_left();
	ERR_FAIL_COND_V(remaining < 4, ERR_UNAVAILABLE);
	uint8_t lbuf[4];
	ring_buffer.copy(lbuf, 0, 4);
	remaining -= 4;
	uint32_t len = decode_uint32(lbuf);
	ERR_FAIL_COND_V(remaining < (int)len, ERR_UNAVAILABLE);

	ring_buffer.read(lbuf, 4); //get rid of first 4 bytes
	ring_buffer.read(&temp_buffer[0], len); // read packet

	*r_buffer = &temp_buffer[0];
	r_buffer_size = len;
	return OK;
}

Error PacketPeerStream::put_packet(const uint8_t *p_buffer, int p_buffer_size) {

	ERR_FAIL_COND_V(peer.is_null(), ERR_UNCONFIGURED);
	Error err = _poll_buffer(); //won't hurt to poll here too

	if (err)
		return err;

	if (p_buffer_size == 0)
		return OK;

	ERR_FAIL_COND_V(p_buffer_size < 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_buffer_size + 4 > temp_buffer.size(), ERR_INVALID_PARAMETER);

	encode_uint32(p_buffer_size, &temp_buffer[0]);
	uint8_t *dst = &temp_buffer[4];
	for (int i = 0; i < p_buffer_size; i++)
		dst[i] = p_buffer[i];

	return peer->put_data(&temp_buffer[0], p_buffer_size + 4);
}

int PacketPeerStream::get_max_packet_size() const {

	return temp_buffer.size();
}

void PacketPeerStream::set_stream_peer(const Ref<StreamPeer> &p_peer) {

	//ERR_FAIL_COND(p_peer.is_null());

	if (p_peer.ptr() != peer.ptr()) {
		ring_buffer.advance_read(ring_buffer.data_left()); // reset the ring buffer
	};

	peer = p_peer;
}

void PacketPeerStream::set_input_buffer_max_size(int p_max_size) {

	//warning may lose packets
	ERR_EXPLAIN("Buffer in use, resizing would cause loss of data");
	ERR_FAIL_COND(ring_buffer.data_left());
	ring_buffer.resize(nearest_shift(p_max_size + 4));
	temp_buffer.resize(nearest_power_of_2(p_max_size + 4));
}

PacketPeerStream::PacketPeerStream() {

	int rbsize = GLOBAL_GET("network/packets/packet_stream_peer_max_buffer_po2");

	ring_buffer.resize(rbsize);
	temp_buffer.resize(1 << rbsize);
}
