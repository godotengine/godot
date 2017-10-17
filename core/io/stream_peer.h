/*************************************************************************/
/*  stream_peer.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef STREAM_PEER_H
#define STREAM_PEER_H

#include "reference.h"

class StreamPeer : public Reference {
	GDCLASS(StreamPeer, Reference);
	OBJ_CATEGORY("Networking");

protected:
	static void _bind_methods();

	//bind helpers
	Error _put_data(const PoolVector<uint8_t> &p_data);
	Array _put_partial_data(const PoolVector<uint8_t> &p_data);

	Array _get_data(int p_bytes);
	Array _get_partial_data(int p_bytes);

	bool big_endian;

public:
	virtual Error put_data(const uint8_t *p_data, int p_bytes) = 0; ///< put a whole chunk of data, blocking until it sent
	virtual Error put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) = 0; ///< put as much data as possible, without blocking.

	virtual Error get_data(uint8_t *p_buffer, int p_bytes) = 0; ///< read p_bytes of data, if p_bytes > available, it will block
	virtual Error get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) = 0; ///< read as much data as p_bytes into buffer, if less was read, return in r_received

	virtual int get_available_bytes() const = 0;

	void set_big_endian(bool p_enable);
	bool is_big_endian_enabled() const;

	void put_8(int8_t p_val);
	void put_u8(uint8_t p_val);
	void put_16(int16_t p_val);
	void put_u16(uint16_t p_val);
	void put_32(int32_t p_val);
	void put_u32(uint32_t p_val);
	void put_64(int64_t p_val);
	void put_u64(uint64_t p_val);
	void put_float(float p_val);
	void put_double(double p_val);
	void put_utf8_string(const String &p_string);
	void put_var(const Variant &p_variant);

	uint8_t get_u8();
	int8_t get_8();
	uint16_t get_u16();
	int16_t get_16();
	uint32_t get_u32();
	int32_t get_32();
	uint64_t get_u64();
	int64_t get_64();
	float get_float();
	float get_double();
	String get_string(int p_bytes);
	String get_utf8_string(int p_bytes);
	Variant get_var();

	StreamPeer() { big_endian = false; }
};

class StreamPeerBuffer : public StreamPeer {

	GDCLASS(StreamPeerBuffer, StreamPeer);

	PoolVector<uint8_t> data;
	int pointer;

protected:
	static void _bind_methods();

public:
	Error put_data(const uint8_t *p_data, int p_bytes);
	Error put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent);

	Error get_data(uint8_t *p_buffer, int p_bytes);
	Error get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received);

	virtual int get_available_bytes() const;

	void seek(int p_pos);
	int get_size() const;
	int get_position() const;
	void resize(int p_size);

	void set_data_array(const PoolVector<uint8_t> &p_data);
	PoolVector<uint8_t> get_data_array() const;

	void clear();

	Ref<StreamPeerBuffer> duplicate() const;

	StreamPeerBuffer();
};

#endif // STREAM_PEER_H
