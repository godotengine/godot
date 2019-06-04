/*************************************************************************/
/*  stream_peer.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "stream_peer.h"
#include "io/marshalls.h"

Error StreamPeer::_put_data(const DVector<uint8_t> &p_data) {

	int len = p_data.size();
	if (len == 0)
		return OK;
	DVector<uint8_t>::Read r = p_data.read();
	return put_data(&r[0], len);
}

Array StreamPeer::_put_partial_data(const DVector<uint8_t> &p_data) {

	Array ret;

	int len = p_data.size();
	if (len == 0) {
		ret.push_back(OK);
		ret.push_back(0);
		return ret;
	}

	DVector<uint8_t>::Read r = p_data.read();
	int sent;
	Error err = put_partial_data(&r[0], len, sent);

	if (err != OK) {
		sent = 0;
	}
	ret.push_back(err);
	ret.push_back(sent);
	return ret;
}

Array StreamPeer::_get_data(int p_bytes) {

	Array ret;

	DVector<uint8_t> data;
	data.resize(p_bytes);
	if (data.size() != p_bytes) {

		ret.push_back(ERR_OUT_OF_MEMORY);
		ret.push_back(DVector<uint8_t>());
		return ret;
	}

	DVector<uint8_t>::Write w = data.write();
	Error err = get_data(&w[0], p_bytes);
	w = DVector<uint8_t>::Write();
	ret.push_back(err);
	ret.push_back(data);
	return ret;
}

Array StreamPeer::_get_partial_data(int p_bytes) {

	Array ret;

	DVector<uint8_t> data;
	data.resize(p_bytes);
	if (data.size() != p_bytes) {

		ret.push_back(ERR_OUT_OF_MEMORY);
		ret.push_back(DVector<uint8_t>());
		return ret;
	}

	DVector<uint8_t>::Write w = data.write();
	int received;
	Error err = get_partial_data(&w[0], p_bytes, received);
	w = DVector<uint8_t>::Write();

	if (err != OK) {
		data.resize(0);
	} else if (received != data.size()) {

		data.resize(received);
	}

	ret.push_back(err);
	ret.push_back(data);
	return ret;
}

void StreamPeer::set_big_endian(bool p_enable) {

	big_endian = p_enable;
}

bool StreamPeer::is_big_endian_enabled() const {

	return big_endian;
}

void StreamPeer::put_u8(uint8_t p_val) {
	put_data((const uint8_t *)&p_val, 1);
}

void StreamPeer::put_8(int8_t p_val) {

	put_data((const uint8_t *)&p_val, 1);
}
void StreamPeer::put_u16(uint16_t p_val) {

	if (big_endian) {
		p_val = BSWAP16(p_val);
	}
	uint8_t buf[2];
	encode_uint16(p_val, buf);
	put_data(buf, 2);
}
void StreamPeer::put_16(int16_t p_val) {

	if (big_endian) {
		p_val = BSWAP16(p_val);
	}
	uint8_t buf[2];
	encode_uint16(p_val, buf);
	put_data(buf, 2);
}
void StreamPeer::put_u32(uint32_t p_val) {

	if (big_endian) {
		p_val = BSWAP32(p_val);
	}
	uint8_t buf[4];
	encode_uint32(p_val, buf);
	put_data(buf, 4);
}
void StreamPeer::put_32(int32_t p_val) {

	if (big_endian) {
		p_val = BSWAP32(p_val);
	}
	uint8_t buf[4];
	encode_uint32(p_val, buf);
	put_data(buf, 4);
}
void StreamPeer::put_u64(uint64_t p_val) {

	if (big_endian) {
		p_val = BSWAP64(p_val);
	}
	uint8_t buf[8];
	encode_uint64(p_val, buf);
	put_data(buf, 8);
}
void StreamPeer::put_64(int64_t p_val) {

	if (big_endian) {
		p_val = BSWAP64(p_val);
	}
	uint8_t buf[8];
	encode_uint64(p_val, buf);
	put_data(buf, 8);
}
void StreamPeer::put_float(float p_val) {

	uint8_t buf[4];

	encode_float(p_val, buf);
	if (big_endian) {
		uint32_t *p32 = (uint32_t *)buf;
		*p32 = BSWAP32(*p32);
	}

	put_data(buf, 4);
}
void StreamPeer::put_double(double p_val) {

	uint8_t buf[8];
	encode_double(p_val, buf);
	if (big_endian) {
		uint64_t *p64 = (uint64_t *)buf;
		*p64 = BSWAP64(*p64);
	}
	put_data(buf, 8);
}
void StreamPeer::put_utf8_string(const String &p_string) {

	CharString cs = p_string.utf8();
	put_data((const uint8_t *)cs.get_data(), cs.length());
}
void StreamPeer::put_var(const Variant &p_variant) {

	int len = 0;
	Vector<uint8_t> buf;
	encode_variant(p_variant, NULL, len);
	buf.resize(len);
	put_32(len);
	encode_variant(p_variant, buf.ptr(), len);
	put_data(buf.ptr(), buf.size());
}

uint8_t StreamPeer::get_u8() {

	uint8_t buf[1];
	get_data(buf, 1);
	return buf[0];
}
int8_t StreamPeer::get_8() {

	uint8_t buf[1];
	get_data(buf, 1);
	return buf[0];
}
uint16_t StreamPeer::get_u16() {

	uint8_t buf[2];
	get_data(buf, 2);
	uint16_t r = decode_uint16(buf);
	if (big_endian) {
		r = BSWAP16(r);
	}
	return r;
}
int16_t StreamPeer::get_16() {

	uint8_t buf[2];
	get_data(buf, 2);
	uint16_t r = decode_uint16(buf);
	if (big_endian) {
		r = BSWAP16(r);
	}
	return r;
}
uint32_t StreamPeer::get_u32() {

	uint8_t buf[4];
	get_data(buf, 4);
	uint32_t r = decode_uint32(buf);
	if (big_endian) {
		r = BSWAP32(r);
	}
	return r;
}
int32_t StreamPeer::get_32() {

	uint8_t buf[4];
	get_data(buf, 4);
	uint32_t r = decode_uint32(buf);
	if (big_endian) {
		r = BSWAP32(r);
	}
	return r;
}
uint64_t StreamPeer::get_u64() {

	uint8_t buf[8];
	get_data(buf, 8);
	uint64_t r = decode_uint64(buf);
	if (big_endian) {
		r = BSWAP64(r);
	}
	return r;
}
int64_t StreamPeer::get_64() {

	uint8_t buf[8];
	get_data(buf, 8);
	uint64_t r = decode_uint64(buf);
	if (big_endian) {
		r = BSWAP64(r);
	}
	return r;
}
float StreamPeer::get_float() {

	uint8_t buf[4];
	get_data(buf, 4);

	if (big_endian) {
		uint32_t *p32 = (uint32_t *)buf;
		*p32 = BSWAP32(*p32);
	}

	return decode_float(buf);
}

float StreamPeer::get_double() {

	uint8_t buf[8];
	get_data(buf, 8);

	if (big_endian) {
		uint64_t *p64 = (uint64_t *)buf;
		*p64 = BSWAP64(*p64);
	}

	return decode_double(buf);
}
String StreamPeer::get_string(int p_bytes) {

	ERR_FAIL_COND_V(p_bytes < 0, String());

	Vector<char> buf;
	buf.resize(p_bytes + 1);
	get_data((uint8_t *)&buf[0], p_bytes);
	buf[p_bytes] = 0;
	return buf.ptr();
}
String StreamPeer::get_utf8_string(int p_bytes) {

	ERR_FAIL_COND_V(p_bytes < 0, String());

	Vector<uint8_t> buf;
	buf.resize(p_bytes);
	get_data(buf.ptr(), p_bytes);

	String ret;
	ret.parse_utf8((const char *)buf.ptr(), buf.size());
	return ret;
}
Variant StreamPeer::get_var() {

	int len = get_32();
	Vector<uint8_t> var;
	var.resize(len);
	get_data(var.ptr(), len);

	Variant ret;
	decode_variant(ret, var.ptr(), len);
	return ret;
}

void StreamPeer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("put_data", "data"), &StreamPeer::_put_data);
	ObjectTypeDB::bind_method(_MD("put_partial_data", "data"), &StreamPeer::_put_partial_data);

	ObjectTypeDB::bind_method(_MD("get_data", "bytes"), &StreamPeer::_get_data);
	ObjectTypeDB::bind_method(_MD("get_partial_data", "bytes"), &StreamPeer::_get_partial_data);

	ObjectTypeDB::bind_method(_MD("get_available_bytes"), &StreamPeer::get_available_bytes);

	ObjectTypeDB::bind_method(_MD("set_big_endian", "enable"), &StreamPeer::set_big_endian);
	ObjectTypeDB::bind_method(_MD("is_big_endian_enabled"), &StreamPeer::is_big_endian_enabled);

	ObjectTypeDB::bind_method(_MD("put_8", "val"), &StreamPeer::put_8);
	ObjectTypeDB::bind_method(_MD("put_u8", "val"), &StreamPeer::put_u8);
	ObjectTypeDB::bind_method(_MD("put_16", "val"), &StreamPeer::put_16);
	ObjectTypeDB::bind_method(_MD("put_u16", "val"), &StreamPeer::put_u16);
	ObjectTypeDB::bind_method(_MD("put_32", "val"), &StreamPeer::put_32);
	ObjectTypeDB::bind_method(_MD("put_u32", "val"), &StreamPeer::put_u32);
	ObjectTypeDB::bind_method(_MD("put_64", "val"), &StreamPeer::put_64);
	ObjectTypeDB::bind_method(_MD("put_u64", "val"), &StreamPeer::put_u64);
	ObjectTypeDB::bind_method(_MD("put_float", "val"), &StreamPeer::put_float);
	ObjectTypeDB::bind_method(_MD("put_double", "val"), &StreamPeer::put_double);
	ObjectTypeDB::bind_method(_MD("put_utf8_string", "val"), &StreamPeer::put_utf8_string);
	ObjectTypeDB::bind_method(_MD("put_var", "val:Variant"), &StreamPeer::put_var);

	ObjectTypeDB::bind_method(_MD("get_8"), &StreamPeer::get_8);
	ObjectTypeDB::bind_method(_MD("get_u8"), &StreamPeer::get_u8);
	ObjectTypeDB::bind_method(_MD("get_16"), &StreamPeer::get_16);
	ObjectTypeDB::bind_method(_MD("get_u16"), &StreamPeer::get_u16);
	ObjectTypeDB::bind_method(_MD("get_32"), &StreamPeer::get_32);
	ObjectTypeDB::bind_method(_MD("get_u32"), &StreamPeer::get_u32);
	ObjectTypeDB::bind_method(_MD("get_64"), &StreamPeer::get_64);
	ObjectTypeDB::bind_method(_MD("get_u64"), &StreamPeer::get_u64);
	ObjectTypeDB::bind_method(_MD("get_float"), &StreamPeer::get_float);
	ObjectTypeDB::bind_method(_MD("get_double"), &StreamPeer::get_double);
	ObjectTypeDB::bind_method(_MD("get_string", "bytes"), &StreamPeer::get_string);
	ObjectTypeDB::bind_method(_MD("get_utf8_string", "bytes"), &StreamPeer::get_utf8_string);
	ObjectTypeDB::bind_method(_MD("get_var:Variant"), &StreamPeer::get_var);
}

////////////////////////////////

void StreamPeerBuffer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("seek", "pos"), &StreamPeerBuffer::seek);
	ObjectTypeDB::bind_method(_MD("get_size"), &StreamPeerBuffer::get_size);
	ObjectTypeDB::bind_method(_MD("get_pos"), &StreamPeerBuffer::get_pos);
	ObjectTypeDB::bind_method(_MD("resize", "size"), &StreamPeerBuffer::resize);
	ObjectTypeDB::bind_method(_MD("set_data_array", "data"), &StreamPeerBuffer::set_data_array);
	ObjectTypeDB::bind_method(_MD("get_data_array"), &StreamPeerBuffer::get_data_array);
	ObjectTypeDB::bind_method(_MD("clear"), &StreamPeerBuffer::clear);
	ObjectTypeDB::bind_method(_MD("duplicate"), &StreamPeerBuffer::duplicate);
}

Error StreamPeerBuffer::put_data(const uint8_t *p_data, int p_bytes) {

	if (p_bytes <= 0)
		return OK;

	if (pointer + p_bytes > data.size()) {
		data.resize(pointer + p_bytes);
	}

	DVector<uint8_t>::Write w = data.write();
	copymem(&w[pointer], p_data, p_bytes);

	pointer += p_bytes;
	return OK;
}

Error StreamPeerBuffer::put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) {

	r_sent = p_bytes;
	return put_data(p_data, p_bytes);
}

Error StreamPeerBuffer::get_data(uint8_t *p_buffer, int p_bytes) {

	int recv;
	get_partial_data(p_buffer, p_bytes, recv);
	if (recv != p_bytes)
		return ERR_INVALID_PARAMETER;

	return OK;
}
Error StreamPeerBuffer::get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) {

	if (pointer + p_bytes > data.size()) {
		r_received = data.size() - pointer;
		if (r_received <= 0) {
			r_received = 0;
			return OK; //you got 0
		}
	} else {
		r_received = p_bytes;
	}

	DVector<uint8_t>::Read r = data.read();
	copymem(p_buffer, r.ptr() + pointer, r_received);

	pointer += r_received;
	// FIXME: return what? OK or ERR_*
}

int StreamPeerBuffer::get_available_bytes() const {

	return data.size() - pointer;
}

void StreamPeerBuffer::seek(int p_pos) {

	ERR_FAIL_COND(p_pos < 0);
	ERR_FAIL_COND(p_pos > data.size());
	pointer = p_pos;
}
int StreamPeerBuffer::get_size() const {

	return data.size();
}

int StreamPeerBuffer::get_pos() const {

	return pointer;
}

void StreamPeerBuffer::resize(int p_size) {

	data.resize(p_size);
}

void StreamPeerBuffer::set_data_array(const DVector<uint8_t> &p_data) {

	data = p_data;
	pointer = 0;
}

DVector<uint8_t> StreamPeerBuffer::get_data_array() const {

	return data;
}

void StreamPeerBuffer::clear() {

	data.resize(0);
	pointer = 0;
}

Ref<StreamPeerBuffer> StreamPeerBuffer::duplicate() const {

	Ref<StreamPeerBuffer> spb;
	spb.instance();
	spb->data = data;
	return spb;
}

StreamPeerBuffer::StreamPeerBuffer() {

	pointer = 0;
}
