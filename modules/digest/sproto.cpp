/*************************************************************************/
/*  sproto.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "sproto.h"
extern "C" {
#include "sproto/sproto.h"
};

void Sproto::dump() {

	ERR_FAIL_COND(proto == NULL);
	sproto_dump(proto);
}

Dictionary Sproto::protocol() {

	return Dictionary();
}

void Sproto::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("dump"),&Sproto::dump);
	ObjectTypeDB::bind_method(_MD("encode","type","variant"),&Sproto::encode);
	ObjectTypeDB::bind_method(_MD("decode","stream"),&Sproto::decode);
	ObjectTypeDB::bind_method(_MD("protocol",""),&Sproto::protocol);
}

Sproto::Sproto()
	: proto(NULL)
{
}

Sproto::~Sproto() {

	if(proto != NULL)
		sproto_release(proto);
}

//ByteArray Snappy::compress(const ByteArray& p_input) {
//
//	ByteArray::Read r = p_input.read();
//	uint32_t max_compressed_length = csnappy_max_compressed_length(p_input.size());
//
//	ByteArray output;
//	output.resize(max_compressed_length);
//	ByteArray::Write w = output.write();
//
//	static char working_memory[CSNAPPY_WORKMEM_BYTES];
//
//	uint32_t compressed_length;
//	csnappy_compress((const char *) r.ptr(), p_input.size(), (char *) w.ptr(), &compressed_length, working_memory, CSNAPPY_WORKMEM_BYTES_POWER_OF_TWO);
//	if(compressed_length > 0) {
//
//		w = ByteArray::Write();
//		output.resize(compressed_length);
//		return output;
//	}
//
//	return ByteArray();
//}
//
//ByteArray Snappy::uncompress(const ByteArray& p_input) {
//
//	ByteArray::Read r = p_input.read();
//	uint32_t uncompressed_length;
//	if(csnappy_get_uncompressed_length((const char *) r.ptr(), p_input.size(), &uncompressed_length) != CSNAPPY_E_HEADER_BAD) {
//		
//		ByteArray output;
//		output.resize(uncompressed_length);
//		ByteArray::Write w = output.write();
//
//		if(csnappy_decompress((const char *) r.ptr(), p_input.size(), (char *) w.ptr(), uncompressed_length) == CSNAPPY_E_OK)
//			return output;
//	}
//
//	return ByteArray();
//}
//
//void Snappy::_bind_methods() {
//
//	ObjectTypeDB::bind_method(_MD("compress"),&Snappy::compress);
//	ObjectTypeDB::bind_method(_MD("uncompress"),&Snappy::uncompress);
//}
