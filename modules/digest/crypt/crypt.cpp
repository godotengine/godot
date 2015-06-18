/*************************************************************************/
/*  crypt.cpp                                                            */
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
#include "crypt.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "sha1.inl"
#include "des.inl"
#include "hmac.inl"

static void 
Hash(const char * str, int sz, uint8_t key[8]) {
	uint32_t djb_hash = 5381L;
	uint32_t js_hash = 1315423911L;

	int i;
	for (i=0;i<sz;i++) {
		uint8_t c = (uint8_t)str[i];
		djb_hash += (djb_hash << 5) + c;
		js_hash ^= ((js_hash << 5) + c + (js_hash >> 2));
	}

	key[0] = djb_hash & 0xff;
	key[1] = (djb_hash >> 8) & 0xff;
	key[2] = (djb_hash >> 16) & 0xff;
	key[3] = (djb_hash >> 24) & 0xff;

	key[4] = js_hash & 0xff;
	key[5] = (js_hash >> 8) & 0xff;
	key[6] = (js_hash >> 16) & 0xff;
	key[7] = (js_hash >> 24) & 0xff;
}

ByteArray Crypt::hashkey(const ByteArray& p_key) {

	ByteArray realkey;
	realkey.resize(8);

	ByteArray::Read r = p_key.read();
	ByteArray::Write w = realkey.write();
	Hash((const char *) r.ptr(), p_key.size(), w.ptr());

	return realkey;
}

ByteArray Crypt::randomkey() {

	ByteArray tmp;
	tmp.resize(8);
	ByteArray::Write w = tmp.write();
	for(int i = 0; i < 8; i++)
		w[i] = rand() % 0xff;

	return tmp;
}

static bool
des_key(const ByteArray& p_key, uint32_t SK[32]) {

	ERR_EXPLAIN("Invalid key size(not equal 8bytes)");
	ERR_FAIL_COND_V(p_key.size() != 8, false);
	ByteArray::Read r = p_key.read();

	des_main_ks(SK, r.ptr());
	return true;
}


ByteArray Crypt::desencode(const ByteArray& p_key, const ByteArray& p_text) {

	uint32_t SK[32];
	ERR_FAIL_COND_V(!des_key(p_key, SK), ByteArray());

	size_t textsz = p_text.size();
	ByteArray::Read r = p_text.read();
	size_t chunksz = (textsz + 8) & ~7;

	ByteArray buffer;
	buffer.resize(chunksz);
	ByteArray::Write w = buffer.write();

	int i;
	for(i = 0; i < textsz - 7; i += 8)
		des_crypt(SK, r.ptr() + i, w.ptr() + i);

	int bytes = textsz - i;

	uint8_t tail[8];
	int j;
	for (j=0;j<8;j++) {
		if (j < bytes) {
			tail[j] = r[i + j];
		} else if (j==bytes) {
			tail[j] = 0x80;
		} else {
			tail[j] = 0;
		}
	}
	des_crypt(SK, tail, w.ptr() + i);
	return buffer;
}

ByteArray Crypt::desdecode(const ByteArray& p_key, const ByteArray& p_text) {

	uint32_t ESK[32];
	ERR_FAIL_COND_V(!des_key(p_key, ESK), ByteArray());
	uint32_t SK[32];
	for(int i = 0; i < 32; i += 2) {
		SK[i] = ESK[30 - i];
		SK[i + 1] = ESK[31 - i];
	}
	size_t textsz = p_text.size();
	ByteArray::Read r = p_text.read();
	ERR_EXPLAIN("Invalid des crypt text length " + String::num(textsz));
	ERR_FAIL_COND_V(((textsz & 7) || textsz == 0), ByteArray());

	ByteArray buffer;
	buffer.resize(textsz);
	ByteArray::Write w = buffer.write();

	int i;
	for (i=0;i<textsz;i+=8)
		des_crypt(SK, r.ptr() + i, w.ptr() + i);

	int padding = 1;
	for (i=textsz-1;i>=textsz-8;i--) {
		if (buffer[i] == 0) {
			padding++;
		} else if (buffer[i] == 0x80) {
			break;
		} else {
			ERR_EXPLAIN("Invalid des crypt text");
			ERR_FAIL_V(ByteArray());
		}
	}
	ERR_EXPLAIN("Invalid des crypt text");
	ERR_FAIL_COND_V(padding > 8, ByteArray());
	w = ByteArray::Write();
	buffer.resize(textsz - padding);
	return buffer;
}

ByteArray Crypt::hexencode(const ByteArray& p_raw) {

	static char hex[] = "0123456789abcdef";
	size_t sz = p_raw.size();
	ByteArray buffer;
	buffer.resize(sz * 2);
	ByteArray::Read r = p_raw.read();
	ByteArray::Write w = buffer.write();

	for(int i = 0; i < sz; i++) {
		w[i * 2] = hex[r[i] >> 4];
		w[i * 2 + 1] = hex[r[i] & 0xf];
	}
	return buffer;
}

#define HEX(v,c) { char tmp = (char) c; if (tmp >= '0' && tmp <= '9') { v = tmp-'0'; } else { v = tmp - 'a' + 10; } }

ByteArray Crypt::hexdecode(const ByteArray& p_hex) {

	size_t sz = p_hex.size();
	ERR_EXPLAIN("Invalid hex text size " + String::num(sz));
	ERR_FAIL_COND_V(sz & 2, ByteArray());

	ByteArray buffer;
	buffer.resize(sz / 2);
	ByteArray::Write w = buffer.write();
	ByteArray::Read r = p_hex.read();

	int i;
	for(i = 0; i < sz; i += 2) {
		uint8_t hi, low;
		HEX(hi, r[i]);
		HEX(low, r[i + 1]);
		if(hi > 16 || low > 16) {
			ERR_EXPLAIN("Invalid hex text");
			ERR_FAIL_V(ByteArray());
		}
		w[i / 2] = hi << 4 | low;
	}
	return buffer;
}

static bool
read64(const ByteArray& p_x, const ByteArray& p_y, uint32_t xx[2], uint32_t yy[2]) {

	ERR_EXPLAIN("Invalid uint64 x");
	ERR_FAIL_COND_V(p_x.size() != 8, false);
	ERR_EXPLAIN("Invalid uint64 y");
	ERR_FAIL_COND_V(p_y.size() != 8, false);

	ByteArray::Read x = p_x.read();
	ByteArray::Read y = p_y.read();

	xx[0] = x[0] | x[1]<<8 | x[2]<<16 | x[3]<<24;
	xx[1] = x[4] | x[5]<<8 | x[6]<<16 | x[7]<<24;
	yy[0] = y[0] | y[1]<<8 | y[2]<<16 | y[3]<<24;
	yy[1] = y[4] | y[5]<<8 | y[6]<<16 | y[7]<<24;

	return true;
}

static ByteArray
pushqword(uint32_t result[2]) {

	ByteArray buffer;
	buffer.resize(8);
	ByteArray::Write tmp = buffer.write();

	tmp[0] = result[0] & 0xff;
	tmp[1] = (result[0] >> 8 )& 0xff;
	tmp[2] = (result[0] >> 16 )& 0xff;
	tmp[3] = (result[0] >> 24 )& 0xff;
	tmp[4] = result[1] & 0xff;
	tmp[5] = (result[1] >> 8 )& 0xff;
	tmp[6] = (result[1] >> 16 )& 0xff;
	tmp[7] = (result[1] >> 24 )& 0xff;

	return buffer;
}

ByteArray Crypt::hmac64(const ByteArray& p_x, const ByteArray& p_y) {

	uint32_t x[2], y[2];
	ERR_FAIL_COND_V(!read64(p_x, p_y, x, y), ByteArray());
	uint32_t result[2];
	hmac(x,y,result);
	return pushqword(result);
}

// powmodp64 for DH-key exchange

// The biggest 64bit prime
#define P 0xffffffffffffffc5ull

static inline uint64_t
mul_mod_p(uint64_t a, uint64_t b) {
	uint64_t m = 0;
	while(b) {
		if(b&1) {
			uint64_t t = P-a;
			if ( m >= t) {
				m -= t;
			} else {
				m += a;
			}
		}
		if (a >= P - a) {
			a = a * 2 - P;
		} else {
			a = a * 2;
		}
		b>>=1;
	}
	return m;
}

static inline uint64_t
pow_mod_p(uint64_t a, uint64_t b) {
	if (b==1) {
		return a;
	}
	uint64_t t = pow_mod_p(a, b>>1);
	t = mul_mod_p(t,t);
	if (b % 2) {
		t = mul_mod_p(t, a);
	}
	return t;
}

// calc a^b % p
static uint64_t
powmodp(uint64_t a, uint64_t b) {
	if (a > P)
		a%=P;
	return pow_mod_p(a,b);
}

static ByteArray
push64(uint64_t r) {

	ByteArray buffer;
	buffer.resize(8);
	ByteArray::Write tmp = buffer.write();

	tmp[0] = r & 0xff;
	tmp[1] = (r >> 8 )& 0xff;
	tmp[2] = (r >> 16 )& 0xff;
	tmp[3] = (r >> 24 )& 0xff;
	tmp[4] = (r >> 32 )& 0xff;
	tmp[5] = (r >> 40 )& 0xff;
	tmp[6] = (r >> 48 )& 0xff;
	tmp[7] = (r >> 56 )& 0xff;

	return buffer;
}

#define G 5

ByteArray Crypt::dhexchange(const ByteArray& p_raw) {

	size_t sz = p_raw.size();
	ERR_EXPLAIN("Invalid dh uint64 key");
	ERR_FAIL_COND_V(sz != 8, ByteArray());
	ByteArray::Read x = p_raw.read();

	uint32_t xx[2];
	xx[0] = x[0] | x[1]<<8 | x[2]<<16 | x[3]<<24;
	xx[1] = x[4] | x[5]<<8 | x[6]<<16 | x[7]<<24;

	uint64_t r = powmodp(5,	(uint64_t)xx[0] | (uint64_t)xx[1]<<32);
	return push64(r);
}

ByteArray Crypt::dhsecret(const ByteArray& p_x, const ByteArray& p_y) {

	uint32_t x[2], y[2];
	ERR_FAIL_COND_V(!read64(p_x, p_y, x, y), ByteArray());

	uint64_t r = powmodp((uint64_t)x[0] | (uint64_t)x[1]<<32,
		(uint64_t)y[0] | (uint64_t)y[1]<<32);

	return push64(r);
}

ByteArray Crypt::base64encode(const ByteArray& p_raw) {

	static const char* encoding = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	size_t sz = p_raw.size();
	ByteArray::Read text = p_raw.read();
	int encode_sz = (sz + 2)/3*4;
	ByteArray tmp;
	tmp.resize(encode_sz);
	ByteArray::Write buffer = tmp.write();
	int i,j;
	j=0;
	for (i=0;i<(int)sz-2;i+=3) {
		uint32_t v = text[i] << 16 | text[i+1] << 8 | text[i+2];
		buffer[j] = encoding[v >> 18];
		buffer[j+1] = encoding[(v >> 12) & 0x3f];
		buffer[j+2] = encoding[(v >> 6) & 0x3f];
		buffer[j+3] = encoding[(v) & 0x3f];
		j+=4;
	}
	int padding = sz-i;
	uint32_t v;
	switch(padding) {
	case 1 :
		v = text[i];
		buffer[j] = encoding[v >> 2];
		buffer[j+1] = encoding[(v & 3) << 4];
		buffer[j+2] = '=';
		buffer[j+3] = '=';
		break;
	case 2 :
		v = text[i] << 8 | text[i+1];
		buffer[j] = encoding[v >> 10];
		buffer[j+1] = encoding[(v >> 4) & 0x3f];
		buffer[j+2] = encoding[(v & 0xf) << 2];
		buffer[j+3] = '=';
		break;
	}
	return tmp;
}

static inline int
b64index(uint8_t c) {
	static const int decoding[] = {62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,-1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51};
	int decoding_size = sizeof(decoding)/sizeof(decoding[0]);
	if (c<43) {
		return -1;
	}
	c -= 43;
	if (c>=decoding_size)
		return -1;
	return decoding[c];
}

ByteArray Crypt::base64decode(const ByteArray& p_base64) {

	size_t sz = p_base64.size();
	ByteArray::Read text = p_base64.read();
	int decode_sz = (sz+3)/4*3;
	ByteArray tmp;
	tmp.resize(decode_sz);
	ByteArray::Write buffer = tmp.write();
	int i,j;
	int output = 0;
	for (i=0;i<sz;) {
		int padding = 0;
		int c[4];
		for (j=0;j<4;) {
			if (i>=sz) {
				ERR_EXPLAIN("Invalid base64 text");
				ERR_FAIL_V(ByteArray());
			}
			c[j] = b64index(text[i]);
			if (c[j] == -1) {
				++i;
				continue;
			}
			if (c[j] == -2) {
				++padding;
			}
			++i;
			++j;
		}
		uint32_t v;
		switch (padding) {
		case 0:
			v = (unsigned)c[0] << 18 | c[1] << 12 | c[2] << 6 | c[3];
			buffer[output] = v >> 16;
			buffer[output+1] = (v >> 8) & 0xff;
			buffer[output+2] = v & 0xff;
			output += 3;
			break;
		case 1:
			if (c[3] != -2 || (c[2] & 3)!=0) {
				ERR_EXPLAIN("Invalid base64 text");
				ERR_FAIL_V(ByteArray());
			}
			v = (unsigned)c[0] << 10 | c[1] << 4 | c[2] >> 2 ;
			buffer[output] = v >> 8;
			buffer[output+1] = v & 0xff;
			output += 2;
			break;
		case 2:
			if (c[3] != -2 || c[2] != -2 || (c[1] & 0xf) !=0)  {
				ERR_EXPLAIN("Invalid base64 text");
				ERR_FAIL_V(ByteArray());
			}
			v = (unsigned)c[0] << 2 | c[1] >> 4;
			buffer[output] = v;
			++ output;
			break;
		default: {
			ERR_EXPLAIN("Invalid base64 text");
			ERR_FAIL_V(ByteArray());
		}
		}
	}
	buffer = ByteArray::Write();
	tmp.resize(output);
	return tmp;
}

ByteArray Crypt::sha1(const ByteArray& p_raw) {

	size_t sz = p_raw.size();
	ByteArray::Read buffer = p_raw.read();
	ByteArray tmp;
	tmp.resize(SHA1_DIGEST_SIZE);
	ByteArray::Write digest = tmp.write();
	SHA1_CTX ctx;
	sat_SHA1_Init(&ctx);
	sat_SHA1_Update(&ctx, buffer.ptr(), sz);
	sat_SHA1_Final(&ctx, digest.ptr());
	return tmp;
}

ByteArray Crypt::hmac_sha1(const ByteArray& p_key, const ByteArray& p_text) {

	size_t key_sz = p_key.size();
	ByteArray::Read key = p_key.read();
	size_t text_sz = p_text.size();
	ByteArray::Read text = p_text.read();
	SHA1_CTX ctx1, ctx2;
	ByteArray tmp;
	tmp.resize(SHA1_DIGEST_SIZE);
	ByteArray::Write digest1 = tmp.write();
	//uint8_t digest1[SHA1_DIGEST_SIZE];
	uint8_t digest2[SHA1_DIGEST_SIZE];
	uint8_t rkey[BLOCKSIZE];
	memset(rkey, 0, BLOCKSIZE);

	if (key_sz > BLOCKSIZE) {
		SHA1_CTX ctx;
		sat_SHA1_Init(&ctx);
		sat_SHA1_Update(&ctx, key.ptr(), key_sz);
		sat_SHA1_Final(&ctx, rkey);
		key_sz = SHA1_DIGEST_SIZE;
	} else {
		memcpy(rkey, key.ptr(), key_sz);
	}

	xor_key(rkey, 0x5c5c5c5c);
	sat_SHA1_Init(&ctx1);
	sat_SHA1_Update(&ctx1, rkey, BLOCKSIZE);

	xor_key(rkey, 0x5c5c5c5c ^ 0x36363636);
	sat_SHA1_Init(&ctx2);
	sat_SHA1_Update(&ctx2, rkey, BLOCKSIZE);
	sat_SHA1_Update(&ctx2, text.ptr(), text_sz);
	sat_SHA1_Final(&ctx2, digest2);

	sat_SHA1_Update(&ctx1, digest2, SHA1_DIGEST_SIZE);
	sat_SHA1_Final(&ctx1, digest1.ptr());

	return tmp;
}

ByteArray Crypt::hmac_hash(const ByteArray& p_key, const ByteArray& p_text) {

	uint32_t key[2];
	size_t sz = p_key.size();
	ByteArray::Read x = p_key.read();
	ERR_EXPLAIN("Invalid uint64 key");
	ERR_FAIL_COND_V(sz != 8, ByteArray());
	key[0] = x[0] | x[1]<<8 | x[2]<<16 | x[3]<<24;
	key[1] = x[4] | x[5]<<8 | x[6]<<16 | x[7]<<24;
	ByteArray::Read text = p_text.read();
	uint8_t h[8];
	Hash((const char *) text.ptr(),(int)sz,h);
	uint32_t htext[2];
	htext[0] = h[0] | h[1]<<8 | h[2]<<16 | h[3]<<24;
	htext[1] = h[4] | h[5]<<8 | h[6]<<16 | h[7]<<24;
	uint32_t result[2];
	hmac(htext,key,result);
	return pushqword(result);
}

void Crypt::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("hashkey","key"),&Crypt::hashkey);
	ObjectTypeDB::bind_method(_MD("randomkey"),&Crypt::randomkey);
	ObjectTypeDB::bind_method(_MD("desencode","key","text"),&Crypt::desencode);
	ObjectTypeDB::bind_method(_MD("desdecode","key","text"),&Crypt::desdecode);
	ObjectTypeDB::bind_method(_MD("hexencode","raw"),&Crypt::hexencode);
	ObjectTypeDB::bind_method(_MD("hexdecode","hex"),&Crypt::hexdecode);
	ObjectTypeDB::bind_method(_MD("hmac64","x","y"),&Crypt::hmac64);
	ObjectTypeDB::bind_method(_MD("dhexchange","raw"),&Crypt::dhexchange);
	ObjectTypeDB::bind_method(_MD("dhsecret","x","y"),&Crypt::dhsecret);
	ObjectTypeDB::bind_method(_MD("base64encode","raw"),&Crypt::base64encode);
	ObjectTypeDB::bind_method(_MD("base64decode","base64"),&Crypt::base64decode);
	ObjectTypeDB::bind_method(_MD("sha1","raw"),&Crypt::sha1);
	ObjectTypeDB::bind_method(_MD("hmac_sha1","key","raw"),&Crypt::hmac_sha1);
	ObjectTypeDB::bind_method(_MD("hmac_hash","raw"),&Crypt::hmac_hash);
}
