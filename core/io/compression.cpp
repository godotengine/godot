/*************************************************************************/
/*  compression.cpp                                                      */
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
#include "compression.h"
#include "os/copymem.h"
#include "zlib.h"

#include "fastlz.h"
#include "zip_io.h"

int Compression::compress(uint8_t *p_dst, const uint8_t *p_src, int p_src_size, Mode p_mode) {

	switch (p_mode) {
		case MODE_FASTLZ: {

			if (p_src_size < 16) {
				uint8_t src[16];
				zeromem(&src[p_src_size], 16 - p_src_size);
				copymem(src, p_src, p_src_size);
				return fastlz_compress(src, 16, p_dst);
			} else {
				return fastlz_compress(p_src, p_src_size, p_dst);
			}

		} break;
		case MODE_DEFLATE: {

			z_stream strm;
			strm.zalloc = zipio_alloc;
			strm.zfree = zipio_free;
			strm.opaque = Z_NULL;
			int err = deflateInit(&strm, Z_DEFAULT_COMPRESSION);
			if (err != Z_OK)
				return -1;

			strm.avail_in = p_src_size;
			int aout = deflateBound(&strm, p_src_size);
			/*if (aout>p_src_size) {
				deflateEnd(&strm);
				return -1;
			}*/
			strm.avail_out = aout;
			strm.next_in = (Bytef *)p_src;
			strm.next_out = p_dst;
			deflate(&strm, Z_FINISH);
			aout = aout - strm.avail_out;
			deflateEnd(&strm);
			return aout;

		} break;
	}

	ERR_FAIL_V(-1);
}

int Compression::get_max_compressed_buffer_size(int p_src_size, Mode p_mode) {

	switch (p_mode) {
		case MODE_FASTLZ: {

			int ss = p_src_size + p_src_size * 6 / 100;
			if (ss < 66)
				ss = 66;
			return ss;

		} break;
		case MODE_DEFLATE: {

			z_stream strm;
			strm.zalloc = zipio_alloc;
			strm.zfree = zipio_free;
			strm.opaque = Z_NULL;
			int err = deflateInit(&strm, Z_DEFAULT_COMPRESSION);
			if (err != Z_OK)
				return -1;
			int aout = deflateBound(&strm, p_src_size);
			deflateEnd(&strm);
			return aout;
		} break;
	}

	ERR_FAIL_V(-1);
}

int Compression::decompress(uint8_t *p_dst, int p_dst_max_size, const uint8_t *p_src, int p_src_size, Mode p_mode) {

	switch (p_mode) {
		case MODE_FASTLZ: {

			int ret_size = 0;

			if (p_dst_max_size < 16) {
				uint8_t dst[16];
				ret_size = fastlz_decompress(p_src, p_src_size, dst, 16);
				copymem(p_dst, dst, p_dst_max_size);
			} else {
				ret_size = fastlz_decompress(p_src, p_src_size, p_dst, p_dst_max_size);
			}
			return ret_size;
		} break;
		case MODE_DEFLATE: {

			z_stream strm;
			strm.zalloc = zipio_alloc;
			strm.zfree = zipio_free;
			strm.opaque = Z_NULL;
			strm.avail_in = 0;
			strm.next_in = Z_NULL;
			int err = inflateInit(&strm);
			ERR_FAIL_COND_V(err != Z_OK, -1);

			strm.avail_in = p_src_size;
			strm.avail_out = p_dst_max_size;
			strm.next_in = (Bytef *)p_src;
			strm.next_out = p_dst;

			err = inflate(&strm, Z_FINISH);
			int total = strm.total_out;
			inflateEnd(&strm);
			ERR_FAIL_COND_V(err != Z_STREAM_END, -1);
			return total;
		} break;
	}

	ERR_FAIL_V(-1);
}
