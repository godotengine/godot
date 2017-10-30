/*************************************************************************/
/*  compression.cpp                                                      */
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
#include "compression.h"
#include "os/copymem.h"
#include "project_settings.h"
#include "zip_io.h"

#include "thirdparty/misc/fastlz.h"

#include <zlib.h>
#include <zstd.h>

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
		case MODE_DEFLATE:
		case MODE_GZIP: {

			int window_bits = p_mode == MODE_DEFLATE ? 15 : 15 + 16;

			z_stream strm;
			strm.zalloc = zipio_alloc;
			strm.zfree = zipio_free;
			strm.opaque = Z_NULL;
			int level = p_mode == MODE_DEFLATE ? zlib_level : gzip_level;
			int err = deflateInit2(&strm, level, Z_DEFLATED, window_bits, 8, Z_DEFAULT_STRATEGY);
			if (err != Z_OK)
				return -1;

			strm.avail_in = p_src_size;
			int aout = deflateBound(&strm, p_src_size);
			strm.avail_out = aout;
			strm.next_in = (Bytef *)p_src;
			strm.next_out = p_dst;
			deflate(&strm, Z_FINISH);
			aout = aout - strm.avail_out;
			deflateEnd(&strm);
			return aout;

		} break;
		case MODE_ZSTD: {
			ZSTD_CCtx *cctx = ZSTD_createCCtx();
			ZSTD_CCtx_setParameter(cctx, ZSTD_p_compressionLevel, zstd_level);
			if (zstd_long_distance_matching) {
				ZSTD_CCtx_setParameter(cctx, ZSTD_p_enableLongDistanceMatching, 1);
				ZSTD_CCtx_setParameter(cctx, ZSTD_p_windowLog, zstd_window_log_size);
			}
			int max_dst_size = get_max_compressed_buffer_size(p_src_size, MODE_ZSTD);
			int ret = ZSTD_compressCCtx(cctx, p_dst, max_dst_size, p_src, p_src_size, zstd_level);
			ZSTD_freeCCtx(cctx);
			return ret;
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
		case MODE_DEFLATE:
		case MODE_GZIP: {

			int window_bits = p_mode == MODE_DEFLATE ? 15 : 15 + 16;

			z_stream strm;
			strm.zalloc = zipio_alloc;
			strm.zfree = zipio_free;
			strm.opaque = Z_NULL;
			int err = deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, window_bits, 8, Z_DEFAULT_STRATEGY);
			if (err != Z_OK)
				return -1;
			int aout = deflateBound(&strm, p_src_size);
			deflateEnd(&strm);
			return aout;
		} break;
		case MODE_ZSTD: {

			return ZSTD_compressBound(p_src_size);
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
		case MODE_DEFLATE:
		case MODE_GZIP: {

			int window_bits = p_mode == MODE_DEFLATE ? 15 : 15 + 16;

			z_stream strm;
			strm.zalloc = zipio_alloc;
			strm.zfree = zipio_free;
			strm.opaque = Z_NULL;
			strm.avail_in = 0;
			strm.next_in = Z_NULL;
			int err = inflateInit2(&strm, window_bits);
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
		case MODE_ZSTD: {
			ZSTD_DCtx *dctx = ZSTD_createDCtx();
			if (zstd_long_distance_matching) ZSTD_DCtx_setMaxWindowSize(dctx, 1 << zstd_window_log_size);
			int ret = ZSTD_decompressDCtx(dctx, p_dst, p_dst_max_size, p_src, p_src_size);
			ZSTD_freeDCtx(dctx);
			return ret;
		} break;
	}

	ERR_FAIL_V(-1);
}

int Compression::zlib_level = Z_DEFAULT_COMPRESSION;
int Compression::gzip_level = Z_DEFAULT_COMPRESSION;
int Compression::zstd_level = 3;
bool Compression::zstd_long_distance_matching = false;
int Compression::zstd_window_log_size = 27;
