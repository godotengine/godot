/**************************************************************************/
/*  compression.h                                                         */
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

#include "core/templates/vector.h"
#include "core/typedefs.h"

#include <zlib.h>

class Compression {
public:
	static inline int zlib_level = Z_DEFAULT_COMPRESSION;
	static inline int gzip_level = Z_DEFAULT_COMPRESSION;
	static inline int zstd_level = 3;
	static inline bool zstd_long_distance_matching = false;
	static inline int zstd_window_log_size = 27; // ZSTD_WINDOWLOG_LIMIT_DEFAULT
	static inline int gzip_chunk = 16384;

	enum Mode : int32_t {
		MODE_FASTLZ,
		MODE_DEFLATE,
		MODE_ZSTD,
		MODE_GZIP,
		MODE_BROTLI
	};

	static int64_t compress(uint8_t *p_dst, const uint8_t *p_src, int64_t p_src_size, Mode p_mode = MODE_ZSTD);
	static int64_t get_max_compressed_buffer_size(int64_t p_src_size, Mode p_mode = MODE_ZSTD);
	static int64_t decompress(uint8_t *p_dst, int64_t p_dst_max_size, const uint8_t *p_src, int64_t p_src_size, Mode p_mode = MODE_ZSTD);
	static int decompress_dynamic(Vector<uint8_t> *p_dst_vect, int64_t p_max_dst_size, const uint8_t *p_src, int64_t p_src_size, Mode p_mode);
};
