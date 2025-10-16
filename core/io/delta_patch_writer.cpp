/**************************************************************************/
/*  delta_patch_writer.cpp                                                */
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

/*
Adapted to Godot from the Android fork (originally ChromiumOS fork) of the bsdiff project.
*/

/*
Copyright 2017 The Chromium OS Authors. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
	 other materials provided with the distribution.
   * Neither the name of Google Inc. nor the names of its contributors may be used to endorse or promote products derived from this software without specific
	 prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "delta_patch_writer.h"

#include "core/io/compression.h"

#include <string.h>
#include <zstd.h>

bool DeltaPatchWriter::stream_compress(ZSTD_CStream *p_stream, Span<uint8_t> p_input, ByteBuffer &p_output) {
	ZSTD_inBuffer input_buf = {};
	input_buf.src = p_input.ptr();
	input_buf.size = p_input.size();
	input_buf.pos = 0;

	while (input_buf.pos < input_buf.size) {
		ZSTD_outBuffer output_buffer = {};
		output_buffer.dst = tmp_buffer.ptr();
		output_buffer.size = tmp_buffer.size();
		output_buffer.pos = 0;

		size_t result = ZSTD_compressStream(p_stream, &output_buffer, &input_buf);
		ERR_FAIL_COND_V(ZSTD_isError(result), false);

		uint64_t old_size = p_output.size();
		p_output.resize(old_size + output_buffer.pos);
		memcpy(p_output.ptr() + old_size, tmp_buffer.ptr(), output_buffer.pos);
	}

	return true;
}

bool DeltaPatchWriter::stream_end(ZSTD_CStream *p_stream, ByteBuffer &p_output) {
	ZSTD_outBuffer output_buffer = {};
	output_buffer.dst = tmp_buffer.ptr();
	output_buffer.size = tmp_buffer.size();
	output_buffer.pos = 0;

	size_t result = ZSTD_endStream(p_stream, &output_buffer);
	ERR_FAIL_COND_V(ZSTD_isError(result), false);

	uint64_t old_size = p_output.size();
	p_output.resize(old_size + output_buffer.pos);
	memcpy(p_output.ptr() + old_size, tmp_buffer.ptr(), output_buffer.pos);

	return true;
}

void DeltaPatchWriter::encode_int64(int64_t p_value, void *p_dst) {
#ifdef BIG_ENDIAN_ENABLED
	p_value = BSWAP64(p_value);
#endif
	memcpy(p_dst, &p_value, sizeof(int64_t));
}

bool DeltaPatchWriter::Init(size_t /* p_new_size */) {
	tmp_buffer.resize(ZSTD_CStreamOutSize());

	ctrl_stream = ZSTD_createCStream();
	ERR_FAIL_NULL_V(ctrl_stream, false);

	diff_stream = ZSTD_createCStream();
	ERR_FAIL_NULL_V(diff_stream, false);

	extra_stream = ZSTD_createCStream();
	ERR_FAIL_NULL_V(extra_stream, false);

	size_t result = ZSTD_initCStream(ctrl_stream, zstd_level);
	ERR_FAIL_COND_V(ZSTD_isError(result), false);

	result = ZSTD_initCStream(diff_stream, zstd_level);
	ERR_FAIL_COND_V(ZSTD_isError(result), false);

	result = ZSTD_initCStream(extra_stream, zstd_level);
	ERR_FAIL_COND_V(ZSTD_isError(result), false);

	return true;
}

bool DeltaPatchWriter::WriteDiffStream(const uint8_t *p_data, size_t p_size) {
	return stream_compress(diff_stream, Span(p_data, p_size), diff_buffer);
}

bool DeltaPatchWriter::WriteExtraStream(const uint8_t *p_data, size_t p_size) {
	return stream_compress(extra_stream, Span(p_data, p_size), extra_buffer);
}

bool DeltaPatchWriter::AddControlEntry(const ControlEntry &p_entry) {
	uint8_t ctrl[24];
	encode_int64(p_entry.diff_size, ctrl);
	encode_int64(p_entry.extra_size, ctrl + 8);
	encode_int64(p_entry.offset_increment, ctrl + 16);

	new_file_size += p_entry.diff_size + p_entry.extra_size;

	return stream_compress(ctrl_stream, ctrl, ctrl_buffer);
}

bool DeltaPatchWriter::Close() {
	if (!stream_end(ctrl_stream, ctrl_buffer)) {
		return false;
	}

	if (!stream_end(diff_stream, diff_buffer)) {
		return false;
	}

	if (!stream_end(extra_stream, extra_buffer)) {
		return false;
	}

	patch_file->store_8(DELTA_PATCH_VERSION_NUMBER);
	patch_file->store_64(ctrl_buffer.size());
	patch_file->store_64(diff_buffer.size());
	patch_file->store_64(new_file_size);
	patch_file->store_buffer(old_file_md5, 16);
	patch_file->store_buffer(new_file_md5, 16);

	ERR_FAIL_COND_V(patch_file->get_position() != DELTA_PATCH_HEADER_SIZE, false);

	patch_file->store_buffer(ctrl_buffer.ptr(), ctrl_buffer.size());
	patch_file->store_buffer(diff_buffer.ptr(), diff_buffer.size());
	patch_file->store_buffer(extra_buffer.ptr(), extra_buffer.size());

	return true;
}

DeltaPatchWriter::DeltaPatchWriter(Ref<FileAccess> p_patch_file, uint8_t p_old_file_md5[16], uint8_t p_new_file_md5[16], int p_zstd_level) :
		patch_file(p_patch_file),
		zstd_level(p_zstd_level) {
	memcpy(old_file_md5, p_old_file_md5, 16);
	memcpy(new_file_md5, p_new_file_md5, 16);
}

DeltaPatchWriter::~DeltaPatchWriter() {
	if (ctrl_stream != nullptr) {
		ZSTD_freeCStream(ctrl_stream);
		ctrl_stream = nullptr;
	}

	if (diff_stream != nullptr) {
		ZSTD_freeCStream(diff_stream);
		diff_stream = nullptr;
	}

	if (extra_stream != nullptr) {
		ZSTD_freeCStream(extra_stream);
		extra_stream = nullptr;
	}
}
