/**************************************************************************/
/*  delta_patch_reader.cpp                                                */
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

#include "delta_patch_reader.h"

#include "core/crypto/crypto_core.h"
#include "core/io/delta_patch_writer.h"
#include "core/io/file_access_memory.h"

#include <string.h>
#include <zstd.h>

bool DeltaPatchReader::stream_decompress(ZSTD_DStream *p_stream, Span<uint8_t> p_input, size_t *p_input_pos, uint8_t *p_output, size_t p_output_size) {
	ZSTD_inBuffer input_buf = {};
	input_buf.src = p_input.ptr();
	input_buf.size = p_input.size();
	input_buf.pos = *p_input_pos;

	ZSTD_outBuffer output_buf = {};
	output_buf.dst = p_output;
	output_buf.size = p_output_size;
	output_buf.pos = 0;

	while (output_buf.pos < output_buf.size && input_buf.pos < input_buf.size) {
		size_t result = ZSTD_decompressStream(p_stream, &output_buf, &input_buf);
		ERR_FAIL_COND_V(ZSTD_isError(result), false);

		if (result == 0) {
			break;
		}
	}

	*p_input_pos = input_buf.pos;
	return output_buf.pos == output_buf.size;
}

int64_t DeltaPatchReader::decode_int64(const uint8_t *p_src) {
	int64_t value;
	memcpy(&value, p_src, sizeof(int64_t));
#ifdef BIG_ENDIAN_ENABLED
	value = BSWAP64(value);
#endif
	return value;
}

bool DeltaPatchReader::matches_old_md5(Span<uint8_t> p_data) {
	uint8_t data_md5[16];
	Error err = CryptoCore::md5(p_data.ptr(), p_data.size(), data_md5);
	ERR_FAIL_COND_V(err != OK, false);
	return memcmp(data_md5, old_file_md5, 16) == 0;
}

bool DeltaPatchReader::matches_new_md5(Span<uint8_t> p_data) {
	uint8_t data_md5[16];
	Error err = CryptoCore::md5(p_data.ptr(), p_data.size(), data_md5);
	ERR_FAIL_COND_V(err != OK, false);
	return memcmp(data_md5, new_file_md5, 16) == 0;
}

int64_t DeltaPatchReader::read_new_file_size(Span<uint8_t> p_patch_data) {
	int64_t signed_new_file_size = decode_int64(p_patch_data.ptr() + 17);
	ERR_FAIL_COND_V_MSG(signed_new_file_size < 0, 0, "Patch is corrupt.");
	return signed_new_file_size;
}

bool DeltaPatchReader::Init(const uint8_t *p_patch_data, size_t p_patch_size) {
	ERR_FAIL_COND_V(ctrl_stream != nullptr, false);
	ERR_FAIL_COND_V(diff_stream != nullptr, false);
	ERR_FAIL_COND_V(extra_stream != nullptr, false);

	ctrl_buffer.clear();
	diff_buffer.clear();
	extra_buffer.clear();

	memset(old_file_md5, 0, 16);
	memset(new_file_md5, 0, 16);

	new_size = 0;

	ctrl_stream_pos = 0;
	diff_stream_pos = 0;
	extra_stream_pos = 0;

	ERR_FAIL_COND_V_MSG(p_patch_size < DELTA_PATCH_HEADER_SIZE, false, "Patch is corrupt.");

	Ref<FileAccessMemory> patch_file;
	patch_file.instantiate();
	patch_file->open_custom(p_patch_data, p_patch_size);

	uint8_t version_number = patch_file->get_8();
	ERR_FAIL_COND_V_MSG(version_number != DELTA_PATCH_VERSION_NUMBER, false, vformat("Unexpected version number '%d', expected '%d'.", version_number, DELTA_PATCH_VERSION_NUMBER));

	int64_t ctrl_buffer_size = patch_file->get_64();
	ERR_FAIL_COND_V_MSG(ctrl_buffer_size < 0, false, "Patch is corrupt.");
	ERR_FAIL_COND_V_MSG(static_cast<int64_t>(p_patch_size) - DELTA_PATCH_HEADER_SIZE < ctrl_buffer_size, false, "Patch is corrupt.");

	int64_t diff_buffer_size = patch_file->get_64();
	ERR_FAIL_COND_V_MSG(diff_buffer_size < 0, false, "Patch is corrupt.");
	ERR_FAIL_COND_V_MSG(static_cast<int64_t>(p_patch_size) - DELTA_PATCH_HEADER_SIZE - ctrl_buffer_size < diff_buffer_size, false, "Patch is corrupt.");

	int64_t signed_new_file_size = patch_file->get_64();
	ERR_FAIL_COND_V_MSG(signed_new_file_size < 0, false, "Patch is corrupt.");
	new_size = signed_new_file_size;

	patch_file->get_buffer(old_file_md5, 16);
	patch_file->get_buffer(new_file_md5, 16);

	ctrl_buffer.resize(ctrl_buffer_size);
	patch_file->get_buffer(ctrl_buffer.ptr(), ctrl_buffer.size());

	diff_buffer.resize(diff_buffer_size);
	patch_file->get_buffer(diff_buffer.ptr(), diff_buffer.size());

	uint64_t extra_buffer_size = patch_file->get_length() - patch_file->get_position();
	extra_buffer.resize(extra_buffer_size);
	patch_file->get_buffer(extra_buffer.ptr(), extra_buffer.size());

	ctrl_stream = ZSTD_createDStream();
	ERR_FAIL_NULL_V(ctrl_stream, false);

	diff_stream = ZSTD_createDStream();
	ERR_FAIL_NULL_V(diff_stream, false);

	extra_stream = ZSTD_createDStream();
	ERR_FAIL_NULL_V(extra_stream, false);

	size_t result = ZSTD_initDStream(ctrl_stream);
	ERR_FAIL_COND_V(ZSTD_isError(result), false);

	result = ZSTD_initDStream(diff_stream);
	ERR_FAIL_COND_V(ZSTD_isError(result), false);

	result = ZSTD_initDStream(extra_stream);
	ERR_FAIL_COND_V(ZSTD_isError(result), false);

	return true;
}

bool DeltaPatchReader::ParseControlEntry(ControlEntry *p_control_entry) {
	uint8_t ctrl[24];
	bool success = stream_decompress(ctrl_stream, ctrl_buffer, &ctrl_stream_pos, ctrl, sizeof(ctrl));
	ERR_FAIL_COND_V(!success, false);

	p_control_entry->diff_size = decode_int64(ctrl);
	p_control_entry->extra_size = decode_int64(ctrl + 8);
	p_control_entry->offset_increment = decode_int64(ctrl + 16);

	return true;
}

bool DeltaPatchReader::ReadDiffStream(uint8_t *p_data, size_t p_size) {
	return stream_decompress(diff_stream, diff_buffer, &diff_stream_pos, p_data, p_size);
}

bool DeltaPatchReader::ReadExtraStream(uint8_t *p_data, size_t p_size) {
	return stream_decompress(extra_stream, extra_buffer, &extra_stream_pos, p_data, p_size);
}

bool DeltaPatchReader::Finish() {
	if (ctrl_stream != nullptr) {
		ZSTD_freeDStream(ctrl_stream);
		ctrl_stream = nullptr;
	}

	if (diff_stream != nullptr) {
		ZSTD_freeDStream(diff_stream);
		diff_stream = nullptr;
	}

	if (extra_stream != nullptr) {
		ZSTD_freeDStream(extra_stream);
		extra_stream = nullptr;
	}

	return true;
}

DeltaPatchReader::~DeltaPatchReader() {
	Finish();
}
