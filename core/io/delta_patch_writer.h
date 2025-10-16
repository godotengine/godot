/**************************************************************************/
/*  delta_patch_writer.h                                                  */
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

#include "core/io/file_access.h"

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wshadow")
#include <bsdiff/patch_writer_interface.h>
GODOT_GCC_WARNING_POP
#include <zstd.h>

#define DELTA_PATCH_VERSION_V1 1
#define DELTA_PATCH_VERSION_NUMBER DELTA_PATCH_VERSION_V1

#define DELTA_PATCH_HEADER_SIZE 57

class DeltaPatchWriter final : public bsdiff::PatchWriterInterface {
public:
	typedef LocalVector<uint8_t, uint64_t> ByteBuffer;

private:
	ByteBuffer tmp_buffer;
	ByteBuffer ctrl_buffer;
	ByteBuffer diff_buffer;
	ByteBuffer extra_buffer;

	uint8_t old_file_md5[16];
	uint8_t new_file_md5[16];

	Ref<FileAccess> patch_file;

	ZSTD_CStream *ctrl_stream = nullptr;
	ZSTD_CStream *diff_stream = nullptr;
	ZSTD_CStream *extra_stream = nullptr;

	int zstd_level = 3;
	uint64_t new_file_size = 0;

	bool stream_compress(ZSTD_CStream *p_stream, Span<uint8_t> p_input, ByteBuffer &p_output);
	bool stream_end(ZSTD_CStream *p_stream, ByteBuffer &p_output);

	static void encode_int64(int64_t p_value, void *p_dst);

public:
	virtual bool Init(size_t p_new_size) override;
	virtual bool WriteDiffStream(const uint8_t *p_data, size_t p_size) override;
	virtual bool WriteExtraStream(const uint8_t *p_data, size_t p_size) override;
	virtual bool AddControlEntry(const ControlEntry &p_entry) override;
	virtual bool Close() override;

	DeltaPatchWriter(Ref<FileAccess> p_patch_file, uint8_t p_old_file_md5[16], uint8_t p_new_file_md5[16], int p_zstd_level);

	virtual ~DeltaPatchWriter();
};
