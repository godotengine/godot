/**************************************************************************/
/*  delta_patch_reader.h                                                  */
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

#include "core/string/ustring.h"
#include "core/templates/local_vector.h"

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wshadow")
#include <bsdiff/patch_reader_interface.h>
#include <zstd.h>
GODOT_GCC_WARNING_POP

class DeltaPatchReader final : public bsdiff::PatchReaderInterface {
public:
	typedef LocalVector<uint8_t, uint64_t> ByteBuffer;

private:
	ByteBuffer ctrl_buffer;
	ByteBuffer diff_buffer;
	ByteBuffer extra_buffer;

	uint8_t old_file_md5[16];
	uint8_t new_file_md5[16];

	ZSTD_DStream *ctrl_stream = nullptr;
	ZSTD_DStream *diff_stream = nullptr;
	ZSTD_DStream *extra_stream = nullptr;

	uint64_t new_size = 0;

	size_t ctrl_stream_pos = 0;
	size_t diff_stream_pos = 0;
	size_t extra_stream_pos = 0;

	static bool stream_decompress(ZSTD_DStream *p_stream, Span<uint8_t> p_input, size_t *p_input_pos, uint8_t *p_output, size_t p_output_size);
	static int64_t decode_int64(const uint8_t *p_src);

public:
	bool matches_old_md5(Span<uint8_t> p_data);
	bool matches_new_md5(Span<uint8_t> p_data);

	static int64_t read_new_file_size(Span<uint8_t> p_patch_data);

	virtual bool Init(const uint8_t *p_patch_data, size_t p_patch_size) override;

	virtual bool ParseControlEntry(ControlEntry *p_control_entry) override;
	virtual bool ReadDiffStream(uint8_t *p_data, size_t p_size) override;
	virtual bool ReadExtraStream(uint8_t *p_data, size_t p_size) override;

	virtual uint64_t new_file_size() const override { return new_size; }

	virtual bool Finish() override;

	virtual ~DeltaPatchReader();
};
