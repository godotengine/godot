/**************************************************************************/
/*  pck_lzma.cpp                                                          */
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

#include "pck_lzma.h"

#include "core/templates/local_vector.h"

#include <cstring>

extern "C" {
#include <thirdparty/lzma2601/C/7zAlloc.h>
#include <thirdparty/lzma2601/C/Lzma2Dec.h>
#include <thirdparty/lzma2601/C/Lzma2Enc.h>
}

namespace {

ISzAlloc g_alloc = { SzAlloc, SzFree };
ISzAlloc g_alloc_temp = { SzAllocTemp, SzFreeTemp };

struct LzmaInStream {
	ISeqInStream vt;
	const uint8_t *data = nullptr;
	size_t size = 0;
	size_t pos = 0;
};

static SRes _lzma_in_stream_read(ISeqInStreamPtr pp, void *p_buf, size_t *p_size) {
	LzmaInStream *stream = Z7_CONTAINER_FROM_VTBL(pp, LzmaInStream, vt); // NOLINT(modernize-use-bool-literals)
	const size_t requested = *p_size;
	const size_t left = (stream->pos < stream->size) ? stream->size - stream->pos : 0;
	const size_t read_size = MIN(requested, left);
	if (read_size > 0) {
		memcpy(p_buf, stream->data + stream->pos, read_size);
		stream->pos += read_size;
	}
	*p_size = read_size;
	return SZ_OK;
}

struct LzmaOutStream {
	ISeqOutStream vt;
	LocalVector<uint8_t> data;
};

static size_t _lzma_out_stream_write(ISeqOutStreamPtr pp, const void *p_buf, size_t p_size) {
	LzmaOutStream *stream = Z7_CONTAINER_FROM_VTBL(pp, LzmaOutStream, vt); // NOLINT(modernize-use-bool-literals)
	if (p_size == 0) {
		return 0;
	}
	const uint8_t *buf_ptr = reinterpret_cast<const uint8_t *>(p_buf);
	const uint32_t old_size = stream->data.size();
	stream->data.resize(old_size + p_size);
	memcpy(stream->data.ptr() + old_size, buf_ptr, p_size);
	return p_size;
}

static Error _lzma_error_to_godot(int p_res) {
	switch (p_res) {
		case SZ_OK:
			return OK;
		case SZ_ERROR_MEM:
			return ERR_OUT_OF_MEMORY;
		case SZ_ERROR_UNSUPPORTED:
		case SZ_ERROR_PARAM:
			return ERR_INVALID_PARAMETER;
		case SZ_ERROR_INPUT_EOF:
		case SZ_ERROR_DATA:
			return ERR_FILE_CORRUPT;
		default:
			return ERR_CANT_CREATE;
	}
}

} //namespace

Error compress_lzma2(const Vector<uint8_t> &p_src, Vector<uint8_t> &r_dst, const PCKLzmaOptions &p_options) {
	r_dst.clear();
	if (p_src.is_empty()) {
		return OK;
	}

	CLzma2EncProps props;
	Lzma2EncProps_Init(&props);
	props.lzmaProps.level = CLAMP(p_options.compression_level, 0, 9);
	props.lzmaProps.dictSize = (UInt32)CLAMP(p_options.dictionary_size_mb, 1, 1536) * 1024 * 1024;
	props.lzmaProps.fb = CLAMP(p_options.word_size, 5, 273);
	props.lzmaProps.numThreads = CLAMP(p_options.threads, 1, 1);
	props.numTotalThreads = CLAMP(p_options.threads, 1, 1);

	CLzma2EncHandle enc = Lzma2Enc_Create(&g_alloc, &g_alloc_temp);
	ERR_FAIL_NULL_V(enc, ERR_OUT_OF_MEMORY);

	const SRes set_res = Lzma2Enc_SetProps(enc, &props);
	if (set_res != SZ_OK) {
		Lzma2Enc_Destroy(enc);
		return _lzma_error_to_godot(set_res);
	}
	Lzma2Enc_SetDataSize(enc, (UInt64)p_src.size());
	const Byte prop = Lzma2Enc_WriteProperties(enc);

	LzmaInStream in_stream;
	in_stream.vt.Read = _lzma_in_stream_read;
	in_stream.data = p_src.ptr();
	in_stream.size = p_src.size();
	in_stream.pos = 0;

	LzmaOutStream out_stream;
	out_stream.vt.Write = _lzma_out_stream_write;

	const SRes enc_res = Lzma2Enc_Encode2(enc, &out_stream.vt, nullptr, nullptr, &in_stream.vt, nullptr, 0, nullptr);
	Lzma2Enc_Destroy(enc);
	if (enc_res != SZ_OK) {
		return _lzma_error_to_godot(enc_res);
	}

	r_dst.resize(1 + out_stream.data.size());
	r_dst.write[0] = prop;
	if (!out_stream.data.is_empty()) {
		memcpy(r_dst.ptrw() + 1, out_stream.data.ptr(), out_stream.data.size());
	}
	return OK;
}

Error decompress_lzma2(const uint8_t *p_src, uint64_t p_src_size, uint64_t p_expected_size, Vector<uint8_t> &r_dst) {
	r_dst.clear();
	ERR_FAIL_NULL_V(p_src, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_src_size < 1, ERR_FILE_CORRUPT);
	ERR_FAIL_COND_V(p_expected_size > INT32_MAX, ERR_INVALID_PARAMETER);

	const Byte prop = p_src[0];
	const Byte *data = reinterpret_cast<const Byte *>(p_src + 1);
	SizeT data_size = (SizeT)(p_src_size - 1);

	r_dst.resize((int)p_expected_size);
	SizeT dst_size = (SizeT)p_expected_size;
	ELzmaStatus status = LZMA_STATUS_NOT_SPECIFIED;
	const SRes res = Lzma2Decode(reinterpret_cast<Byte *>(r_dst.ptrw()), &dst_size, data, &data_size, prop, LZMA_FINISH_END, &status, &g_alloc);
	if (res != SZ_OK) {
		r_dst.clear();
		return _lzma_error_to_godot(res);
	}

	if (dst_size != (SizeT)p_expected_size || data_size != (SizeT)(p_src_size - 1) || status != LZMA_STATUS_FINISHED_WITH_MARK) {
		r_dst.clear();
		return ERR_FILE_CORRUPT;
	}
	return OK;
}
