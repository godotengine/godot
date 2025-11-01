/**************************************************************************/
/*  delta_encoding.cpp                                                    */
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

#include "delta_encoding.h"

#include <zstd.h>

#define ERR_FAIL_ZSTD_V_MSG(m_result, m_retval, m_msg) \
	ERR_FAIL_COND_V_MSG(ZSTD_isError(m_result), m_retval, vformat("%s Zstandard reported error code %d: \"%s\".", m_msg, ZSTD_getErrorCode(m_result), ZSTD_getErrorString(ZSTD_getErrorCode(m_result))))

struct ZstdCompressionContext {
	ZSTD_CCtx *context = ZSTD_createCCtx();
	~ZstdCompressionContext() { ZSTD_freeCCtx(context); }
	operator ZSTD_CCtx *() { return context; }
};

struct ZstdDecompressionContext {
	ZSTD_DCtx *context = ZSTD_createDCtx();
	~ZstdDecompressionContext() { ZSTD_freeDCtx(context); }
	operator ZSTD_DCtx *() { return context; }
};

struct DeltaHeader {
	uint8_t magic[4];
	uint8_t version;
};

static_assert(alignof(DeltaHeader) == 1); // There shouldn't be any padding.

static constexpr size_t DELTA_HEADER_SIZE = sizeof(DeltaHeader);
static constexpr uint8_t DELTA_MAGIC[4] = { 'G', 'D', 'D', 'L' };
static constexpr int DELTA_VERSION_NUMBER = 1;

Error DeltaEncoding::encode_delta(Span<uint8_t> p_old_data, Span<uint8_t> p_new_data, Vector<uint8_t> &r_delta, int p_compression_level) {
	size_t zstd_result = ZSTD_compressBound(p_new_data.size());
	ERR_FAIL_ZSTD_V_MSG(zstd_result, FAILED, "Failed to encode delta. Calculating compression bounds failed.");

	r_delta.reserve_exact(DELTA_HEADER_SIZE + zstd_result);
	r_delta.resize(DELTA_HEADER_SIZE + zstd_result);

	DeltaHeader header;
	memcpy(header.magic, DELTA_MAGIC, 4);
	header.version = DELTA_VERSION_NUMBER;
	memcpy(r_delta.ptrw(), &header, DELTA_HEADER_SIZE);

	ZstdCompressionContext zstd_context;

	ZSTD_parameters zstd_params = ZSTD_getParams(p_compression_level, p_new_data.size(), p_old_data.size());
	zstd_params.fParams.contentSizeFlag = 1;
	zstd_params.fParams.checksumFlag = 1;

	zstd_result = ZSTD_CCtx_setParams(zstd_context, zstd_params);
	ERR_FAIL_ZSTD_V_MSG(zstd_result, FAILED, "Failed to encode delta. Setting compression parameters failed.");

	zstd_result = ZSTD_CCtx_refPrefix(zstd_context, p_old_data.ptr(), p_old_data.size());
	ERR_FAIL_ZSTD_V_MSG(zstd_result, FAILED, "Failed to encode delta. Setting prefix dictionary failed.");

	zstd_result = ZSTD_compress2(zstd_context, r_delta.ptrw() + DELTA_HEADER_SIZE, r_delta.size() - DELTA_HEADER_SIZE, p_new_data.ptr(), p_new_data.size());
	ERR_FAIL_ZSTD_V_MSG(zstd_result, FAILED, "Failed to encode delta. Compression failed.");

	r_delta.resize(DELTA_HEADER_SIZE + zstd_result);

	return OK;
}

Error DeltaEncoding::decode_delta(Span<uint8_t> p_old_data, Span<uint8_t> p_delta, Vector<uint8_t> &r_new_data) {
	ERR_FAIL_COND_V_MSG(p_delta.size() < DELTA_HEADER_SIZE, ERR_INVALID_DATA, vformat("Failed to decode delta. File size (%d) is too small.", p_delta.size()));

	DeltaHeader header;
	memcpy(&header, p_delta.ptr(), DELTA_HEADER_SIZE);

	ERR_FAIL_COND_V_MSG(memcmp(header.magic, DELTA_MAGIC, 4) != 0, ERR_FILE_CORRUPT, "Failed to decode delta. Header is invalid.");
	ERR_FAIL_COND_V_MSG(header.version != DELTA_VERSION_NUMBER, ERR_FILE_UNRECOGNIZED, vformat("Failed to decode delta. Expected version %d but found %d.", DELTA_VERSION_NUMBER, header.version));

	size_t zstd_result = ZSTD_findDecompressedSize(p_delta.ptr() + DELTA_HEADER_SIZE, p_delta.size() - DELTA_HEADER_SIZE);
	ERR_FAIL_ZSTD_V_MSG(zstd_result, FAILED, "Failed to decode delta. Unable to find decompressed size.");

	r_new_data.reserve_exact(zstd_result);
	r_new_data.resize(zstd_result);

	ZstdDecompressionContext zstd_context;

	zstd_result = ZSTD_DCtx_refPrefix(zstd_context, p_old_data.ptr(), p_old_data.size());
	ERR_FAIL_ZSTD_V_MSG(zstd_result, FAILED, "Failed to decode delta. Setting prefix dictionary failed.");

	zstd_result = ZSTD_decompressDCtx(zstd_context, r_new_data.ptrw(), r_new_data.size(), p_delta.ptr() + DELTA_HEADER_SIZE, p_delta.size() - DELTA_HEADER_SIZE);
	ERR_FAIL_ZSTD_V_MSG(zstd_result, FAILED, "Failed to decode delta. Decompression failed.");
	ERR_FAIL_COND_V(zstd_result != (size_t)r_new_data.size(), ERR_FILE_CORRUPT);

	return OK;
}
