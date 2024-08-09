// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

namespace imageUtil
{
	/// \ingroup ImageUtil
	/// Turns planar 8-bit RGB data into interleaved RGBA data with a constant, predefined alpha.
	/// The destination buffer \a dest must hold "width*height*4" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void InterleaveRGB(const uint8_t* PSD_RESTRICT srcR, const uint8_t* PSD_RESTRICT srcG, const uint8_t* PSD_RESTRICT srcB, uint8_t alpha, uint8_t* PSD_RESTRICT dest, unsigned int width, unsigned int height);

	/// \ingroup ImageUtil
	/// Turns planar 8-bit RGBA data into interleaved RGBA data.
	/// The destination buffer \a dest must hold "width*height*4" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void InterleaveRGBA(const uint8_t* PSD_RESTRICT srcR, const uint8_t* PSD_RESTRICT srcG, const uint8_t* PSD_RESTRICT srcB, const uint8_t* PSD_RESTRICT srcA, uint8_t* PSD_RESTRICT dest, unsigned int width, unsigned int height);


	/// \ingroup ImageUtil
	/// Turns planar 16-bit RGB data into interleaved RGBA data with a constant, predefined alpha.
	/// The destination buffer \a dest must hold "width*height*8" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void InterleaveRGB(const uint16_t* PSD_RESTRICT srcR, const uint16_t* PSD_RESTRICT srcG, const uint16_t* PSD_RESTRICT srcB, uint16_t alpha, uint16_t* PSD_RESTRICT dest, unsigned int width, unsigned int height);

	/// \ingroup ImageUtil
	/// Turns planar 16-bit RGBA data into interleaved RGBA data.
	/// The destination buffer \a dest must hold "width*height*8" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void InterleaveRGBA(const uint16_t* PSD_RESTRICT srcR, const uint16_t* PSD_RESTRICT srcG, const uint16_t* PSD_RESTRICT srcB, const uint16_t* PSD_RESTRICT srcA, uint16_t* PSD_RESTRICT dest, unsigned int width, unsigned int height);


	/// \ingroup ImageUtil
	/// Turns planar 32-bit RGB data into interleaved RGBA data with a constant, predefined alpha.
	/// The destination buffer \a dest must hold "width*height*16" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void InterleaveRGB(const float32_t* PSD_RESTRICT srcR, const float32_t* PSD_RESTRICT srcG, const float32_t* PSD_RESTRICT srcB, float32_t alpha, float32_t* PSD_RESTRICT dest, unsigned int width, unsigned int height);

	/// \ingroup ImageUtil
	/// Turns planar 32-bit RGBA data into interleaved RGBA data.
	/// The destination buffer \a dest must hold "width*height*16" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void InterleaveRGBA(const float32_t* PSD_RESTRICT srcR, const float32_t* PSD_RESTRICT srcG, const float32_t* PSD_RESTRICT srcB, const float32_t* PSD_RESTRICT srcA, float32_t* PSD_RESTRICT dest, unsigned int width, unsigned int height);


	/// \ingroup ImageUtil
	/// Turns interleaved 8-bit RGB data into planar 8-bit data.
	/// The destination buffers must hold "width*height" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void DeinterleaveRGB(const uint8_t* PSD_RESTRICT rgb, uint8_t* PSD_RESTRICT destR, uint8_t* PSD_RESTRICT destG, uint8_t* PSD_RESTRICT destB, unsigned int width, unsigned int height);

	/// \ingroup ImageUtil
	/// Turns interleaved 8-bit RGBA data into planar 8-bit data.
	/// The destination buffers must hold "width*height" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void DeinterleaveRGBA(const uint8_t* PSD_RESTRICT rgba, uint8_t* PSD_RESTRICT destR, uint8_t* PSD_RESTRICT destG, uint8_t* PSD_RESTRICT destB, uint8_t* PSD_RESTRICT destA, unsigned int width, unsigned int height);


	/// \ingroup ImageUtil
	/// Turns interleaved 16-bit RGB data into planar 16-bit data.
	/// The destination buffers must hold "width*height*2" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void DeinterleaveRGB(const uint16_t* PSD_RESTRICT rgb, uint16_t* PSD_RESTRICT destR, uint16_t* PSD_RESTRICT destG, uint16_t* PSD_RESTRICT destB, unsigned int width, unsigned int height);

	/// \ingroup ImageUtil
	/// Turns interleaved 16-bit RGBA data into planar 16-bit data.
	/// The destination buffers must hold "width*height*2" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void DeinterleaveRGBA(const uint16_t* PSD_RESTRICT rgba, uint16_t* PSD_RESTRICT destR, uint16_t* PSD_RESTRICT destG, uint16_t* PSD_RESTRICT destB, uint16_t* PSD_RESTRICT destA, unsigned int width, unsigned int height);


	/// \ingroup ImageUtil
	/// Turns interleaved 32-bit RGB data into planar 32-bit data.
	/// The destination buffers must hold "width*height*4" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void DeinterleaveRGB(const float32_t* PSD_RESTRICT rgb, float32_t* PSD_RESTRICT destR, float32_t* PSD_RESTRICT destG, float32_t* PSD_RESTRICT destB, unsigned int width, unsigned int height);

	/// \ingroup ImageUtil
	/// Turns interleaved 32-bit RGBA data into planar 32-bit data.
	/// The destination buffers must hold "width*height*4" bytes.
	/// \remark All given buffers (both source and destination) must be aligned to 16 bytes.
	void DeinterleaveRGBA(const float32_t* PSD_RESTRICT rgba, float32_t* PSD_RESTRICT destR, float32_t* PSD_RESTRICT destG, float32_t* PSD_RESTRICT destB, float32_t* PSD_RESTRICT destA, unsigned int width, unsigned int height);
}

PSD_NAMESPACE_END
