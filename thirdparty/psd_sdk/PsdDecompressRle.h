// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

namespace imageUtil
{
	/// \ingroup ImageUtil
	/// Decompresses a block of RLE encoded data using the PackBits (http://en.wikipedia.org/wiki/PackBits) algorithm.
	void DecompressRle(const uint8_t* PSD_RESTRICT src, unsigned int srcSize, uint8_t* PSD_RESTRICT dest, unsigned int size);

	/// \ingroup ImageUtil
	/// Compresses a block of data to RLE encoded data using the PackBits (http://en.wikipedia.org/wiki/PackBits) algorithm.
	/// \a dest must hold \a size * 2 bytes.
	unsigned int CompressRle(const uint8_t* PSD_RESTRICT src, uint8_t* PSD_RESTRICT dest, unsigned int size);
}

PSD_NAMESPACE_END
