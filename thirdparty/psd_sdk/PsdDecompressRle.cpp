// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdDecompressRle.h"

#include "PsdAssert.h"
#include "PsdLog.h"
#include <cstring>


PSD_NAMESPACE_BEGIN

namespace imageUtil
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	void DecompressRle(const uint8_t* PSD_RESTRICT src, unsigned int srcSize, uint8_t* PSD_RESTRICT dest, unsigned int size)
	{
		PSD_ASSERT_NOT_NULL(src);
		PSD_ASSERT_NOT_NULL(dest);

		unsigned int bytesRead = 0u;
		unsigned int offset = 0u;
		while (offset < size)
		{
			if (bytesRead >= srcSize)
			{
				PSD_ERROR("DecompressRle", "Malformed RLE data encountered");
				return;
			}

			const uint8_t byte = *src++;
			++bytesRead;

			if (byte == 0x80)
			{
				// byte == -128 (0x80) is a no-op
			}
			// 0x81 - 0XFF
			else if (byte > 0x80)
			{
				// next 257-byte bytes are replicated from the next source byte
				const unsigned int count = static_cast<unsigned int>(257 - byte);

				memset(dest + offset, *src++, count);
				offset += count;

				++bytesRead;
			}
			// 0x00 - 0x7F
			else
			{
				// copy next byte+1 bytes 1-by-1
				const unsigned int count = static_cast<unsigned int>(byte + 1);
				
				memcpy(dest + offset, src, count);

				src += count;
				offset += count;

				bytesRead += count;
			}
		}
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	unsigned int CompressRle(const uint8_t* PSD_RESTRICT src, uint8_t* PSD_RESTRICT dest, unsigned int size)
	{
		PSD_ASSERT_NOT_NULL(src);
		PSD_ASSERT_NOT_NULL(dest);

		unsigned int runLength = 0u;
		unsigned int nonRunLength = 0u;

		unsigned int rleDataSize = 0u;
		for (unsigned int i = 1u; i < size; ++i)
		{
			const uint8_t previous = src[i - 1];
			const uint8_t current = src[i];
			if (previous == current)
			{
				if (nonRunLength != 0u)
				{
					// first repeat of a character

					// write non-run bytes so far
					*dest++ = static_cast<uint8_t>(nonRunLength - 1u);
					memcpy(dest, src + i - nonRunLength - 1u, nonRunLength);
					dest += nonRunLength;
					rleDataSize += 1u + nonRunLength;

					nonRunLength = 0u;
				}

				// belongs to the same run
				++runLength;

				// maximum length of a run is 128
				if (runLength == 128u)
				{
					// need to manually stop this run and write to output
					*dest++ = static_cast<uint8_t>(257u - runLength);
					*dest++ = current;
					rleDataSize += 2u;

					runLength = 0u;
				}
			}
			else
			{
				if (runLength != 0u)
				{
					// include first character and encode this run
					++runLength;

					*dest++ = static_cast<uint8_t>(257u - runLength);
					*dest++ = previous;
					rleDataSize += 2u;

					runLength = 0u;
				}
				else
				{
					++nonRunLength;
				}

				// maximum length of a non-run is 128 bytes
				if (nonRunLength == 128u)
				{
					*dest++ = static_cast<uint8_t>(nonRunLength - 1u);
					memcpy(dest, src + i - nonRunLength, nonRunLength);
					dest += nonRunLength;
					rleDataSize += 1u + nonRunLength;

					nonRunLength = 0u;
				}
			}
		}

		if (runLength != 0u)
		{
			++runLength;

			*dest++ = static_cast<uint8_t>(257u - runLength);
			*dest++ = src[size-1u];
			rleDataSize += 2u;
		}
		else
		{
			++nonRunLength;
			*dest++ = static_cast<uint8_t>(nonRunLength - 1u);
			memcpy(dest, src + size - nonRunLength, nonRunLength);
			dest += nonRunLength;
			rleDataSize += 1u + nonRunLength;
		}

		// pad to an even number of bytes
		if (rleDataSize & 1)
		{
			*dest++ = 0x80;
			++rleDataSize;
		}

		return rleDataSize;
	}
}

PSD_NAMESPACE_END
