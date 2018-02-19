/*
 * ITCompression.cpp
 * -----------------
 * Purpose: Code for IT sample compression and decompression.
 * Notes  : The original Python compression code was written by GreaseMonkey and has been released into the public domain.
 * Authors: OpenMPT Devs
 *          Ben "GreaseMonkey" Russell
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include <ostream>
#include "ITCompression.h"
#include "../common/misc_util.h"
#include "../common/mptIO.h"
#include "ModSample.h"


OPENMPT_NAMESPACE_BEGIN


// Algorithm parameters for 16-Bit samples
struct IT16BitParams
{
	typedef int16 sample_t;
	static const int16 lowerTab[];
	static const int16 upperTab[];
	static const int8 fetchA = 4, lowerB = -8, upperB = 7, defWidth = 17;
	static const int mask = 0xFFFF;
};

const int16 IT16BitParams::lowerTab[] = { 0, -1, -3, -7, -15, -31, -56, -120, -248, -504, -1016, -2040, -4088, -8184, -16376, -32760, -32768 };
const int16 IT16BitParams::upperTab[] = { 0, 1, 3, 7, 15, 31, 55, 119, 247, 503, 1015, 2039, 4087, 8183, 16375, 32759, 32767 };

// Algorithm parameters for 8-Bit samples
struct IT8BitParams
{
	typedef int8 sample_t;
	static const int8 lowerTab[];
	static const int8 upperTab[];
	static const int8 fetchA = 3, lowerB = -4, upperB = 3, defWidth = 9;
	static const int mask = 0xFF;
};

const int8 IT8BitParams::lowerTab[] = { 0, -1, -3, -7, -15, -31, -60, -124, -128 };
const int8 IT8BitParams::upperTab[] = { 0, 1, 3, 7, 15, 31, 59, 123, 127 };

static const int8 ITWidthChangeSize[] = { 4, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };

//////////////////////////////////////////////////////////////////////////////
// IT 2.14 compression


ITCompression::ITCompression(const ModSample &sample, bool it215, std::ostream *f, SmpLength maxLength)
	: file(f)
	, mptSample(sample)
	, is215(it215)
{
	packedData = new (std::nothrow) uint8[bufferSize];
	sampleData = new (std::nothrow) uint8[blockSize];
	packedTotalLength = 0;
	if(packedData == nullptr || sampleData == nullptr)
	{
		return;
	}

	if(maxLength == 0 || maxLength > mptSample.nLength)
		maxLength = mptSample.nLength;
	for(uint8 chn = 0; chn < mptSample.GetNumChannels(); chn++)
	{
		SmpLength offset = 0;
		SmpLength remain = maxLength;
		while(remain > 0)
		{
			// Initialise output buffer and bit writer positions
			packedLength = 2;
			bitPos = 0;
			remBits = 8;
			byteVal = 0;

			if(mptSample.GetElementarySampleSize() > 1)
				Compress<IT16BitParams>(sample.pSample16 + chn, offset, remain);
			else
				Compress<IT8BitParams>(sample.pSample8 + chn, offset, remain);

			if(file) mpt::IO::WriteRaw(*file, packedData, packedLength);
			packedTotalLength += packedLength;

			offset += baseLength;
			remain -= baseLength;
		}
	}

	delete[] packedData;
	delete[] reinterpret_cast<uint8*>(sampleData);
}


template<typename T>
void ITCompression::CopySample(void *target, const void *source, SmpLength offset, SmpLength length, SmpLength skip)
{
	T *out = static_cast<T *>(target);
	const T *in = static_cast<const T *>(source) + offset * skip;
	for(SmpLength i = 0, j = 0; j < length; i += skip, j++)
	{
		out[j] = in[i];
	}
}


// Convert sample to delta values.
template<typename T>
void ITCompression::Deltafy()
{
	T *p = static_cast<T *>(sampleData);
	int oldVal = 0;
	for(SmpLength i = 0; i < baseLength; i++)
	{
		int newVal = p[i];
		p[i] = static_cast<T>(newVal - oldVal);
		oldVal = newVal;
	}
}


template<typename Properties>
void ITCompression::Compress(const void *data, SmpLength offset, SmpLength actualLength)
{
	baseLength = std::min(actualLength, SmpLength(blockSize / sizeof(typename Properties::sample_t)));

	CopySample<typename Properties::sample_t>(sampleData, data, offset, baseLength, mptSample.GetNumChannels());

	Deltafy<typename Properties::sample_t>();
	if(is215)
	{
		Deltafy<typename Properties::sample_t>();
	}

	const int8 defWidth = Properties::defWidth; // gcc static const member reference workaround

	// Initialise bit width table with initial values
	bwt.assign(baseLength, defWidth);

	// Recurse!
	SquishRecurse<Properties>(defWidth, defWidth, defWidth, defWidth - 2, 0, baseLength);
	
	// Write those bits!
	const typename Properties::sample_t *p = static_cast<typename Properties::sample_t *>(sampleData);
	int8 width = defWidth;
	for(size_t i = 0; i < baseLength; i++)
	{
		if(bwt[i] != width)
		{
			if(width <= 6)
			{
				// Mode A: 1 to 6 bits
				MPT_ASSERT(width);
				WriteBits(width, (1 << (width - 1)));
				WriteBits(Properties::fetchA, ConvertWidth(width, bwt[i]));
			} else if(width < defWidth)
			{
				// Mode B: 7 to 8 / 16 bits
				int xv = (1 << (width - 1)) + Properties::lowerB + ConvertWidth(width, bwt[i]);
				WriteBits(width, xv);
			} else
			{
				// Mode C: 9 / 17 bits
				MPT_ASSERT((bwt[i] - 1) >= 0);
				WriteBits(width, (1 << (width - 1)) + bwt[i] - 1);
			}

			width = bwt[i];
		}
		WriteBits(width, static_cast<int>(p[i]) & Properties::mask);
	}

	// Write last byte and update block length
	WriteByte(byteVal);
	packedData[0] = static_cast<uint8>((packedLength - 2) & 0xFF);
	packedData[1] = static_cast<uint8>((packedLength - 2) >> 8);
}


int8 ITCompression::GetWidthChangeSize(int8 w, bool is16)
{
	MPT_ASSERT(w > 0 && static_cast<unsigned int>(w) <= CountOf(ITWidthChangeSize));
	int8 wcs = ITWidthChangeSize[w - 1];
	if(w <= 6 && is16)
		wcs++;
	return wcs;
}


template<typename Properties>
void ITCompression::SquishRecurse(int8 sWidth, int8 lWidth, int8 rWidth, int8 width, SmpLength offset, SmpLength length)
{
	if(width + 1 < 1)
	{
		for(SmpLength i = offset; i < offset + length; i++)
			bwt[i] = sWidth;
		return;
	}

	MPT_ASSERT(width >= 0 && static_cast<unsigned int>(width) < CountOf(Properties::lowerTab));

	SmpLength i = offset;
	SmpLength end = offset + length;
	const typename Properties::sample_t *p = static_cast<typename Properties::sample_t *>(sampleData);

	while(i < end)
	{
		if(p[i] >= Properties::lowerTab[width] && p[i] <= Properties::upperTab[width])
		{
			SmpLength start = i;
			// Check for how long we can keep this bit width
			while(i < end && p[i] >= Properties::lowerTab[width] && p[i] <= Properties::upperTab[width])
			{
				i++;
			}

			const SmpLength blockLength = i - start;
			const int8 xlwidth = start == offset ? lWidth : sWidth;
			const int8 xrwidth = i == end ? rWidth : sWidth;

			const bool is16 = sizeof(typename Properties::sample_t) > 1;
			const int8 wcsl = GetWidthChangeSize(xlwidth, is16);
			const int8 wcss = GetWidthChangeSize(sWidth, is16);
			const int8 wcsw = GetWidthChangeSize(width + 1, is16);

			bool comparison;
			if(i == baseLength)
			{
				SmpLength keepDown = wcsl + (width + 1) * blockLength;
				SmpLength levelLeft = wcsl + sWidth * blockLength;

				if(xlwidth == sWidth)
					levelLeft -= wcsl;

				comparison = (keepDown <= levelLeft);
			} else
			{
				SmpLength keepDown = wcsl + (width + 1) * blockLength + wcsw;
				SmpLength levelLeft = wcsl + sWidth * blockLength + wcss;

				if(xlwidth == sWidth)
					levelLeft -= wcsl;
				if(xrwidth == sWidth)
					levelLeft -= wcss;

				comparison = (keepDown <= levelLeft);
			}
			SquishRecurse<Properties>(comparison ? (width + 1) : sWidth, xlwidth, xrwidth, width - 1, start, blockLength);
		} else
		{
			bwt[i] = sWidth;
			i++;
		}
	}
}


int8 ITCompression::ConvertWidth(int8 curWidth, int8 newWidth)
{
	curWidth--;
	newWidth--;
	MPT_ASSERT(newWidth != curWidth);
	if(newWidth > curWidth)
		newWidth--;
	return newWidth;
}


void ITCompression::WriteBits(int8 width, int v)
{
	while(width > remBits)
	{
		byteVal |= (v << bitPos);
		width -= remBits;
		v >>= remBits;
		bitPos = 0;
		remBits = 8;
		WriteByte(byteVal);
		byteVal = 0;
	}

	if(width > 0)
	{
		byteVal |= (v & ((1 << width) - 1)) << bitPos;
		remBits -= width;
		bitPos += width;
	}
}


void ITCompression::WriteByte(uint8 v)
{
	if(packedLength < bufferSize)
	{
		packedData[packedLength++] = v;
	} else
	{
		// How could this happen, anyway?
		MPT_ASSERT_NOTREACHED();
	}
}


//////////////////////////////////////////////////////////////////////////////
// IT 2.14 decompression


ITDecompression::ITDecompression(FileReader &file, ModSample &sample, bool it215)
	: mptSample(sample)
	, is215(it215)
{
	for(uint8 chn = 0; chn < mptSample.GetNumChannels(); chn++)
	{
		writtenSamples = writePos = 0;
		while(writtenSamples < sample.nLength && file.CanRead(sizeof(uint16)))
		{
			dataSize = file.ReadUint16LE();
			if(!dataSize)
				continue;	// Malformed sample?
			file.ReadRaw(chunk, dataSize);

			// Initialise bit reader
			dataPos = 0;
			bitPos = 0;
			remBits = 8;
			mem1 = mem2 = 0;

			if(mptSample.GetElementarySampleSize() > 1)
				Uncompress<IT16BitParams>(mptSample.pSample16 + chn);
			else
				Uncompress<IT8BitParams>(mptSample.pSample8 + chn);
		}
	}
}


template<typename Properties>
void ITDecompression::Uncompress(typename Properties::sample_t *target)
{
	curLength = std::min(mptSample.nLength - writtenSamples, SmpLength(ITCompression::blockSize / sizeof(typename Properties::sample_t)));

	const int defWidth = Properties::defWidth; // gcc static const member reference workaround

	int width = defWidth;
	while(curLength > 0)
	{
		if(width < 1 || width > defWidth || dataPos >= dataSize)
		{
			// Error!
			return;
		}

		int v = ReadBits(width);
		const int topBit = (1 << (width - 1));
		if(width <= 6)
		{
			// Mode A: 1 to 6 bits
			if(v == topBit)
				ChangeWidth(width, ReadBits(Properties::fetchA));
			else
				Write<Properties>(v, topBit, target);
		} else if(width < defWidth)
		{
			// Mode B: 7 to 8 / 16 bits
			if(v >= topBit + Properties::lowerB && v <= topBit + Properties::upperB)
				ChangeWidth(width, v - (topBit + Properties::lowerB));
			else
				Write<Properties>(v, topBit, target);
		} else
		{
			// Mode C: 9 / 17 bits
			if(v & topBit)
				width = (v & ~topBit) + 1;
			else
				Write<Properties>((v & ~topBit), 0, target);
		}
	}
}


#if MPT_MSVC_AT_LEAST(2017,3) && MPT_MSVC_BEFORE(2017,5)
// Work-around compiler crash in VS2017.3 / cl 19.11.25506
// https://developercommunity.visualstudio.com/content/problem/96687/c1063-and-c1001-while-compiling-trivial-code-in-vs.html
MPT_NOINLINE
#endif
void ITDecompression::ChangeWidth(int &curWidth, int width)
{
	width++;
	if(width >= curWidth)
		width++;
	curWidth = width;
}


#if MPT_MSVC_AT_LEAST(2017,3) && MPT_MSVC_BEFORE(2017,5)
// Work-around compiler crash in VS2017.3 / cl 19.11.25506
// https://developercommunity.visualstudio.com/content/problem/96687/c1063-and-c1001-while-compiling-trivial-code-in-vs.html
MPT_NOINLINE
#endif
int ITDecompression::ReadBits(int width)
{
	int v = 0, vPos = 0, vMask = (1 << width) - 1;
	while(width >= remBits && dataPos < dataSize)
	{
		v |= (chunk[dataPos] >> bitPos) << vPos;
		vPos += remBits;
		width -= remBits;
		dataPos++;
		remBits = 8;
		bitPos = 0;
	}

	if(width > 0 && dataPos < dataSize)
	{
		v |= (chunk[dataPos] >> bitPos) << vPos;
		v &= vMask;
		remBits -= width;
		bitPos += width;
	}
	return v;
}


template<typename Properties>
void ITDecompression::Write(int v, int topBit, typename Properties::sample_t *target)
{
	if(v & topBit)
		v -= (topBit << 1);
	mem1 += v;
	mem2 += mem1;
	target[writePos] = static_cast<typename Properties::sample_t>(static_cast<int>(is215 ? mem2 : mem1));
	writtenSamples++;
	writePos += mptSample.GetNumChannels();
	curLength--;
}


OPENMPT_NAMESPACE_END
