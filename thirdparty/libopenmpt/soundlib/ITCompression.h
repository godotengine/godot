/*
 * ITCompression.h
 * ---------------
 * Purpose: Code for IT sample compression and decompression.
 * Notes  : The original Python compression code was written by GreaseMonkey and has been released into the public domain.
 * Authors: OpenMPT Devs
 *          Ben "GreaseMonkey" Russell
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#include <vector>
#include <iosfwd>
#include "Snd_defs.h"
#include "../common/FileReader.h"


OPENMPT_NAMESPACE_BEGIN

struct ModSample;

class ITCompression
{
public:
	ITCompression(const ModSample &sample, bool it215, std::ostream *f, SmpLength maxLength = 0);
	size_t GetCompressedSize() const { return packedTotalLength; }

	static const size_t bufferSize = 2 + 0xFFFF;	// Our output buffer can't be longer than this.
	static const size_t blockSize = 0x8000;			// Block size (in bytes) in which samples are being processed

protected:
	std::vector<int8> bwt;			// Bit width table for each sampling point
	uint8 *packedData;				// Compressed data for current sample block
	std::ostream *file;				// File to which compressed data will be written (can be nullptr if you only want to find out the sample size)
	void *sampleData;				// Pre-processed sample data for currently compressed sample block
	const ModSample &mptSample;		// Sample that is being processed
	size_t packedLength;			// Size of currently compressed sample block
	size_t packedTotalLength;		// Size of all compressed data so far
	SmpLength baseLength;			// Length of the currently compressed sample block (in samples)

	// Bit writer
	int8 bitPos;	// Current bit position in this byte
	int8 remBits;	// Remaining bits in this byte
	uint8 byteVal;	// Current byte value to be written

	bool is215;		// Use IT2.15 compression (double deltas)

	template<typename T>
	static void CopySample(void *target, const void *source, SmpLength offset, SmpLength length, SmpLength skip);

	template<typename T>
	void Deltafy();

	template<typename Properties>
	void Compress(const void *data, SmpLength offset, SmpLength actualLength);

	static int8 GetWidthChangeSize(int8 w, bool is16);

	template<typename Properties>
	void SquishRecurse(int8 sWidth, int8 lWidth, int8 rWidth, int8 width, SmpLength offset, SmpLength length);

	static int8 ConvertWidth(int8 curWidth, int8 newWidth);
	void WriteBits(int8 width, int v);

	void WriteByte(uint8 v);
};


class ITDecompression
{
public:
	ITDecompression(FileReader &file, ModSample &sample, bool it215);

protected:
	ModSample &mptSample;		// Sample that is being processed

	SmpLength writtenSamples;	// Number of samples so far written on this channel
	SmpLength writePos;			// Absolut write position in sample (for stereo samples)
	SmpLength curLength;		// Length of currently processed block
	unsigned int mem1, mem2;	// Integrator memory
	int dataPos, dataSize;		// Position in and size of input block

	// Bit reader
	int bitPos;					// Current bit position in this byte
	int remBits;				// Remaining bits in this byte

	bool is215;					// Use IT2.15 compression (double deltas)

	mpt::byte chunk[65535];		// Input block

	template<typename Properties>
	void Uncompress(typename Properties::sample_t *target);
	static void ChangeWidth(int &curWidth, int width);
	int ReadBits(int width);

	template<typename Properties>
	void Write(int v, int topbit, typename Properties::sample_t *target);
};


OPENMPT_NAMESPACE_END
