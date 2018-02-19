/*
 * SampleIO.h
 * ----------
 * Purpose: Central code for reading and writing samples. Create your SampleIO object and have a go at the ReadSample and WriteSample functions!
 * Notes  : Not all combinations of possible sample format combinations are implemented, especially for WriteSample.
 *          Using the existing generic sample conversion functors in SampleFormatConverters.h, it should be quite easy to extend the code, though.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


#include "../common/FileReaderFwd.h"


OPENMPT_NAMESPACE_BEGIN


struct ModSample;

// Sample import / export formats
class SampleIO
{
protected:
	typedef uint32 format_type;
	format_type format;

	// Internal bitmasks
	enum Offsets
	{
		bitOffset		= 0,
		channelOffset	= 8,
		endianOffset	= 16,
		encodingOffset	= 24,

		bitMask			= 0xFF << bitOffset,
		channelMask		= 0xFF << channelOffset,
		endianMask		= 0xFF << endianOffset,
		encodingMask	= 0x7F << encodingOffset, // 0xff will overflow signed 32bit int, which is the base type for an enum that fits, causing warnings when shifted
	};

public:
	// Bits per sample
	enum Bitdepth
	{
		_8bit	= 8,
		_16bit	= 16,
		_24bit	= 24,
		_32bit	= 32,
		_64bit	= 64,
	};

	// Number of channels + channel format
	enum Channels
	{
		mono = 0,
		stereoInterleaved,	// LRLRLR...
		stereoSplit,		// LLL...RRR...
	};

	// Sample byte order
	enum Endianness
	{
		littleEndian = 0,
		bigEndian = 1,
	};

	// Sample encoding
	enum Encoding
	{
		signedPCM = 0,      // Integer PCM, signed
		unsignedPCM,        // Integer PCM, unsigned
		deltaPCM,           // Integer PCM, delta-encoded
		floatPCM,           // Floating point PCM
		IT214,              // Impulse Tracker 2.14 compressed
		IT215,              // Impulse Tracker 2.15 compressed
		AMS,                // AMS / Velvet Studio packed
		DMF,                // DMF Huffman compression
		MDL,                // MDL Huffman compression
		PTM8Dto16,          // PTM 8-Bit delta value -> 16-Bit sample
		PCM7to8,            // 8-Bit sample data with unused high bit
		ADPCM,              // 4-Bit ADPCM-packed
		MT2,                // MadTracker 2 stereo delta encoding
		floatPCM15,         // Floating point PCM with 2^15 full scale
		floatPCM23,         // Floating point PCM with 2^23 full scale
		floatPCMnormalize,  // Floating point PCM and data will be normalized while reading
		signedPCMnormalize, // Integer PCM and data will be normalized while reading
		uLaw,               // 8-to-16 bit G.711 u-law compression
		aLaw,               // 8-to-16 bit G.711 a-law compression
	};

	SampleIO(Bitdepth bits = _8bit, Channels channels = mono, Endianness endianness = littleEndian, Encoding encoding = signedPCM)
	{
		format = (bits << bitOffset) | (channels << channelOffset) | (endianness << endianOffset) | (encoding << encodingOffset);
	}

	SampleIO(const SampleIO &other) : format(other.format) { }

	bool operator== (const SampleIO &other) const
	{
		return format == other.format;
	}
	bool operator!= (const SampleIO &other) const
	{
		return !(*this == other);
	}

	void operator|= (Bitdepth bits)
	{
		format = (format & ~bitMask) | (bits << bitOffset);
	}

	void operator|= (Channels channels)
	{
		format = (format & ~channelMask) | (channels << channelOffset);
	}

	void operator|= (Endianness endianness)
	{
		format = (format & ~endianMask) | (endianness << endianOffset);
	}

	void operator|= (Encoding encoding)
	{
		format = (format & ~encodingMask) | (encoding << encodingOffset);
	}

	static inline Endianness GetNativeEndianness()
	{
		const mpt::endian_type endian = mpt::endian();
		MPT_ASSERT((endian == mpt::endian_little) || (endian == mpt::endian_big));
		Endianness result = littleEndian;
		MPT_MAYBE_CONSTANT_IF(endian == mpt::endian_little)
		{
			result = littleEndian;
		}
		MPT_MAYBE_CONSTANT_IF(endian == mpt::endian_big)
		{
			result = bigEndian;
		}
		return result;
	}

	void MayNormalize()
	{
		if(GetBitDepth() == 24 || GetBitDepth() == 32)
		{
			if(GetEncoding() == SampleIO::signedPCM)
			{
				(*this) |= SampleIO::signedPCMnormalize;
			} else if(GetEncoding() == SampleIO::floatPCM)
			{
				(*this) |= SampleIO::floatPCMnormalize;
			}
		}
	}

	// Return 0 in case of variable-length encoded samples.
	uint8 GetEncodedBitsPerSample() const
	{
		uint8 result = 0;
		switch(GetEncoding())
		{
			case signedPCM:// Integer PCM, signed
				result = GetBitDepth();
				break;
			case unsignedPCM://Integer PCM, unsigned
				result = GetBitDepth();
				break;
			case deltaPCM:// Integer PCM, delta-encoded
				result = GetBitDepth();
				break;
			case floatPCM:// Floating point PCM
				result = GetBitDepth();
				break;
			case IT214:// Impulse Tracker 2.14 compressed
				result = 0; // variable-length compressed
				break;
			case IT215:// Impulse Tracker 2.15 compressed
				result = 0; // variable-length compressed
				break;
			case AMS:// AMS / Velvet Studio packed
				result = 0; // variable-length compressed
				break;
			case DMF:// DMF Huffman compression
				result = 0; // variable-length compressed
				break;
			case MDL:// MDL Huffman compression
				result = 0; // variable-length compressed
				break;
			case PTM8Dto16:// PTM 8-Bit delta value -> 16-Bit sample
				result = 16;
				break;
			case PCM7to8:// 8-Bit sample data with unused high bit
				result = 8;
				break;
			case ADPCM:// 4-Bit ADPCM-packed
				result = 4;
				break;
			case MT2:// MadTracker 2 stereo delta encoding
				result = GetBitDepth();
				break;
			case floatPCM15:// Floating point PCM with 2^15 full scale
				result = GetBitDepth();
				break;
			case floatPCM23:// Floating point PCM with 2^23 full scale
				result = GetBitDepth();
				break;
			case floatPCMnormalize:// Floating point PCM and data will be normalized while reading
				result = GetBitDepth();
				break;
			case signedPCMnormalize:// Integer PCM and data will be normalized while reading
				result = GetBitDepth();
				break;
			case uLaw:// G.711 u-law
				result = 8;
				break;
			case aLaw:// G.711 a-law
				result = 8;
				break;
		}
		return result;
	}

	// Return the static header size additional to the raw encoded sample data.
	std::size_t GetEncodedHeaderSize() const
	{
		std::size_t result = 0;
		if(GetEncoding() == ADPCM)
		{
			result = 16;
		}
		return result;
	}

	// Returns true if the encoded size cannot be calculated apriori from the encoding format and the sample length.
	bool IsVariableLengthEncoded() const
	{
		return GetEncodedBitsPerSample() == 0;
	}

	bool UsesFileReaderForDecoding() const
	{
		if(GetEncoding() == IT214 || GetEncoding() == IT215)
		{
			// IT compressed samples use FileReader interface and thus do not need to call GetPinnedRawDataView()
			return true;
		}
		if(GetEncoding() == AMS || GetEncoding() == MDL)
		{
			return true;
		}
		return false;
	}

	// Get bits per sample
	uint8 GetBitDepth() const
	{
		return static_cast<uint8>((format & bitMask) >> bitOffset);
	}
	// Get channel layout
	Channels GetChannelFormat() const
	{
		return static_cast<Channels>((format & channelMask) >> channelOffset);
	}
	// Get number of channels
	uint8 GetNumChannels() const
	{
		return GetChannelFormat() == mono ? 1u : 2u;
	}
	// Get sample byte order
	Endianness GetEndianness() const
	{
		return static_cast<Endianness>((format & endianMask) >> endianOffset);
	}
	// Get sample format / encoding
	Encoding GetEncoding() const
	{
		return static_cast<Encoding>((format & encodingMask) >> encodingOffset);
	}

	// Returns the encoded size of the sample. In case of variable-length encoding returns 0.
	std::size_t CalculateEncodedSize(SmpLength length) const
	{
		if(IsVariableLengthEncoded())
		{
			return 0;
		}
		if(GetEncodedBitsPerSample() % 8 != 0)
		{
			MPT_ASSERT(GetEncoding() == ADPCM && GetEncodedBitsPerSample() == 4);
			return GetEncodedHeaderSize() + (((length + 1) / 2) * GetNumChannels()); // round up
		}
		return GetEncodedHeaderSize() + (length * (GetEncodedBitsPerSample()/8) * GetNumChannels());
	}

	// Read a sample from memory
	size_t ReadSample(ModSample &sample, FileReader &file) const;

#ifndef MODPLUG_NO_FILESAVE
	// Optionally write a sample to file
	size_t WriteSample(std::ostream *f, const ModSample &sample, SmpLength maxSamples = 0) const;
	// Write a sample to file
	size_t WriteSample(std::ostream &f, const ModSample &sample, SmpLength maxSamples = 0) const;
	// Write a sample to file
	size_t WriteSample(FILE *f, const ModSample &sample, SmpLength maxSamples = 0) const;
#endif // MODPLUG_NO_FILESAVE
};


OPENMPT_NAMESPACE_END
