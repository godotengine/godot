/*
 * WAVTools.h
 * ----------
 * Purpose: Definition of WAV file structures and helper functions
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "ChunkReader.h"
#include "Loaders.h"
#include "../common/mptUUID.h"

OPENMPT_NAMESPACE_BEGIN

struct FileTags;

// RIFF header
struct RIFFHeader
{
	// 32-Bit chunk identifiers
	enum RIFFMagic
	{
		idRIFF	= MAGIC4LE('R','I','F','F'),	// magic for WAV files
		idLIST	= MAGIC4LE('L','I','S','T'),	// magic for samples in DLS banks
		idWAVE	= MAGIC4LE('W','A','V','E'),	// type for WAV files
		idwave	= MAGIC4LE('w','a','v','e'),	// type for samples in DLS banks
	};

	uint32le magic;		// RIFF (in WAV files) or LIST (in DLS banks)
	uint32le length;	// Size of the file, not including magic and length
	uint32le type;		// WAVE (in WAV files) or wave (in DLS banks)
};

MPT_BINARY_STRUCT(RIFFHeader, 12)


// General RIFF Chunk header
struct RIFFChunk
{
	// 32-Bit chunk identifiers
	enum ChunkIdentifiers
	{
		idfmt_	= MAGIC4LE('f','m','t',' '),	// Sample format information
		iddata	= MAGIC4LE('d','a','t','a'),	// Sample data
		idpcm_	= MAGIC4LE('p','c','m',' '),	// IMA ADPCM samples
		idfact	= MAGIC4LE('f','a','c','t'),	// Compressed samples
		idsmpl	= MAGIC4LE('s','m','p','l'),	// Sampler and loop information
		idinst	= MAGIC4LE('i','n','s','t'),	// Instrument information
		idLIST	= MAGIC4LE('L','I','S','T'),	// List of chunks
		idxtra	= MAGIC4LE('x','t','r','a'),	// OpenMPT extra infomration
		idcue_	= MAGIC4LE('c','u','e',' '),	// Cue points
		idwsmp	= MAGIC4LE('w','s','m','p'),	// DLS bank samples
		id____	= 0x00000000,	// Found when loading buggy MPT samples

		// Identifiers in "LIST" chunk
		idINAM	= MAGIC4LE('I','N','A','M'), // title
		idISFT	= MAGIC4LE('I','S','F','T'), // software
		idICOP	= MAGIC4LE('I','C','O','P'), // copyright
		idIART	= MAGIC4LE('I','A','R','T'), // artist
		idIPRD	= MAGIC4LE('I','P','R','D'), // product (album)
		idICMT	= MAGIC4LE('I','C','M','T'), // comment
		idIENG	= MAGIC4LE('I','E','N','G'), // engineer
		idISBJ	= MAGIC4LE('I','S','B','J'), // subject
		idIGNR	= MAGIC4LE('I','G','N','R'), // genre
		idICRD	= MAGIC4LE('I','C','R','D'), // date created

		idYEAR  = MAGIC4LE('Y','E','A','R'), // year
		idTRCK  = MAGIC4LE('T','R','C','K'), // track number
		idTURL  = MAGIC4LE('T','U','R','L'), // url
	};

	uint32le id;		// See ChunkIdentifiers
	uint32le length;	// Chunk size without header

	size_t GetLength() const
	{
		return length;
	}

	ChunkIdentifiers GetID() const
	{
		return static_cast<ChunkIdentifiers>(id.get());
	}
};

MPT_BINARY_STRUCT(RIFFChunk, 8)


// Format Chunk
struct WAVFormatChunk
{
	// Sample formats
	enum SampleFormats
	{
		fmtPCM			= 1,
		fmtFloat		= 3,
		fmtALaw			= 6,
		fmtULaw			= 7,
		fmtIMA_ADPCM	= 17,
		fmtMP3			= 85,
		fmtExtensible	= 0xFFFE,
	};

	uint16le format;			// Sample format, see SampleFormats
	uint16le numChannels;		// Number of audio channels
	uint32le sampleRate;		// Sample rate in Hz
	uint32le byteRate;			// Bytes per second (should be freqHz * blockAlign)
	uint16le blockAlign;		// Size of a sample, in bytes (do not trust this value, it's incorrect in some files)
	uint16le bitsPerSample;		// Bits per sample
};

MPT_BINARY_STRUCT(WAVFormatChunk, 16)


// Extension of the WAVFormatChunk structure, used if format == formatExtensible
struct WAVFormatChunkExtension
{
	uint16le size;
	uint16le validBitsPerSample;
	uint32le channelMask;
	GUIDms   subFormat;
};

MPT_BINARY_STRUCT(WAVFormatChunkExtension, 24)


// Sample information chunk
struct WAVSampleInfoChunk
{
	uint32le manufacturer;
	uint32le product;
	uint32le samplePeriod;	// 1000000000 / sampleRate
	uint32le baseNote;		// MIDI base note of sample
	uint32le pitchFraction;
	uint32le SMPTEFormat;
	uint32le SMPTEOffset;
	uint32le numLoops;		// number of loops
	uint32le samplerData;

	// Set up information
	void ConvertToWAV(uint32 freq, uint8 rootNote)
	{
		manufacturer = 0;
		product = 0;
		samplePeriod = 1000000000 / freq;
		if(rootNote != 0)
			baseNote = rootNote - NOTE_MIN;
		else
			baseNote = NOTE_MIDDLEC - NOTE_MIN;
		pitchFraction = 0;
		SMPTEFormat = 0;
		SMPTEOffset = 0;
		numLoops = 0;
		samplerData = 0;
	}
};

MPT_BINARY_STRUCT(WAVSampleInfoChunk, 36)


// Sample loop information chunk (found after WAVSampleInfoChunk in "smpl" chunk)
struct WAVSampleLoop
{
	// Sample Loop Types
	enum LoopType
	{
		loopForward		= 0,
		loopBidi		= 1,
		loopBackward	= 2,
	};

	uint32le identifier;
	uint32le loopType;		// See LoopType
	uint32le loopStart;		// Loop start in samples
	uint32le loopEnd;		// Loop end in samples
	uint32le fraction;
	uint32le playCount;		// Loop Count, 0 = infinite

	// Apply WAV loop information to a mod sample.
	void ApplyToSample(SmpLength &start, SmpLength &end, SmpLength sampleLength, SampleFlags &flags, ChannelFlags enableFlag, ChannelFlags bidiFlag, bool mptLoopFix) const;

	// Convert internal loop information into a WAV loop.
	void ConvertToWAV(SmpLength start, SmpLength end, bool bidi);
};

MPT_BINARY_STRUCT(WAVSampleLoop, 24)


// Instrument information chunk
struct WAVInstrumentChunk
{
	uint8 unshiftedNote;	// Root key of sample, 0...127
	int8  finetune;			// Finetune of root key in cents
	int8  gain;				// in dB
	uint8 lowNote;			// Note range, 0...127
	uint8 highNote;
	uint8 lowVelocity;		// Velocity range, 0...127
	uint8 highVelocity;
};

MPT_BINARY_STRUCT(WAVInstrumentChunk, 7)


// MPT-specific "xtra" chunk
struct WAVExtraChunk
{
	enum Flags
	{
		setPanning	= 0x20,
	};

	uint32le flags;
	uint16le defaultPan;
	uint16le defaultVolume;
	uint16le globalVolume;
	uint16le reserved;
	uint8le  vibratoType;
	uint8le  vibratoSweep;
	uint8le  vibratoDepth;
	uint8le  vibratoRate;

	// Set up sample information
	void ConvertToWAV(const ModSample &sample, MODTYPE modType)
	{
		if(sample.uFlags[CHN_PANNING])
		{
			flags = WAVExtraChunk::setPanning;
		} else
		{
			flags = 0;
		}

		defaultPan = sample.nPan;
		defaultVolume = sample.nVolume;
		globalVolume = sample.nGlobalVol;
		vibratoType = sample.nVibType;
		vibratoSweep = sample.nVibSweep;
		vibratoDepth = sample.nVibDepth;
		vibratoRate = sample.nVibRate;

		if((modType & MOD_TYPE_XM) && (vibratoDepth | vibratoRate))
		{
			// XM vibrato is upside down
			vibratoSweep = 255 - vibratoSweep;
		}
	}
};

MPT_BINARY_STRUCT(WAVExtraChunk, 16)


// Sample cue point structure for the "cue " chunk
struct WAVCuePoint
{
	uint32le id;			// Unique identification value
	uint32le position;		// Play order position
	uint32le riffChunkID;	// RIFF ID of corresponding data chunk
	uint32le chunkStart;	// Byte Offset of Data Chunk
	uint32le blockStart;	// Byte Offset to sample of First Channel
	uint32le offset;		// Byte Offset to sample byte of First Channel

	// Set up sample information
	void ConvertToWAV(uint32 id_, SmpLength offset_)
	{
		id = id_;
		position = offset_;
		riffChunkID = static_cast<uint32>(RIFFChunk::iddata);
		chunkStart = 0;	// we use no Wave List Chunk (wavl) as we have only one data block, so this should be 0.
		blockStart = 0;	// ditto
		offset = offset_;
	}
};

MPT_BINARY_STRUCT(WAVCuePoint, 24)


class WAVReader
{
protected:
	ChunkReader file;
	FileReader sampleData, smplChunk, instChunk, xtraChunk, wsmpChunk, cueChunk;
	ChunkReader::ChunkList<RIFFChunk> infoChunk;

	FileReader::off_t sampleLength;
	WAVFormatChunk formatInfo;
	uint16 subFormat;
	bool isDLS;
	bool mayBeCoolEdit16_8;

public:
	WAVReader(FileReader &inputFile);

	bool IsValid() const { return sampleData.IsValid(); }

	void FindMetadataChunks(ChunkReader::ChunkList<RIFFChunk> &chunks);

	// Self-explanatory getters.
	WAVFormatChunk::SampleFormats GetSampleFormat() const { return IsExtensibleFormat() ? static_cast<WAVFormatChunk::SampleFormats>(subFormat) : static_cast<WAVFormatChunk::SampleFormats>(formatInfo.format.get()); }
	uint16 GetNumChannels() const { return formatInfo.numChannels; }
	uint16 GetBitsPerSample() const { return formatInfo.bitsPerSample; }
	uint32 GetSampleRate() const { return formatInfo.sampleRate; }
	uint16 GetBlockAlign() const { return formatInfo.blockAlign; }
	FileReader GetSampleData() const { return sampleData; }
	FileReader GetWsmpChunk() const { return wsmpChunk; }
	bool IsExtensibleFormat() const { return formatInfo.format == WAVFormatChunk::fmtExtensible; }
	bool MayBeCoolEdit16_8() const { return mayBeCoolEdit16_8; }

	// Get size of a single sample point, in bytes.
	uint16 GetSampleSize() const { return ((GetNumChannels() * GetBitsPerSample()) + 7) / 8; }

	// Get sample length (in samples)
	SmpLength GetSampleLength() const { return mpt::saturate_cast<SmpLength>(sampleLength); }

	// Apply sample settings from file (loop points, MPT extra settings, ...) to a sample.
	void ApplySampleSettings(ModSample &sample, char (&sampleName)[MAX_SAMPLENAME]);
};


#ifndef MODPLUG_NO_FILESAVE

class WAVWriter
{
protected:
	// When writing to a stream: Stream pointer
	std::ostream *s;
	// When writing to memory: Memory address + length
	uint8 *memory;
	size_t memSize;

	// Cursor position
	size_t position;
	// Total number of bytes written to file / memory
	size_t totalSize;

	// Currently written chunk
	size_t chunkStartPos;
	RIFFChunk chunkHeader;

public:
	// Output to stream: Initialize with std::ostream*.
	WAVWriter(std::ostream *stream);
	// Output to clipboard: Initialize with pointer to memory and size of reserved memory.
	WAVWriter(void *mem, size_t size);

	~WAVWriter();

	// Check if anything can be written to the file.
	bool IsValid() const { return s != nullptr || memory != nullptr; }

	// Finalize the file by closing the last open chunk and updating the file header. Returns total size of file.
	size_t Finalize();
	// Begin writing a new chunk to the file.
	void StartChunk(RIFFChunk::ChunkIdentifiers id);

	// Skip some bytes... For example after writing sample data.
	void Skip(size_t numBytes) { Seek(position + numBytes); }
	// Get position in file (not counting any changes done to the file from outside this class, i.e. through GetFile())
	size_t GetPosition() const { return position; }

	// Shrink file size to current position.
	void Truncate() { totalSize = position; }

	// Write some data to the file.
	template<typename T>
	void Write(const T &data)
	{
		Write(&data, sizeof(T));
	}

	// Write a buffer to the file.
	void WriteBuffer(const char *data, size_t size)
	{
		Write(data, size);
	}

	// Write an array to the file.
	template<typename T, size_t size>
	void WriteArray(const T (&data)[size])
	{
		Write(data, sizeof(T) * size);
	}

	// Write the WAV format to the file.
	void WriteFormat(uint32 sampleRate, uint16 bitDepth, uint16 numChannels, WAVFormatChunk::SampleFormats encoding);
	// Write text tags to the file.
	void WriteMetatags(const FileTags &tags);
	// Write a sample loop information chunk to the file.
	void WriteLoopInformation(const ModSample &sample);
	// Write a sample's cue points to the file.
	void WriteCueInformation(const ModSample &sample);
	// Write MPT's sample information chunk to the file.
	void WriteExtraInformation(const ModSample &sample, MODTYPE modType, const char *sampleName = nullptr);

protected:
	void Init();
	// Seek to a position in file.
	void Seek(size_t pos);
	// End current chunk by updating the chunk header and writing a padding byte if necessary.
	void FinalizeChunk();

	// Write some data to the file.
	void Write(const void *data, size_t numBytes);

	// Write a single tag into a open idLIST chunk
	void WriteTag(RIFFChunk::ChunkIdentifiers id, const mpt::ustring &utext);
};

#endif // MODPLUG_NO_FILESAVE

OPENMPT_NAMESPACE_END
