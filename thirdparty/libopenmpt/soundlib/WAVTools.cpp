/*
 * WAVTools.cpp
 * ------------
 * Purpose: Definition of WAV file structures and helper functions
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "WAVTools.h"
#include "Tagging.h"
#ifndef MODPLUG_NO_FILESAVE
#include "../common/mptFileIO.h"
#endif


OPENMPT_NAMESPACE_BEGIN


///////////////////////////////////////////////////////////
// WAV Reading


WAVReader::WAVReader(FileReader &inputFile) : file(inputFile)
{
	file.Rewind();

	RIFFHeader fileHeader;
	isDLS = false;
	subFormat = 0;
	mayBeCoolEdit16_8 = false;
	if(!file.ReadStruct(fileHeader)
		|| (fileHeader.magic != RIFFHeader::idRIFF && fileHeader.magic != RIFFHeader::idLIST)
		|| (fileHeader.type != RIFFHeader::idWAVE && fileHeader.type != RIFFHeader::idwave))
	{
		return;
	}

	isDLS = (fileHeader.magic == RIFFHeader::idLIST);

	auto chunks = file.ReadChunks<RIFFChunk>(2);

	if(chunks.size() >= 4
		&& chunks[1].GetHeader().GetID() == RIFFChunk::iddata
		&& chunks[1].GetHeader().GetLength() % 2u != 0
		&& chunks[2].GetHeader().GetLength() == 0
		&& chunks[3].GetHeader().GetID() == RIFFChunk::id____)
	{
		// Houston, we have a problem: Old versions of (Open)MPT didn't write RIFF padding bytes. -_-
		// Luckily, the only RIFF chunk with an odd size those versions would ever write would be the "data" chunk
		// (which contains the sample data), and its size is only odd iff the sample has an odd length and is in
		// 8-Bit mono format. In all other cases, the sample size (and thus the chunk size) is even.

		// And we're even more lucky: The versions of (Open)MPT in question will always write a relatively small
		// (smaller than 256 bytes) "smpl" chunk after the "data" chunk. This means that after an unpadded sample,
		// we will always read "mpl?" (? being the length of the "smpl" chunk) as the next chunk magic. The first two
		// 32-Bit members of the "smpl" chunk are always zero in our case, so we are going to read a chunk length of 0
		// next and the next chunk magic, which will always consist of four zero bytes. Hooray! We just checked for those
		// four zero bytes and can be pretty confident that we should not have applied padding.
		file.Seek(sizeof(RIFFHeader));
		chunks = file.ReadChunks<RIFFChunk>(1);
	}

	// Read format chunk
	FileReader formatChunk = chunks.GetChunk(RIFFChunk::idfmt_);
	if(!formatChunk.ReadStruct(formatInfo))
	{
		return;
	}
	if(formatInfo.format == WAVFormatChunk::fmtPCM && formatChunk.BytesLeft() == 4)
	{
		uint16 size = formatChunk.ReadIntLE<uint16>();
		uint16 value = formatChunk.ReadIntLE<uint16>();
		if(size == 2 && value == 1)
		{
			// May be Cool Edit 16.8 format.
			// See SampleFormats.cpp for details.
			mayBeCoolEdit16_8 = true;
		}
	} else if(formatInfo.format == WAVFormatChunk::fmtExtensible)
	{
		WAVFormatChunkExtension extFormat;
		if(!formatChunk.ReadStruct(extFormat))
		{
			return;
		}
		subFormat = static_cast<uint16>(mpt::UUID(extFormat.subFormat).GetData1());
	}

	// Read sample data
	sampleData = chunks.GetChunk(RIFFChunk::iddata);

	if(!sampleData.IsValid())
	{
		// The old IMA ADPCM loader code looked for the "pcm " chunk instead of the "data" chunk...
		// Dunno why (Windows XP's audio recorder saves IMA ADPCM files with a "data" chunk), but we will just look for both.
		sampleData = chunks.GetChunk(RIFFChunk::idpcm_);
	}

	// "fact" chunk should contain sample length of compressed samples.
	sampleLength = chunks.GetChunk(RIFFChunk::idfact).ReadUint32LE();

	if((formatInfo.format != WAVFormatChunk::fmtIMA_ADPCM || sampleLength == 0) && GetSampleSize() != 0)
	{
		if((GetBlockAlign() == 0) || (GetBlockAlign() / GetNumChannels() >= 2 * GetSampleSize()))
		{
			// Some samples have an incorrect blockAlign / sample size set (e.g. it's 8 in SQUARE.WAV while it should be 1), so let's better not always trust this value.
			// The idea here is, if block align is off by twice or more, it is unlikely to be describing sample padding inside the block.
			// Ignore it in this case and calculate the length based on the single sample size and number of channels instead.
			sampleLength = sampleData.GetLength() / GetSampleSize();
		} else
		{
			// Correct case (so that 20bit WAVEFORMATEX files work).
			sampleLength = sampleData.GetLength() / GetBlockAlign();
		}
	}

	// Check for loop points, texts, etc...
	FindMetadataChunks(chunks);

	// DLS bank chunk
	wsmpChunk = chunks.GetChunk(RIFFChunk::idwsmp);
}


void WAVReader::FindMetadataChunks(ChunkReader::ChunkList<RIFFChunk> &chunks)
{
	// Read sample loop points and other sampler information
	smplChunk = chunks.GetChunk(RIFFChunk::idsmpl);
	instChunk = chunks.GetChunk(RIFFChunk::idinst);

	// Read sample cues
	cueChunk = chunks.GetChunk(RIFFChunk::idcue_);

	// Read text chunks
	ChunkReader listChunk = chunks.GetChunk(RIFFChunk::idLIST);
	if(listChunk.ReadMagic("INFO"))
	{
		infoChunk = listChunk.ReadChunks<RIFFChunk>(2);
	}

	// Read MPT sample information
	xtraChunk = chunks.GetChunk(RIFFChunk::idxtra);
}


void WAVReader::ApplySampleSettings(ModSample &sample, char (&sampleName)[MAX_SAMPLENAME])
{
	// Read sample name
	FileReader textChunk = infoChunk.GetChunk(RIFFChunk::idINAM);
	if(textChunk.IsValid())
	{
		textChunk.ReadString<mpt::String::nullTerminated>(sampleName, textChunk.GetLength());
	}
	if(isDLS)
	{
		// DLS sample -> sample filename
		mpt::String::Copy(sample.filename, sampleName);
	}

	// Read software name
	const bool isOldMPT = infoChunk.GetChunk(RIFFChunk::idISFT).ReadMagic("Modplug Tracker");
	
	// Convert loops
	WAVSampleInfoChunk sampleInfo;
	smplChunk.Rewind();
	if(smplChunk.ReadStruct(sampleInfo))
	{
		WAVSampleLoop loopData;
		if(sampleInfo.numLoops > 1 && smplChunk.ReadStruct(loopData))
		{
			// First loop: Sustain loop
			loopData.ApplyToSample(sample.nSustainStart, sample.nSustainEnd, sample.nLength, sample.uFlags, CHN_SUSTAINLOOP, CHN_PINGPONGSUSTAIN, isOldMPT);
		}
		// First loop (if only one loop is present) or second loop (if more than one loop is present): Normal sample loop
		if(smplChunk.ReadStruct(loopData))
		{
			loopData.ApplyToSample(sample.nLoopStart, sample.nLoopEnd, sample.nLength, sample.uFlags, CHN_LOOP, CHN_PINGPONGLOOP, isOldMPT);
		}
		//sample.Transpose((60 - sampleInfo.baseNote) / 12.0);
		sample.rootNote = static_cast<uint8>(sampleInfo.baseNote);
		if(sample.rootNote < 128)
			sample.rootNote += NOTE_MIN;
		else
			sample.rootNote = NOTE_NONE;
		sample.SanitizeLoops();
	}

	if(sample.rootNote == NOTE_NONE && instChunk.LengthIsAtLeast(sizeof(WAVInstrumentChunk)))
	{
		WAVInstrumentChunk inst;
		instChunk.Rewind();
		if(instChunk.ReadStruct(inst))
		{
			sample.rootNote = inst.unshiftedNote;
			if(sample.rootNote < 128)
				sample.rootNote += NOTE_MIN;
			else
				sample.rootNote = NOTE_NONE;
		}
	}

	// Read cue points
	if(cueChunk.IsValid())
	{
		uint32 numPoints = cueChunk.ReadUint32LE();
		LimitMax(numPoints, mpt::saturate_cast<uint32>(MPT_ARRAY_COUNT(sample.cues)));
		for(uint32 i = 0; i < numPoints; i++)
		{
			WAVCuePoint cuePoint;
			cueChunk.ReadStruct(cuePoint);
			sample.cues[i] = cuePoint.position;
		}
	}

	// Read MPT extra info
	WAVExtraChunk mptInfo;
	xtraChunk.Rewind();
	if(xtraChunk.ReadStruct(mptInfo))
	{
		if(mptInfo.flags & WAVExtraChunk::setPanning) sample.uFlags.set(CHN_PANNING);

		sample.nPan = std::min<uint16>(mptInfo.defaultPan, 256);
		sample.nVolume = std::min<uint16>(mptInfo.defaultVolume, 256);
		sample.nGlobalVol = std::min<uint16>(mptInfo.globalVolume, 64);
		sample.nVibType = mptInfo.vibratoType;
		sample.nVibSweep = mptInfo.vibratoSweep;
		sample.nVibDepth = mptInfo.vibratoDepth;
		sample.nVibRate = mptInfo.vibratoRate;

		if(xtraChunk.CanRead(MAX_SAMPLENAME))
		{
			// Name present (clipboard only)
			xtraChunk.ReadString<mpt::String::nullTerminated>(sampleName, MAX_SAMPLENAME);
			xtraChunk.ReadString<mpt::String::nullTerminated>(sample.filename, xtraChunk.BytesLeft());
		}
	}
}


// Apply WAV loop information to a mod sample.
void WAVSampleLoop::ApplyToSample(SmpLength &start, SmpLength &end, SmpLength sampleLength, SampleFlags &flags, ChannelFlags enableFlag, ChannelFlags bidiFlag, bool mptLoopFix) const
{
	if(loopEnd == 0)
	{
		// Some WAV files seem to have loops going from 0 to 0... We should ignore those.
		return;
	}
	start = std::min(static_cast<SmpLength>(loopStart), sampleLength);
	end = Clamp(static_cast<SmpLength>(loopEnd), start, sampleLength);
	if(!mptLoopFix && end < sampleLength)
	{
		// RIFF loop end points are inclusive - old versions of MPT didn't consider this.
		end++;
	}

	flags.set(enableFlag);
	if(loopType == loopBidi)
	{
		flags.set(bidiFlag);
	}
}


// Convert internal loop information into a WAV loop.
void WAVSampleLoop::ConvertToWAV(SmpLength start, SmpLength end, bool bidi)
{
	identifier = 0;
	loopType = bidi ? loopBidi : loopForward;
	loopStart = mpt::saturate_cast<uint32>(start);
	// Loop ends are *inclusive* in the RIFF standard, while they're *exclusive* in OpenMPT.
	if(end > start)
	{
		loopEnd = mpt::saturate_cast<uint32>(end - 1);
	} else
	{
		loopEnd = loopStart;
	}
	fraction = 0;
	playCount = 0;
}


#ifndef MODPLUG_NO_FILESAVE

///////////////////////////////////////////////////////////
// WAV Writing


// Output to stream: Initialize with std::ostream*.
WAVWriter::WAVWriter(std::ostream *stream) : s(nullptr), memory(nullptr), memSize(0)
{
	s = stream;
	Init();
}


// Output to clipboard: Initialize with pointer to memory and size of reserved memory.
WAVWriter::WAVWriter(void *mem, size_t size) : s(nullptr), memory(static_cast<uint8 *>(mem)), memSize(size)
{
	Init();
}


WAVWriter::~WAVWriter()
{
	Finalize();
}


// Reset all file variables.
void WAVWriter::Init()
{
	chunkStartPos = 0;
	position = 0;
	totalSize = 0;

	// Skip file header for now
	Seek(sizeof(RIFFHeader));
}


// Finalize the file by closing the last open chunk and updating the file header. Returns total size of file.
size_t WAVWriter::Finalize()
{
	FinalizeChunk();

	RIFFHeader fileHeader;
	fileHeader.magic = RIFFHeader::idRIFF;
	fileHeader.length = static_cast<uint32>(totalSize - 8);
	fileHeader.type = RIFFHeader::idWAVE;

	Seek(0);
	Write(fileHeader);

	s = nullptr;
	memory = nullptr;

	return totalSize;
}


// Write a new chunk header to the file.
void WAVWriter::StartChunk(RIFFChunk::ChunkIdentifiers id)
{
	FinalizeChunk();

	chunkStartPos = position;
	chunkHeader.id = id;
	Skip(sizeof(chunkHeader));
}


// End current chunk by updating the chunk header and writing a padding byte if necessary.
void WAVWriter::FinalizeChunk()
{
	if(chunkStartPos != 0)
	{
		const size_t chunkSize = position - (chunkStartPos + sizeof(RIFFChunk));
		chunkHeader.length = chunkSize;

		size_t curPos = position;
		Seek(chunkStartPos);
		Write(chunkHeader);

		Seek(curPos);
		if((chunkSize % 2u) != 0)
		{
			// Write padding
			uint8 padding = 0;
			Write(padding);
		}

		chunkStartPos = 0;
	}
}


// Seek to a position in file.
void WAVWriter::Seek(size_t pos)
{
	position = pos;
	totalSize = std::max(totalSize, position);

	if(s != nullptr)
	{
		s->seekp(position);
	}
}


// Write some data to the file.
void WAVWriter::Write(const void *data, size_t numBytes)
{
	if(s != nullptr)
	{
		s->write(static_cast<const char*>(data), numBytes);
	} else if(memory != nullptr)
	{
		if(position <= memSize && numBytes <= memSize - position)
		{
			memcpy(memory + position, data, numBytes);
		} else
		{
			// Should never happen - did we calculate a wrong memory size?
			MPT_ASSERT_NOTREACHED();
		}
	}
	position += numBytes;
	totalSize = std::max(totalSize, position);
}


// Write the WAV format to the file.
void WAVWriter::WriteFormat(uint32 sampleRate, uint16 bitDepth, uint16 numChannels, WAVFormatChunk::SampleFormats encoding)
{
	StartChunk(RIFFChunk::idfmt_);
	WAVFormatChunk wavFormat;

	bool extensible = (numChannels > 2);

	wavFormat.format = static_cast<uint16>(extensible ? WAVFormatChunk::fmtExtensible : encoding);
	wavFormat.numChannels = numChannels;
	wavFormat.sampleRate = sampleRate;
	wavFormat.blockAlign = (bitDepth * numChannels + 7) / 8;
	wavFormat.byteRate = wavFormat.sampleRate * wavFormat.blockAlign;
	wavFormat.bitsPerSample = bitDepth;

	Write(wavFormat);

	if(extensible)
	{
		WAVFormatChunkExtension extFormat;
		extFormat.size = sizeof(WAVFormatChunkExtension) - sizeof(uint16);
		extFormat.validBitsPerSample = bitDepth;
		switch(numChannels)
		{
		case 1:
			extFormat.channelMask = 0x0004;	// FRONT_CENTER
			break;
		case 2:
			extFormat.channelMask = 0x0003;	// FRONT_LEFT | FRONT_RIGHT
			break;
		case 3:
			extFormat.channelMask = 0x0103;	// FRONT_LEFT | FRONT_RIGHT | BACK_CENTER
			break;
		case 4:
			extFormat.channelMask = 0x0033;	// FRONT_LEFT | FRONT_RIGHT | BACK_LEFT | BACK_RIGHT
			break;
		default:
			extFormat.channelMask = 0;
			break;
		}
		extFormat.subFormat = mpt::UUID(static_cast<uint16>(encoding), 0x0000, 0x0010, 0x800000AA00389B71ull);
		Write(extFormat);
	}
}


// Write text tags to the file.
void WAVWriter::WriteMetatags(const FileTags &tags)
{
	StartChunk(RIFFChunk::idLIST);
	const char info[] = { 'I', 'N', 'F', 'O' };
	WriteArray(info);

	WriteTag(RIFFChunk::idINAM, tags.title);
	WriteTag(RIFFChunk::idIART, tags.artist);
	WriteTag(RIFFChunk::idIPRD, tags.album);
	WriteTag(RIFFChunk::idICRD, tags.year);
	WriteTag(RIFFChunk::idICMT, tags.comments);
	WriteTag(RIFFChunk::idIGNR, tags.genre);
	WriteTag(RIFFChunk::idTURL, tags.url);
	WriteTag(RIFFChunk::idISFT, tags.encoder);
	//WriteTag(RIFFChunk::      , tags.bpm);
	WriteTag(RIFFChunk::idTRCK, tags.trackno);
}


// Write a single tag into a open idLIST chunk
void WAVWriter::WriteTag(RIFFChunk::ChunkIdentifiers id, const mpt::ustring &utext)
{
	std::string text = mpt::ToCharset(mpt::CharsetWindows1252, utext);
	if(!text.empty())
	{
		const size_t length = text.length() + 1;

		RIFFChunk chunk;
		chunk.id = static_cast<uint32>(id);
		chunk.length = length;
		Write(chunk);
		Write(text.c_str(), length);

		if((length % 2u) != 0)
		{
			uint8 padding = 0;
			Write(padding);
		}
	}
}


// Write a sample loop information chunk to the file.
void WAVWriter::WriteLoopInformation(const ModSample &sample)
{
	if(!sample.uFlags[CHN_LOOP | CHN_SUSTAINLOOP] && !ModCommand::IsNote(sample.rootNote))
	{
		return;
	}

	StartChunk(RIFFChunk::idsmpl);
	WAVSampleInfoChunk info;

	uint32 sampleRate = sample.nC5Speed;
	if(sampleRate == 0)
	{
		sampleRate = ModSample::TransposeToFrequency(sample.RelativeTone, sample.nFineTune);
	}

	info.ConvertToWAV(sampleRate, sample.rootNote);

	// Set up loops
	WAVSampleLoop loops[2];
	if(sample.uFlags[CHN_SUSTAINLOOP])
	{
		loops[info.numLoops++].ConvertToWAV(sample.nSustainStart, sample.nSustainEnd, sample.uFlags[CHN_PINGPONGSUSTAIN]);
	}
	if(sample.uFlags[CHN_LOOP])
	{
		loops[info.numLoops++].ConvertToWAV(sample.nLoopStart, sample.nLoopEnd, sample.uFlags[CHN_PINGPONGLOOP]);
	} else if(sample.uFlags[CHN_SUSTAINLOOP])
	{
		// Since there are no "loop types" to distinguish between sustain and normal loops, OpenMPT assumes
		// that the first loop is a sustain loop if there are two loops. If we only want a sustain loop,
		// we will have to write a second bogus loop.
		loops[info.numLoops++].ConvertToWAV(0, 0, false);
	}

	Write(info);
	for(uint32 i = 0; i < info.numLoops; i++)
	{
		Write(loops[i]);
	}
}


// Write a sample's cue points to the file.
void WAVWriter::WriteCueInformation(const ModSample &sample)
{
	StartChunk(RIFFChunk::idcue_);
	{
		const uint32 numPoints = SwapBytesLE(static_cast<uint32>(CountOf(sample.cues)));
		Write(numPoints);
	}
	for(uint32 i = 0; i < CountOf(sample.cues); i++)
	{
		WAVCuePoint cuePoint;
		cuePoint.ConvertToWAV(i, sample.cues[i]);
		Write(cuePoint);
	}
}


// Write MPT's sample information chunk to the file.
void WAVWriter::WriteExtraInformation(const ModSample &sample, MODTYPE modType, const char *sampleName)
{
	StartChunk(RIFFChunk::idxtra);
	WAVExtraChunk mptInfo;

	mptInfo.ConvertToWAV(sample, modType);
	Write(mptInfo);

	if(sampleName != nullptr)
	{
		// Write sample name (clipboard only)
		char name[MAX_SAMPLENAME];
		mpt::String::Write<mpt::String::nullTerminated>(name, sampleName, MAX_SAMPLENAME);
		WriteArray(name);

		char filename[MAX_SAMPLEFILENAME];
		mpt::String::Write<mpt::String::nullTerminated>(filename, sample.filename);
		WriteArray(filename);
	}
}

#endif // MODPLUG_NO_FILESAVE


OPENMPT_NAMESPACE_END
