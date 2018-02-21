/*
 * SampleFormatFLAC.cpp
 * --------------------
 * Purpose: FLAC sample import.
 * Notes  :
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#ifdef MODPLUG_TRACKER
#include "../mptrack/TrackerSettings.h"
#endif //MODPLUG_TRACKER
#ifndef MODPLUG_NO_FILESAVE
#include "../common/mptFileIO.h"
#endif
#include "../common/misc_util.h"
#include "Tagging.h"
#include "Loaders.h"
#include "WAVTools.h"
#include "ChunkReader.h"
#include "modsmp_ctrl.h"
#include "../soundbase/SampleFormatConverters.h"
#include "../soundbase/SampleFormatCopy.h"
#include "../soundlib/ModSampleCopy.h"
//#include "../common/mptCRC.h"
#include "OggStream.h"
#ifdef MPT_WITH_OGG
#include <ogg/ogg.h>
#endif // MPT_WITH_OGG
#ifdef MPT_WITH_FLAC
#include <FLAC/stream_decoder.h>
#include <FLAC/stream_encoder.h>
#include <FLAC/metadata.h>
#endif // MPT_WITH_FLAC


OPENMPT_NAMESPACE_BEGIN


///////////////////////////////////////////////////////////////////////////////////////////////////
// FLAC Samples

#ifdef MPT_WITH_FLAC

struct FLACDecoder
{
	FileReader &file;
	CSoundFile &sndFile;
	SAMPLEINDEX sample;
	bool ready;

	FLACDecoder(FileReader &f, CSoundFile &sf, SAMPLEINDEX smp) : file(f), sndFile(sf), sample(smp), ready(false) { }

	static FLAC__StreamDecoderReadStatus read_cb(const FLAC__StreamDecoder *, FLAC__byte buffer[], size_t *bytes, void *client_data)
	{
		FileReader &file = static_cast<FLACDecoder *>(client_data)->file;
		if(*bytes > 0)
		{
			FileReader::off_t readBytes = *bytes;
			LimitMax(readBytes, file.BytesLeft());
			file.ReadRaw(buffer, readBytes);
			*bytes = readBytes;
			if(*bytes == 0)
				return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
			else
				return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
		} else
		{
			return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
		}
	}

	static FLAC__StreamDecoderSeekStatus seek_cb(const FLAC__StreamDecoder *, FLAC__uint64 absolute_byte_offset, void *client_data)
	{
		FileReader &file = static_cast<FLACDecoder *>(client_data)->file;
		if(!file.Seek(static_cast<FileReader::off_t>(absolute_byte_offset)))
			return FLAC__STREAM_DECODER_SEEK_STATUS_ERROR;
		else
			return FLAC__STREAM_DECODER_SEEK_STATUS_OK;
	}

	static FLAC__StreamDecoderTellStatus tell_cb(const FLAC__StreamDecoder *, FLAC__uint64 *absolute_byte_offset, void *client_data)
	{
		FileReader &file = static_cast<FLACDecoder *>(client_data)->file;
		*absolute_byte_offset = file.GetPosition();
		return FLAC__STREAM_DECODER_TELL_STATUS_OK;
	}

	static FLAC__StreamDecoderLengthStatus length_cb(const FLAC__StreamDecoder *, FLAC__uint64 *stream_length, void *client_data)
	{
		FileReader &file = static_cast<FLACDecoder *>(client_data)->file;
		*stream_length = file.GetLength();
		return FLAC__STREAM_DECODER_LENGTH_STATUS_OK;
	}

	static FLAC__bool eof_cb(const FLAC__StreamDecoder *, void *client_data)
	{
		FileReader &file = static_cast<FLACDecoder *>(client_data)->file;
		return file.NoBytesLeft();
	}

	static FLAC__StreamDecoderWriteStatus write_cb(const FLAC__StreamDecoder *decoder, const FLAC__Frame *frame, const FLAC__int32 *const buffer[], void *client_data)
	{
		FLACDecoder &client = *static_cast<FLACDecoder *>(client_data);
		ModSample &sample = client.sndFile.GetSample(client.sample);

		if(frame->header.number.sample_number >= sample.nLength || !client.ready)
		{
			// We're reading beyond the sample size already, or we aren't even ready to decode yet!
			return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
		}

		// Number of samples to be copied in this call
		const SmpLength copySamples = std::min(static_cast<SmpLength>(frame->header.blocksize), static_cast<SmpLength>(sample.nLength - frame->header.number.sample_number));
		// Number of target channels
		const uint8 modChannels = sample.GetNumChannels();
		// Offset (in samples) into target data
		const size_t offset = static_cast<size_t>(frame->header.number.sample_number) * modChannels;
		// Source size in bytes
		const size_t srcSize = frame->header.blocksize * 4;
		// Source bit depth
		const unsigned int bps = frame->header.bits_per_sample;

		int8 *sampleData8 = sample.pSample8 + offset;
		int16 *sampleData16 = sample.pSample16 + offset;

		MPT_ASSERT((bps <= 8 && sample.GetElementarySampleSize() == 1) || (bps > 8 && sample.GetElementarySampleSize() == 2));
		MPT_ASSERT(modChannels <= FLAC__stream_decoder_get_channels(decoder));
		MPT_ASSERT(bps == FLAC__stream_decoder_get_bits_per_sample(decoder));
		MPT_UNREFERENCED_PARAMETER(decoder); // decoder is unused if ASSERTs are compiled out

		// Do the sample conversion
		for(uint8 chn = 0; chn < modChannels; chn++)
		{
			if(bps <= 8)
			{
				CopySample<SC::ConversionChain<SC::ConvertShift< int8, int32,  0>, SC::DecodeIdentity<int32> > >(sampleData8  + chn, copySamples, modChannels, buffer[chn], srcSize, 1);
			} else if(bps <= 16)
			{
				CopySample<SC::ConversionChain<SC::ConvertShift<int16, int32,  0>, SC::DecodeIdentity<int32> > >(sampleData16 + chn, copySamples, modChannels, buffer[chn], srcSize, 1);
			} else if(bps <= 24)
			{
				CopySample<SC::ConversionChain<SC::ConvertShift<int16, int32,  8>, SC::DecodeIdentity<int32> > >(sampleData16 + chn, copySamples, modChannels, buffer[chn], srcSize, 1);
			} else if(bps <= 32)
			{
				CopySample<SC::ConversionChain<SC::ConvertShift<int16, int32, 16>, SC::DecodeIdentity<int32> > >(sampleData16 + chn, copySamples, modChannels, buffer[chn], srcSize, 1);
			}
		}

		return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
	}

	static void metadata_cb(const FLAC__StreamDecoder *, const FLAC__StreamMetadata *metadata, void *client_data)
	{
		FLACDecoder &client = *static_cast<FLACDecoder *>(client_data);
		if(client.sample > client.sndFile.GetNumSamples())
		{
			client.sndFile.m_nSamples = client.sample;
		}
		ModSample &sample = client.sndFile.GetSample(client.sample);

		if(metadata->type == FLAC__METADATA_TYPE_STREAMINFO && metadata->data.stream_info.total_samples != 0)
		{
			// Init sample information
			client.sndFile.DestroySampleThreadsafe(client.sample);
			strcpy(client.sndFile.m_szNames[client.sample], "");
			sample.Initialize();
			sample.uFlags.set(CHN_16BIT, metadata->data.stream_info.bits_per_sample > 8);
			sample.uFlags.set(CHN_STEREO, metadata->data.stream_info.channels > 1);
			sample.nLength = mpt::saturate_cast<SmpLength>(metadata->data.stream_info.total_samples);
			sample.nC5Speed = metadata->data.stream_info.sample_rate;
			client.ready = (sample.AllocateSample() != 0);
		} else if(metadata->type == FLAC__METADATA_TYPE_APPLICATION && !memcmp(metadata->data.application.id, "riff", 4) && client.ready)
		{
			// Try reading RIFF loop points and other sample information
			ChunkReader data(mpt::as_span(metadata->data.application.data, metadata->length));
			ChunkReader::ChunkList<RIFFChunk> chunks = data.ReadChunks<RIFFChunk>(2);

			// We're not really going to read a WAV file here because there will be only one RIFF chunk per metadata event, but we can still re-use the code for parsing RIFF metadata...
			WAVReader riffReader(data);
			riffReader.FindMetadataChunks(chunks);
			riffReader.ApplySampleSettings(sample, client.sndFile.m_szNames[client.sample]);
		} else if(metadata->type == FLAC__METADATA_TYPE_VORBIS_COMMENT && client.ready)
		{
			// Try reading Vorbis Comments for sample title, sample rate and loop points
			SmpLength loopStart = 0, loopLength = 0;
			for(FLAC__uint32 i = 0; i < metadata->data.vorbis_comment.num_comments; i++)
			{
				const char *tag = mpt::byte_cast<const char *>(metadata->data.vorbis_comment.comments[i].entry);
				const FLAC__uint32 length = metadata->data.vorbis_comment.comments[i].length;
				if(length > 6 && !mpt::CompareNoCaseAscii(tag, "TITLE=", 6))
				{
					mpt::String::Read<mpt::String::maybeNullTerminated>(client.sndFile.m_szNames[client.sample], tag + 6, length - 6);
				} else if(length > 11 && !mpt::CompareNoCaseAscii(tag, "SAMPLERATE=", 11))
				{
					uint32 sampleRate = ConvertStrTo<uint32>(tag + 11);
					if(sampleRate > 0) sample.nC5Speed = sampleRate;
				} else if(length > 10 && !mpt::CompareNoCaseAscii(tag, "LOOPSTART=", 10))
				{
					loopStart = ConvertStrTo<SmpLength>(tag + 10);
				} else if(length > 11 && !mpt::CompareNoCaseAscii(tag, "LOOPLENGTH=", 11))
				{
					loopLength = ConvertStrTo<SmpLength>(tag + 11);
				}
			}
			if(loopLength > 0)
			{
				sample.nLoopStart = loopStart;
				sample.nLoopEnd = loopStart + loopLength;
				sample.uFlags.set(CHN_LOOP);
				sample.SanitizeLoops();
			}
		}
	}

	static void error_cb(const FLAC__StreamDecoder *, FLAC__StreamDecoderErrorStatus, void *)
	{
	}
};

#endif // MPT_WITH_FLAC


bool CSoundFile::ReadFLACSample(SAMPLEINDEX sample, FileReader &file)
{
#ifdef MPT_WITH_FLAC
	file.Rewind();
	bool isOgg = false;
#ifdef MPT_WITH_OGG
	uint32 oggFlacBitstreamSerial = 0;
#endif
	// Check whether we are dealing with native FLAC, OggFlac or no FLAC at all.
	if(file.ReadMagic("fLaC"))
	{ // ok
		isOgg = false;
#ifdef MPT_WITH_OGG
	} else if(file.ReadMagic("OggS"))
	{ // use libogg to find the first OggFlac stream header
		file.Rewind();
		bool oggOK = false;
		bool needMoreData = true;
		static const long bufsize = 65536;
		std::size_t readSize = 0;
		char *buf = nullptr;
		ogg_sync_state oy;
		MemsetZero(oy);
		ogg_page og;
		MemsetZero(og);
		std::map<uint32, ogg_stream_state*> oggStreams;
		ogg_packet op;
		MemsetZero(op);
		if(ogg_sync_init(&oy) != 0)
		{
			return false;
		}
		while(needMoreData)
		{
			if(file.NoBytesLeft())
			{ // stop at EOF
				oggOK = false;
				needMoreData = false;
				break;
			}
			buf = ogg_sync_buffer(&oy, bufsize);
			if(!buf)
			{
				oggOK = false;
				needMoreData = false;
				break;
			}
			readSize = file.ReadRaw(buf, bufsize);
			if(ogg_sync_wrote(&oy, static_cast<long>(readSize)) != 0)
			{
				oggOK = false;
				needMoreData = false;
				break;
			}
			while(ogg_sync_pageout(&oy, &og) == 1)
			{
				if(!ogg_page_bos(&og))
				{ // we stop scanning when seeing the first noo-begin-of-stream page
					oggOK = false;
					needMoreData = false;
					break;
				}
				uint32 serial = ogg_page_serialno(&og);
				if(!oggStreams[serial])
				{ // previously unseen stream serial
					oggStreams[serial] = new ogg_stream_state();
					MemsetZero(*(oggStreams[serial]));
					if(ogg_stream_init(oggStreams[serial], serial) != 0)
					{
						delete oggStreams[serial];
						oggStreams.erase(serial);
						oggOK = false;
						needMoreData = false;
						break;
					}
				}
				if(ogg_stream_pagein(oggStreams[serial], &og) != 0)
				{ // invalid page
					oggOK = false;
					needMoreData = false;
					break;
				}
				if(ogg_stream_packetout(oggStreams[serial], &op) != 1)
				{ // partial or broken packet, continue with more data
					continue;
				}
				if(op.packetno != 0)
				{ // non-begin-of-stream packet.
					// This should not appear on first page for any known ogg codec,
					// but deal gracefully with badly mused streams in that regard.
					continue;
				}
				FileReader packet(mpt::as_span(op.packet, op.bytes));
				if(packet.ReadIntLE<uint8>() == 0x7f && packet.ReadMagic("FLAC"))
				{ // looks like OggFlac
					oggOK = true;
					oggFlacBitstreamSerial = serial;
					needMoreData = false;
					break;
				}
			}
		}
		while(oggStreams.size() > 0)
		{
			uint32 serial = oggStreams.begin()->first;
			ogg_stream_clear(oggStreams[serial]);
			delete oggStreams[serial];
			oggStreams.erase(serial);
		}
		ogg_sync_clear(&oy);
		if(!oggOK)
		{
			return false;
		}
		isOgg = true;
#else // !MPT_WITH_OGG
	} else if(file.CanRead(78) && file.ReadMagic("OggS"))
	{ // first OggFlac page is exactly 78 bytes long
		// only support plain OggFlac here with the FLAC logical bitstream being the first one
		uint8 oggPageVersion = file.ReadIntLE<uint8>();
		uint8 oggPageHeaderType = file.ReadIntLE<uint8>();
		uint64 oggPageGranulePosition = file.ReadIntLE<uint64>();
		uint32 oggPageBitstreamSerialNumber = file.ReadIntLE<uint32>();
		uint32 oggPageSequenceNumber = file.ReadIntLE<uint32>();
		uint32 oggPageChecksum = file.ReadIntLE<uint32>();
		uint8 oggPageSegments = file.ReadIntLE<uint8>();
		uint8 oggPageSegmentLength = file.ReadIntLE<uint8>();
		if(oggPageVersion != 0)
		{ // unknown Ogg version
			return false;
		}
		if(!(oggPageHeaderType & 0x02) || (oggPageHeaderType& 0x01))
		{ // not BOS or continuation
			return false;
		}
		if(oggPageGranulePosition != 0)
		{ // not starting position
			return false;
		}
		if(oggPageSequenceNumber != 0)
		{ // not first page
			return false;
		}
		// skip CRC check for now
		if(oggPageSegments != 1)
		{ // first OggFlac page must contain exactly 1 segment
			return false;
		}
		if(oggPageSegmentLength != 51)
		{ // segment length must be 51 bytes in OggFlac mapping
			return false;
		}
		if(file.ReadIntLE<uint8>() != 0x7f)
		{ // OggFlac mapping demands 0x7f packet type
			return false;
		}
		if(!file.ReadMagic("FLAC"))
		{ // OggFlac magic
			return false;
		}
		if(file.ReadIntLE<uint8>() != 0x01)
		{ // OggFlac major version
			return false;
		}
		// by now, we are pretty confident that we are not parsing random junk
		isOgg = true;
#endif // MPT_WITH_OGG
	} else
	{
		return false;
	}
	file.Rewind();

	FLAC__StreamDecoder *decoder = FLAC__stream_decoder_new();
	if(decoder == nullptr)
	{
		return false;
	}

#ifdef MPT_WITH_OGG
	if(isOgg)
	{
		// force flac decoding of the logical bitstream that actually is OggFlac
		if(!FLAC__stream_decoder_set_ogg_serial_number(decoder, oggFlacBitstreamSerial))
		{
			FLAC__stream_decoder_delete(decoder);
			return false;
		}
	}
#endif

	// Give me all the metadata!
	FLAC__stream_decoder_set_metadata_respond_all(decoder);

	FLACDecoder client(file, *this, sample);

	// Init decoder
	FLAC__StreamDecoderInitStatus initStatus = isOgg ?
		FLAC__stream_decoder_init_ogg_stream(decoder, FLACDecoder::read_cb, FLACDecoder::seek_cb, FLACDecoder::tell_cb, FLACDecoder::length_cb, FLACDecoder::eof_cb, FLACDecoder::write_cb, FLACDecoder::metadata_cb, FLACDecoder::error_cb, &client)
		:
		FLAC__stream_decoder_init_stream(decoder, FLACDecoder::read_cb, FLACDecoder::seek_cb, FLACDecoder::tell_cb, FLACDecoder::length_cb, FLACDecoder::eof_cb, FLACDecoder::write_cb, FLACDecoder::metadata_cb, FLACDecoder::error_cb, &client)
		;
	if(initStatus != FLAC__STREAM_DECODER_INIT_STATUS_OK)
	{
		FLAC__stream_decoder_delete(decoder);
		return false;
	}

	// Decode file
	FLAC__stream_decoder_process_until_end_of_stream(decoder);
	FLAC__stream_decoder_finish(decoder);
	FLAC__stream_decoder_delete(decoder);

	if(client.ready && Samples[sample].pSample != nullptr)
	{
		Samples[sample].Convert(MOD_TYPE_IT, GetType());
		Samples[sample].PrecomputeLoops(*this, false);
		return true;
	}
#else
	MPT_UNREFERENCED_PARAMETER(sample);
	MPT_UNREFERENCED_PARAMETER(file);
#endif // MPT_WITH_FLAC
	return false;
}


#ifdef MPT_WITH_FLAC
// Helper function for copying OpenMPT's sample data to FLAC's int32 buffer.
template<typename T>
inline static void SampleToFLAC32(FLAC__int32 *dst, const T *src, SmpLength numSamples)
{
	for(SmpLength i = 0; i < numSamples; i++)
	{
		dst[i] = src[i];
	}
};

// RAII-style helper struct for FLAC encoder
struct FLAC__StreamEncoder_RAII
{
	FLAC__StreamEncoder *encoder;
	FILE *f;

	operator FLAC__StreamEncoder *() { return encoder; }

	FLAC__StreamEncoder_RAII() : encoder(FLAC__stream_encoder_new()), f(nullptr) { }
	~FLAC__StreamEncoder_RAII()
	{
		FLAC__stream_encoder_delete(encoder);
		if(f != nullptr) fclose(f);
	}
};

#endif


#ifndef MODPLUG_NO_FILESAVE
bool CSoundFile::SaveFLACSample(SAMPLEINDEX nSample, const mpt::PathString &filename) const
{
#ifdef MPT_WITH_FLAC
	FLAC__StreamEncoder_RAII encoder;
	if(encoder == nullptr)
	{
		return false;
	}

	const ModSample &sample = Samples[nSample];
	uint32 sampleRate = sample.GetSampleRate(GetType());

	// First off, set up all the metadata...
	FLAC__StreamMetadata *metadata[] =
	{
		FLAC__metadata_object_new(FLAC__METADATA_TYPE_VORBIS_COMMENT),
		FLAC__metadata_object_new(FLAC__METADATA_TYPE_APPLICATION),	// MPT sample information
		FLAC__metadata_object_new(FLAC__METADATA_TYPE_APPLICATION),	// Loop points
		FLAC__metadata_object_new(FLAC__METADATA_TYPE_APPLICATION),	// Cue points
	};

	unsigned numBlocks = 2;
	if(metadata[0])
	{
		// Store sample name
		FLAC__StreamMetadata_VorbisComment_Entry entry;
		FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry, "TITLE", m_szNames[nSample]);
		FLAC__metadata_object_vorbiscomment_append_comment(metadata[0], entry, false);
		FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry, "ENCODER", MptVersion::GetOpenMPTVersionStr().c_str());
		FLAC__metadata_object_vorbiscomment_append_comment(metadata[0], entry, false);
		if(sampleRate > FLAC__MAX_SAMPLE_RATE)
		{
			// FLAC only supports a sample rate of up to 655350 Hz.
			// Store the real sample rate in a custom Vorbis comment.
			FLAC__metadata_object_vorbiscomment_entry_from_name_value_pair(&entry, "SAMPLERATE", mpt::fmt::val(sampleRate).c_str());
			FLAC__metadata_object_vorbiscomment_append_comment(metadata[0], entry, false);
		}
	}
	if(metadata[1])
	{
		// Write MPT sample information
		memcpy(metadata[1]->data.application.id, "riff", 4);

		struct
		{
			RIFFChunk header;
			WAVExtraChunk mptInfo;
		} chunk;

		chunk.header.id = RIFFChunk::idxtra;
		chunk.header.length = sizeof(WAVExtraChunk);

		chunk.mptInfo.ConvertToWAV(sample, GetType());

		const uint32 length = sizeof(RIFFChunk) + sizeof(WAVExtraChunk);

		FLAC__metadata_object_application_set_data(metadata[1], reinterpret_cast<FLAC__byte *>(&chunk), length, true);
	}
	if(metadata[numBlocks] && (sample.uFlags[CHN_LOOP | CHN_SUSTAINLOOP] || ModCommand::IsNote(sample.rootNote)))
	{
		// Store loop points / root note information
		memcpy(metadata[numBlocks]->data.application.id, "riff", 4);

		struct
		{
			RIFFChunk header;
			WAVSampleInfoChunk info;
			WAVSampleLoop loops[2];
		} chunk;

		chunk.header.id = RIFFChunk::idsmpl;
		chunk.header.length = sizeof(WAVSampleInfoChunk);

		chunk.info.ConvertToWAV(sample.GetSampleRate(GetType()), sample.rootNote);

		if(sample.uFlags[CHN_SUSTAINLOOP])
		{
			chunk.loops[chunk.info.numLoops++].ConvertToWAV(sample.nSustainStart, sample.nSustainEnd, sample.uFlags[CHN_PINGPONGSUSTAIN]);
			chunk.header.length += sizeof(WAVSampleLoop);
		}
		if(sample.uFlags[CHN_LOOP])
		{
			chunk.loops[chunk.info.numLoops++].ConvertToWAV(sample.nLoopStart, sample.nLoopEnd, sample.uFlags[CHN_PINGPONGLOOP]);
			chunk.header.length += sizeof(WAVSampleLoop);
		}

		const uint32 length = sizeof(RIFFChunk) + chunk.header.length;

		FLAC__metadata_object_application_set_data(metadata[numBlocks], reinterpret_cast<FLAC__byte *>(&chunk), length, true);
		numBlocks++;
	}
	if(metadata[numBlocks] && sample.HasCustomCuePoints())
	{
		// Store cue points
		memcpy(metadata[numBlocks]->data.application.id, "riff", 4);

		struct
		{
			RIFFChunk header;
			uint32le numPoints;
			WAVCuePoint cues[CountOf(sample.cues)];
		} chunk;

		chunk.header.id = RIFFChunk::idcue_;
		chunk.header.length = 4 + sizeof(chunk.cues);
		chunk.numPoints = CountOf(sample.cues);

		for(uint32 i = 0; i < CountOf(sample.cues); i++)
		{
			chunk.cues[i].ConvertToWAV(i, sample.cues[i]);
		}

		const uint32 length = sizeof(RIFFChunk) + chunk.header.length;

		FLAC__metadata_object_application_set_data(metadata[numBlocks], reinterpret_cast<FLAC__byte *>(&chunk), length, true);
		numBlocks++;
	}

	// FLAC allows a maximum sample rate of 655350 Hz.
	// If the real rate is higher, we store it in a Vorbis comment above.
	LimitMax(sampleRate, FLAC__MAX_SAMPLE_RATE);
	if(!FLAC__format_sample_rate_is_subset(sampleRate))
	{
		// FLAC only supports 10 Hz granularity for frequencies above 65535 Hz if the streamable subset is chosen.
		FLAC__stream_encoder_set_streamable_subset(encoder, false);
	}
	FLAC__stream_encoder_set_channels(encoder, sample.GetNumChannels());
	FLAC__stream_encoder_set_bits_per_sample(encoder, sample.GetElementarySampleSize() * 8);
	FLAC__stream_encoder_set_sample_rate(encoder, sampleRate);
	FLAC__stream_encoder_set_total_samples_estimate(encoder, sample.nLength);
	FLAC__stream_encoder_set_metadata(encoder, metadata, numBlocks);
#ifdef MODPLUG_TRACKER
	FLAC__stream_encoder_set_compression_level(encoder, TrackerSettings::Instance().m_FLACCompressionLevel);
#endif // MODPLUG_TRACKER

	bool result = false;
	FLAC__int32 *sampleData = nullptr;

	encoder.f = mpt_fopen(filename, "wb");
	if(encoder.f == nullptr || FLAC__stream_encoder_init_FILE(encoder, encoder.f, nullptr, nullptr) != FLAC__STREAM_ENCODER_INIT_STATUS_OK)
	{
		goto fail;
	}

	// Convert sample data to signed 32-Bit integer array.
	const SmpLength numSamples = sample.nLength * sample.GetNumChannels();
	sampleData = new (std::nothrow) FLAC__int32[numSamples];
	if(sampleData == nullptr)
	{
		goto fail;
	}

	if(sample.GetElementarySampleSize() == 1)
	{
		SampleToFLAC32(sampleData, sample.pSample8, numSamples);
	} else if(sample.GetElementarySampleSize() == 2)
	{
		SampleToFLAC32(sampleData, sample.pSample16, numSamples);
	} else
	{
		MPT_ASSERT_NOTREACHED();
	}

	// Do the actual conversion.
	FLAC__stream_encoder_process_interleaved(encoder, sampleData, sample.nLength);
	result = true;

fail:
	FLAC__stream_encoder_finish(encoder);

	delete[] sampleData;
	for(auto m : metadata)
	{
		FLAC__metadata_object_delete(m);
	}

	return result;
#else
	MPT_UNREFERENCED_PARAMETER(nSample);
	MPT_UNREFERENCED_PARAMETER(filename);
	return false;
#endif // MPT_WITH_FLAC
}
#endif // MODPLUG_NO_FILESAVE


OPENMPT_NAMESPACE_END
