/**************************************************************************/
/*  simple_audio_writer.cpp                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "simple_audio_writer.h"
#include "core/string/print_string.h"
#include "core/os/os.h"
#include "movie_utils.h"

SimpleAudioWriter::SimpleAudioWriter() {
	audio_chunk_count = 0;
}

SimpleAudioWriter::~SimpleAudioWriter() {
	if (f.is_valid() && f->is_open()) {
		close();
	}
}

Error SimpleAudioWriter::open(const String &p_path, uint32_t p_sample_rate, uint32_t p_channels) {
	base_path = p_path;
	mix_rate = p_sample_rate;
	channels = p_channels;
	audio_chunk_count = 0;
	audio_chunk_sizes.clear();
	
	f = FileAccess::open(base_path, FileAccess::WRITE);
	ERR_FAIL_COND_V(f.is_null(), ERR_CANT_OPEN);
	
	const uint32_t bit_depth = 32;
	uint32_t blockalign = bit_depth / 8 * channels;
	audio_block_size = blockalign; // Bytes per sample
	
	// Write AVI header - audio stream only
	f->store_buffer((const uint8_t *)"RIFF", 4);
	f->store_32(0); // Total length (update later)
	f->store_buffer((const uint8_t *)"AVI ", 4);
	f->store_buffer((const uint8_t *)"LIST", 4);
	f->store_32(140); // hdrl size: 4 + 4 + 4 + 56 + 4 + 4 + 84 (audio stream only)
	f->store_buffer((const uint8_t *)"hdrl", 4);
	f->store_buffer((const uint8_t *)"avih", 4);
	f->store_32(56);

	f->store_32(1000000); // Microsecs per frame (not very important for audio)
	f->store_32(mix_rate * blockalign); // Max bytes per second
	f->store_32(0); // Padding Granularity
	f->store_32(16);
	f->store_32(0); // Total frames (not important for audio stream)
	f->store_32(0); // Initial frames
	f->store_32(1); // Streams (only 1 audio stream)
	f->store_32(12288); // Suggested buffer size
	f->store_32(0); // Width (0 for audio stream)
	f->store_32(0); // Height (0 for audio stream)
	for (uint32_t i = 0; i < 4; i++) {
		f->store_32(0); // Reserved
	}
	
	// Audio stream header
	f->store_buffer((const uint8_t *)"LIST", 4);
	f->store_32(84); // strl size: 4 + 4 + 4 + 48 + 4 + 4 + 16
	f->store_buffer((const uint8_t *)"strl", 4);
	f->store_buffer((const uint8_t *)"strh", 4);
	f->store_32(48);
	f->store_buffer((const uint8_t *)"auds", 4);
	f->store_32(0); // Handler
	f->store_32(0); // Flags
	f->store_16(0); // Priority
	f->store_16(0); // Language
	f->store_32(0); // Initial Frames
	f->store_32(blockalign); // Scale
	f->store_32(mix_rate * blockalign); // Rate
	f->store_32(0); // Start
	total_audio_frames_ofs = f->get_position();
	f->store_32(0); // Number of frames (to be updated later)
	f->store_32(12288); // Suggested Buffer Size
	f->store_32(0xFFFFFFFF); // Quality
	f->store_32(blockalign); // Block Align

	f->store_buffer((const uint8_t *)"strf", 4);
	f->store_32(16); // Standard format, no extra fields
	f->store_16(1); // Compression code, standard PCM
	f->store_16(channels);
	f->store_32(mix_rate); // Samples (frames) / Sec
	f->store_32(mix_rate * blockalign); // Bytes / sec
	f->store_16(blockalign); // Block align
	f->store_16(bit_depth); // Bits per sample

	// movi part starts
	f->store_buffer((const uint8_t *)"LIST", 4);
	movi_data_ofs = f->get_position();
	f->store_32(0); // Size (to be updated later)
	f->store_buffer((const uint8_t *)"movi", 4);

	if (MovieDebugUtils::is_stdout_verbose()) {
		print_line(String("SimpleAudioWriter: Starting audio recording to ") + base_path);
		print_line(String("Sample rate: ") + String::num_int64(mix_rate) + "Hz, Channels: " + String::num_int64(channels));
	}
	
	return OK;
}

Error SimpleAudioWriter::write_audio_chunk(const int32_t *p_audio_data, int p_frame_count) {
	ERR_FAIL_COND_V(f.is_null(), ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(!p_audio_data || p_frame_count <= 0, ERR_INVALID_PARAMETER);
	
	uint32_t data_size = p_frame_count * channels * 4; // 32-bit PCM

	f->store_buffer((const uint8_t *)"00wb", 4); // Stream 0, Audio (wave buffer)
	f->store_32(data_size);
	f->store_buffer((const uint8_t *)p_audio_data, data_size);
	
	// Byte alignment
	if (data_size & 1) {
		f->store_8(0);
		data_size++;
	}
	
	audio_chunk_sizes.push_back(data_size);
	audio_chunk_count++;

	return OK;
}

void SimpleAudioWriter::close() {
	if (!f.is_valid() || !f->is_open()) {
		return;
	}
	
	// Write index - only contains audio chunks
	f->store_buffer((const uint8_t *)"idx1", 4);
	f->store_32(4 * 4 * audio_chunk_count); // 16 bytes per index entry
	uint32_t ofs = 4;
	uint32_t all_data_size = 0;
	
	for (uint32_t i = 0; i < audio_chunk_count; i++) {
		f->store_buffer((const uint8_t *)"00wb", 4);
		f->store_32(16); // AVI_KEYFRAME
		f->store_32(ofs);
		f->store_32(audio_chunk_sizes[i]);

		ofs += audio_chunk_sizes[i] + 8;
		all_data_size += audio_chunk_sizes[i];
	}

	// Update size information in the file header
	uint32_t file_size = f->get_position();
	f->seek(4);
	f->store_32(file_size - 8);
	f->seek(total_audio_frames_ofs);
	// Calculate total number of audio samples
	uint32_t total_samples = 0;
	for (uint32_t i = 0; i < audio_chunk_count; i++) {
		total_samples += (audio_chunk_sizes[i] / (channels * 4));
	}
	f->store_32(total_samples);
	f->seek(movi_data_ofs);
	f->store_32(all_data_size + 4 + 16 * audio_chunk_count);

	f.unref();
	
	if (MovieDebugUtils::is_stdout_verbose()) {
		print_line(String("SimpleAudioWriter: Audio recording completed, total chunks: ") + String::num_int64(audio_chunk_count));
		print_line(String("Total samples: ") + String::num_int64(total_samples));
	}
} 