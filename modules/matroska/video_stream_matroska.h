/**************************************************************************/
/*  video_stream_matroska.h                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/io/resource_loader.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/texture_rd.h"
#include "scene/resources/video_stream.h"
#include "scene/resources/video_stream_encoding.h"

#define EBML_HEADER_ID 0x1A45DFA3
#define EBML_VERSION_ID 0x4286
#define EBML_READ_VERSION_ID 0x42F7
#define EBML_MAX_ID_LENGTH_ID 0x42F2
#define EBML_MAX_SIZE_LENGTH_ID 0x42F3
#define EBML_DOC_TYPE_ID 0x4282
#define EBML_DOC_TYPE_VERSION_ID 0x4287
#define EBML_DOC_TYPE_READ_VERSION_ID 0x4285
#define EBML_DOC_TYPE_EXTENSION_ID 0x4281
#define EBML_DOC_TYPE_EXTENSION_NAME_ID 0x4283
#define EBML_DOC_TYPE_EXTENSION_VERSION_ID 0x4284

#define EBML_CRC32_ID 0xbf
#define EBML_VOID_ID 0xec

#define MATROSKA_SEGMENT_ID 0x18538067

#define MATROSKA_SEEK_HEAD_ID 0x114D9B74
#define MATROSKA_SEEK_ID 0x4DBB
#define MATROSKA_SEEK_TARGET_ID 0x53AB
#define MATROSKA_SEEK_POSITION_ID 0x53AC

#define MATROSKA_SEGMENT_INFO_ID 0x1549A966
#define MATROSKA_SEGMENT_INFO_UUID_ID 0x73A4
#define MATROSKA_SEGMENT_INFO_FILENAME_ID 0x7384
#define MATROSKA_SEGMENT_INFO_PREV_UUID_ID 0x3CB923
#define MATROSKA_SEGMENT_INFO_PREV_FILENAME_ID 0x3C83AB
#define MATROSKA_SEGMENT_INFO_NEXT_UUID_ID 0x3EB923
#define MATROSKA_SEGMENT_INFO_NEXT_FILENAME_ID 0x3E83BB
#define MATROSKA_SEGMENT_INFO_FAMILY_ID 0x4444
#define MATROSKA_SEGMENT_INFO_CHAPTER_TRANSLATE_ID 0x6924
// TODO: other chapter translate things
#define MATROSKA_SEGMENT_INFO_TIME_SCALE_ID 0x2AD7B1
#define MATROSKA_SEGMENT_INFO_DURATION_ID 0x4489
#define MATROSKA_SEGMENT_INFO_DATE_UTC_ID 0x4461
#define MATROSKA_SEGMENT_INFO_TITLE_ID 0x7BA9
#define MATROSKA_SEGMENT_INFO_MUXING_APP_ID 0x4D80
#define MATROSKA_SEGMENT_INFO_WRITING_APP_ID 0x5741

#define MATROSKA_TRACKS_ID 0x1654AE6B
#define MATROSKA_TRACK_ENTRY_ID 0xAE
#define MATROSKA_TRACK_NUMBER_ID 0xd7
#define MATROSKA_TRACK_UID_ID 0x73C5
#define MATROSKA_TRACK_TYPE 0x83

#define MATROSKA_TRACK_FLAG_ENABLED 0xB9
#define MATROSKA_TRACK_FLAG_DEFAULT 0x88
#define MATROSKA_TRACK_FLAG_FORCED 0x55AA
#define MATROSKA_TRACK_FLAG_HEARING_IMPAIRED 0x55AB
#define MATROSKA_TRACK_FLAG_VISUAL_IMPAIRED 0x55AC
#define MATROSKA_TRACK_FLAG_TEXT_DESCRIPTIONS 0x55AD
#define MATROSKA_TRACK_FLAG_ORIGINAL 0x55AE
#define MATROSKA_TRACK_FLAG_COMMENTARY 0x55AF
#define MATROSKA_TRACK_FLAG_LACING 0x9C

#define MATRSOKA_TRACK_DEFAULT_DURATION 0x23E383
#define MATRSOKA_TRACK_DEFAULT_DECODED_FIELD_DURATION 0x234E7A

#define MATROSKA_CHAPTERS_ID 0x1043A770
#define MATROSKA_CLUSTER_ID 0x1F43B675
#define MATROSKA_CUES_ID 0x1C53BB6B
#define MATROSKA_ATTACHEMENTS_ID 0x1941A469
#define MATROSKA_TAGS_ID 0x1254C367

class VideoStreamMatroska : public VideoStream {
	GDCLASS(VideoStreamMatroska, VideoStream);

public:
	void set_audio_track(int p_track) override;
	Ref<VideoStreamPlayback> instantiate_playback() override;
};

class VideoStreamPlaybackMatroska : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackMatroska, VideoStreamPlayback);

private:
	struct EbmlHeader {
		uint64_t version;
		uint64_t read_version;

		uint64_t max_id_length;
		uint64_t max_size_length;

		String doc_type;
		uint64_t doc_type_version;
		uint64_t doc_type_read_version;
	};

	struct EbmlHeader header;

	struct SeekHead {
		uint64_t segment_info_position = 0;
		uint64_t tracks_position = 0;
		uint64_t cues_position = 0;
		uint64_t tags_position = 0;
	};

	struct SegmentInfo {
		void *uuid;
		String filename;

		void *prev_uuid;
		String prev_filename;

		void *next_uuid;
		String next_filename;

		// multiple
		void *segment_family;

		// TODO
		void *chapter_translate;

		uint64_t time_scale;
		double duration;
		void *creation_date;

		String title;

		String muxing_app;
		String writing_app;
	};

	struct Tracks;
	struct Chapters;
	struct Clusters;
	struct Cues;
	struct Attachments;
	struct Tags;

	struct Segment {
		uint8_t *src = nullptr;

		SeekHead seek_head;
		SegmentInfo info;
	};

	Segment segment;
	Vector<uint64_t> clusters;

	Ref<VideoStreamEncoding> video_stream_encoding = nullptr;

	uint width = 0;
	uint height = 0;

	RID cluster;
	Ref<Texture2DArrayRD> rd_cluster;

	Ref<ImageTexture> image_texture;

	bool playing = false;

	uint64_t read_id(uint8_t *p_stream, uint32_t *r_read);
	uint64_t read_size(uint8_t *p_stream, uint32_t *r_read);

	int64_t read_int(uint8_t *p_stream, uint32_t *r_read);
	uint64_t read_uint(uint8_t *p_stream, uint32_t *r_read);
	double read_float(uint8_t *p_stream, uint32_t *r_read);
	String read_string(uint8_t *p_stream, uint32_t *r_read);

	Error parse_ebml_header(uint8_t *p_stream, uint32_t *r_read, EbmlHeader *r_header);

	Error parse_segment(uint8_t *p_stream, uint32_t *r_read, Segment *r_segment);
	Error parse_seek_head(uint8_t *p_stream, uint32_t *r_read, SeekHead *r_seak_head);
	Error parse_segment_info(uint8_t *p_stream, uint32_t *r_read, SegmentInfo *r_segment_info);
	Error parse_tracks(uint8_t *p_stream, uint32_t *r_read);
	Error parse_chapters(uint8_t *p_stream, uint32_t *r_read);
	Error parse_cluster(uint8_t *p_stream, uint32_t *r_read);
	Error parse_cues(uint8_t *p_stream, uint32_t *r_read);
	Error parse_attachments(uint8_t *p_stream, uint32_t *r_read);
	Error parse_tags(uint8_t *p_stream, uint32_t *r_read);

public:
	void set_file(const String &p_file);

	virtual void play() override;
	virtual void stop() override;
	virtual bool is_playing() const override;

	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	virtual double get_length() const override;

	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;

	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(double p_delta) override;

	virtual int get_channels() const override;
	virtual int get_mix_rate() const override;

	virtual void set_audio_track(int p_idx) override;

	VideoStreamPlaybackMatroska();
};

class ResourceFormatLoaderMatroska : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
};
