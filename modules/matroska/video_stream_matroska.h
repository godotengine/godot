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
#include "core/templates/hash_map.h"
#include "matroska.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/video_stream.h"
#include "scene/resources/video_stream_encoding.h"
#include "servers/rendering/rendering_device.h"

class VideoStreamMatroska : public VideoStream {
	GDCLASS(VideoStreamMatroska, VideoStream);

public:
	void set_audio_track(int p_track) override;
	Ref<VideoStreamPlayback> instantiate_playback() override;
};

class VideoStreamPlaybackMatroska : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackMatroska, VideoStreamPlayback);

private:
	EbmlHeader header;
	Segment segment;

	String path;
	const uint8_t *origin = nullptr;
	const uint8_t *src = nullptr;

	Ref<VideoStreamEncoding> video_stream_encoding = nullptr;

	uint32_t width = 0;
	uint32_t height = 0;

	RenderingDevice *local_device;

	RID video_session;

	RID yuv_sampler;
	RID yuv_shader;
	RID yuv_pipeline;
	Vector<RID> dst_yuv_pool;
	Vector<RID> dst_rgba_pool;

	size_t counter;
	Ref<ImageTexture> image_texture;

	bool playing = false;

	void _skip_id(uint64_t p_id);

	uint64_t read_id();
	uint64_t read_size();

	int64_t read_int();
	uint64_t read_uint();
	double read_float();
	String read_string();

	Error parse_ebml_header(EbmlHeader *r_header);
	Error parse_segment(Segment *r_segment);

	Error parse_seek_head(SeekHead *r_seak_head);
	Error parse_segment_info(SegmentInfo *r_segment_info);
	Error parse_tracks(Vector<Track> r_tracks);
	Error parse_chapters();
	Error parse_cluster(Cluster *r_cluster);
	Error parse_cues();
	Error parse_attachments();
	Error parse_tags();

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
	~VideoStreamPlaybackMatroska();
};

class ResourceFormatLoaderMatroska : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
};
