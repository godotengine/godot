/**************************************************************************/
/*  video_stream_av1.h                                                    */
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

#include "scene/resources/video_stream_encoding.h"
#include "servers/rendering/rendering_device.h"

class VideoStreamAV1 : public VideoStreamEncoding {
	GDCLASS(VideoStreamAV1, VideoStreamEncoding);

private:
	const uint8_t *src = nullptr;
	uint8_t shift = 0;

	VideoCodingAV1SequenceHeader av1_sequence_header;

	// Decoder State.
	uint8_t ref_valid[VIDEO_CODING_AV1_NUM_REF_FRAMES];
	uint8_t ref_frame_id[VIDEO_CODING_AV1_NUM_REF_FRAMES];
	VideoCodingAV1FrameType ref_frame_types[VIDEO_CODING_AV1_NUM_REF_FRAMES];
	uint8_t ref_order_hint[VIDEO_CODING_AV1_NUM_REF_FRAMES];

	uint8_t saved_order_hints[VIDEO_CODING_AV1_NUM_REF_FRAMES][VIDEO_CODING_AV1_REFS_PER_FRAME];

	int8_t loop_filter_ref_deltas[VIDEO_CODING_AV1_NUM_REF_FRAMES][VIDEO_CODING_AV1_TOTAL_REFS_PER_FRAME];
	int8_t loop_filter_mode_deltas[VIDEO_CODING_AV1_NUM_REF_FRAMES][2];

	uint8_t segmentation_feature_enabled[VIDEO_CODING_AV1_NUM_REF_FRAMES][VIDEO_CODING_AV1_MAX_SEGMENTS];
	uint8_t segmentation_feature_data[VIDEO_CODING_AV1_NUM_REF_FRAMES][VIDEO_CODING_AV1_MAX_SEGMENTS][VIDEO_CODING_AV1_SEG_LVL_MAX];

	RenderingDevice *local_device;

	VideoProfile video_profile;
	RID video_session;

	RID texture_sampler;

	uint64_t read_bits(uint8_t p_bits);
	uint64_t read_uvlc();
	uint64_t read_leb128();
	int64_t read_su(uint8_t p_bits);

	int64_t read_delta_q();

	uint8_t tile_log2(uint32_t p_blk, uint32_t p_target);

	int8_t get_relative_distance(uint8_t a, uint8_t b);

	bool parse_open_bitstream_unit();
	VideoCodingAV1SequenceHeader parse_sequence_header();
	VideoDecodeAV1Frame parse_frame_header();
	VideoDecodeAV1Frame parse_frame(uint32_t p_size);

	void parse_tile_info(uint32_t mi_cols, uint32_t mi_rows, VideoCodingAV1TileInfo *r_tile_info);
	void parse_quantization(VideoCodingAV1Quantization *r_quantization);
	void parse_segmentation(uint8_t p_primary_ref_frame, VideoCodingAV1Segmentation *r_segmentation);
	void parse_loop_filter(bool p_coded_lossless, bool p_allow_intrabc, VideoCodingAV1LoopFilter *r_loop_filter);
	void parse_cdef(bool p_coded_lossless, bool p_allow_intrabc, VideoCodingAV1CDEF *r_cdef);
	void parse_loop_restoration(bool p_all_lossless, bool p_allow_intrabc, VideoCodingAV1LoopRestoration *r_loop_restoration);
	void parse_global_motion(bool p_is_intra, bool p_allow_high_precision_mv, VideoCodingAV1GlobalMotion *r_global_motion);
	void parse_film_grain(bool p_show_frame, bool p_showable_frame, VideoCodingAV1FrameType p_frame_type, VideoCodingAV1FilmGrain *r_film_grain);

public:
	virtual Error parse_container_metadata(const uint8_t *p_stream, uint64_t p_size) final override;
	virtual Error parse_container_block(const uint8_t *p_stream, size_t p_size, Vector<size_t> *r_offsets, Vector<size_t> *r_sizes) final override;

	virtual void set_rendering_device(RenderingDevice *p_coding_device) final override;
	virtual RID create_video_session(RD::VideoSessionInfo p_session_template) final override;
	virtual RID create_texture_sampler(RD::SamplerState p_sampler_template) final override;
	virtual RID create_texture(RD::TextureFormat p_texture_template) final override;
	virtual void decode_frame(Span<uint8_t> p_frame_data, RID p_dst_texture) final override;

	VideoStreamAV1();
};
