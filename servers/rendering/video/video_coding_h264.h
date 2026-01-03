/**************************************************************************/
/*  video_coding_h264.h                                                   */
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

#include "core/templates/vector.h"

#include <cstdint>

enum VideoCodingH264ChromaFormatIdc {
	VIDEO_CODING_H264_CHROMA_FORMAT_IDC_MONOCHROME = 0,
	VIDEO_CODING_H264_CHROMA_FORMAT_IDC_420 = 1,
	VIDEO_CODING_H264_CHROMA_FORMAT_IDC_422 = 2,
	VIDEO_CODING_H264_CHROMA_FORMAT_IDC_444 = 3,
};

enum VideoCodingH264NalUnitType {
	VIDEO_CODING_H264_NAL_UNIT_TYPE_UNSPECIFIED = 0,
	VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE = 1,
	VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE_A = 2,
	VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE_B = 3,
	VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE_C = 4,
	VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE_IDR = 5,
	VIDEO_CODING_H264_NAL_UNIT_TYPE_SUPPLEMENTAL_ENHACEMENT_INFORMATION = 6,
	VIDEO_CODING_H264_NAL_UNIT_TYPE_SEQUENCE_PARAMETER_SET = 7,
	VIDEO_CODING_H264_NAL_UNIT_TYPE_PICTURE_PARAMETER_SET = 8,
};

enum VideoCodingH264ProfileIdc {
	VIDEO_CODING_H264_PROFILE_IDC_BASELINE = 66,
	VIDEO_CODING_H264_PROFILE_IDC_MAIN = 77,
	VIDEO_CODING_H264_PROFILE_IDC_HIGH = 100,
	VIDEO_CODING_H264_PROFILE_IDC_HIGH_PREDICTIVE = 244,
};

enum VideoCodingH264PictureLayout {
	VIDEO_CODING_H264_PICTURE_LAYOUT_PROGRESSIVE = 0,
	VIDEO_CODING_H264_PICTURE_LAYOUT_INTERLACED_INTERLEAVED = 1,
	VIDEO_CODING_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES = 2,
};

enum VideoCodingH264PocType {
	VIDEO_CODING_H264_POC_TYPE_0,
	VIDEO_CODING_H264_POC_TYPE_1,
	VIDEO_CODING_H264_POC_TYPE_2,
};

enum VideoCodingH264WeightedBipredIdc {

};

enum VideoCodingH264AspectRatioIdc {
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_UNSPECIFIED = 0,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_SQUARE = 1,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_12_11 = 2,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_10_11 = 3,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_16_11 = 4,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_40_33 = 5,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_24_11 = 6,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_20_11 = 7,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_32_11 = 8,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_80_33 = 9,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_18_11 = 10,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_15_11 = 11,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_64_33 = 12,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_160_99 = 13,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_4_3 = 14,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_3_2 = 15,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_2_1 = 16,
	VIDEO_CODING_H264_ASPECT_RATIO_IDC_EXTENDED_SAR = 255,
};

struct VideoCodingH264SequenceParameterSet {
	VideoCodingH264ProfileIdc profile_idc;

	bool constraint_set0_flag;
	bool constraint_set1_flag;
	bool constraint_set2_flag;
	bool constraint_set3_flag;
	bool constraint_set4_flag;
	bool constraint_set5_flag;

	uint32_t level_idc;
	uint8_t seq_parameter_set_id;

	VideoCodingH264ChromaFormatIdc chroma_format_idc = VIDEO_CODING_H264_CHROMA_FORMAT_IDC_420;
	bool separate_colour_plane_flag = false;
	uint8_t bit_depth_luma_minus8 = 0;
	uint8_t bit_depth_chroma_minus8 = 0;

	bool qpprime_y_zero_transform_bypass_flag = false;

	// TODO scaling lists
	bool seq_scaling_matrix_present_flag = false;
	struct {
		uint16_t scaling_list_present_mask;
		uint16_t use_default_scaling_matrix_mask;
	} scaling_lists;

	uint8_t log2_max_frame_num_minus4;

	VideoCodingH264PocType pic_order_cnt_type;

	uint8_t log2_max_pic_order_cnt_lsb_minus4;

	uint8_t delta_pic_order_always_zero_flag;
	int32_t offset_for_non_ref_pic;
	int32_t offset_for_top_to_bottom_field;
	uint8_t num_ref_frames_in_pic_order_cnt_cycle;
	Vector<int32_t> offset_for_ref_frame;

	uint8_t max_num_ref_frames;
	bool gaps_in_frame_num_value_allowed_flag;

	uint32_t pic_width_in_mbs_minus1;
	uint32_t pic_height_in_map_units_minus1;

	bool frame_mbs_only_flag;
	bool mb_adaptive_frame_field_flag = false;
	bool direct_8x8_inference_flag;

	bool frame_cropping_flag;
	uint32_t frame_crop_left_offset = 0;
	uint32_t frame_crop_right_offset = 0;
	uint32_t frame_crop_top_offset = 0;
	uint32_t frame_crop_bottom_offset = 0;

	bool vui_parameters_present_flag;
	struct {
		bool aspect_ratio_info_present_flag;
		VideoCodingH264AspectRatioIdc aspect_ratio_idc;
		uint16_t sar_width;
		uint16_t sar_height;

		bool overscan_info_present_flag;
		bool overscan_appropriate_flag;

		bool video_signal_type_present_flag;
		uint8_t video_format;
		bool video_full_range_flag;
		bool color_description_present_flag;
		uint8_t colour_primaries;
		uint8_t transfer_characteristics;
		uint8_t matrix_coefficients;

		bool chroma_loc_info_present_flag;
		uint8_t chroma_sample_loc_type_top_field;
		uint8_t chroma_sample_loc_type_bottom_field;

		bool timing_info_present_flag;
		uint32_t num_units_in_tick;
		uint32_t time_scale;
		bool fixed_frame_rate_flag;

		bool nal_hrd_parameters_present_flag;
		bool vcl_hrd_parameters_present_flag;

		bool bitstream_restriction_flag;
		uint8_t max_num_reorder_frames;
		uint8_t max_dec_frame_buffering;

		// TODO hdr parameters
	} vui;
};

struct VideoCodingH264PictureParameterSet {
	uint8_t seq_parameter_set_id;
	uint8_t pic_parameter_set_id;

	uint8_t entropy_coding_mode_flag;
	uint8_t bottom_field_pic_order_in_frame_present_flag;

	uint8_t num_ref_idx_l0_default_active_minus1;
	uint8_t num_ref_idx_l1_default_active_minus1;

	uint8_t weighted_pred_flag;
	VideoCodingH264WeightedBipredIdc weighted_bipred_idc;

	int8_t pic_init_qp_minus26;
	int8_t pic_init_qs_minus26;
	int8_t chroma_qp_index_offset;

	uint8_t deblocking_filter_control_present_flag;
	uint8_t constrained_intra_pred_flag;
	uint8_t redundant_pic_cnt_present_flag;

	uint8_t transform_8x8_mode_flag;

	// TODO
	uint8_t pic_scaling_matrix_present_flag;
	struct {
	} scaling_lists;

	int8_t second_chroma_qp_index_offset;
};
