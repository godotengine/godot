/**************************************************************************/
/*  video_coding_av1.h                                                    */
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

#define VIDEO_CODING_AV1_NUM_REF_FRAMES 8
#define VIDEO_CODING_AV1_REFS_PER_FRAME 7
#define VIDEO_CODING_AV1_TOTAL_REFS_PER_FRAME 8
#define VIDEO_CODING_AV1_MAX_TILE_COLS 64
#define VIDEO_CODING_AV1_MAX_TILE_ROWS 64
#define VIDEO_CODING_AV1_MAX_SEGMENTS 8
#define VIDEO_CODING_AV1_SEG_LVL_MAX 8
#define VIDEO_CODING_AV1_PRIMARY_REF_NONE 7
#define VIDEO_CODING_AV1_SELECT_INTEGER_MV 2
#define VIDEO_CODING_AV1_SELECT_SCREEN_CONTENT_TOOLS 2
#define VIDEO_CODING_AV1_SKIP_MODE_FRAMES 2
#define VIDEO_CODING_AV1_MAX_LOOP_FILTER_STRENGTHS 4
#define VIDEO_CODING_AV1_LOOP_FILTER_ADJUSTMENTS 2
#define VIDEO_CODING_AV1_MAX_CDEF_FILTER_STRENGTHS 8
#define VIDEO_CODING_AV1_MAX_NUM_PLANES 3
#define VIDEO_CODING_AV1_GLOBAL_MOTION_PARAMS 6
#define VIDEO_CODING_AV1_MAX_NUM_Y_POINTS 14
#define VIDEO_CODING_AV1_MAX_NUM_CB_POINTS 10
#define VIDEO_CODING_AV1_MAX_NUM_CR_POINTS 10
#define VIDEO_CODING_AV1_MAX_NUM_POS_LUMA 24
#define VIDEO_CODING_AV1_MAX_NUM_POS_CHROMA 25

#define VIDEO_CODING_AV1_MAX_LOOP_FILTER 63

static constexpr uint8_t video_coding_av1_segmentation_bits[VIDEO_CODING_AV1_SEG_LVL_MAX] = {
	8,
	6,
	6,
	6,
	6,
	3,
	0,
	0
};

static constexpr bool video_coding_av1_segmentation_signed[VIDEO_CODING_AV1_SEG_LVL_MAX] = {
	true,
	true,
	true,
	true,
	true,
	false,
	false,
	false,
};

static constexpr uint8_t video_coding_av1_segmentation_feature_max[VIDEO_CODING_AV1_SEG_LVL_MAX] = {
	255,
	VIDEO_CODING_AV1_MAX_LOOP_FILTER,
	VIDEO_CODING_AV1_MAX_LOOP_FILTER,
	VIDEO_CODING_AV1_MAX_LOOP_FILTER,
	VIDEO_CODING_AV1_MAX_LOOP_FILTER,
	7,
	0,
	0
};

enum VideoCodingAV1ObuType {
	VIDEO_CODING_AV1_OBU_TYPE_SEQUENCE_HEADER = 1,
	VIDEO_CODING_AV1_OBU_TYPE_TEMPORAL_DELIMITER = 2,
	VIDEO_CODING_AV1_OBU_TYPE_FRAME_HEADER = 3,
	VIDEO_CODING_AV1_OBU_TYPE_TILE_GROUP = 4,
	VIDEO_CODING_AV1_OBU_TYPE_METADATA = 5,
	VIDEO_CODING_AV1_OBU_TYPE_FRAME = 6,
	VIDEO_CODING_AV1_OBU_TYPE_REDUNDANT_FRAME_HEADER = 7,
	VIDEO_CODING_AV1_OBU_TYPE_TILE_LIST = 8,
	VIDEO_CODING_AV1_OBU_TYPE_TILE_PADDING = 15,
};

enum VideoCodingAV1Profile {
	VIDEO_CODING_AV1_PROFILE_MAIN = 0,
	VIDEO_CODING_AV1_PROFILE_HIGH = 1,
	VIDEO_CODING_AV1_PROFILE_PROFESSIONAL = 2,
};

enum VideoCodingAV1Level {
	VIDEO_CODING_AV1_LEVEL_2_0 = 0,
	VIDEO_CODING_AV1_LEVEL_2_1 = 1,
	VIDEO_CODING_AV1_LEVEL_2_2 = 2,
	VIDEO_CODING_AV1_LEVEL_2_3 = 3,
	VIDEO_CODING_AV1_LEVEL_3_0 = 4,
	VIDEO_CODING_AV1_LEVEL_3_1 = 5,
	VIDEO_CODING_AV1_LEVEL_3_2 = 6,
	VIDEO_CODING_AV1_LEVEL_3_3 = 7,
	VIDEO_CODING_AV1_LEVEL_4_0 = 8,
	VIDEO_CODING_AV1_LEVEL_4_1 = 9,
	VIDEO_CODING_AV1_LEVEL_4_2 = 10,
	VIDEO_CODING_AV1_LEVEL_4_3 = 11,
	VIDEO_CODING_AV1_LEVEL_5_0 = 12,
	VIDEO_CODING_AV1_LEVEL_5_1 = 13,
	VIDEO_CODING_AV1_LEVEL_5_2 = 14,
	VIDEO_CODING_AV1_LEVEL_5_3 = 15,
	VIDEO_CODING_AV1_LEVEL_6_0 = 16,
	VIDEO_CODING_AV1_LEVEL_6_1 = 17,
	VIDEO_CODING_AV1_LEVEL_6_2 = 18,
	VIDEO_CODING_AV1_LEVEL_6_3 = 19,
	VIDEO_CODING_AV1_LEVEL_7_0 = 20,
	VIDEO_CODING_AV1_LEVEL_7_1 = 21,
	VIDEO_CODING_AV1_LEVEL_7_2 = 22,
	VIDEO_CODING_AV1_LEVEL_7_3 = 23,
};

enum VideoCodingAV1FrameType {
	VIDEO_CODING_AV1_FRAME_TYPE_KEY = 0,
	VIDEO_CODING_AV1_FRAME_TYPE_INTER = 1,
	VIDEO_CODING_AV1_FRAME_TYPE_INTRA_ONLY = 2,
	VIDEO_CODING_AV1_FRAME_TYPE_SWITCH = 3,
};

enum VideoCodingAV1ReferenceName {
	VIDEO_CODING_AV1_REFERENCE_NAME_INTRA_FRAME = 0,
	VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME = 1,
	VIDEO_CODING_AV1_REFERENCE_NAME_LAST2_FRAME = 2,
	VIDEO_CODING_AV1_REFERENCE_NAME_LAST3_FRAME = 3,
	VIDEO_CODING_AV1_REFERENCE_NAME_GOLDEN_FRAME = 4,
	VIDEO_CODING_AV1_REFERENCE_NAME_BWDREF_FRAME = 5,
	VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF2_FRAME = 6,
	VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF_FRAME = 7,
};

enum VideoCodingAV1InterpolationFilter {
	VIDEO_CODING_AV1_INTERPOLATION_FILTER_EIGHTTAP = 0,
	VIDEO_CODING_AV1_INTERPOLATION_FILTER_EIGHTTAP_SMOOTH = 1,
	VIDEO_CODING_AV1_INTERPOLATION_FILTER_EIGHTTAP_SHARP = 2,
	VIDEO_CODING_AV1_INTERPOLATION_FILTER_BILINEAR = 3,
	VIDEO_CODING_AV1_INTERPOLATION_FILTER_SWITCHABLE = 4,
};

enum VideoCodingAV1TxMode {
	VIDEO_CODING_AV1_TX_MODE_ONLY_4X4 = 0,
	VIDEO_CODING_AV1_TX_MODE_LARGEST = 1,
	VIDEO_CODING_AV1_TX_MODE_SELECT = 2,
};

enum VideoCodingAV1FrameRestorationType {
	VIDEO_CODING_AV1_FRAME_RESTORATION_TYPE_NONE = 0,
	VIDEO_CODING_AV1_FRAME_RESTORATION_TYPE_WIENER = 1,
	VIDEO_CODING_AV1_FRAME_RESTORATION_TYPE_SGRPROJ = 2,
	VIDEO_CODING_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE = 3,
};

enum VideoCodingAV1ColorPrimaries {
	VIDEO_CODING_AV1_COLOR_PRIMARIES_BT_709 = 1,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_BT_UNSPECIFIED = 2,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_BT_470_M = 4,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_BT_470_B_G = 5,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_BT_601 = 6,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_SMPTE_240 = 7,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_GENERIC_FILM = 8,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_BT_2020 = 9,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_XYZ = 10,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_SMPTE_431 = 11,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_SMPTE_432 = 12,
	VIDEO_CODING_AV1_COLOR_PRIMARIES_EBU_3213 = 22,
};

enum VideoCodingAV1TransferCharacteristics {
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_RESERVED_0 = 0,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_BT_709 = 1,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED = 2,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_RESERVED_3 = 3,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_BT_470_M = 4,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_BT_470_B_G = 5,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_BT_601 = 6,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_SMPTE_240 = 7,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_LINEAR = 8,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_LOG_100 = 9,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_LOG_100_SQRT10 = 10,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_IEC_61966 = 11,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_BT_1361 = 12,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_SRGB = 13,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_BT_2020_10_BIT = 14,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_BT_2020_12_BIT = 15,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_SMPTE_2084 = 16,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_SMPTE_428 = 17,
	VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_HLG = 18,
};

enum VideoCodingAV1MatrixCoefficients {
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_IDENTITY = 0,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_BT_709 = 1,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED = 2,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_RESERVED_3 = 3,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_FCC = 4,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_BT_470_B_G = 5,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_BT_601 = 6,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_SMPTE_240 = 7,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_SMPTE_YCGCO = 8,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_BT_2020_NCL = 9,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_BT_2020_CL = 10,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_SMPTE_2085 = 11,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_CHROMAT_NCL = 12,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_CHROMAT_CL = 13,
	VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_ICTCP = 14,
};

enum VideoCodingAV1ChromaSamplePosition {
	VIDEO_CODING_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN = 0,
	VIDEO_CODING_AV1_CHROMA_SAMPLE_POSITION_VERTICAL = 1,
	VIDEO_CODING_AV1_CHROMA_SAMPLE_POSITION_COLOCATED = 2,
	VIDEO_CODING_AV1_CHROMA_SAMPLE_POSITION_RESERVED = 3,
};

struct VideoCodingAV1ColorConfig {
	uint8_t bit_depth;

	bool monochrome_flag;

	bool color_description_present_flag;
	VideoCodingAV1ColorPrimaries color_primaries;
	VideoCodingAV1TransferCharacteristics transfer_characteristics;
	VideoCodingAV1MatrixCoefficients matrix_coefficients;

	bool color_range_flag;
	uint8_t subsampling_x;
	uint8_t subsampling_y;
	VideoCodingAV1ChromaSamplePosition chroma_sample_position;

	bool separate_uv_delta_q;
};

struct VideoCodingAV1TimingInfo {
	uint32_t num_units_in_display_tick;
	uint32_t time_scale;

	bool equal_picture_interval;
	uint32_t num_ticks_per_picture_minus_1;
};

struct VideoCodingAV1DecoderModelInfo {
	uint8_t buffer_delay_length_minus_1;
	uint32_t num_units_in_decoding_tick;
	uint8_t buffer_removal_time_length_minus_1;
	uint8_t frame_presentation_time_length_minus_1;
};

struct VideoCodingAV1LoopFilter {
	uint8_t loop_filter_level[VIDEO_CODING_AV1_MAX_LOOP_FILTER_STRENGTHS];
	uint8_t loop_filter_ref_deltas[VIDEO_CODING_AV1_TOTAL_REFS_PER_FRAME];
	uint8_t loop_filter_mode_deltas[VIDEO_CODING_AV1_LOOP_FILTER_ADJUSTMENTS];

	uint8_t loop_filter_sharpness;
	bool loop_filter_delta_enabled;
	bool loop_filter_delta_update;
	bool update_mode_delta;
};

struct VideoCodingAV1Quantization {
	uint8_t base_q_idx;

	bool diff_uv_delta;

	uint8_t delta_q_y_dc;
	uint8_t delta_q_u_dc;
	uint8_t delta_q_u_ac;
	uint8_t delta_q_v_dc;
	uint8_t delta_q_v_ac;

	bool using_qmatrix;

	uint8_t qm_y;
	uint8_t qm_u;
	uint8_t qm_v;
};

struct VideoCodingAV1Segmentation {
	uint8_t feature_enabled[VIDEO_CODING_AV1_MAX_SEGMENTS];
	int16_t feature_data[VIDEO_CODING_AV1_MAX_SEGMENTS][VIDEO_CODING_AV1_SEG_LVL_MAX];
};

struct VideoCodingAV1TileInfo {
	bool uniform_tile_spacing_flag;

	uint8_t tile_cols;
	uint8_t tile_rows;

	uint16_t context_update_tile_id;
};

struct VideoCodingAV1CDEF {
};

struct VideoCodingAV1LoopRestoration {
};

struct VideoCodingAV1GlobalMotion {
};

struct VideoCodingAV1FilmGrain {
};

struct VideoCodingAV1SequenceHeader {
	uint8_t seq_profile;

	bool still_picture_flag;
	bool reduced_still_picture_header_flag;

	uint8_t operating_points_cnt_minus_1;
	Vector<uint32_t> operating_point_idcs;
	Vector<uint8_t> seq_level_idcs;
	Vector<uint8_t> seq_tiers;

	Vector<bool> decoder_model_present_for_this_op;

	Vector<bool> initial_display_delay_present_for_this_op;
	Vector<uint8_t> initial_display_delay_minus_1;

	bool timing_info_present_flag;
	bool decoder_model_info_present_flag;

	VideoCodingAV1TimingInfo timing_info;
	VideoCodingAV1DecoderModelInfo decoder_model_info;

	bool initial_display_delay_present_flag;

	uint8_t frame_width_bits_minus_1;
	uint8_t frame_height_bits_minus_1;
	uint64_t max_frame_width_minus_1;
	uint64_t max_frame_height_minus_1;

	bool frame_id_numbers_present_flag;
	uint8_t delta_frame_id_length_minus_2;
	uint8_t additional_frame_id_length_minus_1;

	bool use_128x128_superblock_flag;
	bool enable_filter_intra_flag;
	bool enable_intra_edge_filter_flag;
	bool enable_interintra_compound_flag;
	bool enable_masked_compound_flag;
	bool enable_warped_motion_flag;
	bool enable_dual_filter_flag;
	bool enable_order_hint_flag;
	bool enable_jnt_comp_flag;
	bool enable_ref_frame_mvs_flag;

	uint8_t seq_force_screen_content_tools;
	uint8_t seq_force_integer_mv;

	uint8_t order_hint_bits;

	bool enable_superres_flag;
	bool enable_cdef_flag;
	bool enable_restoration_flag;

	bool film_grain_params_present_flag;

	VideoCodingAV1ColorConfig color_config;
};
