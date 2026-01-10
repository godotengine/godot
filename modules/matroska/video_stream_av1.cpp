/**************************************************************************/
/*  video_stream_av1.cpp                                                  */
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

#include "video_stream_av1.h"

#include "core/string/print_string.h"
#include "core/typedefs.h"
#include "servers/rendering/rendering_device_commons.h"
#include "servers/rendering/video/video_coding_av1.h"

uint64_t VideoStreamAV1::read_bits(uint8_t p_bits) {
	uint8_t bits = p_bits % 8;
	uint8_t bytes = p_bits / 8;

	uint64_t encoded = 0;
	for (uint8_t i = 0; i < bytes; i++) {
		encoded = (encoded << 8) | src[i];
	}

	src += bytes;

	uint8_t partial_byte = shift + bits;
	if (partial_byte > 8) {
		encoded = (encoded << 8) | src[0];
		encoded = (encoded << 8) | src[1];

		src += 1;
		bytes += 2;
	} else if (partial_byte == 8) {
		encoded = (encoded << 8) | src[0];

		src += 1;
		bytes += 1;
	} else if (partial_byte > 0) {
		encoded = (encoded << 8) | src[0];
		bytes += 1;
	}

	uint8_t start = 8 * bytes - 1 - shift;
	uint8_t end = start + 1 - p_bits;
	shift = (shift + bits) % 8;

	uint64_t value = 0;
	for (uint8_t offset = start + 1; offset > end; offset--) {
		value |= encoded & (1ULL << (offset - 1));
	}

	return value >> end;
}

uint64_t VideoStreamAV1::read_uvlc() {
	uint8_t leading_zeroes = 0;
	while (!read_bits(1)) {
		leading_zeroes++;
	}

	if (leading_zeroes >= 32) {
		return (1ULL << 32) - 1;
	}

	uint64_t value = read_bits(leading_zeroes);
	return value + (1ULL << leading_zeroes) - 1;
}

uint64_t VideoStreamAV1::read_leb128() {
	uint64_t value = 0;

	for (uint32_t i = 0; i < 8; i++) {
		bool stop = (src[0] & 0x80) == 0;
		value |= (src[0] & 0x7f) << (i * 7);

		src += 1;

		if (stop) {
			break;
		}
	}

	return value;
}

int64_t VideoStreamAV1::read_su(uint8_t p_bits) {
	int64_t value = read_bits(p_bits);
	uint64_t sign_mask = 1 << (p_bits - 1);
	if (value & sign_mask) {
		value -= 2 * sign_mask;
	}

	return value;
}

int64_t VideoStreamAV1::read_delta_q() {
	int64_t delta_q = 0;
	bool delta_coded = read_bits(1);
	if (delta_coded) {
		delta_q = read_su(7);
	}

	return delta_q;
}

uint8_t VideoStreamAV1::tile_log2(uint32_t p_blk, uint32_t p_target) {
	uint8_t k = 0;
	while ((p_blk << k) < p_target) {
		k++;
	}

	return k;
}

int8_t VideoStreamAV1::get_relative_distance(uint8_t a, uint8_t b) {
	int16_t diff = a - b;
	uint8_t m = 1 << (av1_sequence_header.order_hint_bits - 1);
	diff = (diff & (m - 1)) - (diff & m);
	return diff;
}

bool VideoStreamAV1::parse_open_bitstream_unit() {
	read_bits(1); // obu_forbidden_bit
	uint8_t obu_type = read_bits(4);
	bool obu_extension_flag = read_bits(1) > 0;
	bool obu_has_size_field = read_bits(1) > 0;
	read_bits(1); // obu_reserved_1bit

	if (obu_extension_flag) {
		read_bits(3); // temporal_id
		read_bits(2); // spacial_id
		read_bits(3); // extension_header_reserved_3bits
	}

	uint64_t obu_size;
	if (obu_has_size_field) {
		obu_size = read_leb128();
	} else {
		ERR_FAIL_V_MSG(false, "Unknown OBU size, refusing to decode");
	}

	const uint8_t *obu_start = src;

	bool is_frame = false;
	switch (obu_type) {
		case VIDEO_CODING_AV1_OBU_TYPE_SEQUENCE_HEADER: {
			av1_sequence_header = parse_sequence_header();
		} break;

		case VIDEO_CODING_AV1_OBU_TYPE_FRAME_HEADER: {
			// TODO: aaaaaaaaaaaaaaaaa
			is_frame = false;
		} break;

		case VIDEO_CODING_AV1_OBU_TYPE_METADATA: {
		} break;

		case VIDEO_CODING_AV1_OBU_TYPE_FRAME: {
			is_frame = true;
		} break;

		default: {
			WARN_PRINT(vformat("Unknown OBU Type (%d) [%d]", obu_type, obu_size));
		}
	}

	src = obu_start + obu_size;
	shift = 0;

	return is_frame;
}

VideoCodingAV1SequenceHeader VideoStreamAV1::parse_sequence_header() {
	VideoCodingAV1SequenceHeader sequence_header = {};

	sequence_header.seq_profile = read_bits(3);
	sequence_header.still_picture_flag = read_bits(1) > 0;

	sequence_header.reduced_still_picture_header_flag = read_bits(1) > 0;
	if (sequence_header.reduced_still_picture_header_flag) {
		sequence_header.timing_info_present_flag = false;
		sequence_header.decoder_model_info_present_flag = false;
		sequence_header.initial_display_delay_present_flag = false;
		sequence_header.operating_points_cnt_minus_1 = 0;
		sequence_header.operating_point_idcs.push_back(0);
		sequence_header.seq_level_idcs.push_back(read_bits(5));
		sequence_header.seq_tiers.push_back(0);
		sequence_header.decoder_model_present_for_this_op.push_back(false);
		sequence_header.initial_display_delay_present_for_this_op.push_back(false);
	} else {
		sequence_header.timing_info_present_flag = read_bits(1) > 0;
		if (sequence_header.timing_info_present_flag) {
			sequence_header.timing_info.num_units_in_display_tick = read_bits(32);
			sequence_header.timing_info.time_scale = read_bits(32);

			sequence_header.timing_info.equal_picture_interval = read_bits(1);
			if (sequence_header.timing_info.equal_picture_interval) {
				sequence_header.timing_info.num_ticks_per_picture_minus_1 = read_uvlc();
			}

			sequence_header.decoder_model_info_present_flag = read_bits(1) > 0;
			if (sequence_header.decoder_model_info_present_flag) {
				sequence_header.decoder_model_info.buffer_delay_length_minus_1 = read_bits(5);
				sequence_header.decoder_model_info.num_units_in_decoding_tick = read_bits(32);
				sequence_header.decoder_model_info.buffer_removal_time_length_minus_1 = read_bits(5);
				sequence_header.decoder_model_info.frame_presentation_time_length_minus_1 = read_bits(5);
			}
		} else {
			sequence_header.decoder_model_info_present_flag = false;
		}

		sequence_header.initial_display_delay_present_flag = read_bits(1) > 0;

		sequence_header.operating_points_cnt_minus_1 = read_bits(5);
		sequence_header.operating_point_idcs.resize(sequence_header.operating_points_cnt_minus_1 + 1);
		sequence_header.seq_level_idcs.resize(sequence_header.operating_points_cnt_minus_1 + 1);
		sequence_header.seq_tiers.resize(sequence_header.operating_points_cnt_minus_1 + 1);
		sequence_header.decoder_model_present_for_this_op.resize(sequence_header.operating_points_cnt_minus_1 + 1);
		sequence_header.initial_display_delay_present_for_this_op.resize(sequence_header.operating_points_cnt_minus_1 + 1);
		sequence_header.initial_display_delay_minus_1.resize(sequence_header.operating_points_cnt_minus_1 + 1);

		for (uint64_t i = 0; i <= sequence_header.operating_points_cnt_minus_1; i++) {
			sequence_header.operating_point_idcs.set(i, read_bits(12));
			sequence_header.seq_level_idcs.set(i, read_bits(5));

			if (sequence_header.seq_level_idcs[i] > 7) {
				sequence_header.seq_tiers.set(i, read_bits(1));
			} else {
				sequence_header.seq_tiers.set(i, 0);
			}

			if (sequence_header.decoder_model_info_present_flag) {
				sequence_header.decoder_model_present_for_this_op.set(i, read_bits(1));
				if (sequence_header.decoder_model_present_for_this_op[i]) {
					uint8_t size = sequence_header.decoder_model_info.buffer_delay_length_minus_1 + 1;
					read_bits(size); // decoder_buffer_delay
					read_bits(size); // encoder_buffer_delay
					read_bits(1); // low_delay_mode_flag
				}
			}

			if (sequence_header.initial_display_delay_present_flag) {
				sequence_header.initial_display_delay_present_for_this_op.set(i, read_bits(1));
				if (sequence_header.initial_display_delay_present_for_this_op[i]) {
					sequence_header.initial_display_delay_minus_1.set(i, read_bits(4));
				}
			}
		}
	}

	sequence_header.frame_width_bits_minus_1 = read_bits(4);
	sequence_header.frame_height_bits_minus_1 = read_bits(4);
	sequence_header.max_frame_width_minus_1 = read_bits(sequence_header.frame_width_bits_minus_1 + 1);
	sequence_header.max_frame_height_minus_1 = read_bits(sequence_header.frame_height_bits_minus_1 + 1);

	if (sequence_header.reduced_still_picture_header_flag) {
		sequence_header.frame_id_numbers_present_flag = false;
	} else {
		sequence_header.frame_id_numbers_present_flag = read_bits(1) > 0;
	}

	if (sequence_header.frame_id_numbers_present_flag) {
		sequence_header.delta_frame_id_length_minus_2 = read_bits(4);
		sequence_header.additional_frame_id_length_minus_1 = read_bits(3);
	}

	sequence_header.use_128x128_superblock_flag = read_bits(1) > 0;
	sequence_header.enable_filter_intra_flag = read_bits(1) > 0;
	sequence_header.enable_intra_edge_filter_flag = read_bits(1) > 0;

	if (sequence_header.reduced_still_picture_header_flag) {
		sequence_header.enable_interintra_compound_flag = false;
		sequence_header.enable_masked_compound_flag = false;
		sequence_header.enable_warped_motion_flag = false;
		sequence_header.enable_dual_filter_flag = false;
		sequence_header.enable_order_hint_flag = false;
		sequence_header.enable_jnt_comp_flag = false;
		sequence_header.enable_ref_frame_mvs_flag = false;
		sequence_header.seq_force_screen_content_tools = VIDEO_CODING_AV1_SELECT_SCREEN_CONTENT_TOOLS;
		sequence_header.seq_force_integer_mv = VIDEO_CODING_AV1_SELECT_INTEGER_MV;
		sequence_header.order_hint_bits = 0;
	} else {
		sequence_header.enable_interintra_compound_flag = read_bits(1) > 0;
		sequence_header.enable_masked_compound_flag = read_bits(1) > 0;
		sequence_header.enable_warped_motion_flag = read_bits(1) > 0;
		sequence_header.enable_dual_filter_flag = read_bits(1) > 0;

		sequence_header.enable_order_hint_flag = read_bits(1) > 0;
		if (sequence_header.enable_order_hint_flag) {
			sequence_header.enable_jnt_comp_flag = read_bits(1) > 0;
			sequence_header.enable_ref_frame_mvs_flag = read_bits(1) > 0;
		} else {
			sequence_header.enable_jnt_comp_flag = false;
			sequence_header.enable_ref_frame_mvs_flag = false;
		}

		bool seq_choose_screen_content_tools = read_bits(1);
		if (seq_choose_screen_content_tools) {
			sequence_header.seq_force_screen_content_tools = VIDEO_CODING_AV1_SELECT_SCREEN_CONTENT_TOOLS;
		} else {
			sequence_header.seq_force_screen_content_tools = read_bits(1);
		}

		if (sequence_header.seq_force_screen_content_tools > 0) {
			bool seq_choose_integer_mv = read_bits(1) > 0;
			if (seq_choose_integer_mv) {
				sequence_header.seq_force_integer_mv = VIDEO_CODING_AV1_SELECT_INTEGER_MV;
			} else {
				sequence_header.seq_force_integer_mv = read_bits(1);
			}
		} else {
			sequence_header.seq_force_integer_mv = VIDEO_CODING_AV1_SELECT_INTEGER_MV;
		}

		if (sequence_header.enable_order_hint_flag) {
			sequence_header.order_hint_bits = read_bits(3) + 1;
		} else {
			sequence_header.order_hint_bits = 0;
		}
	}

	sequence_header.enable_superres_flag = read_bits(1) > 0;
	sequence_header.enable_cdef_flag = read_bits(1) > 0;
	sequence_header.enable_restoration_flag = read_bits(1) > 0;

	bool high_bitdepth = read_bits(1) > 0;
	if (sequence_header.seq_profile == 2 && high_bitdepth) {
		bool twelve_bit = read_bits(1) > 0;
		if (twelve_bit) {
			sequence_header.color_config.bit_depth = 12;
		} else {
			sequence_header.color_config.bit_depth = 10;
		}
	} else {
		if (high_bitdepth) {
			sequence_header.color_config.bit_depth = 10;
		} else {
			sequence_header.color_config.bit_depth = 8;
		}
	}

	if (sequence_header.seq_profile == 1) {
		sequence_header.color_config.monochrome_flag = false;
	} else {
		sequence_header.color_config.monochrome_flag = read_bits(1);
	}

	sequence_header.color_config.color_description_present_flag = read_bits(1) > 0;
	if (sequence_header.color_config.color_description_present_flag) {
		sequence_header.color_config.color_primaries = VideoCodingAV1ColorPrimaries(read_bits(8));
		sequence_header.color_config.transfer_characteristics = VideoCodingAV1TransferCharacteristics(read_bits(8));
		sequence_header.color_config.matrix_coefficients = VideoCodingAV1MatrixCoefficients(read_bits(8));
	} else {
		sequence_header.color_config.color_primaries = VIDEO_CODING_AV1_COLOR_PRIMARIES_BT_UNSPECIFIED;
		sequence_header.color_config.transfer_characteristics = VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED;
		sequence_header.color_config.matrix_coefficients = VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED;
	}

	if (sequence_header.color_config.monochrome_flag) {
		sequence_header.color_config.color_range_flag = read_bits(1) > 0;
		sequence_header.color_config.subsampling_x = 1;
		sequence_header.color_config.subsampling_y = 1;
		sequence_header.color_config.chroma_sample_position = VIDEO_CODING_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN;
		sequence_header.color_config.separate_uv_delta_q = false;
	} else if (sequence_header.color_config.color_primaries == VIDEO_CODING_AV1_COLOR_PRIMARIES_BT_709 && sequence_header.color_config.transfer_characteristics == VIDEO_CODING_AV1_TRANSFER_CHARACTERISTICS_SRGB && sequence_header.color_config.matrix_coefficients == VIDEO_CODING_AV1_MATRIX_COEFFICIENTS_IDENTITY) {
		sequence_header.color_config.color_range_flag = true;
		sequence_header.color_config.subsampling_x = 0;
		sequence_header.color_config.subsampling_y = 0;
		sequence_header.color_config.separate_uv_delta_q = read_bits(1);
	} else {
		sequence_header.color_config.color_range_flag = read_bits(1) > 0;

		if (sequence_header.seq_profile == 0) {
			sequence_header.color_config.subsampling_x = 1;
			sequence_header.color_config.subsampling_y = 1;
		} else if (sequence_header.seq_profile == 1) {
			sequence_header.color_config.subsampling_x = 0;
			sequence_header.color_config.subsampling_y = 0;
		} else {
			if (sequence_header.color_config.bit_depth == 12) {
				sequence_header.color_config.subsampling_x = read_bits(1);
				if (sequence_header.color_config.subsampling_x) {
					sequence_header.color_config.subsampling_y = read_bits(1);
				} else {
					sequence_header.color_config.subsampling_y = 0;
				}
			} else {
				sequence_header.color_config.subsampling_x = 1;
				sequence_header.color_config.subsampling_y = 0;
			}

			if (sequence_header.color_config.subsampling_x && sequence_header.color_config.subsampling_y) {
				sequence_header.color_config.chroma_sample_position = VideoCodingAV1ChromaSamplePosition(read_bits(2));
			}
		}

		sequence_header.color_config.separate_uv_delta_q = read_bits(1);
	}

	sequence_header.film_grain_params_present_flag = read_bits(1) > 0;

	return sequence_header;
}

VideoDecodeAV1Frame VideoStreamAV1::parse_frame_header() {
	VideoDecodeAV1Frame std_frame_header = {};

	uint64_t id_len = 0;
	if (av1_sequence_header.frame_id_numbers_present_flag) {
		id_len = av1_sequence_header.additional_frame_id_length_minus_1 + av1_sequence_header.delta_frame_id_length_minus_2 + 3;
	}

	uint8_t all_frames = (1 << VIDEO_CODING_AV1_NUM_REF_FRAMES) - 1;

	bool show_existing_frame;
	bool show_frame;
	bool showable_frame;

	bool is_intra;

	if (av1_sequence_header.reduced_still_picture_header_flag) {
		show_existing_frame = false;
		std_frame_header.frame_type = VIDEO_CODING_AV1_FRAME_TYPE_KEY;
		is_intra = true;
		show_frame = true;
		showable_frame = false;
	} else {
		show_existing_frame = read_bits(1);
		if (show_existing_frame) {
			uint8_t frame_to_show_map_idx = read_bits(3);

			if (av1_sequence_header.decoder_model_info_present_flag && !av1_sequence_header.timing_info.equal_picture_interval) {
				read_bits(av1_sequence_header.decoder_model_info.frame_presentation_time_length_minus_1 + 1); // frame_presentation_time
			}

			std_frame_header.refresh_frame_flags = 0;
			if (av1_sequence_header.frame_id_numbers_present_flag) {
				uint32_t display_frame_id = read_bits(id_len);
			}

			std_frame_header.frame_type = ref_frame_types[frame_to_show_map_idx];
			if (std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_KEY) {
				std_frame_header.refresh_frame_flags = all_frames;
			}

			if (av1_sequence_header.film_grain_params_present_flag) {
				//TODO: load_grain_params
			}

			return std_frame_header;
		}

		std_frame_header.frame_type = VideoCodingAV1FrameType(read_bits(2));
		is_intra = std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_KEY || std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_INTRA_ONLY;

		show_frame = read_bits(1);
		if (show_frame && av1_sequence_header.decoder_model_info_present_flag && !av1_sequence_header.timing_info.equal_picture_interval) {
			read_bits(av1_sequence_header.decoder_model_info.frame_presentation_time_length_minus_1 + 1); // frame_presentation_time
		}

		if (show_frame) {
			showable_frame = std_frame_header.frame_type != VIDEO_CODING_AV1_FRAME_TYPE_KEY;
		} else {
			showable_frame = read_bits(1);
		}

		if (std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_SWITCH || (std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_KEY && show_frame)) {
			std_frame_header.error_resilient_mode_flag = true;
		} else {
			std_frame_header.error_resilient_mode_flag = read_bits(1);
		}
	}

	if (std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_KEY && show_frame) {
		for (uint64_t i = 0; i < VIDEO_CODING_AV1_NUM_REF_FRAMES; i++) {
			ref_valid[i] = 0;
			ref_order_hint[i] = 0;
		}

		for (uint64_t i = 0; i < VIDEO_CODING_AV1_REFS_PER_FRAME; i++) {
			std_frame_header.order_hints[VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME + i] = 0;
		}
	}

	std_frame_header.disable_cdf_update_flag = read_bits(1);

	if (av1_sequence_header.seq_force_screen_content_tools == VIDEO_CODING_AV1_SELECT_SCREEN_CONTENT_TOOLS) {
		std_frame_header.allow_screen_content_tools_flag = read_bits(1);
	} else {
		std_frame_header.allow_screen_content_tools_flag = av1_sequence_header.seq_force_screen_content_tools;
	}

	if (std_frame_header.allow_screen_content_tools_flag) {
		if (av1_sequence_header.seq_force_integer_mv == VIDEO_CODING_AV1_SELECT_INTEGER_MV) {
			std_frame_header.force_integer_mv_flag = read_bits(1);
		} else {
			std_frame_header.force_integer_mv_flag = av1_sequence_header.seq_force_integer_mv;
		}
	} else {
		std_frame_header.force_integer_mv_flag = false;
	}

	if (is_intra) {
		std_frame_header.force_integer_mv_flag = true;
	}

	if (av1_sequence_header.frame_id_numbers_present_flag) {
		std_frame_header.current_frame_id = read_bits(id_len);
		WARN_PRINT("Skipping mark_ref_frames");
		// TODO: mark_ref_frames()
	} else {
		std_frame_header.current_frame_id = 0;
	}

	if (std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_SWITCH) {
		std_frame_header.frame_size_override_flag = true;
	} else if (av1_sequence_header.reduced_still_picture_header_flag) {
		std_frame_header.frame_size_override_flag = false;
	} else {
		std_frame_header.frame_size_override_flag = read_bits(1);
	}

	std_frame_header.order_hint = read_bits(av1_sequence_header.order_hint_bits);

	if (is_intra || std_frame_header.error_resilient_mode_flag) {
		std_frame_header.primary_ref_frame = VIDEO_CODING_AV1_PRIMARY_REF_NONE;
	} else {
		std_frame_header.primary_ref_frame = read_bits(3);
	}

	if (av1_sequence_header.decoder_model_info_present_flag) {
		std_frame_header.buffer_removal_time_present_flag = read_bits(1);
		if (std_frame_header.buffer_removal_time_present_flag) {
			WARN_PRINT("Skipping using temporal/spatial id");
			for (uint64_t i = 0; i <= av1_sequence_header.operating_points_cnt_minus_1; i++) {
				uint16_t op_idc = av1_sequence_header.operating_point_idcs[i];
				bool in_temporal_layer = (op_idc >> 0) & 1; //TODO: use temporal_id
				bool in_spacial_layer = (op_idc >> 8) & 1; //TODO: use spacial_id
				if (op_idc == 0 || (in_temporal_layer && in_spacial_layer)) {
					read_bits(av1_sequence_header.decoder_model_info.buffer_removal_time_length_minus_1); // buffer_removal_time
				}
			}
		}
	}

	std_frame_header.allow_high_precision_mv = false;
	std_frame_header.use_ref_frame_mvs = false;
	std_frame_header.allow_intrabc = false;

	if (std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_SWITCH || (std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_KEY && show_frame)) {
		std_frame_header.refresh_frame_flags = all_frames;
	} else {
		std_frame_header.refresh_frame_flags = read_bits(8);
	}

	if (!is_intra || std_frame_header.refresh_frame_flags != all_frames) {
		if (std_frame_header.error_resilient_mode_flag && av1_sequence_header.enable_order_hint_flag) {
			for (uint64_t i = 0; i < VIDEO_CODING_AV1_NUM_REF_FRAMES; i++) {
				uint8_t inner_ref_order_hint = read_bits(av1_sequence_header.order_hint_bits);
				if (inner_ref_order_hint != ref_order_hint[i]) {
					ref_valid[i] = false;
				}
			}
		}
	}

	if (is_intra) {
		if (std_frame_header.frame_size_override_flag) {
			uint64_t frame_width_minus_1 = read_bits(av1_sequence_header.frame_width_bits_minus_1 + 1);
			uint64_t frame_height_minus_1 = read_bits(av1_sequence_header.frame_height_bits_minus_1 + 1);
			print_line(vformat("Frame Size Override (%dx%d)", frame_width_minus_1 + 1, frame_height_minus_1 + 1));
		}

		if (av1_sequence_header.enable_superres_flag) {
			std_frame_header.use_superres = read_bits(1);
		} else {
			std_frame_header.use_superres = false;
		}

		if (std_frame_header.use_superres) {
			std_frame_header.coded_denom = read_bits(3);
		}

		std_frame_header.render_and_frame_size_different = read_bits(1);
		if (std_frame_header.render_and_frame_size_different) {
			uint16_t render_width_minus_1 = read_bits(16);
			uint16_t render_height_minus_1 = read_bits(16);
			print_line(vformat("Render Size (%dx%d)", render_width_minus_1 + 1, render_height_minus_1 + 1));
		}

		if (std_frame_header.allow_screen_content_tools_flag && !std_frame_header.use_superres) {
			std_frame_header.allow_intrabc = read_bits(1);
		}
	} else {
		if (!av1_sequence_header.enable_order_hint_flag) {
			std_frame_header.frame_refs_short_signaling = false;
		} else {
			std_frame_header.frame_refs_short_signaling = read_bits(1);
			if (std_frame_header.frame_refs_short_signaling) {
				read_bits(3); // last_frame_idx
				read_bits(3); // gold_frame_idx
				// TODO: set_frame_refs()
				WARN_PRINT("Skipping set_frame_refs");
			}
		}

		for (uint64_t i = 0; i < VIDEO_CODING_AV1_REFS_PER_FRAME; i++) {
			if (!std_frame_header.frame_refs_short_signaling) {
				std_frame_header.ref_frame_idx[i] = read_bits(3);

				if (av1_sequence_header.frame_id_numbers_present_flag) {
					uint32_t delta_frame_id_minus_1 = read_bits(av1_sequence_header.delta_frame_id_length_minus_2 + 2);
					std_frame_header.expected_frame_id[i] = std_frame_header.current_frame_id + (1 << id_len);
					std_frame_header.expected_frame_id[i] -= delta_frame_id_minus_1 + 1;
					std_frame_header.expected_frame_id[i] %= (1 << id_len);
				}
			}
		}

		if (std_frame_header.frame_size_override_flag && !std_frame_header.error_resilient_mode_flag) {
			// TODO: frame_size_with_refs()
			WARN_PRINT("Skipping frame_size_with_refs");
		} else {
			if (std_frame_header.frame_size_override_flag) {
				uint64_t frame_width_minus_1 = read_bits(av1_sequence_header.frame_width_bits_minus_1 + 1);
				uint64_t frame_height_minus_1 = read_bits(av1_sequence_header.frame_height_bits_minus_1 + 1);
				print_line(vformat("Frame Size Override (%dx%d)", frame_width_minus_1 + 1, frame_height_minus_1 + 1));
			}

			if (av1_sequence_header.enable_superres_flag) {
				std_frame_header.use_superres = read_bits(1);
			} else {
				std_frame_header.use_superres = false;
			}

			if (std_frame_header.use_superres) {
				std_frame_header.coded_denom = read_bits(3);
			}

			std_frame_header.render_and_frame_size_different = read_bits(1);
			if (std_frame_header.render_and_frame_size_different) {
				uint16_t render_width_minus_1 = read_bits(16);
				uint16_t render_height_minus_1 = read_bits(16);
				print_line(vformat("Render Size (%dx%d)", render_width_minus_1 + 1, render_height_minus_1 + 1));
			}
		}

		if (std_frame_header.force_integer_mv_flag) {
			std_frame_header.allow_high_precision_mv = false;
		} else {
			std_frame_header.allow_high_precision_mv = read_bits(1);
		}

		std_frame_header.is_filter_switchable = read_bits(1);
		if (std_frame_header.is_filter_switchable) {
			std_frame_header.interpolation_filter = VIDEO_CODING_AV1_INTERPOLATION_FILTER_SWITCHABLE;
		} else {
			std_frame_header.interpolation_filter = VideoCodingAV1InterpolationFilter(read_bits(2));
		}

		std_frame_header.is_motion_mode_switchable = read_bits(1);

		if (std_frame_header.error_resilient_mode_flag || !av1_sequence_header.enable_ref_frame_mvs_flag) {
			std_frame_header.use_ref_frame_mvs = false;
		} else {
			std_frame_header.use_ref_frame_mvs = read_bits(1);
		}

		std_frame_header.ref_frame_sign_bias = 0;
		for (uint64_t i = 0; i < VIDEO_CODING_AV1_REFS_PER_FRAME; i++) {
			uint8_t ref_frame = VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME + i;
			uint8_t hint = ref_order_hint[std_frame_header.ref_frame_idx[i]];
			std_frame_header.order_hints[ref_frame] = hint;

			int8_t relative_distance = get_relative_distance(hint, std_frame_header.order_hint);
			if (av1_sequence_header.enable_order_hint_flag && relative_distance > 0) {
				std_frame_header.ref_frame_sign_bias |= 1 << i;
			}
		}
	}

	if (av1_sequence_header.reduced_still_picture_header_flag || std_frame_header.disable_cdf_update_flag) {
		std_frame_header.disable_frame_end_update_cdf = true;
	} else {
		std_frame_header.disable_frame_end_update_cdf = read_bits(1);
	}

	if (std_frame_header.primary_ref_frame == VIDEO_CODING_AV1_PRIMARY_REF_NONE) {
		// TODO: init_non_coeff_cdfs

		for (uint32_t segment = 0; segment < VIDEO_CODING_AV1_MAX_SEGMENTS; segment++) {
			std_frame_header.segmentation.feature_enabled[segment] = 0;
			for (uint32_t lvl = 0; lvl < VIDEO_CODING_AV1_SEG_LVL_MAX; lvl++) {
				std_frame_header.segmentation.feature_data[segment][lvl] = 0;
			}
		}

		// TODO PrevSegmentIds

		for (uint32_t ref = VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME; ref < VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF_FRAME; ref++) {
			std_frame_header.global_motion.gm_type[ref] = VIDEO_CODING_AV1_WARP_MODEL_IDENTITY;
		}

		std_frame_header.loop_filter.loop_filter_delta_enabled = true;
		std_frame_header.loop_filter.loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_INTRA_FRAME] = 1;
		std_frame_header.loop_filter.loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME] = 0;
		std_frame_header.loop_filter.loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_LAST2_FRAME] = 0;
		std_frame_header.loop_filter.loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_LAST3_FRAME] = 0;
		std_frame_header.loop_filter.loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_BWDREF_FRAME] = 0;
		std_frame_header.loop_filter.loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_GOLDEN_FRAME] = -1;
		std_frame_header.loop_filter.loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF_FRAME] = -1;
		std_frame_header.loop_filter.loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF2_FRAME] = -1;
		std_frame_header.loop_filter.loop_filter_mode_deltas[0] = 0;
		std_frame_header.loop_filter.loop_filter_mode_deltas[1] = 0;
	} else {
		// TODO: load_cdfs

		uint8_t prev_frame = std_frame_header.ref_frame_idx[std_frame_header.primary_ref_frame];

		// TODO load global motion

		// load_loop_filter_params()
		std_frame_header.loop_filter.loop_filter_mode_deltas[0] = loop_filter_mode_deltas[prev_frame][0];
		std_frame_header.loop_filter.loop_filter_mode_deltas[1] = loop_filter_mode_deltas[prev_frame][1];
		for (uint8_t j = 0; j < VIDEO_CODING_AV1_TOTAL_REFS_PER_FRAME; j++) {
			std_frame_header.loop_filter.loop_filter_ref_deltas[j] = loop_filter_ref_deltas[prev_frame][j];
		}

		// load_segmentation_params()
		for (uint8_t segment = 0; segment < VIDEO_CODING_AV1_MAX_SEGMENTS; segment++) {
			std_frame_header.segmentation.feature_enabled[segment] = segmentation_feature_enabled[prev_frame][segment];
			for (uint8_t lvl = 0; lvl < VIDEO_CODING_AV1_SEG_LVL_MAX; lvl++) {
				std_frame_header.segmentation.feature_data[segment][lvl] = segmentation_feature_data[prev_frame][segment][lvl];
			}
		}
	}

	if (std_frame_header.use_ref_frame_mvs) {
		// TODO: motion field estimation
	}

	// TODO: lots of stuff is render size is different
	uint32_t mi_cols = 2 * ((av1_sequence_header.max_frame_width_minus_1 + 8) >> 3);
	uint32_t mi_rows = 2 * ((av1_sequence_header.max_frame_height_minus_1 + 8) >> 3);
	parse_tile_info(mi_cols, mi_rows, &std_frame_header.tile_info);

	parse_quantization(&std_frame_header.quantization);
	parse_segmentation(std_frame_header.primary_ref_frame, &std_frame_header.segmentation);

	if (std_frame_header.quantization.base_q_idx > 0) {
		std_frame_header.delta_q_present = read_bits(1);
		if (std_frame_header.delta_q_present) {
			std_frame_header.delta_q_res = read_bits(2);
		} else {
			std_frame_header.delta_q_res = 0;
		}
	} else {
		std_frame_header.delta_q_present = false;
		std_frame_header.delta_q_res = 0;
	}

	if (std_frame_header.delta_q_present) {
		if (!std_frame_header.allow_intrabc) {
			std_frame_header.delta_lf_present = read_bits(1);
			if (std_frame_header.delta_lf_present) {
				std_frame_header.delta_lf_res = read_bits(2);
				std_frame_header.delta_lf_multi = read_bits(1);
			} else {
				std_frame_header.delta_lf_res = 0;
				std_frame_header.delta_lf_multi = false;
			}
		} else {
			std_frame_header.delta_lf_present = false;
			std_frame_header.delta_lf_res = 0;
			std_frame_header.delta_lf_multi = false;
		}
	}

	if (std_frame_header.primary_ref_frame == VIDEO_CODING_AV1_PRIMARY_REF_NONE) {
		// TODO: init_coeff_cdfs
	} else {
		// TODO: load_previous_segment_ids
	}

	bool coded_lossless = true;
	for (uint32_t segment = 0; segment < VIDEO_CODING_AV1_MAX_SEGMENTS; segment++) {
		uint8_t qindex;
		if (std_frame_header.segmentation.segmentation_enabled && (std_frame_header.segmentation.feature_enabled[segment] & (1 << VIDEO_CODING_AV1_SEG_LVL_ALT_Q))) {
			uint32_t segment_data = std_frame_header.segmentation.feature_data[segment][VIDEO_CODING_AV1_SEG_LVL_ALT_Q];
			qindex = std_frame_header.quantization.base_q_idx + segment_data;
		} else {
			qindex = std_frame_header.quantization.base_q_idx;
		}

		bool lossless_array = qindex == 0 && std_frame_header.quantization.delta_q_y_dc == 0 && std_frame_header.quantization.delta_q_u_dc == 0 && std_frame_header.quantization.delta_q_u_ac == 0 && std_frame_header.quantization.delta_q_v_dc == 0 && std_frame_header.quantization.delta_q_v_ac == 0;

		if (!lossless_array) {
			coded_lossless = false;
		}

		if (std_frame_header.quantization.using_qmatrix) {
			// TODO: SegQMLevel
			WARN_PRINT("Skipping SegQMLevel");
		}
	}

	// TODO: frame width == upscaled width
	bool all_lossless = coded_lossless && true;

	parse_loop_filter(coded_lossless, std_frame_header.allow_intrabc, &std_frame_header.loop_filter);
	parse_cdef(coded_lossless, std_frame_header.allow_intrabc, &std_frame_header.cdef);
	parse_loop_restoration(all_lossless, std_frame_header.allow_intrabc, &std_frame_header.loop_restoration);

	if (coded_lossless) {
		std_frame_header.tx_mode = VIDEO_CODING_AV1_TX_MODE_ONLY_4X4;
	} else {
		bool tx_mode_select = read_bits(1);
		if (tx_mode_select) {
			std_frame_header.tx_mode = VIDEO_CODING_AV1_TX_MODE_SELECT;
		} else {
			std_frame_header.tx_mode = VIDEO_CODING_AV1_TX_MODE_LARGEST;
		}
	}

	if (is_intra) {
		std_frame_header.reference_select = false;
	} else {
		std_frame_header.reference_select = read_bits(1);
	}

	if (is_intra || !std_frame_header.reference_select || !av1_sequence_header.enable_order_hint_flag) {
		std_frame_header.skip_mode_allowed = false;
	} else {
		int8_t forwards_idx = -1;
		int8_t forwards_hint = 0;
		int8_t backwards_idx = -1;
		int8_t backwards_hint = 0;
		for (uint8_t i = 0; i < VIDEO_CODING_AV1_REFS_PER_FRAME; i++) {
			int8_t ref_hint = ref_order_hint[std_frame_header.ref_frame_idx[i]];
			if (get_relative_distance(ref_hint, std_frame_header.order_hint) < 0) {
				if (forwards_idx < 0 || get_relative_distance(ref_hint, forwards_hint) > 0) {
					forwards_idx = i;
					forwards_hint = ref_hint;
				}
			} else if (get_relative_distance(ref_hint, std_frame_header.order_hint) > 0) {
				if (backwards_idx < 0 || get_relative_distance(ref_hint, backwards_hint) < 0) {
					backwards_idx = i;
					backwards_hint = ref_hint;
				}
			}
		}

		if (forwards_idx < 0) {
			std_frame_header.skip_mode_allowed = false;
		} else if (backwards_idx >= 0) {
			std_frame_header.skip_mode_allowed = true;
			std_frame_header.skip_mode_frame[0] = VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME + MIN(forwards_idx, backwards_idx);
			std_frame_header.skip_mode_frame[1] = VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME + MAX(forwards_idx, backwards_idx);
		} else {
			int8_t second_forwards_idx = -1;
			int8_t second_forwards_hint = -1;
			for (uint8_t i = 0; i < VIDEO_CODING_AV1_REFS_PER_FRAME; i++) {
				int8_t ref_hint = ref_order_hint[std_frame_header.ref_frame_idx[i]];
				if (get_relative_distance(ref_hint, forwards_hint) < 0) {
					if (second_forwards_idx < 0 || get_relative_distance(ref_hint, second_forwards_hint) > 0) {
						second_forwards_idx = i;
						second_forwards_hint = ref_hint;
					}
				}
			}

			if (second_forwards_idx < 0) {
				std_frame_header.skip_mode_allowed = false;
			} else {
				std_frame_header.skip_mode_allowed = true;
				std_frame_header.skip_mode_frame[0] = VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME + MIN(forwards_idx, second_forwards_idx);
				std_frame_header.skip_mode_frame[1] = VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME + MAX(forwards_idx, second_forwards_idx);
			}
		}
	}

	if (std_frame_header.skip_mode_allowed) {
		std_frame_header.skip_mode_present = read_bits(1);
	} else {
		std_frame_header.skip_mode_present = false;
	}

	if (is_intra || std_frame_header.error_resilient_mode_flag || !av1_sequence_header.enable_warped_motion_flag) {
		std_frame_header.allow_warped_motion = false;
	} else {
		std_frame_header.allow_warped_motion = read_bits(1);
	}

	std_frame_header.reduced_tx_set = read_bits(1);

	parse_global_motion(is_intra, std_frame_header.allow_high_precision_mv, &std_frame_header.global_motion);
	parse_film_grain(show_frame, showable_frame, std_frame_header.frame_type, &std_frame_header.film_grain);

	return std_frame_header;
}

VideoDecodeAV1Frame VideoStreamAV1::parse_frame(uint32_t p_size) {
	VideoDecodeAV1Frame frame = parse_frame_header();
	if (shift != 0) {
		src += 1;
		shift = 0;
	}

	// Finalize the decode process by updating references.
	for (size_t i = 0; i < VIDEO_CODING_AV1_NUM_REF_FRAMES; i++) {
		if ((frame.refresh_frame_flags & (1 << i)) == 0) {
			continue;
		}

		ref_valid[i] = true;
		ref_frame_id[i] = frame.current_frame_id;
		//ref_upscaled_width[i] = 67;
		//ref_frame_width[i] = 67;
		//ref_frame_height[i] = 67;
		//ref_render_width[i] = 67;
		//ref_render_height[i] = 67;
		//ref_mi_cols[i] = 67;
		//ref_mi_rows[i] = 67;
		ref_frame_types[i] = frame.frame_type;
		ref_order_hint[i] = frame.order_hint;
		//ref_subsampling_x = 67;
		//ref_subsampling_y = 67;
		//ref_bit_depth = 67;
		//! saved_order_hints = ????;
		//frame_store = 67;
		//frame_store = 67;
		//saved_ref_frames = 67;
		//saved_mvs = 67
		//! saved_gm_params[i] = ????;
		//! saved_segement_ids[i] = ?????;
		// save_cdfs()
		// save_grain_params()

		// save_loop_filter_params()
		loop_filter_mode_deltas[i][0] = frame.loop_filter.loop_filter_mode_deltas[0];
		loop_filter_mode_deltas[i][1] = frame.loop_filter.loop_filter_mode_deltas[1];
		for (uint8_t j = 0; j < VIDEO_CODING_AV1_TOTAL_REFS_PER_FRAME; j++) {
			loop_filter_ref_deltas[i][j] = frame.loop_filter.loop_filter_ref_deltas[j];
		}

		// save_segmentation_params()
		for (uint8_t segment = 0; segment < VIDEO_CODING_AV1_MAX_SEGMENTS; segment++) {
			segmentation_feature_enabled[i][segment] = frame.segmentation.feature_enabled[segment];
			for (uint8_t lvl = 0; lvl < VIDEO_CODING_AV1_SEG_LVL_MAX; lvl++) {
				segmentation_feature_data[i][segment][lvl] = frame.segmentation.feature_data[segment][lvl];
			}
		}
	}

	return frame;
}

void VideoStreamAV1::parse_tile_info(uint32_t p_mi_cols, uint32_t p_mi_rows, VideoCodingAV1TileInfo *r_tile_info) {
	uint32_t sb_cols;
	uint32_t sb_rows;
	uint8_t sb_shift;

	if (av1_sequence_header.use_128x128_superblock_flag) {
		sb_cols = (p_mi_cols + 31) >> 5;
		sb_rows = (p_mi_rows + 31) >> 5;
		sb_shift = 5;
	} else {
		sb_cols = (p_mi_cols + 15) >> 4;
		sb_rows = (p_mi_rows + 15) >> 4;
		sb_shift = 4;
	}

	uint8_t sb_size = sb_shift + 2;
	uint32_t max_tile_width_sb = VIDEO_CODING_AV1_MAX_TILE_WIDTH >> sb_size;
	uint32_t max_tile_area_sb = VIDEO_CODING_AV1_MAX_TILE_AREA >> (2 * sb_size);
	uint8_t min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
	uint8_t max_log2_tile_cols = tile_log2(1, MIN(sb_cols, VIDEO_CODING_AV1_MAX_TILE_COLS));
	uint8_t max_log2_tile_rows = tile_log2(1, MIN(sb_rows, VIDEO_CODING_AV1_MAX_TILE_ROWS));
	uint8_t min_log2_tiles = MAX(min_log2_tile_cols, tile_log2(max_tile_area_sb, sb_rows * sb_cols));

	uint8_t tile_cols_log2;
	uint8_t tile_rows_log2;

	r_tile_info->uniform_tile_spacing_flag = read_bits(1);
	if (r_tile_info->uniform_tile_spacing_flag) {
		tile_cols_log2 = min_log2_tile_cols;
		while (tile_cols_log2 < max_log2_tile_cols) {
			bool increment_tile_cols_log2 = read_bits(1);
			if (increment_tile_cols_log2) {
				tile_cols_log2++;
			} else {
				break;
			}
		}

		uint32_t tile_width_sb = (sb_cols + (1 << tile_cols_log2) - 1) >> tile_cols_log2;
		for (uint32_t start_sb = 0; start_sb < sb_cols; start_sb += tile_width_sb) {
			r_tile_info->mi_col_starts.push_back(start_sb << sb_shift);
			r_tile_info->width_in_sbs_minus_1.push_back(tile_width_sb - 1);
		}

		r_tile_info->tile_cols = r_tile_info->mi_col_starts.size();
		r_tile_info->mi_col_starts.push_back(p_mi_cols);

		tile_rows_log2 = MAX(min_log2_tiles - tile_cols_log2, 0);
		while (tile_rows_log2 < max_log2_tile_rows) {
			bool increment_tile_rows_log2 = read_bits(1);
			if (increment_tile_rows_log2) {
				tile_rows_log2++;
			} else {
				break;
			}
		}

		uint32_t tile_height_sb = (sb_rows + (1 << tile_rows_log2) - 1) >> tile_rows_log2;
		for (uint32_t start_sb = 0; start_sb < sb_rows; start_sb += tile_height_sb) {
			r_tile_info->mi_row_starts.push_back(start_sb << sb_shift);
			r_tile_info->height_in_sbs_minus_1.push_back(tile_height_sb - 1);
		}

		r_tile_info->tile_rows = r_tile_info->mi_row_starts.size();
		r_tile_info->mi_row_starts.push_back(p_mi_rows);
	} else {
		// TODO
		WARN_PRINT("Burn in hell");
		tile_cols_log2 = 67;
		tile_rows_log2 = 67;
	}

	if (tile_cols_log2 > 0 || tile_rows_log2 > 0) {
		r_tile_info->context_update_tile_id = read_bits(tile_cols_log2 + tile_rows_log2);
		r_tile_info->tile_size_bytes_minus_1 = read_bits(2);
	} else {
		r_tile_info->context_update_tile_id = 0;
	}
}

void VideoStreamAV1::parse_quantization(VideoCodingAV1Quantization *r_quantization) {
	r_quantization->base_q_idx = read_bits(8);

	r_quantization->delta_q_y_dc = read_delta_q();
	if (!av1_sequence_header.color_config.monochrome_flag) {
		if (av1_sequence_header.color_config.separate_uv_delta_q) {
			r_quantization->diff_uv_delta = read_bits(1);
		} else {
			r_quantization->diff_uv_delta = false;
		}

		r_quantization->delta_q_u_dc = read_delta_q();
		r_quantization->delta_q_u_ac = read_delta_q();

		if (r_quantization->diff_uv_delta) {
			r_quantization->delta_q_v_dc = read_delta_q();
			r_quantization->delta_q_v_ac = read_delta_q();
		} else {
			r_quantization->delta_q_v_dc = r_quantization->delta_q_u_dc;
			r_quantization->delta_q_v_ac = r_quantization->delta_q_u_ac;
		}
	} else {
		r_quantization->delta_q_u_dc = 0;
		r_quantization->delta_q_u_ac = 0;
		r_quantization->delta_q_v_dc = 0;
		r_quantization->delta_q_v_ac = 0;
	}

	r_quantization->using_qmatrix = read_bits(1);
	if (r_quantization->using_qmatrix) {
		r_quantization->qm_y = read_bits(4);
		r_quantization->qm_u = read_bits(4);

		if (!av1_sequence_header.color_config.separate_uv_delta_q) {
			r_quantization->qm_v = r_quantization->qm_u;
		} else {
			r_quantization->qm_v = read_bits(4);
		}
	}
}

void VideoStreamAV1::parse_segmentation(uint8_t p_primary_ref_frame, VideoCodingAV1Segmentation *r_segmentation) {
	r_segmentation->segmentation_enabled = read_bits(1);
	if (r_segmentation->segmentation_enabled) {
		if (p_primary_ref_frame == VIDEO_CODING_AV1_PRIMARY_REF_NONE) {
			r_segmentation->segmentation_update_map = true;
			r_segmentation->segmentation_temporal_update = false;
			r_segmentation->segmentation_update_data = true;
		} else {
			r_segmentation->segmentation_update_map = read_bits(1);
			if (r_segmentation->segmentation_update_map) {
				r_segmentation->segmentation_temporal_update = read_bits(1);
			}

			r_segmentation->segmentation_update_data = read_bits(1);
		}

		if (r_segmentation->segmentation_update_data) {
			for (size_t i = 0; i < VIDEO_CODING_AV1_MAX_SEGMENTS; i++) {
				r_segmentation->feature_enabled[i] = 0;
				for (size_t j = 0; j < VIDEO_CODING_AV1_SEG_LVL_MAX; j++) {
					int64_t feature_value = 0;
					bool feature_enabled = read_bits(1);

					if (feature_enabled) {
						r_segmentation->feature_enabled[i] |= (1 << j);
						uint8_t size = video_coding_av1_segmentation_bits[j];
						int8_t limit = video_coding_av1_segmentation_feature_max[j];

						if (video_coding_av1_segmentation_signed[j]) {
							feature_value = read_su(size + 1);
							r_segmentation->feature_data[i][j] = CLAMP(feature_value, -limit, limit);
						} else {
							feature_value = read_bits(size);
							r_segmentation->feature_data[i][j] = CLAMP(feature_value, 0, limit);
						}
					}
				}
			}
		}
	} else {
		for (uint64_t i = 0; i < VIDEO_CODING_AV1_MAX_SEGMENTS; i++) {
			for (uint64_t j = 0; j < VIDEO_CODING_AV1_SEG_LVL_MAX; j++) {
				r_segmentation->feature_enabled[i] = 0;
				r_segmentation->feature_data[i][j] = 0;
			}
		}
	}
}

void VideoStreamAV1::parse_loop_filter(bool p_coded_lossless, bool p_allow_intrabc, VideoCodingAV1LoopFilter *r_loop_filter) {
	if (p_coded_lossless || p_allow_intrabc) {
		r_loop_filter->loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_INTRA_FRAME] = 1;
		r_loop_filter->loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME] = 0;
		r_loop_filter->loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_LAST2_FRAME] = 0;
		r_loop_filter->loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_LAST3_FRAME] = 0;
		r_loop_filter->loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_BWDREF_FRAME] = 0;
		r_loop_filter->loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_GOLDEN_FRAME] = -1;
		r_loop_filter->loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF_FRAME] = -1;
		r_loop_filter->loop_filter_ref_deltas[VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF2_FRAME] = -1;
		for (uint32_t i = 0; i < 2; i++) {
			r_loop_filter->loop_filter_level[i] = 0;
			r_loop_filter->loop_filter_mode_deltas[i] = 0;
		}
	} else {
		r_loop_filter->loop_filter_level[0] = read_bits(6);
		r_loop_filter->loop_filter_level[1] = read_bits(6);
		if (!av1_sequence_header.color_config.monochrome_flag) {
			if (r_loop_filter->loop_filter_level[0] || r_loop_filter->loop_filter_level[1]) {
				r_loop_filter->loop_filter_level[2] = read_bits(6);
				r_loop_filter->loop_filter_level[3] = read_bits(6);
			}
		}

		r_loop_filter->loop_filter_sharpness = read_bits(3);
		r_loop_filter->loop_filter_delta_enabled = read_bits(1);
		if (r_loop_filter->loop_filter_delta_enabled) {
			r_loop_filter->loop_filter_delta_update = read_bits(1);
			if (r_loop_filter->loop_filter_delta_update) {
				r_loop_filter->update_ref_delta = 0;
				for (uint32_t i = 0; i < VIDEO_CODING_AV1_TOTAL_REFS_PER_FRAME; i++) {
					bool update_ref_delta = read_bits(1);
					if (update_ref_delta) {
						r_loop_filter->update_ref_delta |= 1 << i;
						r_loop_filter->loop_filter_ref_deltas[i] = read_su(7);
					}
				}

				r_loop_filter->update_mode_delta = 0;
				for (uint32_t i = 0; i < 2; i++) {
					bool update_mode_delta = read_bits(1);
					if (update_mode_delta) {
						r_loop_filter->update_mode_delta |= 1 << i;
						r_loop_filter->loop_filter_mode_deltas[i] = read_su(7);
					}
				}
			}
		}
	}
}

void VideoStreamAV1::parse_cdef(bool p_coded_lossless, bool p_allow_intrabc, VideoCodingAV1CDEF *r_cdef) {
	if (p_coded_lossless || p_allow_intrabc || !av1_sequence_header.enable_cdef_flag) {
		r_cdef->cdef_bits = 0;
		r_cdef->cdef_y_pri_strength[0] = 0;
		r_cdef->cdef_y_sec_strength[0] = 0;
		r_cdef->cdef_uv_pri_strength[0] = 0;
		r_cdef->cdef_uv_sec_strength[0] = 0;
		r_cdef->cdef_damping_minus_3 = 0;
	} else {
		r_cdef->cdef_damping_minus_3 = read_bits(2);
		r_cdef->cdef_bits = read_bits(2);
		for (uint32_t i = 0; i < (1 << r_cdef->cdef_bits); i++) {
			r_cdef->cdef_y_pri_strength[i] = read_bits(4);
			r_cdef->cdef_y_sec_strength[i] = read_bits(2);
			if (r_cdef->cdef_y_sec_strength[i] == 3) {
				r_cdef->cdef_y_sec_strength[i] += 1;
			}

			if (!av1_sequence_header.color_config.monochrome_flag) {
				r_cdef->cdef_uv_pri_strength[i] = read_bits(4);
				r_cdef->cdef_uv_sec_strength[i] = read_bits(2);
				if (r_cdef->cdef_uv_sec_strength[i] == 3) {
					r_cdef->cdef_uv_sec_strength[i] += 1;
				}
			}
		}
	}
}

void VideoStreamAV1::parse_loop_restoration(bool p_all_lossless, bool p_allow_intrabc, VideoCodingAV1LoopRestoration *r_loop_restoration) {
	if (p_all_lossless || p_allow_intrabc || !av1_sequence_header.enable_restoration_flag) {
		for (uint32_t i = 0; i < VIDEO_CODING_AV1_MAX_NUM_PLANES; i++) {
			r_loop_restoration->frame_restoration_type[i] = VIDEO_CODING_AV1_FRAME_RESTORATION_TYPE_NONE;
		}

		r_loop_restoration->uses_lr = false;
	} else {
		r_loop_restoration->uses_lr = false;
		r_loop_restoration->uses_chroma_lr = false;

		for (uint32_t i = 0; i < VIDEO_CODING_AV1_MAX_NUM_PLANES; i++) {
			uint16_t lr_type = read_bits(2);
			r_loop_restoration->frame_restoration_type[i] = video_coding_av1_remap_lr_type[lr_type];
			if (r_loop_restoration->frame_restoration_type[i] != VIDEO_CODING_AV1_FRAME_RESTORATION_TYPE_NONE) {
				r_loop_restoration->uses_lr = true;
				if (i > 0) {
					r_loop_restoration->uses_chroma_lr = true;
				}
			}
		}

		if (r_loop_restoration->uses_lr) {
			uint8_t lr_unit_shift = 0;
			if (av1_sequence_header.use_128x128_superblock_flag) {
				lr_unit_shift = read_bits(1) + 1;
			} else {
				lr_unit_shift = read_bits(1);
				if (lr_unit_shift) {
					lr_unit_shift += read_bits(1);
				}
			}

			r_loop_restoration->loop_restoration_size[0] = VIDEO_CODING_AV1_RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift);

			uint8_t lr_uv_shift = 0;
			if (av1_sequence_header.color_config.subsampling_x && av1_sequence_header.color_config.subsampling_y && r_loop_restoration->uses_chroma_lr) {
				lr_uv_shift = read_bits(1);
			}

			r_loop_restoration->loop_restoration_size[1] = r_loop_restoration->loop_restoration_size[0] >> lr_uv_shift;
			r_loop_restoration->loop_restoration_size[2] = r_loop_restoration->loop_restoration_size[0] >> lr_uv_shift;
		}
	}
}

void VideoStreamAV1::parse_global_motion(bool p_is_intra, bool p_allow_high_precision_mv, VideoCodingAV1GlobalMotion *r_global_motion) {
	for (uint32_t ref = VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME; ref <= VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF_FRAME; ref++) {
		r_global_motion->gm_type[ref] = VIDEO_CODING_AV1_WARP_MODEL_IDENTITY;
		for (uint8_t i = 0; i < 6; i++) {
			if (i % 3 == 2) {
				r_global_motion->gm_params[ref][i] = 1 << VIDEO_CODING_AV1_WARPEDMODEL_PREC_BITS;
			} else {
				r_global_motion->gm_params[ref][i] = 0;
			}
		}
	}

	if (p_is_intra) {
		return;
	}

	for (uint32_t ref = VIDEO_CODING_AV1_REFERENCE_NAME_LAST_FRAME; ref <= VIDEO_CODING_AV1_REFERENCE_NAME_ALTREF_FRAME; ref++) {
		bool is_global = read_bits(1);
		if (is_global) {
			bool is_rot_zoom = read_bits(1);
			if (is_rot_zoom) {
				//TODO type = ROTZOOM
			} else {
				bool is_translation = read_bits(1);
				if (is_translation) {
					//TODO type = TRANSLATION
				} else {
					//TODO type = AFFINE
				}
			}
		} else {
			//TODO type = IDENTITY
		}

		// TODO: skipping a lot of global parameter things
	}
}

// TODO: everything
void VideoStreamAV1::parse_film_grain(bool p_show_frame, bool p_showable_frame, VideoCodingAV1FrameType p_frame_type, VideoCodingAV1FilmGrain *r_film_grain) {
	r_film_grain->apply_grain = false;
	return;

	if (!av1_sequence_header.film_grain_params_present_flag || (!p_show_frame && !p_showable_frame)) {
		// TODO reset grain params
	} else {
		r_film_grain->apply_grain = read_bits(1);
		if (!r_film_grain->apply_grain) {
			// TODO reset grain params
			WARN_PRINT("Reset grain params");
		} else {
			WARN_PRINT("Skipping way too much stuff");
			//uint16_t grain_seed = read_bits(16);
			bool update_grain;
			if (p_frame_type == VIDEO_CODING_AV1_FRAME_TYPE_INTER) {
				update_grain = read_bits(1);
			} else {
				update_grain = true;
			}

			if (!update_grain) {
			}
		}
	}
}

Error VideoStreamAV1::parse_container_metadata(const uint8_t *p_stream, uint64_t p_size) {
	src = p_stream;
	shift = 0;

	read_bits(1); // marker.
	uint8_t version = read_bits(7);
	ERR_FAIL_COND_V(version != 1, OK);

	av1_sequence_header.seq_profile = read_bits(3);
	read_bits(5); // seq_level_idx_0
	read_bits(1); // seq_tier_0

	bool high_bitdepth = read_bits(1) > 0;
	bool twelve_bit = read_bits(1) > 0;
	bool monochrome = read_bits(1) > 0;
	bool chroma_subsampling_x = read_bits(1) > 0;
	bool chroma_subsampling_y = read_bits(1) > 0;
	uint8_t chroma_subsampling_position = read_bits(2);
	read_bits(3); // reserved.

	if (high_bitdepth && twelve_bit) {
		video_profile.chroma_bit_depth = 12;
		video_profile.luma_bit_depth = 12;
	} else if (high_bitdepth && !twelve_bit) {
		video_profile.chroma_bit_depth = 10;
		video_profile.luma_bit_depth = 10;
	} else {
		video_profile.chroma_bit_depth = 8;
		video_profile.luma_bit_depth = 8;
	}

	if (monochrome) {
		video_profile.chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_MONOCHROME;
	} else if (chroma_subsampling_x && chroma_subsampling_y) {
		video_profile.chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_420;
	} else if (chroma_subsampling_x && !chroma_subsampling_y) {
		video_profile.chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_422;
	} else {
		video_profile.chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_444;
	}

	bool initial_presentation_delay_present = read_bits(1) > 0;
	if (initial_presentation_delay_present) {
		read_bits(4); // initial_presentation_delay_minus_one
	} else {
		read_bits(4); // reserved.
	}

	while (src < p_stream + p_size) {
		parse_open_bitstream_unit();
	}

	return OK;
}

Error VideoStreamAV1::parse_container_block(const uint8_t *p_stream, size_t p_size, Vector<size_t> *r_offsets, Vector<size_t> *r_sizes) {
	src = p_stream;
	shift = 0;

	while (src < p_stream + p_size) {
		const uint8_t *obu_start = src;
		bool is_frame = parse_open_bitstream_unit();
		if (is_frame) {
			r_offsets->push_back(obu_start - p_stream);
			r_sizes->push_back(src - obu_start);
		}
	}

	return OK;
}

void VideoStreamAV1::set_rendering_device(RenderingDevice *p_local_device) {
	local_device = p_local_device;
}

RID VideoStreamAV1::create_video_session(RD::VideoSessionInfo p_session_template) {
	p_session_template.profile = video_profile;
	p_session_template.width = av1_sequence_header.max_frame_width_minus_1 + 1;
	p_session_template.height = av1_sequence_header.max_frame_height_minus_1 + 1;
	p_session_template.max_active_reference_pictures = VIDEO_CODING_AV1_NUM_REF_FRAMES - 1;

	video_session = local_device->video_session_create(p_session_template);
	local_device->video_session_add_av1_parameters(video_session, av1_sequence_header);
	return video_session;
}

RID VideoStreamAV1::create_texture_sampler(RD::SamplerState p_sampler_template) {
	// TODO override parameters
	texture_sampler = local_device->sampler_create(p_sampler_template);
	return texture_sampler;
}

RID VideoStreamAV1::create_texture(RD::TextureFormat p_texture_template) {
	// TODO override parameters
	Vector<VideoProfile> video_profiles;
	video_profiles.push_back(video_profile);

	p_texture_template.video_profiles = video_profiles;

	RD::TextureView texture_view;
	texture_view.ycbcr_sampler = texture_sampler;

	return local_device->texture_create(p_texture_template, texture_view);
}

void VideoStreamAV1::decode_frame(Span<uint8_t> p_frame_data, RID p_dst_texture) {
	src = p_frame_data.begin();
	shift = 0;

	read_bits(1); // obu_forbidden_bit
	uint8_t obu_type = read_bits(4);
	bool obu_extension_flag = read_bits(1) > 0;
	bool obu_has_size_field = read_bits(1) > 0;
	read_bits(1); // obu_reserved_1bit

	if (obu_extension_flag) {
		uint8_t temporal_id = read_bits(3);
		uint8_t spacial_id = read_bits(2);
		read_bits(3); // extension_header_reserved_3bits
		print_line("temporal_id", temporal_id);
		print_line("spacial_id", spacial_id);
	}

	if (obu_has_size_field) {
		read_leb128(); // obu_size
	} else {
		ERR_FAIL_MSG("Unknown OBU size, refusing to decode");
	}

	if (obu_type == VIDEO_CODING_AV1_OBU_TYPE_FRAME) {
		VideoDecodeAV1Frame frame_header = parse_frame(p_frame_data.size());
		frame_header.tile_start = src - p_frame_data.begin();
		frame_header.tile_size = p_frame_data.size() - (src - p_frame_data.begin());
		local_device->video_session_decode(video_session, p_frame_data, p_dst_texture, &frame_header);
	} else if (obu_type == VIDEO_CODING_AV1_OBU_TYPE_FRAME_HEADER) {
		// May update __stuff__ ?
		print_line("Parsing frame header");
		fflush(stdout);
		parse_frame_header();
	}
}

VideoStreamAV1::VideoStreamAV1() {
	local_device = RD::get_singleton();

	video_profile.operation = VIDEO_OPERATION_DECODE_AV1;
}
