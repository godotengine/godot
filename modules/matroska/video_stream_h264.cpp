/**************************************************************************/
/*  video_stream_h264.cpp                                                 */
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

#include "video_stream_h264.h"

#include "core/error/error_macros.h"
#include "core/string/print_string.h"
#include "core/variant/variant.h"
#include "servers/rendering/rendering_device_commons.h"
#include "servers/rendering/video/video_coding_common.h"
#include "servers/rendering/video/video_coding_h264.h"

uint64_t VideoStreamH264::read_bits(uint8_t p_amount) {
	uint8_t bits = p_amount % 8;
	uint8_t bytes = p_amount / 8;
	uint32_t emulation_prevention_mask = 0xffffff;

	uint64_t encoded = 0;
	for (uint8_t i = 0; i < bytes; i++) {
		encoded = (encoded << 8) | src[i];

		if ((encoded & emulation_prevention_mask) == 3) {
			encoded = encoded >> 8;
			src += 1;
			i--;
		}
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
	uint8_t end = start + 1 - p_amount;
	shift = (shift + bits) % 8;

	uint64_t value = 0;
	for (uint8_t offset = start + 1; offset > end; offset--) {
		value |= encoded & (1ULL << (offset - 1));
	}

	value >>= end;
	return value;
}

uint64_t VideoStreamH264::read_ue() {
	uint64_t bits = 0;
	while (read_bits(1) == 0) {
		bits += 1;
	}

	uint64_t rest = read_bits(bits);
	return (1ULL << bits) + rest - 1;
}

int64_t VideoStreamH264::read_se() {
	int64_t code = read_ue();
	if (code % 2 == 0) {
		return -code / 2;
	} else {
		return -(code + 1) / 2;
	}
}

VideoCodingH264NalUnitType VideoStreamH264::parse_nal_unit(uint64_t p_size, VideoDecodeH264SliceHeader *r_h264_slice_header) {
	const uint8_t *nal_start = src;

	uint8_t header = read_bits(8);
	uint8_t nal_ref_idc = (header & 0b1100000) >> 5;
	VideoCodingH264NalUnitType nal_unit_type = VideoCodingH264NalUnitType(header & 0b11111);

	switch (nal_unit_type) {
		case VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE: {
			print_line(vformat("Standard Frame [%d]", p_size));
			*r_h264_slice_header = parse_slice_header(p_size - 1, nal_ref_idc != 0, false);
		} break;

		case VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE_IDR: {
			print_line(vformat("IDR Frame [%d]", p_size));
			*r_h264_slice_header = parse_slice_header(p_size - 1, nal_ref_idc != 0, true);
		} break;

		case VIDEO_CODING_H264_NAL_UNIT_TYPE_SUPPLEMENTAL_ENHACEMENT_INFORMATION: {
			print_line(vformat("Supplemental Information [%d]", p_size));
		} break;

		case VIDEO_CODING_H264_NAL_UNIT_TYPE_SEQUENCE_PARAMETER_SET: {
			print_line(vformat("Sequence Parameter Set [%d]", p_size));
			VideoCodingH264SequenceParameterSet sps = parse_sequence_parameter_set(p_size - 1);
			if (sps.seq_parameter_set_id < sps_sets.size()) {
				sps_sets.set(sps.seq_parameter_set_id, sps);
			} else {
				sps_sets.insert(sps.seq_parameter_set_id, sps);
			}
		} break;

		case VIDEO_CODING_H264_NAL_UNIT_TYPE_PICTURE_PARAMETER_SET: {
			print_line(vformat("Picture Parameter Set [%d]", p_size));
			VideoCodingH264PictureParameterSet pps = parse_picture_parameter_set(p_size - 1);
			if (pps.pic_parameter_set_id < pps_sets.size()) {
				pps_sets.set(pps.pic_parameter_set_id, pps);
			} else {
				pps_sets.insert(pps.pic_parameter_set_id, pps);
			}
		} break;

		default: {
			print_line(vformat("Unknown NAL Unit (%d) [%d]", nal_unit_type, p_size));
		}
	}

	src = nal_start + p_size;
	shift = 0;

	return nal_unit_type;
}

VideoCodingH264SequenceParameterSet VideoStreamH264::parse_sequence_parameter_set(uint64_t p_size) {
	VideoCodingH264SequenceParameterSet sequence_parameter_set = {};

	sequence_parameter_set.profile_idc = VideoCodingH264ProfileIdc(read_bits(8));

	uint8_t flags = read_bits(8);
	sequence_parameter_set.constraint_set0_flag = (flags & (1 << 7)) > 0;
	sequence_parameter_set.constraint_set1_flag = (flags & (1 << 6)) > 0;
	sequence_parameter_set.constraint_set2_flag = (flags & (1 << 5)) > 0;
	sequence_parameter_set.constraint_set3_flag = (flags & (1 << 4)) > 0;
	sequence_parameter_set.constraint_set4_flag = (flags & (1 << 3)) > 0;
	sequence_parameter_set.constraint_set5_flag = (flags & (1 << 2)) > 0;

	sequence_parameter_set.level_idc = read_bits(8);

	sequence_parameter_set.seq_parameter_set_id = read_ue();

	if (sequence_parameter_set.profile_idc == VIDEO_CODING_H264_PROFILE_IDC_HIGH || sequence_parameter_set.profile_idc == VIDEO_CODING_H264_PROFILE_IDC_HIGH_PREDICTIVE) {
		uint8_t chroma_format = read_ue();
		if (chroma_format == 0) {
			sequence_parameter_set.chroma_format_idc = VIDEO_CODING_H264_CHROMA_FORMAT_IDC_MONOCHROME;
		} else if (chroma_format == 1) {
			sequence_parameter_set.chroma_format_idc = VIDEO_CODING_H264_CHROMA_FORMAT_IDC_420;
		} else if (chroma_format == 2) {
			sequence_parameter_set.chroma_format_idc = VIDEO_CODING_H264_CHROMA_FORMAT_IDC_422;
		} else if (chroma_format == 3) {
			sequence_parameter_set.chroma_format_idc = VIDEO_CODING_H264_CHROMA_FORMAT_IDC_444;
		}

		if (sequence_parameter_set.chroma_format_idc == VIDEO_CODING_H264_CHROMA_FORMAT_IDC_444) {
			sequence_parameter_set.separate_colour_plane_flag = read_bits(1) > 0;
		}

		sequence_parameter_set.bit_depth_luma_minus8 = read_ue();
		sequence_parameter_set.bit_depth_chroma_minus8 = read_ue();

		sequence_parameter_set.qpprime_y_zero_transform_bypass_flag = read_bits(1) > 0;

		sequence_parameter_set.seq_scaling_matrix_present_flag = read_bits(1) > 0;
		if (sequence_parameter_set.seq_scaling_matrix_present_flag) {
			uint64_t size = sequence_parameter_set.chroma_format_idc != VIDEO_CODING_H264_CHROMA_FORMAT_IDC_444 ? 8 : 12;
			for (uint64_t i = 0; i < size; i++) {
				uint8_t present = read_bits(1) > 0;
				sequence_parameter_set.scaling_lists.scaling_list_present_mask |= present << i;
				if (present) {
					if (i < 6) {
						// TODO
						print_line("skipping 4x4 seq scaling lists");
					} else {
						print_line("skipping 8x8 seq scaling lists");
					}
				}
			}
		}
	}

	sequence_parameter_set.log2_max_frame_num_minus4 = read_ue();
	sequence_parameter_set.pic_order_cnt_type = VideoCodingH264PocType(read_ue());

	if (sequence_parameter_set.pic_order_cnt_type == VIDEO_CODING_H264_POC_TYPE_0) {
		sequence_parameter_set.log2_max_pic_order_cnt_lsb_minus4 = read_ue();
	} else if (sequence_parameter_set.pic_order_cnt_type == VIDEO_CODING_H264_POC_TYPE_1) {
		sequence_parameter_set.delta_pic_order_always_zero_flag = read_bits(1) > 0;

		sequence_parameter_set.offset_for_non_ref_pic = read_se();
		sequence_parameter_set.offset_for_top_to_bottom_field = read_se();

		sequence_parameter_set.num_ref_frames_in_pic_order_cnt_cycle = read_ue();

		sequence_parameter_set.offset_for_ref_frame.clear();
		for (uint8_t i = 0; i < sequence_parameter_set.num_ref_frames_in_pic_order_cnt_cycle; i++) {
			int32_t offset_for_ref_frame = read_se();
			sequence_parameter_set.offset_for_ref_frame.push_back(offset_for_ref_frame);
		}
	}

	sequence_parameter_set.max_num_ref_frames = read_ue();

	sequence_parameter_set.gaps_in_frame_num_value_allowed_flag = read_bits(1) > 0;

	sequence_parameter_set.pic_width_in_mbs_minus1 = read_ue();
	sequence_parameter_set.pic_height_in_map_units_minus1 = read_ue();

	sequence_parameter_set.frame_mbs_only_flag = read_bits(1) > 0;
	if (!sequence_parameter_set.frame_mbs_only_flag) {
		sequence_parameter_set.mb_adaptive_frame_field_flag = read_bits(1) > 0;
	}

	sequence_parameter_set.direct_8x8_inference_flag = read_bits(1) > 0;

	sequence_parameter_set.frame_cropping_flag = read_bits(1) > 0;
	if (sequence_parameter_set.frame_cropping_flag) {
		sequence_parameter_set.frame_crop_left_offset = read_ue();
		sequence_parameter_set.frame_crop_right_offset = read_ue();
		sequence_parameter_set.frame_crop_top_offset = read_ue();
		sequence_parameter_set.frame_crop_bottom_offset = read_ue();
	}

	sequence_parameter_set.vui_parameters_present_flag = read_bits(1) > 0;
	if (sequence_parameter_set.vui_parameters_present_flag) {
		sequence_parameter_set.vui.aspect_ratio_info_present_flag = read_bits(1) > 0;
		if (sequence_parameter_set.vui.aspect_ratio_info_present_flag) {
			sequence_parameter_set.vui.aspect_ratio_idc = VideoCodingH264AspectRatioIdc(read_bits(8));
			if (sequence_parameter_set.vui.aspect_ratio_idc == VIDEO_CODING_H264_ASPECT_RATIO_IDC_EXTENDED_SAR) {
				sequence_parameter_set.vui.sar_width = read_bits(16);
				sequence_parameter_set.vui.sar_height = read_bits(16);
			}
		}

		sequence_parameter_set.vui.overscan_info_present_flag = read_bits(1) > 0;
		if (sequence_parameter_set.vui.overscan_info_present_flag) {
			sequence_parameter_set.vui.overscan_appropriate_flag = read_bits(1) > 0;
		}

		sequence_parameter_set.vui.video_signal_type_present_flag = read_bits(1) > 0;
		if (sequence_parameter_set.vui.video_signal_type_present_flag) {
			sequence_parameter_set.vui.video_format = read_bits(3);

			sequence_parameter_set.vui.video_full_range_flag = read_bits(1) > 0;

			sequence_parameter_set.vui.color_description_present_flag = read_bits(1) > 0;
			if (sequence_parameter_set.vui.color_description_present_flag) {
				sequence_parameter_set.vui.colour_primaries = read_bits(8);
				sequence_parameter_set.vui.transfer_characteristics = read_bits(8);
				sequence_parameter_set.vui.matrix_coefficients = read_bits(8);
			}
		}

		sequence_parameter_set.vui.chroma_loc_info_present_flag = read_bits(1) > 0;
		if (sequence_parameter_set.vui.chroma_loc_info_present_flag) {
			sequence_parameter_set.vui.chroma_sample_loc_type_top_field = read_ue();
			sequence_parameter_set.vui.chroma_sample_loc_type_bottom_field = read_ue();
		}

		sequence_parameter_set.vui.timing_info_present_flag = read_bits(1) > 0;
		if (sequence_parameter_set.vui.timing_info_present_flag) {
			sequence_parameter_set.vui.num_units_in_tick = read_bits(32);
			sequence_parameter_set.vui.time_scale = read_bits(32);
			sequence_parameter_set.vui.fixed_frame_rate_flag = read_bits(1) > 0;
		}

		sequence_parameter_set.vui.nal_hrd_parameters_present_flag = read_bits(1) > 0;
		if (sequence_parameter_set.vui.nal_hrd_parameters_present_flag) {
			print_line("skipping nal_hrd_parameters_present_flag");
		}

		sequence_parameter_set.vui.vcl_hrd_parameters_present_flag = read_bits(1) > 0;
		if (sequence_parameter_set.vui.vcl_hrd_parameters_present_flag) {
			print_line("skipping vcl_hrd_parameters_present_flag");
		}

		if (sequence_parameter_set.vui.nal_hrd_parameters_present_flag || sequence_parameter_set.vui.vcl_hrd_parameters_present_flag) {
			read_bits(1); // low_delay_hrd_flag
		}

		read_bits(1); // pic_struct_present_flag
		sequence_parameter_set.vui.bitstream_restriction_flag = read_bits(1) > 0;
		if (sequence_parameter_set.vui.bitstream_restriction_flag) {
			read_bits(1); // motion_vectors_over_pic_boundaries_flag

			read_ue(); // max_bytes_per_pic_denom
			read_ue(); // max_bits_per_mb_denom

			read_ue(); // log2_max_mv_length_horizontal
			read_ue(); // log2_max_mv_length_vertical

			sequence_parameter_set.vui.max_num_reorder_frames = read_ue();
			sequence_parameter_set.vui.max_dec_frame_buffering = read_ue();
		}
	}

	return sequence_parameter_set;
}

VideoCodingH264PictureParameterSet VideoStreamH264::parse_picture_parameter_set(uint64_t p_size) {
	const uint8_t *pps_start = src;
	VideoCodingH264PictureParameterSet picture_parameter_set = {};

	picture_parameter_set.pic_parameter_set_id = read_ue();
	picture_parameter_set.seq_parameter_set_id = read_ue();

	picture_parameter_set.entropy_coding_mode_flag = read_bits(1) > 0;
	picture_parameter_set.bottom_field_pic_order_in_frame_present_flag = read_bits(1) > 0;

	uint64_t num_slice_groups_minus1 = read_ue();
	if (num_slice_groups_minus1 > 0) {
		// TODO
		print_line("skipping slice groups");
	}

	picture_parameter_set.num_ref_idx_l0_default_active_minus1 = read_ue();
	picture_parameter_set.num_ref_idx_l1_default_active_minus1 = read_ue();

	picture_parameter_set.weighted_pred_flag = read_bits(1) > 0;
	picture_parameter_set.weighted_bipred_idc = VideoCodingH264WeightedBipredIdc(read_bits(2));

	picture_parameter_set.pic_init_qp_minus26 = read_se();
	picture_parameter_set.pic_init_qs_minus26 = read_se();
	picture_parameter_set.chroma_qp_index_offset = read_se();

	picture_parameter_set.deblocking_filter_control_present_flag = read_bits(1) > 0;
	picture_parameter_set.constrained_intra_pred_flag = read_bits(1) > 0;
	picture_parameter_set.redundant_pic_cnt_present_flag = read_bits(1) > 0;

	if (src < pps_start + p_size) {
		picture_parameter_set.transform_8x8_mode_flag = read_bits(1) > 0;

		picture_parameter_set.pic_scaling_matrix_present_flag = read_bits(1) > 0;
		if (picture_parameter_set.pic_scaling_matrix_present_flag) {
			// TODO scaling lists
		}

		picture_parameter_set.second_chroma_qp_index_offset = read_se();
	}

	return picture_parameter_set;
}

VideoDecodeH264SliceHeader VideoStreamH264::parse_slice_header(uint64_t p_size, bool p_is_reference, bool p_is_idr) {
	VideoDecodeH264SliceHeader slice_header = {};

	slice_header.is_reference = p_is_reference;
	slice_header.is_intra = p_is_idr;

	// TODO support interlaced video
	slice_header.complementary_field_pair = false;

	read_ue(); // first_mb_in_slice

	uint64_t slice_type = read_ue();
	switch (slice_type) {
		case 0: {
			print_line("slice_type = P");
		} break;

		case 1: {
			print_line("slice_type = B");
		} break;

		case 2: {
			print_line("slice_type = I");
		} break;

		case 3: {
			print_line("slice_type = SP");
		} break;

		case 4: {
			print_line("slice_type = SI");
		} break;

		case 5: {
			print_line("slice_type = P (Only)");
		} break;

		case 6: {
			print_line("slice_type = B (Only)");
		} break;

		case 7: {
			print_line("slice_type = I (Only)");
		} break;

		case 8: {
			print_line("slice_type = SP (Only)");
		} break;

		case 9: {
			print_line("slice_type = SI (Only)");
		} break;

		default: {
			print_line(vformat("Unknown slice type %d", slice_type));
		}
	}

	VideoCodingH264PictureParameterSet active_pps = {};
	VideoCodingH264SequenceParameterSet active_sps = {};

	slice_header.pic_parameter_set_id = read_ue();
	for (VideoCodingH264PictureParameterSet potential_pps : pps_sets) {
		if (potential_pps.pic_parameter_set_id == slice_header.pic_parameter_set_id) {
			active_pps = potential_pps;
			break;
		}
	}

	slice_header.seq_parameter_set_id = active_pps.seq_parameter_set_id;
	for (VideoCodingH264SequenceParameterSet potential_sps : sps_sets) {
		if (potential_sps.seq_parameter_set_id == active_pps.seq_parameter_set_id) {
			active_sps = potential_sps;
			break;
		}
	}

	if (active_sps.separate_colour_plane_flag) {
		read_bits(2); // colour_plane_id
	}

	uint64_t frame_num_size = active_sps.log2_max_frame_num_minus4 + 4;
	slice_header.frame_num = read_bits(frame_num_size);
	print_line(vformat("frame number %d", slice_header.frame_num));

	if (!active_sps.frame_mbs_only_flag) {
		slice_header.field_pic_flag = read_bits(1) > 0;
		if (slice_header.field_pic_flag) {
			slice_header.bottom_field_flag = read_bits(1) > 0;
		}
	}

	if (p_is_idr) {
		slice_header.idr_pic_id = read_ue();
		prev_pic_order_cnt_lsb = 0;
		prev_pic_order_cnt_msb = 0;
		prev_frame_num_offset = 0;
		prev_frame_num = 0;
	}

	uint64_t pic_order_cnt_msb = 0;
	uint32_t pic_order_cnt_lsb = 0;
	uint64_t max_pic_order_cnt_lsb = 1ULL << (active_sps.log2_max_pic_order_cnt_lsb_minus4 + 4);

	int64_t delta_pic_order_cnt[2] = { 0, 0 };

	if (active_sps.pic_order_cnt_type == VIDEO_CODING_H264_POC_TYPE_0) {
		uint64_t pic_order_cnt_lsb_size = active_sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
		pic_order_cnt_lsb = read_bits(pic_order_cnt_lsb_size);
		print_line(vformat("pic order cnt lsb %d", pic_order_cnt_lsb));

		if (active_pps.bottom_field_pic_order_in_frame_present_flag && !slice_header.field_pic_flag) {
			read_se(); // delta_pic_order_cnt_bottom
		}
	} else if (active_sps.pic_order_cnt_type == VIDEO_CODING_H264_POC_TYPE_1 && !active_sps.delta_pic_order_always_zero_flag) {
		delta_pic_order_cnt[0] = read_se();

		if (active_pps.bottom_field_pic_order_in_frame_present_flag && !slice_header.field_pic_flag) {
			delta_pic_order_cnt[1] = read_se();
		}
	}

	if (active_sps.pic_order_cnt_type == VIDEO_CODING_H264_POC_TYPE_0) {
		if (pic_order_cnt_lsb < prev_pic_order_cnt_lsb && prev_pic_order_cnt_lsb - pic_order_cnt_lsb >= (max_pic_order_cnt_lsb / 2)) {
			pic_order_cnt_msb = prev_pic_order_cnt_msb + max_pic_order_cnt_lsb;
		} else if (pic_order_cnt_lsb > prev_pic_order_cnt_lsb && pic_order_cnt_lsb - prev_pic_order_cnt_lsb > (max_pic_order_cnt_lsb / 2)) {
			pic_order_cnt_msb = prev_pic_order_cnt_msb - max_pic_order_cnt_lsb;
		} else {
			pic_order_cnt_msb = prev_pic_order_cnt_msb;
		}

		slice_header.pic_order_cnt_top_field = pic_order_cnt_msb + pic_order_cnt_lsb;
		slice_header.pic_order_cnt_bottom_field = pic_order_cnt_msb + pic_order_cnt_lsb;

		prev_pic_order_cnt_lsb = pic_order_cnt_lsb;
		prev_pic_order_cnt_msb = pic_order_cnt_msb;
	} else if (active_sps.pic_order_cnt_type == VIDEO_CODING_H264_POC_TYPE_1) {
		uint64_t frame_num_offset;
		if (p_is_idr) {
			frame_num_offset = 0;
		} else if (prev_frame_num_offset > slice_header.frame_num) {
			frame_num_offset = prev_frame_num_offset + frame_num_size;
		} else {
			frame_num_offset = prev_frame_num_offset;
		}

		uint64_t abs_frame_num = 0;
		if (active_sps.num_ref_frames_in_pic_order_cnt_cycle != 0) {
			abs_frame_num = frame_num_offset + slice_header.frame_num;
		} else {
			abs_frame_num = 0;
		}

		if (!p_is_reference && abs_frame_num > 0) {
			abs_frame_num -= 1;
		}

		uint64_t expected_pic_order_cnt = 0;
		if (abs_frame_num > 0) {
			uint64_t pic_order_cnt_cycle_cnt = (abs_frame_num - 1) / active_sps.num_ref_frames_in_pic_order_cnt_cycle;
			uint64_t frame_num_in_pic_order_cnt_cycle = (abs_frame_num - 1) % active_sps.num_ref_frames_in_pic_order_cnt_cycle;

			uint64_t expected_delta_per_pic_order_cnt_cycle = 0;
			for (uint64_t i = 0; i < active_sps.num_ref_frames_in_pic_order_cnt_cycle; i++) {
				expected_delta_per_pic_order_cnt_cycle = active_sps.offset_for_ref_frame[i];
			}

			expected_pic_order_cnt = pic_order_cnt_cycle_cnt * expected_delta_per_pic_order_cnt_cycle;
			for (uint64_t i = 0; i < frame_num_in_pic_order_cnt_cycle; i++) {
				expected_pic_order_cnt += active_sps.offset_for_ref_frame[i];
			}
		} else {
			expected_pic_order_cnt = 0;
		}

		if (!p_is_reference) {
			expected_pic_order_cnt += active_sps.offset_for_non_ref_pic;
		}

		slice_header.pic_order_cnt_top_field = expected_pic_order_cnt + delta_pic_order_cnt[0];
		slice_header.pic_order_cnt_bottom_field = slice_header.pic_order_cnt_top_field + delta_pic_order_cnt[1] + active_sps.offset_for_top_to_bottom_field;

		prev_frame_num_offset = frame_num_offset;
	} else if (active_sps.pic_order_cnt_type == VIDEO_CODING_H264_POC_TYPE_2) {
		uint64_t frame_num_offset;
		if (p_is_idr) {
			frame_num_offset = 0;
		} else if (prev_frame_num_offset > slice_header.frame_num) {
			frame_num_offset = prev_frame_num_offset + frame_num_size;
		} else {
			frame_num_offset = prev_frame_num_offset;
		}

		uint64_t tmp_pic_order_cnt;
		if (p_is_idr) {
			tmp_pic_order_cnt = 0;
		} else if (!p_is_reference) {
			tmp_pic_order_cnt = 2 * (frame_num_offset + slice_header.frame_num) - 1;
		} else {
			tmp_pic_order_cnt = 2 * (frame_num_offset + slice_header.frame_num);
		}

		// TODO does leaving it like this break interlaced video?
		slice_header.pic_order_cnt_top_field = tmp_pic_order_cnt;
		slice_header.pic_order_cnt_bottom_field = tmp_pic_order_cnt;

		prev_frame_num_offset = frame_num_offset;
	}

	print_line("pic order cnt", slice_header.pic_order_cnt_top_field);
	return slice_header;
}

// The Matroska "codec private" data for H264 is an AVCDecoderConfigurationRecord
void VideoStreamH264::parse_container_metadata(const uint8_t *p_stream, uint64_t p_size) {
	src = p_stream;
	shift = 0;

	uint8_t configuration_version = read_bits(8);
	ERR_FAIL_COND_MSG(configuration_version > 1, vformat("AVCDecoderConfigurationRecord version (%d) is greater than 1", configuration_version));

	target_profile_idc = VideoCodingH264ProfileIdc(read_bits(8));
	minimum_profile_idc = VideoCodingH264ProfileIdc(read_bits(8));

	video_profile.operation = VIDEO_OPERATION_DECODE_H264;
	video_profile.h264_profile_idc = target_profile_idc;
	video_profile.h264_picture_layout = VIDEO_CODING_H264_PICTURE_LAYOUT_PROGRESSIVE;

	target_level_idc = read_bits(8);

	length_size = (read_bits(8) & 0b11) + 1;

	uint8_t sps_set_count = read_bits(8) & 0b11111;
	for (uint8_t set = 0; set < sps_set_count; set++) {
		uint16_t sps_size = read_bits(16);
		parse_nal_unit(sps_size, nullptr);
	}

	uint8_t pps_set_count = read_bits(8);
	for (uint8_t set = 0; set < pps_set_count; set++) {
		uint16_t pps_size = read_bits(16);
		parse_nal_unit(pps_size, nullptr);
	}

	if (target_profile_idc == VIDEO_CODING_H264_PROFILE_IDC_HIGH) {
		uint8_t chroma_format = read_bits(8) & 0b11;
		if (chroma_format == 0) {
			video_profile.chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_MONOCHROME;
		} else if (chroma_format == 1) {
			video_profile.chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_420;
		} else if (chroma_format == 2) {
			video_profile.chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_422;
		} else if (chroma_format == 3) {
			video_profile.chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_444;
		}

		video_profile.luma_bit_depth = (read_bits(8) & 0b111) + 8;
		video_profile.chroma_bit_depth = (read_bits(8) & 0b111) + 8;

		uint8_t sps_ext_sets = read_bits(8);
		for (uint8_t set = 0; set < sps_ext_sets; set++) {
			uint16_t sps_ext_size = read_bits(16);
			parse_nal_unit(sps_ext_size, nullptr);
		}
	}

	sps_sets.clear();
	pps_sets.clear();
}

void VideoStreamH264::set_rendering_device(RenderingDevice *p_local_device) {
	coding_device = p_local_device;
}

RID VideoStreamH264::create_video_session(uint32_t p_width, uint32_t p_height) {
	coding_device->video_profile_get_capabilities(video_profile);
	coding_device->video_profile_get_format_properties(video_profile);

	video_session = coding_device->video_session_create(video_profile, p_width, p_height);
	coding_device->video_session_add_h264_parameters(video_session, sps_sets, pps_sets);
	return video_session;
}

RID VideoStreamH264::create_texture_sampler(RD::SamplerState &p_sampler_template) {
	// TODO override parameters
	texture_sampler = coding_device->sampler_create(p_sampler_template);
	return texture_sampler;
}

RID VideoStreamH264::create_texture(RD::TextureFormat &p_texture_template) {
	// TODO override parameters
	Vector<VideoProfile> video_profiles;
	video_profiles.push_back(video_profile);

	p_texture_template.video_profiles = video_profiles;

	// TODO: how do we know this?
	//p_texture_template.height += 8;

	RD::TextureView texture_view;
	texture_view.ycbcr_sampler = texture_sampler;

	return coding_device->texture_create(p_texture_template, texture_view);
}

void VideoStreamH264::parse_container_block(Vector<uint8_t> p_buffer, RID p_dst_texture) {
	print_line(vformat("------------------Decoding Block [%d]------------------------", p_buffer.size()));
	src = p_buffer.ptr();
	shift = 0;

	bool seen_sps = false;
	bool seen_pps = false;

	while (src < p_buffer.ptr() + p_buffer.size()) {
		uint64_t nal_size = read_bits(length_size * 8);
		const uint8_t *nal_start = src;

		VideoDecodeH264SliceHeader slice_header = {};
		VideoCodingH264NalUnitType nal_unit_type = parse_nal_unit(nal_size, &slice_header);

		if (nal_unit_type == VIDEO_CODING_H264_NAL_UNIT_TYPE_SEQUENCE_PARAMETER_SET) {
			seen_sps = true;
		}

		if (nal_unit_type == VIDEO_CODING_H264_NAL_UNIT_TYPE_PICTURE_PARAMETER_SET) {
			seen_pps = true;
		}

		if (seen_sps && seen_pps) {
			create_video_session(1920, 1088);
			coding_device->video_session_begin();
			seen_sps = false;
			seen_pps = false;
		}

		if (nal_unit_type == VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE || nal_unit_type == VIDEO_CODING_H264_NAL_UNIT_TYPE_CODED_SLICE_IDR) {
			Span<uint8_t> slice_span = Span(nal_start, nal_size);
			coding_device->video_session_decode_h264(video_session, slice_span, slice_header, p_dst_texture);
		}
	}
}

VideoStreamH264::VideoStreamH264() {
	coding_device = RD::get_singleton();
}
