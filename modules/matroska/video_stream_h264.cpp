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
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_device_commons.h"

// TODO make Godot versions of all Vulkan structs
#include <vk_video/vulkan_video_codec_h264std.h>
#include <vk_video/vulkan_video_codec_h264std_decode.h>
#include <vulkan/vulkan_core.h>

RID VideoStreamH264::create_video_profile() {
	video_profile = RD::get_singleton()->video_profile_create(chroma_subsampling, luma_bit_depth, chroma_bit_depth);
	RD::get_singleton()->video_profile_bind_h264_decoding_metadata(video_profile, target_profile_idc, RenderingDeviceCommons::VIDEO_CODING_H264_PICTURE_LAYOUT_PROGRESSIVE);
	return video_profile;
}

void VideoStreamH264::decode_cluster() {
	RD::VideoCodingListID video_coding_list = RD::get_singleton()->video_coding_list_begin(video_profile, active_sps, active_pps);
	RD::get_singleton()->video_coding_list_bind_texure(video_coding_list, 1980, 1080, slice_spans.size());

	for (uint64_t i = 0; i < slice_spans.size(); i++) {
		RD::get_singleton()->video_coding_list_decode(video_coding_list, slice_spans[i], slice_metadatas[i], i);
	}

	RD::get_singleton()->video_coding_list_end();
}

// The Matroska "codec private" data for H264 is an AVCDecoderConfigurationRecord
void VideoStreamH264::parse_container_metadata(uint8_t *p_stream, uint64_t p_size) {
	uint8_t configuration_version = p_stream[0];
	ERR_FAIL_COND_MSG(configuration_version > 1, vformat("AVCDecoderConfigurationRecord version (%d) is greater than 1", configuration_version));

	target_profile_idc = RD::VideoCodingH264ProfileIdc(p_stream[1]);
	minimum_profile_idc = RD::VideoCodingH264ProfileIdc(p_stream[2]);

	target_level_idc = p_stream[3];

	// TODO what is this?
	uint8_t length_size = (p_stream[4] & 0b11) + 1;
	print_line(vformat("length size %d", length_size));

	uint8_t sps_sets = p_stream[5] & 0b11111;
	for (uint8_t set = 0; set < sps_sets; set++) {
		uint16_t sps_size = (p_stream[6] << 8) | p_stream[7];
		parse_nal_unit(p_stream + 8);
		p_stream += 2 + sps_size;
	}

	uint8_t pps_sets = p_stream[6];
	for (uint8_t set = 0; set < pps_sets; set++) {
		uint16_t pps_size = (p_stream[7] << 8) | p_stream[8];
		parse_nal_unit(p_stream + 9);
		p_stream += 2 + pps_size;
	}

	if (target_profile_idc == RenderingDeviceCommons::VIDEO_CODING_H264_PROFILE_IDC_HIGH) {
		uint8_t chroma_format = p_stream[7] & 0b11;
		if (chroma_format == 0) {
			chroma_subsampling = RenderingDeviceCommons::CHROMA_SUBSAMPLING_MONOCHROME;
		} else if (chroma_format == 1) {
			chroma_subsampling = RenderingDeviceCommons::CHROMA_SUBSAMPLING_420;
		} else if (chroma_format == 2) {
			chroma_subsampling = RenderingDeviceCommons::CHROMA_SUBSAMPLING_422;
		} else if (chroma_format == 3) {
			chroma_subsampling = RenderingDeviceCommons::CHROMA_SUBSAMPLING_444;
			// TODO separate color plane flag
		}

		luma_bit_depth = (p_stream[8] & 0b111) + 8;
		chroma_bit_depth = (p_stream[9] & 0b111) + 8;

		uint8_t sps_ext_sets = p_stream[10];
		for (uint8_t set = 0; set < sps_ext_sets; set++) {
			uint16_t sps_ext_size = (p_stream[11] << 8) | p_stream[12];
			parse_nal_unit(p_stream + 13);
			p_stream += sps_ext_size + 2;
		}
	}
}

void VideoStreamH264::parse_container_block(uint8_t *p_stream, uint64_t p_size) {
	// TODO figure out what's in the first 3 bytes
	uint32_t data = (p_stream[0] << 16) | (p_stream[1] << 8) | (p_stream[2]);
	print_line(vformat("leading bytes [%d, %d, %d] (%d), size %d", p_stream[0], p_stream[1], p_stream[2], data, p_size));
	if (parse_nal_unit(p_stream + 4)) {
		Vector<uint8_t> block;
		block.resize(p_size);
		memcpy(block.ptrw(), p_stream, p_size);
		slice_spans.push_back(block);
	}
}

bool VideoStreamH264::parse_nal_unit(uint8_t *p_stream) {
	uint8_t nal_ref_idc = (p_stream[0] & 0b1100000) >> 5;
	uint8_t nal_unit_type = p_stream[0] & 0b11111;
	p_stream += 1;

	switch (nal_unit_type) {
		case 1: {
			StdVideoDecodeH264PictureInfo slice_info = parse_slice_header(p_stream);
			slice_info.flags.is_reference = nal_ref_idc != 0;
			slice_metadatas.push_back(slice_info);
			return true;
		} break;

		case 7: {
			active_sps = parse_sequence_parameter_set(p_stream);
		} break;

		case 8: {
			active_pps = parse_picture_parameter_set(p_stream);
		} break;

		default: {
			WARN_PRINT(vformat("Unknown NAL unit type %d", nal_unit_type));
		}
	}

	return false;
}

StdVideoH264SequenceParameterSet VideoStreamH264::parse_sequence_parameter_set(uint8_t *p_stream) {
	print_line("Extremely cool SPS");
	StdVideoH264SequenceParameterSet sequence_parameter_set = {};

	sequence_parameter_set.profile_idc = StdVideoH264ProfileIdc(p_stream[0]);

	sequence_parameter_set.flags.constraint_set0_flag = (p_stream[1] & (1 << 7)) > 0;
	sequence_parameter_set.flags.constraint_set1_flag = (p_stream[1] & (1 << 6)) > 0;
	sequence_parameter_set.flags.constraint_set2_flag = (p_stream[1] & (1 << 5)) > 0;
	sequence_parameter_set.flags.constraint_set3_flag = (p_stream[1] & (1 << 4)) > 0;
	sequence_parameter_set.flags.constraint_set4_flag = (p_stream[1] & (1 << 3)) > 0;
	sequence_parameter_set.flags.constraint_set5_flag = (p_stream[1] & (1 << 2)) > 0;

	uint8_t level_idc = p_stream[2];
	switch (level_idc) {
		case 40: {
			sequence_parameter_set.level_idc = STD_VIDEO_H264_LEVEL_IDC_4_0;
		} break;

		default: {
			// TODO default to this.target_level_idc
			WARN_PRINT(vformat("Unhandled level %d", level_idc));
		}
	}

	// From here on they start using crazy ue encoding
	p_stream += 3;
	uint8_t shift = 7;
	uint8_t read = 0;

	sequence_parameter_set.seq_parameter_set_id = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	sequence_parameter_set.chroma_format_idc = StdVideoH264ChromaFormatIdc(parse_ue(p_stream, &shift, &read));
	p_stream += read;

	sequence_parameter_set.bit_depth_luma_minus8 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	sequence_parameter_set.bit_depth_chroma_minus8 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	sequence_parameter_set.flags.qpprime_y_zero_transform_bypass_flag = (p_stream[0] & (1 << shift)) > 0;
	sequence_parameter_set.flags.seq_scaling_matrix_present_flag = (p_stream[0] & (1 << (shift - 1))) > 0;
	p_stream += 1;
	shift = 7;

	if (sequence_parameter_set.flags.seq_scaling_matrix_present_flag) {
		// TODO
		print_line("skipping seq_scaling_matrix_present_flag");
	}

	sequence_parameter_set.log2_max_frame_num_minus4 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	sequence_parameter_set.pic_order_cnt_type = StdVideoH264PocType(parse_ue(p_stream, &shift, &read));
	p_stream += read;

	if (sequence_parameter_set.pic_order_cnt_type == STD_VIDEO_H264_POC_TYPE_0) {
		sequence_parameter_set.log2_max_pic_order_cnt_lsb_minus4 = parse_ue(p_stream, &shift, &read);
		p_stream += read;
	} else if (sequence_parameter_set.pic_order_cnt_type == STD_VIDEO_H264_POC_TYPE_1) {
		// TODO
		print_line("skipping pic_order_cnt_type type 2");
	}

	sequence_parameter_set.max_num_ref_frames = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	sequence_parameter_set.flags.gaps_in_frame_num_value_allowed_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	sequence_parameter_set.pic_width_in_mbs_minus1 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	sequence_parameter_set.pic_height_in_map_units_minus1 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	sequence_parameter_set.flags.frame_mbs_only_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	if (!sequence_parameter_set.flags.frame_mbs_only_flag) {
		sequence_parameter_set.flags.mb_adaptive_frame_field_flag = (p_stream[0] & (1 << shift)) > 0;
		p_stream += shift == 0;
		shift = (shift + 7) % 8;
	}

	sequence_parameter_set.flags.direct_8x8_inference_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	sequence_parameter_set.flags.frame_cropping_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	if (sequence_parameter_set.flags.frame_cropping_flag) {
		sequence_parameter_set.frame_crop_left_offset = parse_ue(p_stream, &shift, &read);
		p_stream += read;

		sequence_parameter_set.frame_crop_right_offset = parse_ue(p_stream, &shift, &read);
		p_stream += read;

		sequence_parameter_set.frame_crop_top_offset = parse_ue(p_stream, &shift, &read);
		p_stream += read;

		sequence_parameter_set.frame_crop_bottom_offset = parse_ue(p_stream, &shift, &read);
		p_stream += read;
	}

	sequence_parameter_set.flags.vui_parameters_present_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	if (sequence_parameter_set.flags.vui_parameters_present_flag) {
		// TODO
		print_line("skipping vui_parameters_present_flag");
	}

	return sequence_parameter_set;
}

StdVideoH264PictureParameterSet VideoStreamH264::parse_picture_parameter_set(uint8_t *p_stream) {
	print_line("Extremely cool PPS");

	StdVideoH264PictureParameterSet picture_parameter_set = {};
	uint8_t shift = 7;
	uint8_t read = 0;

	picture_parameter_set.pic_parameter_set_id = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	picture_parameter_set.seq_parameter_set_id = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	picture_parameter_set.flags.entropy_coding_mode_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	picture_parameter_set.flags.bottom_field_pic_order_in_frame_present_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	uint64_t num_slice_groups_minus1 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	if (num_slice_groups_minus1 > 0) {
		// TODO
		print_line("skipping slice groups");
	}

	picture_parameter_set.num_ref_idx_l0_default_active_minus1 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	picture_parameter_set.num_ref_idx_l1_default_active_minus1 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	picture_parameter_set.flags.weighted_pred_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	picture_parameter_set.weighted_bipred_idc = StdVideoH264WeightedBipredIdc((p_stream[0] & (0b11 << shift)) >> shift);
	p_stream += shift == 0;
	shift = (shift + 6) % 8;

	picture_parameter_set.pic_init_qp_minus26 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	picture_parameter_set.pic_init_qs_minus26 = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	picture_parameter_set.chroma_qp_index_offset = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	picture_parameter_set.flags.deblocking_filter_control_present_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	picture_parameter_set.flags.constrained_intra_pred_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	picture_parameter_set.flags.redundant_pic_cnt_present_flag = (p_stream[0] & (1 << shift)) > 0;
	p_stream += shift == 0;
	shift = (shift + 7) % 8;

	return picture_parameter_set;
}

StdVideoDecodeH264PictureInfo VideoStreamH264::parse_slice_header(uint8_t *p_stream) {
	print_line("Extremely cool slice header");

	StdVideoDecodeH264PictureInfo slice_header = {};
	slice_header.seq_parameter_set_id = active_sps.seq_parameter_set_id;

	uint8_t shift = 7;
	uint8_t read = 0;

	parse_ue(p_stream, &shift, &read); // first_mb_in_slice
	p_stream += read;

	uint64_t slice_type = parse_ue(p_stream, &shift, &read);
	p_stream += read;

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
		}

		case 8: {
			print_line("slice_type = SP (Only)");
		}

		case 9: {
			print_line("slice_type = SI (Only)");
		}
	}

	slice_header.flags.is_intra = slice_type == 0 || slice_type == 1 || slice_type == 2 || slice_type == 5 || slice_type == 6 || slice_type == 7;
	slice_header.flags.IdrPicFlag = slice_type == 5;

	slice_header.pic_parameter_set_id = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	// TODO sps separate color plane flag
	if (false /* separate color plane flag*/) {
		// colour_plane_id
		p_stream += shift == 0;
		shift = (shift + 6) % 8;
	}

	slice_header.frame_num = parse_ue(p_stream, &shift, &read);
	p_stream += read;

	if (!active_sps.flags.frame_mbs_only_flag) {
		slice_header.flags.field_pic_flag = (p_stream[0] & (1 << shift)) > 0;
		p_stream += shift == 0;
		shift = (shift + 7) % 8;

		if (slice_header.flags.field_pic_flag) {
			slice_header.flags.bottom_field_flag = (p_stream[0] & (1 << shift)) > 0;
			p_stream += shift == 0;
			shift = (shift + 7) % 8;
		}
	}

	if (slice_header.flags.IdrPicFlag) {
		slice_header.idr_pic_id = parse_ue(p_stream, &shift, &read);
		p_stream += read;
	}

	// TODO
	slice_header.flags.complementary_field_pair = false;
	slice_header.PicOrderCnt[0] = 0;
	slice_header.PicOrderCnt[1] = 0;

	return slice_header;
}

uint64_t VideoStreamH264::parse_ue(uint8_t *p_stream, uint8_t *shift, uint8_t *read) {
	uint64_t mask = 0;
	for (uint8_t i = 0; i <= *shift; i++) {
		mask |= (1 << i);
	}

	uint64_t bits = 0;
	uint64_t bytes = 0;
	uint64_t encoding = p_stream[0] & mask;

	while ((encoding & (1 << *shift)) == 0) {
		bits += 1;
		if (*shift == 0) {
			encoding = (encoding << 8) | p_stream[0];
			p_stream += 1;
			bytes += 1;
			*shift = 7;
		} else {
			*shift -= 1;
		}
	}

	if (bits > *shift) {
		encoding = (encoding << 8) | p_stream[0];
		bytes += 1;
		*shift += 8;
	} else if (bits == *shift) {
		bytes += 1;
	}

	uint64_t value = 0;
	for (uint8_t offset = 0; offset <= bits; offset++) {
		value |= encoding & (1 << (*shift - offset));
	}

	value = (value >> (*shift - bits));
	*shift = (*shift + 7 - bits) % 8;
	*read = bytes;

	return value - 1;
}

int64_t VideoStreamH264::parse_se(uint8_t *p_stream, uint8_t *shift, uint8_t *read) {
	uint64_t mask = 0;
	for (uint8_t i = 0; i <= *shift; i++) {
		mask |= (1 << i);
	}

	uint64_t bits = 0;
	uint64_t bytes = 0;
	uint64_t encoding = p_stream[0] & mask;

	while ((encoding & (1 << *shift)) == 0) {
		bits += 1;
		if (*shift == 0) {
			encoding = (encoding << 8) | p_stream[0];
			p_stream += 1;
			bytes += 1;
			*shift = 7;
		} else {
			*shift -= 1;
		}
	}

	if (bits > *shift) {
		encoding = (encoding << 8) | p_stream[0];
		bytes += 1;
		*shift += 8;
	} else if (bits == *shift) {
		bytes += 1;
	}

	int64_t value = 0;
	for (uint8_t offset = 0; offset <= bits; offset++) {
		value |= encoding & (1 << (*shift - offset));
	}

	value = (value >> (*shift - bits));
	*shift = (*shift + 7 - bits) % 8;
	*read = bytes;

	return value - 1;
}
