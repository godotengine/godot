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
	RD::get_singleton()->video_profile_bind_h264_decoding_metadata(video_profile, target_profile_idc, RD::VIDEO_CODING_H264_PICTURE_LAYOUT_PROGRESSIVE);

	RenderingDeviceCommons::TextureFormat dpb_format;
	dpb_format.format = RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	dpb_format.width = 1980;
	dpb_format.height = 1080;
	dpb_format.depth = 0;
	dpb_format.array_layers = 17;
	dpb_format.mipmaps = 1;
	dpb_format.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
	//dpb_format.samples = RenderingDeviceCommons::TEXTURE_SAMPLES_16; // TODO huh?
	dpb_format.usage_bits = RD::TEXTURE_USAGE_VIDEO_DECODE_DPB_BIT;
	dpb_format.shareable_formats.clear();
	dpb_format.is_resolve_buffer = false;
	dpb_format.is_discardable = false;

	dpb = RD::get_singleton()->texture_create_for_video_coding(dpb_format, RD::TextureView(), video_profile);

	RenderingDeviceCommons::TextureFormat dst_format;
	dst_format.format = RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	dst_format.width = 1980;
	dst_format.height = 1080;
	dst_format.depth = 1;
	dst_format.array_layers = 120;
	dst_format.mipmaps = 1;
	dst_format.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
	//dst_format.samples = RenderingDeviceCommons::TEXTURE_SAMPLES_16; // TODO huh?
	dst_format.usage_bits = RD::TEXTURE_USAGE_VIDEO_DECODE_DST_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
	dst_format.shareable_formats.clear();
	dst_format.is_resolve_buffer = false;
	dst_format.is_discardable = false;

	dst_texture = RD::get_singleton()->texture_create_for_video_coding(dst_format, RD::TextureView(), video_profile);

	return video_profile;
}

// The Matroska "codec private" data for H264 is an AVCDecoderConfigurationRecord
void VideoStreamH264::parse_container_metadata(const uint8_t *p_stream, uint64_t p_size) {
	src = p_stream;
	shift = 7;

	uint8_t configuration_version = read_bits(8);
	ERR_FAIL_COND_MSG(configuration_version > 1, vformat("AVCDecoderConfigurationRecord version (%d) is greater than 1", configuration_version));

	target_profile_idc = RD::VideoCodingH264ProfileIdc(read_bits(8));
	minimum_profile_idc = RD::VideoCodingH264ProfileIdc(read_bits(8));

	target_level_idc = read_bits(8);

	// TODO what is this?
	uint8_t length_size = (read_bits(8) & 0b11) + 1;
	print_line(vformat("length size %d", length_size));

	uint8_t sps_sets = read_bits(8) & 0b11111;
	for (uint8_t set = 0; set < sps_sets; set++) {
		const uint8_t *start = src;
		uint16_t sps_size = read_bits(16);
		parse_nal_unit(sps_size);
		src = start + 2 + sps_size;
		shift = 7;
	}

	uint8_t pps_sets = read_bits(8);
	for (uint8_t set = 0; set < pps_sets; set++) {
		const uint8_t *start = src;
		uint16_t pps_size = read_bits(16);
		parse_nal_unit(pps_size);
		src = start + 2 + pps_size;
		shift = 7;
	}

	if (target_profile_idc == RD::VIDEO_CODING_H264_PROFILE_IDC_HIGH) {
		uint8_t chroma_format = read_bits(8) & 0b11;
		if (chroma_format == 0) {
			chroma_subsampling = RD::CHROMA_SUBSAMPLING_MONOCHROME;
		} else if (chroma_format == 1) {
			chroma_subsampling = RD::CHROMA_SUBSAMPLING_420;
		} else if (chroma_format == 2) {
			chroma_subsampling = RD::CHROMA_SUBSAMPLING_422;
		} else if (chroma_format == 3) {
			chroma_subsampling = RD::CHROMA_SUBSAMPLING_444;
		}

		luma_bit_depth = (read_bits(8) & 0b111) + 8;
		chroma_bit_depth = (read_bits(8) & 0b111) + 8;

		uint8_t sps_ext_sets = read_bits(8);
		for (uint8_t set = 0; set < sps_ext_sets; set++) {
			const uint8_t *start = src;
			uint16_t sps_ext_size = read_bits(16);
			parse_nal_unit(sps_ext_size);
			src = start + 2 + sps_ext_size;
		}
	}
}

void VideoStreamH264::begin_cluster() {
	print_line("----------BEGIN CLUSTER--------------------");
	video_coding_list = RD::get_singleton()->video_coding_list_begin(video_profile, dpb, active_sps, active_pps);

	target_dpb_layer = 0;
	target_dst_layer = 0;
}

void VideoStreamH264::append_container_block(Vector<uint8_t> p_buffer) {
	print_line(vformat("Block size [%d] bytes", p_buffer.size()));
	src = p_buffer.ptr();
	shift = 7;

	int64_t total_read = 0;
	while (total_read < p_buffer.size()) {
		uint64_t nal_size = read_bits(32);
		const uint8_t *start = src;

		parse_nal_unit(nal_size);
		total_read += nal_size + 4;
		src = start + nal_size;
		shift = 7;
	}

	print_line("--------------------------------------");
}

RID VideoStreamH264::end_cluster() {
	print_line("----------END CLUSTER--------------------");
	RD::get_singleton()->video_coding_list_end();
	return dst_texture;
}

void VideoStreamH264::parse_nal_unit(uint64_t p_size) {
	const uint8_t *start = src;

	uint8_t header = read_bits(8);
	uint8_t nal_ref_idc = (header & 0b1100000) >> 5;
	uint8_t nal_unit_type = header & 0b11111;

	switch (nal_unit_type) {
		case 1: {
			uint64_t buffer_size = p_size + 4;
			buffer_size += 128 - (buffer_size % 128);
			RID buffer = RD::get_singleton()->storage_buffer_create_video_session(buffer_size, video_profile, Span<uint8_t>(), RD::STORAGE_BUFFER_USAGE_VIDEO_DECODE_SRC);

			RD::get_singleton()->buffer_update(buffer, 0, 4, &p_size);
			RD::get_singleton()->buffer_update(buffer, 4, p_size, start);
			RD::get_singleton()->_flush_and_stall_for_all_frames();

			StdVideoDecodeH264PictureInfo slice_info = parse_slice_header(p_size - 1, false);
			slice_info.flags.is_reference = nal_ref_idc != 0;
			RD::get_singleton()->video_coding_list_decode(video_coding_list, buffer, dst_texture, slice_info, target_dst_layer);
			target_dst_layer += 1;

			String is_reference = nal_ref_idc != 0 ? "reference" : "non-reference";
			print_line(vformat("Read %d/%d bytes of a %s slice header", (uint64_t)(src - start), p_size, is_reference));
		} break;

		case 5: {
			uint64_t buffer_size = p_size + 4;
			buffer_size += 128 - (buffer_size % 128);
			RID buffer = RD::get_singleton()->storage_buffer_create_video_session(buffer_size, video_profile, Span<uint8_t>(), RD::STORAGE_BUFFER_USAGE_VIDEO_DECODE_SRC);

			RD::get_singleton()->buffer_update(buffer, 0, 4, &p_size);
			RD::get_singleton()->buffer_update(buffer, 4, p_size, start);
			RD::get_singleton()->_flush_and_stall_for_all_frames();

			StdVideoDecodeH264PictureInfo slice_info = parse_slice_header(p_size - 1, true);
			slice_info.flags.is_reference = nal_ref_idc != 0;
			RD::get_singleton()->video_coding_list_decode(video_coding_list, buffer, dst_texture, slice_info, target_dst_layer);
			target_dst_layer += 1;

			print_line(vformat("Read %d/%d bytes of an IDR slice header", (uint64_t)(src - start), p_size));
		} break;

		case 6: {
			print_line(vformat("Skipping %d bytes of supplemental enhancement information", p_size));
		} break;

		case 7: {
			active_sps = parse_sequence_parameter_set(p_size - 1);
			print_line(vformat("Read %d/%d bytes of an SPS", (uint64_t)(src - start), p_size));
		} break;

		case 8: {
			active_pps = parse_picture_parameter_set(p_size - 1);
			print_line(vformat("Read %d/%d bytes of a PPS", (uint64_t)(src - start), p_size));
		} break;

		default: {
			print_line(vformat("Unknown NAL unit type %d", nal_unit_type));
		}
	}
}

StdVideoH264SequenceParameterSet VideoStreamH264::parse_sequence_parameter_set(uint64_t p_size) {
	StdVideoH264SequenceParameterSet sequence_parameter_set = {};

	sequence_parameter_set.profile_idc = StdVideoH264ProfileIdc(read_bits(8));

	uint8_t flags = read_bits(8);
	sequence_parameter_set.flags.constraint_set0_flag = (flags & (1 << 7)) > 0;
	sequence_parameter_set.flags.constraint_set1_flag = (flags & (1 << 6)) > 0;
	sequence_parameter_set.flags.constraint_set2_flag = (flags & (1 << 5)) > 0;
	sequence_parameter_set.flags.constraint_set3_flag = (flags & (1 << 4)) > 0;
	sequence_parameter_set.flags.constraint_set4_flag = (flags & (1 << 3)) > 0;
	sequence_parameter_set.flags.constraint_set5_flag = (flags & (1 << 2)) > 0;

	uint8_t level_idc = read_bits(8);
	switch (level_idc) {
		case 40: {
			sequence_parameter_set.level_idc = STD_VIDEO_H264_LEVEL_IDC_4_0;
		} break;

		default: {
			// TODO default to this.target_level_idc
			WARN_PRINT(vformat("Unhandled level %d", level_idc));
		}
	}

	sequence_parameter_set.seq_parameter_set_id = read_ue();

	if (sequence_parameter_set.profile_idc == STD_VIDEO_H264_PROFILE_IDC_HIGH || sequence_parameter_set.profile_idc == STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE) {
		sequence_parameter_set.chroma_format_idc = StdVideoH264ChromaFormatIdc(read_ue());

		if (sequence_parameter_set.chroma_format_idc == STD_VIDEO_H264_CHROMA_FORMAT_IDC_444) {
			sequence_parameter_set.flags.separate_colour_plane_flag = read_bits(1) > 0;
		}

		sequence_parameter_set.bit_depth_luma_minus8 = read_ue();
		sequence_parameter_set.bit_depth_chroma_minus8 = read_ue();

		sequence_parameter_set.flags.qpprime_y_zero_transform_bypass_flag = read_bits(1) > 0;
		sequence_parameter_set.flags.seq_scaling_matrix_present_flag = read_bits(1) > 0;

		if (sequence_parameter_set.flags.seq_scaling_matrix_present_flag) {
			uint64_t size = sequence_parameter_set.chroma_format_idc != STD_VIDEO_H264_CHROMA_FORMAT_IDC_444 ? 8 : 12;
			for (uint64_t i = 0; i < size; i++) {
				bool present = read_bits(1) > 0;
				if (present) {
					if (i < 6) {
						print_line("skipping 4x4 seq scaling lists");
					} else {
						print_line("skipping 8x8 seq scaling lists");
					}
				}
			}
		}
	}

	sequence_parameter_set.log2_max_frame_num_minus4 = read_ue();
	sequence_parameter_set.pic_order_cnt_type = StdVideoH264PocType(read_ue());

	if (sequence_parameter_set.pic_order_cnt_type == STD_VIDEO_H264_POC_TYPE_0) {
		sequence_parameter_set.log2_max_pic_order_cnt_lsb_minus4 = read_ue();
	} else if (sequence_parameter_set.pic_order_cnt_type == STD_VIDEO_H264_POC_TYPE_1) {
		// TODO
		print_line("skipping pic_order_cnt_type type 2");
	}

	sequence_parameter_set.max_num_ref_frames = read_ue();

	sequence_parameter_set.flags.gaps_in_frame_num_value_allowed_flag = read_bits(1) > 0;

	sequence_parameter_set.pic_width_in_mbs_minus1 = read_ue();
	sequence_parameter_set.pic_height_in_map_units_minus1 = read_ue();

	sequence_parameter_set.flags.frame_mbs_only_flag = read_bits(1) > 0;

	if (!sequence_parameter_set.flags.frame_mbs_only_flag) {
		sequence_parameter_set.flags.mb_adaptive_frame_field_flag = read_bits(1) > 0;
	}

	sequence_parameter_set.flags.direct_8x8_inference_flag = read_bits(1) > 0;
	sequence_parameter_set.flags.frame_cropping_flag = read_bits(1) > 0;

	if (sequence_parameter_set.flags.frame_cropping_flag) {
		sequence_parameter_set.frame_crop_left_offset = read_ue();
		sequence_parameter_set.frame_crop_right_offset = read_ue();
		sequence_parameter_set.frame_crop_top_offset = read_ue();
		sequence_parameter_set.frame_crop_bottom_offset = read_ue();
	}

	sequence_parameter_set.flags.vui_parameters_present_flag = read_bits(1) > 0;
	if (sequence_parameter_set.flags.vui_parameters_present_flag) {
		StdVideoH264SequenceParameterSetVui sps_vui;

		sps_vui.flags.aspect_ratio_info_present_flag = read_bits(1) > 0;
		if (sps_vui.flags.aspect_ratio_info_present_flag) {
			sps_vui.aspect_ratio_idc = StdVideoH264AspectRatioIdc(read_bits(8));
			if (sps_vui.aspect_ratio_idc == STD_VIDEO_H264_ASPECT_RATIO_IDC_EXTENDED_SAR) {
				sps_vui.sar_width = read_bits(16);
				sps_vui.sar_height = read_bits(16);
			}
		}

		sps_vui.flags.overscan_info_present_flag = read_bits(1) > 0;
		if (sps_vui.flags.overscan_info_present_flag) {
			sps_vui.flags.overscan_appropriate_flag = read_bits(1) > 0;
		}

		sps_vui.flags.video_signal_type_present_flag = read_bits(1) > 0;
		if (sps_vui.flags.video_signal_type_present_flag) {
			sps_vui.video_format = read_bits(3);

			sps_vui.flags.video_full_range_flag = read_bits(1) > 0;

			sps_vui.flags.color_description_present_flag = read_bits(1) > 0;
			if (sps_vui.flags.color_description_present_flag) {
				sps_vui.colour_primaries = read_bits(8);
				sps_vui.transfer_characteristics = read_bits(8);
				sps_vui.matrix_coefficients = read_bits(8);
			}
		}

		sps_vui.flags.chroma_loc_info_present_flag = read_bits(1) > 0;
		if (sps_vui.flags.chroma_loc_info_present_flag) {
			sps_vui.chroma_sample_loc_type_top_field = read_ue();
			sps_vui.chroma_sample_loc_type_bottom_field = read_ue();
		}

		sps_vui.flags.timing_info_present_flag = read_bits(1) > 0;
		if (sps_vui.flags.timing_info_present_flag) {
			sps_vui.num_units_in_tick = read_bits(32);
			sps_vui.time_scale = read_bits(32);
			sps_vui.flags.fixed_frame_rate_flag = read_bits(1) > 0;
		}

		sps_vui.flags.nal_hrd_parameters_present_flag = read_bits(1) > 0;
		if (sps_vui.flags.nal_hrd_parameters_present_flag) {
			print_line("skipping nal_hrd_parameters_present_flag");
		}

		sps_vui.flags.vcl_hrd_parameters_present_flag = read_bits(1) > 0;
		if (sps_vui.flags.vcl_hrd_parameters_present_flag) {
			print_line("skipping vcl_hrd_parameters_present_flag");
		}

		if (sps_vui.flags.nal_hrd_parameters_present_flag || sps_vui.flags.vcl_hrd_parameters_present_flag) {
			read_bits(1); // low_delay_hrd_flag
		}

		read_bits(1); // pic_struct_present_flag
		sps_vui.flags.bitstream_restriction_flag = read_bits(1) > 0;
		if (sps_vui.flags.bitstream_restriction_flag) {
			read_bits(1); // motion_vectors_over_pic_boundaries_flag

			read_ue(); // max_bytes_per_pic_denom
			read_ue(); // max_bits_per_mb_denom

			read_ue(); // log2_max_mv_length_horizontal
			read_ue(); // log2_max_mv_length_vertical

			sps_vui.max_num_reorder_frames = read_ue();
			sps_vui.max_dec_frame_buffering = read_ue();
		}
	}

	return sequence_parameter_set;
}

StdVideoH264PictureParameterSet VideoStreamH264::parse_picture_parameter_set(uint64_t p_size) {
	StdVideoH264PictureParameterSet picture_parameter_set = {};

	picture_parameter_set.pic_parameter_set_id = read_ue();
	picture_parameter_set.seq_parameter_set_id = read_ue();

	picture_parameter_set.flags.entropy_coding_mode_flag = read_bits(1) > 0;
	picture_parameter_set.flags.bottom_field_pic_order_in_frame_present_flag = read_bits(1) > 0;

	uint64_t num_slice_groups_minus1 = read_ue();
	if (num_slice_groups_minus1 > 0) {
		// TODO
		print_line("skipping slice groups");
	}

	picture_parameter_set.num_ref_idx_l0_default_active_minus1 = read_ue();
	picture_parameter_set.num_ref_idx_l1_default_active_minus1 = read_ue();

	picture_parameter_set.flags.weighted_pred_flag = read_bits(1) > 0;
	picture_parameter_set.weighted_bipred_idc = StdVideoH264WeightedBipredIdc(read_bits(2));

	picture_parameter_set.pic_init_qp_minus26 = read_ue();
	picture_parameter_set.pic_init_qs_minus26 = read_ue();
	picture_parameter_set.chroma_qp_index_offset = read_ue();

	picture_parameter_set.flags.deblocking_filter_control_present_flag = read_bits(1) > 0;
	picture_parameter_set.flags.constrained_intra_pred_flag = read_bits(1) > 0;
	picture_parameter_set.flags.redundant_pic_cnt_present_flag = read_bits(1) > 0;

	return picture_parameter_set;
}

StdVideoDecodeH264PictureInfo VideoStreamH264::parse_slice_header(uint64_t p_size, bool p_is_idr) {
	StdVideoDecodeH264PictureInfo slice_header = {};
	slice_header.seq_parameter_set_id = active_sps.seq_parameter_set_id;

	slice_header.flags.IdrPicFlag = p_is_idr;

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

	slice_header.flags.is_intra = slice_type == 2 || slice_type == 4 || slice_type == 7 || slice_type == 9;

	slice_header.pic_parameter_set_id = read_ue();

	if (active_sps.flags.separate_colour_plane_flag) {
		read_bits(2); // colour_plane_id
	}

	slice_header.frame_num = read_ue(); // apparently is u(v)

	if (!active_sps.flags.frame_mbs_only_flag) {
		slice_header.flags.field_pic_flag = read_bits(1) > 0;
		if (slice_header.flags.field_pic_flag) {
			slice_header.flags.bottom_field_flag = read_bits(1) > 0;
		}
	}

	if (slice_header.flags.IdrPicFlag) {
		slice_header.idr_pic_id = read_ue();
	}

	// TODO
	slice_header.flags.complementary_field_pair = false;
	slice_header.PicOrderCnt[0] = 0;
	slice_header.PicOrderCnt[1] = 0;
	return slice_header;
}

uint64_t VideoStreamH264::read_bits(uint8_t p_amount) {
	uint64_t encoded = src[0];
	while (p_amount > shift + 1) {
		encoded = (encoded << 8) | src[1];
		src += 1;
		shift += 8;
	}

	uint64_t value = 0;
	for (uint8_t offset = 0; offset < p_amount; offset++) {
		value |= encoded & (1 << (shift - offset));
	}

	value = (value >> (1 + shift - p_amount));

	shift = (shift + 8 - p_amount) % 8;
	src += (shift + 1) / 8;

	return value;
}

uint64_t VideoStreamH264::read_ue() {
	uint64_t bits = 0;
	uint64_t encoded = src[0];

	while ((encoded & (1 << shift)) == 0) {
		bits += 1;
		if (shift == 0) {
			encoded = (encoded << 8) | src[1];
			src += 1;
			shift = 7;
		} else {
			shift -= 1;
		}
	}

	return read_bits(bits + 1) - 1;
}

int64_t VideoStreamH264::read_se() {
	return static_cast<int64_t>(read_ue());
}
