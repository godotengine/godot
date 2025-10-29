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
#include "servers/rendering/rendering_device_commons.h"
#include "servers/rendering/video/av1.h"
#include "servers/rendering/video/av1_decode.h"

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

bool VideoStreamAV1::parse_open_bitstream_unit(VideoDecodeAV1Frame *r_av1_frame) {
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

	uint64_t obu_size;
	if (obu_has_size_field) {
		obu_size = read_leb128();
	} else {
		ERR_FAIL_V_MSG(false, "CANNOT DECODE");
	}

	const uint8_t *obu_start = src;

	bool is_frame = false;
	switch (obu_type) {
		case VIDEO_CODING_AV1_OBU_TYPE_SEQUENCE_HEADER: {
			print_line(vformat("OBU Sequence Header [%d]", obu_size));
			av1_sequence_header = parse_sequence_header();
		} break;

		case VIDEO_CODING_AV1_OBU_TYPE_FRAME_HEADER: {
			print_line(vformat("OBU Frame Header [%d]", obu_size));
			*r_av1_frame = parse_frame_header();
		} break;

		case VIDEO_CODING_AV1_OBU_TYPE_FRAME: {
			is_frame = true;
			print_line(vformat("OBU Frame [%d]", obu_size));
			*r_av1_frame = parse_frame();
		} break;

		default: {
			print_line(vformat("Unknown OBU Type (%d) [%d]", obu_type, obu_size));
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
		//operating_points_cnt_minus_1 = 0;
		//operating_point_idc[0] = 0
		read_bits(5); //seq_level_idx[0]
		//seq_tier[0] = 0u pa
		//decoder_model_present_for_this_op[0] = 0
		//initial_display_delay_present_for_this_op[0] = 0
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

		uint8_t operating_points_cnt_minus_1 = read_bits(5);
		for (uint64_t i = 0; i <= operating_points_cnt_minus_1; i++) {
			read_bits(12); // operating_point_idc
			uint8_t seq_level_idx = read_bits(5);

			if (seq_level_idx > 7) {
				read_bits(1);
			}

			if (sequence_header.decoder_model_info_present_flag) {
				bool decoder_model_present_for_this_op = read_bits(1);
				if (decoder_model_present_for_this_op) {
					uint8_t size = sequence_header.decoder_model_info.buffer_delay_length_minus_1 + 1;
					read_bits(size); // decoder_buffer_delay
					read_bits(size); // encoder_buffer_delay
					read_bits(1); // low_delay_mode_flag
				}
			}

			if (sequence_header.initial_display_delay_present_flag) {
				bool initial_display_delay_present_for_this_op = read_bits(1);
				if (initial_display_delay_present_for_this_op) {
					read_bits(4);
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

	bool show_existing_frame;
	bool show_frame;
	bool showable_frame;

	if (av1_sequence_header.reduced_still_picture_header_flag) {
		show_existing_frame = false;
		std_frame_header.frame_type = VIDEO_CODING_AV1_FRAME_TYPE_KEY;
		show_frame = true;
		showable_frame = false;
	} else {
		show_existing_frame = read_bits(1);
		if (show_existing_frame) {
			//uint32_t frame_to_show_map_idx = read_bits(3);

			// if (decoder_model_info_present_flag && !equal_picture_interval) {}

			if (av1_sequence_header.frame_id_numbers_present_flag) {
				uint32_t display_frame_id = read_bits(id_len);
				print_line("display_frame_id", display_frame_id);
			}

			return std_frame_header;
		}

		std_frame_header.frame_type = VideoCodingAV1FrameType(read_bits(2));

		show_frame = read_bits(1);
		// if (decoder_model_info_present_flag && !equal_picture_interval) {}

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
		}

		for (uint64_t i = 0; i < VIDEO_CODING_AV1_REFS_PER_FRAME; i++) {
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

	if (std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_KEY || std_frame_header.frame_type == VIDEO_CODING_AV1_FRAME_TYPE_INTRA_ONLY) {
		std_frame_header.force_integer_mv_flag = true;
	}

	if (av1_sequence_header.frame_id_numbers_present_flag) {
		std_frame_header.current_frame_id = read_bits(id_len);
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

	if (showable_frame) {
	}

	return std_frame_header;
}

VideoDecodeAV1Frame VideoStreamAV1::parse_frame() {
	return parse_frame_header();
}

void VideoStreamAV1::parse_container_metadata(const uint8_t *p_stream, uint64_t p_size) {
	src = p_stream;
	shift = 0;

	read_bits(1); // marker.
	uint8_t version = read_bits(7);
	ERR_FAIL_COND(version != 1);

	uint8_t seq_profile = read_bits(3);
	uint8_t seq_level_idx_0 = read_bits(5);
	uint8_t seq_tier_0 = read_bits(1);
	print_line("seq_profile", seq_profile);
	print_line("seq_level_idx_0", seq_level_idx_0);
	print_line("seq_tier_0", seq_tier_0);

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
		video_profile.chroma_subsampling = RD::VIDEO_CHROMA_SUBSAMPLING_MONOCHROME;
	} else if (chroma_subsampling_x && chroma_subsampling_y) {
		video_profile.chroma_subsampling = RD::VIDEO_CHROMA_SUBSAMPLING_420;
	} else if (chroma_subsampling_x && !chroma_subsampling_y) {
		video_profile.chroma_subsampling = RD::VIDEO_CHROMA_SUBSAMPLING_422;
	} else {
		video_profile.chroma_subsampling = RD::VIDEO_CHROMA_SUBSAMPLING_444;
	}

	print_line("chroma_subsampling_position", chroma_subsampling_position);

	bool initial_presentation_delay_present = read_bits(1) > 0;
	if (initial_presentation_delay_present) {
		uint8_t initial_presentation_delay_minus_one = read_bits(4);
		print_line("initial_presentation_delay_minus_one", initial_presentation_delay_minus_one);
	} else {
		read_bits(4); // reserved.
	}

	while (src < p_stream + p_size) {
		parse_open_bitstream_unit(0);
	}
}

void VideoStreamAV1::set_rendering_device(RenderingDevice *p_local_device) {
	local_device = p_local_device;
}

RID VideoStreamAV1::create_video_session(uint32_t p_width, uint32_t p_height) {
	local_device->video_profile_get_capabilities(video_profile);
	local_device->video_profile_get_format_properties(video_profile);

	video_session = local_device->video_session_create(video_profile, p_width, p_height);
	local_device->video_session_add_av1_parameters(video_session, av1_sequence_header);
	return video_session;
}

RID VideoStreamAV1::create_texture_sampler(RD::SamplerState &p_sampler_template) {
	// TODO override parameters
	texture_sampler = local_device->sampler_create(p_sampler_template);
	return texture_sampler;
}

RID VideoStreamAV1::create_texture(RD::TextureFormat &p_texture_template) {
	// TODO override parameters
	Vector<RD::VideoProfile> video_profiles;
	video_profiles.push_back(video_profile);

	p_texture_template.video_profiles = video_profiles;

	RD::TextureView texture_view;
	texture_view.ycbcr_sampler = texture_sampler;

	return local_device->texture_create(p_texture_template, texture_view);
}

void VideoStreamAV1::parse_container_block(Vector<uint8_t> p_block, RID p_dst_texture) {
	print_line(vformat("-----------Decoding block [%d]-----------", p_block.size()));

	src = p_block.ptr();
	shift = 0;

	while (src < p_block.ptr() + p_block.size()) {
		VideoDecodeAV1Frame av1_frame_header = {};

		const uint8_t *obu_start = src;
		bool is_frame = parse_open_bitstream_unit(&av1_frame_header);
		uint64_t obu_size = src - obu_start;

		if (is_frame) {
			Span<uint8_t> frame_span = Span(obu_start, obu_size);
			local_device->video_session_decode_av1(video_session, frame_span, av1_frame_header, p_dst_texture);
		}
	}
}

VideoStreamAV1::VideoStreamAV1() {
	local_device = RD::get_singleton();

	video_profile.operation = RD::VIDEO_OPERATION_DECODE_AV1;
}
