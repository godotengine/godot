/**************************************************************************/
/*  video_stream_matroska.cpp                                             */
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

#include "video_stream_matroska.h"

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/math/color.h"
#include "core/string/print_string.h"
#include "core/variant/variant.h"
#include "matroska.h"
#include "modules/matroska/video_stream_av1.h"
#include "modules/matroska/video_stream_h264.h"
#include "scene/resources/texture.h"
#include "scene/resources/video_stream_encoding.h"
#include "servers/audio/audio_server.h"
#include "servers/rendering/rendering_device.h"

#include <cstddef>
#include <cstdint>

void VideoStreamMatroska::set_audio_track(int p_track) {
	audio_track = p_track;
}

Ref<VideoStreamPlayback> VideoStreamMatroska::instantiate_playback() {
	Ref<VideoStreamPlaybackMatroska> stream_playback;
	stream_playback.instantiate();
	stream_playback->set_audio_track(audio_track);

	Error err = stream_playback->set_file(file);
	if (err != OK) {
		return nullptr;
	}

	return stream_playback;
}

void VideoStreamPlaybackMatroska::_skip_id(uint64_t p_id) {
	uint64_t size = read_size();
	WARN_PRINT(vformat("Unhandled element with ID [%x] of size [%d]", p_id, size));
	read_ptr += size;
}

uint64_t VideoStreamPlaybackMatroska::read_id() {
	uint8_t octet_length = 0;
	uint8_t byte = src[read_ptr];

	bool found_marker = false;
	for (uint8_t i = 0; i < 8; i++) {
		octet_length += 1;
		if ((byte & (128 >> i)) > 0) {
			found_marker = true;
			break;
		}
	}

	if (!found_marker) {
		ERR_PRINT("Failed to parse EBML ID");
		return 0;
	}

	uint64_t id = byte;
	for (uint32_t i = 1; i < octet_length; i++) {
		id = (id << 8) | src[read_ptr + i];
	}

	read_ptr += octet_length;
	return id;
}

uint64_t VideoStreamPlaybackMatroska::read_size() {
	uint8_t octet_length = 0;
	uint8_t byte = src[read_ptr];

	bool found_marker = false;
	for (uint8_t i = 0; i < 8; i++) {
		octet_length += 1;
		if ((byte & (128 >> i)) > 0) {
			found_marker = true;
			break;
		}
	}

	if (!found_marker) {
		ERR_PRINT("Failed to parse EBML size");
		return 1;
	}

	uint64_t size = byte ^ (1 << (8 - octet_length));
	for (uint32_t i = 1; i < octet_length; i++) {
		size = (size << 8) | src[read_ptr + i];
	}

	read_ptr += octet_length;
	return size;
}

int64_t VideoStreamPlaybackMatroska::read_int() {
	uint64_t size = read_size();

	int64_t value = 0;
	for (uint32_t i = 0; i < size; i++) {
		value = (value << 8) | src[read_ptr + i];
	}

	read_ptr += size;
	return value;
}

uint64_t VideoStreamPlaybackMatroska::read_uint() {
	uint64_t size = read_size();

	uint64_t value = 0;
	for (uint32_t i = 0; i < size; i++) {
		value = (value << 8) | src[read_ptr + i];
	}

	read_ptr += size;
	return value;
}

double VideoStreamPlaybackMatroska::read_float() {
	uint64_t size = read_size();

	char *src_big = (char *)src + read_ptr;
	if (size == 4) {
		char src_little[4];

		src_little[0] = src_big[3];
		src_little[1] = src_big[2];
		src_little[2] = src_big[1];
		src_little[3] = src_big[0];
		read_ptr += size;

		float value = 0;
		memcpy(&value, src_little, 4);
		return value;
	} else {
		char src_little[8];

		src_little[0] = src_big[7];
		src_little[1] = src_big[6];
		src_little[2] = src_big[5];
		src_little[3] = src_big[4];
		src_little[4] = src_big[3];
		src_little[5] = src_big[2];
		src_little[6] = src_big[1];
		src_little[7] = src_big[0];
		read_ptr += size;

		double value = 0;
		memcpy(&value, src_little, 8);
		return value;
	}
}

String VideoStreamPlaybackMatroska::read_string() {
	uint64_t size = read_size();

	Span<char> span = Span((const char *)src + read_ptr, size);

	String str = String();
	str.append_ascii(span);

	read_ptr += size;
	return str;
}

Error VideoStreamPlaybackMatroska::parse_ebml_header(EbmlHeader *r_header) {
	uint64_t header_size = read_size();
	size_t header_start = read_ptr;

	while (read_ptr < header_start + header_size) {
		uint64_t potential_id = read_id();

		if (potential_id == EBML_ID_VERSION) {
			r_header->version = read_uint();
			continue;
		}

		if (potential_id == EBML_ID_READ_VERSION) {
			r_header->read_version = read_uint();
			continue;
		}

		if (potential_id == EBML_ID_MAX_LENGTH) {
			r_header->max_id_length = read_uint();
			continue;
		}

		if (potential_id == EBML_ID_MAX_SIZE_LENGTH) {
			r_header->max_size_length = read_uint();
			continue;
		}

		if (potential_id == EBML_ID_DOC_TYPE) {
			r_header->doc_type = read_string();
			continue;
		}

		if (potential_id == EBML_ID_DOC_TYPE_VERSION) {
			r_header->doc_type_version = read_uint();
			continue;
		}

		if (potential_id == EBML_ID_DOC_TYPE_READ_VERSION) {
			r_header->doc_type_read_version = read_uint();
			continue;
		}

		if (potential_id == EBML_ID_DOC_TYPE_EXTENSION) {
			uint64_t extension_size = read_size();
			read_ptr += extension_size;
			continue;
		}

		if (potential_id == EBML_ID_DOC_TYPE_EXTENSION_NAME) {
			String extension_name = read_string();
			continue;
		}

		if (potential_id == EBML_ID_DOC_TYPE_EXTENSION_VERSION) {
			read_uint();
			continue;
		}

		_skip_id(potential_id);
	}

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_segment(Segment *r_segment) {
	uint64_t segment_size = read_size();
	size_t segment_start = read_ptr;

	bool use_seek_head = false;
	while (read_ptr < segment_start + segment_size) {
		uint64_t potential_id = read_id();

		if (potential_id == MATROSKA_ID_SEEK_HEAD) {
			parse_seek_head(&r_segment->seek_head);
			use_seek_head = true;
			break;
		}

		if (potential_id == MATROSKA_ID_SEGMENT_INFO) {
			parse_segment_info(&r_segment->info);
			continue;
		}

		if (potential_id == MATROSKA_ID_TRACKS) {
			parse_tracks(r_segment->tracks);
			continue;
		}

		if (potential_id == MATROSKA_ID_CHAPTERS) {
			parse_chapters();
			continue;
		}

		if (potential_id == MATROSKA_ID_CLUSTER) {
			Cluster cluster;
			parse_cluster(&cluster);
			continue;
		}

		if (potential_id == MATROSKA_ID_CUES) {
			parse_cues();
			continue;
		}

		if (potential_id == MATROSKA_ID_ATTACHEMENTS) {
			parse_attachments();
			continue;
		}

		if (potential_id == MATROSKA_ID_TAGS) {
			parse_tags();
			continue;
		}

		_skip_id(potential_id);
	}

	if (use_seek_head) {
		if (r_segment->seek_head.segment_info_position != 0) {
			read_ptr = segment_start + r_segment->seek_head.segment_info_position;
			read_id();
			parse_segment_info(&r_segment->info);
		}

		if (r_segment->seek_head.tracks_position != 0) {
			read_ptr = segment_start + r_segment->seek_head.tracks_position;
			read_id();
			parse_tracks(r_segment->tracks);
		}

		if (r_segment->seek_head.cues_position != 0) {
			read_ptr = segment_start + r_segment->seek_head.cues_position;
			read_id();
			parse_cues();

			for (Cluster &cluster : segment.clusters) {
				read_ptr = segment_start + cluster.position;
				read_id();
				parse_cluster(&cluster);
			}
		}

		if (r_segment->seek_head.tags_position != 0) {
			read_ptr = segment_start + r_segment->seek_head.tags_position;
			read_id();
			parse_tags();
		}
	}

	read_ptr = segment_start + segment_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_seek_head(SeekHead *r_seak_head) {
	uint64_t seek_head_size = read_size();
	size_t seek_head_start = read_ptr;

	while (read_ptr < seek_head_start + seek_head_size) {
		uint64_t potential_id = read_id();

		if (potential_id == MATROSKA_ID_SEEK) {
			uint64_t inner_size = read_size();
			size_t seek_start = read_ptr;

			struct {
				uint64_t id = 0;
				uint64_t position = 0;
			} seek;

			while (read_ptr < seek_start + inner_size) {
				uint64_t inner_id = read_id();

				if (inner_id == MATROSKA_ID_SEEK_ID) {
					seek.id = read_uint();
				}

				if (inner_id == MATROSKA_ID_SEEK_POSITION) {
					seek.position = read_uint();
				}
			}

			if (seek.id == MATROSKA_ID_SEGMENT_INFO) {
				r_seak_head->segment_info_position = seek.position;
			} else if (seek.id == MATROSKA_ID_TRACKS) {
				r_seak_head->tracks_position = seek.position;
			} else if (seek.id == MATROSKA_ID_CUES) {
				r_seak_head->cues_position = seek.position;
			} else if (seek.id == MATROSKA_ID_TAGS) {
				r_seak_head->tags_position = seek.position;
			}
		}
	}

	read_ptr = seek_head_start + seek_head_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_segment_info(SegmentInfo *r_segment_info) {
	uint64_t segment_info_size = read_size();
	size_t segment_info_start = read_ptr;

	while (read_ptr < segment_info_start + segment_info_size) {
		uint64_t id = read_id();

		// TODO use CRC?
		if (id == EBML_ID_CRC32) {
			uint64_t crc_size = read_size();
			read_ptr += crc_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_UUID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->uuid, src + read_ptr, uuid_size);
			read_ptr += uuid_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_FILENAME) {
			r_segment_info->filename = read_string();
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_PREV_UUID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->prev_uuid, src + read_ptr, uuid_size);
			read_ptr += uuid_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_PREV_FILENAME) {
			r_segment_info->prev_filename = read_string();
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_NEXT_UUID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->next_uuid, src + read_ptr, uuid_size);
			read_ptr += uuid_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_NEXT_FILENAME) {
			r_segment_info->next_filename = read_string();
			continue;
		}

		// TODO
		if (id == MATROSKA_ID_SEGMENT_INFO_FAMILY) {
			uint64_t family_size = read_size();
			read_ptr += family_size;
			continue;
		}

		// TODO
		if (id == MATROSKA_ID_SEGMENT_INFO_CHAPTER_TRANSLATE) {
			uint64_t chapter_size = read_size();
			read_ptr += chapter_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_TIME_SCALE) {
			r_segment_info->time_scale = read_uint();
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_DURATION) {
			r_segment_info->duration = read_float();
			continue;
		}

		// TODO
		if (id == MATROSKA_ID_SEGMENT_INFO_DATE_UTC) {
			uint64_t date_size = read_size();
			read_ptr += date_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_TITLE) {
			r_segment_info->title = read_string();
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_MUXING_APP) {
			r_segment_info->muxing_app = read_string();
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_WRITING_APP) {
			r_segment_info->writing_app = read_string();
			continue;
		}

		_skip_id(id);
	}

	read_ptr = segment_info_start + segment_info_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_tracks(Vector<Track> r_tracks) {
	uint64_t tracks_size = read_size();
	size_t tracks_start = read_ptr;

	while (read_ptr < tracks_start + tracks_size) {
		uint64_t id = read_id();

		// TODO use CRC
		if (id == EBML_ID_CRC32) {
			uint64_t crc_size = read_size();
			read_ptr += crc_size;
			continue;
		}

		if (id == MATROSKA_ID_TRACK_ENTRY) {
			uint64_t entry_size = read_size();
			size_t entry_start = read_ptr;

			Track track = {};
			while (read_ptr < entry_start + entry_size) {
				uint64_t inner_id = read_id();

				if (inner_id == EBML_ID_VOID) {
					uint64_t void_size = read_size();
					read_ptr += void_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_NUMBER) {
					track.track_number = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_UID) {
					track.track_uid = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_TYPE) {
					uint64_t track_type = read_uint();
					if (track_type != 1 && track_type != 2) {
						WARN_PRINT(vformat("Unknown track type %d", track_type));
					}
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_ENABLED) {
					track.flag_enabled = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_DEFAULT) {
					track.flag_default = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_FORCED) {
					track.flag_forced = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_HEARING_IMPAIRED) {
					track.flag_hearing_impaired = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_VISUAL_IMPAIRED) {
					track.flag_visual_impaired = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_TEXT_DESCRIPTIONS) {
					track.flag_text_descriptions = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_ORIGINAL) {
					track.flag_original = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_COMMENTARY) {
					track.flag_commentary = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_FLAG_LACING) {
					track.flag_lacing = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_DEFAULT_DURATION) {
					track.default_duration = read_uint();
					block_duration = double(track.default_duration) / 1.0e9;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_DEFAULT_DECODED_FIELD_DURATION) {
					track.default_decoded_field_duration = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_TIMESTAMP_SCALE) {
					track.track_timestamp_scale = read_float();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_MAX_BLOCK_ADDITION_ID) {
					track.max_block_addition_id = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_BLOCK_ADDITION_MAPPING) {
					uint64_t block_additions_size = read_size();
					read_ptr += block_additions_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_NAME) {
					track.name = read_string();
					continue;
				}

				// BCP47 language codes take priority over Matroska language codes
				if (inner_id == MATROSKA_ID_TRACK_LANGUAGE) {
					String language = read_string();
					if (track.language.is_empty()) {
						track.language = language;
					}
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_LANGUAGE_BCP47) {
					track.language = read_string();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_CODEC_ID) {
					track.codec_id = read_string();
					if (track.codec_id == "V_MPEG4/ISO/AVC") {
						video_stream_encoding = memnew(VideoStreamH264);
					} else if (track.codec_id == "V_AV1") {
						video_stream_encoding = memnew(VideoStreamAV1);
					}
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_CODEC_PRIVATE) {
					uint64_t codec_size = read_size();
					if (track.track_number == 1 && video_stream_encoding.is_valid()) {
						video_stream_encoding->parse_container_metadata(src + read_ptr, codec_size);
					}

					read_ptr += codec_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_CODEC_NAME) {
					track.coded_name = read_string();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_ATTACHMENT_LINK) {
					track.attachment_link = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_CODEC_DELAY) {
					track.codec_delay = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_SEEK_PRE_ROLL) {
					track.seek_pre_roll = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_TRANSLATE) {
					uint64_t track_translate_size = read_size();
					read_ptr += track_translate_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_VIDEO) {
					uint64_t video_size = read_size();
					size_t video_start = read_ptr;

					while (read_ptr < video_start + video_size) {
						uint64_t video_id = read_id();

						if (video_id == MATROSKA_ID_TRACK_VIDEO_PIXEL_WIDTH) {
							width = read_uint();
							continue;
						}

						if (video_id == MATROSKA_ID_TRACK_VIDEO_PIXEL_HEIGHT) {
							height = read_uint();
							continue;
						}

						_skip_id(video_id);
					}

					read_ptr = video_start + video_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_AUDIO) {
					uint64_t audio_size = read_size();
					read_ptr += audio_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_OPERATION) {
					uint64_t track_operation_size = read_size();
					read_ptr += track_operation_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_CONTENT_ENCODINGS) {
					uint64_t content_encodings_size = read_size();
					read_ptr += content_encodings_size;
					continue;
				}

				_skip_id(inner_id);
			}

			r_tracks.push_back(track);
			continue;
		}

		_skip_id(id);
	}

	read_ptr = tracks_start + tracks_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_chapters() {
	uint64_t chapters_size = read_size();
	read_ptr += chapters_size;

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_cluster(Cluster *r_cluster) {
	uint64_t cluster_size = read_size();
	size_t cluster_start = read_ptr;

	while (read_ptr < cluster_start + cluster_size) {
		uint64_t id = read_id();

		if (id == EBML_ID_CRC32) {
			uint64_t crc_size = read_size();
			read_ptr += crc_size;
			continue;
		}

		if (id == MATROSKA_ID_CLUSTER_TIMESTAMP) {
			r_cluster->time = read_uint();
			continue;
		}

		if (id == MATROSKA_ID_CLUSTER_SIMPLE_BLOCK) {
			uint64_t block_size = read_size();
			size_t block_start = read_ptr;

			uint64_t target_track = read_size();

			//skip timestamp
			read_ptr += 2;

			uint8_t flags = src[read_ptr];
			uint8_t lacing = flags & 0x06;
			// ...other flags
			read_ptr += 1;

			Vector<size_t> lace_sizes;
			if (lacing == 0) {
				// No lacing: there is only 1 block
				lace_sizes.push_back(block_size - (read_ptr - block_start));
			} else if (lacing == 1) {
				// Xiph lacing
				size_t total_size = 0;

				// First byte: lace count -1
				size_t lace_count = src[read_ptr];
				read_ptr += 1;

				for (size_t i = 0; i < lace_count; i++) {
					size_t lace_size = 0;
					while (src[read_ptr] == 255) {
						total_size += 255;
						lace_size += 255;
						read_ptr += 1;
					}

					total_size += src[read_ptr];
					lace_size += src[read_ptr];
					read_ptr += 1;

					lace_sizes.push_back(lace_size);
				}

				// Final lace's size is derived from total block size
				lace_sizes.push_back(block_size - total_size - (read_ptr - block_start));
			} else if (lacing == 2) {
				// Fixed size lacing

				// First byte: lace count -1
				size_t lace_count = src[read_ptr] + 1;
				read_ptr += 1;

				// All laces have the same size
				size_t lace_size = (block_size - (read_ptr - block_start)) / lace_count;
				for (size_t i = 0; i < lace_count; i++) {
					lace_sizes.push_back(lace_size);
				}
			} else if (lacing == 3) {
				// EBML lacing
				size_t total_size = 0;

				// First byte: lace count -1
				size_t lace_count = src[read_ptr];
				read_ptr += 1;

				// First size is read normally
				size_t previous_size = read_size();
				lace_sizes.push_back(previous_size);

				for (size_t i = 1; i < lace_count; i++) {
					// Subsequent sizes are used as a delta from the previous
					int64_t lace_start = read_ptr;
					int64_t lace_vint = read_size();
					int64_t lace_delta = lace_vint + 1 - (1 << (7 * (read_ptr - lace_start) - 1));

					previous_size += lace_delta;
					lace_sizes.push_back(previous_size);
				}

				// Final lace's size is derived from total block size
				lace_sizes.push_back(block_size - total_size - (read_ptr - block_start));
			}

			for (size_t lace_size : lace_sizes) {
				if (target_track == 1 && video_stream_encoding.is_valid()) {
					Vector<VideoStreamEncoding::ParsedFrame> frames;
					video_stream_encoding->parse_container_block(src + read_ptr, lace_size, &frames);
					for (int64_t i = 0; i < frames.size(); i++) {
						Cluster::Block block = {};

						block.header_position = read_ptr + frames[i].header_offset;
						block.header_size = frames[i].header_size;
						block.frame_size = frames[i].frame_size;

						r_cluster->blocks.append(block);
					}
				}

				read_ptr += lace_size;
			}

			read_ptr = block_start + block_size;
			continue;
		}

		if (id == MATROSKA_ID_CLUSTER_BLOCK_GROUP) {
			uint64_t skipped_size = read_size();
			read_ptr += skipped_size;
			continue;
		}

		_skip_id(id);
	}

	read_ptr = cluster_start + cluster_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_cues() {
	uint64_t cues_size = read_size();
	size_t cues_start = read_ptr;

	while (read_ptr < cues_start + cues_size) {
		uint64_t id = read_id();

		if (id == EBML_ID_CRC32) {
			uint64_t crc_size = read_size();
			read_ptr += crc_size;
			continue;
		}

		if (id == MATROSKA_ID_CUES_CUE_POINT) {
			uint64_t element_size = read_size();
			size_t element_start = read_ptr;

			Cluster cluster = {};
			while (read_ptr < element_start + element_size) {
				uint64_t element_id = read_id();

				if (element_id == MATROSKA_ID_CUES_CUE_POINT_TIME) {
					cluster.time = read_uint();
					continue;
				}

				if (element_id == MATROSKA_ID_CUES_CUE_POINT_POSITIONS) {
					uint64_t positions_size = read_size();
					size_t positions_start = read_ptr;

					while (read_ptr < positions_start + positions_size) {
						uint64_t positions_id = read_id();

						if (positions_id == MATROSKA_ID_CUES_CUE_POINT_POSITIONS_TRACK) {
							cluster.target_track = read_uint();
							continue;
						}

						if (positions_id == MATROSKA_ID_CUES_CUE_POINT_POSITIONS_CLUSTER_POSITION) {
							cluster.position = read_uint();
							continue;
						}

						// TODO is relative position at all useful?
						if (positions_id == MATROSKA_ID_CUES_CUE_POINT_POSITIONS_RELATIVE_POSITION) {
							read_uint();
							continue;
						}

						_skip_id(positions_id);
					}

					continue;
				}

				_skip_id(element_id);
			}

			segment.clusters.push_back(cluster);
			continue;
		}

		_skip_id(id);
	}

	read_ptr = cues_start + cues_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_attachments() {
	uint64_t attachements_size = read_size();
	read_ptr += attachements_size;

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_tags() {
	uint64_t tags_size = read_size();
	read_ptr += tags_size;

	return OK;
}

void VideoStreamPlaybackMatroska::decode_frame() {
	Cluster cluster = segment.clusters[cluster_index];
	Cluster::Block block = cluster.blocks[block_index];

	Error err;
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ, &err);

	file->seek(block.header_position);
	Vector<uint8_t> frame_data = file->get_buffer(block.header_size);

	uint8_t *src_buffer = video_stream_encoding->queue_decode(frame_data, block.frame_size);
	file->seek(block.header_position);
	file->get_buffer(src_buffer, block.frame_size);

	block_index += 1;
	if (block_index == cluster.blocks.size()) {
		block_index = 0;
		cluster_index += 1;
	}
}

void VideoStreamPlaybackMatroska::present_frame() {
	Vector<uint8_t> data = video_stream_encoding->present_frame();
	block_position += block_duration;

	Ref<Image> frame;
	frame.instantiate();
	frame->set_data(width, height, false, Image::FORMAT_RGBA8, data);

	image_texture->set_image(frame);
}

Error VideoStreamPlaybackMatroska::set_file(const String &p_file) {
	path = p_file;
	Vector<uint8_t> buffer = FileAccess::get_file_as_bytes(p_file);

	src = buffer.ptr();
	while (read_ptr < (size_t)buffer.size()) {
		uint64_t id = read_id();

		if (id == EBML_ID_HEADER) {
			Error err = parse_ebml_header(&header);
			if (err != OK) {
				ERR_PRINT("error parsing ebml header");
			}
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT) {
			Error err = parse_segment(&segment);
			if (err != OK) {
				ERR_PRINT("error parsing matroska segment");
			}
			break;
		}

		_skip_id(id);
	}

	//TODO: error if there's no compatible audio
	ERR_FAIL_COND_V_MSG(video_stream_encoding.is_null(), FAILED, "No compatible video format found");

	return OK;
}

void VideoStreamPlaybackMatroska::play() {
	playing = true;

	if (video_stream_encoding.is_null()) {
		return;
	}

	RD::VideoSessionProfile video_session_info = {};
	video_session_info.max_width = width;
	video_session_info.max_height = height;

	// Decode stage objects
	//TODO: be more precise with ycbcr sampler
	RD::SamplerState sampler_info;
	sampler_info.enable_ycbcr = true;

	RD::TextureFormat dst_rgba_format;
	dst_rgba_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	dst_rgba_format.width = width;
	dst_rgba_format.height = height;
	dst_rgba_format.depth = 1;
	dst_rgba_format.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	video_stream_encoding->initialize(video_session_info, sampler_info, dst_rgba_format);

	for (size_t i = 0; i < buffered_frames; i++) {
		decode_frame();
	}

	video_stream_encoding->submit_decode();
	present_frame();
}

void VideoStreamPlaybackMatroska::stop() {
	playing = false;
}

bool VideoStreamPlaybackMatroska::is_playing() const {
	return playing;
}

void VideoStreamPlaybackMatroska::set_paused(bool p_paused) {
	// TODO
}

bool VideoStreamPlaybackMatroska::is_paused() const {
	// TODO
	return false;
}

double VideoStreamPlaybackMatroska::get_length() const {
	double scale = double(segment.info.time_scale) / 1.0e9;
	return scale * segment.info.duration;
}

double VideoStreamPlaybackMatroska::get_playback_position() const {
	return playback_position;
}

void VideoStreamPlaybackMatroska::seek(double p_time) {
	// TODO
}

Ref<Texture2D> VideoStreamPlaybackMatroska::get_texture() const {
	return image_texture;
}

// TODO
void VideoStreamPlaybackMatroska::update(double p_delta) {
	playback_position += p_delta;
	if (playback_position < block_position + block_duration) {
		return;
	}

	// TODO: audio
	present_frame();
	decode_frame();
	video_stream_encoding->submit_decode();
}

int VideoStreamPlaybackMatroska::get_channels() const {
	return 1;
}

int VideoStreamPlaybackMatroska::get_mix_rate() const {
	// TODO
	return AudioDriverManager::DEFAULT_MIX_RATE;
}

void VideoStreamPlaybackMatroska::set_audio_track(int p_idx) {
	// TODO
}

VideoStreamPlaybackMatroska::VideoStreamPlaybackMatroska() {
	image_texture.instantiate();
}

Ref<Resource> ResourceFormatLoaderMatroska::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		if (r_error) {
			*r_error = ERR_CANT_OPEN;
		}
		return Ref<Resource>();
	}

	Ref<VideoStreamMatroska> matroska_stream;
	matroska_stream.instantiate();
	matroska_stream->set_file(p_path);

	if (r_error) {
		*r_error = OK;
	}

	return matroska_stream;
}

void ResourceFormatLoaderMatroska::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("mkv");
	p_extensions->push_back("webm");
}

bool ResourceFormatLoaderMatroska::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "VideoStream");
}

String ResourceFormatLoaderMatroska::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "mkv" || el == "webm") {
		return "VideoStreamMatroska";
	}
	return "";
}
