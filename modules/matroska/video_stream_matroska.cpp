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
#include "servers/audio/audio_server.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_device_binds.h"
#include "servers/rendering/rendering_server.h"
#include "ycbcr_sampler.glsl.gen.h"

#include <cstdint>

void VideoStreamMatroska::set_audio_track(int p_track) {
	audio_track = p_track;
}

Ref<VideoStreamPlayback> VideoStreamMatroska::instantiate_playback() {
	Ref<VideoStreamPlaybackMatroska> stream_playback;
	stream_playback.instantiate();
	stream_playback->set_audio_track(audio_track);
	stream_playback->set_file(file);
	return stream_playback;
}

void VideoStreamPlaybackMatroska::_skip_id(uint64_t p_id) {
	uint64_t size = read_size();
	WARN_PRINT(vformat("Unhandled element with ID [%x] of size [%d]", p_id, size));
	src += size;
}

uint64_t VideoStreamPlaybackMatroska::read_id() {
	uint8_t octet_length = 0;
	uint8_t byte = src[0];

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
		id = (id << 8) | src[i];
	}

	src += octet_length;
	return id;
}

uint64_t VideoStreamPlaybackMatroska::read_size() {
	uint8_t octet_length = 0;
	uint8_t byte = src[0];

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
		size = (size << 8) | src[i];
	}

	src += octet_length;
	return size;
}

int64_t VideoStreamPlaybackMatroska::read_int() {
	uint64_t size = read_size();

	int64_t value = 0;
	for (uint32_t i = 0; i < size; i++) {
		value = (value << 8) | src[i];
	}

	src += size;
	return value;
}

uint64_t VideoStreamPlaybackMatroska::read_uint() {
	uint64_t size = read_size();

	uint64_t value = 0;
	for (uint32_t i = 0; i < size; i++) {
		value = (value << 8) | src[i];
	}

	src += size;
	return value;
}

double VideoStreamPlaybackMatroska::read_float() {
	uint64_t size = read_size();

	char *src_big = (char *)src;
	if (size == 4) {
		char src_little[4];

		src_little[0] = src_big[3];
		src_little[1] = src_big[2];
		src_little[2] = src_big[1];
		src_little[3] = src_big[0];
		src += size;

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
		src += size;

		double value = 0;
		memcpy(&value, src_little, 8);
		return value;
	}
}

String VideoStreamPlaybackMatroska::read_string() {
	uint64_t size = read_size();

	Span<char> span = Span((const char *)src, size);

	String str = String();
	str.append_ascii(span);

	src += size;
	return str;
}

Error VideoStreamPlaybackMatroska::parse_ebml_header(EbmlHeader *r_header) {
	uint64_t header_size = read_size();
	const uint8_t *header_start = src;

	while (src < header_start + header_size) {
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
			src += extension_size;
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
	const uint8_t *segment_start = src;

	bool use_seek_head = false;
	while (src < segment_start + segment_size) {
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
			src = segment_start + r_segment->seek_head.segment_info_position;
			read_id();
			parse_segment_info(&r_segment->info);
		}

		if (r_segment->seek_head.tracks_position != 0) {
			src = segment_start + r_segment->seek_head.tracks_position;
			read_id();
			parse_tracks(r_segment->tracks);
		}

		if (r_segment->seek_head.cues_position != 0) {
			src = segment_start + r_segment->seek_head.cues_position;
			read_id();
			parse_cues();

			for (Cluster &cluster : segment.clusters) {
				src = segment_start + cluster.position;
				read_id();
				parse_cluster(&cluster);
			}
		}

		if (r_segment->seek_head.tags_position != 0) {
			src = segment_start + r_segment->seek_head.tags_position;
			read_id();
			parse_tags();
		}
	}

	src = segment_start + segment_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_seek_head(SeekHead *r_seak_head) {
	uint64_t seek_head_size = read_size();
	const uint8_t *seek_head_start = src;

	while (src < seek_head_start + seek_head_size) {
		uint64_t potential_id = read_id();

		if (potential_id == MATROSKA_ID_SEEK) {
			int64_t inner_size = read_size();
			const uint8_t *seek_start = src;

			struct {
				uint64_t id = 0;
				uint64_t position = 0;
			} seek;

			while (src < seek_start + inner_size) {
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

	src = seek_head_start + seek_head_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_segment_info(SegmentInfo *r_segment_info) {
	int64_t segment_info_size = read_size();
	const uint8_t *segment_info_start = src;

	while (src < segment_info_start + segment_info_size) {
		uint64_t id = read_id();

		// TODO use CRC?
		if (id == EBML_ID_CRC32) {
			uint64_t crc_size = read_size();
			src += crc_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_UUID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->uuid, src, uuid_size);
			src += uuid_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_FILENAME) {
			r_segment_info->filename = read_string();
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_PREV_UUID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->prev_uuid, src, uuid_size);
			src += uuid_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_PREV_FILENAME) {
			r_segment_info->prev_filename = read_string();
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_NEXT_UUID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->next_uuid, src, uuid_size);
			src += uuid_size;
			continue;
		}

		if (id == MATROSKA_ID_SEGMENT_INFO_NEXT_FILENAME) {
			r_segment_info->next_filename = read_string();
			continue;
		}

		// TODO
		if (id == MATROSKA_ID_SEGMENT_INFO_FAMILY) {
			uint64_t family_size = read_size();
			src += family_size;
			continue;
		}

		// TODO
		if (id == MATROSKA_ID_SEGMENT_INFO_CHAPTER_TRANSLATE) {
			uint64_t chapter_size = read_size();
			src += chapter_size;
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
			src += date_size;
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

	src = segment_info_start + segment_info_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_tracks(Vector<Track> r_tracks) {
	int64_t tracks_size = read_size();
	const uint8_t *tracks_start = src;

	while (src < tracks_start + tracks_size) {
		uint64_t id = read_id();

		// TODO use CRC
		if (id == EBML_ID_CRC32) {
			uint64_t crc_size = read_size();
			src += crc_size;
			continue;
		}

		if (id == MATROSKA_ID_TRACK_ENTRY) {
			int64_t entry_size = read_size();
			const uint8_t *entry_start = src;

			Track track = {};
			while (src < entry_start + entry_size) {
				uint64_t inner_id = read_id();

				if (inner_id == EBML_ID_VOID) {
					uint64_t void_size = read_size();
					src += void_size;
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
						print_line(vformat("Unknown track type %d", track_type));
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
					src += block_additions_size;
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
					print_line(vformat("Track #%d (%s)", track.track_number, track.codec_id));
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
						video_stream_encoding->parse_container_metadata(src, codec_size);
					}

					src += codec_size;
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
					src += track_translate_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_VIDEO) {
					int64_t video_size = read_size();
					const uint8_t *video_start = src;

					while (src < video_start + video_size) {
						uint64_t video_id = read_id();

						if (video_id == MATROSKA_ID_TRACK_VIDEO_PIXEL_WIDTH) {
							width = read_uint();
							continue;
						}

						if (video_id == MATROSKA_ID_TRACK_VIDEO_PIXEL_HEIGHT) {
							height = read_uint();
							continue;
						}
					}

					src = video_start + video_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_AUDIO) {
					uint64_t audio_size = read_size();
					src += audio_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_OPERATION) {
					uint64_t track_operation_size = read_size();
					src += track_operation_size;
					continue;
				}

				if (inner_id == MATROSKA_ID_TRACK_CONTENT_ENCODINGS) {
					uint64_t content_encodings_size = read_size();
					src += content_encodings_size;
					continue;
				}

				_skip_id(inner_id);
			}

			r_tracks.push_back(track);
			continue;
		}

		_skip_id(id);
	}

	src = tracks_start + tracks_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_chapters() {
	uint64_t chapters_size = read_size();
	src += chapters_size;

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_cluster(Cluster *r_cluster) {
	int64_t cluster_size = read_size();
	const uint8_t *cluster_start = src;

	Vector<Cluster::Block> blocks;
	while (src < cluster_start + cluster_size) {
		uint64_t id = read_id();

		if (id == EBML_ID_CRC32) {
			uint64_t crc_size = read_size();
			src += crc_size;
			continue;
		}

		if (id == MATROSKA_ID_CLUSTER_TIMESTAMP) {
			r_cluster->time = read_uint();
			continue;
		}

		if (id == MATROSKA_ID_CLUSTER_SIMPLE_BLOCK) {
			Cluster::Block block = {};

			uint64_t block_size = read_size();
			const uint8_t *block_start = src;

			uint64_t target_track = read_size();

			uint16_t timestamp = (src[0] << 8) | src[1];
			src += 2;

			if (target_track == 1) {
				r_cluster->time_to_layer.insert(timestamp, r_cluster->time_to_layer.size());
			}

			uint8_t flags = src[0];
			block.key = (flags & 0x80) > 0;
			block.invisible = (flags & 0x08) > 0;
			uint8_t lacing = flags & 0x06;
			block.discardable = (flags & 0x01) > 0;
			src += 1;

			if (lacing != 0) {
				ERR_PRINT("UNUSABLE LACING");
			}

			if (target_track == 1) {
				block.position = src - origin;
				block.size = block_size - 4;
				r_cluster->blocks.append(block);
			}

			src = block_start + block_size;
			continue;
		}

		if (id == MATROSKA_ID_CLUSTER_BLOCK_GROUP) {
			uint64_t skipped_size = read_size();
			src += skipped_size;
			continue;
		}

		_skip_id(id);
	}

	src = cluster_start + cluster_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_cues() {
	int64_t cues_size = read_size();
	const uint8_t *cues_start = src;

	while (src < cues_start + cues_size) {
		uint64_t id = read_id();

		if (id == EBML_ID_CRC32) {
			uint64_t crc_size = read_size();
			src += crc_size;
			continue;
		}

		if (id == MATROSKA_ID_CUES_CUE_POINT) {
			int64_t element_size = read_size();
			const uint8_t *element_start = src;

			Cluster cluster = {};
			while (src < element_start + element_size) {
				uint64_t element_id = read_id();

				if (element_id == MATROSKA_ID_CUES_CUE_POINT_TIME) {
					cluster.time = read_uint();
					continue;
				}

				if (element_id == MATROSKA_ID_CUES_CUE_POINT_POSITIONS) {
					int64_t positions_size = read_size();
					const uint8_t *positions_start = src;

					while (src < positions_start + positions_size) {
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

	src = cues_start + cues_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_attachments() {
	uint64_t attachements_size = read_size();
	src += attachements_size;

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_tags() {
	uint64_t tags_size = read_size();
	src += tags_size;

	return OK;
}

void VideoStreamPlaybackMatroska::set_file(const String &p_file) {
	path = p_file;
	Vector<uint8_t> buffer = FileAccess::get_file_as_bytes(p_file);

	origin = buffer.ptr();
	src = buffer.ptr();

	while (src < buffer.ptr() + buffer.size()) {
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

	if (video_stream_encoding.is_null()) {
		return;
	}

	// All Matroska metadata is done, now create the yuv sampler, yuv image pool and dst image
	video_stream_encoding->set_rendering_device(local_device);

	// Decode stage objects
	//TODO: be more precise with ycbcr sampler
	RD::SamplerState sampler_info;
	sampler_info.enable_ycbcr = true;

	yuv_sampler = video_stream_encoding->create_texture_sampler(sampler_info);

	RD::TextureFormat dst_yuv_format;
	dst_yuv_format.format = RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	dst_yuv_format.width = width;
	dst_yuv_format.height = height + 8; //TODO: stream encoding uses a mutable reference not a copy
	dst_yuv_format.depth = 1;
	dst_yuv_format.usage_bits = RD::TEXTURE_USAGE_VIDEO_DECODE_DST_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	for (uint32_t i = 0; i < 2; i++) {
		dst_yuv_pool.push_back(video_stream_encoding->create_texture(dst_yuv_format));
	}

	// Compute Shader objects
	RD::TextureFormat dst_rgba_format;
	dst_rgba_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	dst_rgba_format.width = width;
	dst_rgba_format.height = height;
	dst_rgba_format.depth = 1;
	dst_rgba_format.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	for (uint32_t i = 0; i < 2; i++) {
		dst_rgba_pool.push_back(local_device->texture_create(dst_rgba_format, RD::TextureView()));
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform immutable_sampler;
	immutable_sampler.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	immutable_sampler.binding = 0;
	immutable_sampler.immutable_sampler = true;
	immutable_sampler.append_id(yuv_sampler);
	immutable_sampler.append_id(dst_rgba_pool[0]);
	uniforms.push_back(immutable_sampler);

	Ref<RDShaderFile> yuv_shader_src;
	yuv_shader_src.instantiate();
	yuv_shader_src->parse_versions_from_text(ycbcr_sampler_shader_glsl);

	Vector<RD::ShaderStageSPIRVData> yuv_spirv = yuv_shader_src->get_spirv_stages();
	Vector<uint8_t> yuv_bytecode = local_device->shader_compile_binary_from_spirv(yuv_spirv);

	yuv_shader = local_device->shader_create_placeholder();
	yuv_shader = local_device->shader_create_from_bytecode_with_samplers(yuv_bytecode, yuv_shader, uniforms);
	yuv_pipeline = local_device->compute_pipeline_create(yuv_shader);
}

void VideoStreamPlaybackMatroska::play() {
	playing = true;

	if (video_stream_encoding.is_null()) {
		return;
	}

	Error err;
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ, &err);

	Cluster cluster = segment.clusters[0];
	print_line(vformat("------------Begin Matroska Cluster [%d]------------", cluster.blocks.size()));

	for (uint32_t i = 0; i < 2; i++) {
		RID dst_yuv = dst_yuv_pool[i];
		Cluster::Block block = cluster.blocks[i];

		file->seek(block.position);
		Vector<uint8_t> frame = file->get_buffer(block.size);

		video_stream_encoding->parse_container_block(frame, dst_yuv);
	}

	local_device->video_session_end();
	print_line("------------End Matroska Cluster------------");

	Vector<RD::Uniform> uniforms;
	RD::Uniform src_texture = {};
	src_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	src_texture.binding = 0;
	src_texture.append_id(yuv_sampler);
	src_texture.append_id(dst_yuv_pool[0]);
	uniforms.push_back(src_texture);

	RD::Uniform dst_texture = {};
	dst_texture.uniform_type = RD::UNIFORM_TYPE_IMAGE;
	dst_texture.binding = 1;
	dst_texture.append_id(dst_rgba_pool[0]);
	uniforms.push_back(dst_texture);

	RID uniform_set = local_device->uniform_set_create(uniforms, yuv_shader, 0);

	// Bind things
	RD::ComputeListID compute_list = local_device->compute_list_begin();
	local_device->compute_list_bind_compute_pipeline(compute_list, yuv_pipeline);
	local_device->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
	local_device->compute_list_dispatch(compute_list, 1920, 1080, 1);
	local_device->compute_list_end();

	// Execute
	local_device->submit();
	local_device->sync();
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
	// TODO
	return 0.0;
}

void VideoStreamPlaybackMatroska::seek(double p_time) {
	// TODO
}

Ref<Texture2D> VideoStreamPlaybackMatroska::get_texture() const {
	return image_texture;
}

// TODO
void VideoStreamPlaybackMatroska::update(double p_delta) {
	RID src_rgba = dst_rgba_pool[0];
	Vector<uint8_t> data = local_device->texture_get_data(src_rgba, 0);

	Ref<Image> frame;
	frame.instantiate();
	frame->set_data(width, height, false, Image::FORMAT_RGBA8, data);

	image_texture->set_image(frame);
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

	local_device = RenderingServer::get_singleton()->create_local_rendering_device();
}

VideoStreamPlaybackMatroska::~VideoStreamPlaybackMatroska() {
	if (local_device) {
		memdelete(local_device);
	}
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
