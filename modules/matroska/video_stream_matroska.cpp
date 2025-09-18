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
#include "scene/resources/texture.h"
#include "servers/audio_server.h"
#include "video_stream_h264.h"

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
	int64_t header_size = read_size();
	const uint8_t *header_start = src;

	while (src - header_start < header_size) {
		uint64_t potential_id = read_id();

		if (potential_id == EBML_VERSION_ID) {
			r_header->version = read_uint();
			continue;
		}

		if (potential_id == EBML_READ_VERSION_ID) {
			r_header->read_version = read_uint();
			continue;
		}

		if (potential_id == EBML_MAX_ID_LENGTH_ID) {
			r_header->max_id_length = read_uint();
			continue;
		}

		if (potential_id == EBML_MAX_SIZE_LENGTH_ID) {
			r_header->max_size_length = read_uint();
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_ID) {
			r_header->doc_type = read_string();
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_VERSION_ID) {
			r_header->doc_type_version = read_uint();
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_READ_VERSION_ID) {
			r_header->doc_type_read_version = read_uint();
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_EXTENSION_ID) {
			uint64_t extension_size = read_size();
			src += extension_size;
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_EXTENSION_NAME_ID) {
			String extension_name = read_string();
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_EXTENSION_VERSION_ID) {
			read_uint();
			continue;
		}

		uint64_t size = read_size();
		src += size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", potential_id, size));
	}

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_segment(Segment *r_segment) {
	int64_t segment_size = read_size();
	const uint8_t *segment_start = src;

	r_segment->start = src - origin;

	bool use_seek_head = false;
	while (src - segment_start < segment_size) {
		uint64_t potential_id = read_id();

		if (potential_id == MATROSKA_SEEK_HEAD_ID) {
			parse_seek_head(&r_segment->seek_head);
			use_seek_head = true;
			break;
		}

		if (potential_id == MATROSKA_SEGMENT_INFO_ID) {
			parse_segment_info(&r_segment->info);
			continue;
		}

		if (potential_id == MATROSKA_TRACKS_ID) {
			parse_tracks(r_segment->tracks);
			continue;
		}

		if (potential_id == MATROSKA_CHAPTERS_ID) {
			parse_chapters();
			continue;
		}

		if (potential_id == MATROSKA_CLUSTER_ID) {
			Cluster cluster;
			parse_cluster(&cluster);
			continue;
		}

		if (potential_id == MATROSKA_CUES_ID) {
			parse_cues();
			continue;
		}

		if (potential_id == MATROSKA_ATTACHEMENTS_ID) {
			parse_attachments();
			continue;
		}

		if (potential_id == MATROSKA_TAGS_ID) {
			parse_tags();
			continue;
		}

		uint64_t size = read_size();
		src += size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", potential_id, size));
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

			for (Cluster &cluster : clusters) {
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
	int64_t seek_head_size = read_size();
	const uint8_t *seek_head_start = src;

	while (src - seek_head_start < seek_head_size) {
		uint64_t potential_id = read_id();

		if (potential_id == MATROSKA_SEEK_ID) {
			int64_t inner_size = read_size();
			const uint8_t *seek_start = src;

			struct {
				uint64_t id = 0;
				uint64_t position = 0;
			} seek;

			while (src - seek_start < inner_size) {
				uint64_t inner_id = read_id();

				if (inner_id == MATROSKA_SEEK_TARGET_ID) {
					seek.id = read_uint();
				}

				if (inner_id == MATROSKA_SEEK_POSITION_ID) {
					seek.position = read_uint();
				}
			}

			if (seek.id == MATROSKA_SEGMENT_INFO_ID) {
				r_seak_head->segment_info_position = seek.position;
			} else if (seek.id == MATROSKA_TRACKS_ID) {
				r_seak_head->tracks_position = seek.position;
			} else if (seek.id == MATROSKA_CUES_ID) {
				r_seak_head->cues_position = seek.position;
			} else if (seek.id == MATROSKA_TAGS_ID) {
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

	while (src - segment_info_start < segment_info_size) {
		uint64_t id = read_id();

		// TODO use CRC?
		if (id == EBML_CRC32_ID) {
			uint64_t size = read_size();
			src += size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_UUID_ID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->uuid, src, uuid_size);
			src += uuid_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_FILENAME_ID) {
			r_segment_info->filename = read_string();
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_PREV_UUID_ID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->prev_uuid, src, uuid_size);
			src += uuid_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_PREV_FILENAME_ID) {
			r_segment_info->prev_filename = read_string();
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_NEXT_UUID_ID) {
			uint64_t uuid_size = read_size();
			memcpy(r_segment_info->next_uuid, src, uuid_size);
			src += uuid_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_NEXT_FILENAME_ID) {
			r_segment_info->next_filename = read_string();
			continue;
		}

		// TODO
		if (id == MATROSKA_SEGMENT_INFO_FAMILY_ID) {
			uint64_t family_size = read_size();
			src += family_size;
			continue;
		}

		// TODO
		if (id == MATROSKA_SEGMENT_INFO_CHAPTER_TRANSLATE_ID) {
			uint64_t chapter_size = read_size();
			src += chapter_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_TIME_SCALE_ID) {
			r_segment_info->time_scale = read_uint();
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_DURATION_ID) {
			r_segment_info->duration = read_float();
			continue;
		}

		// TODO
		if (id == MATROSKA_SEGMENT_INFO_DATE_UTC_ID) {
			uint64_t date_size = read_size();
			src += date_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_TITLE_ID) {
			r_segment_info->title = read_string();
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_MUXING_APP_ID) {
			r_segment_info->muxing_app = read_string();
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_WRITING_APP_ID) {
			r_segment_info->writing_app = read_string();
			continue;
		}

		uint64_t skipped_size = read_size();
		src += skipped_size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", id, skipped_size));
	}

	src = segment_info_start + segment_info_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_tracks(Vector<Track> r_tracks) {
	int64_t tracks_size = read_size();
	const uint8_t *tracks_start = src;

	while (src - tracks_start < tracks_size) {
		uint64_t id = read_id();

		// TODO use CRC
		if (id == EBML_CRC32_ID) {
			uint64_t crc_size = read_size();
			src += crc_size;
			continue;
		}

		if (id == MATROSKA_TRACK_ENTRY_ID) {
			int64_t entry_size = read_size();
			const uint8_t *entry_start = src;

			Track track = {};
			while (src - entry_start < entry_size) {
				uint64_t inner_id = read_id();

				if (inner_id == 0xec) {
					uint64_t void_size = read_size();
					src += void_size;
					continue;
				}

				if (inner_id == MATROSKA_TRACK_NUMBER_ID) {
					track.track_number = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_UID_ID) {
					track.track_uid = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_TYPE) {
					uint64_t track_type = read_uint();
					if (track_type != 1 && track_type != 2) {
						print_line(vformat("Unknown track type %d", track_type));
					}
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_ENABLED) {
					track.flag_enabled = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_DEFAULT) {
					track.flag_default = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_FORCED) {
					track.flag_forced = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_HEARING_IMPAIRED) {
					track.flag_hearing_impaired = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_VISUAL_IMPAIRED) {
					track.flag_visual_impaired = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_TEXT_DESCRIPTIONS) {
					track.flag_text_descriptions = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_ORIGINAL) {
					track.flag_original = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_COMMENTARY) {
					track.flag_commentary = read_uint();
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_LACING) {
					track.flag_lacing = read_uint();
					continue;
				}

				if (inner_id == MATRSOKA_TRACK_DEFAULT_DURATION) {
					track.default_duration = read_uint();
					continue;
				}

				if (inner_id == MATRSOKA_TRACK_DEFAULT_DECODED_FIELD_DURATION) {
					track.default_decoded_field_duration = read_uint();
					continue;
				}

				if (inner_id == 0x23314F) {
					track.track_timestamp_scale = read_float();
					continue;
				}

				if (inner_id == 0x55ee) {
					track.max_block_addition_id = read_uint();
					continue;
				}

				// block additions mapping
				if (inner_id == 0x41E4) {
					uint64_t block_additions_size = read_size();
					src += block_additions_size;
					continue;
				}

				if (inner_id == 0x536E) {
					track.name = read_string();
					continue;
				}

				// BCP47 language codes take priority over Matroska language codes
				if (inner_id == 0x22b59c) {
					String language = read_string();
					if (track.language.is_empty()) {
						track.language = language;
					}
					continue;
				}

				if (inner_id == 0x22B59D) {
					track.language = read_string();
					continue;
				}

				if (inner_id == 0x86) {
					track.codec_id = read_string();
					print_line(vformat("Track #%d (%s)", track.track_number, track.codec_id));
					if (track.codec_id == "V_MPEG4/ISO/AVC") {
						video_stream_encoding = memnew(VideoStreamH264);
					}
					continue;
				}

				if (inner_id == 0x63A2) {
					uint64_t codec_size = read_size();
					if (track.track_number == 1 && video_stream_encoding.is_valid()) {
						video_stream_encoding->parse_container_metadata(src, codec_size);
					}

					src += codec_size;
					continue;
				}

				if (inner_id == 0x258688) {
					track.coded_name = read_string();
					continue;
				}

				if (inner_id == 0x7446) {
					track.attachment_link = read_uint();
					continue;
				}

				if (inner_id == 0x56aa) {
					track.codec_delay = read_uint();
					continue;
				}

				if (inner_id == 0x56BB) {
					track.seek_pre_roll = read_uint();
					continue;
				}

				if (inner_id == 0x6624) {
					uint64_t track_translate_size = read_size();
					src += track_translate_size;
					continue;
				}

				if (inner_id == 0xe0) {
					int64_t video_size = read_size();
					const uint8_t *video_start = src;

					while (src - video_start < video_size) {
						uint64_t video_id = read_id();

						if (video_id == 0xB0) {
							width = read_uint();
							print_line(vformat("width %d", width));
							continue;
						}

						if (video_id == 0xBA) {
							height = read_uint();
							print_line(vformat("height %d", height));
							continue;
						}
					}

					src = video_start + video_size;
					continue;
				}

				if (inner_id == 0xe1) {
					uint64_t audio_size = read_size();
					src += audio_size;
					continue;
				}

				if (inner_id == 0xe2) {
					uint64_t track_operation_size = read_size();
					src += track_operation_size;
					continue;
				}

				if (inner_id == 0x6D80) {
					uint64_t content_encodings_size = read_size();
					src += content_encodings_size;
					continue;
				}

				uint64_t inner_size = read_size();
				src += inner_size;
				WARN_PRINT(vformat("Unhandled element with ID %x of size %d", inner_id, inner_size));
			}

			r_tracks.push_back(track);
			continue;
		}

		uint64_t outer_size = read_size();
		src += outer_size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", id, outer_size));
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
	while (src - cluster_start < cluster_size) {
		uint64_t id = read_id();

		if (id == EBML_CRC32_ID) {
			uint64_t crc_size = read_size();
			src += crc_size;
			continue;
		}

		if (id == 0xE7) {
			r_cluster->time = read_uint();
			continue;
		}

		if (id == 0xA3) {
			Cluster::Block block = {};

			uint64_t block_size = read_size();
			const uint8_t *block_start = src;

			uint64_t target_track = read_size();

			uint16_t timestamp = (src[0] << 8) | src[1];
			src += 2;

			if (target_track == 1) {
				r_cluster->time_to_layer.insert(timestamp, r_cluster->time_to_layer.size());
			}

			//uint8_t flags = src[0];
			src += 1;

			if (target_track == 1) {
				block.position = src - origin;
				block.size = block_size - (src - block_start);
				r_cluster->blocks.append(block);
			}

			src = block_start + block_size;
			continue;
		}

		uint64_t skipped_size = read_size();
		src += skipped_size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", id, skipped_size));
	}

	src = cluster_start + cluster_size;
	return OK;
}

Error VideoStreamPlaybackMatroska::parse_cues() {
	int64_t cues_size = read_size();
	const uint8_t *cues_start = src;

	while (src - cues_start < cues_size) {
		uint64_t id = read_id();

		if (id == EBML_CRC32_ID) {
			uint64_t crc_size = read_size();
			src += crc_size;
			continue;
		}

		if (id == 0xBB) {
			int64_t element_size = read_size();
			const uint8_t *element_start = src;

			Cluster cluster = {};
			while (src - element_start < element_size) {
				uint64_t element_id = read_id();

				if (element_id == 0xB3) {
					cluster.time = read_uint();
					continue;
				}

				if (element_id == 0xB7) {
					int64_t positions_size = read_size();
					const uint8_t *positions_start = src;

					while (src - positions_start < positions_size) {
						uint64_t positions_id = read_id();

						if (positions_id == 0xF7) {
							cluster.target_track = read_uint();
							continue;
						}

						if (positions_id == 0xF1) {
							cluster.position = read_uint();
							continue;
						}

						// TODO is relative position at all useful?
						if (positions_id == 0xF0) {
							read_uint();
							continue;
						}

						uint64_t skipped_size = read_size();
						src += skipped_size;
						ERR_PRINT(vformat("Missing id %x of size %d", element_id, skipped_size));
					}

					continue;
				}

				uint64_t skipped_size = read_size();
				src += skipped_size;
				ERR_PRINT(vformat("Missing id %x", element_id));
			}

			clusters.push_back(cluster);
			continue;
		}

		uint64_t skipped_size = read_size();
		src += skipped_size;
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

	while (src - origin < buffer.size()) {
		uint64_t id = read_id();

		if (id == EBML_HEADER_ID) {
			Error err = parse_ebml_header(&header);
			if (err != OK) {
				ERR_PRINT("error parsing ebml header");
			}
			continue;
		}

		if (id == MATROSKA_SEGMENT_ID) {
			Error err = parse_segment(&segment);
			if (err != OK) {
				ERR_PRINT("error parsing matroska segment");
			}
			break;
		}

		uint64_t size = read_size();
		src += size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", id, size));
	}
}

void VideoStreamPlaybackMatroska::play() {
	playing = true;

	if (video_stream_encoding.is_null()) {
		return;
	}

	cluster_rid = video_stream_encoding->create_video_session(width, height);

	Error err;
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ, &err);

	Cluster cluster = clusters[0];
	video_stream_encoding->begin_cluster();

	Cluster::Block block = cluster.blocks[0];
	file->seek(block.position);
	Vector<uint8_t> frame = file->get_buffer(block.size);
	video_stream_encoding->append_container_block(frame);

	cluster_rid = video_stream_encoding->end_cluster();
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
	return;
	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(cluster_rid, 0);

	Ref<Image> frame;
	frame.instantiate();
	frame->set_data(width, height, true, Image::FORMAT_RGBA8, data);

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
}

bool ResourceFormatLoaderMatroska::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "VideoStream");
}

String ResourceFormatLoaderMatroska::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "mkv") {
		return "VideoStreamMatroska";
	}
	return "";
}
