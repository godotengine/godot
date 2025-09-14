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

uint64_t VideoStreamPlaybackMatroska::read_id(uint8_t *p_stream, uint32_t *r_read) {
	uint8_t octet_length = 0;
	uint8_t byte = p_stream[0];

	bool found_marker = false;
	for (uint8_t i = 0; i < 8; i++) {
		octet_length += 1;
		if ((byte & (128 >> i)) > 0) {
			found_marker = true;
			break;
		}
	}

	if (!found_marker) {
		*r_read = 0;
		return 0;
	}

	*r_read = octet_length;

	uint64_t id = byte;
	for (uint32_t i = 1; i < octet_length; i++) {
		id = (id << 8) | p_stream[i];
	}

	return id;
}

uint64_t VideoStreamPlaybackMatroska::read_size(uint8_t *p_stream, uint32_t *r_read) {
	uint8_t octet_length = 0;
	uint8_t byte = p_stream[0];

	bool found_marker = false;
	for (uint8_t i = 0; i < 8; i++) {
		octet_length += 1;
		if ((byte & (128 >> i)) > 0) {
			found_marker = true;
			break;
		}
	}

	if (!found_marker) {
		WARN_PRINT("failed parse");
		*r_read = 1;
		return 1;
	}

	*r_read = octet_length;

	uint64_t size = byte ^ (1 << (8 - octet_length));
	for (uint32_t i = 1; i < octet_length; i++) {
		size = (size << 8) | p_stream[i];
	}

	return size;
}

int64_t VideoStreamPlaybackMatroska::read_int(uint8_t *p_stream, uint32_t *r_read) {
	uint64_t size = read_size(p_stream, r_read);
	p_stream += *r_read;
	*r_read += size;

	int64_t value = 0;
	for (uint32_t i = 0; i < size; i++) {
		value = (value << 8) | p_stream[i];
	}

	return value;
}

uint64_t VideoStreamPlaybackMatroska::read_uint(uint8_t *p_stream, uint32_t *r_read) {
	uint64_t size = read_size(p_stream, r_read);
	p_stream += *r_read;
	*r_read += size;

	uint64_t value = 0;
	for (uint32_t i = 0; i < size; i++) {
		value = (value << 8) | p_stream[i];
	}

	return value;
}

double VideoStreamPlaybackMatroska::read_float(uint8_t *p_stream, uint32_t *r_read) {
	uint64_t size = read_size(p_stream, r_read);
	p_stream += *r_read;
	*r_read += size;

	char *src_big = (char *)p_stream;
	if (size == 4) {
		char src_little[4];

		src_little[0] = src_big[3];
		src_little[1] = src_big[2];
		src_little[2] = src_big[1];
		src_little[3] = src_big[0];

		double value = 0;
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

		double value = 0;
		memcpy(&value, src_little, 8);
		return value;
	}
}

String VideoStreamPlaybackMatroska::read_string(uint8_t *p_stream, uint32_t *r_read) {
	uint64_t size = read_size(p_stream, r_read);
	char *src = (char *)(p_stream + *r_read);
	*r_read += size;

	TightLocalVector<char> buffer;
	buffer.resize(size);
	strncpy(buffer.ptr(), src, size);

	String ret;
	ret.append_ascii(buffer);
	return ret;
}

Error VideoStreamPlaybackMatroska::parse_ebml_header(uint8_t *p_stream, uint32_t *r_read, EbmlHeader *r_header) {
	uint32_t read = 0;

	uint64_t header_size = read_size(p_stream, &read);
	p_stream += read;
	*r_read = read;

	uint32_t total_read = 0;
	while (total_read < header_size) {
		uint64_t potential_id = read_id(p_stream, &read);
		p_stream += read;
		total_read += read;

		if (potential_id == EBML_VERSION_ID) {
			r_header->version = read_uint(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == EBML_READ_VERSION_ID) {
			r_header->read_version = read_uint(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == EBML_MAX_ID_LENGTH_ID) {
			r_header->max_id_length = read_uint(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == EBML_MAX_SIZE_LENGTH_ID) {
			r_header->max_size_length = read_uint(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_ID) {
			r_header->doc_type = read_string(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_VERSION_ID) {
			r_header->doc_type_version = read_uint(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_READ_VERSION_ID) {
			r_header->doc_type_read_version = read_uint(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_EXTENSION_ID) {
			uint64_t size = read_size(p_stream, &read);
			p_stream += read + size;
			total_read += read + size;
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_EXTENSION_NAME_ID) {
			uint64_t size = read_size(p_stream, &read);
			p_stream += read + size;
			total_read += read + size;
			continue;
		}

		if (potential_id == EBML_DOC_TYPE_EXTENSION_VERSION_ID) {
			uint64_t size = read_size(p_stream, &read);
			p_stream += read + size;
			total_read += read + size;
			continue;
		}

		uint64_t size = read_size(p_stream, &read);
		p_stream += read + size;
		total_read += read + size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", potential_id, size));
	}

	*r_read += total_read;

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_segment(uint8_t *p_stream, uint32_t *r_read, Segment *r_segment) {
	uint32_t read = 0;

	uint64_t segment_size = read_size(p_stream, &read);
	p_stream += read;
	*r_read += read;

	r_segment->src = p_stream;

	bool use_seek_head = false;
	uint32_t total_read = 0;
	while (total_read < segment_size) {
		uint64_t potential_id = read_id(p_stream, &read);
		p_stream += read;
		total_read += read;

		if (potential_id == MATROSKA_SEEK_HEAD_ID) {
			parse_seek_head(p_stream, &read, &r_segment->seek_head);
			p_stream += read;
			total_read += read;
			use_seek_head = true;
			break;
		}

		if (potential_id == MATROSKA_SEGMENT_INFO_ID) {
			parse_segment_info(p_stream, &read, &r_segment->info);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == MATROSKA_TRACKS_ID) {
			parse_tracks(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == MATROSKA_CHAPTERS_ID) {
			parse_chapters(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == MATROSKA_CLUSTER_ID) {
			parse_cluster(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == MATROSKA_CUES_ID) {
			parse_cues(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == MATROSKA_ATTACHEMENTS_ID) {
			parse_attachments(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (potential_id == MATROSKA_TAGS_ID) {
			parse_tags(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		uint64_t size = read_size(p_stream, &read);
		p_stream += read + size;
		total_read += read + size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", potential_id, size));
	}

	if (use_seek_head) {
		if (r_segment->seek_head.segment_info_position != 0) {
			uint8_t *segment_info_start = r_segment->src + r_segment->seek_head.segment_info_position;
			read_id(segment_info_start, &read);
			parse_segment_info(segment_info_start + read, &read, &r_segment->info);
		}

		if (r_segment->seek_head.tracks_position != 0) {
			uint8_t *tracks_start = r_segment->src + r_segment->seek_head.tracks_position;
			read_id(tracks_start, &read);
			parse_tracks(tracks_start + read, &read);
		}

		if (r_segment->seek_head.cues_position != 0) {
			uint8_t *cues_start = r_segment->src + r_segment->seek_head.cues_position;
			read_id(cues_start, &read);
			parse_cues(cues_start + read, &read);

			uint8_t *cluster_start = r_segment->src + clusters[0];
			read_id(cluster_start, &read);
			parse_cluster(cluster_start + read, &read);
		}

		if (r_segment->seek_head.tags_position != 0) {
			uint8_t *tags_start = r_segment->src + r_segment->seek_head.tags_position;
			read_id(tags_start, &read);
			parse_tags(tags_start + read, &read);
		}
	}

	*r_read += total_read;

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_seek_head(uint8_t *p_stream, uint32_t *r_read, SeekHead *r_seak_head) {
	uint32_t read = 0;

	uint64_t size = read_size(p_stream, &read);
	p_stream += read;
	*r_read += read + size;

	uint32_t total_read = 0;
	while (total_read < size) {
		uint64_t id = read_id(p_stream, &read);
		p_stream += read;
		total_read += read;

		if (id == MATROSKA_SEEK_ID) {
			uint64_t inner_size = read_size(p_stream, &read);
			p_stream += read;
			total_read += read + inner_size;

			struct {
				uint64_t id = 0;
				uint64_t position = 0;
			} seek;

			uint32_t inner_total_read = 0;
			while (inner_total_read < inner_size) {
				uint64_t inner_id = read_id(p_stream, &read);
				p_stream += read;
				inner_total_read += read;

				if (inner_id == MATROSKA_SEEK_TARGET_ID) {
					seek.id = read_uint(p_stream, &read);
					p_stream += read;
					inner_total_read += read;
				}

				if (inner_id == MATROSKA_SEEK_POSITION_ID) {
					seek.position = read_uint(p_stream, &read);
					p_stream += read;
					inner_total_read += read;
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

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_segment_info(uint8_t *p_stream, uint32_t *r_read, SegmentInfo *r_segment_info) {
	uint32_t read = 0;

	uint64_t size = read_size(p_stream, &read);
	p_stream += read;
	*r_read += read + size;

	uint64_t total_read = 0;
	while (total_read < size) {
		uint64_t id = read_id(p_stream, &read);
		p_stream += read;
		total_read += read;

		if (id == EBML_CRC32_ID) {
			// CRC sizes are little-endian (unlike every other EBML int)
			p_stream += 5;
			total_read += 5;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_UUID_ID) {
			uint64_t uuid_size = read_size(p_stream, &read);
			p_stream += read + uuid_size;
			total_read += read + uuid_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_FILENAME_ID) {
			r_segment_info->filename = read_string(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_PREV_UUID_ID) {
			uint64_t uuid_size = read_size(p_stream, &read);
			p_stream += read + uuid_size;
			total_read += read + uuid_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_PREV_FILENAME_ID) {
			String filename = read_string(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_NEXT_UUID_ID) {
			uint64_t uuid_size = read_size(p_stream, &read);
			p_stream += read + uuid_size;
			total_read += read + uuid_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_NEXT_FILENAME_ID) {
			String filename = read_string(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_FAMILY_ID) {
			uint64_t family_size = read_size(p_stream, &read);
			p_stream += read + family_size;
			total_read += read + family_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_CHAPTER_TRANSLATE_ID) {
			uint64_t chapter_size = read_size(p_stream, &read);
			p_stream += read + chapter_size;
			total_read += read + chapter_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_TIME_SCALE_ID) {
			r_segment_info->time_scale = read_uint(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_DURATION_ID) {
			r_segment_info->duration = read_float(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_DATE_UTC_ID) {
			uint64_t date_size = read_size(p_stream, &read);
			p_stream += read + date_size;
			total_read += read + date_size;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_TITLE_ID) {
			String title = read_string(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_MUXING_APP_ID) {
			r_segment_info->muxing_app = read_string(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		if (id == MATROSKA_SEGMENT_INFO_WRITING_APP_ID) {
			r_segment_info->writing_app = read_string(p_stream, &read);
			p_stream += read;
			total_read += read;
			continue;
		}

		uint64_t inner_size = read_size(p_stream, &read);
		p_stream += read + inner_size;
		total_read += read + inner_size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", id, size));
	}

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_tracks(uint8_t *p_stream, uint32_t *r_read) {
	uint32_t read = 0;

	uint64_t tracks_size = read_size(p_stream, &read);
	p_stream += read;
	*r_read += read + tracks_size;

	uint64_t total_read = 0;
	while (total_read < tracks_size) {
		uint64_t id = read_id(p_stream, &read);
		p_stream += read;
		total_read += read;

		if (id == EBML_CRC32_ID) {
			p_stream += 5;
			total_read += 5;
			continue;
		}

		uint64_t track_number;

		if (id == MATROSKA_TRACK_ENTRY_ID) {
			uint64_t entry_size = read_size(p_stream, &read);
			p_stream += read;
			total_read += read + entry_size;

			uint64_t total_entry_read = 0;
			while (total_entry_read < entry_size) {
				uint64_t inner_id = read_id(p_stream, &read);
				p_stream += read;
				total_entry_read += read;

				if (inner_id == MATROSKA_TRACK_NUMBER_ID) {
					track_number = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Track number %d", track_number));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_UID_ID) {
					uint64_t track_uuid = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Track uuid %d", track_uuid));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_TYPE) {
					uint64_t track_type = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					if (track_type == 1) {
						print_line("Video Track");
					} else if (track_type == 2) {
						print_line("Audio Track");
					} else {
						print_line("Some Other Track", track_type);
					}
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_ENABLED) {
					uint64_t enabled = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Enabled %d", enabled));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_DEFAULT) {
					uint64_t flag_default = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Default %d", flag_default));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_FORCED) {
					uint64_t forced = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Forced %d", forced));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_HEARING_IMPAIRED) {
					uint64_t hearing_impaired = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Hearing Impaired %d", hearing_impaired));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_VISUAL_IMPAIRED) {
					uint64_t visual_impaired = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Visually Impaired %d", visual_impaired));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_TEXT_DESCRIPTIONS) {
					uint64_t text_descriptions = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Text descriptions %d", text_descriptions));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_ORIGINAL) {
					uint64_t original = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Original %d", original));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_COMMENTARY) {
					uint64_t commentary = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Commentary %d", commentary));
					continue;
				}

				if (inner_id == MATROSKA_TRACK_FLAG_LACING) {
					uint64_t lacing = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Lacing %d", lacing));
					continue;
				}

				if (inner_id == MATRSOKA_TRACK_DEFAULT_DURATION) {
					uint64_t duration = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Default duration %d", duration));
					continue;
				}

				if (inner_id == MATRSOKA_TRACK_DEFAULT_DECODED_FIELD_DURATION) {
					uint64_t duration = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Default field duration %d", duration));
					continue;
				}

				if (inner_id == MATRSOKA_TRACK_DEFAULT_DECODED_FIELD_DURATION) {
					uint64_t duration = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Default field duration %d", duration));
					continue;
				}

				if (inner_id == 0x22b59c) {
					String language = read_string(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line("language", language);
					continue;
				}

				if (inner_id == 0x22B59D) {
					String language = read_string(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line("language (bcp47)", language);
					continue;
				}

				if (inner_id == 0xe0) {
					uint64_t video_size = read_size(p_stream, &read);
					p_stream += read + video_size;
					total_entry_read += read + video_size;
					width = 1920;
					height = 1080;
					print_line("video object");
					continue;
				}

				if (inner_id == 0x63A2) {
					uint64_t codec_size = read_size(p_stream, &read);
					p_stream += read;
					total_entry_read += read + codec_size;
					print_line("codec metadata", codec_size);
					if (track_number == 1 && video_stream_encoding.is_valid()) {
						video_stream_encoding->parse_container_metadata(p_stream, codec_size);
					}

					p_stream += codec_size;
					continue;
				}

				if (inner_id == 0x56aa) {
					uint64_t codec_delay = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Codec delay %d", codec_delay));
					continue;
				}

				if (inner_id == 0x56BB) {
					uint64_t seek_pre_roll = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line(vformat("Seek pre-roll %d", seek_pre_roll));
					continue;
				}

				if (inner_id == 0xe1) {
					uint64_t audio_size = read_size(p_stream, &read);
					p_stream += read + audio_size;
					total_entry_read += read + audio_size;
					print_line("audio object");
					continue;
				}

				if (inner_id == 0x86) {
					String codec = read_string(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					if (codec == "V_MPEG4/ISO/AVC") {
						video_stream_encoding = memnew(VideoStreamH264);
					}
					print_line(codec);
					continue;
				}

				if (inner_id == 0x55ee) {
					uint64_t max_blocks = read_uint(p_stream, &read);
					p_stream += read;
					total_entry_read += read;
					print_line("max blocks", max_blocks);
					continue;
				}

				// TODO fill in elements
				// no clue what this element is
				if (inner_id == 0xec) {
					uint64_t ec_size = read_size(p_stream, &read);
					p_stream += read + ec_size;
					total_entry_read += read + ec_size;
					continue;
				}

				uint64_t inner_size = read_size(p_stream, &read);
				p_stream += read + inner_size;
				total_entry_read += read + inner_size;
				WARN_PRINT(vformat("Unhandled element with ID %x of size %d", inner_id, inner_size));
			}

			continue;
		}

		uint64_t outer_size = read_size(p_stream, &read);
		p_stream += read + outer_size;
		total_read += read + outer_size;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", id, outer_size));
	}

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_chapters(uint8_t *p_stream, uint32_t *r_read) {
	uint32_t read = 0;

	uint64_t size = read_size(p_stream, &read);
	*r_read += read;
	*r_read += size;

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_cluster(uint8_t *p_stream, uint32_t *r_read) {
	uint32_t read = 0;

	uint64_t cluster_size = read_size(p_stream, &read);
	p_stream += read;
	*r_read += read + cluster_size;

	uint64_t total_read = 0;
	while (total_read < cluster_size) {
		uint64_t id = read_id(p_stream, &read);
		p_stream += read;
		total_read += read;

		if (id == 0xE7) {
			uint64_t time = read_uint(p_stream, &read);
			p_stream += read;
			total_read += read;
			print_line(vformat("cluster time %d", time));
			continue;
		}

		if (id == 0xA3) {
			uint64_t block_size = read_size(p_stream, &read);
			uint64_t frame_size = block_size;
			p_stream += read;
			total_read += read + block_size;

			uint64_t target_track = read_size(p_stream, &read);
			p_stream += read;
			frame_size -= read;
			if (target_track == 1) {
				print_line("---------------------");
			}

			uint16_t timestamp = (p_stream[0] << 8) | p_stream[1];
			p_stream += 2;
			frame_size -= 2;
			if (target_track == 1) {
				print_line(vformat("timestamp %d", timestamp));
			}

			uint8_t flags = p_stream[0];
			p_stream += 1;
			frame_size -= 1;

			if (target_track == 1 && video_stream_encoding.is_valid()) {
				print_line(vformat("key: %s", (flags & (1 << 7)) > 0));
				video_stream_encoding->parse_container_block(p_stream, frame_size);
			}

			p_stream += frame_size;
			continue;
		}

		ERR_PRINT(vformat("Unknown id %x", id));
		uint64_t skipped_size = read_size(p_stream, &read);
		p_stream += read + skipped_size;
		total_read += read + skipped_size;
	}

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_cues(uint8_t *p_stream, uint32_t *r_read) {
	uint32_t read = 0;

	uint64_t cues_size = read_size(p_stream, &read);
	p_stream += read;
	*r_read += read + cues_size;

	uint64_t total_read = 0;
	while (total_read < cues_size) {
		uint64_t id = read_id(p_stream, &read);
		p_stream += read;
		total_read += read;

		if (id == EBML_CRC32_ID) {
			uint64_t crc_size = read_size(p_stream, &read);
			p_stream += read + crc_size;
			total_read += read + crc_size;
			continue;
		}

		if (id == 0xBB) {
			uint64_t element_size = read_size(p_stream, &read);
			p_stream += read;
			total_read += read + element_size;

			uint64_t element_read = 0;
			while (element_read < element_size) {
				uint64_t element_id = read_id(p_stream, &read);
				p_stream += read;
				element_read += read;

				if (element_id == 0xB3) {
					uint64_t time = read_uint(p_stream, &read);
					p_stream += read;
					element_read += read;
					print_line(vformat("Cue Time %d", time));
					continue;
				}

				if (element_id == 0xB7) {
					uint64_t positions_size = read_size(p_stream, &read);
					p_stream += read;
					element_read += read + positions_size;

					uint64_t positions_read = 0;
					while (positions_read < positions_size) {
						uint64_t positions_id = read_id(p_stream, &read);
						p_stream += read;
						positions_read += read;

						if (positions_id == 0xF7) {
							uint64_t track = read_uint(p_stream, &read);
							p_stream += read;
							positions_read += read;
							print_line(vformat("target track %d", track));
							continue;
						}

						if (positions_id == 0xF1) {
							uint64_t position = read_uint(p_stream, &read);
							p_stream += read;
							positions_read += read;
							print_line(vformat("target position %d", position));
							clusters.push_back(position);
							continue;
						}

						if (positions_id == 0xF0) {
							uint64_t relative_position = read_uint(p_stream, &read);
							p_stream += read;
							positions_read += read;
							print_line(vformat("relative position %d", relative_position));
							continue;
						}

						ERR_PRINT(vformat("Missing id %x", element_id));
					}

					continue;
				}

				ERR_PRINT(vformat("Missing id %x", element_id));
			}

			continue;
		}

		uint64_t skipped_size = read_size(p_stream, &read);
		p_stream += read + skipped_size;
		total_read += read + skipped_size;
	}

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_attachments(uint8_t *p_stream, uint32_t *r_read) {
	uint32_t read = 0;

	uint64_t size = read_size(p_stream, &read);
	*r_read += read;
	*r_read += size;

	return OK;
}

Error VideoStreamPlaybackMatroska::parse_tags(uint8_t *p_stream, uint32_t *r_read) {
	uint32_t read = 0;

	uint64_t size = read_size(p_stream, &read);
	*r_read += read;
	*r_read += size;

	return OK;
}

void VideoStreamPlaybackMatroska::set_file(const String &p_file) {
	Vector<uint8_t> buffer = FileAccess::get_file_as_bytes(p_file);

	uint8_t *ptr = (uint8_t *)buffer.ptr();
	uint32_t read = 0;
	uint32_t total_read = 0;

	while (total_read < buffer.size()) {
		uint64_t id = read_id(ptr, &read);
		ptr += read;
		total_read += read;

		if (id == EBML_HEADER_ID) {
			Error err = parse_ebml_header(ptr, &read, &header);
			ptr += read;
			total_read += read;
			continue;
		}

		if (id == MATROSKA_SEGMENT_ID) {
			Error err = parse_segment(ptr, &read, &segment);
			ptr += read;
			total_read += read;
			break;
		}

		uint64_t size = read_size(ptr, &read);
		ptr += read;
		total_read += read;
		WARN_PRINT(vformat("Unhandled element with ID %x of size %d", id, size));
	}
}

void VideoStreamPlaybackMatroska::play() {
	playing = true;

	if (video_stream_encoding.is_null()) {
		return;
	}

	video_stream_encoding->create_video_profile();
	cluster = video_stream_encoding->decode_cluster();
	sleep(1);

	rd_cluster->set_texture_rd_rid(cluster);
	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(cluster, 0);

	Ref<Image> frame;
	frame.instantiate();
	frame->set_data(1980, 1080, false, Image::FORMAT_RGBAF, data);

	image_texture->set_image(frame);
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
	rd_cluster.instantiate();
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
