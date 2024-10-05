/**************************************************************************/
/*  resource_importer_ogg_vorbis.cpp                                      */
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

#include "resource_importer_ogg_vorbis.h"

#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "scene/resources/texture.h"

#ifdef TOOLS_ENABLED
#include "editor/import/audio_stream_import_settings.h"
#endif

#include <ogg/ogg.h>
#include <vorbis/codec.h>

String ResourceImporterOggVorbis::get_importer_name() const {
	return "oggvorbisstr";
}

String ResourceImporterOggVorbis::get_visible_name() const {
	return "oggvorbisstr";
}

void ResourceImporterOggVorbis::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ogg");
}

String ResourceImporterOggVorbis::get_save_extension() const {
	return "oggvorbisstr";
}

String ResourceImporterOggVorbis::get_resource_type() const {
	return "AudioStreamOggVorbis";
}

bool ResourceImporterOggVorbis::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterOggVorbis::get_preset_count() const {
	return 0;
}

String ResourceImporterOggVorbis::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterOggVorbis::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "loop"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "loop_offset"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "bpm", PROPERTY_HINT_RANGE, "0,400,0.01,or_greater"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "beat_count", PROPERTY_HINT_RANGE, "0,512,or_greater"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "bar_beats", PROPERTY_HINT_RANGE, "2,32,or_greater"), 4));
}

#ifdef TOOLS_ENABLED

bool ResourceImporterOggVorbis::has_advanced_options() const {
	return true;
}

void ResourceImporterOggVorbis::show_advanced_options(const String &p_path) {
	Ref<AudioStreamOggVorbis> ogg_stream = _load_from_file(p_path);
	if (ogg_stream.is_valid()) {
		AudioStreamImportSettingsDialog::get_singleton()->edit(p_path, "oggvorbisstr", ogg_stream);
	}
}
#endif

Error ResourceImporterOggVorbis::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	bool loop = p_options["loop"];
	double loop_offset = p_options["loop_offset"];
	double bpm = p_options["bpm"];
	int beat_count = p_options["beat_count"];
	int bar_beats = p_options["bar_beats"];

	Error err = OK;
	Ref<AudioStreamOggVorbis> ogg_vorbis_stream = load_from_file(p_source_file, err);
	if (err != OK) {
		return err;
	}

	ogg_vorbis_stream->set_loop(loop);
	ogg_vorbis_stream->set_loop_offset(loop_offset);
	ogg_vorbis_stream->set_bpm(bpm);
	ogg_vorbis_stream->set_beat_count(beat_count);
	ogg_vorbis_stream->set_bar_beats(bar_beats);

	return ResourceSaver::save(ogg_vorbis_stream, p_save_path + ".oggvorbisstr");
}

ResourceImporterOggVorbis::ResourceImporterOggVorbis() {
}

void ResourceImporterOggVorbis::_bind_methods() {
	ClassDB::bind_static_method("ResourceImporterOggVorbis", D_METHOD("load_from_buffer", "buffer"), &ResourceImporterOggVorbis::_load_from_buffer);
	ClassDB::bind_static_method("ResourceImporterOggVorbis", D_METHOD("load_from_file", "path"), &ResourceImporterOggVorbis::_load_from_file);
}

Ref<AudioStreamOggVorbis> ResourceImporterOggVorbis::load_from_buffer(const Vector<uint8_t> &p_file_data, Error &r_error) {
	Ref<AudioStreamOggVorbis> ogg_vorbis_stream;
	ogg_vorbis_stream.instantiate();

	Ref<OggPacketSequence> ogg_packet_sequence;
	ogg_packet_sequence.instantiate();

	ogg_stream_state stream_state;
	ogg_sync_state sync_state;
	ogg_page page;
	ogg_packet packet;
	bool initialized_stream = false;

	ogg_sync_init(&sync_state);
	int err;
	size_t cursor = 0;
	size_t packet_count = 0;
	bool done = false;
	while (!done) {
		err = ogg_sync_check(&sync_state);
		if (err != OK) {
			r_error = ERR_INVALID_DATA;
			ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Ogg sync error " + itos(err));
		}
		while (ogg_sync_pageout(&sync_state, &page) != 1) {
			if (cursor >= size_t(p_file_data.size())) {
				done = true;
				break;
			}
			err = ogg_sync_check(&sync_state);
			if (err != OK) {
				r_error = ERR_INVALID_DATA;
				ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Ogg sync error " + itos(err));
			}
			char *sync_buf = ogg_sync_buffer(&sync_state, OGG_SYNC_BUFFER_SIZE);
			err = ogg_sync_check(&sync_state);
			if (err != OK) {
				r_error = ERR_INVALID_DATA;
				ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Ogg sync error " + itos(err));
			}
			size_t copy_size = p_file_data.size() - cursor;
			if (copy_size > OGG_SYNC_BUFFER_SIZE) {
				copy_size = OGG_SYNC_BUFFER_SIZE;
			}
			memcpy(sync_buf, &p_file_data[cursor], copy_size);
			ogg_sync_wrote(&sync_state, copy_size);
			cursor += copy_size;
			err = ogg_sync_check(&sync_state);
			if (err != OK) {
				r_error = ERR_INVALID_DATA;
				ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Ogg sync error " + itos(err));
			}
		}
		if (done) {
			break;
		}
		err = ogg_sync_check(&sync_state);
		if (err != OK) {
			r_error = ERR_INVALID_DATA;
			ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Ogg sync error " + itos(err));
		}

		// Have a page now.
		if (!initialized_stream) {
			if (ogg_stream_init(&stream_state, ogg_page_serialno(&page))) {
				r_error = ERR_OUT_OF_MEMORY;
				ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Failed allocating memory for Ogg Vorbis stream.");
			}
			initialized_stream = true;
		}
		ogg_stream_pagein(&stream_state, &page);
		err = ogg_stream_check(&stream_state);
		if (err != OK) {
			r_error = ERR_INVALID_DATA;
			ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Ogg stream error " + itos(err));
		}
		int desync_iters = 0;

		RBMap<uint64_t, Vector<Vector<uint8_t>>> sorted_packets;
		int64_t granule_pos = 0;

		while (true) {
			err = ogg_stream_packetout(&stream_state, &packet);
			if (err == -1) {
				// According to the docs this is usually recoverable, but don't sit here spinning forever.
				desync_iters++;
				WARN_PRINT_ONCE("Desync during ogg import.");
				if (desync_iters > 100) {
					r_error = ERR_INVALID_DATA;
					ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Packet sync issue during Ogg import");
				}
				continue;
			} else if (err == 0) {
				// Not enough data to fully reconstruct a packet. Go on to the next page.
				break;
			}
			if (packet_count == 0 && vorbis_synthesis_idheader(&packet) == 0) {
				print_verbose("Found a non-vorbis-header packet in a header position");
				// Clearly this logical stream is not a vorbis stream, so destroy it and try again with the next page.
				if (initialized_stream) {
					ogg_stream_clear(&stream_state);
					initialized_stream = false;
				}
				break;
			}
			if (packet.granulepos > granule_pos) {
				granule_pos = packet.granulepos;
			}

			if (packet.bytes > 0) {
				PackedByteArray data;
				data.resize(packet.bytes);
				memcpy(data.ptrw(), packet.packet, packet.bytes);
				sorted_packets[granule_pos].push_back(data);
				packet_count++;
			}
		}
		Vector<Vector<uint8_t>> packet_data;
		for (const KeyValue<uint64_t, Vector<Vector<uint8_t>>> &pair : sorted_packets) {
			for (const Vector<uint8_t> &packets : pair.value) {
				packet_data.push_back(packets);
			}
		}
		if (initialized_stream && packet_data.size() > 0) {
			ogg_packet_sequence->push_page(ogg_page_granulepos(&page), packet_data);
		}
	}
	if (initialized_stream) {
		ogg_stream_clear(&stream_state);
	}
	ogg_sync_clear(&sync_state);

	if (ogg_packet_sequence->get_packet_granule_positions().is_empty()) {
		r_error = ERR_INVALID_DATA;
		ERR_FAIL_V_MSG(Ref<AudioStreamOggVorbis>(), "Ogg Vorbis decoding failed. Check that your data is a valid Ogg Vorbis audio stream.");
	}

	ogg_vorbis_stream->set_packet_sequence(ogg_packet_sequence);

	return ogg_vorbis_stream;
}

Ref<AudioStreamOggVorbis> ResourceImporterOggVorbis::_load_from_buffer(const Vector<uint8_t> &p_file_data) {
	Error err = OK;
	Ref<AudioStreamOggVorbis> as = load_from_buffer(p_file_data, err);
	if (err) {
		return Ref<AudioStreamOggVorbis>();
	}
	return as;
}

Ref<AudioStreamOggVorbis> ResourceImporterOggVorbis::load_from_file(const String &p_path, Error &r_error) {
	Error err = OK;

	Vector<uint8_t> file_data = FileAccess::get_file_as_bytes(p_path, &err);

	if (err != OK) {
		r_error = err;
		ERR_FAIL_V(Ref<AudioStreamOggVorbis>());
	}

	return load_from_buffer(file_data, r_error);
}

Ref<AudioStreamOggVorbis> ResourceImporterOggVorbis::_load_from_file(const String &p_path) {
	Error err = OK;
	Ref<AudioStreamOggVorbis> as = load_from_file(p_path, err);
	if (err != OK) {
		return Ref<AudioStreamOggVorbis>();
	}
	return as;
}
