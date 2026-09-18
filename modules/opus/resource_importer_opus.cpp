/**************************************************************************/
/*  resource_importer_opus.cpp                                            */
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

#include "resource_importer_opus.h"

#include "audio_stream_opus.h"

#include "core/io/resource_saver.h"

#ifdef TOOLS_ENABLED
#include "editor/import/audio_stream_import_settings.h"
#endif

#include <ogg/ogg.h>
#include <opus/opus.h>

String ResourceImporterOpus::get_importer_name() const {
	return "oggopusstr";
}

String ResourceImporterOpus::get_visible_name() const {
	return "Ogg Opus";
}

void ResourceImporterOpus::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("opus");
}

String ResourceImporterOpus::get_save_extension() const {
	return "oggopusstr";
}

String ResourceImporterOpus::get_resource_type() const {
	return "AudioStreamOpus";
}

bool ResourceImporterOpus::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterOpus::get_preset_count() const {
	return 0;
}

String ResourceImporterOpus::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterOpus::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "loop"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "loop_offset"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "bpm", PROPERTY_HINT_RANGE, "0,400,0.01,or_greater"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "beat_count", PROPERTY_HINT_RANGE, "0,512,or_greater"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "bar_beats", PROPERTY_HINT_RANGE, "2,32,or_greater"), 4));
}

#ifdef TOOLS_ENABLED
bool ResourceImporterOpus::has_advanced_options() const {
	return true;
}

void ResourceImporterOpus::show_advanced_options(const String &p_path) {
	Ref<AudioStreamOpus> opus_stream = AudioStreamOpus::load_from_file(p_path);
	if (opus_stream.is_valid()) {
		AudioStreamImportSettingsDialog::get_singleton()->edit(p_path, "oggopusstr", opus_stream);
	}
}
#endif

Error ResourceImporterOpus::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	bool loop = p_options["loop"];
	double loop_offset = p_options["loop_offset"];
	double bpm = p_options["bpm"];
	int beat_count = p_options["beat_count"];
	int bar_beats = p_options["bar_beats"];

	Ref<AudioStreamOpus> opus_stream = AudioStreamOpus::load_from_file(p_source_file);
	if (opus_stream.is_null()) {
		return ERR_CANT_OPEN;
	}

	opus_stream->set_loop(loop);
	opus_stream->set_loop_offset(loop_offset);
	opus_stream->set_bpm(bpm);
	opus_stream->set_beat_count(beat_count);
	opus_stream->set_bar_beats(bar_beats);

	return ResourceSaver::save(opus_stream, p_save_path + ".oggopusstr");
}

void ResourceImporterOpus::_bind_methods() {
}

ResourceImporterOpus::ResourceImporterOpus() {
}
