/*************************************************************************/
/*  resource_importer_ogg_opus.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "audio_stream_ogg_opus.h"

#include "resource_importer_ogg_opus.h"

#include "core/io/resource_saver.h"
#include "core/os/file_access.h"
#include "scene/resources/texture.h"

String ResourceImporterOGGOpus::get_importer_name() const {
	return "oggopus";
}

String ResourceImporterOGGOpus::get_visible_name() const {
	return "OGGOpus";
}

void ResourceImporterOGGOpus::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("opus");
}

String ResourceImporterOGGOpus::get_save_extension() const {
	return "opusstr";
}

String ResourceImporterOGGOpus::get_resource_type() const {
	return "AudioStreamOGGOpus";
}

bool ResourceImporterOGGOpus::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterOGGOpus::get_preset_count() const {
	return 0;
}

String ResourceImporterOGGOpus::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterOGGOpus::get_import_options(List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "loop"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "loop_offset"), 0));
}

Error ResourceImporterOGGOpus::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	bool loop = p_options["loop"];
	float loop_offset = p_options["loop_offset"];

	FileAccess *f = FileAccess::open(p_source_file, FileAccess::READ);

	ERR_FAIL_COND_V_MSG(!f, ERR_CANT_OPEN, "Cannot open file '" + p_source_file + "'.");

	size_t len = f->get_len();

	Vector<uint8_t> data;
	data.resize(len);
	uint8_t *w = data.ptrw();

	f->get_buffer(w, len);

	memdelete(f);

	Ref<AudioStreamOGGOpus> ogg_stream;
	ogg_stream.instance();

	ogg_stream->set_data(data);
	ERR_FAIL_COND_V(!ogg_stream->get_data().size(), ERR_FILE_CORRUPT);
	ogg_stream->set_loop(loop);
	ogg_stream->set_loop_offset(loop_offset);

	return ResourceSaver::save(p_save_path + ".opusstr", ogg_stream);
}

ResourceImporterOGGOpus::ResourceImporterOGGOpus() {
}
