/*************************************************************************/
/*  resource_importer_ogg_vorbis.cpp                                     */
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

#include "resource_importer_ogg_vorbis.h"

#include "core/io/resource_saver.h"
#include "core/os/file_access.h"
#include "scene/resources/texture.h"

String ResourceImporterOGGVorbis::get_importer_name() const {
	return "ogg_vorbis";
}

String ResourceImporterOGGVorbis::get_visible_name() const {
	return "OGGVorbis";
}

void ResourceImporterOGGVorbis::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ogg");
}

String ResourceImporterOGGVorbis::get_save_extension() const {
	return "oggstr";
}

String ResourceImporterOGGVorbis::get_resource_type() const {
	return "AudioStreamOGGVorbis";
}

bool ResourceImporterOGGVorbis::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterOGGVorbis::get_preset_count() const {
	return 0;
}

String ResourceImporterOGGVorbis::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterOGGVorbis::get_import_options(List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "loop"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "loop_offset"), 0));
}

Error ResourceImporterOGGVorbis::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	bool loop = p_options["loop"];
	float loop_offset = p_options["loop_offset"];

	FileAccess *f = FileAccess::open(p_source_file, FileAccess::READ);

	ERR_FAIL_COND_V_MSG(!f, ERR_CANT_OPEN, "Cannot open file '" + p_source_file + "'.");

	uint64_t len = f->get_length();

	Vector<uint8_t> data;
	data.resize(len);
	uint8_t *w = data.ptrw();

	f->get_buffer(w, len);

	memdelete(f);

	Ref<AudioStreamOGGVorbis> ogg_stream;
	ogg_stream.instance();

	ogg_stream->set_data(data);
	ERR_FAIL_COND_V(!ogg_stream->get_data().size(), ERR_FILE_CORRUPT);
	ogg_stream->set_loop(loop);
	ogg_stream->set_loop_offset(loop_offset);

	return ResourceSaver::save(p_save_path + ".oggstr", ogg_stream);
}

ResourceImporterOGGVorbis::ResourceImporterOGGVorbis() {
}
