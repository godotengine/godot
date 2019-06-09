/*************************************************************************/
/*  resource_importer_xm.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "resource_importer_music.h"

#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "core/os/file_access.h"
#include "audio_stream_music.h"

const float TRIM_DB_LIMIT = -50;
const int TRIM_FADE_OUT_FRAMES = 500;

String ResourceImporterXM::get_importer_name() const {

	return "xm";
}

String ResourceImporterXM::get_visible_name() const {

	return "Fasttracker II module";
}
void ResourceImporterXM::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("xm");
}
String ResourceImporterXM::get_save_extension() const {
	return "xm";
}

String ResourceImporterXM::get_resource_type() const {

	return "AudioStreamMusic";
}

bool ResourceImporterXM::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

	return true;
}

int ResourceImporterXM::get_preset_count() const {
	return 0;
}
String ResourceImporterXM::get_preset_name(int p_idx) const {

	return String();
}

void ResourceImporterXM::get_import_options(List<ImportOption> *r_options, int p_preset) const {

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/8_bit"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/mono"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/max_rate"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "force/max_rate_hz", PROPERTY_HINT_EXP_RANGE, "11025,192000,1"), 44100));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/trim"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/normalize"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/loop"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/mode", PROPERTY_HINT_ENUM, "Disabled,RAM (Ima-ADPCM)"), 0));
}

Error ResourceImporterXM::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {

	FileAccess *f = FileAccess::open(p_source_file, FileAccess::READ);
	if (!f) {
		ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);
	}

	size_t len = f->get_len();

	Vector<uint8_t> data;
	data.resize(len);

	f->get_buffer(data.ptrw(), len);

	memdelete(f);

	f = FileAccess::open(p_save_path + ".xm", FileAccess::WRITE);

	//save the header GDIM
	//const uint8_t header[4] = { 'G', 'D', 'X', 'M' };
//	f->store_buffer(header, 4);
	//SAVE the extension (so it can be recognized by the loader later
	//f->store_pascal_string(p_source_file.get_extension().to_lower());
	//SAVE the actual image
	f->store_buffer(data.ptr(), len);

	memdelete(f);

	return OK;
}

ResourceImporterXM::ResourceImporterXM() {
}
