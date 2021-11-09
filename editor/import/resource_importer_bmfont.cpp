/*************************************************************************/
/*  resource_importer_bmfont.cpp                                         */
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

#include "resource_importer_bmfont.h"

#include "core/io/resource_saver.h"

String ResourceImporterBMFont::get_importer_name() const {
	return "font_data_bmfont";
}

String ResourceImporterBMFont::get_visible_name() const {
	return "Font Data (AngelCode BMFont)";
}

void ResourceImporterBMFont::get_recognized_extensions(List<String> *p_extensions) const {
	if (p_extensions) {
		p_extensions->push_back("font");
		p_extensions->push_back("fnt");
	}
}

String ResourceImporterBMFont::get_save_extension() const {
	return "fontdata";
}

String ResourceImporterBMFont::get_resource_type() const {
	return "FontData";
}

bool ResourceImporterBMFont::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {
	return true;
}

void ResourceImporterBMFont::get_import_options(List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compress"), true));
}

Error ResourceImporterBMFont::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	print_verbose("Importing BMFont font from: " + p_source_file);

	Ref<FontData> font;
	font.instantiate();

	Error err = font->load_bitmap_font(p_source_file);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot load font to file \"" + p_source_file + "\".");

	int flg = ResourceSaver::SaverFlags::FLAG_BUNDLE_RESOURCES | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;
	if ((bool)p_options["compress"]) {
		flg |= ResourceSaver::SaverFlags::FLAG_COMPRESS;
	}

	print_verbose("Saving to: " + p_save_path + ".fontdata");
	err = ResourceSaver::save(p_save_path + ".fontdata", font, flg);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save font to file \"" + p_save_path + ".res\".");
	print_verbose("Done saving to: " + p_save_path + ".fontdata");
	return OK;
}

ResourceImporterBMFont::ResourceImporterBMFont() {
}
