/**************************************************************************/
/*  resource_importer_svg.cpp                                             */
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

#include "resource_importer_svg.h"

#include "core/io/file_access.h"
#include "scene/resources/dpi_texture.h"

String ResourceImporterSVG::get_importer_name() const {
	return "svg";
}

String ResourceImporterSVG::get_visible_name() const {
	return "DPITexture";
}

void ResourceImporterSVG::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("svg");
}

String ResourceImporterSVG::get_save_extension() const {
	return "dpitex";
}

String ResourceImporterSVG::get_resource_type() const {
	return "DPITexture";
}

bool ResourceImporterSVG::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterSVG::get_preset_count() const {
	return 0;
}

String ResourceImporterSVG::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterSVG::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "base_scale", PROPERTY_HINT_RANGE, "0.001,100,0.001"), 1.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "saturation", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), 1.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::DICTIONARY, "color_map"), Dictionary()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compress"), true));
}

Error ResourceImporterSVG::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	String source = FileAccess::get_file_as_string(p_source_file);
	ERR_FAIL_COND_V_MSG(source.is_empty(), ERR_CANT_OPEN, vformat("Cannot open file from path \"%s\".", p_source_file));

	double base_scale = p_options["base_scale"];
	double saturation = p_options["saturation"];
	Dictionary color_map = p_options["color_map"];

	Ref<DPITexture> dpi_tex = DPITexture::create_from_string(source, base_scale, saturation, color_map);
	ERR_FAIL_COND_V_MSG(dpi_tex->get_rid().is_null(), ERR_CANT_OPEN, vformat("Failed loading SVG, unsupported or invalid SVG data in \"%s\".", p_source_file));

	int flg = 0;
	if ((bool)p_options["compress"]) {
		flg |= ResourceSaver::SaverFlags::FLAG_COMPRESS;
	}

	print_verbose("Saving to: " + p_save_path + ".dpitex");
	Error err = ResourceSaver::save(dpi_tex, p_save_path + ".dpitex", flg);
	ERR_FAIL_COND_V_MSG(err != OK, err, vformat("Cannot save DPI texture to file \"%s.dpitex\".", p_save_path));
	print_verbose("Done saving to: " + p_save_path + ".dpitex");

	return OK;
}
