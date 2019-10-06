/*************************************************************************/
/*  resource_importer_animated_image.cpp                                 */
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

#include "resource_importer_animated_image.h"

#include "core/os/file_access.h"
#include "scene/resources/texture.h"

void ResourceImporterAnimatedImage::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("gif");
}

String ResourceImporterAnimatedImage::get_save_extension() const {

	return "aimg";
}

bool ResourceImporterAnimatedImage::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

	return true;
}

int ResourceImporterAnimatedImage::get_preset_count() const {

	return 0;
}

String ResourceImporterAnimatedImage::get_preset_name(int p_idx) const {

	return "";
}

void ResourceImporterAnimatedImage::get_import_options(List<ImportOption> *r_options, int p_preset) const {

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "flags/repeat", PROPERTY_HINT_ENUM, "Disabled,Enabled,Mirrored"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "flags/filter"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "flags/mipmaps"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "flags/anisotropic"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "flags/srgb", PROPERTY_HINT_ENUM, "Disable,Enable,Detect"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "max_frames", PROPERTY_HINT_RANGE, "0, 99999, 1"), 0));
}

Error ResourceImporterAnimatedImage::import_animated_image(AnimatedImage::ImportType type, const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options) {

	int repeat = p_options["flags/repeat"];
	bool filter = p_options["flags/filter"];
	bool mipmaps = p_options["flags/mipmaps"];
	bool anisotropic = p_options["flags/anisotropic"];
	int srgb = p_options["flags/srgb"];
	int max_frames = p_options["max_frames"];

	int tex_flags = 0;
	if (repeat > 0)
		tex_flags |= Texture::FLAG_REPEAT;
	if (repeat == 2)
		tex_flags |= Texture::FLAG_MIRRORED_REPEAT;
	if (filter)
		tex_flags |= Texture::FLAG_FILTER;
	if (mipmaps)
		tex_flags |= Texture::FLAG_MIPMAPS;
	if (anisotropic)
		tex_flags |= Texture::FLAG_ANISOTROPIC_FILTER;
	if (srgb == 1)
		tex_flags |= Texture::FLAG_CONVERT_TO_LINEAR;

	FileAccess *f = FileAccess::open(p_source_file, FileAccess::READ);
	ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);
	size_t len = f->get_len();

	Vector<uint8_t> data;
	data.resize(len);
	f->get_buffer(data.ptrw(), len);
	f->close();
	memdelete(f);

	f = FileAccess::open(p_save_path + ".aimg", FileAccess::WRITE);
	ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);

	const uint8_t header[6] = { 'G', 'D', 'A', 'I', 'M', 'G' };
	f->store_buffer(header, 6);
	f->store_8(AnimatedImage::GIF); // The only supported format right now.
	f->store_8(type);
	f->store_32(tex_flags);
	f->store_32(max_frames);
	f->store_buffer(data.ptr(), len);
	f->close();
	memdelete(f);

	return OK;
}
