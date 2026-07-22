/**************************************************************************/
/*  image_loader.cpp                                                      */
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

#include "image_loader.h"

#include "core/object/class_db.h"

void ImageFormatLoader::_bind_methods() {
	BIND_BITFIELD_FLAG(FLAG_NONE);
	BIND_BITFIELD_FLAG(FLAG_FORCE_LINEAR);
	BIND_BITFIELD_FLAG(FLAG_CONVERT_COLORS);
}

bool ImageFormatLoader::recognize(const String &p_extension) const {
	List<String> extensions;
	get_recognized_extensions(&extensions);
	for (const String &E : extensions) {
		if (E.nocasecmp_to(p_extension) == 0) {
			return true;
		}
	}

	return false;
}

Error ImageFormatLoaderExtension::load_image(Ref<Image> p_image, Ref<FileAccess> p_fileaccess, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	Error err = ERR_UNAVAILABLE;
	GDVIRTUAL_CALL(_load_image, p_image, p_fileaccess, p_flags, p_scale, err);
	return err;
}

void ImageFormatLoaderExtension::get_recognized_extensions(List<String> *p_extension) const {
	PackedStringArray ext;
	if (GDVIRTUAL_CALL(_get_recognized_extensions, ext)) {
		for (int i = 0; i < ext.size(); i++) {
			p_extension->push_back(ext[i]);
		}
	}
}

void ImageFormatLoaderExtension::add_format_loader() {
	ImageLoader::add_image_format_loader(this);
}

void ImageFormatLoaderExtension::remove_format_loader() {
	ImageLoader::remove_image_format_loader(this);
}

void ImageFormatLoaderExtension::_bind_methods() {
	GDVIRTUAL_BIND(_get_recognized_extensions);
	GDVIRTUAL_BIND(_load_image, "image", "fileaccess", "flags", "scale");
	ClassDB::bind_method(D_METHOD("add_format_loader"), &ImageFormatLoaderExtension::add_format_loader);
	ClassDB::bind_method(D_METHOD("remove_format_loader"), &ImageFormatLoaderExtension::remove_format_loader);
}

Error ImageLoader::load_image(const String &p_file, Ref<Image> p_image, Ref<FileAccess> p_custom, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	ERR_FAIL_COND_V_MSG(p_image.is_null(), ERR_INVALID_PARAMETER, "Can't load an image: invalid Image object.");
	const String file = ResourceUID::ensure_path(p_file);

	Ref<FileAccess> f = p_custom;
	if (f.is_null()) {
		Error err;
		f = FileAccess::open(file, FileAccess::READ, &err);
		ERR_FAIL_COND_V_MSG(f.is_null(), err, vformat("Error opening file '%s'.", file));
	}

	String extension = file.get_extension();

	for (int i = 0; i < loader.size(); i++) {
		if (!loader[i]->recognize(extension)) {
			continue;
		}
		Error err = loader.write[i]->load_image(p_image, f, p_flags, p_scale);
		if (err != OK) {
			ERR_PRINT(vformat("Error loading image: '%s'.", file));
		}

		if (err != ERR_FILE_UNRECOGNIZED) {
			return err;
		}
	}

	return ERR_FILE_UNRECOGNIZED;
}

void ImageLoader::get_recognized_extensions(List<String> *p_extensions) {
	for (int i = 0; i < loader.size(); i++) {
		loader[i]->get_recognized_extensions(p_extensions);
	}
}

Ref<ImageFormatLoader> ImageLoader::recognize(const String &p_extension) {
	for (int i = 0; i < loader.size(); i++) {
		if (loader[i]->recognize(p_extension)) {
			return loader[i];
		}
	}

	return nullptr;
}

void ImageLoader::add_image_format_loader(Ref<ImageFormatLoader> p_loader) {
	loader.push_back(p_loader);
}

void ImageLoader::remove_image_format_loader(Ref<ImageFormatLoader> p_loader) {
	loader.erase(p_loader);
}

void ImageLoader::cleanup() {
	while (loader.size()) {
		remove_image_format_loader(loader[0]);
	}
}
