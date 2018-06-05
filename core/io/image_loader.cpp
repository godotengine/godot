/*************************************************************************/
/*  image_loader.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "image_loader.h"

#include "print_string.h"
bool ImageFormatLoader::recognize(const String &p_extension) const {

	List<String> extensions;
	get_recognized_extensions(&extensions);
	for (List<String>::Element *E = extensions.front(); E; E = E->next()) {

		if (E->get().nocasecmp_to(p_extension) == 0)
			return true;
	}

	return false;
}

Error ImageLoader::load_image(String p_file, Ref<Image> p_image, FileAccess *p_custom, bool p_force_linear, float p_scale) {
	ERR_FAIL_COND_V(p_image.is_null(), ERR_INVALID_PARAMETER);

	FileAccess *f = p_custom;
	if (!f) {
		Error err;
		f = FileAccess::open(p_file, FileAccess::READ, &err);
		if (!f) {
			ERR_PRINTS("Error opening file: " + p_file);
			return err;
		}
	}

	String extension = p_file.get_extension();

	for (int i = 0; i < loader.size(); i++) {

		if (!loader[i]->recognize(extension))
			continue;
		Error err = loader[i]->load_image(p_image, f, p_force_linear, p_scale);

		if (err != ERR_FILE_UNRECOGNIZED) {

			if (!p_custom)
				memdelete(f);

			return err;
		}
	}

	if (!p_custom)
		memdelete(f);

	return ERR_FILE_UNRECOGNIZED;
}

void ImageLoader::get_recognized_extensions(List<String> *p_extensions) {

	for (int i = 0; i < loader.size(); i++) {

		loader[i]->get_recognized_extensions(p_extensions);
	}
}

ImageFormatLoader *ImageLoader::recognize(const String &p_extension) {

	for (int i = 0; i < loader.size(); i++) {

		if (loader[i]->recognize(p_extension))
			return loader[i];
	}

	return NULL;
}

Vector<ImageFormatLoader *> ImageLoader::loader;

void ImageLoader::add_image_format_loader(ImageFormatLoader *p_loader) {

	loader.push_back(p_loader);
}

void ImageLoader::remove_image_format_loader(ImageFormatLoader *p_loader) {

	loader.erase(p_loader);
}

const Vector<ImageFormatLoader *> &ImageLoader::get_image_format_loaders() {

	return loader;
}

void ImageLoader::cleanup() {

    while (loader.size()) {
        remove_image_format_loader(loader[0]);
    }
}
