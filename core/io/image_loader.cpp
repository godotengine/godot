/*************************************************************************/
/*  image_loader.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

		if (E->get().nocasecmp_to(p_extension.get_extension()) == 0)
			return true;
	}

	return false;
}

Error ImageLoader::load_image(String p_file, Image *p_image, FileAccess *p_custom) {

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

	for (int i = 0; i < loader_count; i++) {

		if (!loader[i]->recognize(extension))
			continue;
		Error err = loader[i]->load_image(p_image, f);

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

	for (int i = 0; i < loader_count; i++) {

		loader[i]->get_recognized_extensions(p_extensions);
	}
}

bool ImageLoader::recognize(const String &p_extension) {

	for (int i = 0; i < loader_count; i++) {

		if (loader[i]->recognize(p_extension))
			return true;
	}

	return false;
}

ImageFormatLoader *ImageLoader::loader[MAX_LOADERS];
int ImageLoader::loader_count = 0;

void ImageLoader::add_image_format_loader(ImageFormatLoader *p_loader) {

	ERR_FAIL_COND(loader_count >= MAX_LOADERS);
	loader[loader_count++] = p_loader;
}
