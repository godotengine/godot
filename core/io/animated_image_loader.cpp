/*************************************************************************/
/*  animated_image_loader.cpp                                            */
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

#include "animated_image_loader.h"

#define HEADER_SIZE 6

AnimatedImageFormatLoader::~AnimatedImageFormatLoader() {
}

Vector<AnimatedImageFormatLoader *> ResourceFormatLoaderAnimatedImage::loaders;

void ResourceFormatLoaderAnimatedImage::add_animated_image_format_loader(AnimatedImageFormatLoader *p_loader) {

	loaders.push_back(p_loader);
}

void ResourceFormatLoaderAnimatedImage::remove_animated_image_format_loader(AnimatedImageFormatLoader *p_loader) {

	loaders.erase(p_loader);
}

RES ResourceFormatLoaderAnimatedImage::load(const String &p_path, const String &p_original_path, Error *r_error) {

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!f) {

		if (r_error)
			*r_error = err;

		return RES();
	}

	uint8_t header[HEADER_SIZE] = { 0 };
	int header_size = f->get_buffer(header, HEADER_SIZE);

	if (header_size != HEADER_SIZE) {

		if (r_error)
			*r_error = ERR_FILE_CORRUPT;

		return RES();
	}

	AnimatedImage::SourceFormat format;
	AnimatedImage::ImportType type = AnimatedImage::ANIMATED_TEXTURE;
	int tex_flags = 0;
	int max_frames = 0;

	uint8_t imported_header[HEADER_SIZE] = { 'G', 'D', 'A', 'I', 'M', 'G' };
	if (memcmp(&header[0], &imported_header[0], HEADER_SIZE) == 0) { // The file is imported.

		format = AnimatedImage::SourceFormat(f->get_8());
		type = AnimatedImage::ImportType(f->get_8());
		tex_flags = f->get_32();
		max_frames = f->get_32();
	} else {

		if (header[0] == 'G') { // The only supported format right now.

			format = AnimatedImage::GIF;
		} else {

			f->close();
			memdelete(f);

			if (r_error)
				*r_error = ERR_FILE_CORRUPT;

			return RES();
		}

		f->seek(0); // Reset the cursor to the beginning of the file.
	}

	bool loaded = false;
	Ref<AnimatedImage> animated_image = memnew(AnimatedImage);

	for (int i = 0; i < loaders.size(); i++) {

		if (loaders[i]->recognize_format(format)) {
			err = loaders[i]->load_animated_image(animated_image, f, max_frames);
			loaded = true;
		}
	}

	f->close();
	memdelete(f);

	if (err != OK || !loaded) {

		if (r_error)
			*r_error = err;

		return RES();
	}

	RES result;

	switch (type) {
		case AnimatedImage::ANIMATED_TEXTURE: {

			result = animated_image->to_animated_texture(tex_flags, max_frames);
		} break;
		case AnimatedImage::SPRITE_FRAMES: {

			result = animated_image->to_sprite_frames(tex_flags, max_frames);
		} break;
		default: {

			if (r_error)
				*r_error = ERR_FILE_CORRUPT;

			return RES();
		}
	}

	if (r_error)
		*r_error = OK;

	return result;
}

void ResourceFormatLoaderAnimatedImage::get_recognized_extensions(List<String> *p_extensions) const {

	for (int i = 0; i < loaders.size(); i++)
		loaders[i]->get_recognized_extensions(p_extensions);

	p_extensions->push_back("aimg");
}

String ResourceFormatLoaderAnimatedImage::get_resource_type(const String &p_path) const {

	String ext = p_path.get_extension();

	List<String> extensions;
	get_recognized_extensions(&extensions);
	for (List<String>::Element *E = extensions.front(); E; E = E->next()) {

		if (E->get().nocasecmp_to(ext) == 0)
			return "AnimatedImage";
	}

	return "";
}

bool ResourceFormatLoaderAnimatedImage::handles_type(const String &p_type) const {

	return (p_type == "AnimatedTexture" || p_type == "SpriteFrames");
}
