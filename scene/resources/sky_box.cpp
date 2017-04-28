/*************************************************************************/
/*  sky_box.cpp                                                          */
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
#include "sky_box.h"
#include "io/image_loader.h"

void SkyBox::set_radiance_size(RadianceSize p_size) {
	ERR_FAIL_INDEX(p_size, RADIANCE_SIZE_MAX);

	radiance_size = p_size;
	_radiance_changed();
}

SkyBox::RadianceSize SkyBox::get_radiance_size() const {

	return radiance_size;
}

void SkyBox::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_radiance_size", "size"), &SkyBox::set_radiance_size);
	ClassDB::bind_method(D_METHOD("get_radiance_size"), &SkyBox::get_radiance_size);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "radiance_size", PROPERTY_HINT_ENUM, "256,512,1024,2048"), "set_radiance_size", "get_radiance_size");

	BIND_CONSTANT(RADIANCE_SIZE_256);
	BIND_CONSTANT(RADIANCE_SIZE_512);
	BIND_CONSTANT(RADIANCE_SIZE_1024);
	BIND_CONSTANT(RADIANCE_SIZE_2048);
	BIND_CONSTANT(RADIANCE_SIZE_MAX);
}

SkyBox::SkyBox() {
	radiance_size = RADIANCE_SIZE_512;
}

/////////////////////////////////////////

void ImageSkyBox::_radiance_changed() {

	if (cube_map_valid) {
		static const int size[RADIANCE_SIZE_MAX] = {
			256, 512, 1024, 2048
		};
		VS::get_singleton()->skybox_set_texture(sky_box, cube_map, size[get_radiance_size()]);
	}
}

void ImageSkyBox::set_image_path(ImagePath p_image, const String &p_path) {

	ERR_FAIL_INDEX(p_image, IMAGE_PATH_MAX);
	image_path[p_image] = p_path;

	bool all_ok = true;
	for (int i = 0; i < IMAGE_PATH_MAX; i++) {
		if (image_path[i] == String()) {
			all_ok = false;
		}
	}

	cube_map_valid = false;

	if (all_ok) {

		Image images[IMAGE_PATH_MAX];
		int w = 0, h = 0;
		Image::Format format;

		for (int i = 0; i < IMAGE_PATH_MAX; i++) {
			Error err = ImageLoader::load_image(image_path[i], &images[i]);
			if (err) {
				ERR_PRINTS("Error loading image for skybox: " + image_path[i]);
				return;
			}

			if (i == 0) {
				w = images[0].get_width();
				h = images[0].get_height();
				format = images[0].get_format();
			} else {
				if (images[i].get_width() != w || images[i].get_height() != h || images[i].get_format() != format) {
					ERR_PRINTS("Image size mismatch (" + itos(images[i].get_width()) + "," + itos(images[i].get_height()) + ":" + Image::get_format_name(images[i].get_format()) + " when it should be " + itos(w) + "," + itos(h) + ":" + Image::get_format_name(format) + "): " + image_path[i]);
					return;
				}
			}
		}

		VS::get_singleton()->texture_allocate(cube_map, w, h, format, VS::TEXTURE_FLAG_FILTER | VS::TEXTURE_FLAG_CUBEMAP | VS::TEXTURE_FLAG_MIPMAPS);
		for (int i = 0; i < IMAGE_PATH_MAX; i++) {
			VS::get_singleton()->texture_set_data(cube_map, images[i], VS::CubeMapSide(i));
		}

		cube_map_valid = true;
		_radiance_changed();
	}
}

String ImageSkyBox::get_image_path(ImagePath p_image) const {

	ERR_FAIL_INDEX_V(p_image, IMAGE_PATH_MAX, String());
	return image_path[p_image];
}

RID ImageSkyBox::get_rid() const {

	return sky_box;
}

void ImageSkyBox::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_image_path", "image", "path"), &ImageSkyBox::set_image_path);
	ClassDB::bind_method(D_METHOD("get_image_path", "image"), &ImageSkyBox::get_image_path);

	List<String> extensions;
	ImageLoader::get_recognized_extensions(&extensions);
	String hints;
	for (List<String>::Element *E = extensions.front(); E; E = E->next()) {
		if (hints != String()) {
			hints += ",";
		}
		hints += "*." + E->get();
	}

	ADD_GROUP("Image Path", "image_path_");
	ADD_PROPERTYI(PropertyInfo(Variant::STRING, "image_path_negative_x", PROPERTY_HINT_FILE, hints), "set_image_path", "get_image_path", IMAGE_PATH_NEGATIVE_X);
	ADD_PROPERTYI(PropertyInfo(Variant::STRING, "image_path_positive_x", PROPERTY_HINT_FILE, hints), "set_image_path", "get_image_path", IMAGE_PATH_POSITIVE_X);
	ADD_PROPERTYI(PropertyInfo(Variant::STRING, "image_path_negative_y", PROPERTY_HINT_FILE, hints), "set_image_path", "get_image_path", IMAGE_PATH_NEGATIVE_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::STRING, "image_path_positive_y", PROPERTY_HINT_FILE, hints), "set_image_path", "get_image_path", IMAGE_PATH_POSITIVE_Y);
	ADD_PROPERTYI(PropertyInfo(Variant::STRING, "image_path_negative_z", PROPERTY_HINT_FILE, hints), "set_image_path", "get_image_path", IMAGE_PATH_NEGATIVE_Z);
	ADD_PROPERTYI(PropertyInfo(Variant::STRING, "image_path_positive_z", PROPERTY_HINT_FILE, hints), "set_image_path", "get_image_path", IMAGE_PATH_POSITIVE_Z);

	BIND_CONSTANT(IMAGE_PATH_NEGATIVE_X);
	BIND_CONSTANT(IMAGE_PATH_POSITIVE_X);
	BIND_CONSTANT(IMAGE_PATH_NEGATIVE_Y);
	BIND_CONSTANT(IMAGE_PATH_POSITIVE_Y);
	BIND_CONSTANT(IMAGE_PATH_NEGATIVE_Z);
	BIND_CONSTANT(IMAGE_PATH_POSITIVE_Z);
	BIND_CONSTANT(IMAGE_PATH_MAX);
}

ImageSkyBox::ImageSkyBox() {

	cube_map = VS::get_singleton()->texture_create();
	sky_box = VS::get_singleton()->skybox_create();
	cube_map_valid = false;
}

ImageSkyBox::~ImageSkyBox() {

	VS::get_singleton()->free(cube_map);
	VS::get_singleton()->free(sky_box);
}
