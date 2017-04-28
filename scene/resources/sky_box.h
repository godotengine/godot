/*************************************************************************/
/*  sky_box.h                                                            */
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
#ifndef SKYBOX_H
#define SKYBOX_H

#include "scene/resources/texture.h"

class SkyBox : public Resource {
	GDCLASS(SkyBox, Resource);

public:
	enum RadianceSize {
		RADIANCE_SIZE_256,
		RADIANCE_SIZE_512,
		RADIANCE_SIZE_1024,
		RADIANCE_SIZE_2048,
		RADIANCE_SIZE_MAX
	};

private:
	RadianceSize radiance_size;

protected:
	static void _bind_methods();
	virtual void _radiance_changed() = 0;

public:
	void set_radiance_size(RadianceSize p_size);
	RadianceSize get_radiance_size() const;
	SkyBox();
};

VARIANT_ENUM_CAST(SkyBox::RadianceSize)

class ImageSkyBox : public SkyBox {
	GDCLASS(ImageSkyBox, SkyBox);

public:
	enum ImagePath {
		IMAGE_PATH_NEGATIVE_X,
		IMAGE_PATH_POSITIVE_X,
		IMAGE_PATH_NEGATIVE_Y,
		IMAGE_PATH_POSITIVE_Y,
		IMAGE_PATH_NEGATIVE_Z,
		IMAGE_PATH_POSITIVE_Z,
		IMAGE_PATH_MAX
	};

private:
	RID cube_map;
	RID sky_box;
	bool cube_map_valid;

	String image_path[IMAGE_PATH_MAX];

protected:
	static void _bind_methods();
	virtual void _radiance_changed();

public:
	void set_image_path(ImagePath p_image, const String &p_path);
	String get_image_path(ImagePath p_image) const;

	virtual RID get_rid() const;

	ImageSkyBox();
	~ImageSkyBox();
};

VARIANT_ENUM_CAST(ImageSkyBox::ImagePath)

#endif // SKYBOX_H
