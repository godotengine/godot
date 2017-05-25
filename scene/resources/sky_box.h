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
#ifndef Sky_H
#define Sky_H

#include "scene/resources/texture.h"

class Sky : public Resource {
	GDCLASS(Sky, Resource);

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
	Sky();
};

VARIANT_ENUM_CAST(Sky::RadianceSize)

class PanoramaSky : public Sky {
	GDCLASS(PanoramaSky, Sky);

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
	RID sky;
	Ref<Texture> panorama;

protected:
	static void _bind_methods();
	virtual void _radiance_changed();

public:
	void set_panorama(const Ref<Texture> &p_panorama);
	Ref<Texture> get_panorama() const;

	virtual RID get_rid() const;

	PanoramaSky();
	~PanoramaSky();
};

VARIANT_ENUM_CAST(PanoramaSky::ImagePath)

#endif // Sky_H
