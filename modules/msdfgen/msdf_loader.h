/*************************************************************************/
/*  msdf_loader.h                                                        */
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

#ifndef MSDF_LOADER_H
#define MSDF_LOADER_H

#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "scene/resources/texture.h"

#include "msdfgen.h"

class MSDFLoader : public Object {
	GDCLASS(MSDFLoader, Object);

public:
	enum SDFType {
		SDF_TRUE,
		SDF_PSEUDO,
		SDF_MULTICHANNEL,
		SDF_COMBINED
	};

private:
	bool dirty = true;

	float px_range = 4;
	SDFType sdf_type = SDF_COMBINED;

	msdfgen::Point2 position;
	msdfgen::Contour *contour = nullptr;
	msdfgen::Shape shape;

	Ref<Image> data;

protected:
	static void _bind_methods();

public:
	Error load_svg(const String &p_path, float p_scale = 1.0);

	void set_px_range(float p_range);
	float get_px_range() const;

	void set_sdf_type(int p_type);
	int get_sdf_type() const;

	void clear_shape();
	void move_to(const Vector2 &p_to);
	void line_to(const Vector2 &p_to);
	void conic_to(const Vector2 &p_ctrl, const Vector2 &p_to);
	void cubic_to(const Vector2 &p_ctrl1, const Vector2 &p_ctrl2, const Vector2 &p_to);

	Ref<Image> get_data();
};

VARIANT_ENUM_CAST(MSDFLoader::SDFType);

#endif
