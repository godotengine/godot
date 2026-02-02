/**************************************************************************/
/*  usd_light.h                                                           */
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

#pragma once

#include "core/io/resource.h"

class USDLight : public Resource {
	GDCLASS(USDLight, Resource);

public:
	enum LightType {
		DISTANT,
		SPHERE,
		DISK,
		RECT,
		CYLINDER,
		DOME,
	};

private:
	LightType type = SPHERE;
	Color color = Color(1, 1, 1);
	float intensity = 1.0;
	float exposure = 0.0;
	float radius = 0.5;
	float width = 1.0;
	float height = 1.0;
	float cone_angle = 90.0;
	float cone_softness = 0.0;
	bool cast_shadows = true;
	String dome_texture;

protected:
	static void _bind_methods();

public:
	LightType get_type() const;
	void set_type(LightType p_type);

	Color get_color() const;
	void set_color(const Color &p_color);

	float get_intensity() const;
	void set_intensity(float p_intensity);

	float get_exposure() const;
	void set_exposure(float p_exposure);

	float get_radius() const;
	void set_radius(float p_radius);

	float get_width() const;
	void set_width(float p_width);

	float get_height() const;
	void set_height(float p_height);

	float get_cone_angle() const;
	void set_cone_angle(float p_cone_angle);

	float get_cone_softness() const;
	void set_cone_softness(float p_cone_softness);

	bool get_cast_shadows() const;
	void set_cast_shadows(bool p_cast_shadows);

	String get_dome_texture() const;
	void set_dome_texture(const String &p_path);
};

VARIANT_ENUM_CAST(USDLight::LightType);
