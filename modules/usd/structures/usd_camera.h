/**************************************************************************/
/*  usd_camera.h                                                          */
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

class USDCamera : public Resource {
	GDCLASS(USDCamera, Resource);

public:
	enum ProjectionType {
		PERSPECTIVE,
		ORTHOGRAPHIC,
	};

private:
	ProjectionType projection = PERSPECTIVE;
	float focal_length = 50.0;
	float horizontal_aperture = 20.955;
	float vertical_aperture = 15.2908;
	float near_clip = 0.1;
	float far_clip = 1000.0;

protected:
	static void _bind_methods();

public:
	ProjectionType get_projection() const;
	void set_projection(ProjectionType p_projection);

	float get_focal_length() const;
	void set_focal_length(float p_focal_length);

	float get_horizontal_aperture() const;
	void set_horizontal_aperture(float p_horizontal_aperture);

	float get_vertical_aperture() const;
	void set_vertical_aperture(float p_vertical_aperture);

	float get_near_clip() const;
	void set_near_clip(float p_near_clip);

	float get_far_clip() const;
	void set_far_clip(float p_far_clip);

	float get_fov() const;
};

VARIANT_ENUM_CAST(USDCamera::ProjectionType);
