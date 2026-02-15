/**************************************************************************/
/*  render_scene_data.h                                                   */
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

#include "core/object/class_db.h"
#include "core/object/gdvirtual.gen.h"
#include "core/object/object.h"
#include "core/object/script_language.h"
#include "core/variant/variant.h"

class RenderSceneData : public Object {
	GDCLASS(RenderSceneData, Object);

protected:
	static void _bind_methods();

public:
	virtual Transform3D get_cam_transform() const = 0;
	virtual Projection get_cam_projection() const = 0;

	virtual PackedFloat32Array get_transformed_projection_data() const = 0;

	virtual float get_z_far() const = 0;
	virtual float get_z_near() const = 0;
	virtual float get_aspect() const = 0;
	virtual Vector2 get_viewport_half_extents() const = 0;
	virtual Vector2 get_far_plane_half_extents() const = 0;
	virtual int get_pixels_per_meter(int p_for_pixel_width) const = 0;

	virtual uint32_t get_view_count() const = 0;
	virtual Vector3 get_view_eye_offset(uint32_t p_view) const = 0;
	virtual Projection get_view_projection(uint32_t p_view) const = 0;

	virtual RID get_uniform_buffer() const = 0;
};

class RenderSceneDataExtension : public RenderSceneData {
	GDCLASS(RenderSceneDataExtension, RenderSceneData);

protected:
	static void _bind_methods();

public:
	virtual Transform3D get_cam_transform() const override;
	virtual Projection get_cam_projection() const override;

	virtual PackedFloat32Array get_transformed_projection_data() const override;

	virtual float get_z_far() const override;
	virtual float get_z_near() const override;
	virtual float get_aspect() const override;
	virtual Vector2 get_viewport_half_extents() const override;
	virtual Vector2 get_far_plane_half_extents() const override;
	virtual int get_pixels_per_meter(int p_for_pixel_width) const override;

	virtual uint32_t get_view_count() const override;
	virtual Vector3 get_view_eye_offset(uint32_t p_view) const override;
	virtual Projection get_view_projection(uint32_t p_view) const override;

	virtual RID get_uniform_buffer() const override;

	GDVIRTUAL0RC(Transform3D, _get_cam_transform)
	GDVIRTUAL0RC(Projection, _get_cam_projection)

	GDVIRTUAL0RC(PackedFloat32Array, _get_transformed_projection_data)

	GDVIRTUAL0RC(float, _get_z_far)
	GDVIRTUAL0RC(float, _get_z_near)
	GDVIRTUAL0RC(float, _get_aspect)
	GDVIRTUAL0RC(Vector2, _get_viewport_half_extents)
	GDVIRTUAL0RC(Vector2, _get_far_plane_half_extents)
	GDVIRTUAL1RC(int, _get_pixels_per_meter, int)

	GDVIRTUAL0RC(uint32_t, _get_view_count)
	GDVIRTUAL1RC(Vector3, _get_view_eye_offset, uint32_t)
	GDVIRTUAL1RC(Projection, _get_view_projection, uint32_t)

	GDVIRTUAL0RC(RID, _get_uniform_buffer)
};
