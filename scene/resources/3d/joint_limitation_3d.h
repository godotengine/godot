/**************************************************************************/
/*  joint_limitation_3d.h                                                 */
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

#ifdef TOOLS_ENABLED
#include "scene/resources/surface_tool.h"
#endif // TOOLS_ENABLED

class JointLimitation3D : public Resource {
	GDCLASS(JointLimitation3D, Resource);

protected:
	// Directions are normalized vector from Vector(0, 0, 0). Space is defined by _make_space(), must return normalized vector.
	virtual Vector3 _solve(const Vector3 &p_direction) const;

public:
	// Define temporary space based on rest and forward vector.
	virtual Quaternion make_space(const Vector3 &p_local_forward_vector, const Vector3 &p_local_right_vector, const Quaternion &p_rotation_offset) const;

	Vector3 solve(const Vector3 &p_local_forward_vector, const Vector3 &p_local_right_vector, const Quaternion &p_rotation_offset, const Vector3 &p_local_current_vector) const;

#ifdef TOOLS_ENABLED
	virtual void draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const; // For drawing gizmo.
#endif // TOOLS_ENABLED
};
