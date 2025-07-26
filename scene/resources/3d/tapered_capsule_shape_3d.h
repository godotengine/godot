/**************************************************************************/
/*  tapered_capsule_shape_3d.h                                            */
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

#include "scene/resources/3d/shape_3d.h"
#include "servers/physics_3d/physics_server_3d.h"

class ArrayMesh;

class TaperedCapsuleShape3D : public Shape3D {
	GDCLASS(TaperedCapsuleShape3D, Shape3D);
	real_t radius_top = 0.5;
	real_t radius_bottom = 0.5;
	real_t mid_height = 1.0; // Height of the cylindrical part

protected:
	static void _bind_methods();

	virtual void _update_shape() override;

public:
	void set_radius_top(real_t p_radius_top);
	real_t get_radius_top() const;

	void set_radius_bottom(real_t p_radius_bottom);
	real_t get_radius_bottom() const;

	void set_mid_height(real_t p_mid_height);
	real_t get_mid_height() const;

	void set_height(real_t p_height);
	real_t get_height() const;

	virtual Vector<Vector3> get_debug_mesh_lines() const override;
	virtual Ref<ArrayMesh> get_debug_arraymesh_faces(const Color &p_modulate) const override;
	virtual real_t get_enclosing_radius() const override;

	virtual Variant get_data() const;
	virtual void set_data(const Variant &p_data);

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_CAPSULE; } // Use capsule type for physics server compatibility

	TaperedCapsuleShape3D();
};
