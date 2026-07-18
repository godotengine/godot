/**************************************************************************/
/*  primitive_geometry_3d.h                                               */
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

#include "core/math/aabb.h"
#include "core/math/math_defs.h"
#include "core/math/triangle_mesh.h"
#include "core/math/vector3.h"
#include "core/object/object.h"
#include "core/templates/rid.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/3d/shape_3d.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"

// Core primitive geometry class, handles collision and materials
class PrimitiveGeometry3D : public GeometryInstance3D {
	GDCLASS(PrimitiveGeometry3D, GeometryInstance3D);

protected:
	Ref<Material> material;
	Ref<PrimitiveMesh> mesh;

#ifndef PHYSICS_3D_DISABLED
	Ref<Shape3D> shape;

private:
	bool use_collision = true;
	uint32_t collision_layer = 1;
	uint32_t collision_mask = 1;
	real_t collision_priority = 1.0;
	RID collision_body;
#endif // PHYSICS_3D_DISABLED

protected:
	void _notification(int p_what);
	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

public:
	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

#ifndef PHYSICS_3D_DISABLED
	void initiate_collision();
	void destroy_collision();

	void set_use_collision(bool p_enable);
	bool is_using_collision() const;

	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_layer_value(int p_layer_number, bool p_value);
	bool get_collision_layer_value(int p_layer_number) const;

	void set_collision_mask_value(int p_layer_number, bool p_value);
	bool get_collision_mask_value(int p_layer_number) const;

	void set_collision_priority(real_t p_priority);
	real_t get_collision_priority() const;
#endif

	AABB get_aabb() const override;
	Ref<TriangleMesh> generate_triangle_mesh() const override;

private:
	virtual void instantiate_mesh() = 0;
#ifndef PHYSICS_3D_DISABLED
	virtual void instantiate_collision_shape() = 0;
#endif

};

// All the primitive geometries

class Box3D : public PrimitiveGeometry3D {
	GDCLASS(Box3D, PrimitiveGeometry3D);

	Vector3 size = Vector3(1.f, 1.f, 1.f);

protected:
	static void _bind_methods();

private:
	void instantiate_mesh() override;
#ifndef PHYSICS_3D_DISABLED
	void instantiate_collision_shape() override;
#endif

public:
	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;

	Box3D() {};
};

class Sphere3D : public PrimitiveGeometry3D {
	GDCLASS(Sphere3D, PrimitiveGeometry3D);

	float radius = 1.f;

protected:
	static void _bind_methods();

private:
	void instantiate_mesh() override;
#ifndef PHYSICS_3D_DISABLED
	void instantiate_collision_shape() override;
#endif

public:
	void set_radius(const float p_radius);
	float get_radius() const;

	Sphere3D() {};
};

class Cylinder3D : public PrimitiveGeometry3D {
	GDCLASS(Cylinder3D, PrimitiveGeometry3D);

	float radius = 0.5f;
	float height = 2.0f;

protected:
	static void _bind_methods();

private:
	void instantiate_mesh() override;
#ifndef PHYSICS_3D_DISABLED
	void instantiate_collision_shape() override;
#endif

public:
	void set_radius(const float p_radius);
	float get_radius() const;
	void set_height(const float p_height);
	float get_height() const;

	Cylinder3D() {};
};

class Capsule3D : public PrimitiveGeometry3D {
	GDCLASS(Capsule3D, PrimitiveGeometry3D);

	float radius = 0.5f;
	float height = 2.0f;

protected:
	static void _bind_methods();

private:
	void instantiate_mesh() override;
#ifndef PHYSICS_3D_DISABLED
	void instantiate_collision_shape() override;
#endif

public:
	void set_radius(const float p_radius);
	float get_radius() const;
	void set_height(const float p_height);
	float get_height() const;

	Capsule3D() {};
};
