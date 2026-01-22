/**************************************************************************/
/*  gltf_physics_material.h                                               */
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

#include "scene/resources/physics_material.h"

// GLTFPhysicsMaterial is an intermediary between Godot's PhysicsMaterial
// and the OMI_physics_body extension's physicsMaterial property.
// https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0/OMI_physics_body

class GLTFPhysicsMaterial : public Resource {
	GDCLASS(GLTFPhysicsMaterial, Resource)

public:
	enum CombineMode {
		COMBINE_AVERAGE,
		COMBINE_MINIMUM,
		COMBINE_MAXIMUM,
		COMBINE_MULTIPLY,
	};

protected:
	static void _bind_methods();

private:
	real_t static_friction = 0.6;
	real_t dynamic_friction = 0.6;
	real_t restitution = 0.0;
	CombineMode friction_combine = COMBINE_AVERAGE;
	CombineMode restitution_combine = COMBINE_AVERAGE;

public:
	real_t get_static_friction() const;
	void set_static_friction(real_t p_static_friction);

	real_t get_dynamic_friction() const;
	void set_dynamic_friction(real_t p_dynamic_friction);

	real_t get_restitution() const;
	void set_restitution(real_t p_restitution);

	CombineMode get_friction_combine() const;
	void set_friction_combine(CombineMode p_friction_combine);

	CombineMode get_restitution_combine() const;
	void set_restitution_combine(CombineMode p_restitution_combine);

	static Ref<GLTFPhysicsMaterial> from_resource(const Ref<PhysicsMaterial> &p_material);
	Ref<PhysicsMaterial> to_resource() const;

	static Ref<GLTFPhysicsMaterial> from_dictionary(const Dictionary &p_dictionary);
	Dictionary to_dictionary() const;

	bool operator==(const GLTFPhysicsMaterial &p_other) const;
	bool operator!=(const GLTFPhysicsMaterial &p_other) const;
};

VARIANT_ENUM_CAST(GLTFPhysicsMaterial::CombineMode);
