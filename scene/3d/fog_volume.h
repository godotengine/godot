/*************************************************************************/
/*  fog_volume.h                                                         */
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

#ifndef FOG_VOLUME_H
#define FOG_VOLUME_H

#include "core/templates/rid.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/main/node.h"
#include "scene/main/viewport.h"
#include "scene/resources/material.h"

class FogVolume : public VisualInstance3D {
	GDCLASS(FogVolume, VisualInstance3D);

	Vector3 extents = Vector3(1, 1, 1);
	Ref<Material> material;
	RS::FogVolumeShape shape = RS::FOG_VOLUME_SHAPE_BOX;

	RID volume;

protected:
	_FORCE_INLINE_ RID _get_volume() { return volume; }
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;

public:
	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const;

	void set_shape(RS::FogVolumeShape p_type);
	RS::FogVolumeShape get_shape() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	virtual AABB get_aabb() const override;
	virtual Vector<Face3> get_faces(uint32_t p_usage_flags) const override { return Vector<Face3>(); }
	TypedArray<String> get_configuration_warnings() const override;

	FogVolume();
	~FogVolume();
};

#endif // FOG_VOLUME_H
