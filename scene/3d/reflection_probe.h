/**************************************************************************/
/*  reflection_probe.h                                                    */
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

#ifndef REFLECTION_PROBE_H
#define REFLECTION_PROBE_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/sky.h"
#include "scene/resources/texture.h"
#include "servers/visual_server.h"

class ReflectionProbe : public VisualInstance {
	GDCLASS(ReflectionProbe, VisualInstance);

public:
	enum UpdateMode {
		UPDATE_ONCE,
		UPDATE_ALWAYS,
	};

private:
	RID probe;
	float intensity;
	float max_distance;
	Vector3 extents;
	Vector3 origin_offset;
	bool box_projection;
	bool enable_shadows;
	bool interior;
	Color interior_ambient;
	float interior_ambient_energy;
	float interior_ambient_probe_contribution;

	uint32_t cull_mask;
	UpdateMode update_mode;

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;

public:
	void set_intensity(float p_intensity);
	float get_intensity() const;

	void set_interior_ambient(Color p_ambient);
	Color get_interior_ambient() const;

	void set_interior_ambient_energy(float p_energy);
	float get_interior_ambient_energy() const;

	void set_interior_ambient_probe_contribution(float p_contribution);
	float get_interior_ambient_probe_contribution() const;

	void set_max_distance(float p_distance);
	float get_max_distance() const;

	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const;

	void set_origin_offset(const Vector3 &p_extents);
	Vector3 get_origin_offset() const;

	void set_as_interior(bool p_enable);
	bool is_set_as_interior() const;

	void set_enable_box_projection(bool p_enable);
	bool is_box_projection_enabled() const;

	void set_enable_shadows(bool p_enable);
	bool are_shadows_enabled() const;

	void set_cull_mask(uint32_t p_layers);
	uint32_t get_cull_mask() const;

	void set_update_mode(UpdateMode p_mode);
	UpdateMode get_update_mode() const;

	virtual AABB get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	ReflectionProbe();
	~ReflectionProbe();
};

VARIANT_ENUM_CAST(ReflectionProbe::UpdateMode);

#endif // REFLECTION_PROBE_H
