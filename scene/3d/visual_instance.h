/*************************************************************************/
/*  visual_instance.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef VISUAL_INSTANCE_H
#define VISUAL_INSTANCE_H

#include "face3.h"
#include "rid.h"
#include "scene/3d/spatial.h"
#include "scene/resources/material.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class VisualInstance : public Spatial {

	GDCLASS(VisualInstance, Spatial);
	OBJ_CATEGORY("3D Visual Nodes");

	RID instance;
	uint32_t layers;

	RID _get_visual_instance_rid() const;

protected:
	void _update_visibility();

	void _notification(int p_what);
	static void _bind_methods();

public:
	enum GetFacesFlags {
		FACES_SOLID = 1, // solid geometry
		FACES_ENCLOSING = 2,
		FACES_DYNAMIC = 4 // dynamic object geometry

	};

	RID get_instance() const;
	virtual AABB get_aabb() const = 0;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const = 0;

	virtual AABB get_transformed_aabb() const; // helper

	void set_base(const RID &p_base);

	void set_layer_mask(uint32_t p_mask);
	uint32_t get_layer_mask() const;

	VisualInstance();
	~VisualInstance();
};

class GeometryInstance : public VisualInstance {

	GDCLASS(GeometryInstance, VisualInstance);

public:
	enum Flags {
		FLAG_USE_BAKED_LIGHT = VS::INSTANCE_FLAG_USE_BAKED_LIGHT,
		FLAG_MAX = VS::INSTANCE_FLAG_MAX,
	};

	enum ShadowCastingSetting {
		SHADOW_CASTING_SETTING_OFF = VS::SHADOW_CASTING_SETTING_OFF,
		SHADOW_CASTING_SETTING_ON = VS::SHADOW_CASTING_SETTING_ON,
		SHADOW_CASTING_SETTING_DOUBLE_SIDED = VS::SHADOW_CASTING_SETTING_DOUBLE_SIDED,
		SHADOW_CASTING_SETTING_SHADOWS_ONLY = VS::SHADOW_CASTING_SETTING_SHADOWS_ONLY
	};

private:
	bool flags[FLAG_MAX];
	ShadowCastingSetting shadow_casting_setting;
	Ref<Material> material_override;
	float lod_min_distance;
	float lod_max_distance;
	float lod_min_hysteresis;
	float lod_max_hysteresis;

	float extra_cull_margin;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_flag(Flags p_flag, bool p_value);
	bool get_flag(Flags p_flag) const;

	void set_cast_shadows_setting(ShadowCastingSetting p_shadow_casting_setting);
	ShadowCastingSetting get_cast_shadows_setting() const;

	void set_lod_min_distance(float p_dist);
	float get_lod_min_distance() const;

	void set_lod_max_distance(float p_dist);
	float get_lod_max_distance() const;

	void set_lod_min_hysteresis(float p_dist);
	float get_lod_min_hysteresis() const;

	void set_lod_max_hysteresis(float p_dist);
	float get_lod_max_hysteresis() const;

	void set_material_override(const Ref<Material> &p_material);
	Ref<Material> get_material_override() const;

	void set_extra_cull_margin(float p_margin);
	float get_extra_cull_margin() const;

	GeometryInstance();
};

VARIANT_ENUM_CAST(GeometryInstance::Flags);
VARIANT_ENUM_CAST(GeometryInstance::ShadowCastingSetting);

#endif
