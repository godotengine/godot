/**************************************************************************/
/*  light.h                                                               */
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

#ifndef LIGHT_H
#define LIGHT_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/texture.h"
#include "servers/visual_server.h"

class Light : public VisualInstance {
	GDCLASS(Light, VisualInstance);
	OBJ_CATEGORY("3D Light Nodes");

public:
	enum Param {
		PARAM_ENERGY = VS::LIGHT_PARAM_ENERGY,
		PARAM_INDIRECT_ENERGY = VS::LIGHT_PARAM_INDIRECT_ENERGY,
		PARAM_SIZE = VS::LIGHT_PARAM_SIZE,
		PARAM_SPECULAR = VS::LIGHT_PARAM_SPECULAR,
		PARAM_RANGE = VS::LIGHT_PARAM_RANGE,
		PARAM_ATTENUATION = VS::LIGHT_PARAM_ATTENUATION,
		PARAM_SPOT_ANGLE = VS::LIGHT_PARAM_SPOT_ANGLE,
		PARAM_SPOT_ATTENUATION = VS::LIGHT_PARAM_SPOT_ATTENUATION,
		PARAM_CONTACT_SHADOW_SIZE = VS::LIGHT_PARAM_CONTACT_SHADOW_SIZE,
		PARAM_SHADOW_MAX_DISTANCE = VS::LIGHT_PARAM_SHADOW_MAX_DISTANCE,
		PARAM_SHADOW_SPLIT_1_OFFSET = VS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET,
		PARAM_SHADOW_SPLIT_2_OFFSET = VS::LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET,
		PARAM_SHADOW_SPLIT_3_OFFSET = VS::LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET,
		PARAM_SHADOW_NORMAL_BIAS = VS::LIGHT_PARAM_SHADOW_NORMAL_BIAS,
		PARAM_SHADOW_BIAS = VS::LIGHT_PARAM_SHADOW_BIAS,
		PARAM_SHADOW_BIAS_SPLIT_SCALE = VS::LIGHT_PARAM_SHADOW_BIAS_SPLIT_SCALE,
		PARAM_SHADOW_FADE_START = VS::LIGHT_PARAM_SHADOW_FADE_START,
		PARAM_MAX = VS::LIGHT_PARAM_MAX
	};

	enum BakeMode {
		BAKE_DISABLED,
		BAKE_INDIRECT,
		BAKE_ALL
	};

private:
	Color color;
	float param[PARAM_MAX];
	Color shadow_color;
	bool shadow;
	bool negative;
	bool reverse_cull;
	uint32_t cull_mask;
	VS::LightType type;
	bool editor_only;
	void _update_visibility();
	BakeMode bake_mode;

	// bind helpers

	virtual void owner_changed_notify();

protected:
	RID light;

	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const;

	Light(VisualServer::LightType p_type);

public:
	VS::LightType get_light_type() const { return type; }

	void set_editor_only(bool p_editor_only);
	bool is_editor_only() const;

	void set_param(Param p_param, float p_value);
	float get_param(Param p_param) const;

	void set_shadow(bool p_enable);
	bool has_shadow() const;

	void set_negative(bool p_enable);
	bool is_negative() const;

	void set_cull_mask(uint32_t p_cull_mask);
	uint32_t get_cull_mask() const;

	void set_color(const Color &p_color);
	Color get_color() const;

	void set_shadow_color(const Color &p_shadow_color);
	Color get_shadow_color() const;

	void set_shadow_reverse_cull_face(bool p_enable);
	bool get_shadow_reverse_cull_face() const;

	void set_bake_mode(BakeMode p_mode);
	BakeMode get_bake_mode() const;

	virtual AABB get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	Light();
	~Light();
};

VARIANT_ENUM_CAST(Light::Param);
VARIANT_ENUM_CAST(Light::BakeMode);

class DirectionalLight : public Light {
	GDCLASS(DirectionalLight, Light);

public:
	enum ShadowMode {
		SHADOW_ORTHOGONAL,
		SHADOW_PARALLEL_2_SPLITS,
		SHADOW_PARALLEL_3_SPLITS,
		SHADOW_PARALLEL_4_SPLITS
	};

	enum ShadowDepthRange {
		SHADOW_DEPTH_RANGE_STABLE = VS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_STABLE,
		SHADOW_DEPTH_RANGE_OPTIMIZED = VS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_OPTIMIZED,
	};

private:
	bool blend_splits;
	ShadowMode shadow_mode;
	ShadowDepthRange shadow_depth_range;

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const;

public:
	void set_shadow_mode(ShadowMode p_mode);
	ShadowMode get_shadow_mode() const;

	void set_shadow_depth_range(ShadowDepthRange p_range);
	ShadowDepthRange get_shadow_depth_range() const;

	void set_blend_splits(bool p_enable);
	bool is_blend_splits_enabled() const;

	DirectionalLight();
};

VARIANT_ENUM_CAST(DirectionalLight::ShadowMode)
VARIANT_ENUM_CAST(DirectionalLight::ShadowDepthRange)

class OmniLight : public Light {
	GDCLASS(OmniLight, Light);

public:
	// omni light
	enum ShadowMode {
		SHADOW_DUAL_PARABOLOID,
		SHADOW_CUBE,
	};

	// omni light
	enum ShadowDetail {
		SHADOW_DETAIL_VERTICAL,
		SHADOW_DETAIL_HORIZONTAL
	};

private:
	ShadowMode shadow_mode;
	ShadowDetail shadow_detail;

protected:
	static void _bind_methods();

public:
	void set_shadow_mode(ShadowMode p_mode);
	ShadowMode get_shadow_mode() const;

	void set_shadow_detail(ShadowDetail p_detail);
	ShadowDetail get_shadow_detail() const;

	OmniLight();
};

VARIANT_ENUM_CAST(OmniLight::ShadowMode)
VARIANT_ENUM_CAST(OmniLight::ShadowDetail)

class SpotLight : public Light {
	GDCLASS(SpotLight, Light);

protected:
	static void _bind_methods();

public:
	virtual String get_configuration_warning() const;

	SpotLight() :
			Light(VisualServer::LIGHT_SPOT) {}
};

#endif // LIGHT_H
