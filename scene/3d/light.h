/*************************************************************************/
/*  light.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef LIGHT_H
#define LIGHT_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/texture.h"
#include "servers/visual_server.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Light : public VisualInstance {

	OBJ_TYPE(Light, VisualInstance);
	OBJ_CATEGORY("3D Light Nodes");

public:
	enum Parameter {
		PARAM_RADIUS = VisualServer::LIGHT_PARAM_RADIUS,
		PARAM_ENERGY = VisualServer::LIGHT_PARAM_ENERGY,
		PARAM_ATTENUATION = VisualServer::LIGHT_PARAM_ATTENUATION,
		PARAM_SPOT_ANGLE = VisualServer::LIGHT_PARAM_SPOT_ANGLE,
		PARAM_SPOT_ATTENUATION = VisualServer::LIGHT_PARAM_SPOT_ATTENUATION,
		PARAM_SHADOW_DARKENING = VisualServer::LIGHT_PARAM_SHADOW_DARKENING,
		PARAM_SHADOW_Z_OFFSET = VisualServer::LIGHT_PARAM_SHADOW_Z_OFFSET,
		PARAM_SHADOW_Z_SLOPE_SCALE = VisualServer::LIGHT_PARAM_SHADOW_Z_SLOPE_SCALE,
		PARAM_SHADOW_ESM_MULTIPLIER = VisualServer::LIGHT_PARAM_SHADOW_ESM_MULTIPLIER,
		PARAM_SHADOW_BLUR_PASSES = VisualServer::LIGHT_PARAM_SHADOW_BLUR_PASSES,
		PARAM_MAX = VisualServer::LIGHT_PARAM_MAX
	};

	enum LightColor {

		COLOR_DIFFUSE = VisualServer::LIGHT_COLOR_DIFFUSE,
		COLOR_SPECULAR = VisualServer::LIGHT_COLOR_SPECULAR
	};

	enum BakeMode {

		BAKE_MODE_DISABLED,
		BAKE_MODE_INDIRECT,
		BAKE_MODE_INDIRECT_AND_SHADOWS,
		BAKE_MODE_FULL

	};

	enum Operator {

		OPERATOR_ADD,
		OPERATOR_SUB
	};

private:
	Ref<Texture> projector;
	float vars[PARAM_MAX];
	Color colors[3];

	BakeMode bake_mode;
	VisualServer::LightType type;
	bool shadows;
	bool enabled;
	bool editor_only;
	Operator op;

	void _update_visibility();
	// bind helpers

protected:
	RID light;

	virtual bool _can_gizmo_scale() const;
	virtual RES _get_gizmo_geometry() const;

	static void _bind_methods();
	void _notification(int p_what);

	Light(VisualServer::LightType p_type);

public:
	VS::LightType get_light_type() const { return type; }

	void set_parameter(Parameter p_var, float p_value);
	float get_parameter(Parameter p_var) const;

	void set_color(LightColor p_color, const Color &p_value);
	Color get_color(LightColor p_color) const;

	void set_project_shadows(bool p_enabled);
	bool has_project_shadows() const;

	void set_projector(const Ref<Texture> &p_projector);
	Ref<Texture> get_projector() const;

	void set_operator(Operator p_op);
	Operator get_operator() const;

	void set_bake_mode(BakeMode p_bake_mode);
	BakeMode get_bake_mode() const;

	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_editor_only(bool p_editor_only);
	bool is_editor_only() const;

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void approximate_opengl_attenuation(float p_constant, float p_linear, float p_quadratic, float p_radius_treshold = 0.5);

	Light();
	~Light();
};

VARIANT_ENUM_CAST(Light::Parameter);
VARIANT_ENUM_CAST(Light::LightColor);
VARIANT_ENUM_CAST(Light::Operator);
VARIANT_ENUM_CAST(Light::BakeMode);

class DirectionalLight : public Light {

	OBJ_TYPE(DirectionalLight, Light);

public:
	enum ShadowMode {
		SHADOW_ORTHOGONAL,
		SHADOW_PERSPECTIVE,
		SHADOW_PARALLEL_2_SPLITS,
		SHADOW_PARALLEL_4_SPLITS
	};
	enum ShadowParam {
		SHADOW_PARAM_MAX_DISTANCE,
		SHADOW_PARAM_PSSM_SPLIT_WEIGHT,
		SHADOW_PARAM_PSSM_ZOFFSET_SCALE
	};

private:
	ShadowMode shadow_mode;
	float shadow_param[3];

protected:
	static void _bind_methods();

public:
	void set_shadow_mode(ShadowMode p_mode);
	ShadowMode get_shadow_mode() const;

	void set_shadow_max_distance(float p_distance);
	float get_shadow_max_distance() const;
	void set_shadow_param(ShadowParam p_param, float p_value);
	float get_shadow_param(ShadowParam p_param) const;

	DirectionalLight();
};

VARIANT_ENUM_CAST(DirectionalLight::ShadowMode);
VARIANT_ENUM_CAST(DirectionalLight::ShadowParam);

class OmniLight : public Light {

	OBJ_TYPE(OmniLight, Light);

protected:
	static void _bind_methods();

public:
	OmniLight() :
			Light(VisualServer::LIGHT_OMNI) { set_parameter(PARAM_SHADOW_Z_OFFSET, 0.001); }
};

class SpotLight : public Light {

	OBJ_TYPE(SpotLight, Light);

protected:
	static void _bind_methods();

public:
	SpotLight() :
			Light(VisualServer::LIGHT_SPOT) {}
};

#endif
