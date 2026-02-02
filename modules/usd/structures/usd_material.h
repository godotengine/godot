/**************************************************************************/
/*  usd_material.h                                                        */
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
#include "scene/resources/material.h"

class USDMaterial : public Resource {
	GDCLASS(USDMaterial, Resource);

private:
	String name;

	// MaterialX XML source (non-empty when this material uses MaterialX).
	String materialx_xml;

	// UsdPreviewSurface parameters.
	Color diffuse_color = Color(0.18, 0.18, 0.18);
	float metallic = 0.0;
	float roughness = 0.5;
	Color emissive_color = Color(0, 0, 0);
	float opacity = 1.0;
	float ior = 1.5;
	float clearcoat = 0.0;
	float clearcoat_roughness = 0.01;

	// Texture asset paths (resolved).
	String diffuse_texture;
	String metallic_texture;
	String roughness_texture;
	String normal_texture;
	String emissive_texture;
	String occlusion_texture;
	String opacity_texture;

protected:
	static void _bind_methods();

public:
	String get_name() const;
	void set_name(const String &p_name);

	String get_materialx_xml() const;
	void set_materialx_xml(const String &p_xml);
	bool has_materialx() const;

	Color get_diffuse_color() const;
	void set_diffuse_color(const Color &p_color);

	float get_metallic() const;
	void set_metallic(float p_metallic);

	float get_roughness() const;
	void set_roughness(float p_roughness);

	Color get_emissive_color() const;
	void set_emissive_color(const Color &p_color);

	float get_opacity() const;
	void set_opacity(float p_opacity);

	float get_ior() const;
	void set_ior(float p_ior);

	float get_clearcoat() const;
	void set_clearcoat(float p_clearcoat);

	float get_clearcoat_roughness() const;
	void set_clearcoat_roughness(float p_clearcoat_roughness);

	String get_diffuse_texture() const;
	void set_diffuse_texture(const String &p_path);

	String get_metallic_texture() const;
	void set_metallic_texture(const String &p_path);

	String get_roughness_texture() const;
	void set_roughness_texture(const String &p_path);

	String get_normal_texture() const;
	void set_normal_texture(const String &p_path);

	String get_emissive_texture() const;
	void set_emissive_texture(const String &p_path);

	String get_occlusion_texture() const;
	void set_occlusion_texture(const String &p_path);

	String get_opacity_texture() const;
	void set_opacity_texture(const String &p_path);

	Ref<StandardMaterial3D> to_material(const String &p_base_path) const;

	// If materialx_xml is set, compile it into a ShaderMaterial using
	// USDMaterialXConverter; otherwise return null.
	Ref<ShaderMaterial> to_shader_material(const String &p_base_path) const;
};
