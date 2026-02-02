/**************************************************************************/
/*  usd_material.cpp                                                      */
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

#include "usd_material.h"

#include "usd_materialx_converter.h"

#include "core/io/resource_loader.h"
#include "scene/resources/image_texture.h"

void USDMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &USDMaterial::get_name);
	ClassDB::bind_method(D_METHOD("set_name", "name"), &USDMaterial::set_name);
	ClassDB::bind_method(D_METHOD("get_diffuse_color"), &USDMaterial::get_diffuse_color);
	ClassDB::bind_method(D_METHOD("set_diffuse_color", "color"), &USDMaterial::set_diffuse_color);
	ClassDB::bind_method(D_METHOD("get_metallic"), &USDMaterial::get_metallic);
	ClassDB::bind_method(D_METHOD("set_metallic", "metallic"), &USDMaterial::set_metallic);
	ClassDB::bind_method(D_METHOD("get_roughness"), &USDMaterial::get_roughness);
	ClassDB::bind_method(D_METHOD("set_roughness", "roughness"), &USDMaterial::set_roughness);
	ClassDB::bind_method(D_METHOD("get_emissive_color"), &USDMaterial::get_emissive_color);
	ClassDB::bind_method(D_METHOD("set_emissive_color", "color"), &USDMaterial::set_emissive_color);
	ClassDB::bind_method(D_METHOD("get_opacity"), &USDMaterial::get_opacity);
	ClassDB::bind_method(D_METHOD("set_opacity", "opacity"), &USDMaterial::set_opacity);
	ClassDB::bind_method(D_METHOD("get_ior"), &USDMaterial::get_ior);
	ClassDB::bind_method(D_METHOD("set_ior", "ior"), &USDMaterial::set_ior);
	ClassDB::bind_method(D_METHOD("get_clearcoat"), &USDMaterial::get_clearcoat);
	ClassDB::bind_method(D_METHOD("set_clearcoat", "clearcoat"), &USDMaterial::set_clearcoat);
	ClassDB::bind_method(D_METHOD("get_clearcoat_roughness"), &USDMaterial::get_clearcoat_roughness);
	ClassDB::bind_method(D_METHOD("set_clearcoat_roughness", "clearcoat_roughness"), &USDMaterial::set_clearcoat_roughness);
	ClassDB::bind_method(D_METHOD("get_diffuse_texture"), &USDMaterial::get_diffuse_texture);
	ClassDB::bind_method(D_METHOD("set_diffuse_texture", "path"), &USDMaterial::set_diffuse_texture);
	ClassDB::bind_method(D_METHOD("get_metallic_texture"), &USDMaterial::get_metallic_texture);
	ClassDB::bind_method(D_METHOD("set_metallic_texture", "path"), &USDMaterial::set_metallic_texture);
	ClassDB::bind_method(D_METHOD("get_roughness_texture"), &USDMaterial::get_roughness_texture);
	ClassDB::bind_method(D_METHOD("set_roughness_texture", "path"), &USDMaterial::set_roughness_texture);
	ClassDB::bind_method(D_METHOD("get_normal_texture"), &USDMaterial::get_normal_texture);
	ClassDB::bind_method(D_METHOD("set_normal_texture", "path"), &USDMaterial::set_normal_texture);
	ClassDB::bind_method(D_METHOD("get_emissive_texture"), &USDMaterial::get_emissive_texture);
	ClassDB::bind_method(D_METHOD("set_emissive_texture", "path"), &USDMaterial::set_emissive_texture);
	ClassDB::bind_method(D_METHOD("get_occlusion_texture"), &USDMaterial::get_occlusion_texture);
	ClassDB::bind_method(D_METHOD("set_occlusion_texture", "path"), &USDMaterial::set_occlusion_texture);
	ClassDB::bind_method(D_METHOD("get_opacity_texture"), &USDMaterial::get_opacity_texture);
	ClassDB::bind_method(D_METHOD("set_opacity_texture", "path"), &USDMaterial::set_opacity_texture);
	ClassDB::bind_method(D_METHOD("to_material", "base_path"), &USDMaterial::to_material);
	ClassDB::bind_method(D_METHOD("to_shader_material", "base_path"), &USDMaterial::to_shader_material);
	ClassDB::bind_method(D_METHOD("get_materialx_xml"), &USDMaterial::get_materialx_xml);
	ClassDB::bind_method(D_METHOD("set_materialx_xml", "xml"), &USDMaterial::set_materialx_xml);
	ClassDB::bind_method(D_METHOD("has_materialx"), &USDMaterial::has_materialx);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "materialx_xml"), "set_materialx_xml", "get_materialx_xml");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "name"), "set_name", "get_name");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "diffuse_color"), "set_diffuse_color", "get_diffuse_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "metallic"), "set_metallic", "get_metallic");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "roughness"), "set_roughness", "get_roughness");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "emissive_color"), "set_emissive_color", "get_emissive_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "opacity"), "set_opacity", "get_opacity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ior"), "set_ior", "get_ior");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "clearcoat"), "set_clearcoat", "get_clearcoat");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "clearcoat_roughness"), "set_clearcoat_roughness", "get_clearcoat_roughness");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "diffuse_texture"), "set_diffuse_texture", "get_diffuse_texture");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "metallic_texture"), "set_metallic_texture", "get_metallic_texture");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "roughness_texture"), "set_roughness_texture", "get_roughness_texture");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "normal_texture"), "set_normal_texture", "get_normal_texture");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "emissive_texture"), "set_emissive_texture", "get_emissive_texture");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "occlusion_texture"), "set_occlusion_texture", "get_occlusion_texture");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "opacity_texture"), "set_opacity_texture", "get_opacity_texture");
}

String USDMaterial::get_name() const {
	return name;
}

void USDMaterial::set_name(const String &p_name) {
	name = p_name;
}

String USDMaterial::get_materialx_xml() const {
	return materialx_xml;
}

void USDMaterial::set_materialx_xml(const String &p_xml) {
	materialx_xml = p_xml;
}

bool USDMaterial::has_materialx() const {
	return !materialx_xml.is_empty();
}

Color USDMaterial::get_diffuse_color() const {
	return diffuse_color;
}

void USDMaterial::set_diffuse_color(const Color &p_color) {
	diffuse_color = p_color;
}

float USDMaterial::get_metallic() const {
	return metallic;
}

void USDMaterial::set_metallic(float p_metallic) {
	metallic = p_metallic;
}

float USDMaterial::get_roughness() const {
	return roughness;
}

void USDMaterial::set_roughness(float p_roughness) {
	roughness = p_roughness;
}

Color USDMaterial::get_emissive_color() const {
	return emissive_color;
}

void USDMaterial::set_emissive_color(const Color &p_color) {
	emissive_color = p_color;
}

float USDMaterial::get_opacity() const {
	return opacity;
}

void USDMaterial::set_opacity(float p_opacity) {
	opacity = p_opacity;
}

float USDMaterial::get_ior() const {
	return ior;
}

void USDMaterial::set_ior(float p_ior) {
	ior = p_ior;
}

float USDMaterial::get_clearcoat() const {
	return clearcoat;
}

void USDMaterial::set_clearcoat(float p_clearcoat) {
	clearcoat = p_clearcoat;
}

float USDMaterial::get_clearcoat_roughness() const {
	return clearcoat_roughness;
}

void USDMaterial::set_clearcoat_roughness(float p_clearcoat_roughness) {
	clearcoat_roughness = p_clearcoat_roughness;
}

String USDMaterial::get_diffuse_texture() const {
	return diffuse_texture;
}

void USDMaterial::set_diffuse_texture(const String &p_path) {
	diffuse_texture = p_path;
}

String USDMaterial::get_metallic_texture() const {
	return metallic_texture;
}

void USDMaterial::set_metallic_texture(const String &p_path) {
	metallic_texture = p_path;
}

String USDMaterial::get_roughness_texture() const {
	return roughness_texture;
}

void USDMaterial::set_roughness_texture(const String &p_path) {
	roughness_texture = p_path;
}

String USDMaterial::get_normal_texture() const {
	return normal_texture;
}

void USDMaterial::set_normal_texture(const String &p_path) {
	normal_texture = p_path;
}

String USDMaterial::get_emissive_texture() const {
	return emissive_texture;
}

void USDMaterial::set_emissive_texture(const String &p_path) {
	emissive_texture = p_path;
}

String USDMaterial::get_occlusion_texture() const {
	return occlusion_texture;
}

void USDMaterial::set_occlusion_texture(const String &p_path) {
	occlusion_texture = p_path;
}

String USDMaterial::get_opacity_texture() const {
	return opacity_texture;
}

void USDMaterial::set_opacity_texture(const String &p_path) {
	opacity_texture = p_path;
}

static Ref<Texture2D> _load_texture(const String &p_base_path, const String &p_texture_path) {
	if (p_texture_path.is_empty()) {
		return Ref<Texture2D>();
	}

	// Build the full path: if the texture path is relative, resolve it
	// against the base directory of the USD file.
	String full_path = p_texture_path;
	if (!p_texture_path.is_absolute_path()) {
		full_path = p_base_path.path_join(p_texture_path);
	}

	// Attempt to load through the ResourceLoader so that imported
	// textures (.import cache) are resolved automatically.
	Ref<Resource> res = ResourceLoader::load(full_path);
	if (res.is_valid()) {
		Ref<Texture2D> tex = res;
		if (tex.is_valid()) {
			return tex;
		}
	}

	return Ref<Texture2D>();
}

Ref<StandardMaterial3D> USDMaterial::to_material(const String &p_base_path) const {
	Ref<StandardMaterial3D> mat;
	mat.instantiate();

	// Albedo / diffuse.
	mat->set_albedo(diffuse_color);

	Ref<Texture2D> albedo_tex = _load_texture(p_base_path, diffuse_texture);
	if (albedo_tex.is_valid()) {
		mat->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, albedo_tex);
	}

	// Metallic.
	mat->set_metallic(metallic);

	Ref<Texture2D> metallic_tex = _load_texture(p_base_path, metallic_texture);
	if (metallic_tex.is_valid()) {
		mat->set_texture(BaseMaterial3D::TEXTURE_METALLIC, metallic_tex);
	}

	// Roughness.
	mat->set_roughness(roughness);

	Ref<Texture2D> roughness_tex = _load_texture(p_base_path, roughness_texture);
	if (roughness_tex.is_valid()) {
		mat->set_texture(BaseMaterial3D::TEXTURE_ROUGHNESS, roughness_tex);
	}

	// Emission.
	if (emissive_color != Color(0, 0, 0)) {
		mat->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
		mat->set_emission(emissive_color);
		mat->set_emission_energy_multiplier(1.0);
	}

	Ref<Texture2D> emission_tex = _load_texture(p_base_path, emissive_texture);
	if (emission_tex.is_valid()) {
		mat->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
		mat->set_texture(BaseMaterial3D::TEXTURE_EMISSION, emission_tex);
	}

	// Normal map.
	Ref<Texture2D> normal_tex = _load_texture(p_base_path, normal_texture);
	if (normal_tex.is_valid()) {
		mat->set_feature(BaseMaterial3D::FEATURE_NORMAL_MAPPING, true);
		mat->set_texture(BaseMaterial3D::TEXTURE_NORMAL, normal_tex);
	}

	// Ambient occlusion.
	Ref<Texture2D> ao_tex = _load_texture(p_base_path, occlusion_texture);
	if (ao_tex.is_valid()) {
		mat->set_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION, true);
		mat->set_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION, ao_tex);
	}

	// Transparency.
	if (opacity < 1.0) {
		mat->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
		Color albedo_with_alpha = diffuse_color;
		albedo_with_alpha.a = opacity;
		mat->set_albedo(albedo_with_alpha);
	}

	Ref<Texture2D> opacity_tex = _load_texture(p_base_path, opacity_texture);
	if (opacity_tex.is_valid()) {
		mat->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
		// When an opacity texture is present, use it as the albedo texture
		// alpha channel source. If an albedo texture is already set, the
		// artist is expected to have pre-multiplied alpha; otherwise we set
		// the opacity texture as albedo so the alpha channel is available.
		if (!albedo_tex.is_valid()) {
			mat->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, opacity_tex);
		}
	}

	// Clearcoat.
	if (clearcoat > 0.0) {
		mat->set_feature(BaseMaterial3D::FEATURE_CLEARCOAT, true);
		mat->set_clearcoat(clearcoat);
		mat->set_clearcoat_roughness(clearcoat_roughness);
	}

	return mat;
}

Ref<ShaderMaterial> USDMaterial::to_shader_material(const String &p_base_path) const {
	if (materialx_xml.is_empty()) {
		return Ref<ShaderMaterial>();
	}

	Ref<USDMaterialXConverter> converter;
	converter.instantiate();
	return converter->convert_materialx_from_xml(materialx_xml, p_base_path);
}
