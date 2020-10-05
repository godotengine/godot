/*************************************************************************/
/*  fbx_material.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_FBX_MATERIAL_H
#define GODOT_FBX_MATERIAL_H

#include "core/reference.h"
#include "core/ustring.h"
#include "modules/fbx/tools/import_utils.h"

struct FBXMaterial : public Reference {
	String material_name = String();
	mutable const Assimp::FBX::Material *material = nullptr;

	/* Godot materials
	 *** Texture Maps:
	 * Albedo - color, texture
	 * Metallic - specular, metallic, texture
	 * Roughness - roughness, texture
	 * Emission - color, texture
	 * Normal Map - scale, texture
	 * Ambient Occlusion - texture
	 * Refraction - scale, texture
	 *** Has Settings for:
	 * UV1 - SCALE, OFFSET
	 * UV2 - SCALE, OFFSET
	 *** Flags for
	 * Transparent
	 * Cull Mode
	 */

	enum class MapMode {
		AlbedoM = 0,
		MetallicM,
		SpecularM,
		EmissionM,
		RoughnessM,
		NormalM,
		AmbientOcclusionM,
		RefractionM,
		ReflectionM,
	};
//
	// TODO make this static?
	const std::map<std::string, StandardMaterial3D::Transparency> fbx_feature_mapping_desc = {
		/* Transparent */
		{ "TransparentColor", StandardMaterial3D::Transparency::TRANSPARENCY_ALPHA },
		{ "Maya|opacity", StandardMaterial3D::Transparency::TRANSPARENCY_ALPHA }
	};

	// TODO make this static?
	const std::map<std::string, StandardMaterial3D::TextureParam> fbx_texture_mapping_desc = {
		/* Diffuse */
		{ "Maya|base", StandardMaterial3D::TextureParam::TEXTURE_ALBEDO },
		{ "DiffuseColor", StandardMaterial3D::TextureParam::TEXTURE_ALBEDO },
		{ "Maya|DiffuseTexture", StandardMaterial3D::TextureParam::TEXTURE_ALBEDO },
		{ "Maya|baseColor", StandardMaterial3D::TextureParam::TEXTURE_ALBEDO },
		{ "Maya|baseColor|file", StandardMaterial3D::TextureParam::TEXTURE_ALBEDO },
		{ "3dsMax|Parameters|base_color_map", StandardMaterial3D::TextureParam::TEXTURE_ALBEDO },
		{ "Maya|TEX_color_map|file", StandardMaterial3D::TextureParam::TEXTURE_ALBEDO },
		{ "Maya|TEX_color_map", StandardMaterial3D::TextureParam::TEXTURE_ALBEDO },
		/* Emission */
		{ "EmissiveColor", StandardMaterial3D::TextureParam::TEXTURE_EMISSION },
		{ "EmissiveFactor", StandardMaterial3D::TextureParam::TEXTURE_EMISSION },
		{ "Maya|emissionColor", StandardMaterial3D::TextureParam::TEXTURE_EMISSION },
		{ "Maya|emissionColor|file", StandardMaterial3D::TextureParam::TEXTURE_EMISSION },
		{ "3dsMax|Parameters|emission_map", StandardMaterial3D::TextureParam::TEXTURE_EMISSION },
		{ "Maya|TEX_emissive_map", StandardMaterial3D::TextureParam::TEXTURE_EMISSION },
		{ "Maya|TEX_emissive_map|file", StandardMaterial3D::TextureParam::TEXTURE_EMISSION },
		/* Metallic */
		{ "Maya|metalness", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "Maya|metalness|file", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "3dsMax|Parameters|metalness_map", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "Maya|TEX_metallic_map", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "Maya|TEX_metallic_map|file", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "SpecularColor", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "Maya|specularColor", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "Maya|SpecularTexture", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "Maya|SpecularTexture|file", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		{ "ShininessExponent", StandardMaterial3D::TextureParam::TEXTURE_METALLIC },
		/* Roughness */
		{ "Maya|diffuseRoughness", StandardMaterial3D::TextureParam::TEXTURE_ROUGHNESS },
		{ "Maya|diffuseRoughness|file", StandardMaterial3D::TextureParam::TEXTURE_ROUGHNESS },
		{ "3dsMax|Parameters|roughness_map", StandardMaterial3D::TextureParam::TEXTURE_ROUGHNESS },
		{ "Maya|TEX_roughness_map", StandardMaterial3D::TextureParam::TEXTURE_ROUGHNESS },
		{ "Maya|TEX_roughness_map|file", StandardMaterial3D::TextureParam::TEXTURE_ROUGHNESS },
		{ "ReflectionFactor", StandardMaterial3D::TextureParam::TEXTURE_ROUGHNESS },
		{ "Maya|specularRoughness", StandardMaterial3D::TextureParam::TEXTURE_ROUGHNESS },
		/* Normal */
		{ "NormalMap", StandardMaterial3D::TextureParam::TEXTURE_NORMAL },
		{ "Bump", StandardMaterial3D::TextureParam::TEXTURE_NORMAL },
		{ "3dsMax|Parameters|bump_map", StandardMaterial3D::TextureParam::TEXTURE_NORMAL },
		{ "Maya|NormalTexture", StandardMaterial3D::TextureParam::TEXTURE_NORMAL },
		{ "Maya|normalCamera", StandardMaterial3D::TextureParam::TEXTURE_NORMAL },
		{ "Maya|normalCamera|file", StandardMaterial3D::TextureParam::TEXTURE_NORMAL },
		{ "Maya|TEX_normal_map", StandardMaterial3D::TextureParam::TEXTURE_NORMAL },
		{ "Maya|TEX_normal_map|file", StandardMaterial3D::TextureParam::TEXTURE_NORMAL },
		/* AO */
		{ "Maya|TEX_ao_map", StandardMaterial3D::TextureParam::TEXTURE_AMBIENT_OCCLUSION },
		{ "Maya|TEX_ao_map|file", StandardMaterial3D::TextureParam::TEXTURE_AMBIENT_OCCLUSION },
		//	{"TransparentColor",StandardMaterial3D::TextureParam::TEXTURE_CHANNEL_ALPHA },
		//	{"TransparencyFactor",StandardMaterial3D::TextureParam::TEXTURE_CHANNEL_ALPHA }
	};

	// TODO make this static?
	enum PropertyDesc {
		PROPERTY_DESC_NOT_FOUND,
		PROPERTY_DESC_ALBEDO_COLOR,
		PROPERTY_DESC_TRANSPARENT,
		PROPERTY_DESC_METALLIC,
		PROPERTY_DESC_ROUGHNESS,
		PROPERTY_DESC_COAT,
		PROPERTY_DESC_COAT_ROUGHNESS,
		PROPERTY_DESC_EMISSIVE,
		PROPERTY_DESC_EMISSIVE_COLOR,
		PROPERTY_DESC_IGNORE
	};

	const std::map<std::string, PropertyDesc> fbx_properties_desc = {
		/* Albedo */
		{ "DiffuseColor", PROPERTY_DESC_ALBEDO_COLOR },
		{ "Maya|baseColor", PROPERTY_DESC_ALBEDO_COLOR },

		/* Transparent */
		{ "Opacity", PROPERTY_DESC_TRANSPARENT },
		{ "TransparencyFactor", PROPERTY_DESC_TRANSPARENT },
		{ "Maya|opacity", PROPERTY_DESC_TRANSPARENT },

		/* Metallic */
		{ "Shininess", PROPERTY_DESC_METALLIC },
		{ "Reflectivity", PROPERTY_DESC_METALLIC },
		{ "Maya|metalness", PROPERTY_DESC_METALLIC },

		/* Roughness */
		{ "Maya|diffuseRoughness", PROPERTY_DESC_ROUGHNESS },

		/* Coat */
		{ "Maya|coat", PROPERTY_DESC_COAT },

		/* Coat roughness */
		{ "Maya|coatRoughness", PROPERTY_DESC_COAT_ROUGHNESS },

		/* Emissive */
		{ "Maya|emission", PROPERTY_DESC_EMISSIVE },

		/* Emissive color */
		{ "EmissiveColor", PROPERTY_DESC_EMISSIVE_COLOR },
		{ "Maya|emissionColor", PROPERTY_DESC_EMISSIVE_COLOR },

		/* Ignore */
		{ "Maya", PROPERTY_DESC_IGNORE },
		{ "Diffuse", PROPERTY_DESC_IGNORE },
		{ "Maya|TypeId", PROPERTY_DESC_IGNORE },
		{ "Ambient", PROPERTY_DESC_IGNORE },
		{ "AmbientColor", PROPERTY_DESC_IGNORE },
		{ "ShininessExponent", PROPERTY_DESC_IGNORE },
		{ "Specular", PROPERTY_DESC_IGNORE },
		{ "SpecularColor", PROPERTY_DESC_IGNORE },
		{ "SpecularFactor", PROPERTY_DESC_IGNORE },
		//{ "BumpFactor", PROPERTY_DESC_IGNORE },
		{ "Maya|exitToBackground", PROPERTY_DESC_IGNORE },
		{ "Maya|indirectDiffuse", PROPERTY_DESC_IGNORE },
		{ "Maya|indirectSpecular", PROPERTY_DESC_IGNORE },
		{ "Maya|internalReflections", PROPERTY_DESC_IGNORE },
		{ "DiffuseFactor", PROPERTY_DESC_IGNORE },
		{ "AmbientFactor", PROPERTY_DESC_IGNORE },
		{ "ReflectionColor", PROPERTY_DESC_IGNORE },
		{ "Emissive", PROPERTY_DESC_IGNORE },
		{ "Maya|coatColor", PROPERTY_DESC_IGNORE },
		{ "Maya|coatNormal", PROPERTY_DESC_IGNORE },
		{ "Maya|coatIOR", PROPERTY_DESC_IGNORE },
	};

	struct TextureFileMapping {
		StandardMaterial3D::TextureParam map_mode = StandardMaterial3D::TEXTURE_ALBEDO;
		String name = String();
		const Assimp::FBX::Texture *texture = nullptr;
	};

	/* storing the texture properties like color */
	template <class T>
	struct TexturePropertyMapping : Reference {
		StandardMaterial3D::TextureParam map_mode = StandardMaterial3D::TextureParam::TEXTURE_ALBEDO;
		const T property = T();
	};

	static void add_search_string(String p_filename, String p_current_directory, String search_directory, Vector<String> &texture_search_paths);

	static String find_texture_path_by_filename(const String p_filename, const String p_current_directory);

	String get_material_name() const;

	void set_imported_material(const Assimp::FBX::Material *p_material);

	struct MaterialInfo {
		Vector<TextureFileMapping> textures;
		Vector<StandardMaterial3D::Feature> features;
	};
	/// Extracts the material information.
	MaterialInfo extract_material_info(const Assimp::FBX::Material *material) const;

	Ref<StandardMaterial3D> import_material(ImportState &state);
};

#endif // GODOT_FBX_MATERIAL_H
