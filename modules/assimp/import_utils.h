/*************************************************************************/
/*  import_utils.h                                                       */
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

#ifndef IMPORT_UTILS_IMPORTER_ASSIMP_H
#define IMPORT_UTILS_IMPORTER_ASSIMP_H

#include "core/io/image_loader.h"
#include "import_state.h"

#include <assimp/SceneCombiner.h>
#include <assimp/cexport.h>
#include <assimp/cimport.h>
#include <assimp/matrix4x4.h>
#include <assimp/pbrmaterial.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/LogStream.hpp>
#include <assimp/Logger.hpp>
#include <string>

using namespace AssimpImporter;

#define AI_PROPERTIES aiTextureType_UNKNOWN, 0
#define AI_NULL 0, 0
#define AI_MATKEY_FBX_MAYA_BASE_COLOR_FACTOR "$raw.Maya|baseColor"
#define AI_MATKEY_FBX_MAYA_METALNESS_FACTOR "$raw.Maya|metalness"
#define AI_MATKEY_FBX_MAYA_DIFFUSE_ROUGHNESS_FACTOR "$raw.Maya|diffuseRoughness"

#define AI_MATKEY_FBX_MAYA_EMISSION_TEXTURE "$raw.Maya|emissionColor|file"
#define AI_MATKEY_FBX_MAYA_EMISSIVE_FACTOR "$raw.Maya|emission"
#define AI_MATKEY_FBX_MAYA_METALNESS_TEXTURE "$raw.Maya|metalness|file"
#define AI_MATKEY_FBX_MAYA_METALNESS_UV_XFORM "$raw.Maya|metalness|uvtrafo"
#define AI_MATKEY_FBX_MAYA_DIFFUSE_ROUGHNESS_TEXTURE "$raw.Maya|diffuseRoughness|file"
#define AI_MATKEY_FBX_MAYA_DIFFUSE_ROUGHNESS_UV_XFORM "$raw.Maya|diffuseRoughness|uvtrafo"
#define AI_MATKEY_FBX_MAYA_BASE_COLOR_TEXTURE "$raw.Maya|baseColor|file"
#define AI_MATKEY_FBX_MAYA_BASE_COLOR_UV_XFORM "$raw.Maya|baseColor|uvtrafo"
#define AI_MATKEY_FBX_MAYA_NORMAL_TEXTURE "$raw.Maya|normalCamera|file"
#define AI_MATKEY_FBX_MAYA_NORMAL_UV_XFORM "$raw.Maya|normalCamera|uvtrafo"

#define AI_MATKEY_FBX_NORMAL_TEXTURE "$raw.Maya|normalCamera|file"
#define AI_MATKEY_FBX_NORMAL_UV_XFORM "$raw.Maya|normalCamera|uvtrafo"

#define AI_MATKEY_FBX_MAYA_STINGRAY_DISPLACEMENT_SCALING_FACTOR "$raw.Maya|displacementscaling"
#define AI_MATKEY_FBX_MAYA_STINGRAY_BASE_COLOR_FACTOR "$raw.Maya|base_color"
#define AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_FACTOR "$raw.Maya|emissive"
#define AI_MATKEY_FBX_MAYA_STINGRAY_METALLIC_FACTOR "$raw.Maya|metallic"
#define AI_MATKEY_FBX_MAYA_STINGRAY_ROUGHNESS_FACTOR "$raw.Maya|roughness"
#define AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_INTENSITY_FACTOR "$raw.Maya|emissive_intensity"

#define AI_MATKEY_FBX_MAYA_STINGRAY_NORMAL_TEXTURE "$raw.Maya|TEX_normal_map|file"
#define AI_MATKEY_FBX_MAYA_STINGRAY_NORMAL_UV_XFORM "$raw.Maya|TEX_normal_map|uvtrafo"
#define AI_MATKEY_FBX_MAYA_STINGRAY_COLOR_TEXTURE "$raw.Maya|TEX_color_map|file"
#define AI_MATKEY_FBX_MAYA_STINGRAY_COLOR_UV_XFORM "$raw.Maya|TEX_color_map|uvtrafo"
#define AI_MATKEY_FBX_MAYA_STINGRAY_METALLIC_TEXTURE "$raw.Maya|TEX_metallic_map|file"
#define AI_MATKEY_FBX_MAYA_STINGRAY_METALLIC_UV_XFORM "$raw.Maya|TEX_metallic_map|uvtrafo"
#define AI_MATKEY_FBX_MAYA_STINGRAY_ROUGHNESS_TEXTURE "$raw.Maya|TEX_roughness_map|file"
#define AI_MATKEY_FBX_MAYA_STINGRAY_ROUGHNESS_UV_XFORM "$raw.Maya|TEX_roughness_map|uvtrafo"
#define AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_TEXTURE "$raw.Maya|TEX_emissive_map|file"
#define AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_UV_XFORM "$raw.Maya|TEX_emissive_map|uvtrafo"
#define AI_MATKEY_FBX_MAYA_STINGRAY_AO_TEXTURE "$raw.Maya|TEX_ao_map|file"
#define AI_MATKEY_FBX_MAYA_STINGRAY_AO_UV_XFORM "$raw.Maya|TEX_ao_map|uvtrafo"

/**
 * Assimp Utils
 * Conversion tools / glue code to convert from assimp to godot
*/
class AssimpUtils {
public:
	/**
	 * calculate tangents for mesh data from assimp data
	 */
	static void calc_tangent_from_mesh(const aiMesh *ai_mesh, int i, int tri_index, int index, Color *w) {
		const aiVector3D normals = ai_mesh->mAnimMeshes[i]->mNormals[tri_index];
		const Vector3 godot_normal = Vector3(normals.x, normals.y, normals.z);
		const aiVector3D tangent = ai_mesh->mAnimMeshes[i]->mTangents[tri_index];
		const Vector3 godot_tangent = Vector3(tangent.x, tangent.y, tangent.z);
		const aiVector3D bitangent = ai_mesh->mAnimMeshes[i]->mBitangents[tri_index];
		const Vector3 godot_bitangent = Vector3(bitangent.x, bitangent.y, bitangent.z);
		float d = godot_normal.cross(godot_tangent).dot(godot_bitangent) > 0.0f ? 1.0f : -1.0f;
		Color plane_tangent = Color(tangent.x, tangent.y, tangent.z, d);
		w[index] = plane_tangent;
	}

	struct AssetImportFbx {
		enum ETimeMode {
			TIME_MODE_DEFAULT = 0,
			TIME_MODE_120 = 1,
			TIME_MODE_100 = 2,
			TIME_MODE_60 = 3,
			TIME_MODE_50 = 4,
			TIME_MODE_48 = 5,
			TIME_MODE_30 = 6,
			TIME_MODE_30_DROP = 7,
			TIME_MODE_NTSC_DROP_FRAME = 8,
			TIME_MODE_NTSC_FULL_FRAME = 9,
			TIME_MODE_PAL = 10,
			TIME_MODE_CINEMA = 11,
			TIME_MODE_1000 = 12,
			TIME_MODE_CINEMA_ND = 13,
			TIME_MODE_CUSTOM = 14,
			TIME_MODE_TIME_MODE_COUNT = 15
		};
		enum UpAxis {
			UP_VECTOR_AXIS_X = 1,
			UP_VECTOR_AXIS_Y = 2,
			UP_VECTOR_AXIS_Z = 3
		};
		enum FrontAxis {
			FRONT_PARITY_EVEN = 1,
			FRONT_PARITY_ODD = 2,
		};

		enum CoordAxis {
			COORD_RIGHT = 0,
			COORD_LEFT = 1
		};
	};

	/** Get assimp string
    * automatically filters the string data
    */
	static String get_assimp_string(const aiString &p_string) {
		//convert an assimp String to a Godot String
		String name;
		name.parse_utf8(p_string.C_Str() /*,p_string.length*/);
		if (name.find(":") != -1) {
			String replaced_name = name.split(":")[1];
			print_verbose("Replacing " + name + " containing : with " + replaced_name);
			name = replaced_name;
		}

		return name;
	}

	static String get_anim_string_from_assimp(const aiString &p_string) {
		String name;
		name.parse_utf8(p_string.C_Str() /*,p_string.length*/);
		if (name.find(":") != -1) {
			String replaced_name = name.split(":")[1];
			print_verbose("Replacing " + name + " containing : with " + replaced_name);
			name = replaced_name;
		}
		return name;
	}

	/**
     * No filter logic get_raw_string_from_assimp
     * This just convers the aiString to a parsed utf8 string
     * Without removing special chars etc
     */
	static String get_raw_string_from_assimp(const aiString &p_string) {
		String name;
		name.parse_utf8(p_string.C_Str() /*,p_string.length*/);
		return name;
	}

	static Ref<Animation> import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps) {
		return Ref<Animation>();
	}

	/**
     * Converts aiMatrix4x4 to godot Transform
    */
	static const Transform assimp_matrix_transform(const aiMatrix4x4 p_matrix) {
		aiMatrix4x4 matrix = p_matrix;
		Transform xform;
		xform.set(matrix.a1, matrix.a2, matrix.a3, matrix.b1, matrix.b2, matrix.b3, matrix.c1, matrix.c2, matrix.c3, matrix.a4, matrix.b4, matrix.c4);
		return xform;
	}

	/** Get fbx fps for time mode meta data
     */
	static float get_fbx_fps(int32_t time_mode, const aiScene *p_scene) {
		switch (time_mode) {
			case AssetImportFbx::TIME_MODE_DEFAULT:
				return 24; //hack
			case AssetImportFbx::TIME_MODE_120:
				return 120;
			case AssetImportFbx::TIME_MODE_100:
				return 100;
			case AssetImportFbx::TIME_MODE_60:
				return 60;
			case AssetImportFbx::TIME_MODE_50:
				return 50;
			case AssetImportFbx::TIME_MODE_48:
				return 48;
			case AssetImportFbx::TIME_MODE_30:
				return 30;
			case AssetImportFbx::TIME_MODE_30_DROP:
				return 30;
			case AssetImportFbx::TIME_MODE_NTSC_DROP_FRAME:
				return 29.9700262f;
			case AssetImportFbx::TIME_MODE_NTSC_FULL_FRAME:
				return 29.9700262f;
			case AssetImportFbx::TIME_MODE_PAL:
				return 25;
			case AssetImportFbx::TIME_MODE_CINEMA:
				return 24;
			case AssetImportFbx::TIME_MODE_1000:
				return 1000;
			case AssetImportFbx::TIME_MODE_CINEMA_ND:
				return 23.976f;
			case AssetImportFbx::TIME_MODE_CUSTOM:
				int32_t frame_rate = -1;
				p_scene->mMetaData->Get("FrameRate", frame_rate);
				return frame_rate;
		}
		return 0;
	}

	/**
      * Get global transform for the current node - so we can use world space rather than
      * local space coordinates
      * useful if you need global - although recommend using local wherever possible over global
      * as you could break fbx scaling :)
      */
	static Transform _get_global_assimp_node_transform(const aiNode *p_current_node) {
		aiNode const *current_node = p_current_node;
		Transform xform;
		while (current_node != nullptr) {
			xform = assimp_matrix_transform(current_node->mTransformation) * xform;
			current_node = current_node->mParent;
		}
		return xform;
	}

	/**
	  * Find hardcoded textures from assimp which could be in many different directories
	  */
	static void find_texture_path(const String &p_path, _Directory &dir, String &path, bool &found, String extension) {
		Vector<String> paths;
		paths.push_back(path.get_basename() + extension);
		paths.push_back(path + extension);
		paths.push_back(path);
		paths.push_back(p_path.get_base_dir().plus_file(path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file(path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file(path.get_file()));
		paths.push_back(p_path.get_base_dir().plus_file("textures/" + path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("textures/" + path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("textures/" + path.get_file()));
		paths.push_back(p_path.get_base_dir().plus_file("Textures/" + path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("Textures/" + path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("Textures/" + path.get_file()));
		paths.push_back(p_path.get_base_dir().plus_file("../Textures/" + path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("../Textures/" + path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("../Textures/" + path.get_file()));
		paths.push_back(p_path.get_base_dir().plus_file("../textures/" + path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("../textures/" + path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("../textures/" + path.get_file()));
		paths.push_back(p_path.get_base_dir().plus_file("texture/" + path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("texture/" + path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("texture/" + path.get_file()));
		paths.push_back(p_path.get_base_dir().plus_file("Texture/" + path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("Texture/" + path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("Texture/" + path.get_file()));
		paths.push_back(p_path.get_base_dir().plus_file("../Texture/" + path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("../Texture/" + path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("../Texture/" + path.get_file()));
		paths.push_back(p_path.get_base_dir().plus_file("../texture/" + path.get_file().get_basename() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("../texture/" + path.get_file() + extension));
		paths.push_back(p_path.get_base_dir().plus_file("../texture/" + path.get_file()));
		for (int i = 0; i < paths.size(); i++) {
			if (dir.file_exists(paths[i])) {
				found = true;
				path = paths[i];
				return;
			}
		}
	}

	/** find the texture path for the supplied fbx path inside godot
      * very simple lookup for subfolders etc for a texture which may or may not be in a directory
      */
	static void find_texture_path(const String &r_p_path, String &r_path, bool &r_found) {
		_Directory dir;

		List<String> exts;
		ImageLoader::get_recognized_extensions(&exts);

		Vector<String> split_path = r_path.get_basename().split("*");
		if (split_path.size() == 2) {
			r_found = true;
			return;
		}

		if (dir.file_exists(r_p_path.get_base_dir() + r_path.get_file())) {
			r_path = r_p_path.get_base_dir() + r_path.get_file();
			r_found = true;
			return;
		}

		for (int32_t i = 0; i < exts.size(); i++) {
			if (r_found) {
				return;
			}
			find_texture_path(r_p_path, dir, r_path, r_found, "." + exts[i]);
		}
	}

	/**
	  * set_texture_mapping_mode
	  * Helper to check the mapping mode of the texture (repeat, clamp and mirror)
	  */
	static void set_texture_mapping_mode(aiTextureMapMode *map_mode, Ref<ImageTexture> texture) {
		ERR_FAIL_COND(texture.is_null());
		ERR_FAIL_COND(map_mode == nullptr);
		// FIXME: Commented out during Vulkan port.
		/*
		aiTextureMapMode tex_mode = map_mode[0];

		int32_t flags = Texture2D::FLAGS_DEFAULT;
		if (tex_mode == aiTextureMapMode_Wrap) {
			//Default
		} else if (tex_mode == aiTextureMapMode_Clamp) {
			flags = flags & ~Texture2D::FLAG_REPEAT;
		} else if (tex_mode == aiTextureMapMode_Mirror) {
			flags = flags | Texture2D::FLAG_MIRRORED_REPEAT;
		}
		texture->set_flags(flags);
		*/
	}

	/**
	  * Load or load from cache image :)
	  */
	static Ref<Image> load_image(ImportState &state, const aiScene *p_scene, String p_path) {
		Map<String, Ref<Image>>::Element *match = state.path_to_image_cache.find(p_path);

		// if our cache contains this image then don't bother
		if (match) {
			return match->get();
		}

		Vector<String> split_path = p_path.get_basename().split("*");
		if (split_path.size() == 2) {
			size_t texture_idx = split_path[1].to_int();
			ERR_FAIL_COND_V(texture_idx >= p_scene->mNumTextures, Ref<Image>());
			aiTexture *tex = p_scene->mTextures[texture_idx];
			String filename = AssimpUtils::get_raw_string_from_assimp(tex->mFilename);
			filename = filename.get_file();
			print_verbose("Open Asset Import: Loading embedded texture " + filename);
			if (tex->mHeight == 0) {
				if (tex->CheckFormat("png")) {
					ERR_FAIL_COND_V(Image::_png_mem_loader_func == nullptr, Ref<Image>());
					Ref<Image> img = Image::_png_mem_loader_func((uint8_t *)tex->pcData, tex->mWidth);
					ERR_FAIL_COND_V(img.is_null(), Ref<Image>());
					state.path_to_image_cache.insert(p_path, img);
					return img;
				} else if (tex->CheckFormat("jpg")) {
					ERR_FAIL_COND_V(Image::_jpg_mem_loader_func == nullptr, Ref<Image>());
					Ref<Image> img = Image::_jpg_mem_loader_func((uint8_t *)tex->pcData, tex->mWidth);
					ERR_FAIL_COND_V(img.is_null(), Ref<Image>());
					state.path_to_image_cache.insert(p_path, img);
					return img;
				} else if (tex->CheckFormat("dds")) {
					ERR_FAIL_COND_V_MSG(true, Ref<Image>(), "Open Asset Import: Embedded dds not implemented");
				}
			} else {
				Ref<Image> img;
				img.instance();
				PackedByteArray arr;
				uint32_t size = tex->mWidth * tex->mHeight;
				arr.resize(size);
				memcpy(arr.ptrw(), tex->pcData, size);
				ERR_FAIL_COND_V(arr.size() % 4 != 0, Ref<Image>());
				//ARGB8888 to RGBA8888
				for (int32_t i = 0; i < arr.size() / 4; i++) {
					arr.ptrw()[(4 * i) + 3] = arr[(4 * i) + 0];
					arr.ptrw()[(4 * i) + 0] = arr[(4 * i) + 1];
					arr.ptrw()[(4 * i) + 1] = arr[(4 * i) + 2];
					arr.ptrw()[(4 * i) + 2] = arr[(4 * i) + 3];
				}
				img->create(tex->mWidth, tex->mHeight, true, Image::FORMAT_RGBA8, arr);
				ERR_FAIL_COND_V(img.is_null(), Ref<Image>());
				state.path_to_image_cache.insert(p_path, img);
				return img;
			}
			return Ref<Image>();
		} else {
			Ref<Texture2D> texture = ResourceLoader::load(p_path);
			ERR_FAIL_COND_V(texture.is_null(), Ref<Image>());
			Ref<Image> image = texture->get_data();
			ERR_FAIL_COND_V(image.is_null(), Ref<Image>());
			state.path_to_image_cache.insert(p_path, image);
			return image;
		}

		return Ref<Image>();
	}

	/* create texture from assimp data, if found in path */
	static bool CreateAssimpTexture(
			AssimpImporter::ImportState &state,
			aiString texture_path,
			String &filename,
			String &path,
			AssimpImageData &image_state) {
		filename = get_raw_string_from_assimp(texture_path);
		path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
		bool found = false;
		find_texture_path(state.path, path, found);
		if (found) {
			image_state.raw_image = AssimpUtils::load_image(state, state.assimp_scene, path);
			if (image_state.raw_image.is_valid()) {
				image_state.texture.instance();
				image_state.texture->create_from_image(image_state.raw_image);
				// FIXME: Commented out during Vulkan port.
				//image_state.texture->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSY);
				return true;
			}
		}

		return false;
	}
	/** GetAssimpTexture
	  * Designed to retrieve textures for you
	  */
	static bool GetAssimpTexture(
			AssimpImporter::ImportState &state,
			aiMaterial *ai_material,
			aiTextureType texture_type,
			String &filename,
			String &path,
			AssimpImageData &image_state) {
		aiString ai_filename = aiString();
		if (AI_SUCCESS == ai_material->GetTexture(texture_type, 0, &ai_filename, nullptr, nullptr, nullptr, nullptr, image_state.map_mode)) {
			return CreateAssimpTexture(state, ai_filename, filename, path, image_state);
		}

		return false;
	}
};

#endif // IMPORT_UTILS_IMPORTER_ASSIMP_H
