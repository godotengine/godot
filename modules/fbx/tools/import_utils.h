/*************************************************************************/
/*  import_utils.h                                                       */
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

#ifndef IMPORT_UTILS_FBX_IMPORTER_H
#define IMPORT_UTILS_FBX_IMPORTER_H

#include "core/io/image_loader.h"

#include "data/import_state.h"
#include "fbx_parser/FBXDocument.h"

#include <string>

#define CONVERT_FBX_TIME(time) static_cast<double>(time) / 46186158000LL

/**
 * Import Utils
 * Conversion tools / glue code to convert from FBX to Godot
 */
class ImportUtils {
public:
	///	Convert a vector from degrees to radians.
	static Vector3 deg2rad(const Vector3 &p_rotation);

	///	Convert a vector from radians to degrees.
	static Vector3 rad2deg(const Vector3 &p_rotation);

	/// Converts rotation order vector (in rad) to quaternion.
	static Basis EulerToBasis(FBXDocParser::Model::RotOrder mode, const Vector3 &p_rotation);

	/// Converts rotation order vector (in rad) to quaternion.
	static Quaternion EulerToQuaternion(FBXDocParser::Model::RotOrder mode, const Vector3 &p_rotation);

	/// Converts basis into rotation order vector (in rad).
	static Vector3 BasisToEuler(FBXDocParser::Model::RotOrder mode, const Basis &p_rotation);

	/// Converts quaternion into rotation order vector (in rad).
	static Vector3 QuaternionToEuler(FBXDocParser::Model::RotOrder mode, const Quaternion &p_rotation);

	static void debug_xform(String name, const Transform3D &t) {
		print_verbose(name + " " + t.origin + " rotation: " + (t.basis.get_euler() * (180 / Math_PI)));
	}

	static String FBXNodeToName(const std::string &name) {
		// strip Model:: prefix, avoiding ambiguities (i.e. don't strip if
		// this causes ambiguities, well possible between empty identifiers,
		// such as "Model::" and ""). Make sure the behaviour is consistent
		// across multiple calls to FixNodeName().

		// We must remove this from the name
		// Some bones have this
		// SubDeformer::
		// Meshes, Joints have this, some other IK elements too.
		// Model::

		String node_name = String(name.c_str());

		if (node_name.substr(0, 7) == "Model::") {
			node_name = node_name.substr(7, node_name.length() - 7);
			return node_name.replace(":", "");
		}

		if (node_name.substr(0, 13) == "SubDeformer::") {
			node_name = node_name.substr(13, node_name.length() - 13);
			return node_name.replace(":", "");
		}

		if (node_name.substr(0, 11) == "AnimStack::") {
			node_name = node_name.substr(11, node_name.length() - 11);
			return node_name.replace(":", "");
		}

		if (node_name.substr(0, 15) == "AnimCurveNode::") {
			node_name = node_name.substr(15, node_name.length() - 15);
			return node_name.replace(":", "");
		}

		if (node_name.substr(0, 11) == "AnimCurve::") {
			node_name = node_name.substr(11, node_name.length() - 11);
			return node_name.replace(":", "");
		}

		if (node_name.substr(0, 10) == "Geometry::") {
			node_name = node_name.substr(10, node_name.length() - 10);
			return node_name.replace(":", "");
		}

		if (node_name.substr(0, 10) == "Material::") {
			node_name = node_name.substr(10, node_name.length() - 10);
			return node_name.replace(":", "");
		}

		if (node_name.substr(0, 9) == "Texture::") {
			node_name = node_name.substr(9, node_name.length() - 9);
			return node_name.replace(":", "");
		}

		return node_name.replace(":", "");
	}

	static std::string FBXAnimMeshName(const std::string &name) {
		if (name.length()) {
			size_t indexOf = name.find_first_of("::");
			if (indexOf != std::string::npos && indexOf < name.size() - 2) {
				return name.substr(indexOf + 2);
			}
		}
		return name.length() ? name : "AnimMesh";
	}

	static Vector3 safe_import_vector3(const Vector3 &p_vec) {
		Vector3 vector = p_vec;
		if (Math::is_zero_approx(vector.x)) {
			vector.x = 0;
		}

		if (Math::is_zero_approx(vector.y)) {
			vector.y = 0;
		}

		if (Math::is_zero_approx(vector.z)) {
			vector.z = 0;
		}
		return vector;
	}

	static void debug_xform(String name, const Basis &t) {
		//print_verbose(name + " rotation: " + (t.get_euler() * (180 / Math_PI)));
	}

	static Vector3 FixAxisConversions(Vector3 input) {
		return Vector3(input.x, input.y, input.z);
	}

	static void AlignMeshAxes(std::vector<Vector3> &vertex_data) {
		for (size_t x = 0; x < vertex_data.size(); x++) {
			vertex_data[x] = FixAxisConversions(vertex_data[x]);
		}
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

	/** Get fbx fps for time mode meta data
	 */
	static float get_fbx_fps(int32_t time_mode) {
		switch (time_mode) {
			case AssetImportFbx::TIME_MODE_DEFAULT:
				return 24;
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
				return -1;
		}
		return 0;
	}

	static float get_fbx_fps(const FBXDocParser::FileGlobalSettings *FBXSettings) {
		int time_mode = FBXSettings->TimeMode();

		// get the animation FPS
		float frames_per_second = get_fbx_fps(time_mode);

		// handle animation custom FPS time.
		if (time_mode == ImportUtils::AssetImportFbx::TIME_MODE_CUSTOM) {
			print_verbose("FBX Animation has custom FPS setting");
			frames_per_second = FBXSettings->CustomFrameRate();

			// not our problem this is the modeller, we can print as an error so they can fix the source.
			if (frames_per_second == 0) {
				print_error("Custom animation time in file is set to 0 value, animation won't play, please edit your file to correct the FPS value");
			}
		}
		return frames_per_second;
	}

	/**
	 * Find hardcoded textures from assimp which could be in many different directories
	 */

	/**
	 * set_texture_mapping_mode
	 * Helper to check the mapping mode of the texture (repeat, clamp and mirror)
	 */
	// static void set_texture_mapping_mode(aiTextureMapMode *map_mode, Ref<ImageTexture> texture) {
	// 	ERR_FAIL_COND(texture.is_null());
	// 	ERR_FAIL_COND(map_mode == nullptr);
	// 	aiTextureMapMode tex_mode = map_mode[0];

	// 	int32_t flags = Texture::FLAGS_DEFAULT;
	// 	if (tex_mode == aiTextureMapMode_Wrap) {
	// 		//Default
	// 	} else if (tex_mode == aiTextureMapMode_Clamp) {
	// 		flags = flags & ~Texture::FLAG_REPEAT;
	// 	} else if (tex_mode == aiTextureMapMode_Mirror) {
	// 		flags = flags | Texture::FLAG_MIRRORED_REPEAT;
	// 	}
	// 	texture->set_flags(flags);
	// }

	/**
	 * Load or load from cache image :)
	 * We need to upgrade this in the later version :) should not be hard
	 */
	//static Ref<Image> load_image(ImportState &state, const aiScene *p_scene, String p_path){
	// Map<String, Ref<Image> >::Element *match = state.path_to_image_cache.find(p_path);

	// // if our cache contains this image then don't bother
	// if (match) {
	// 	return match->get();
	// }

	// Vector<String> split_path = p_path.get_basename().split("*");
	// if (split_path.size() == 2) {
	// 	size_t texture_idx = split_path[1].to_int();
	// 	ERR_FAIL_COND_V(texture_idx >= p_scene->mNumTextures, Ref<Image>());
	// 	aiTexture *tex = p_scene->mTextures[texture_idx];
	// 	String filename = AssimpUtils::get_raw_string_from_assimp(tex->mFilename);
	// 	filename = filename.get_file();
	// 	print_verbose("Open Asset Import: Loading embedded texture " + filename);
	// 	if (tex->mHeight == 0) {
	// 		if (tex->CheckFormat("png")) {
	// 			Ref<Image> img = Image::_png_mem_loader_func((uint8_t *)tex->pcData, tex->mWidth);
	// 			ERR_FAIL_COND_V(img.is_null(), Ref<Image>());
	// 			state.path_to_image_cache.insert(p_path, img);
	// 			return img;
	// 		} else if (tex->CheckFormat("jpg")) {
	// 			Ref<Image> img = Image::_jpg_mem_loader_func((uint8_t *)tex->pcData, tex->mWidth);
	// 			ERR_FAIL_COND_V(img.is_null(), Ref<Image>());
	// 			state.path_to_image_cache.insert(p_path, img);
	// 			return img;
	// 		} else if (tex->CheckFormat("dds")) {
	// 			ERR_FAIL_COND_V_MSG(true, Ref<Image>(), "Open Asset Import: Embedded dds not implemented");
	// 		}
	// 	} else {
	// 		Ref<Image> img;
	// 		img.instantiate();
	// 		PoolByteArray arr;
	// 		uint32_t size = tex->mWidth * tex->mHeight;
	// 		arr.resize(size);
	// 		memcpy(arr.write().ptr(), tex->pcData, size);
	// 		ERR_FAIL_COND_V(arr.size() % 4 != 0, Ref<Image>());
	// 		//ARGB8888 to RGBA8888
	// 		for (int32_t i = 0; i < arr.size() / 4; i++) {
	// 			arr.write().ptr()[(4 * i) + 3] = arr[(4 * i) + 0];
	// 			arr.write().ptr()[(4 * i) + 0] = arr[(4 * i) + 1];
	// 			arr.write().ptr()[(4 * i) + 1] = arr[(4 * i) + 2];
	// 			arr.write().ptr()[(4 * i) + 2] = arr[(4 * i) + 3];
	// 		}
	// 		img->create(tex->mWidth, tex->mHeight, true, Image::FORMAT_RGBA8, arr);
	// 		ERR_FAIL_COND_V(img.is_null(), Ref<Image>());
	// 		state.path_to_image_cache.insert(p_path, img);
	// 		return img;
	// 	}
	// 	return Ref<Image>();
	// } else {
	// 	Ref<Texture> texture = ResourceLoader::load(p_path);
	// 	ERR_FAIL_COND_V(texture.is_null(), Ref<Image>());
	// 	Ref<Image> image = texture->get_image();
	// 	ERR_FAIL_COND_V(image.is_null(), Ref<Image>());
	// 	state.path_to_image_cache.insert(p_path, image);
	// 	return image;
	// }

	// return Ref<Image>();
	//}

	// /* create texture from assimp data, if found in path */
	// static bool CreateAssimpTexture(
	// 		AssimpImporter::ImportState &state,
	// 		aiString texture_path,
	// 		String &filename,
	// 		String &path,
	// 		AssimpImageData &image_state) {
	// 	filename = get_raw_string_from_assimp(texture_path);
	// 	path = state.path.get_base_dir().plus_file(filename.replace("\\", "/"));
	// 	bool found = false;
	// 	find_texture_path(state.path, path, found);
	// 	if (found) {
	// 		image_state.raw_image = AssimpUtils::load_image(state, state.assimp_scene, path);
	// 		if (image_state.raw_image.is_valid()) {
	// 			image_state.texture.instantiate();
	// 			image_state.texture->create_from_image(image_state.raw_image);
	// 			image_state.texture->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSY);
	// 			return true;
	// 		}
	// 	}

	// 	return false;
	// }
	// /** GetAssimpTexture
	//   * Designed to retrieve textures for you
	//   */
	// static bool GetAssimpTexture(
	// 		AssimpImporter::ImportState &state,
	// 		aiMaterial *ai_material,
	// 		aiTextureType texture_type,
	// 		String &filename,
	// 		String &path,
	// 		AssimpImageData &image_state) {
	// 	aiString ai_filename = aiString();
	// 	if (AI_SUCCESS == ai_material->GetTexture(texture_type, 0, &ai_filename, nullptr, nullptr, nullptr, nullptr, image_state.map_mode)) {
	// 		return CreateAssimpTexture(state, ai_filename, filename, path, image_state);
	// 	}

	// 	return false;
	// }
};

// Apply the transforms so the basis will have scale 1.
Transform3D get_unscaled_transform(const Transform3D &p_initial, real_t p_scale);

/// Uses the Newell's method to compute any polygon normal.
/// The polygon must be at least size of 3 or bigger.
Vector3 get_poly_normal(const std::vector<Vector3> &p_vertices);

#endif // IMPORT_UTILS_FBX_IMPORTER_H
