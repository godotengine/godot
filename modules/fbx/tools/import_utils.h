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

#ifndef IMPORT_UTILS_FBX_IMPORTER_H
#define IMPORT_UTILS_FBX_IMPORTER_H

#include "modules/fbx/data/import_state.h"
#include "thirdparty/assimp_fbx/FBXDocument.h"

#include "core/io/image_loader.h"
#include <string>

#define CONVERT_FBX_TIME(time) static_cast<double>(time) / 46186158000LL

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
	static Basis EulerToBasis(Assimp::FBX::Model::RotOrder mode, const Vector3 &p_rotation);

	/// Converts rotation order vector (in rad) to quaternion.
	static Quat EulerToQuaternion(Assimp::FBX::Model::RotOrder mode, const Vector3 &p_rotation);

	/// Converts basis into rotation order vector (in rad).
	static Vector3 BasisToEuler(Assimp::FBX::Model::RotOrder mode, const Basis &p_rotation);

	/// Converts quaternion into rotation order vector (in rad).
	static Vector3 QuaternionToEuler(Assimp::FBX::Model::RotOrder mode, const Quat &p_rotation);

	static void debug_xform(String name, const Transform &t) {
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
		if (Math::is_equal_approx(0, vector.x)) {
			vector.x = 0;
		}

		if (Math::is_equal_approx(0, vector.y)) {
			vector.y = 0;
		}

		if (Math::is_equal_approx(0, vector.z)) {
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
			case AssetImportFbx::TIME_MODE_DEFAULT: return 24;
			case AssetImportFbx::TIME_MODE_120: return 120;
			case AssetImportFbx::TIME_MODE_100: return 100;
			case AssetImportFbx::TIME_MODE_60: return 60;
			case AssetImportFbx::TIME_MODE_50: return 50;
			case AssetImportFbx::TIME_MODE_48: return 48;
			case AssetImportFbx::TIME_MODE_30: return 30;
			case AssetImportFbx::TIME_MODE_30_DROP: return 30;
			case AssetImportFbx::TIME_MODE_NTSC_DROP_FRAME: return 29.9700262f;
			case AssetImportFbx::TIME_MODE_NTSC_FULL_FRAME: return 29.9700262f;
			case AssetImportFbx::TIME_MODE_PAL: return 25;
			case AssetImportFbx::TIME_MODE_CINEMA: return 24;
			case AssetImportFbx::TIME_MODE_1000: return 1000;
			case AssetImportFbx::TIME_MODE_CINEMA_ND: return 23.976f;
			case AssetImportFbx::TIME_MODE_CUSTOM: return -1;
		}
		return 0;
	}

	static float get_fbx_fps(const Assimp::FBX::FileGlobalSettings *FBXSettings) {
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
};

// Apply the transforms so the basis will have scale 1.
Transform get_unscaled_transform(const Transform &p_initial, real_t p_scale);

/// Uses the Newell's method to compute any polygon normal.
/// The polygon must be at least size of 3 or bigger.
Vector3 get_poly_normal(const std::vector<Vector3> &p_vertices);

#endif // IMPORT_UTILS_FBX_IMPORTER_H
