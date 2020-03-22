/*************************************************************************/
/*  editor_scene_importer_assimp.h                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef EDITOR_SCENE_FBX_IMPORTER_H
#define EDITOR_SCENE_FBX_IMPORTER_H

#ifdef TOOLS_ENABLED

#include "core/bind/core_bind.h"
#include "core/io/resource_importer.h"
#include "core/vector.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"

#include <assimp/matrix4x4.h>
#include <assimp/types.h>
#include <assimp/Importer.hpp>

#include "import_state.h"
#include "import_utils.h"
#include <code/FBX/FBXDocument.h>
#include <code/FBX/FBXImportSettings.h>
#include <code/FBX/FBXMeshGeometry.h>
#include <code/FBX/FBXUtil.h>
#include <map>

#define CONVERT_FBX_TIME(time) static_cast<double>(time) / 46186158000LL

using namespace AssimpImporter;

class EditorSceneImporterFBX : public EditorSceneImporter {
private:
	GDCLASS(EditorSceneImporterFBX, EditorSceneImporter);

	struct AssetImportAnimation {
		enum Interpolation {
			INTERP_LINEAR,
			INTERP_STEP,
			INTERP_CATMULLROMSPLINE,
			INTERP_CUBIC_SPLINE
		};
	};

	enum TransformationComp {
		TransformationComp_Translation,
		TransformationComp_Scaling,
		TransformationComp_Rotation,
		TransformationComp_RotationOffset,
		TransformationComp_RotationPivot,
		TransformationComp_PreRotation,
		TransformationComp_PostRotation,
		TransformationComp_ScalingOffset,
		TransformationComp_ScalingPivot,
		TransformationComp_GeometricTranslation,
		TransformationComp_GeometricRotation,
		TransformationComp_GeometricScaling,
		TransformationComp_MAXIMUM
	};

	struct BoneInfo {
		Vector<int> bone_id;
		Vector<float> weights;
	};

	// ------------------------------------------------------------------------------------------------
	template <typename T>
	const T *ProcessDOMConnection(
			const Assimp::FBX::Document *doc,
			const char *element_to_find,
			uint64_t current_element,
			bool reverse = false) {

		const std::vector<const Assimp::FBX::Connection *> &conns = reverse ? doc->GetConnectionsByDestinationSequenced(current_element) : doc->GetConnectionsBySourceSequenced(current_element);
		//print_verbose("[doc] looking for " + String(element_to_find));
		// using the temp pattern here so we can debug before it returns
		// in some cases we return too early, with 'deformer object base class' in wrong place
		// in assimp this means we can accidentally return too early...
		const T *return_obj = nullptr;

		for (const Assimp::FBX::Connection *con : conns) {
			const Assimp::FBX::Object *const source_object = reverse ? con->DestinationObject() : con->SourceObject();
			const Assimp::FBX::Object *const dest_object = reverse ? con->SourceObject() : con->DestinationObject();
			if (source_object && dest_object != nullptr) {
				//print_verbose("[doc] connection name: " + String(source_object->Name().c_str()) + ", dest: " + String(dest_object->Name().c_str()));
				const T *temp = dynamic_cast<const T *>(reverse ? source_object : dest_object);
				if (temp) {
					return_obj = temp;
				}
			}
		}

		if (return_obj != nullptr) {
			//print_verbose("[doc] returned valid element");
			//print_verbose("Found object for bone");
			return return_obj;
		}

		// safe to return nothing, need to use nullptr here as nullptr is used internally for FBX document.
		return nullptr;
	}

	// create fbx mesh function
	MeshInstance *create_fbx_mesh(Ref<FBXMeshVertexData> renderer_mesh_data, const Assimp::FBX::MeshGeometry *mesh_geometry, const Assimp::FBX::Model *model);

	Quat EulerToQuaternion(Assimp::FBX::Model::RotOrder mode, const Vector3 &rotation);

	Ref<Mesh> _generate_mesh_from_surface_indices(ImportState &state, const Vector<int> &p_surface_indices,
			const aiNode *assimp_node, Ref<Skin> &skin,
			Skeleton *&skeleton_assigned);

	Transform ComputePivotTransform(
			Transform chain[TransformationComp_MAXIMUM],
			Transform &geometric_transform) {

		// Maya pivots
		Transform T = chain[TransformationComp_Translation];
		Transform Roff = chain[TransformationComp_RotationOffset];
		Transform Rp = chain[TransformationComp_RotationPivot];
		Transform Rpre = chain[TransformationComp_PreRotation];
		Transform R = chain[TransformationComp_Rotation];
		Transform Rpost = chain[TransformationComp_PostRotation];
		Transform Soff = chain[TransformationComp_ScalingOffset];
		Transform Sp = chain[TransformationComp_ScalingPivot];
		Transform S = chain[TransformationComp_Scaling];

		// 3DS Max Pivots
		Transform OT = chain[TransformationComp_GeometricTranslation];
		Transform OR = chain[TransformationComp_GeometricRotation];
		Transform OS = chain[TransformationComp_GeometricScaling];

		// Calculate 3DS max pivot transform - use geometric space (e.g doesn't effect children nodes only the current node)
		geometric_transform = OT * OR * OS;
		// Calculate standard maya pivots
		return T * Roff * Rp * Rpre * R * Rpost.inverse() * Rp.inverse() * Soff * Sp * S * Sp.inverse();
		//return Sp.inverse() * S * Sp * Soff * Rp.inverse() * Rpost.inverse() * R * Rpre * Rp * Roff * T;
	}

	Transform GenFBXTransform(
			const Assimp::FBX::PropertyTable &props,
			const Assimp::FBX::Model::RotOrder &rot,
			const Assimp::FBX::TransformInheritance &inheritType,
			Transform &geometric_transform) {
		bool ok = false;

		Transform chain[TransformationComp_MAXIMUM];

		// Identity everything
		for (int x = 0; x < TransformationComp_MAXIMUM; x++) {
			chain[x] = Transform();
		}

		// Rotation matrix
		const Vector3 &PreRotation = Assimp::FBX::PropertyGet<Vector3>(props, "PreRotation", ok);
		if (ok) {
			chain[TransformationComp_PreRotation] = Transform(EulerToQuaternion(rot, PreRotation));
		}

		// Rotation matrix
		const Vector3 &PostRotation = Assimp::FBX::PropertyGet<Vector3>(props, "PostRotation", ok);
		if (ok) {
			chain[TransformationComp_PostRotation] = Transform(EulerToQuaternion(rot, PostRotation));
		}

		// Pivot translation
		const Vector3 &RotationPivot = Assimp::FBX::PropertyGet<Vector3>(props, "RotationPivot", ok);
		if (ok) {
			chain[TransformationComp_RotationPivot].origin = RotationPivot;
		}

		// Pivot translation
		const Vector3 &RotationOffset = Assimp::FBX::PropertyGet<Vector3>(props, "RotationOffset", ok);
		if (ok) {
			chain[TransformationComp_RotationOffset].origin = RotationOffset;
		}

		// Pivot translation
		const Vector3 &ScalingOffset = Assimp::FBX::PropertyGet<Vector3>(props, "ScalingOffset", ok);
		if (ok) {
			chain[TransformationComp_ScalingOffset].origin = ScalingOffset;
		}

		// Pivot translation
		const Vector3 &ScalingPivot = Assimp::FBX::PropertyGet<Vector3>(props, "ScalingPivot", ok);
		if (ok) {
			chain[TransformationComp_ScalingPivot].origin = ScalingPivot;
		}

		// Transform translation
		const Vector3 &Translation = Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Translation", ok);
		if (ok) {
			chain[TransformationComp_Translation].origin = Translation;
		}

		// Transform scaling
		const Vector3 &Scaling = Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Scaling", ok);
		if (ok) {
			chain[TransformationComp_Scaling].basis.scale(Scaling);
		}

		// Rotation matrix
		const Vector3 &Rotation = Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Rotation", ok);
		if (ok) {
			chain[TransformationComp_Rotation] = Transform(EulerToQuaternion(rot, Rotation));
		}

		// post node scaling
		const Vector3 &GeometricScaling = Assimp::FBX::PropertyGet<Vector3>(props, "GeometricScaling", ok);
		if (ok) {
			chain[TransformationComp_GeometricScaling].basis.scale(GeometricScaling);
		}

		// post node rotation
		const Vector3 &GeometricRotation = Assimp::FBX::PropertyGet<Vector3>(props, "GeometricRotation", ok);
		if (ok) {
			chain[TransformationComp_GeometricRotation] = Transform(EulerToQuaternion(rot, GeometricRotation));
		}

		// post node translation
		const Vector3 &GeometricTranslation = Assimp::FBX::PropertyGet<Vector3>(props, "GeometricTranslation", ok);
		if (ok) {
			chain[TransformationComp_GeometricTranslation].origin = GeometricTranslation;
		}

		// handler to do the math
		return ComputePivotTransform(chain, geometric_transform);
	}

	Transform GenFBXTransform(
			const Assimp::FBX::Model &model,
			Transform &geometric_transform) {
		return GenFBXTransform(
				model.Props(),
				model.RotationOrder(),
				model.InheritType(),
				geometric_transform);
	}

	String FBXNodeToName(const std::string &name) {
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

		return node_name.replace(":", "");
	}

	void CacheNodeInformation(Transform p_parent_transform, Ref<FBXBone> p_parent_bone,
			ImportState &state, const Assimp::FBX::Document *p_doc,
			uint64_t p_id);
	void BuildDocumentNodes(Transform parent_transform, ImportState &state, const Assimp::FBX::Document *doc, uint64_t id, Ref<FBXNode> fbx_parent);

	Spatial *_generate_scene(const String &p_path, const Assimp::FBX::Document *p_document,
			const uint32_t p_flags,
			int p_bake_fps, const int32_t p_max_bone_weights);

	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time, AssetImportAnimation::Interpolation p_interp);
	void _register_project_setting_import(const String generic, const String import_setting_string, const Vector<String> &exts, List<String> *r_extensions, const bool p_enabled) const;

	struct ImportFormat {
		Vector<String> extensions;
		bool is_default;
	};

protected:
	static void _bind_methods();

public:
	EditorSceneImporterFBX() {
	}
	~EditorSceneImporterFBX() {
	}

	virtual void get_extensions(List<String> *r_extensions) const;
	virtual uint32_t get_import_flags() const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = NULL);
	Ref<Image> load_image(ImportState &state, const aiScene *p_scene, String p_path);

    void
    GenFBXWeightInfo(const Ref<FBXMeshVertexData> &renderer_mesh_data, const Assimp::FBX::MeshGeometry *mesh_geometry,
                     Ref<SurfaceTool> &st, size_t vertex_id) const;
};

#endif // TOOLS_ENABLED
#endif // EDITOR_SCENE_FBX_IMPORTER_H
