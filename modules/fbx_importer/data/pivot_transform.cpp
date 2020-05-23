/*************************************************************************/
/*  pivot_transform.cpp 	                                             */
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

#include "pivot_transform.h"
#include <modules/fbx_importer/tools/import_utils.h>

void PivotTransform::ReadTransformChain() {
	const Assimp::FBX::PropertyTable &props = fbx_model->Props();
	const Assimp::FBX::Model::RotOrder &rot = fbx_model->RotationOrder();
	const Assimp::FBX::TransformInheritance &inheritType = fbx_model->InheritType();
	inherit_type = inheritType; // copy the inherit type we need it in the second step.
	print_verbose("Model: " + String(fbx_model->Name().c_str()) + " Has inherit type: " + itos(fbx_model->InheritType()));
	bool ok = false;
	raw_pre_rotation = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "PreRotation", ok));
	if (ok) {
		pre_rotation = AssimpUtils::EulerToQuaternion(Assimp::FBX::Model::RotOrder_EulerXYZ, raw_pre_rotation);
		print_verbose("valid pre_rotation: " + raw_pre_rotation + " euler conversion: " + (pre_rotation.get_euler() * (180 / Math_PI)));
	}
	raw_post_rotation = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "PostRotation", ok));
	if (ok) {
		post_rotation = AssimpUtils::EulerToQuaternion(Assimp::FBX::Model::RotOrder_EulerXYZ, raw_post_rotation);
		print_verbose("valid post_rotation: " + raw_post_rotation + " euler conversion: " + (pre_rotation.get_euler() * (180 / Math_PI)));
	}
	const Vector3 &RotationPivot = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "RotationPivot", ok));
	if (ok) {
		rotation_pivot = AssimpUtils::FixAxisConversions(RotationPivot);
	}
	const Vector3 &RotationOffset = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "RotationOffset", ok));
	if (ok) {
		rotation_offset = AssimpUtils::FixAxisConversions(RotationOffset);
	}
	const Vector3 &ScalingOffset = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "ScalingOffset", ok));
	if (ok) {
		scaling_offset = AssimpUtils::FixAxisConversions(ScalingOffset);
	}
	const Vector3 &ScalingPivot = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "ScalingPivot", ok));
	if (ok) {
		scaling_pivot = AssimpUtils::FixAxisConversions(ScalingPivot);
	}
	const Vector3 &Translation = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Translation", ok));
	if (ok) {
		translation = AssimpUtils::FixAxisConversions(Translation);
	}
	raw_rotation = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Rotation", ok));
	if (ok) {
		rotation = AssimpUtils::EulerToQuaternion(rot, raw_rotation);
	}
	const Vector3 &Scaling = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Scaling", ok));
	if (ok) {
		scaling = Scaling;
	}
	const Vector3 &GeometricScaling = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "GeometricScaling", ok));
	if (ok) {
		geometric_scaling = GeometricScaling;
	}
	const Vector3 &GeometricRotation = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "GeometricRotation", ok));
	if (ok) {
		geometric_rotation = AssimpUtils::EulerToQuaternion(rot, GeometricRotation);
	}
	const Vector3 &GeometricTranslation = safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "GeometricTranslation", ok));
	if (ok) {
		geometric_translation = AssimpUtils::FixAxisConversions(GeometricTranslation);
	}
}
void PivotTransform::ComputePivotTransform() {
	// in maya we use geometric_rotation too
	// print_verbose("pre_rotation : " + (pre_rotation.get_euler() * (180 / Math_PI)));
	// print_verbose("post_rotation : " + (post_rotation.get_euler() * (180 / Math_PI)));
	// print_verbose("rotation : " + (rotation.get_euler() * (180 / Math_PI)));
	// print_verbose("geometric_rotation : " + (geometric_rotation.get_euler() * (180 / Math_PI)));

	Transform T, Roff, Rp, Soff, Sp, S;

	// Here I assume this is the operation which needs done.
	// Its WorldTransform * V

	// Origin pivots
	T.set_origin(translation);
	Roff.set_origin(rotation_offset);
	Rp.set_origin(rotation_pivot);
	Soff.set_origin(scaling_offset);
	Sp.set_origin(scaling_pivot);

	// Scaling node
	S.scale(scaling);
	Local_Scaling_Matrix = S; // copy for when node / child is looking for the value of this.

	// Rotation pivots
	Transform Rpre = Transform(pre_rotation);
	Transform R = Transform(rotation);
	Transform Rpost = Transform(post_rotation);

	Transform parent_global_xform;
	Transform parent_local_scaling_m;

	if (parent_transform.is_valid()) {
		parent_global_xform = parent_transform->GlobalTransform;
		parent_local_scaling_m = parent_transform->Local_Scaling_Matrix;
	}

	Transform local_rotation_m, parent_global_rotation_m;
	Quat parent_global_rotation = parent_global_xform.basis.get_rotation_quat();
	parent_global_rotation_m.basis.set_quat(parent_global_rotation);
	local_rotation_m = Rpre * R * Rpost;

	//Basis parent_global_rotation = Basis(parent_global_xform.get_basis().get_rotation_quat().normalized());

	Transform local_shear_scaling, parent_shear_scaling, parent_shear_rotation, parent_shear_translation;
	Vector3 parent_translation = parent_global_xform.get_origin();
	parent_shear_translation.origin = parent_translation;
	parent_shear_rotation = parent_shear_translation.inverse() * parent_global_xform;
	parent_shear_scaling = parent_global_rotation_m.inverse() * parent_shear_rotation;
	local_shear_scaling = S;

	// Inherit type handler - we don't care about T here, just reordering RSrs etc.
	Transform global_rotation_scale;
	if (inherit_type == Assimp::FBX::Transform_RrSs) {
		global_rotation_scale = parent_global_rotation_m * local_rotation_m * parent_shear_scaling * local_shear_scaling;
	} else if (inherit_type == Assimp::FBX::Transform_RSrs) {
		global_rotation_scale = parent_global_rotation_m * parent_shear_scaling * local_rotation_m * local_shear_scaling;
	} else if (inherit_type == Assimp::FBX::Transform_Rrs) {
		Transform parent_global_shear_m_noLocal = parent_shear_scaling * parent_local_scaling_m.inverse();
		global_rotation_scale = parent_global_rotation_m * local_rotation_m * parent_global_shear_m_noLocal * local_shear_scaling;
	}

	LocalTransform = T * Roff * Rp * Rpre * R * Rpost.inverse() * Rp.inverse() * Soff * Sp * S * Sp.inverse();

	Vector3 local_translation_pivoted = LocalTransform.origin;
	Vector3 global_translation_pivoted = parent_global_xform.xform(local_translation_pivoted);
	GlobalTransform = Transform();
	GlobalTransform.origin = global_translation_pivoted;

	GlobalTransform = GlobalTransform * global_rotation_scale;

	AssimpUtils::debug_xform("local xform calculation", LocalTransform);
	print_verbose("scale of node: " + S.basis.get_scale_local());
	print_verbose("---------------------------------------------------------------");
}
void PivotTransform::Execute() {
	ReadTransformChain();
	ComputePivotTransform();

	AssimpUtils::debug_xform("global xform: ", GlobalTransform);
	computed_global_xform = true;
}
