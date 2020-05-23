#include "fbx_bone.h"
#include "fbx_node.h"
#include "fbx_skeleton.h"
#include "import_state.h"
#include "pivot_transform.h"

Ref<FBXNode> FBXBone::get_link(const ImportState &state) const {
	ERR_FAIL_COND_V_MSG(cluster == nullptr || cluster->TargetNode() == nullptr, Ref<FBXNode>(), "invalid link for bone");

	Ref<FBXNode> link_node;
	uint64_t id = cluster->TargetNode()->ID();
	if (state.fbx_target_map.has(id)) {
		link_node = state.fbx_target_map[id];
	} else {
		print_error("link node not found for " + itos(id));
	}

	// the node in space this is for, like if it's FOR a target.
	return link_node;
}

/* right now we just get single skin working and we can patch in the multiple tomorrow - per skin not per bone. */
// this will work for multiple meshes :) awesomeness.
// okay so these formula's are complex and need proper understanding of
// shear, pivots, geometric pivots, pre rotation and post rotation
// additionally DO NOT EDIT THIS if your blender file isn't working.
// Contact RevoluPowered Gordon MacPherson if you are contemplating making edits to this.
Transform FBXBone::get_vertex_skin_xform(const ImportState &state, Transform mesh_global_position) {
	print_verbose("get_vertex_skin_xform: " + bone_name);
	ERR_FAIL_COND_V_MSG(cluster == nullptr, Transform(), "[serious] unable to resolve the fbx cluster for this bone " + bone_name);
	// these methods will ONLY work for Maya.
	if (cluster->TransformAssociateModelValid()) {
		//print_error("additive skinning in use");
		Transform associate_global_init_position = cluster->TransformAssociateModel();
		Transform associate_global_current_position = Transform();
		Transform reference_global_init_position = cluster->GetTransform();
		Transform cluster_global_init_position = cluster->TransformLink();
		Ref<FBXNode> link_node = get_link(state);
		Transform cluster_global_current_position = link_node.is_valid() && link_node->pivot_transform.is_valid() ? link_node->pivot_transform->GlobalTransform : Transform();

		vertex_transform_matrix = reference_global_init_position.affine_inverse() * associate_global_init_position * associate_global_current_position.affine_inverse() *
								  cluster_global_current_position * cluster_global_init_position.affine_inverse() * reference_global_init_position;
	} else {
		//print_error("non additive skinning is in use");
		Transform reference_global_position = cluster->GetTransform();
		Transform reference_global_current_position = mesh_global_position;
		//Transform geometric_pivot = Transform(); // we do not use this - 3ds max only
		Transform global_init_position = cluster->TransformLink();
		Transform cluster_relative_init_position = global_init_position.affine_inverse() * reference_global_position;
		Transform cluster_relative_position_inverse = reference_global_current_position.affine_inverse() * global_init_position;
		vertex_transform_matrix = cluster_relative_position_inverse * cluster_relative_init_position;
	}

	return vertex_transform_matrix;
}