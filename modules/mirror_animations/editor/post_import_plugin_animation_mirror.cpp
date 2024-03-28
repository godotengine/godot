#ifdef TOOLS_ENABLED

#include "post_import_plugin_animation_mirror.h"

#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"

void PostImportPluginAnimationMirror::get_internal_import_options(InternalImportCategory p_category, List<ResourceImporter::ImportOption> *r_options) {
	EditorScenePostImportPlugin::get_internal_import_options(p_category, r_options);
}

void PostImportPluginAnimationMirror::internal_process(InternalImportCategory p_category, Node *p_base_scene, Node *p_node, Ref<Resource> p_resource, const Dictionary &p_options) {
	EditorScenePostImportPlugin::internal_process(p_category, p_base_scene, p_node, p_resource, p_options);

	if (p_category == INTERNAL_IMPORT_CATEGORY_ANIMATION) {
		if (!static_cast<bool>(p_options["settings/mirror"])) {
			return;
		}

		Animation *animation = Object::cast_to<Animation>(p_resource.ptr());
		if (animation == nullptr) {
			return;
		}

		AnimationPlayer *animation_player = nullptr;

		TypedArray<Node> nodes = p_base_scene->find_children("*", "AnimationPlayer");
		while (nodes.size()) {
			AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(nodes.pop_back());
			if (!ap->has_animation(animation->get_name())) {
				continue;
			}

			// sanity check for same animation name in different players
			if (ap->get_animation(animation->get_name()) != animation) {
				continue;
			}

			animation_player = ap;
			break;
		}

		const int track_count = animation->get_track_count();
		for (int i = 0; i < track_count; i++) {
			mirror_node_path(animation, animation_player, i);

			switch (animation->track_get_type(i)) {
				case Animation::TYPE_POSITION_3D:
					mirror_position_track(animation, i);
					break;
				case Animation::TYPE_ROTATION_3D:
					mirror_rotation_track(animation, i);
					break;
			}
		}
	}
}

PostImportPluginAnimationMirror::PostImportPluginAnimationMirror() {
}

void PostImportPluginAnimationMirror::mirror_node_path(Animation *p_animation, const AnimationPlayer *p_animation_player, const int &p_track_index) const {
	const auto node_path = p_animation->track_get_path(p_track_index);
	if (node_path.get_subname_count() <= 0) {
		return;
	}

	Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(p_animation_player->get_node(p_animation_player->get_root_node())->get_node(node_path));
	ERR_FAIL_COND_MSG(skeleton == nullptr, vformat("Can't mirror animation \"%s\" - no skeleton found for track path \"%s\"", p_animation->get_name(), node_path));

	const auto &bone_name = node_path.get_subname(node_path.get_subname_count() - 1);
	const auto &bone_index = skeleton->find_bone(bone_name);

	if (bone_index < 0) {
		return;
	}

	const auto &bone_counterpart_name = skeleton->get_bone_counterpart_name(bone_index);
	if (bone_counterpart_name == StringName()) {
		return;
	}

	auto sub_names = node_path.get_subnames();
	sub_names.set(sub_names.size() - 1, bone_counterpart_name);

	const auto mirror_path = NodePath(node_path.get_names(), sub_names, node_path.is_absolute());
	p_animation->track_set_path(p_track_index, mirror_path);
}

void PostImportPluginAnimationMirror::mirror_position_track(Animation *p_animation, const int &p_track_index) {
	const auto scale = Vector3(-1, 1, 1);

	for (int i = 0; i < p_animation->track_get_key_count(p_track_index); i++) {
		const auto value = p_animation->track_get_key_value(p_track_index, i);

		if (value.get_type() != Variant::VECTOR3) {
			continue;
		}

		const auto newValue = static_cast<Vector3>(value) * scale;
		p_animation->track_set_key_value(p_track_index, i, newValue);
	}
}

void PostImportPluginAnimationMirror::mirror_rotation_track(Animation *p_animation, const int &p_track_index) {
	for (int i = 0; i < p_animation->track_get_key_count(p_track_index); i++) {
		const auto value = p_animation->track_get_key_value(p_track_index, i);

		if (value.get_type() != Variant::QUATERNION) {
			continue;
		}

		const auto current_rotation = static_cast<Quaternion>(value);
		const auto new_rotation = Quaternion(-current_rotation.x, current_rotation.y, current_rotation.z, -current_rotation.w);

		p_animation->track_set_key_value(p_track_index, i, new_rotation);
	}
}

#endif // TOOLS_ENABLED
