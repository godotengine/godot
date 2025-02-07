#include "flecs_packed_scene_module.h"
#include "../flecs_entity_node.h"
#include "scene/gui/tree.h"
#include "flecs_root_node_module.h"

void FlecsPackedSceneMod::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_scene"), &FlecsPackedSceneMod::get_scene);
	ClassDB::bind_method(D_METHOD("set_scene"), &FlecsPackedSceneMod::set_scene);
	ClassDB::bind_method(D_METHOD("is_auto_initialize"), &FlecsPackedSceneMod::is_auto_initialize);
	ClassDB::bind_method(D_METHOD("set_auto_initialize"), &FlecsPackedSceneMod::set_auto_initialize);
	ClassDB::bind_method(D_METHOD("is_add_to_root"), &FlecsPackedSceneMod::is_add_to_root);
	ClassDB::bind_method(D_METHOD("set_add_to_root"), &FlecsPackedSceneMod::set_add_to_root);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "packed_scene", PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_scene", "get_scene");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_initialize"), "set_auto_initialize", "is_auto_initialize");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "add_to_root"), "set_add_to_root", "is_add_to_root");
}

TypedArray<FlecsMod> FlecsPackedSceneMod::get_required_modules() const {
	TypedArray<FlecsMod> deps;
	deps.append(memnew(FlecsRootNodeMod)); 
	return deps;
}


void FlecsPackedSceneMod::initialize(flecs::entity &prefab, flecs::world &world) {
	world.import <modules::FlecsPackedSceneModule>();

	prefab.set<components::FlecsPackedScene>({ packed_scene });
}


void FlecsPackedSceneMod::post_instantiate(Ref<FlecsEntity> entity, flecs::world &world) {
	if (auto_initialize && add_to_root) {
		if (packed_scene.is_valid()) {
			Node *node = packed_scene->instantiate();
			Node *root = FlecsWorld::get_singleton()->get_tree()->get_current_scene();
			root->add_child(node);
			node->set_owner(root);

			FlecsEntityNode *entity_node = memnew(FlecsEntityNode);
			node->add_child(entity_node);
			entity_node->set_owner(root);
			entity_node->set_owned_by_flecs(true);
			entity_node->set_entity(entity);
		}
	}
}

void FlecsPackedSceneMod::initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) {
	if (auto_initialize && !add_to_root) {
		const components::FlecsPackedScene *packed_scene_to_instantiate = entity->get_entity()->get_entity().get<components::FlecsPackedScene>();
		if (packed_scene_to_instantiate && packed_scene_to_instantiate->value.is_valid()) {
			Node *node = packed_scene_to_instantiate->value->instantiate();
			entity->add_child(node);
			node->set_owner(entity);
		}
	}
}

Ref<PackedScene> FlecsPackedSceneMod::get_scene() const {
	return packed_scene;
}

void FlecsPackedSceneMod::set_scene(Ref<PackedScene> p_scene) {
	packed_scene = p_scene;
}

bool FlecsPackedSceneMod::is_auto_initialize() const {
	return auto_initialize;
}

void FlecsPackedSceneMod::set_auto_initialize(bool p_auto_initialize) {
	auto_initialize = p_auto_initialize;
}

bool FlecsPackedSceneMod::is_add_to_root() const {
	return add_to_root;
}

void FlecsPackedSceneMod::set_add_to_root(bool p_add_to_root) {
	add_to_root = p_add_to_root;
}
