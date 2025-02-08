#include "flecs_entity_node.h"

void FlecsEntityNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_prefab", "prefab"), &FlecsEntityNode::set_prefab);
	ClassDB::bind_method(D_METHOD("get_prefab"), &FlecsEntityNode::get_prefab);
	ClassDB::bind_method(D_METHOD("set_entity", "entity"), &FlecsEntityNode::set_entity);
	ClassDB::bind_method(D_METHOD("get_entity"), &FlecsEntityNode::get_entity);
	ClassDB::bind_method(D_METHOD("set_owned_by_flecs", "owned_by_flecs"), &FlecsEntityNode::set_owned_by_flecs);
	ClassDB::bind_method(D_METHOD("get_owned_by_flecs"), &FlecsEntityNode::get_owned_by_flecs);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "prefab", PROPERTY_HINT_RESOURCE_TYPE, "FlecsPrefab"), "set_prefab", "get_prefab");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "entity", PROPERTY_HINT_RESOURCE_TYPE, "FlecsEntity"), "set_entity", "get_entity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "owned_by_flecs"), "set_owned_by_flecs", "get_owned_by_flecs");
}

void FlecsEntityNode::_notification(int p_what) {
	// Never run in editor
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	switch(p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if(prefab != nullptr && prefab.is_valid() && !owned_by_flecs) {

				if(!entity.is_valid()) {
					entity = prefab->instantiate();
				}
				entity->get_prefab()->initialize_entity_data(this);
			}

		}

		case NOTIFICATION_EXIT_TREE: {
			if(entity.is_valid() && owned_by_flecs) {
				entity->destroy();
			}
		}
	}
}

void FlecsEntityNode::set_prefab(Ref<FlecsPrefab> p_prefab) {
	prefab = p_prefab;
}

Ref<FlecsPrefab> FlecsEntityNode::get_prefab() const {
	return prefab;
}

void FlecsEntityNode::set_entity(Ref<FlecsEntity> p_entity) {
	entity = p_entity;

	if (owned_by_flecs) {
		entity->get_prefab()->initialize_entity_data(this);
	}
}

Ref<FlecsEntity> FlecsEntityNode::get_entity() const {
	return entity;
}

void FlecsEntityNode::set_owned_by_flecs(bool p_owned_by_flecs) {
	owned_by_flecs = p_owned_by_flecs;
}

bool FlecsEntityNode::get_owned_by_flecs() const {
	return owned_by_flecs;
}
