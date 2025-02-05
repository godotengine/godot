// register_types.cpp
#include "register_types.h"

#include "core/object/class_db.h"
#include "flecs_singleton.h"
#include "flecs_world.h"
#include "flecs_entity.h"
#include "flecs_component.h"
#include "flecs_mod.h"
#include "flecs_prefab.h"
#include "modules/flecs_transform_module.h"
#include "components/flecs_transform_component.h"
#include "flecs_entity_node.h"
#include "modules/flecs_root_node_module.h"
#include "components/flecs_root_node_component.h"

void initialize_flecs_module(ModuleInitializationLevel p_level) {

	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

	ClassDB::register_class<FlecsSingleton>();
	ClassDB::register_class<FlecsWorld>();
	ClassDB::register_class<FlecsEntity>();
	ClassDB::register_abstract_class<FlecsComponent>();
	ClassDB::register_abstract_class<FlecsMod>();
	ClassDB::register_class<FlecsPrefab>();
	ClassDB::register_class<FlecsEntityNode>();
	ClassDB::register_class<FlecsTransformMod>();
	ClassDB::register_class<FlecsTransformComponent>();
	ClassDB::register_class<FlecsRootNodeMod>();
	ClassDB::register_class<FlecsRootNodeComponent>();

	FlecsComponent::generate_component_enum();

}

void uninitialize_flecs_module(ModuleInitializationLevel p_level) {

	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}
