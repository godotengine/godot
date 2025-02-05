#ifndef FLECS_ROOT_NODE_MODULE_H
#define FLECS_ROOT_NODE_MODULE_H


#include "../components/flecs_root_node_component.h"
#include "../flecs_mod.h"
#include "../flecs_world.h"
#include "core/object/class_db.h"
#include "core/object/object.h"


namespace modules {

struct FlecsRootNodeModule {
	FlecsRootNodeModule(flecs::world &world) {

		world.module<FlecsRootNodeModule>();
		world.component<components::FlecsRootNode>();
	}
};

} //namespace modules

class FlecsRootNodeMod : public FlecsMod {
	GDCLASS(FlecsRootNodeMod, FlecsMod);


protected:
	static void _bind_methods();

public:
	virtual void initialize(flecs::entity &prefab, flecs::world &world) override;
	virtual void initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) override;
};

#endif