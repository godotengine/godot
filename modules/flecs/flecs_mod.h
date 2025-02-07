#ifndef FLECS_MOD_H
#define FLECS_MOD_H

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "flecs_world.h"
#include "flecs_entity.h"

class FlecsEntityNode;

class FlecsMod : public Resource {
	GDCLASS(FlecsMod, Resource);


protected:
	static void _bind_methods();

public:

	virtual TypedArray<FlecsMod> get_required_modules() const { return TypedArray<FlecsMod>(); }
	

	virtual void initialize(flecs::entity &prefab, flecs::world &world) = 0;

	virtual void post_instantiate(Ref<FlecsEntity> entity, flecs::world &world) {};

	virtual void initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) {};
};

#endif // FLECS_MOD_H