#ifndef FLECS_ENTITY_NODE_H
#define FLECS_ENTITY_NODE_H


#include "core/object/class_db.h"
#include "flecs_world.h"
#include "scene/main/node.h"
#include "flecs_prefab.h"
#include "flecs_entity.h"

class FlecsEntityNode : public Node {
	GDCLASS(FlecsEntityNode, Node);


private:
	Ref<FlecsPrefab> prefab = nullptr;
	Ref<FlecsEntity> entity = nullptr;
	bool owned_by_flecs = true;


protected:
	static void _bind_methods();
	void _notification(int p_what);

public:

	void set_prefab(Ref<FlecsPrefab> p_prefab);
	Ref<FlecsPrefab> get_prefab() const;

	void set_entity(Ref<FlecsEntity> p_entity);
	Ref<FlecsEntity> get_entity() const;

	void set_owned_by_flecs(bool p_owned_by_flecs);
	bool get_owned_by_flecs() const;

};

#endif // FLECS_ENTITY_NODE_H