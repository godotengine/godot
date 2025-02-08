#ifndef FLECS_ROOT_NODE_COMPONENT_H
#define FLECS_ROOT_NODE_COMPONENT_H


#include "../flecs_component.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "modules/jolt_physics/objects/jolt_body_3d.h"

namespace components {

struct FlecsRootNode {
	Node3D *value{ nullptr };
	RID physics_body_rid = RID();
	JoltBody3D * jolt_body{ nullptr };
};


struct FlecsRootNodeToGodotSyncTag{};
struct GodotToFlecsRootNodeSyncTag{};
struct HasPhysicsBody3DTag{};


} // namespace components

class FlecsRootNodeComponent : public FlecsComponent {
	GDCLASS(FlecsRootNodeComponent, FlecsComponent);


protected:
	static void _bind_methods();

public:
	Node3D *get_root_node() const;
	void set_root_node(Node3D *p_root_node);

	virtual void add_component(flecs::entity p_entity) override;
	virtual void remove_component(flecs::entity p_entity) override;
	virtual bool has_component(flecs::entity p_entity) const override;

};

#endif // FLECS_ROOT_NODE_COMPONENT_H