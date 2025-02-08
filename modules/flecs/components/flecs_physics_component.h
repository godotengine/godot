#ifndef FLECS_PHYSICS_COMPONENT_H
#define FLECS_PHYSICS_COMPONENT_H


#include "../flecs_component.h"
#include "core/object/class_db.h"
#include "core/object/object.h"

namespace components {

struct FlecsForce {
	Vector3 value{ Vector3() };
	Vector3 position_force{ Vector3() };
	Vector3 position{ Vector3() };
};

struct FlecsImpulse {
	Vector3 value{ Vector3() };
	Vector3 position_impulse{ Vector3() };
	Vector3 position{ Vector3() };
};

} // namespace components


class FlecsPhysicsComponent : public FlecsComponent {
	GDCLASS(FlecsPhysicsComponent, FlecsComponent);


protected:
	static void _bind_methods();

public:

	virtual void add_component(flecs::entity p_entity) override;
	virtual void remove_component(flecs::entity p_entity) override;
	virtual bool has_component(flecs::entity p_entity) const override;

	void apply_force(const Vector3 &p_force);
	void apply_impulse(const Vector3 &p_impulse);
	void apply_force_at_position(const Vector3 &p_force, const Vector3 &p_position);
	void apply_impulse_at_position(const Vector3 &p_impulse, const Vector3 &p_position);


	Vector3 get_force() const;
	Vector3 get_impulse() const;

	Vector3 get_force_at_position() const;
	Vector3 get_impulse_at_position() const;




};

#endif // FLECS_PHYSICS_COMPONENT_H