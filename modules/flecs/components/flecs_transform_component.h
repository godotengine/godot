#ifndef FLECS_TRANSFORM_COMPONENT_H
#define FLECS_TRANSFORM_COMPONENT_H

#include "../flecs_component.h"
#include "core/object/class_db.h"
#include "core/object/object.h"

namespace components {

struct FlecsLocation {
	Vector3 value{ Vector3() };
};

struct FlecsRotation {
	Quaternion value{ Quaternion() };
};

struct FlecsScale {
	Vector3 value{ Vector3(1, 1, 1) };
};

} // namespace components

class FlecsTransformComponent : public FlecsComponent {
	GDCLASS(FlecsTransformComponent, FlecsComponent);

protected:
	static void _bind_methods();

public:
	Vector3 get_location() const;
	Quaternion get_rotation() const;
	Vector3 get_scale() const;

	void set_location(Vector3 location);
	void set_rotation(Quaternion rotation);
	void set_scale(Vector3 scale);

	virtual void add_component(flecs::entity p_entity) override;
	virtual void remove_component(flecs::entity p_entity) override;
	virtual bool has_component(flecs::entity p_entity) const override;

};

#endif // FLECS_TRANSFORM_COMPONENT_H