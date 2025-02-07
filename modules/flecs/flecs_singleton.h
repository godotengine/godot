// flecs_World.h
#ifndef FLECS_SINGLETON_H
#define FLECS_SINGLETON_H

#include "flecs_world.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "thirdparty/flecs.h"

class FlecsSingleton : public Object {
	GDCLASS(FlecsSingleton, Object)

protected:
	static void _bind_methods() {
	}


public:
	virtual void _register_singleton(FlecsWorld* flecs_world) {}
};

#endif // FLECS_SINGLETON_H
