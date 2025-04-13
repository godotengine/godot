//
// Created by rfish on 4/12/2025.
//

#ifndef GODOT_FROM_SOURCE_COMPONENT_H
#define GODOT_FROM_SOURCE_COMPONENT_H


#include "core/object/object.h"
#include "core/io/resource.h"


class Component : public Resource {
	GDCLASS(Component, Resource);

protected:
	static void _bind_methods();


public:
	Component() = default;
};

#endif //GODOT_FROM_SOURCE_COMPONENT_H
