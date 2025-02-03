// register_types.h
#ifndef FLECS_REGISTER_TYPES_H
#define FLECS_REGISTER_TYPES_H

#include "modules/register_module_types.h"

void initialize_flecs_module(ModuleInitializationLevel p_level);
void uninitialize_flecs_module(ModuleInitializationLevel p_level);

void _add_to_scene_tree();
#endif // FLECS_REGISTER_TYPES_H
