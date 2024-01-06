#ifndef LUA_REGISTER_TYPES_H
#define LUA_REGISTER_TYPES_H

#ifndef LAPI_GDEXTENSION
#include "modules/register_module_types.h"
#else
#include <godot_cpp/core/class_db.hpp>
using namespace godot;
#endif
void initialize_godot_luaAPI_module(ModuleInitializationLevel p_level);
void uninitialize_godot_luaAPI_module(ModuleInitializationLevel p_level);
#endif
