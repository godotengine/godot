#ifndef LUA_LIBRARIES_H
#define LUA_LIBRARIES_H

#include <lua/lua.hpp>

#ifndef LAPI_GDEXTENSION
#include "core/string/ustring.h"

#include "core/variant/variant.h"
#else
#include <godot_cpp/variant/string.hpp>

using namespace godot;
#endif

bool loadLuaLibrary(lua_State *L, String libraryName);

#endif