#ifndef LAPI_LUA_HPP
#define LAPI_LUA_HPP

#if defined(LAPI_LUAJIT)

#include <luaJIT_riscv/src/lua.hpp>

#elif defined(LAPI_51)
extern "C" {
#include <lua51/lauxlib.h>
#include <lua51/lua.h>
#include <lua51/lualib.h>
}

#else // LUAJIT ONLY

extern "C" {
#include <lua/lauxlib.h>
#include <lua/lua.h>
#include <lua/lualib.h>
}
#endif

#endif
