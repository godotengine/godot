/*
* The code below can be used to make a Lua core that does not contain the
* parsing modules (lcode, llex, lparser), which represent 35% of the total core.
* You'll only be able to load binary files and strings, precompiled with luac.
* (Of course, you'll have to build luac with the original parsing modules!)
*
* To use this module, simply compile it ("make noparser" does that) and list
* its object file before the Lua libraries. The linker should then not load
* the parsing modules. To try it, do "make luab".
*
* If you also want to avoid the dump module (ldump.o), define NODUMP.
* #define NODUMP
*/

#define LUA_CORE

#include "llex.h"
#include "lparser.h"
#include "lzio.h"

LUAI_FUNC void luaX_init (lua_State *L) {
  UNUSED(L);
}

LUAI_FUNC Proto *luaY_parser (lua_State *L, ZIO *z, Mbuffer *buff, const char *name) {
  UNUSED(z);
  UNUSED(buff);
  UNUSED(name);
  lua_pushliteral(L,"parser not loaded");
  lua_error(L);
  return NULL;
}

#ifdef NODUMP
#include "lundump.h"

LUAI_FUNC int luaU_dump (lua_State* L, const Proto* f, lua_Writer w, void* data, int strip) {
  UNUSED(f);
  UNUSED(w);
  UNUSED(data);
  UNUSED(strip);
#if 1
  UNUSED(L);
  return 0;
#else
  lua_pushliteral(L,"dumper not loaded");
  lua_error(L);
#endif
}
#endif
