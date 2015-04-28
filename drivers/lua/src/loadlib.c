/*
** $Id: loadlib.c,v 1.52.1.4 2009/09/09 13:17:16 roberto Exp $
** Dynamic library loader for Lua
** See Copyright Notice in lua.h
**
** This module contains an implementation of loadlib for Unix systems
** that have dlfcn, an implementation for Darwin (Mac OS X), an
** implementation for Windows, and a stub for other systems.
*/


#include <stdlib.h>
#include <string.h>


#define loadlib_c
#define LUA_LIB

#include "lua.h"

#include "lauxlib.h"
#include "lualib.h"


/* prefix for open functions in C libraries */
#define LUA_POF		"luaopen_"

/* separator for open functions in C libraries */
#define LUA_OFSEP	"_"


#define LIBPREFIX	"LOADLIB: "

#define POF		LUA_POF
#define LIB_FAIL	"open"


/* error codes for ll_loadfunc */
#define ERRLIB		1
#define ERRFUNC		2

#define setprogdir(L)		((void)0)


static void ll_unloadlib (void *lib);
static void *ll_load (lua_State *L, const char *path);
static lua_CFunction ll_sym (lua_State *L, void *lib, const char *sym);



#if defined(LUA_DL_DLOPEN)
/*
** {========================================================================
** This is an implementation of loadlib based on the dlfcn interface.
** The dlfcn interface is available in Linux, SunOS, Solaris, IRIX, FreeBSD,
** NetBSD, AIX 4.2, HPUX 11, and  probably most other Unix flavors, at least
** as an emulation layer on top of native functions.
** =========================================================================
*/

#include <dlfcn.h>

static void ll_unloadlib (void *lib) {
  dlclose(lib);
}


static void *ll_load (lua_State *L, const char *path) {
  void *lib = dlopen(path, RTLD_NOW);
  if (lib == NULL) lua_pushstring(L, dlerror());
  return lib;
}


static lua_CFunction ll_sym (lua_State *L, void *lib, const char *sym) {
  lua_CFunction f = (lua_CFunction)dlsym(lib, sym);
  if (f == NULL) lua_pushstring(L, dlerror());
  return f;
}

/* }====================================================== */



#elif defined(LUA_DL_DLL)
/*
** {======================================================================
** This is an implementation of loadlib for Windows using native functions.
** =======================================================================
*/

#include <windows.h>


#undef setprogdir

static void setprogdir (lua_State *L) {
  char buff[MAX_PATH + 1];
  char *lb;
  DWORD nsize = sizeof(buff)/sizeof(char);
  DWORD n = GetModuleFileNameA(NULL, buff, nsize);
  if (n == 0 || n == nsize || (lb = strrchr(buff, '\\')) == NULL)
    luaL_error(L, "unable to get ModuleFileName");
  else {
    *lb = '\0';
    luaL_gsub(L, lua_tostring(L, -1), LUA_EXECDIR, buff);
    lua_remove(L, -2);  /* remove original string */
  }
}


static void pusherror (lua_State *L) {
  int error = GetLastError();
  char buffer[128];
  if (FormatMessageA(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM,
      NULL, error, 0, buffer, sizeof(buffer), NULL))
    lua_pushstring(L, buffer);
  else
    lua_pushfstring(L, "system error %d\n", error);
}

static void ll_unloadlib (void *lib) {
  FreeLibrary((HINSTANCE)lib);
}


static void *ll_load (lua_State *L, const char *path) {
  HINSTANCE lib = LoadLibraryA(path);
  if (lib == NULL) pusherror(L);
  return lib;
}


static lua_CFunction ll_sym (lua_State *L, void *lib, const char *sym) {
  lua_CFunction f = (lua_CFunction)GetProcAddress((HINSTANCE)lib, sym);
  if (f == NULL) pusherror(L);
  return f;
}

/* }====================================================== */



#elif defined(LUA_DL_DYLD)
/*
** {======================================================================
** Native Mac OS X / Darwin Implementation
** =======================================================================
*/

#include <mach-o/dyld.h>


/* Mac appends a `_' before C function names */
#undef POF
#define POF	"_" LUA_POF


static void pusherror (lua_State *L) {
  const char *err_str;
  const char *err_file;
  NSLinkEditErrors err;
  int err_num;
  NSLinkEditError(&err, &err_num, &err_file, &err_str);
  lua_pushstring(L, err_str);
}


static const char *errorfromcode (NSObjectFileImageReturnCode ret) {
  switch (ret) {
    case NSObjectFileImageInappropriateFile:
      return "file is not a bundle";
    case NSObjectFileImageArch:
      return "library is for wrong CPU type";
    case NSObjectFileImageFormat:
      return "bad format";
    case NSObjectFileImageAccess:
      return "cannot access file";
    case NSObjectFileImageFailure:
    default:
      return "unable to load library";
  }
}


static void ll_unloadlib (void *lib) {
  NSUnLinkModule((NSModule)lib, NSUNLINKMODULE_OPTION_RESET_LAZY_REFERENCES);
}


static void *ll_load (lua_State *L, const char *path) {
  NSObjectFileImage img;
  NSObjectFileImageReturnCode ret;
  /* this would be a rare case, but prevents crashing if it happens */
  if(!_dyld_present()) {
    lua_pushliteral(L, "dyld not present");
    return NULL;
  }
  ret = NSCreateObjectFileImageFromFile(path, &img);
  if (ret == NSObjectFileImageSuccess) {
    NSModule mod = NSLinkModule(img, path, NSLINKMODULE_OPTION_PRIVATE |
                       NSLINKMODULE_OPTION_RETURN_ON_ERROR);
    NSDestroyObjectFileImage(img);
    if (mod == NULL) pusherror(L);
    return mod;
  }
  lua_pushstring(L, errorfromcode(ret));
  return NULL;
}


static lua_CFunction ll_sym (lua_State *L, void *lib, const char *sym) {
  NSSymbol nss = NSLookupSymbolInModule((NSModule)lib, sym);
  if (nss == NULL) {
    lua_pushfstring(L, "symbol " LUA_QS " not found", sym);
    return NULL;
  }
  return (lua_CFunction)NSAddressOfSymbol(nss);
}

/* }====================================================== */



#else
/*
** {======================================================
** Fallback for other systems
** =======================================================
*/

#undef LIB_FAIL
#define LIB_FAIL	"absent"


#define DLMSG	"dynamic libraries not enabled; check your Lua installation"


static void ll_unloadlib (void *lib) {
  (void)lib;  /* to avoid warnings */
}


static void *ll_load (lua_State *L, const char *path) {
  (void)path;  /* to avoid warnings */
  lua_pushliteral(L, DLMSG);
  return NULL;
}


static lua_CFunction ll_sym (lua_State *L, void *lib, const char *sym) {
  (void)lib; (void)sym;  /* to avoid warnings */
  lua_pushliteral(L, DLMSG);
  return NULL;
}

/* }====================================================== */
#endif



static void **ll_register (lua_State *L, const char *path) {
  void **plib;
  lua_pushfstring(L, "%s%s", LIBPREFIX, path);
  lua_gettable(L, LUA_REGISTRYINDEX);  /* check library in registry? */
  if (!lua_isnil(L, -1))  /* is there an entry? */
    plib = (void **)lua_touserdata(L, -1);
  else {  /* no entry yet; create one */
    lua_pop(L, 1);
    plib = (void **)lua_newuserdata(L, sizeof(const void *));
    *plib = NULL;
    luaL_getmetatable(L, "_LOADLIB");
    lua_setmetatable(L, -2);
    lua_pushfstring(L, "%s%s", LIBPREFIX, path);
    lua_pushvalue(L, -2);
    lua_settable(L, LUA_REGISTRYINDEX);
  }
  return plib;
}


/*
** __gc tag method: calls library's `ll_unloadlib' function with the lib
** handle
*/
static int gctm (lua_State *L) {
  void **lib = (void **)luaL_checkudata(L, 1, "_LOADLIB");
  if (*lib) ll_unloadlib(*lib);
  *lib = NULL;  /* mark library as closed */
  return 0;
}


static int ll_loadfunc (lua_State *L, const char *path, const char *sym) {
  void **reg = ll_register(L, path);
  if (*reg == NULL) *reg = ll_load(L, path);
  if (*reg == NULL)
    return ERRLIB;  /* unable to load library */
  else {
    lua_CFunction f = ll_sym(L, *reg, sym);
    if (f == NULL)
      return ERRFUNC;  /* unable to find function */
    lua_pushcfunction(L, f);
    return 0;  /* return function */
  }
}


static int ll_loadlib (lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  const char *init = luaL_checkstring(L, 2);
  int stat = ll_loadfunc(L, path, init);
  if (stat == 0)  /* no errors? */
    return 1;  /* return the loaded function */
  else {  /* error; error message is on stack top */
    lua_pushnil(L);
    lua_insert(L, -2);
    lua_pushstring(L, (stat == ERRLIB) ?  LIB_FAIL : "init");
    return 3;  /* return nil, error message, and where */
  }
}



/*
** {======================================================
** 'require' function
** =======================================================
*/


static int readable (const char *filename) {
  FILE *f = fopen(filename, "r");  /* try to open file */
  if (f == NULL) return 0;  /* open failed */
  fclose(f);
  return 1;
}


static const char *pushnexttemplate (lua_State *L, const char *path) {
  const char *l;
  while (*path == *LUA_PATHSEP) path++;  /* skip separators */
  if (*path == '\0') return NULL;  /* no more templates */
  l = strchr(path, *LUA_PATHSEP);  /* find next separator */
  if (l == NULL) l = path + strlen(path);
  lua_pushlstring(L, path, l - path);  /* template */
  return l;
}


static const char *findfile (lua_State *L, const char *name,
                                           const char *pname) {
  const char *path;
  name = luaL_gsub(L, name, ".", LUA_DIRSEP);
  lua_getfield(L, LUA_ENVIRONINDEX, pname);
  path = lua_tostring(L, -1);
  if (path == NULL)
    luaL_error(L, LUA_QL("package.%s") " must be a string", pname);
  lua_pushliteral(L, "");  /* error accumulator */
  while ((path = pushnexttemplate(L, path)) != NULL) {
    const char *filename;
    filename = luaL_gsub(L, lua_tostring(L, -1), LUA_PATH_MARK, name);
    lua_remove(L, -2);  /* remove path template */
    if (readable(filename))  /* does file exist and is readable? */
      return filename;  /* return that file name */
    lua_pushfstring(L, "\n\tno file " LUA_QS, filename);
    lua_remove(L, -2);  /* remove file name */
    lua_concat(L, 2);  /* add entry to possible error message */
  }
  return NULL;  /* not found */
}


static void loaderror (lua_State *L, const char *filename) {
  luaL_error(L, "error loading module " LUA_QS " from file " LUA_QS ":\n\t%s",
                lua_tostring(L, 1), filename, lua_tostring(L, -1));
}


static int loader_Lua (lua_State *L) {
  const char *filename;
  const char *name = luaL_checkstring(L, 1);
  filename = findfile(L, name, "path");
  if (filename == NULL) return 1;  /* library not found in this path */
  if (luaL_loadfile(L, filename) != 0)
    loaderror(L, filename);
  return 1;  /* library loaded successfully */
}


static const char *mkfuncname (lua_State *L, const char *modname) {
  const char *funcname;
  const char *mark = strchr(modname, *LUA_IGMARK);
  if (mark) modname = mark + 1;
  funcname = luaL_gsub(L, modname, ".", LUA_OFSEP);
  funcname = lua_pushfstring(L, POF"%s", funcname);
  lua_remove(L, -2);  /* remove 'gsub' result */
  return funcname;
}


static int loader_C (lua_State *L) {
  const char *funcname;
  const char *name = luaL_checkstring(L, 1);
  const char *filename = findfile(L, name, "cpath");
  if (filename == NULL) return 1;  /* library not found in this path */
  funcname = mkfuncname(L, name);
  if (ll_loadfunc(L, filename, funcname) != 0)
    loaderror(L, filename);
  return 1;  /* library loaded successfully */
}


static int loader_Croot (lua_State *L) {
  const char *funcname;
  const char *filename;
  const char *name = luaL_checkstring(L, 1);
  const char *p = strchr(name, '.');
  int stat;
  if (p == NULL) return 0;  /* is root */
  lua_pushlstring(L, name, p - name);
  filename = findfile(L, lua_tostring(L, -1), "cpath");
  if (filename == NULL) return 1;  /* root not found */
  funcname = mkfuncname(L, name);
  if ((stat = ll_loadfunc(L, filename, funcname)) != 0) {
    if (stat != ERRFUNC) loaderror(L, filename);  /* real error */
    lua_pushfstring(L, "\n\tno module " LUA_QS " in file " LUA_QS,
                       name, filename);
    return 1;  /* function not found */
  }
  return 1;
}


static int loader_preload (lua_State *L) {
  const char *name = luaL_checkstring(L, 1);
  lua_getfield(L, LUA_ENVIRONINDEX, "preload");
  if (!lua_istable(L, -1))
    luaL_error(L, LUA_QL("package.preload") " must be a table");
  lua_getfield(L, -1, name);
  if (lua_isnil(L, -1))  /* not found? */
    lua_pushfstring(L, "\n\tno field package.preload['%s']", name);
  return 1;
}


static const int sentinel_ = 0;
#define sentinel	((void *)&sentinel_)


static int ll_require (lua_State *L) {
  const char *name = luaL_checkstring(L, 1);
  int i;
  lua_settop(L, 1);  /* _LOADED table will be at index 2 */
  lua_getfield(L, LUA_REGISTRYINDEX, "_LOADED");
  lua_getfield(L, 2, name);
  if (lua_toboolean(L, -1)) {  /* is it there? */
    if (lua_touserdata(L, -1) == sentinel)  /* check loops */
      luaL_error(L, "loop or previous error loading module " LUA_QS, name);
    return 1;  /* package is already loaded */
  }
  /* else must load it; iterate over available loaders */
  lua_getfield(L, LUA_ENVIRONINDEX, "loaders");
  if (!lua_istable(L, -1))
    luaL_error(L, LUA_QL("package.loaders") " must be a table");
  lua_pushliteral(L, "");  /* error message accumulator */
  for (i=1; ; i++) {
    lua_rawgeti(L, -2, i);  /* get a loader */
    if (lua_isnil(L, -1))
      luaL_error(L, "module " LUA_QS " not found:%s",
                    name, lua_tostring(L, -2));
    lua_pushstring(L, name);
    lua_call(L, 1, 1);  /* call it */
    if (lua_isfunction(L, -1))  /* did it find module? */
      break;  /* module loaded successfully */
    else if (lua_isstring(L, -1))  /* loader returned error message? */
      lua_concat(L, 2);  /* accumulate it */
    else
      lua_pop(L, 1);
  }
  lua_pushlightuserdata(L, sentinel);
  lua_setfield(L, 2, name);  /* _LOADED[name] = sentinel */
  lua_pushstring(L, name);  /* pass name as argument to module */
  lua_call(L, 1, 1);  /* run loaded module */
  if (!lua_isnil(L, -1))  /* non-nil return? */
    lua_setfield(L, 2, name);  /* _LOADED[name] = returned value */
  lua_getfield(L, 2, name);
  if (lua_touserdata(L, -1) == sentinel) {   /* module did not set a value? */
    lua_pushboolean(L, 1);  /* use true as result */
    lua_pushvalue(L, -1);  /* extra copy to be returned */
    lua_setfield(L, 2, name);  /* _LOADED[name] = true */
  }
  return 1;
}

/* }====================================================== */



/*
** {======================================================
** 'module' function
** =======================================================
*/
  

static void setfenv (lua_State *L) {
  lua_Debug ar;
  if (lua_getstack(L, 1, &ar) == 0 ||
      lua_getinfo(L, "f", &ar) == 0 ||  /* get calling function */
      lua_iscfunction(L, -1))
    luaL_error(L, LUA_QL("module") " not called from a Lua function");
  lua_pushvalue(L, -2);
  lua_setfenv(L, -2);
  lua_pop(L, 1);
}


static void dooptions (lua_State *L, int n) {
  int i;
  for (i = 2; i <= n; i++) {
    lua_pushvalue(L, i);  /* get option (a function) */
    lua_pushvalue(L, -2);  /* module */
    lua_call(L, 1, 0);
  }
}


static void modinit (lua_State *L, const char *modname) {
  const char *dot;
  lua_pushvalue(L, -1);
  lua_setfield(L, -2, "_M");  /* module._M = module */
  lua_pushstring(L, modname);
  lua_setfield(L, -2, "_NAME");
  dot = strrchr(modname, '.');  /* look for last dot in module name */
  if (dot == NULL) dot = modname;
  else dot++;
  /* set _PACKAGE as package name (full module name minus last part) */
  lua_pushlstring(L, modname, dot - modname);
  lua_setfield(L, -2, "_PACKAGE");
}


static int ll_module (lua_State *L) {
  const char *modname = luaL_checkstring(L, 1);
  int loaded = lua_gettop(L) + 1;  /* index of _LOADED table */
  lua_getfield(L, LUA_REGISTRYINDEX, "_LOADED");
  lua_getfield(L, loaded, modname);  /* get _LOADED[modname] */
  if (!lua_istable(L, -1)) {  /* not found? */
    lua_pop(L, 1);  /* remove previous result */
    /* try global variable (and create one if it does not exist) */
    if (luaL_findtable(L, LUA_GLOBALSINDEX, modname, 1) != NULL)
      return luaL_error(L, "name conflict for module " LUA_QS, modname);
    lua_pushvalue(L, -1);
    lua_setfield(L, loaded, modname);  /* _LOADED[modname] = new table */
  }
  /* check whether table already has a _NAME field */
  lua_getfield(L, -1, "_NAME");
  if (!lua_isnil(L, -1))  /* is table an initialized module? */
    lua_pop(L, 1);
  else {  /* no; initialize it */
    lua_pop(L, 1);
    modinit(L, modname);
  }
  lua_pushvalue(L, -1);
  setfenv(L);
  dooptions(L, loaded - 1);
  return 0;
}


static int ll_seeall (lua_State *L) {
  luaL_checktype(L, 1, LUA_TTABLE);
  if (!lua_getmetatable(L, 1)) {
    lua_createtable(L, 0, 1); /* create new metatable */
    lua_pushvalue(L, -1);
    lua_setmetatable(L, 1);
  }
  lua_pushvalue(L, LUA_GLOBALSINDEX);
  lua_setfield(L, -2, "__index");  /* mt.__index = _G */
  return 0;
}


/* }====================================================== */



/* auxiliary mark (for internal use) */
#define AUXMARK		"\1"

static void setpath (lua_State *L, const char *fieldname, const char *envname,
                                   const char *def) {
  const char *path = getenv(envname);
  if (path == NULL)  /* no environment variable? */
    lua_pushstring(L, def);  /* use default */
  else {
    /* replace ";;" by ";AUXMARK;" and then AUXMARK by default path */
    path = luaL_gsub(L, path, LUA_PATHSEP LUA_PATHSEP,
                              LUA_PATHSEP AUXMARK LUA_PATHSEP);
    luaL_gsub(L, path, AUXMARK, def);
    lua_remove(L, -2);
  }
  setprogdir(L);
  lua_setfield(L, -2, fieldname);
}


static const luaL_Reg pk_funcs[] = {
  {"loadlib", ll_loadlib},
  {"seeall", ll_seeall},
  {NULL, NULL}
};


static const luaL_Reg ll_funcs[] = {
  {"module", ll_module},
  {"require", ll_require},
  {NULL, NULL}
};


static const lua_CFunction loaders[] =
  {loader_preload, loader_Lua, loader_C, loader_Croot, NULL};


LUALIB_API int luaopen_package (lua_State *L) {
  int i;
  /* create new type _LOADLIB */
  luaL_newmetatable(L, "_LOADLIB");
  lua_pushcfunction(L, gctm);
  lua_setfield(L, -2, "__gc");
  /* create `package' table */
  luaL_register(L, LUA_LOADLIBNAME, pk_funcs);
#if defined(LUA_COMPAT_LOADLIB) 
  lua_getfield(L, -1, "loadlib");
  lua_setfield(L, LUA_GLOBALSINDEX, "loadlib");
#endif
  lua_pushvalue(L, -1);
  lua_replace(L, LUA_ENVIRONINDEX);
  /* create `loaders' table */
  lua_createtable(L, sizeof(loaders)/sizeof(loaders[0]) - 1, 0);
  /* fill it with pre-defined loaders */
  for (i=0; loaders[i] != NULL; i++) {
    lua_pushcfunction(L, loaders[i]);
    lua_rawseti(L, -2, i+1);
  }
  lua_setfield(L, -2, "loaders");  /* put it in field `loaders' */
  setpath(L, "path", LUA_PATH, LUA_PATH_DEFAULT);  /* set field `path' */
  setpath(L, "cpath", LUA_CPATH, LUA_CPATH_DEFAULT); /* set field `cpath' */
  /* store config information */
  lua_pushliteral(L, LUA_DIRSEP "\n" LUA_PATHSEP "\n" LUA_PATH_MARK "\n"
                     LUA_EXECDIR "\n" LUA_IGMARK);
  lua_setfield(L, -2, "config");
  /* set field `loaded' */
  luaL_findtable(L, LUA_REGISTRYINDEX, "_LOADED", 2);
  lua_setfield(L, -2, "loaded");
  /* set field `preload' */
  lua_newtable(L);
  lua_setfield(L, -2, "preload");
  lua_pushvalue(L, LUA_GLOBALSINDEX);
  luaL_register(L, NULL, ll_funcs);  /* open lib into global table */
  lua_pop(L, 1);
  return 1;  /* return 'package' table */
}

