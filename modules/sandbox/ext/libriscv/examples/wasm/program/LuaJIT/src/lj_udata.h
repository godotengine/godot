/*
** Userdata handling.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_UDATA_H
#define _LJ_UDATA_H

#include "lj_obj.h"

LJ_FUNC GCudata *lj_udata_new(lua_State *L, MSize sz, GCtab *env);
LJ_FUNC void LJ_FASTCALL lj_udata_free(global_State *g, GCudata *ud);
#if LJ_64
LJ_FUNC void * LJ_FASTCALL lj_lightud_intern(lua_State *L, void *p);
#endif

#endif
