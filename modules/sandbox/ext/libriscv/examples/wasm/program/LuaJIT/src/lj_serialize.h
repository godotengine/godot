/*
** Object de/serialization.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_SERIALIZE_H
#define _LJ_SERIALIZE_H

#include "lj_obj.h"
#include "lj_buf.h"

#if LJ_HASBUFFER

#define LJ_SERIALIZE_DEPTH	100	/* Default depth. */

LJ_FUNC void LJ_FASTCALL lj_serialize_dict_prep_str(lua_State *L, GCtab *dict);
LJ_FUNC void LJ_FASTCALL lj_serialize_dict_prep_mt(lua_State *L, GCtab *dict);
LJ_FUNC SBufExt * LJ_FASTCALL lj_serialize_put(SBufExt *sbx, cTValue *o);
LJ_FUNC char * LJ_FASTCALL lj_serialize_get(SBufExt *sbx, TValue *o);
LJ_FUNC GCstr * LJ_FASTCALL lj_serialize_encode(lua_State *L, cTValue *o);
LJ_FUNC void lj_serialize_decode(lua_State *L, TValue *o, GCstr *str);
#if LJ_HASJIT
LJ_FUNC MSize LJ_FASTCALL lj_serialize_peektype(SBufExt *sbx);
#endif

#endif

#endif
