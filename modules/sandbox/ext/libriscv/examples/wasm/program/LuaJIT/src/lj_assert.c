/*
** Internal assertions.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_assert_c
#define LUA_CORE

#if defined(LUA_USE_ASSERT) || defined(LUA_USE_APICHECK)

#include <stdio.h>

#include "lj_obj.h"

void lj_assert_fail(global_State *g, const char *file, int line,
		    const char *func, const char *fmt, ...)
{
  va_list argp;
  va_start(argp, fmt);
  fprintf(stderr, "LuaJIT ASSERT %s:%d: %s: ", file, line, func);
  vfprintf(stderr, fmt, argp);
  fputc('\n', stderr);
  va_end(argp);
  UNUSED(g);  /* May be NULL. TODO: optionally dump state. */
  abort();
}

#endif
