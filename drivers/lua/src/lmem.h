/*
** $Id: lmem.h,v 1.31.1.1 2007/12/27 13:02:25 roberto Exp $
** Interface to Memory Manager
** See Copyright Notice in lua.h
*/

#ifndef lmem_h
#define lmem_h


#include <stddef.h>

#include "llimits.h"
#include "lua.h"

#define MEMERRMSG	"not enough memory"


#define luaM_reallocv(L,b,on,n,e) \
	((cast(size_t, (n)+1) <= MAX_SIZET/(e)) ?  /* +1 to avoid warnings */ \
		luaM_realloc_(L, (b), (on)*(e), (n)*(e)) : \
		luaM_toobig(L))

#define luaM_freemem(L, b, s)	luaM_realloc_(L, (b), (s), 0)
#define luaM_free(L, b)		luaM_realloc_(L, (b), sizeof(*(b)), 0)
#define luaM_freearray(L, b, n, t)   luaM_reallocv(L, (b), n, 0, sizeof(t))

#define luaM_malloc(L,t)	luaM_realloc_(L, NULL, 0, (t))
#define luaM_new(L,t)		cast(t *, luaM_malloc(L, sizeof(t)))
#define luaM_newvector(L,n,t) \
		cast(t *, luaM_reallocv(L, NULL, 0, n, sizeof(t)))

#define luaM_growvector(L,v,nelems,size,t,limit,e) \
          if ((nelems)+1 > (size)) \
            ((v)=cast(t *, luaM_growaux_(L,v,&(size),sizeof(t),limit,e)))

#define luaM_reallocvector(L, v,oldn,n,t) \
   ((v)=cast(t *, luaM_reallocv(L, v, oldn, n, sizeof(t))))


LUAI_FUNC void *luaM_realloc_ (lua_State *L, void *block, size_t oldsize,
                                                          size_t size);
LUAI_FUNC void *luaM_toobig (lua_State *L);
LUAI_FUNC void *luaM_growaux_ (lua_State *L, void *block, int *size,
                               size_t size_elem, int limit,
                               const char *errormsg);

#endif

