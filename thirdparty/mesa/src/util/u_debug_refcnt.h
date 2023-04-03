/**************************************************************************
 *
 * Copyright 2010 Luca Barbieri
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER(S) AND/OR ITS SUPPLIERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef U_DEBUG_REFCNT_H_
#define U_DEBUG_REFCNT_H_

#include "util/detect.h"
#include "pipe/p_state.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*debug_reference_descriptor)(char*, const struct pipe_reference*);

#if defined(DEBUG)

extern int debug_refcnt_state;

void
debug_reference_slowpath(const struct pipe_reference* p,
                         debug_reference_descriptor get_desc, int change);

static inline void
debug_reference(const struct pipe_reference* p,
                debug_reference_descriptor get_desc, int change)
{
   if (debug_refcnt_state >= 0)
      debug_reference_slowpath(p, get_desc, change);
}

#else

static inline void
debug_reference(UNUSED const struct pipe_reference* p,
                UNUSED debug_reference_descriptor get_desc, UNUSED int change)
{
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* U_DEBUG_REFCNT_H_ */
