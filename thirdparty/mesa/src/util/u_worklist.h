/*
 * Copyright (c) 2022 Collabora Ltd.
 * Copyright © 2014 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */


#ifndef _U_WORKLIST_
#define _U_WORKLIST_

#include "util/bitset.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Represents a double-ended queue of unique entries. Each entry must have a
 * unique index in the range [0, num_entries). Internally, entries are tracked
 * as pointers to that index (so you can go from the index pointer back to a
 * containing structure). This requires index pointers to remain valid while
 * they are in the worklist (i.e. no realloc).
 *
 * The worklist data structure guarantees that each entry is in the queue at
 * most once. Pushing an entry onto either end of the queue is a no-op if the
 * entry is already in the queue. Internally, the data structure maintains a
 * bitset of present entries.
 */
typedef struct {
   /* The total size of the worklist */
   unsigned size;

   /* The number of entries currently in the worklist */
   unsigned count;

   /* The offset in the array of entries at which the list starts */
   unsigned start;

   /* A bitset of all of the entries currently present in the worklist */
   BITSET_WORD *present;

   /* The actual worklist */
   unsigned **entries;
} u_worklist;

void u_worklist_init(u_worklist *w, unsigned num_entries, void *mem_ctx);

void u_worklist_fini(u_worklist *w);

static inline bool
u_worklist_is_empty(const u_worklist *w)
{
   return w->count == 0;
}

void u_worklist_push_head_index(u_worklist *w, unsigned *block);

unsigned *u_worklist_peek_head_index(const u_worklist *w);

unsigned *u_worklist_pop_head_index(u_worklist *w);

unsigned *u_worklist_peek_tail_index(const u_worklist *w);

void u_worklist_push_tail_index(u_worklist *w, unsigned *block);

unsigned *u_worklist_pop_tail_index(u_worklist *w);

#define u_worklist_push_tail(w, block, index) \
   u_worklist_push_tail_index(w, &((block)->index))

#define u_worklist_push_head(w, block, index) \
   u_worklist_push_head_index(w, &((block)->index))

#define u_worklist_pop_head(w, entry_t, index) \
   container_of(u_worklist_pop_head_index(w), entry_t, index)

#define u_worklist_pop_tail(w, entry_t, index) \
   container_of(u_worklist_pop_tail_index(w), entry_t, index)

#define u_worklist_peek_head(w, entry_t, index) \
   container_of(u_worklist_peek_head_index(w), entry_t, index)

#define u_worklist_peek_tail(w, entry_t, index) \
   container_of(u_worklist_peek_tail_index(w), entry_t, index)

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _U_WORKLIST_ */
