/*
 * Copyright © 2019 Broadcom
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
 */

#ifndef DAG_H
#define DAG_H

#include <stdint.h>
#include "util/list.h"
#include "util/u_dynarray.h"

#ifdef __cplusplus
extern "C" {
#endif

struct dag_edge {
   struct dag_node *child;
   /* User-defined data associated with the edge. */
   uintptr_t data;
};

struct dag_node {
   /* Position in the DAG heads list (or a self-link) */
   struct list_head link;
   /* Array struct edge to the children. */
   struct util_dynarray edges;
   uint32_t parent_count;
};

struct dag {
   struct list_head heads;
};

struct dag *dag_create(void *mem_ctx);
void dag_init_node(struct dag *dag, struct dag_node *node);
void dag_add_edge(struct dag_node *parent, struct dag_node *child, uintptr_t data);
void dag_add_edge_max_data(struct dag_node *parent, struct dag_node *child, uintptr_t data);
void dag_remove_edge(struct dag *dag, struct dag_edge *edge);
void dag_traverse_bottom_up(struct dag *dag, void (*cb)(struct dag_node *node,
                                                        void *data), void *data);
void dag_prune_head(struct dag *dag, struct dag_node *node);
void dag_validate(struct dag *dag, void (*cb)(const struct dag_node *node,
                                              void *data), void *data);

#ifdef __cplusplus
}
#endif

#endif
