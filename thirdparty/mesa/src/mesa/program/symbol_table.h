/*
 * Copyright © 2008 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef MESA_SYMBOL_TABLE_H
#define MESA_SYMBOL_TABLE_H

#ifdef __cplusplus
extern "C" {
#endif

struct _mesa_symbol_table;

extern void _mesa_symbol_table_push_scope(struct _mesa_symbol_table *table);

extern void _mesa_symbol_table_pop_scope(struct _mesa_symbol_table *table);

extern int _mesa_symbol_table_add_symbol(struct _mesa_symbol_table *symtab,
                                         const char *name, void *declaration);

extern int _mesa_symbol_table_replace_symbol(struct _mesa_symbol_table *table,
                                             const char *name,
                                             void *declaration);

extern int
_mesa_symbol_table_add_global_symbol(struct _mesa_symbol_table *symtab,
                                     const char *name,
                                     void *declaration);

extern int _mesa_symbol_table_symbol_scope(struct _mesa_symbol_table *table,
                                           const char *name);

extern void *_mesa_symbol_table_find_symbol(struct _mesa_symbol_table *symtab,
                                            const char *name);

extern struct _mesa_symbol_table *_mesa_symbol_table_ctor(void);

extern void _mesa_symbol_table_dtor(struct _mesa_symbol_table *);

#ifdef __cplusplus
}
#endif

#endif /* MESA_SYMBOL_TABLE_H */
