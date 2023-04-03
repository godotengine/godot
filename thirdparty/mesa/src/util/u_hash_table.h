/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/**
 * General purpose hash table.
 */

#ifndef U_HASH_TABLE_H_
#define U_HASH_TABLE_H_


#include "pipe/p_defines.h"
#include "util/hash_table.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a hash table where the keys are generic pointers.
 */
struct hash_table *
util_hash_table_create_ptr_keys(void);


/**
 * Create a hash table where the keys are device FDs.
 */
struct hash_table *
util_hash_table_create_fd_keys(void);


void *
util_hash_table_get(struct hash_table *ht,
                    void *key);


int
util_hash_table_foreach(struct hash_table *ht,
                        int (*callback)
                        (void *key, void *value, void *data),
                        void *data);

#ifdef __cplusplus
}
#endif

#endif /* U_HASH_TABLE_H_ */
