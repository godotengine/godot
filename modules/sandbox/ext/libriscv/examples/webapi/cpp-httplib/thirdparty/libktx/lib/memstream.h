/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2010-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file
 * @~English
 *
 * @brief Interface of ktxStream for memory.
 *
 * @author Maksim Kolesin
 * @author Georg Kolling, Imagination Technology
 * @author Mark Callow, HI Corporation
 */

#ifndef MEMSTREAM_H
#define MEMSTREAM_H

#include "ktx.h"

/*
 * Initialize a ktxStream to a ktxMemStream with internally
 * allocated memory. Can be read or written.
 */
KTX_error_code ktxMemStream_construct(ktxStream* str,
                                      ktx_bool_t freeOnDestruct);
/*
 * Initialize a ktxStream to a read-only ktxMemStream reading
 * from an array of bytes.
 */
KTX_error_code ktxMemStream_construct_ro(ktxStream* str,
                                         const ktx_uint8_t* pBytes,
                                         const ktx_size_t size);
void ktxMemStream_destruct(ktxStream* str);

KTX_error_code ktxMemStream_getdata(ktxStream* str, ktx_uint8_t** ppBytes);

#endif /* MEMSTREAM_H */
