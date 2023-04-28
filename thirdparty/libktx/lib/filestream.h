/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2010-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Author: Maksim Kolesin from original code
 * by Mark Callow and Georg Kolling
 */

#ifndef FILESTREAM_H
#define FILESTREAM_H

#include "ktx.h"

/*
 * ktxFileInit: Initialize a ktxStream to a ktxFileStream with a FILE object
 */
KTX_error_code ktxFileStream_construct(ktxStream* str, FILE* file,
                                       ktx_bool_t closeFileOnDestruct);

void ktxFileStream_destruct(ktxStream* str);

#endif /* FILESTREAM_H */
