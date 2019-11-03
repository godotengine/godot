/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef VPX_MEM_INCLUDE_VPX_MEM_INTRNL_H_
#define VPX_MEM_INCLUDE_VPX_MEM_INTRNL_H_
#include "./vpx_config.h"

#define ADDRESS_STORAGE_SIZE      sizeof(size_t)

#ifndef DEFAULT_ALIGNMENT
# if defined(VXWORKS)
#  define DEFAULT_ALIGNMENT        32        /*default addr alignment to use in
calls to vpx_* functions other
than vpx_memalign*/
# else
#  define DEFAULT_ALIGNMENT        (2 * sizeof(void*))  /* NOLINT */
# endif
#endif

/*returns an addr aligned to the byte boundary specified by align*/
#define align_addr(addr,align) (void*)(((size_t)(addr) + ((align) - 1)) & (size_t)-(align))

#endif  // VPX_MEM_INCLUDE_VPX_MEM_INTRNL_H_
