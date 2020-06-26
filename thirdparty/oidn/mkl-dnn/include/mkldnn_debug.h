/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* DO NOT EDIT, AUTO-GENERATED */

#ifndef MKLDNN_DEBUG_H
#define MKLDNN_DEBUG_H

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/* All symbols shall be internal unless marked as MKLDNN_API */
#if defined _WIN32 || defined __CYGWIN__
#   define MKLDNN_HELPER_DLL_IMPORT __declspec(dllimport)
#   define MKLDNN_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#   if __GNUC__ >= 4
#       define MKLDNN_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
#       define MKLDNN_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
#   else
#       define MKLDNN_HELPER_DLL_IMPORT
#       define MKLDNN_HELPER_DLL_EXPORT
#   endif
#endif

#ifdef MKLDNN_DLL
#   ifdef MKLDNN_DLL_EXPORTS
#       define MKLDNN_API MKLDNN_HELPER_DLL_EXPORT
#   else
#       define MKLDNN_API MKLDNN_HELPER_DLL_IMPORT
#   endif
#else
#   define MKLDNN_API
#endif

#if defined (__GNUC__)
#   define MKLDNN_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#   define MKLDNN_DEPRECATED __declspec(deprecated)
#else
#   define MKLDNN_DEPRECATED
#endif

#include "mkldnn_types.h"
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#ifdef __cplusplus
extern "C" {
#endif

const char MKLDNN_API *mkldnn_status2str(mkldnn_status_t v);
const char MKLDNN_API *mkldnn_dt2str(mkldnn_data_type_t v);
const char MKLDNN_API *mkldnn_fmt_kind2str(mkldnn_format_kind_t v);
const char MKLDNN_API *mkldnn_fmt_tag2str(mkldnn_format_tag_t v);
const char MKLDNN_API *mkldnn_prop_kind2str(mkldnn_prop_kind_t v);
const char MKLDNN_API *mkldnn_prim_kind2str(mkldnn_primitive_kind_t v);
const char MKLDNN_API *mkldnn_alg_kind2str(mkldnn_alg_kind_t v);
const char MKLDNN_API *mkldnn_rnn_direction2str(mkldnn_rnn_direction_t v);

/** Forms a format string for a given memory descriptor.
 *
 * The format is defined as: 'dt:[p|o|0]:fmt_kind:fmt:extra'.
 * Here:
 *  - dt       -- data type
 *  - p        -- indicates there is non-trivial padding
 *  - o        -- indicates there is non-trivial padding offset
 *  - 0        -- indicates there is non-trivial offset0
 *  - fmt_kind -- format kind (blocked, wino, etc...)
 *  - fmt      -- extended format string (format_kind specific)
 *  - extra    -- shows extra fields (underspecified)
 */
int MKLDNN_API mkldnn_md2fmt_str(char *fmt_str, size_t fmt_str_len,
        const mkldnn_memory_desc_t *md);

/** Forms a dimension string for a given memory descriptor.
 *
 * The format is defined as: 'dim0xdim1x...xdimN
 */
int MKLDNN_API mkldnn_md2dim_str(char *dim_str, size_t dim_str_len,
        const mkldnn_memory_desc_t *md);

#ifdef __cplusplus
}
#endif

#endif
