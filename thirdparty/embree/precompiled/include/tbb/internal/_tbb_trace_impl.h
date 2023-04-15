/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef _FGT_TBB_TRACE_IMPL_H
#define _FGT_TBB_TRACE_IMPL_H

#include "../tbb_profiling.h"

namespace tbb {
    namespace internal {

#if TBB_PREVIEW_ALGORITHM_TRACE
        static inline void fgt_algorithm( string_index t, void *algorithm, void *parent ) {
            itt_make_task_group( ITT_DOMAIN_FLOW, algorithm, ALGORITHM, parent, ALGORITHM, t );
        }
        static inline void fgt_begin_algorithm( string_index t, void *algorithm ) {
            itt_task_begin( ITT_DOMAIN_FLOW, algorithm, ALGORITHM, NULL, FLOW_NULL, t );
        }
        static inline void fgt_end_algorithm( void * ) {
            itt_task_end( ITT_DOMAIN_FLOW );
        }
        static inline void fgt_alg_begin_body( string_index t, void *body, void *algorithm ) {
            itt_task_begin( ITT_DOMAIN_FLOW, body, FLOW_BODY, algorithm, ALGORITHM, t );
        }
        static inline void fgt_alg_end_body( void * ) {
            itt_task_end( ITT_DOMAIN_FLOW );
        }

#else // TBB_PREVIEW_ALGORITHM_TRACE

        static inline void fgt_algorithm( string_index /*t*/, void * /*algorithm*/, void * /*parent*/ ) { }
        static inline void fgt_begin_algorithm( string_index /*t*/, void * /*algorithm*/ ) { }
        static inline void fgt_end_algorithm( void * ) { }
        static inline void fgt_alg_begin_body( string_index /*t*/, void * /*body*/, void * /*algorithm*/ ) { }
        static inline void fgt_alg_end_body( void * ) { }

#endif // TBB_PREVIEW_ALGORITHM_TRACEE

    } // namespace internal
} // namespace tbb

#endif
