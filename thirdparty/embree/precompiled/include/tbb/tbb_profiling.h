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

#ifndef __TBB_profiling_H
#define __TBB_profiling_H

#define __TBB_tbb_profiling_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

namespace tbb {
    namespace internal {

        // include list of index names
        #define TBB_STRING_RESOURCE(index_name,str) index_name,
        enum string_index {
           #include "internal/_tbb_strings.h"
           NUM_STRINGS
        };
        #undef TBB_STRING_RESOURCE

        enum itt_relation
        {
        __itt_relation_is_unknown = 0,
        __itt_relation_is_dependent_on,         /**< "A is dependent on B" means that A cannot start until B completes */
        __itt_relation_is_sibling_of,           /**< "A is sibling of B" means that A and B were created as a group */
        __itt_relation_is_parent_of,            /**< "A is parent of B" means that A created B */
        __itt_relation_is_continuation_of,      /**< "A is continuation of B" means that A assumes the dependencies of B */
        __itt_relation_is_child_of,             /**< "A is child of B" means that A was created by B (inverse of is_parent_of) */
        __itt_relation_is_continued_by,         /**< "A is continued by B" means that B assumes the dependencies of A (inverse of is_continuation_of) */
        __itt_relation_is_predecessor_to        /**< "A is predecessor to B" means that B cannot start until A completes (inverse of is_dependent_on) */
        };

    }
}

// Check if the tools support is enabled
#if (_WIN32||_WIN64||__linux__) && !__MINGW32__ && TBB_USE_THREADING_TOOLS

#if _WIN32||_WIN64
#include <stdlib.h>  /* mbstowcs_s */
#endif
#include "tbb_stddef.h"

namespace tbb {
    namespace internal {

#if _WIN32||_WIN64
        void __TBB_EXPORTED_FUNC itt_set_sync_name_v3( void *obj, const wchar_t* name );
        inline size_t multibyte_to_widechar( wchar_t* wcs, const char* mbs, size_t bufsize) {
#if _MSC_VER>=1400
            size_t len;
            mbstowcs_s( &len, wcs, bufsize, mbs, _TRUNCATE );
            return len;   // mbstowcs_s counts null terminator
#else
            size_t len = mbstowcs( wcs, mbs, bufsize );
            if(wcs && len!=size_t(-1) )
                wcs[len<bufsize-1? len: bufsize-1] = wchar_t('\0');
            return len+1; // mbstowcs does not count null terminator
#endif
        }
#else
        void __TBB_EXPORTED_FUNC itt_set_sync_name_v3( void *obj, const char* name );
#endif
    } // namespace internal
} // namespace tbb

//! Macro __TBB_DEFINE_PROFILING_SET_NAME(T) defines "set_name" methods for sync objects of type T
/** Should be used in the "tbb" namespace only.
    Don't place semicolon after it to avoid compiler warnings. **/
#if _WIN32||_WIN64
    #define __TBB_DEFINE_PROFILING_SET_NAME(sync_object_type)                       \
        namespace profiling {                                                       \
            inline void set_name( sync_object_type& obj, const wchar_t* name ) {    \
                tbb::internal::itt_set_sync_name_v3( &obj, name );                  \
            }                                                                       \
            inline void set_name( sync_object_type& obj, const char* name ) {       \
                size_t len = tbb::internal::multibyte_to_widechar(NULL, name, 0);   \
                wchar_t *wname = new wchar_t[len];                                  \
                tbb::internal::multibyte_to_widechar(wname, name, len);             \
                set_name( obj, wname );                                             \
                delete[] wname;                                                     \
            }                                                                       \
        }
#else /* !WIN */
    #define __TBB_DEFINE_PROFILING_SET_NAME(sync_object_type)                       \
        namespace profiling {                                                       \
            inline void set_name( sync_object_type& obj, const char* name ) {       \
                tbb::internal::itt_set_sync_name_v3( &obj, name );                  \
            }                                                                       \
        }
#endif /* !WIN */

#else /* no tools support */

#if _WIN32||_WIN64
    #define __TBB_DEFINE_PROFILING_SET_NAME(sync_object_type)               \
        namespace profiling {                                               \
            inline void set_name( sync_object_type&, const wchar_t* ) {}    \
            inline void set_name( sync_object_type&, const char* ) {}       \
        }
#else /* !WIN */
    #define __TBB_DEFINE_PROFILING_SET_NAME(sync_object_type)               \
        namespace profiling {                                               \
            inline void set_name( sync_object_type&, const char* ) {}       \
        }
#endif /* !WIN */

#endif /* no tools support */

#include "atomic.h"

// Need these to work regardless of tools support
namespace tbb {
    namespace internal {

        enum notify_type {prepare=0, cancel, acquired, releasing};

        const uintptr_t NUM_NOTIFY_TYPES = 4; // set to # elements in enum above

        void __TBB_EXPORTED_FUNC call_itt_notify_v5(int t, void *ptr);
        void __TBB_EXPORTED_FUNC itt_store_pointer_with_release_v3(void *dst, void *src);
        void* __TBB_EXPORTED_FUNC itt_load_pointer_with_acquire_v3(const void *src);
        void* __TBB_EXPORTED_FUNC itt_load_pointer_v3( const void* src );
        enum itt_domain_enum { ITT_DOMAIN_FLOW=0, ITT_DOMAIN_MAIN=1, ITT_DOMAIN_ALGO=2, ITT_NUM_DOMAINS };

        void __TBB_EXPORTED_FUNC itt_make_task_group_v7( itt_domain_enum domain, void *group, unsigned long long group_extra,
                                                         void *parent, unsigned long long parent_extra, string_index name_index );
        void __TBB_EXPORTED_FUNC itt_metadata_str_add_v7( itt_domain_enum domain, void *addr, unsigned long long addr_extra,
                                                          string_index key, const char *value );
        void __TBB_EXPORTED_FUNC itt_metadata_ptr_add_v11( itt_domain_enum domain, void *addr, unsigned long long addr_extra,
                                                           string_index key, void* value );
        void __TBB_EXPORTED_FUNC itt_relation_add_v7( itt_domain_enum domain, void *addr0, unsigned long long addr0_extra,
                                                      itt_relation relation, void *addr1, unsigned long long addr1_extra );
        void __TBB_EXPORTED_FUNC itt_task_begin_v7( itt_domain_enum domain, void *task, unsigned long long task_extra,
                                                    void *parent, unsigned long long parent_extra, string_index name_index );
        void __TBB_EXPORTED_FUNC itt_task_end_v7( itt_domain_enum domain );

        void __TBB_EXPORTED_FUNC itt_region_begin_v9( itt_domain_enum domain, void *region, unsigned long long region_extra,
                                                      void *parent, unsigned long long parent_extra, string_index name_index );
        void __TBB_EXPORTED_FUNC itt_region_end_v9( itt_domain_enum domain, void *region, unsigned long long region_extra );

        // two template arguments are to workaround /Wp64 warning with tbb::atomic specialized for unsigned type
        template <typename T, typename U>
        inline void itt_store_word_with_release(tbb::atomic<T>& dst, U src) {
#if TBB_USE_THREADING_TOOLS
            // This assertion should be replaced with static_assert
            __TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized.");
            itt_store_pointer_with_release_v3(&dst, (void *)uintptr_t(src));
#else
            dst = src;
#endif // TBB_USE_THREADING_TOOLS
        }

        template <typename T>
        inline T itt_load_word_with_acquire(const tbb::atomic<T>& src) {
#if TBB_USE_THREADING_TOOLS
            // This assertion should be replaced with static_assert
            __TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized.");
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
            // Workaround for overzealous compiler warnings
            #pragma warning (push)
            #pragma warning (disable: 4311)
#endif
            T result = (T)itt_load_pointer_with_acquire_v3(&src);
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
            #pragma warning (pop)
#endif
            return result;
#else
            return src;
#endif // TBB_USE_THREADING_TOOLS
        }

        template <typename T>
        inline void itt_store_word_with_release(T& dst, T src) {
#if TBB_USE_THREADING_TOOLS
            // This assertion should be replaced with static_assert
            __TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized.");
            itt_store_pointer_with_release_v3(&dst, (void *)src);
#else
            __TBB_store_with_release(dst, src);
#endif // TBB_USE_THREADING_TOOLS
        }

        template <typename T>
        inline T itt_load_word_with_acquire(const T& src) {
#if TBB_USE_THREADING_TOOLS
            // This assertion should be replaced with static_assert
            __TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized");
            return (T)itt_load_pointer_with_acquire_v3(&src);
#else
            return __TBB_load_with_acquire(src);
#endif // TBB_USE_THREADING_TOOLS
        }

        template <typename T>
        inline void itt_hide_store_word(T& dst, T src) {
#if TBB_USE_THREADING_TOOLS
            //TODO: This assertion should be replaced with static_assert
            __TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized");
            itt_store_pointer_with_release_v3(&dst, (void *)src);
#else
            dst = src;
#endif
        }

        //TODO: rename to itt_hide_load_word_relaxed
        template <typename T>
        inline T itt_hide_load_word(const T& src) {
#if TBB_USE_THREADING_TOOLS
            //TODO: This assertion should be replaced with static_assert
            __TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized.");
            return (T)itt_load_pointer_v3(&src);
#else
            return src;
#endif
        }

#if TBB_USE_THREADING_TOOLS
        inline void call_itt_notify(notify_type t, void *ptr) {
            call_itt_notify_v5((int)t, ptr);
        }

        inline void itt_make_task_group( itt_domain_enum domain, void *group, unsigned long long group_extra,
                                         void *parent, unsigned long long parent_extra, string_index name_index ) {
            itt_make_task_group_v7( domain, group, group_extra, parent, parent_extra, name_index );
        }

        inline void itt_metadata_str_add( itt_domain_enum domain, void *addr, unsigned long long addr_extra,
                                          string_index key, const char *value ) {
            itt_metadata_str_add_v7( domain, addr, addr_extra, key, value );
        }
        
        inline void register_node_addr(itt_domain_enum domain, void *addr, unsigned long long addr_extra,
            string_index key, void *value) {
            itt_metadata_ptr_add_v11(domain, addr, addr_extra, key, value);
        }

        inline void itt_relation_add( itt_domain_enum domain, void *addr0, unsigned long long addr0_extra,
                                      itt_relation relation, void *addr1, unsigned long long addr1_extra ) {
            itt_relation_add_v7( domain, addr0, addr0_extra, relation, addr1, addr1_extra );
        }

        inline void itt_task_begin( itt_domain_enum domain, void *task, unsigned long long task_extra,
                                                        void *parent, unsigned long long parent_extra, string_index name_index ) {
            itt_task_begin_v7( domain, task, task_extra, parent, parent_extra, name_index );
        }

        inline void itt_task_end( itt_domain_enum domain ) {
            itt_task_end_v7( domain );
        }

        inline void itt_region_begin( itt_domain_enum domain, void *region, unsigned long long region_extra,
                                      void *parent, unsigned long long parent_extra, string_index name_index ) {
            itt_region_begin_v9( domain, region, region_extra, parent, parent_extra, name_index );
        }

        inline void itt_region_end( itt_domain_enum domain, void *region, unsigned long long region_extra  ) {
            itt_region_end_v9( domain, region, region_extra );
        }
#else
        inline void register_node_addr( itt_domain_enum /*domain*/, void* /*addr*/, unsigned long long /*addr_extra*/, string_index /*key*/, void* /*value*/ ) {}
        inline void call_itt_notify(notify_type /*t*/, void* /*ptr*/) {}

        inline void itt_make_task_group( itt_domain_enum /*domain*/, void* /*group*/, unsigned long long /*group_extra*/,
                                         void* /*parent*/, unsigned long long /*parent_extra*/, string_index /*name_index*/ ) {}

        inline void itt_metadata_str_add( itt_domain_enum /*domain*/, void* /*addr*/, unsigned long long /*addr_extra*/,
                                          string_index /*key*/, const char* /*value*/ ) {}

        inline void itt_relation_add( itt_domain_enum /*domain*/, void* /*addr0*/, unsigned long long /*addr0_extra*/,
                                      itt_relation /*relation*/, void* /*addr1*/, unsigned long long /*addr1_extra*/ ) {}

        inline void itt_task_begin( itt_domain_enum /*domain*/, void* /*task*/, unsigned long long /*task_extra*/,
                                    void* /*parent*/, unsigned long long /*parent_extra*/, string_index /*name_index*/ ) {}

        inline void itt_task_end( itt_domain_enum /*domain*/ ) {}

        inline void itt_region_begin( itt_domain_enum /*domain*/, void* /*region*/, unsigned long long /*region_extra*/,
                                      void* /*parent*/, unsigned long long /*parent_extra*/, string_index /*name_index*/ ) {}

        inline void itt_region_end( itt_domain_enum /*domain*/, void* /*region*/, unsigned long long /*region_extra*/ ) {}
#endif // TBB_USE_THREADING_TOOLS

    } // namespace internal
} // namespace tbb

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
#include <string>

namespace tbb {
namespace profiling {
namespace interface10 {

#if TBB_USE_THREADING_TOOLS && !(TBB_USE_THREADING_TOOLS == 2)
class event {
/** This class supports user event traces through itt.
    Common use-case is tagging data flow graph tasks (data-id)
    and visualization by Intel Advisor Flow Graph Analyzer (FGA)  **/
//  TODO: Replace implementation by itt user event api.

    const std::string my_name;

    static void emit_trace(const std::string &input) {
        itt_metadata_str_add( tbb::internal::ITT_DOMAIN_FLOW, NULL, tbb::internal::FLOW_NULL, tbb::internal::USER_EVENT, ( "FGA::DATAID::" + input ).c_str() );
    }

public:
    event(const std::string &input)
              : my_name( input )
    { }

    void emit() {
        emit_trace(my_name);
    }

    static void emit(const std::string &description) {
        emit_trace(description);
    }

};
#else // TBB_USE_THREADING_TOOLS && !(TBB_USE_THREADING_TOOLS == 2)
// Using empty struct if user event tracing is disabled:
struct event {
    event(const std::string &) { }

    void emit() { }

    static void emit(const std::string &) { }
};
#endif // TBB_USE_THREADING_TOOLS && !(TBB_USE_THREADING_TOOLS == 2)

} // interfaceX
using interface10::event;
} // namespace profiling
} // namespace tbb
#endif // TBB_PREVIEW_FLOW_GRAPH_TRACE

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_tbb_profiling_H_include_area

#endif /* __TBB_profiling_H */
