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

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_tbb_thread_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_tbb_thread_H
#pragma message("TBB Warning: tbb/tbb_thread.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_tbb_thread_H
#define __TBB_tbb_thread_H

#define __TBB_tbb_thread_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_stddef.h"

#if _WIN32||_WIN64
#include "machine/windows_api.h"
#define __TBB_NATIVE_THREAD_ROUTINE unsigned WINAPI
#define __TBB_NATIVE_THREAD_ROUTINE_PTR(r) unsigned (WINAPI* r)( void* )
namespace tbb { namespace internal {
#if __TBB_WIN8UI_SUPPORT
    typedef size_t thread_id_type;
#else  // __TBB_WIN8UI_SUPPORT
    typedef DWORD thread_id_type;
#endif // __TBB_WIN8UI_SUPPORT
}} //namespace tbb::internal
#else
#define __TBB_NATIVE_THREAD_ROUTINE void*
#define __TBB_NATIVE_THREAD_ROUTINE_PTR(r) void* (*r)( void* )
#include <pthread.h>
namespace tbb { namespace internal {
    typedef pthread_t thread_id_type;
}} //namespace tbb::internal
#endif // _WIN32||_WIN64

#include "atomic.h"
#include "internal/_tbb_hash_compare_impl.h"
#include "tick_count.h"

#include __TBB_STD_SWAP_HEADER
#include <iosfwd>

namespace tbb {

namespace internal {
    class tbb_thread_v3;
}

inline void swap( internal::tbb_thread_v3& t1, internal::tbb_thread_v3& t2 ) __TBB_NOEXCEPT(true);

namespace internal {

    //! Allocate a closure
    void* __TBB_EXPORTED_FUNC allocate_closure_v3( size_t size );
    //! Free a closure allocated by allocate_closure_v3
    void __TBB_EXPORTED_FUNC free_closure_v3( void* );

    struct thread_closure_base {
        void* operator new( size_t size ) {return allocate_closure_v3(size);}
        void operator delete( void* ptr ) {free_closure_v3(ptr);}
    };

    template<class F> struct thread_closure_0: thread_closure_base {
        F function;

        static __TBB_NATIVE_THREAD_ROUTINE start_routine( void* c ) {
            thread_closure_0 *self = static_cast<thread_closure_0*>(c);
            self->function();
            delete self;
            return 0;
        }
        thread_closure_0( const F& f ) : function(f) {}
    };
    //! Structure used to pass user function with 1 argument to thread.
    template<class F, class X> struct thread_closure_1: thread_closure_base {
        F function;
        X arg1;
        //! Routine passed to Windows's _beginthreadex by thread::internal_start() inside tbb.dll
        static __TBB_NATIVE_THREAD_ROUTINE start_routine( void* c ) {
            thread_closure_1 *self = static_cast<thread_closure_1*>(c);
            self->function(self->arg1);
            delete self;
            return 0;
        }
        thread_closure_1( const F& f, const X& x ) : function(f), arg1(x) {}
    };
    template<class F, class X, class Y> struct thread_closure_2: thread_closure_base {
        F function;
        X arg1;
        Y arg2;
        //! Routine passed to Windows's _beginthreadex by thread::internal_start() inside tbb.dll
        static __TBB_NATIVE_THREAD_ROUTINE start_routine( void* c ) {
            thread_closure_2 *self = static_cast<thread_closure_2*>(c);
            self->function(self->arg1, self->arg2);
            delete self;
            return 0;
        }
        thread_closure_2( const F& f, const X& x, const Y& y ) : function(f), arg1(x), arg2(y) {}
    };

    //! Versioned thread class.
    class tbb_thread_v3 {
#if __TBB_IF_NO_COPY_CTOR_MOVE_SEMANTICS_BROKEN
        // Workaround for a compiler bug: declaring the copy constructor as public
        // enables use of the moving constructor.
        // The definition is not provided in order to prohibit copying.
    public:
#endif
        tbb_thread_v3(const tbb_thread_v3&); // = delete;   // Deny access
    public:
#if _WIN32||_WIN64
        typedef HANDLE native_handle_type;
#else
        typedef pthread_t native_handle_type;
#endif // _WIN32||_WIN64

        class id;
        //! Constructs a thread object that does not represent a thread of execution.
        tbb_thread_v3() __TBB_NOEXCEPT(true) : my_handle(0)
#if _WIN32||_WIN64
            , my_thread_id(0)
#endif // _WIN32||_WIN64
        {}

        //! Constructs an object and executes f() in a new thread
        template <class F> explicit tbb_thread_v3(F f) {
            typedef internal::thread_closure_0<F> closure_type;
            internal_start(closure_type::start_routine, new closure_type(f));
        }
        //! Constructs an object and executes f(x) in a new thread
        template <class F, class X> tbb_thread_v3(F f, X x) {
            typedef internal::thread_closure_1<F,X> closure_type;
            internal_start(closure_type::start_routine, new closure_type(f,x));
        }
        //! Constructs an object and executes f(x,y) in a new thread
        template <class F, class X, class Y> tbb_thread_v3(F f, X x, Y y) {
            typedef internal::thread_closure_2<F,X,Y> closure_type;
            internal_start(closure_type::start_routine, new closure_type(f,x,y));
        }

#if __TBB_CPP11_RVALUE_REF_PRESENT
        tbb_thread_v3(tbb_thread_v3&& x) __TBB_NOEXCEPT(true)
            : my_handle(x.my_handle)
#if _WIN32||_WIN64
            , my_thread_id(x.my_thread_id)
#endif
        {
            x.internal_wipe();
        }
        tbb_thread_v3& operator=(tbb_thread_v3&& x) __TBB_NOEXCEPT(true) {
            internal_move(x);
            return *this;
        }
    private:
        tbb_thread_v3& operator=(const tbb_thread_v3& x); // = delete;
    public:
#else  // __TBB_CPP11_RVALUE_REF_PRESENT
        tbb_thread_v3& operator=(tbb_thread_v3& x) {
            internal_move(x);
            return *this;
        }
#endif // __TBB_CPP11_RVALUE_REF_PRESENT

        void swap( tbb_thread_v3& t ) __TBB_NOEXCEPT(true) {tbb::swap( *this, t );}
        bool joinable() const __TBB_NOEXCEPT(true) {return my_handle!=0; }
        //! The completion of the thread represented by *this happens before join() returns.
        void __TBB_EXPORTED_METHOD join();
        //! When detach() returns, *this no longer represents the possibly continuing thread of execution.
        void __TBB_EXPORTED_METHOD detach();
        ~tbb_thread_v3() {if( joinable() ) detach();}
        inline id get_id() const __TBB_NOEXCEPT(true);
        native_handle_type native_handle() { return my_handle; }

        //! The number of hardware thread contexts.
        /** Before TBB 3.0 U4 this methods returned the number of logical CPU in
            the system. Currently on Windows, Linux and FreeBSD it returns the
            number of logical CPUs available to the current process in accordance
            with its affinity mask.

            NOTE: The return value of this method never changes after its first
            invocation. This means that changes in the process affinity mask that
            took place after this method was first invoked will not affect the
            number of worker threads in the TBB worker threads pool. **/
        static unsigned __TBB_EXPORTED_FUNC hardware_concurrency() __TBB_NOEXCEPT(true);
    private:
        native_handle_type my_handle;
#if _WIN32||_WIN64
        thread_id_type my_thread_id;
#endif // _WIN32||_WIN64

        void internal_wipe() __TBB_NOEXCEPT(true) {
            my_handle = 0;
#if _WIN32||_WIN64
            my_thread_id = 0;
#endif
        }
        void internal_move(tbb_thread_v3& x) __TBB_NOEXCEPT(true) {
            if (joinable()) detach();
            my_handle = x.my_handle;
#if _WIN32||_WIN64
            my_thread_id = x.my_thread_id;
#endif // _WIN32||_WIN64
            x.internal_wipe();
        }

        /** Runs start_routine(closure) on another thread and sets my_handle to the handle of the created thread. */
        void __TBB_EXPORTED_METHOD internal_start( __TBB_NATIVE_THREAD_ROUTINE_PTR(start_routine),
                             void* closure );
        friend void __TBB_EXPORTED_FUNC move_v3( tbb_thread_v3& t1, tbb_thread_v3& t2 );
        friend void tbb::swap( tbb_thread_v3& t1, tbb_thread_v3& t2 ) __TBB_NOEXCEPT(true);
    };

    class tbb_thread_v3::id {
        thread_id_type my_id;
        id( thread_id_type id_ ) : my_id(id_) {}

        friend class tbb_thread_v3;
    public:
        id() __TBB_NOEXCEPT(true) : my_id(0) {}

        friend bool operator==( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
        friend bool operator!=( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
        friend bool operator<( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
        friend bool operator<=( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
        friend bool operator>( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);
        friend bool operator>=( tbb_thread_v3::id x, tbb_thread_v3::id y ) __TBB_NOEXCEPT(true);

        template<class charT, class traits>
        friend std::basic_ostream<charT, traits>&
        operator<< (std::basic_ostream<charT, traits> &out,
                    tbb_thread_v3::id id)
        {
            out << id.my_id;
            return out;
        }
        friend tbb_thread_v3::id __TBB_EXPORTED_FUNC thread_get_id_v3();

        friend inline size_t tbb_hasher( const tbb_thread_v3::id& id ) {
            __TBB_STATIC_ASSERT(sizeof(id.my_id) <= sizeof(size_t), "Implementation assumes that thread_id_type fits into machine word");
            return tbb::tbb_hasher(id.my_id);
        }

        // A workaround for lack of tbb::atomic<id> (which would require id to be POD in C++03).
        friend id atomic_compare_and_swap(id& location, const id& value, const id& comparand){
            return as_atomic(location.my_id).compare_and_swap(value.my_id, comparand.my_id);
        }
    }; // tbb_thread_v3::id

    tbb_thread_v3::id tbb_thread_v3::get_id() const __TBB_NOEXCEPT(true) {
#if _WIN32||_WIN64
        return id(my_thread_id);
#else
        return id(my_handle);
#endif // _WIN32||_WIN64
    }

    void __TBB_EXPORTED_FUNC move_v3( tbb_thread_v3& t1, tbb_thread_v3& t2 );
    tbb_thread_v3::id __TBB_EXPORTED_FUNC thread_get_id_v3();
    void __TBB_EXPORTED_FUNC thread_yield_v3();
    void __TBB_EXPORTED_FUNC thread_sleep_v3(const tick_count::interval_t &i);

    inline bool operator==(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
    {
        return x.my_id == y.my_id;
    }
    inline bool operator!=(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
    {
        return x.my_id != y.my_id;
    }
    inline bool operator<(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
    {
        return x.my_id < y.my_id;
    }
    inline bool operator<=(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
    {
        return x.my_id <= y.my_id;
    }
    inline bool operator>(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
    {
        return x.my_id > y.my_id;
    }
    inline bool operator>=(tbb_thread_v3::id x, tbb_thread_v3::id y) __TBB_NOEXCEPT(true)
    {
        return x.my_id >= y.my_id;
    }

} // namespace internal;

//! Users reference thread class by name tbb_thread
__TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::thread is deprecated, use std::thread") typedef internal::tbb_thread_v3 tbb_thread;

using internal::operator==;
using internal::operator!=;
using internal::operator<;
using internal::operator>;
using internal::operator<=;
using internal::operator>=;

inline void move( tbb_thread& t1, tbb_thread& t2 ) {
    internal::move_v3(t1, t2);
}

inline void swap( internal::tbb_thread_v3& t1, internal::tbb_thread_v3& t2 )  __TBB_NOEXCEPT(true) {
    std::swap(t1.my_handle, t2.my_handle);
#if _WIN32||_WIN64
    std::swap(t1.my_thread_id, t2.my_thread_id);
#endif /* _WIN32||_WIN64 */
}

namespace this_tbb_thread {
    __TBB_DEPRECATED_IN_VERBOSE_MODE inline tbb_thread::id get_id() { return internal::thread_get_id_v3(); }
    //! Offers the operating system the opportunity to schedule another thread.
    __TBB_DEPRECATED_IN_VERBOSE_MODE inline void yield() { internal::thread_yield_v3(); }
    //! The current thread blocks at least until the time specified.
    __TBB_DEPRECATED_IN_VERBOSE_MODE inline void sleep(const tick_count::interval_t &i) {
        internal::thread_sleep_v3(i);
    }
}  // namespace this_tbb_thread

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_tbb_thread_H_include_area

#endif /* __TBB_tbb_thread_H */
