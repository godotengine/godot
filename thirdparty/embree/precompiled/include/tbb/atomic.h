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

#if !defined(__TBB_show_deprecation_message_atomic_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_atomic_H
#pragma message("TBB Warning: tbb/atomic.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_atomic_H
#define __TBB_atomic_H

#define __TBB_atomic_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include <cstddef>

#if _MSC_VER
#define __TBB_LONG_LONG __int64
#else
#define __TBB_LONG_LONG long long
#endif /* _MSC_VER */

#include "tbb_machine.h"

#if _MSC_VER && !__INTEL_COMPILER
    // Suppress overzealous compiler warnings till the end of the file
    #pragma warning (push)
    #pragma warning (disable: 4244 4267 4512)
#endif

namespace tbb {

//! Specifies memory semantics.
enum memory_semantics {
    //! Sequential consistency
    full_fence,
    //! Acquire
    acquire,
    //! Release
    release,
    //! No ordering
    relaxed
};

//! @cond INTERNAL
namespace internal {

#if __TBB_ALIGNAS_PRESENT
    #define __TBB_DECL_ATOMIC_FIELD(t,f,a) alignas(a) t f;
#elif __TBB_ATTRIBUTE_ALIGNED_PRESENT
    #define __TBB_DECL_ATOMIC_FIELD(t,f,a) t f  __attribute__ ((aligned(a)));
#elif __TBB_DECLSPEC_ALIGN_PRESENT
    #define __TBB_DECL_ATOMIC_FIELD(t,f,a) __declspec(align(a)) t f;
#else
    #error Do not know syntax for forcing alignment.
#endif

template<size_t S>
struct atomic_rep;           // Primary template declared, but never defined.

template<>
struct atomic_rep<1> {       // Specialization
    typedef int8_t word;
};
template<>
struct atomic_rep<2> {       // Specialization
    typedef int16_t word;
};
template<>
struct atomic_rep<4> {       // Specialization
#if _MSC_VER && !_WIN64
    // Work-around that avoids spurious /Wp64 warnings
    typedef intptr_t word;
#else
    typedef int32_t word;
#endif
};
#if __TBB_64BIT_ATOMICS
template<>
struct atomic_rep<8> {       // Specialization
    typedef int64_t word;
};
#endif

template<typename value_type, size_t size>
struct aligned_storage;

//the specializations are needed to please MSVC syntax of __declspec(align()) which accept _literal_ constants only
#if __TBB_ATOMIC_CTORS
    #define ATOMIC_STORAGE_PARTIAL_SPECIALIZATION(S)                  \
    template<typename value_type>                                     \
    struct aligned_storage<value_type,S> {                            \
        __TBB_DECL_ATOMIC_FIELD(value_type,my_value,S)                \
        aligned_storage() = default ;                                 \
        constexpr aligned_storage(value_type value):my_value(value){} \
    };                                                                \

#else
    #define ATOMIC_STORAGE_PARTIAL_SPECIALIZATION(S)                  \
    template<typename value_type>                                     \
    struct aligned_storage<value_type,S> {                            \
        __TBB_DECL_ATOMIC_FIELD(value_type,my_value,S)                \
    };                                                                \

#endif

template<typename value_type>
struct aligned_storage<value_type,1> {
    value_type my_value;
#if __TBB_ATOMIC_CTORS
    aligned_storage() = default ;
    constexpr aligned_storage(value_type value):my_value(value){}
#endif
};

ATOMIC_STORAGE_PARTIAL_SPECIALIZATION(2)
ATOMIC_STORAGE_PARTIAL_SPECIALIZATION(4)
#if __TBB_64BIT_ATOMICS
ATOMIC_STORAGE_PARTIAL_SPECIALIZATION(8)
#endif

template<size_t Size, memory_semantics M>
struct atomic_traits;        // Primary template declared, but not defined.

#define __TBB_DECL_FENCED_ATOMIC_PRIMITIVES(S,M)                                                         \
    template<> struct atomic_traits<S,M> {                                                               \
        typedef atomic_rep<S>::word word;                                                                \
        inline static word compare_and_swap( volatile void* location, word new_value, word comparand ) { \
            return __TBB_machine_cmpswp##S##M(location,new_value,comparand);                             \
        }                                                                                                \
        inline static word fetch_and_add( volatile void* location, word addend ) {                       \
            return __TBB_machine_fetchadd##S##M(location,addend);                                        \
        }                                                                                                \
        inline static word fetch_and_store( volatile void* location, word value ) {                      \
            return __TBB_machine_fetchstore##S##M(location,value);                                       \
        }                                                                                                \
    };

#define __TBB_DECL_ATOMIC_PRIMITIVES(S)                                                                  \
    template<memory_semantics M>                                                                         \
    struct atomic_traits<S,M> {                                                                          \
        typedef atomic_rep<S>::word word;                                                                \
        inline static word compare_and_swap( volatile void* location, word new_value, word comparand ) { \
            return __TBB_machine_cmpswp##S(location,new_value,comparand);                                \
        }                                                                                                \
        inline static word fetch_and_add( volatile void* location, word addend ) {                       \
            return __TBB_machine_fetchadd##S(location,addend);                                           \
        }                                                                                                \
        inline static word fetch_and_store( volatile void* location, word value ) {                      \
            return __TBB_machine_fetchstore##S(location,value);                                          \
        }                                                                                                \
    };

template<memory_semantics M>
struct atomic_load_store_traits;    // Primary template declaration

#define __TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(M)                      \
    template<> struct atomic_load_store_traits<M> {                     \
        template <typename T>                                           \
        inline static T load( const volatile T& location ) {            \
            return __TBB_load_##M( location );                          \
        }                                                               \
        template <typename T>                                           \
        inline static void store( volatile T& location, T value ) {     \
            __TBB_store_##M( location, value );                         \
        }                                                               \
    }

#if __TBB_USE_FENCED_ATOMICS
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(1,full_fence)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(2,full_fence)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(4,full_fence)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(1,acquire)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(2,acquire)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(4,acquire)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(1,release)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(2,release)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(4,release)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(1,relaxed)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(2,relaxed)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(4,relaxed)
#if __TBB_64BIT_ATOMICS
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(8,full_fence)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(8,acquire)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(8,release)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(8,relaxed)
#endif
#else /* !__TBB_USE_FENCED_ATOMICS */
__TBB_DECL_ATOMIC_PRIMITIVES(1)
__TBB_DECL_ATOMIC_PRIMITIVES(2)
__TBB_DECL_ATOMIC_PRIMITIVES(4)
#if __TBB_64BIT_ATOMICS
__TBB_DECL_ATOMIC_PRIMITIVES(8)
#endif
#endif /* !__TBB_USE_FENCED_ATOMICS */

__TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(full_fence);
__TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(acquire);
__TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(release);
__TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(relaxed);

//! Additive inverse of 1 for type T.
/** Various compilers issue various warnings if -1 is used with various integer types.
    The baroque expression below avoids all the warnings (we hope). */
#define __TBB_MINUS_ONE(T) (T(T(0)-T(1)))

//! Base class that provides basic functionality for atomic<T> without fetch_and_add.
/** Works for any type T that has the same size as an integral type, has a trivial constructor/destructor,
    and can be copied/compared by memcpy/memcmp. */
template<typename T>
struct atomic_impl {
protected:
    aligned_storage<T,sizeof(T)> my_storage;
private:
    //TODO: rechecks on recent versions of gcc if union is still the _only_ way to do a conversion without warnings
    //! Union type used to convert type T to underlying integral type.
    template<typename value_type>
    union converter {
        typedef typename atomic_rep<sizeof(value_type)>::word bits_type;
        converter(){}
        converter(value_type a_value) : value(a_value) {}
        value_type value;
        bits_type bits;
    };

    template<typename value_t>
    static typename converter<value_t>::bits_type to_bits(value_t value){
        return converter<value_t>(value).bits;
    }
    template<typename value_t>
    static value_t to_value(typename converter<value_t>::bits_type bits){
        converter<value_t> u;
        u.bits = bits;
        return u.value;
    }

    template<typename value_t>
    union ptr_converter;            //Primary template declared, but never defined.

    template<typename value_t>
    union ptr_converter<value_t *> {
        ptr_converter(){}
        ptr_converter(value_t* a_value) : value(a_value) {}
        value_t* value;
        uintptr_t bits;
    };
    //TODO: check if making to_bits accepting reference (thus unifying it with to_bits_ref)
    //does not hurt performance
    template<typename value_t>
    static typename converter<value_t>::bits_type & to_bits_ref(value_t& value){
        //TODO: this #ifdef is temporary workaround, as union conversion seems to fail
        //on suncc for 64 bit types for 32 bit target
        #if !__SUNPRO_CC
            return *(typename converter<value_t>::bits_type*)ptr_converter<value_t*>(&value).bits;
        #else
            return *(typename converter<value_t>::bits_type*)(&value);
        #endif
    }


public:
    typedef T value_type;

#if __TBB_ATOMIC_CTORS
    atomic_impl() = default ;
    constexpr atomic_impl(value_type value):my_storage(value){}
#endif
    template<memory_semantics M>
    value_type fetch_and_store( value_type value ) {
          return to_value<value_type>(
                  internal::atomic_traits<sizeof(value_type),M>::fetch_and_store( &my_storage.my_value, to_bits(value) )
          );
    }

    value_type fetch_and_store( value_type value ) {
        return fetch_and_store<full_fence>(value);
    }

    template<memory_semantics M>
    value_type compare_and_swap( value_type value, value_type comparand ) {
        return to_value<value_type>(
                internal::atomic_traits<sizeof(value_type),M>::compare_and_swap( &my_storage.my_value, to_bits(value), to_bits(comparand) )
        );
    }

    value_type compare_and_swap( value_type value, value_type comparand ) {
        return compare_and_swap<full_fence>(value,comparand);
    }

    operator value_type() const volatile {                // volatile qualifier here for backwards compatibility
        return  to_value<value_type>(
                __TBB_load_with_acquire( to_bits_ref(my_storage.my_value) )
        );
    }

    template<memory_semantics M>
    value_type load () const {
        return to_value<value_type>(
                internal::atomic_load_store_traits<M>::load( to_bits_ref(my_storage.my_value) )
        );
    }

    value_type load () const {
        return load<acquire>();
    }

    template<memory_semantics M>
    void store ( value_type value ) {
        internal::atomic_load_store_traits<M>::store( to_bits_ref(my_storage.my_value), to_bits(value));
    }

    void store ( value_type value ) {
        store<release>( value );
    }

protected:
    value_type store_with_release( value_type rhs ) {
       //TODO: unify with store<release>
        __TBB_store_with_release( to_bits_ref(my_storage.my_value), to_bits(rhs) );
        return rhs;
    }
};

//! Base class that provides basic functionality for atomic<T> with fetch_and_add.
/** I is the underlying type.
    D is the difference type.
    StepType should be char if I is an integral type, and T if I is a T*. */
template<typename I, typename D, typename StepType>
struct atomic_impl_with_arithmetic: atomic_impl<I> {
public:
    typedef I value_type;
#if    __TBB_ATOMIC_CTORS
    atomic_impl_with_arithmetic() = default ;
    constexpr atomic_impl_with_arithmetic(value_type value): atomic_impl<I>(value){}
#endif
    template<memory_semantics M>
    value_type fetch_and_add( D addend ) {
        return value_type(internal::atomic_traits<sizeof(value_type),M>::fetch_and_add( &this->my_storage.my_value, addend*sizeof(StepType) ));
    }

    value_type fetch_and_add( D addend ) {
        return fetch_and_add<full_fence>(addend);
    }

    template<memory_semantics M>
    value_type fetch_and_increment() {
        return fetch_and_add<M>(1);
    }

    value_type fetch_and_increment() {
        return fetch_and_add(1);
    }

    template<memory_semantics M>
    value_type fetch_and_decrement() {
        return fetch_and_add<M>(__TBB_MINUS_ONE(D));
    }

    value_type fetch_and_decrement() {
        return fetch_and_add(__TBB_MINUS_ONE(D));
    }

public:
    value_type operator+=( D value ) {
        return fetch_and_add(value)+value;
    }

    value_type operator-=( D value ) {
        // Additive inverse of value computed using binary minus,
        // instead of unary minus, for sake of avoiding compiler warnings.
        return operator+=(D(0)-value);
    }

    value_type operator++() {
        return fetch_and_add(1)+1;
    }

    value_type operator--() {
        return fetch_and_add(__TBB_MINUS_ONE(D))-1;
    }

    value_type operator++(int) {
        return fetch_and_add(1);
    }

    value_type operator--(int) {
        return fetch_and_add(__TBB_MINUS_ONE(D));
    }
};

} /* Internal */
//! @endcond

//! Primary template for atomic.
/** See the Reference for details.
    @ingroup synchronization */
template<typename T>
struct __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::atomic is deprecated, use std::atomic")
atomic: internal::atomic_impl<T> {
#if __TBB_ATOMIC_CTORS
    atomic() = default;
    constexpr atomic(T arg): internal::atomic_impl<T>(arg) {}
    constexpr atomic<T>(const atomic<T>& rhs): internal::atomic_impl<T>(rhs) {}
#endif
    T operator=( T rhs ) {
        // "this" required here in strict ISO C++ because store_with_release is a dependent name
        return this->store_with_release(rhs);
    }
    atomic<T>& operator=( const atomic<T>& rhs ) {this->store_with_release(rhs); return *this;}
};

#if __TBB_ATOMIC_CTORS
    #define __TBB_DECL_ATOMIC(T)                                                                    \
        template<> struct __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::atomic is deprecated, use std::atomic")  \
        atomic<T>: internal::atomic_impl_with_arithmetic<T,T,char> {                                \
            atomic() = default;                                                                     \
            constexpr atomic(T arg): internal::atomic_impl_with_arithmetic<T,T,char>(arg) {}        \
            constexpr atomic<T>(const atomic<T>& rhs):                                              \
                      internal::atomic_impl_with_arithmetic<T,T,char>(rhs) {}                       \
                                                                                                    \
            T operator=( T rhs ) {return store_with_release(rhs);}                                  \
            atomic<T>& operator=( const atomic<T>& rhs ) {store_with_release(rhs); return *this;}   \
        };
#else
    #define __TBB_DECL_ATOMIC(T)                                                                    \
        template<> struct __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::atomic is deprecated, use std::atomic")  \
        atomic<T>: internal::atomic_impl_with_arithmetic<T,T,char> {                                \
            T operator=( T rhs ) {return store_with_release(rhs);}                                  \
            atomic<T>& operator=( const atomic<T>& rhs ) {store_with_release(rhs); return *this;}   \
        };
#endif

#if __TBB_64BIT_ATOMICS
//TODO: consider adding non-default (and atomic) copy constructor for 32bit platform
__TBB_DECL_ATOMIC(__TBB_LONG_LONG)
__TBB_DECL_ATOMIC(unsigned __TBB_LONG_LONG)
#else
// test_atomic will verify that sizeof(long long)==8
#endif
__TBB_DECL_ATOMIC(long)
__TBB_DECL_ATOMIC(unsigned long)

#if _MSC_VER && !_WIN64
#if __TBB_ATOMIC_CTORS
/* Special version of __TBB_DECL_ATOMIC that avoids gratuitous warnings from cl /Wp64 option.
   It is identical to __TBB_DECL_ATOMIC(unsigned) except that it replaces operator=(T)
   with an operator=(U) that explicitly converts the U to a T.  Types T and U should be
   type synonyms on the platform.  Type U should be the wider variant of T from the
   perspective of /Wp64. */
#define __TBB_DECL_ATOMIC_ALT(T,U) \
    template<> struct __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::atomic is deprecated, use std::atomic") \
    atomic<T>: internal::atomic_impl_with_arithmetic<T,T,char> {                               \
        atomic() = default ;                                                                   \
        constexpr atomic(T arg): internal::atomic_impl_with_arithmetic<T,T,char>(arg) {}       \
        constexpr atomic<T>(const atomic<T>& rhs):                                             \
                  internal::atomic_impl_with_arithmetic<T,T,char>(rhs) {}                      \
                                                                                               \
        T operator=( U rhs ) {return store_with_release(T(rhs));}                              \
        atomic<T>& operator=( const atomic<T>& rhs ) {store_with_release(rhs); return *this;}  \
    };
#else
#define __TBB_DECL_ATOMIC_ALT(T,U) \
    template<> struct __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::atomic is deprecated, use std::atomic") \
    atomic<T>: internal::atomic_impl_with_arithmetic<T,T,char> {                               \
        T operator=( U rhs ) {return store_with_release(T(rhs));}                              \
        atomic<T>& operator=( const atomic<T>& rhs ) {store_with_release(rhs); return *this;}  \
    };
#endif
__TBB_DECL_ATOMIC_ALT(unsigned,size_t)
__TBB_DECL_ATOMIC_ALT(int,ptrdiff_t)
#else
__TBB_DECL_ATOMIC(unsigned)
__TBB_DECL_ATOMIC(int)
#endif /* _MSC_VER && !_WIN64 */

__TBB_DECL_ATOMIC(unsigned short)
__TBB_DECL_ATOMIC(short)
__TBB_DECL_ATOMIC(char)
__TBB_DECL_ATOMIC(signed char)
__TBB_DECL_ATOMIC(unsigned char)

#if !_MSC_VER || defined(_NATIVE_WCHAR_T_DEFINED)
__TBB_DECL_ATOMIC(wchar_t)
#endif /* _MSC_VER||!defined(_NATIVE_WCHAR_T_DEFINED) */

//! Specialization for atomic<T*> with arithmetic and operator->.
template<typename T> struct __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::atomic is deprecated, use std::atomic")
atomic<T*>: internal::atomic_impl_with_arithmetic<T*,ptrdiff_t,T> {
#if __TBB_ATOMIC_CTORS
    atomic() = default ;
    constexpr atomic(T* arg): internal::atomic_impl_with_arithmetic<T*,ptrdiff_t,T>(arg) {}
    constexpr atomic(const atomic<T*>& rhs): internal::atomic_impl_with_arithmetic<T*,ptrdiff_t,T>(rhs) {}
#endif
    T* operator=( T* rhs ) {
        // "this" required here in strict ISO C++ because store_with_release is a dependent name
        return this->store_with_release(rhs);
    }
    atomic<T*>& operator=( const atomic<T*>& rhs ) {
        this->store_with_release(rhs); return *this;
    }
    T* operator->() const {
        return (*this);
    }
};

//! Specialization for atomic<void*>, for sake of not allowing arithmetic or operator->.
template<> struct __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::atomic is deprecated, use std::atomic")
atomic<void*>: internal::atomic_impl<void*> {
#if __TBB_ATOMIC_CTORS
    atomic() = default ;
    constexpr atomic(void* arg): internal::atomic_impl<void*>(arg) {}
    constexpr atomic(const atomic<void*>& rhs): internal::atomic_impl<void*>(rhs) {}
#endif
    void* operator=( void* rhs ) {
        // "this" required here in strict ISO C++ because store_with_release is a dependent name
        return this->store_with_release(rhs);
    }
    atomic<void*>& operator=( const atomic<void*>& rhs ) {
        this->store_with_release(rhs); return *this;
    }
};

// Helpers to workaround ugly syntax of calling template member function of a
// template class with template argument dependent on template parameters.

template <memory_semantics M, typename T>
T load ( const atomic<T>& a ) { return a.template load<M>(); }

template <memory_semantics M, typename T>
void store ( atomic<T>& a, T value ) { a.template store<M>(value); }

namespace interface6{
//! Make an atomic for use in an initialization (list), as an alternative to zero-initialization or normal assignment.
template<typename T>
atomic<T> make_atomic(T t) {
    atomic<T> a;
    store<relaxed>(a,t);
    return a;
}
}
using interface6::make_atomic;

namespace internal {
template<memory_semantics M, typename T >
void swap(atomic<T> & lhs, atomic<T> & rhs){
    T tmp = load<M>(lhs);
    store<M>(lhs,load<M>(rhs));
    store<M>(rhs,tmp);
}

// only to aid in the gradual conversion of ordinary variables to proper atomics
template<typename T>
inline atomic<T>& as_atomic( T& t ) {
    return (atomic<T>&)t;
}
} // namespace tbb::internal

} // namespace tbb

#if _MSC_VER && !__INTEL_COMPILER
    #pragma warning (pop)
#endif // warnings are restored

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_atomic_H_include_area

#endif /* __TBB_atomic_H */
