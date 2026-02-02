//
// Copyright (c) 2014-2018 Martin Moene
//
// https://github.com/martinmoene/optional-lite
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifndef NONSTD_OPTIONAL_LITE_HPP
#define NONSTD_OPTIONAL_LITE_HPP

#define optional_lite_MAJOR  3
#define optional_lite_MINOR  3
#define optional_lite_PATCH  0

#define optional_lite_VERSION  optional_STRINGIFY(optional_lite_MAJOR) "." optional_STRINGIFY(optional_lite_MINOR) "." optional_STRINGIFY(optional_lite_PATCH)

#define optional_STRINGIFY(  x )  optional_STRINGIFY_( x )
#define optional_STRINGIFY_( x )  #x


// optional-lite configuration:

#define optional_OPTIONAL_DEFAULT  0
#define optional_OPTIONAL_NONSTD   1
#define optional_OPTIONAL_STD      2

// TinyUSDZ mod. Force use nonstd implementation even on C++17 to avoid possible linkage issue.
#if defined( optional_CONFIG_SELECT_OPTIONAL )
#undef optional_CONFIG_SELECT_OPTIONAL
#endif
#define optional_CONFIG_SELECT_OPTIONAL (optional_OPTIONAL_NONSTD)

// tweak header support:

#ifdef __has_include
# if __has_include(<nonstd/optional.tweak.hpp>)
#  include <nonstd/optional.tweak.hpp>
# endif
#define optional_HAVE_TWEAK_HEADER  1
#else
#define optional_HAVE_TWEAK_HEADER  0
//# pragma message("optional.hpp: Note: Tweak header not supported.")
#endif

// optional selection and configuration:

#if !defined( optional_CONFIG_SELECT_OPTIONAL )
# define optional_CONFIG_SELECT_OPTIONAL  ( optional_HAVE_STD_OPTIONAL ? optional_OPTIONAL_STD : optional_OPTIONAL_NONSTD )
#endif

// Control presence of exception handling (try and auto discover):

// TinyUSDZ mod. Force disable exception.
#if defined( optional_CONFIG_NO_EXCEPTIONS )
#undef optional_CONFIG_NO_EXCEPTIONS
#define optional_CONFIG_NO_EXCEPTIONS 1
#endif

#ifndef optional_CONFIG_NO_EXCEPTIONS
# if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#  define optional_CONFIG_NO_EXCEPTIONS  0
# else
#  define optional_CONFIG_NO_EXCEPTIONS  1
# endif
#endif

// C++ language version detection (C++20 is speculative):
// Note: VC14.0/1900 (VS2015) lacks too much from C++14.

#ifndef   optional_CPLUSPLUS
# if defined(_MSVC_LANG ) && !defined(__clang__)
#  define optional_CPLUSPLUS  (_MSC_VER == 1900 ? 201103L : _MSVC_LANG )
# else
#  define optional_CPLUSPLUS  __cplusplus
# endif
#endif

#define optional_CPP98_OR_GREATER  ( optional_CPLUSPLUS >= 199711L )
#define optional_CPP11_OR_GREATER  ( optional_CPLUSPLUS >= 201103L )
#define optional_CPP11_OR_GREATER_ ( optional_CPLUSPLUS >= 201103L )
#define optional_CPP14_OR_GREATER  ( optional_CPLUSPLUS >= 201402L )
#define optional_CPP17_OR_GREATER  ( optional_CPLUSPLUS >= 201703L )
#define optional_CPP20_OR_GREATER  ( optional_CPLUSPLUS >= 202000L )

// C++ language version (represent 98 as 3):

#define optional_CPLUSPLUS_V  ( optional_CPLUSPLUS / 100 - (optional_CPLUSPLUS > 200000 ? 2000 : 1994) )

// Use C++17 std::optional if available and requested:

#if optional_CPP17_OR_GREATER && defined(__has_include )
# if __has_include( <optional> )
#  define optional_HAVE_STD_OPTIONAL  1
# else
#  define optional_HAVE_STD_OPTIONAL  0
# endif
#else
# define  optional_HAVE_STD_OPTIONAL  0
#endif

#define optional_USES_STD_OPTIONAL  ( (optional_CONFIG_SELECT_OPTIONAL == optional_OPTIONAL_STD) || ((optional_CONFIG_SELECT_OPTIONAL == optional_OPTIONAL_DEFAULT) && optional_HAVE_STD_OPTIONAL) )

//
// in_place: code duplicated in any-lite, expected-lite, optional-lite, value-ptr-lite, variant-lite:
//

#ifndef nonstd_lite_HAVE_IN_PLACE_TYPES
#define nonstd_lite_HAVE_IN_PLACE_TYPES  1

// C++17 std::in_place in <utility>:

#if optional_CPP17_OR_GREATER

#include <utility>

namespace nonstd {

using std::in_place;
using std::in_place_type;
using std::in_place_index;
using std::in_place_t;
using std::in_place_type_t;
using std::in_place_index_t;

#define nonstd_lite_in_place_t(      T)  std::in_place_t
#define nonstd_lite_in_place_type_t( T)  std::in_place_type_t<T>
#define nonstd_lite_in_place_index_t(K)  std::in_place_index_t<K>

#define nonstd_lite_in_place(      T)    std::in_place_t{}
#define nonstd_lite_in_place_type( T)    std::in_place_type_t<T>{}
#define nonstd_lite_in_place_index(K)    std::in_place_index_t<K>{}

} // namespace nonstd

#else // optional_CPP17_OR_GREATER

#include <cstddef>

namespace nonstd {
namespace detail {

template< class T >
struct in_place_type_tag {};

template< std::size_t K >
struct in_place_index_tag {};

} // namespace detail

struct in_place_t {};

template< class T >
inline in_place_t in_place( detail::in_place_type_tag<T> /*unused*/ = detail::in_place_type_tag<T>() )
{
    return in_place_t();
}

template< std::size_t K >
inline in_place_t in_place( detail::in_place_index_tag<K> /*unused*/ = detail::in_place_index_tag<K>() )
{
    return in_place_t();
}

template< class T >
inline in_place_t in_place_type( detail::in_place_type_tag<T> /*unused*/ = detail::in_place_type_tag<T>() )
{
    return in_place_t();
}

template< std::size_t K >
inline in_place_t in_place_index( detail::in_place_index_tag<K> /*unused*/ = detail::in_place_index_tag<K>() )
{
    return in_place_t();
}

// mimic templated typedef:

#define nonstd_lite_in_place_t(      T)  nonstd::in_place_t(&)( nonstd::detail::in_place_type_tag<T>  )
#define nonstd_lite_in_place_type_t( T)  nonstd::in_place_t(&)( nonstd::detail::in_place_type_tag<T>  )
#define nonstd_lite_in_place_index_t(K)  nonstd::in_place_t(&)( nonstd::detail::in_place_index_tag<K> )

#define nonstd_lite_in_place(      T)    nonstd::in_place_type<T>
#define nonstd_lite_in_place_type( T)    nonstd::in_place_type<T>
#define nonstd_lite_in_place_index(K)    nonstd::in_place_index<K>

} // namespace nonstd

#endif // optional_CPP17_OR_GREATER
#endif // nonstd_lite_HAVE_IN_PLACE_TYPES

//
// Using std::optional:
//

#if optional_USES_STD_OPTIONAL

#include <optional>

namespace nonstd {

    using std::optional;
    using std::bad_optional_access;
    using std::hash;

    using std::nullopt;
    using std::nullopt_t;

    using std::operator==;
    using std::operator!=;
    using std::operator<;
    using std::operator<=;
    using std::operator>;
    using std::operator>=;
    using std::make_optional;
    using std::swap;
}

#else // optional_USES_STD_OPTIONAL

#include <cassert>
#include <utility>

// optional-lite alignment configuration:

#ifndef  optional_CONFIG_MAX_ALIGN_HACK
# define optional_CONFIG_MAX_ALIGN_HACK  0
#endif

#ifndef  optional_CONFIG_ALIGN_AS
// no default, used in #if defined()
#endif

#ifndef  optional_CONFIG_ALIGN_AS_FALLBACK
# define optional_CONFIG_ALIGN_AS_FALLBACK  double
#endif

// Compiler warning suppression:

#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wundef"
#elif defined(__GNUC__)
# pragma GCC   diagnostic push
# pragma GCC   diagnostic ignored "-Wundef"
#elif defined(_MSC_VER )
# pragma warning( push )
#endif

// half-open range [lo..hi):
#define optional_BETWEEN( v, lo, hi ) ( (lo) <= (v) && (v) < (hi) )

// Compiler versions:
//
// MSVC++  6.0  _MSC_VER == 1200  optional_COMPILER_MSVC_VERSION ==  60  (Visual Studio 6.0)
// MSVC++  7.0  _MSC_VER == 1300  optional_COMPILER_MSVC_VERSION ==  70  (Visual Studio .NET 2002)
// MSVC++  7.1  _MSC_VER == 1310  optional_COMPILER_MSVC_VERSION ==  71  (Visual Studio .NET 2003)
// MSVC++  8.0  _MSC_VER == 1400  optional_COMPILER_MSVC_VERSION ==  80  (Visual Studio 2005)
// MSVC++  9.0  _MSC_VER == 1500  optional_COMPILER_MSVC_VERSION ==  90  (Visual Studio 2008)
// MSVC++ 10.0  _MSC_VER == 1600  optional_COMPILER_MSVC_VERSION == 100  (Visual Studio 2010)
// MSVC++ 11.0  _MSC_VER == 1700  optional_COMPILER_MSVC_VERSION == 110  (Visual Studio 2012)
// MSVC++ 12.0  _MSC_VER == 1800  optional_COMPILER_MSVC_VERSION == 120  (Visual Studio 2013)
// MSVC++ 14.0  _MSC_VER == 1900  optional_COMPILER_MSVC_VERSION == 140  (Visual Studio 2015)
// MSVC++ 14.1  _MSC_VER >= 1910  optional_COMPILER_MSVC_VERSION == 141  (Visual Studio 2017)
// MSVC++ 14.2  _MSC_VER >= 1920  optional_COMPILER_MSVC_VERSION == 142  (Visual Studio 2019)

#if defined(_MSC_VER ) && !defined(__clang__)
# define optional_COMPILER_MSVC_VER      (_MSC_VER )
# define optional_COMPILER_MSVC_VERSION  (_MSC_VER / 10 - 10 * ( 5 + (_MSC_VER < 1900 ) ) )
#else
# define optional_COMPILER_MSVC_VER      0
# define optional_COMPILER_MSVC_VERSION  0
#endif

#define optional_COMPILER_VERSION( major, minor, patch )  ( 10 * (10 * (major) + (minor) ) + (patch) )

#if defined(__GNUC__) && !defined(__clang__)
# define optional_COMPILER_GNUC_VERSION   optional_COMPILER_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
# define optional_COMPILER_GNUC_VERSION   0
#endif

#if defined(__clang__)
# define optional_COMPILER_CLANG_VERSION  optional_COMPILER_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
# define optional_COMPILER_CLANG_VERSION  0
#endif

#if optional_BETWEEN(optional_COMPILER_MSVC_VERSION, 70, 140 )
# pragma warning( disable: 4345 )   // initialization behavior changed
#endif

#if optional_BETWEEN(optional_COMPILER_MSVC_VERSION, 70, 150 )
# pragma warning( disable: 4814 )   // in C++14 'constexpr' will not imply 'const'
#endif

// Presence of language and library features:

#define optional_HAVE(FEATURE) ( optional_HAVE_##FEATURE )

#ifdef _HAS_CPP0X
# define optional_HAS_CPP0X  _HAS_CPP0X
#else
# define optional_HAS_CPP0X  0
#endif

// Unless defined otherwise below, consider VC14 as C++11 for optional-lite:

#if optional_COMPILER_MSVC_VER >= 1900
# undef  optional_CPP11_OR_GREATER
# define optional_CPP11_OR_GREATER  1
#endif

#define optional_CPP11_90   (optional_CPP11_OR_GREATER_ || optional_COMPILER_MSVC_VER >= 1500)
#define optional_CPP11_100  (optional_CPP11_OR_GREATER_ || optional_COMPILER_MSVC_VER >= 1600)
#define optional_CPP11_110  (optional_CPP11_OR_GREATER_ || optional_COMPILER_MSVC_VER >= 1700)
#define optional_CPP11_120  (optional_CPP11_OR_GREATER_ || optional_COMPILER_MSVC_VER >= 1800)
#define optional_CPP11_140  (optional_CPP11_OR_GREATER_ || optional_COMPILER_MSVC_VER >= 1900)
#define optional_CPP11_141  (optional_CPP11_OR_GREATER_ || optional_COMPILER_MSVC_VER >= 1910)

#define optional_CPP14_000  (optional_CPP14_OR_GREATER)
#define optional_CPP17_000  (optional_CPP17_OR_GREATER)

// gcc >= 4.9, msvc >= vc14.1 (vs17):
#define optional_CPP11_140_G490     ((optional_CPP11_OR_GREATER_ && optional_COMPILER_GNUC_VERSION >= 490) || (optional_COMPILER_MSVC_VER >= 1910))

// clang >= 3.5, msvc >= vc11 (vs12):
#define optional_CPP11_110_C350     ( optional_CPP11_110 && !optional_BETWEEN( optional_COMPILER_CLANG_VERSION, 1, 350 ) )

// clang >= 3.5, gcc >= 5.0, msvc >= vc11 (vs12):
#define optional_CPP11_110_C350_G500 \
    (  optional_CPP11_110 && \
    !( optional_BETWEEN( optional_COMPILER_CLANG_VERSION, 1, 350 ) \
    || optional_BETWEEN( optional_COMPILER_GNUC_VERSION , 1, 500 ) ) )

// Presence of C++11 language features:

#define optional_HAVE_CONSTEXPR_11      optional_CPP11_140
#define optional_HAVE_IS_DEFAULT        optional_CPP11_140
#define optional_HAVE_NOEXCEPT          optional_CPP11_140
#define optional_HAVE_NULLPTR           optional_CPP11_100
#define optional_HAVE_REF_QUALIFIER     optional_CPP11_140_G490
#define optional_HAVE_INITIALIZER_LIST  optional_CPP11_140

// Presence of C++14 language features:

#define optional_HAVE_CONSTEXPR_14      optional_CPP14_000

// Presence of C++17 language features:

#define optional_HAVE_NODISCARD         optional_CPP17_000

// Presence of C++ library features:

#define optional_HAVE_CONDITIONAL       optional_CPP11_120
#define optional_HAVE_REMOVE_CV         optional_CPP11_120
#define optional_HAVE_TYPE_TRAITS       optional_CPP11_90

#define optional_HAVE_TR1_TYPE_TRAITS   (!! optional_COMPILER_GNUC_VERSION )
#define optional_HAVE_TR1_ADD_POINTER   (!! optional_COMPILER_GNUC_VERSION )

#define optional_HAVE_IS_ASSIGNABLE                     optional_CPP11_110_C350
#define optional_HAVE_IS_MOVE_CONSTRUCTIBLE             optional_CPP11_110_C350
#define optional_HAVE_IS_NOTHROW_MOVE_ASSIGNABLE        optional_CPP11_110_C350
#define optional_HAVE_IS_NOTHROW_MOVE_CONSTRUCTIBLE     optional_CPP11_110_C350
#define optional_HAVE_IS_TRIVIALLY_COPY_CONSTRUCTIBLE   optional_CPP11_110_C350_G500
#define optional_HAVE_IS_TRIVIALLY_MOVE_CONSTRUCTIBLE   optional_CPP11_110_C350_G500

// C++ feature usage:

#if optional_HAVE( CONSTEXPR_11 )
# define optional_constexpr  constexpr
#else
# define optional_constexpr  /*constexpr*/
#endif

#if optional_HAVE( IS_DEFAULT )
# define optional_is_default  = default;
#else
# define optional_is_default  {}
#endif

#if optional_HAVE( CONSTEXPR_14 )
# define optional_constexpr14  constexpr
#else
# define optional_constexpr14  /*constexpr*/
#endif

#if optional_HAVE( NODISCARD )
# define optional_nodiscard  [[nodiscard]]
#else
# define optional_nodiscard  /*[[nodiscard]]*/
#endif

#if optional_HAVE( NOEXCEPT )
# define optional_noexcept  noexcept
#else
# define optional_noexcept  /*noexcept*/
#endif

#if optional_HAVE( NULLPTR )
# define optional_nullptr  nullptr
#else
# define optional_nullptr  NULL
#endif

#if optional_HAVE( REF_QUALIFIER )
// NOLINTNEXTLINE( bugprone-macro-parentheses )
# define optional_ref_qual  &
# define optional_refref_qual  &&
#else
# define optional_ref_qual  /*&*/
# define optional_refref_qual  /*&&*/
#endif

// additional includes:

#if optional_CONFIG_NO_EXCEPTIONS
// already included: <cassert>
#else
# include <stdexcept>
#endif

#if optional_CPP11_OR_GREATER
# include <functional>
#endif

#if optional_HAVE( INITIALIZER_LIST )
# include <initializer_list>
#endif

#if optional_HAVE( TYPE_TRAITS )
# include <type_traits>
#elif optional_HAVE( TR1_TYPE_TRAITS )
# include <tr1/type_traits>
#endif

// Method enabling

#if optional_CPP11_OR_GREATER

#define optional_REQUIRES_0(...) \
    template< bool B = (__VA_ARGS__), typename std::enable_if<B, int>::type = 0 >

#define optional_REQUIRES_T(...) \
    , typename std::enable_if< (__VA_ARGS__), int >::type = 0

#define optional_REQUIRES_R(R, ...) \
    typename std::enable_if< (__VA_ARGS__), R>::type

#define optional_REQUIRES_A(...) \
    , typename std::enable_if< (__VA_ARGS__), void*>::type = nullptr

#endif

//
// optional:
//

namespace nonstd { namespace optional_lite {

namespace std11 {

template< class T, T v > struct integral_constant { enum { value = v }; };
template< bool B       > struct bool_constant : integral_constant<bool, B>{};

typedef bool_constant< true  > true_type;
typedef bool_constant< false > false_type;

#if optional_CPP11_OR_GREATER
    using std::move;
#else
    template< typename T > T & move( T & t ) { return t; }
#endif

#if optional_HAVE( CONDITIONAL )
    using std::conditional;
#else
    template< bool B, typename T, typename F > struct conditional              { typedef T type; };
    template<         typename T, typename F > struct conditional<false, T, F> { typedef F type; };
#endif // optional_HAVE_CONDITIONAL

#if optional_HAVE( IS_ASSIGNABLE )
    using std::is_assignable;
#else
    template< class T, class U > struct is_assignable : std11::true_type{};
#endif

#if optional_HAVE( IS_MOVE_CONSTRUCTIBLE )
    using std::is_move_constructible;
#else
    template< class T > struct is_move_constructible : std11::true_type{};
#endif

#if optional_HAVE( IS_NOTHROW_MOVE_ASSIGNABLE )
    using std::is_nothrow_move_assignable;
#else
    template< class T > struct is_nothrow_move_assignable : std11::true_type{};
#endif

#if optional_HAVE( IS_NOTHROW_MOVE_CONSTRUCTIBLE )
    using std::is_nothrow_move_constructible;
#else
    template< class T > struct is_nothrow_move_constructible : std11::true_type{};
#endif

#if optional_HAVE( IS_TRIVIALLY_COPY_CONSTRUCTIBLE )
    using std::is_trivially_copy_constructible;
#else
    template< class T > struct is_trivially_copy_constructible : std11::true_type{};
#endif

#if optional_HAVE( IS_TRIVIALLY_MOVE_CONSTRUCTIBLE )
    using std::is_trivially_move_constructible;
#else
    template< class T > struct is_trivially_move_constructible : std11::true_type{};
#endif

} // namespace std11

#if optional_CPP11_OR_GREATER

/// type traits C++17:

namespace std17 {

#if optional_CPP17_OR_GREATER

using std::is_swappable;
using std::is_nothrow_swappable;

#elif optional_CPP11_OR_GREATER

namespace detail {

using std::swap;

struct is_swappable
{
    template< typename T, typename = decltype( swap( std::declval<T&>(), std::declval<T&>() ) ) >
    static std11::true_type test( int /*unused*/ );

    template< typename >
    static std11::false_type test(...);
};

struct is_nothrow_swappable
{
    // wrap noexcept(expr) in separate function as work-around for VC140 (VS2015):

    template< typename T >
    static constexpr bool satisfies()
    {
        return noexcept( swap( std::declval<T&>(), std::declval<T&>() ) );
    }

    template< typename T >
    static auto test( int /*unused*/ ) -> std11::integral_constant<bool, satisfies<T>()>{}

    template< typename >
    static auto test(...) -> std11::false_type;
};

} // namespace detail

// is [nothow] swappable:

template< typename T >
struct is_swappable : decltype( detail::is_swappable::test<T>(0) ){};

template< typename T >
struct is_nothrow_swappable : decltype( detail::is_nothrow_swappable::test<T>(0) ){};

#endif // optional_CPP17_OR_GREATER

} // namespace std17

/// type traits C++20:

namespace std20 {

template< typename T >
struct remove_cvref
{
    typedef typename std::remove_cv< typename std::remove_reference<T>::type >::type type;
};

} // namespace std20

#endif // optional_CPP11_OR_GREATER

/// class optional

template< typename T >
class optional;

namespace detail {

// C++11 emulation:

struct nulltype{};

template< typename Head, typename Tail >
struct typelist
{
    typedef Head head;
    typedef Tail tail;
};

#if optional_CONFIG_MAX_ALIGN_HACK

// Max align, use most restricted type for alignment:

#define optional_UNIQUE(  name )       optional_UNIQUE2( name, __LINE__ )
#define optional_UNIQUE2( name, line ) optional_UNIQUE3( name, line )
#define optional_UNIQUE3( name, line ) name ## line

#define optional_ALIGN_TYPE( type ) \
    type optional_UNIQUE( _t ); struct_t< type > optional_UNIQUE( _st )

template< typename T >
struct struct_t { T _; };

union max_align_t
{
    optional_ALIGN_TYPE( char );
    optional_ALIGN_TYPE( short int );
    optional_ALIGN_TYPE( int );
    optional_ALIGN_TYPE( long int  );
    optional_ALIGN_TYPE( float  );
    optional_ALIGN_TYPE( double );
    optional_ALIGN_TYPE( long double );
    optional_ALIGN_TYPE( char * );
    optional_ALIGN_TYPE( short int * );
    optional_ALIGN_TYPE( int *  );
    optional_ALIGN_TYPE( long int * );
    optional_ALIGN_TYPE( float * );
    optional_ALIGN_TYPE( double * );
    optional_ALIGN_TYPE( long double * );
    optional_ALIGN_TYPE( void * );

#ifdef HAVE_LONG_LONG
    optional_ALIGN_TYPE( long long );
#endif

    struct Unknown;

    Unknown ( * optional_UNIQUE(_) )( Unknown );
    Unknown * Unknown::* optional_UNIQUE(_);
    Unknown ( Unknown::* optional_UNIQUE(_) )( Unknown );

    struct_t< Unknown ( * )( Unknown)         > optional_UNIQUE(_);
    struct_t< Unknown * Unknown::*            > optional_UNIQUE(_);
    struct_t< Unknown ( Unknown::* )(Unknown) > optional_UNIQUE(_);
};

#undef optional_UNIQUE
#undef optional_UNIQUE2
#undef optional_UNIQUE3

#undef optional_ALIGN_TYPE

#elif defined( optional_CONFIG_ALIGN_AS ) // optional_CONFIG_MAX_ALIGN_HACK

// Use user-specified type for alignment:

#define optional_ALIGN_AS( unused ) \
    optional_CONFIG_ALIGN_AS

#else // optional_CONFIG_MAX_ALIGN_HACK

// Determine POD type to use for alignment:

#define optional_ALIGN_AS( to_align ) \
    typename type_of_size< alignment_types, alignment_of< to_align >::value >::type

template< typename T >
struct alignment_of;

template< typename T >
struct alignment_of_hack
{
    char c;
    T t;
    alignment_of_hack();
};

template< size_t A, size_t S >
struct alignment_logic
{
    enum { value = A < S ? A : S };
};

template< typename T >
struct alignment_of
{
    enum { value = alignment_logic<
        sizeof( alignment_of_hack<T> ) - sizeof(T), sizeof(T) >::value };
};

template< typename List, size_t N >
struct type_of_size
{
    typedef typename std11::conditional<
        N == sizeof( typename List::head ),
            typename List::head,
            typename type_of_size<typename List::tail, N >::type >::type type;
};

template< size_t N >
struct type_of_size< nulltype, N >
{
    typedef optional_CONFIG_ALIGN_AS_FALLBACK type;
};

template< typename T>
struct struct_t { T _; };

#define optional_ALIGN_TYPE( type ) \
    typelist< type , typelist< struct_t< type >

struct Unknown;

typedef
    optional_ALIGN_TYPE( char ),
    optional_ALIGN_TYPE( short ),
    optional_ALIGN_TYPE( int ),
    optional_ALIGN_TYPE( long ),
    optional_ALIGN_TYPE( float ),
    optional_ALIGN_TYPE( double ),
    optional_ALIGN_TYPE( long double ),

    optional_ALIGN_TYPE( char *),
    optional_ALIGN_TYPE( short * ),
    optional_ALIGN_TYPE( int * ),
    optional_ALIGN_TYPE( long * ),
    optional_ALIGN_TYPE( float * ),
    optional_ALIGN_TYPE( double * ),
    optional_ALIGN_TYPE( long double * ),

    optional_ALIGN_TYPE( Unknown ( * )( Unknown ) ),
    optional_ALIGN_TYPE( Unknown * Unknown::*     ),
    optional_ALIGN_TYPE( Unknown ( Unknown::* )( Unknown ) ),

    nulltype
    > > > > > > >    > > > > > > >
    > > > > > > >    > > > > > > >
    > > > > > >
    alignment_types;

#undef optional_ALIGN_TYPE

#endif // optional_CONFIG_MAX_ALIGN_HACK

/// C++03 constructed union to hold value.

template< typename T >
union storage_t
{
//private:
//    template< typename > friend class optional;

    typedef T value_type;

    storage_t() optional_is_default

    explicit storage_t( value_type const & v )
    {
        construct_value( v );
    }

    void construct_value( value_type const & v )
    {
        ::new( value_ptr() ) value_type( v );
    }

#if optional_CPP11_OR_GREATER

    explicit storage_t( value_type && v )
    {
        construct_value( std::move( v ) );
    }

    void construct_value( value_type && v )
    {
        ::new( value_ptr() ) value_type( std::move( v ) );
    }

    template< class... Args >
    void emplace( Args&&... args )
    {
        ::new( value_ptr() ) value_type( std::forward<Args>(args)... );
    }

    template< class U, class... Args >
    void emplace( std::initializer_list<U> il, Args&&... args )
    {
        ::new( value_ptr() ) value_type( il, std::forward<Args>(args)... );
    }

#endif

    void destruct_value()
    {
        value_ptr()->~T();
    }

    optional_nodiscard value_type const * value_ptr() const
    {
        return as<value_type>();
    }

    value_type * value_ptr()
    {
        return as<value_type>();
    }

    optional_nodiscard value_type const & value() const optional_ref_qual
    {
        return * value_ptr();
    }

    value_type & value() optional_ref_qual
    {
        return * value_ptr();
    }

#if optional_HAVE( REF_QUALIFIER )

    optional_nodiscard value_type const && value() const optional_refref_qual
    {
        return std::move( value() );
    }

    value_type && value() optional_refref_qual
    {
        return std::move( value() );
    }

#endif

#if optional_CPP11_OR_GREATER

    using aligned_storage_t = typename std::aligned_storage< sizeof(value_type), alignof(value_type) >::type;
    aligned_storage_t data;

#elif optional_CONFIG_MAX_ALIGN_HACK

    typedef struct { unsigned char data[ sizeof(value_type) ]; } aligned_storage_t;

    max_align_t hack;
    aligned_storage_t data;

#else
    typedef optional_ALIGN_AS(value_type) align_as_type;

    typedef struct { align_as_type data[ 1 + ( sizeof(value_type) - 1 ) / sizeof(align_as_type) ]; } aligned_storage_t;
    aligned_storage_t data;

#   undef optional_ALIGN_AS

#endif // optional_CONFIG_MAX_ALIGN_HACK

    optional_nodiscard void * ptr() optional_noexcept
    {
        return &data;
    }

    optional_nodiscard void const * ptr() const optional_noexcept
    {
        return &data;
    }

    template <typename U>
    optional_nodiscard U * as()
    {
        return reinterpret_cast<U*>( ptr() );
    }

    template <typename U>
    optional_nodiscard U const * as() const
    {
        return reinterpret_cast<U const *>( ptr() );
    }
};

} // namespace detail

/// disengaged state tag

struct nullopt_t
{
    struct init{};
    explicit optional_constexpr nullopt_t( init /*unused*/ ) optional_noexcept {}
};

#if optional_HAVE( CONSTEXPR_11 )
constexpr nullopt_t nullopt{ nullopt_t::init{} };
#else
// extra parenthesis to prevent the most vexing parse:
const nullopt_t nullopt(( nullopt_t::init() ));
#endif

/// optional access error

#if ! optional_CONFIG_NO_EXCEPTIONS

class bad_optional_access : public std::logic_error
{
public:
  explicit bad_optional_access()
  : logic_error( "bad optional access" ) {}
};

#endif //optional_CONFIG_NO_EXCEPTIONS

/// optional

template< typename T>
class optional
{
private:
    template< typename > friend class optional;

    typedef void (optional::*safe_bool)() const;

public:
    typedef T value_type;

    // x.x.3.1, constructors

    // 1a - default construct
    optional_constexpr optional() optional_noexcept
    : has_value_( false )
    , contained()
    {}

    // 1b - construct explicitly empty
    // NOLINTNEXTLINE( google-explicit-constructor, hicpp-explicit-conversions )
    optional_constexpr optional( nullopt_t /*unused*/ ) optional_noexcept
    : has_value_( false )
    , contained()
    {}

    // 2 - copy-construct
#if optional_CPP11_OR_GREATER
    // template< typename U = T
    //     optional_REQUIRES_T(
    //         std::is_copy_constructible<U>::value
    //         || std11::is_trivially_copy_constructible<U>::value
    //     )
    // >
#endif
    optional_constexpr14 optional( optional const & other )
    : has_value_( other.has_value() )
    {
        if ( other.has_value() )
        {
            contained.construct_value( other.contained.value() );
        }
    }

#if optional_CPP11_OR_GREATER

    // 3 (C++11) - move-construct from optional
    template< typename U = T
        optional_REQUIRES_T(
            std11::is_move_constructible<U>::value
            || std11::is_trivially_move_constructible<U>::value
        )
    >
    optional_constexpr14 optional( optional && other )
    // NOLINTNEXTLINE( performance-noexcept-move-constructor )
        noexcept( std11::is_nothrow_move_constructible<T>::value )
    : has_value_( other.has_value() )
    {
        if ( other.has_value() )
        {
            contained.construct_value( std::move( other.contained.value() ) );
        }
    }

    // 4a (C++11) - explicit converting copy-construct from optional
    template< typename U
        optional_REQUIRES_T(
            std::is_constructible<T, U const &>::value
            && !std::is_constructible<T, optional<U> &          >::value
            && !std::is_constructible<T, optional<U> &&         >::value
            && !std::is_constructible<T, optional<U> const &    >::value
            && !std::is_constructible<T, optional<U> const &&   >::value
            && !std::is_convertible<     optional<U> &       , T>::value
            && !std::is_convertible<     optional<U> &&      , T>::value
            && !std::is_convertible<     optional<U> const & , T>::value
            && !std::is_convertible<     optional<U> const &&, T>::value
            && !std::is_convertible<               U const & , T>::value /*=> explicit */
        )
    >
    explicit optional( optional<U> const & other )
    : has_value_( other.has_value() )
    {
        if ( other.has_value() )
        {
            contained.construct_value( T{ other.contained.value() } );
        }
    }
#endif // optional_CPP11_OR_GREATER

    // 4b (C++98 and later) - non-explicit converting copy-construct from optional
    template< typename U
#if optional_CPP11_OR_GREATER
        optional_REQUIRES_T(
            std::is_constructible<T, U const &>::value
            && !std::is_constructible<T, optional<U> &          >::value
            && !std::is_constructible<T, optional<U> &&         >::value
            && !std::is_constructible<T, optional<U> const &    >::value
            && !std::is_constructible<T, optional<U> const &&   >::value
            && !std::is_convertible<     optional<U> &       , T>::value
            && !std::is_convertible<     optional<U> &&      , T>::value
            && !std::is_convertible<     optional<U> const & , T>::value
            && !std::is_convertible<     optional<U> const &&, T>::value
            &&  std::is_convertible<               U const & , T>::value /*=> non-explicit */
        )
#endif // optional_CPP11_OR_GREATER
    >
    // NOLINTNEXTLINE( google-explicit-constructor, hicpp-explicit-conversions )
    /*non-explicit*/ optional( optional<U> const & other )
    : has_value_( other.has_value() )
    {
        if ( other.has_value() )
        {
            contained.construct_value( other.contained.value() );
        }
    }

#if optional_CPP11_OR_GREATER

    // 5a (C++11) - explicit converting move-construct from optional
    template< typename U
        optional_REQUIRES_T(
            std::is_constructible<T, U &&>::value
            && !std::is_constructible<T, optional<U> &          >::value
            && !std::is_constructible<T, optional<U> &&         >::value
            && !std::is_constructible<T, optional<U> const &    >::value
            && !std::is_constructible<T, optional<U> const &&   >::value
            && !std::is_convertible<     optional<U> &       , T>::value
            && !std::is_convertible<     optional<U> &&      , T>::value
            && !std::is_convertible<     optional<U> const & , T>::value
            && !std::is_convertible<     optional<U> const &&, T>::value
            && !std::is_convertible<                     U &&, T>::value /*=> explicit */
        )
    >
    explicit optional( optional<U> && other
    )
    : has_value_( other.has_value() )
    {
        if ( other.has_value() )
        {
            contained.construct_value( T{ std::move( other.contained.value() ) } );
        }
    }

    // 5a (C++11) - non-explicit converting move-construct from optional
    template< typename U
        optional_REQUIRES_T(
            std::is_constructible<T, U &&>::value
            && !std::is_constructible<T, optional<U> &          >::value
            && !std::is_constructible<T, optional<U> &&         >::value
            && !std::is_constructible<T, optional<U> const &    >::value
            && !std::is_constructible<T, optional<U> const &&   >::value
            && !std::is_convertible<     optional<U> &       , T>::value
            && !std::is_convertible<     optional<U> &&      , T>::value
            && !std::is_convertible<     optional<U> const & , T>::value
            && !std::is_convertible<     optional<U> const &&, T>::value
            &&  std::is_convertible<                     U &&, T>::value /*=> non-explicit */
        )
    >
    // NOLINTNEXTLINE( google-explicit-constructor, hicpp-explicit-conversions )
    /*non-explicit*/ optional( optional<U> && other )
    : has_value_( other.has_value() )
    {
        if ( other.has_value() )
        {
            contained.construct_value( std::move( other.contained.value() ) );
        }
    }

    // 6 (C++11) - in-place construct
    template< typename... Args
        optional_REQUIRES_T(
            std::is_constructible<T, Args&&...>::value
        )
    >
    optional_constexpr explicit optional( nonstd_lite_in_place_t(T), Args&&... args )
    : has_value_( true )
    , contained( T( std::forward<Args>(args)...) )
    {}

    // 7 (C++11) - in-place construct,  initializer-list
    template< typename U, typename... Args
        optional_REQUIRES_T(
            std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value
        )
    >
    optional_constexpr explicit optional( nonstd_lite_in_place_t(T), std::initializer_list<U> il, Args&&... args )
    : has_value_( true )
    , contained( T( il, std::forward<Args>(args)...) )
    {}

    // 8a (C++11) - explicit move construct from value
    template< typename U = T
        optional_REQUIRES_T(
            std::is_constructible<T, U&&>::value
            && !std::is_same<typename std20::remove_cvref<U>::type, nonstd_lite_in_place_t(U)>::value
            && !std::is_same<typename std20::remove_cvref<U>::type, optional<T>>::value
            && !std::is_convertible<U&&, T>::value /*=> explicit */
        )
    >
    optional_constexpr explicit optional( U && value )
    : has_value_( true )
    , contained( T{ std::forward<U>( value ) } )
    {}

    // 8b (C++11) - non-explicit move construct from value
    template< typename U = T
        optional_REQUIRES_T(
            std::is_constructible<T, U&&>::value
            && !std::is_same<typename std20::remove_cvref<U>::type, nonstd_lite_in_place_t(U)>::value
            && !std::is_same<typename std20::remove_cvref<U>::type, optional<T>>::value
            && std::is_convertible<U&&, T>::value /*=> non-explicit */
        )
    >
    // NOLINTNEXTLINE( google-explicit-constructor, hicpp-explicit-conversions )
    optional_constexpr /*non-explicit*/ optional( U && value )
    : has_value_( true )
    , contained( std::forward<U>( value ) )
    {}

#else // optional_CPP11_OR_GREATER

    // 8 (C++98)
    optional( value_type const & value )
    : has_value_( true )
    , contained( value )
    {}

#endif // optional_CPP11_OR_GREATER

    // x.x.3.2, destructor

    ~optional()
    {
        if ( has_value() )
        {
            contained.destruct_value();
        }
    }

    // x.x.3.3, assignment

    // 1 (C++98and later) -  assign explicitly empty
    optional & operator=( nullopt_t /*unused*/) optional_noexcept
    {
        reset();
        return *this;
    }

    // 2 (C++98and later) - copy-assign from optional
#if optional_CPP11_OR_GREATER
    // NOLINTNEXTLINE( cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator )
    optional_REQUIRES_R(
        optional &,
        true
//      std::is_copy_constructible<T>::value
//      && std::is_copy_assignable<T>::value
    )
    operator=( optional const & other )
        noexcept(
            std11::is_nothrow_move_assignable<T>::value
            && std11::is_nothrow_move_constructible<T>::value
        )
#else
    optional & operator=( optional const & other )
#endif
    {
        if      ( (has_value() == true ) && (other.has_value() == false) ) { reset(); }
        else if ( (has_value() == false) && (other.has_value() == true ) ) { initialize( *other ); }
        else if ( (has_value() == true ) && (other.has_value() == true ) ) { contained.value() = *other; }
        return *this;
    }

#if optional_CPP11_OR_GREATER

    // 3 (C++11) - move-assign from optional
    // NOLINTNEXTLINE( cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator )
    optional_REQUIRES_R(
        optional &,
        true
//      std11::is_move_constructible<T>::value
//      && std::is_move_assignable<T>::value
    )
    operator=( optional && other ) noexcept
    {
        if      ( (has_value() == true ) && (other.has_value() == false) ) { reset(); }
        else if ( (has_value() == false) && (other.has_value() == true ) ) { initialize( std::move( *other ) ); }
        else if ( (has_value() == true ) && (other.has_value() == true ) ) { contained.value() = std::move( *other ); }
        return *this;
    }

    // 4 (C++11) - move-assign from value
    template< typename U = T >
        // NOLINTNEXTLINE( cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator )
        optional_REQUIRES_R(
            optional &,
            std::is_constructible<T , U>::value
            && std11::is_assignable<T&, U>::value
            && !std::is_same<typename std20::remove_cvref<U>::type, nonstd_lite_in_place_t(U)>::value
            && !std::is_same<typename std20::remove_cvref<U>::type, optional<T>>::value
            && !(std::is_scalar<T>::value && std::is_same<T, typename std::decay<U>::type>::value)
        )
    operator=( U && value )
    {
        if ( has_value() )
        {
            contained.value() = std::forward<U>( value );
        }
        else
        {
            initialize( T( std::forward<U>( value ) ) );
        }
        return *this;
    }

#else // optional_CPP11_OR_GREATER

    // 4 (C++98) - copy-assign from value
    template< typename U /*= T*/ >
    optional & operator=( U const & value )
    {
        if ( has_value() ) contained.value() = value;
        else               initialize( T( value ) );
        return *this;
    }

#endif // optional_CPP11_OR_GREATER

    // 5 (C++98 and later) - converting copy-assign from optional
    template< typename U >
#if optional_CPP11_OR_GREATER
        // NOLINTNEXTLINE( cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator )
        optional_REQUIRES_R(
            optional&,
            std::is_constructible<  T , U const &>::value
            &&  std11::is_assignable< T&, U const &>::value
            && !std::is_constructible<T, optional<U> &          >::value
            && !std::is_constructible<T, optional<U> &&         >::value
            && !std::is_constructible<T, optional<U> const &    >::value
            && !std::is_constructible<T, optional<U> const &&   >::value
            && !std::is_convertible<     optional<U> &       , T>::value
            && !std::is_convertible<     optional<U> &&      , T>::value
            && !std::is_convertible<     optional<U> const & , T>::value
            && !std::is_convertible<     optional<U> const &&, T>::value
            && !std11::is_assignable<  T&, optional<U> &          >::value
            && !std11::is_assignable<  T&, optional<U> &&         >::value
            && !std11::is_assignable<  T&, optional<U> const &    >::value
            && !std11::is_assignable<  T&, optional<U> const &&   >::value
        )
#else
    optional&
#endif // optional_CPP11_OR_GREATER
    operator=( optional<U> const & other )
    {
        return *this = optional( other );
    }

#if optional_CPP11_OR_GREATER

    // 6 (C++11) -  converting move-assign from optional
    template< typename U >
        // NOLINTNEXTLINE( cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator )
        optional_REQUIRES_R(
            optional&,
            std::is_constructible<  T , U>::value
            &&  std11::is_assignable< T&, U>::value
            && !std::is_constructible<T, optional<U> &          >::value
            && !std::is_constructible<T, optional<U> &&         >::value
            && !std::is_constructible<T, optional<U> const &    >::value
            && !std::is_constructible<T, optional<U> const &&   >::value
            && !std::is_convertible<     optional<U> &       , T>::value
            && !std::is_convertible<     optional<U> &&      , T>::value
            && !std::is_convertible<     optional<U> const & , T>::value
            && !std::is_convertible<     optional<U> const &&, T>::value
            && !std11::is_assignable<  T&, optional<U> &          >::value
            && !std11::is_assignable<  T&, optional<U> &&         >::value
            && !std11::is_assignable<  T&, optional<U> const &    >::value
            && !std11::is_assignable<  T&, optional<U> const &&   >::value
        )
    operator=( optional<U> && other )
    {
        return *this = optional( std::move( other ) );
    }

    // 7 (C++11) - emplace
    template< typename... Args
        optional_REQUIRES_T(
            std::is_constructible<T, Args&&...>::value
        )
    >
    T& emplace( Args&&... args )
    {
        *this = nullopt;
        contained.emplace( std::forward<Args>(args)...  );
        has_value_ = true;
        return contained.value();
    }

    // 8 (C++11) - emplace, initializer-list
    template< typename U, typename... Args
        optional_REQUIRES_T(
            std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value
        )
    >
    T& emplace( std::initializer_list<U> il, Args&&... args )
    {
        *this = nullopt;
        contained.emplace( il, std::forward<Args>(args)...  );
        has_value_ = true;
        return contained.value();
    }

#endif // optional_CPP11_OR_GREATER

    // x.x.3.4, swap

    void swap( optional & other )
#if optional_CPP11_OR_GREATER
        noexcept(
            std11::is_nothrow_move_constructible<T>::value
            && std17::is_nothrow_swappable<T>::value
        )
#endif
    {
        using std::swap;
        if      ( (has_value() == true ) && (other.has_value() == true ) ) { swap( **this, *other ); }
        else if ( (has_value() == false) && (other.has_value() == true ) ) { initialize( std11::move(*other) ); other.reset(); }
        else if ( (has_value() == true ) && (other.has_value() == false) ) { other.initialize( std11::move(**this) ); reset(); }
    }

    // x.x.3.5, observers

    optional_constexpr value_type const * operator ->() const
    {
        return assert( has_value() ),
            contained.value_ptr();
    }

    optional_constexpr14 value_type * operator ->()
    {
        return assert( has_value() ),
            contained.value_ptr();
    }

    optional_constexpr value_type const & operator *() const optional_ref_qual
    {
        return assert( has_value() ),
            contained.value();
    }

    optional_constexpr14 value_type & operator *() optional_ref_qual
    {
        return assert( has_value() ),
            contained.value();
    }

#if optional_HAVE( REF_QUALIFIER )

    optional_constexpr value_type const && operator *() const optional_refref_qual
    {
        return std::move( **this );
    }

    optional_constexpr14 value_type && operator *() optional_refref_qual
    {
        return std::move( **this );
    }

#endif

#if optional_CPP11_OR_GREATER
    optional_constexpr explicit operator bool() const optional_noexcept
    {
        return has_value();
    }
#else
    optional_constexpr operator safe_bool() const optional_noexcept
    {
        return has_value() ? &optional::this_type_does_not_support_comparisons : 0;
    }
#endif

    // NOLINTNEXTLINE( modernize-use-nodiscard )
    /*optional_nodiscard*/ optional_constexpr bool has_value() const optional_noexcept
    {
        return has_value_;
    }

    // NOLINTNEXTLINE( modernize-use-nodiscard )
    /*optional_nodiscard*/ optional_constexpr14 value_type const & value() const optional_ref_qual
    {
#if optional_CONFIG_NO_EXCEPTIONS
        assert( has_value() );
#else
        if ( ! has_value() )
        {
            throw bad_optional_access();
        }
#endif
        return contained.value();
    }

    optional_constexpr14 value_type & value() optional_ref_qual
    {
#if optional_CONFIG_NO_EXCEPTIONS
        assert( has_value() );
#else
        if ( ! has_value() )
        {
            throw bad_optional_access();
        }
#endif
        return contained.value();
    }

#if optional_HAVE( REF_QUALIFIER )  &&  ( !optional_COMPILER_GNUC_VERSION || optional_COMPILER_GNUC_VERSION >= 490 )

    // NOLINTNEXTLINE( modernize-use-nodiscard )
    /*optional_nodiscard*/ optional_constexpr value_type const && value() const optional_refref_qual
    {
        return std::move( value() );
    }

    optional_constexpr14 value_type && value() optional_refref_qual
    {
        return std::move( value() );
    }

#endif

#if optional_CPP11_OR_GREATER

    template< typename U >
    optional_constexpr value_type value_or( U && v ) const optional_ref_qual
    {
        return has_value() ? contained.value() : static_cast<T>(std::forward<U>( v ) );
    }

    template< typename U >
    optional_constexpr14 value_type value_or( U && v ) optional_refref_qual
    {
        return has_value() ? std::move( contained.value() ) : static_cast<T>(std::forward<U>( v ) );
    }

#else

    template< typename U >
    optional_constexpr value_type value_or( U const & v ) const
    {
        return has_value() ? contained.value() : static_cast<value_type>( v );
    }

#endif // optional_CPP11_OR_GREATER

    // x.x.3.6, modifiers

    void reset() optional_noexcept
    {
        if ( has_value() )
        {
            contained.destruct_value();
        }

        has_value_ = false;
    }

private:
    void this_type_does_not_support_comparisons() const {}

    template< typename V >
    void initialize( V const & value )
    {
        assert( ! has_value()  );
        contained.construct_value( value );
        has_value_ = true;
    }

#if optional_CPP11_OR_GREATER
    template< typename V >
    void initialize( V && value )
    {
        assert( ! has_value()  );
        contained.construct_value( std::move( value ) );
        has_value_ = true;
    }

#endif

private:
    bool has_value_;
    detail::storage_t< value_type > contained;

};

// Relational operators

template< typename T, typename U >
inline optional_constexpr bool operator==( optional<T> const & x, optional<U> const & y )
{
    return bool(x) != bool(y) ? false : !bool( x ) ? true : *x == *y;
}

template< typename T, typename U >
inline optional_constexpr bool operator!=( optional<T> const & x, optional<U> const & y )
{
    return !(x == y);
}

template< typename T, typename U >
inline optional_constexpr bool operator<( optional<T> const & x, optional<U> const & y )
{
    return (!y) ? false : (!x) ? true : *x < *y;
}

template< typename T, typename U >
inline optional_constexpr bool operator>( optional<T> const & x, optional<U> const & y )
{
    return (y < x);
}

template< typename T, typename U >
inline optional_constexpr bool operator<=( optional<T> const & x, optional<U> const & y )
{
    return !(y < x);
}

template< typename T, typename U >
inline optional_constexpr bool operator>=( optional<T> const & x, optional<U> const & y )
{
    return !(x < y);
}

// Comparison with nullopt

template< typename T >
inline optional_constexpr bool operator==( optional<T> const & x, nullopt_t /*unused*/ ) optional_noexcept
{
    return (!x);
}

template< typename T >
inline optional_constexpr bool operator==( nullopt_t /*unused*/, optional<T> const & x ) optional_noexcept
{
    return (!x);
}

template< typename T >
inline optional_constexpr bool operator!=( optional<T> const & x, nullopt_t /*unused*/ ) optional_noexcept
{
    return bool(x);
}

template< typename T >
inline optional_constexpr bool operator!=( nullopt_t /*unused*/, optional<T> const & x ) optional_noexcept
{
    return bool(x);
}

template< typename T >
inline optional_constexpr bool operator<( optional<T> const & /*unused*/, nullopt_t /*unused*/ ) optional_noexcept
{
    return false;
}

template< typename T >
inline optional_constexpr bool operator<( nullopt_t /*unused*/, optional<T> const & x ) optional_noexcept
{
    return bool(x);
}

template< typename T >
inline optional_constexpr bool operator<=( optional<T> const & x, nullopt_t /*unused*/ ) optional_noexcept
{
    return (!x);
}

template< typename T >
inline optional_constexpr bool operator<=( nullopt_t /*unused*/, optional<T> const & /*unused*/ ) optional_noexcept
{
    return true;
}

template< typename T >
inline optional_constexpr bool operator>( optional<T> const & x, nullopt_t /*unused*/ ) optional_noexcept
{
    return bool(x);
}

template< typename T >
inline optional_constexpr bool operator>( nullopt_t /*unused*/, optional<T> const & /*unused*/ ) optional_noexcept
{
    return false;
}

template< typename T >
inline optional_constexpr bool operator>=( optional<T> const & /*unused*/, nullopt_t /*unused*/ ) optional_noexcept
{
    return true;
}

template< typename T >
inline optional_constexpr bool operator>=( nullopt_t /*unused*/, optional<T> const & x ) optional_noexcept
{
    return (!x);
}

// Comparison with T

template< typename T, typename U >
inline optional_constexpr bool operator==( optional<T> const & x, U const & v )
{
    return bool(x) ? *x == v : false;
}

template< typename T, typename U >
inline optional_constexpr bool operator==( U const & v, optional<T> const & x )
{
    return bool(x) ? v == *x : false;
}

template< typename T, typename U >
inline optional_constexpr bool operator!=( optional<T> const & x, U const & v )
{
    return bool(x) ? *x != v : true;
}

template< typename T, typename U >
inline optional_constexpr bool operator!=( U const & v, optional<T> const & x )
{
    return bool(x) ? v != *x : true;
}

template< typename T, typename U >
inline optional_constexpr bool operator<( optional<T> const & x, U const & v )
{
    return bool(x) ? *x < v : true;
}

template< typename T, typename U >
inline optional_constexpr bool operator<( U const & v, optional<T> const & x )
{
    return bool(x) ? v < *x : false;
}

template< typename T, typename U >
inline optional_constexpr bool operator<=( optional<T> const & x, U const & v )
{
    return bool(x) ? *x <= v : true;
}

template< typename T, typename U >
inline optional_constexpr bool operator<=( U const & v, optional<T> const & x )
{
    return bool(x) ? v <= *x : false;
}

template< typename T, typename U >
inline optional_constexpr bool operator>( optional<T> const & x, U const & v )
{
    return bool(x) ? *x > v : false;
}

template< typename T, typename U >
inline optional_constexpr bool operator>( U const & v, optional<T> const & x )
{
    return bool(x) ? v > *x : true;
}

template< typename T, typename U >
inline optional_constexpr bool operator>=( optional<T> const & x, U const & v )
{
    return bool(x) ? *x >= v : false;
}

template< typename T, typename U >
inline optional_constexpr bool operator>=( U const & v, optional<T> const & x )
{
    return bool(x) ? v >= *x : true;
}

// Specialized algorithms

template< typename T
#if optional_CPP11_OR_GREATER
    optional_REQUIRES_T(
        std11::is_move_constructible<T>::value
        && std17::is_swappable<T>::value )
#endif
>
void swap( optional<T> & x, optional<T> & y )
#if optional_CPP11_OR_GREATER
    noexcept( noexcept( x.swap(y) ) )
#endif
{
    x.swap( y );
}

#if optional_CPP11_OR_GREATER

template< typename T >
optional_constexpr optional< typename std::decay<T>::type > make_optional( T && value )
{
    return optional< typename std::decay<T>::type >( std::forward<T>( value ) );
}

template< typename T, typename...Args >
optional_constexpr optional<T> make_optional( Args&&... args )
{
    return optional<T>( nonstd_lite_in_place(T), std::forward<Args>(args)...);
}

template< typename T, typename U, typename... Args >
optional_constexpr optional<T> make_optional( std::initializer_list<U> il, Args&&... args )
{
    return optional<T>( nonstd_lite_in_place(T), il, std::forward<Args>(args)...);
}

#else

template< typename T >
optional<T> make_optional( T const & value )
{
    return optional<T>( value );
}

#endif // optional_CPP11_OR_GREATER

} // namespace optional_lite

using optional_lite::optional;
using optional_lite::nullopt_t;
using optional_lite::nullopt;

#if ! optional_CONFIG_NO_EXCEPTIONS
using optional_lite::bad_optional_access;
#endif

using optional_lite::make_optional;

} // namespace nonstd

#if optional_CPP11_OR_GREATER

// specialize the std::hash algorithm:

namespace std {

template< class T >
struct hash< nonstd::optional<T> >
{
public:
    std::size_t operator()( nonstd::optional<T> const & v ) const optional_noexcept
    {
        return bool( v ) ? std::hash<T>{}( *v ) : 0;
    }
};

} //namespace std

#endif // optional_CPP11_OR_GREATER

#if defined(__clang__)
# pragma clang diagnostic pop
#elif defined(__GNUC__)
# pragma GCC   diagnostic pop
#elif defined(_MSC_VER )
# pragma warning( pop )
#endif

#endif // optional_USES_STD_OPTIONAL

#endif // NONSTD_OPTIONAL_LITE_HPP
