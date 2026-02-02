// This version targets C++11 and later.
//
// Copyright (C) 2016-2020 Martin Moene.
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// expected lite is based on:
//   A proposal to add a utility class to represent expected monad
//   by Vicente J. Botet Escriba and Pierre Talbot. http:://wg21.link/p0323

#ifndef NONSTD_EXPECTED_LITE_HPP
#define NONSTD_EXPECTED_LITE_HPP

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#define expected_lite_MAJOR  0
#define expected_lite_MINOR  6
#define expected_lite_PATCH  3

#define expected_lite_VERSION  expected_STRINGIFY(expected_lite_MAJOR) "." expected_STRINGIFY(expected_lite_MINOR) "." expected_STRINGIFY(expected_lite_PATCH)

#define expected_STRINGIFY(  x )  expected_STRINGIFY_( x )
#define expected_STRINGIFY_( x )  #x

// expected-lite configuration:

#define nsel_EXPECTED_DEFAULT  0
#define nsel_EXPECTED_NONSTD   1
#define nsel_EXPECTED_STD      2

// tweak header support:

#ifdef __has_include
# if __has_include(<nonstd/expected.tweak.hpp>)
#  include <nonstd/expected.tweak.hpp>
# endif
#define expected_HAVE_TWEAK_HEADER  1
#else
#define expected_HAVE_TWEAK_HEADER  0
//# pragma message("expected.hpp: Note: Tweak header not supported.")
#endif

// expected selection and configuration:

#if !defined( nsel_CONFIG_SELECT_EXPECTED )
# define nsel_CONFIG_SELECT_EXPECTED  ( nsel_HAVE_STD_EXPECTED ? nsel_EXPECTED_STD : nsel_EXPECTED_NONSTD )
#endif

// Proposal revisions:
//
// DXXXXR0: --
// N4015  : -2 (2014-05-26)
// N4109  : -1 (2014-06-29)
// P0323R0:  0 (2016-05-28)
// P0323R1:  1 (2016-10-12)
// -------:
// P0323R2:  2 (2017-06-15)
// P0323R3:  3 (2017-10-15)
// P0323R4:  4 (2017-11-26)
// P0323R5:  5 (2018-02-08)
// P0323R6:  6 (2018-04-02)
// P0323R7:  7 (2018-06-22) *
//
// expected-lite uses 2 and higher

#ifndef  nsel_P0323R
# define nsel_P0323R  7
#endif

// Monadic operations proposal revisions:
//
// P2505R0:  0 (2021-12-12)
// P2505R1:  1 (2022-02-10)
// P2505R2:  2 (2022-04-15)
// P2505R3:  3 (2022-06-05)
// P2505R4:  4 (2022-06-15)
// P2505R5:  5 (2022-09-20) *
//
// expected-lite uses 5

#ifndef  nsel_P2505R
# define nsel_P2505R  5
#endif

// Control presence of C++ exception handling (try and auto discover):

#ifndef nsel_CONFIG_NO_EXCEPTIONS
# if defined(_MSC_VER)
#  include <cstddef>    // for _HAS_EXCEPTIONS
# endif
# if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || (_HAS_EXCEPTIONS)
#  define nsel_CONFIG_NO_EXCEPTIONS  0
# else
#  define nsel_CONFIG_NO_EXCEPTIONS  1
# endif
#endif

// at default use SEH with MSVC for no C++ exceptions

#ifndef  nsel_CONFIG_NO_EXCEPTIONS_SEH
# define nsel_CONFIG_NO_EXCEPTIONS_SEH  ( nsel_CONFIG_NO_EXCEPTIONS && _MSC_VER )
#endif

// C++ language version detection (C++23 is speculative):
// Note: VC14.0/1900 (VS2015) lacks too much from C++14.

#ifndef   nsel_CPLUSPLUS
# if defined(_MSVC_LANG ) && !defined(__clang__)
#  define nsel_CPLUSPLUS  (_MSC_VER == 1900 ? 201103L : _MSVC_LANG )
# else
#  define nsel_CPLUSPLUS  __cplusplus
# endif
#endif

#define nsel_CPP98_OR_GREATER  ( nsel_CPLUSPLUS >= 199711L )
#define nsel_CPP11_OR_GREATER  ( nsel_CPLUSPLUS >= 201103L )
#define nsel_CPP14_OR_GREATER  ( nsel_CPLUSPLUS >= 201402L )
#define nsel_CPP17_OR_GREATER  ( nsel_CPLUSPLUS >= 201703L )
#define nsel_CPP20_OR_GREATER  ( nsel_CPLUSPLUS >= 202002L )
#define nsel_CPP23_OR_GREATER  ( nsel_CPLUSPLUS >= 202300L )

// Use C++23 std::expected if available and requested:

#if nsel_CPP23_OR_GREATER && defined(__has_include )
# if __has_include( <expected> )
#  define nsel_HAVE_STD_EXPECTED  1
# else
#  define nsel_HAVE_STD_EXPECTED  0
# endif
#else
# define  nsel_HAVE_STD_EXPECTED  0
#endif

#define  nsel_USES_STD_EXPECTED  ( (nsel_CONFIG_SELECT_EXPECTED == nsel_EXPECTED_STD) || ((nsel_CONFIG_SELECT_EXPECTED == nsel_EXPECTED_DEFAULT) && nsel_HAVE_STD_EXPECTED) )

//
// in_place: code duplicated in any-lite, expected-lite, expected-lite, value-ptr-lite, variant-lite:
//

#ifndef nonstd_lite_HAVE_IN_PLACE_TYPES
#define nonstd_lite_HAVE_IN_PLACE_TYPES  1

// C++17 std::in_place in <utility>:

#if nsel_CPP17_OR_GREATER

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

#else // nsel_CPP17_OR_GREATER

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
inline in_place_t in_place( detail::in_place_type_tag<T> = detail::in_place_type_tag<T>() )
{
    return in_place_t();
}

template< std::size_t K >
inline in_place_t in_place( detail::in_place_index_tag<K> = detail::in_place_index_tag<K>() )
{
    return in_place_t();
}

template< class T >
inline in_place_t in_place_type( detail::in_place_type_tag<T> = detail::in_place_type_tag<T>() )
{
    return in_place_t();
}

template< std::size_t K >
inline in_place_t in_place_index( detail::in_place_index_tag<K> = detail::in_place_index_tag<K>() )
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

#endif // nsel_CPP17_OR_GREATER
#endif // nonstd_lite_HAVE_IN_PLACE_TYPES

//
// Using std::expected:
//

#if nsel_USES_STD_EXPECTED

#include <expected>

namespace nonstd {

    using std::expected;
//  ...
}

#else // nsel_USES_STD_EXPECTED

#include <cassert>
#include <exception>
#include <functional>
#include <initializer_list>
#include <memory>
#include <new>
#include <system_error>
#include <type_traits>
#include <utility>

// additional includes:

#if nsel_CONFIG_NO_EXCEPTIONS
# if nsel_CONFIG_NO_EXCEPTIONS_SEH
#  include <windows.h>   // for ExceptionCodes
# else
// already included: <cassert>
# endif
#else
# include <stdexcept>
#endif

// C++ feature usage:

#if nsel_CPP11_OR_GREATER
# define nsel_constexpr  constexpr
#else
# define nsel_constexpr  /*constexpr*/
#endif

#if nsel_CPP14_OR_GREATER
# define nsel_constexpr14 constexpr
#else
# define nsel_constexpr14 /*constexpr*/
#endif

#if nsel_CPP17_OR_GREATER
# define nsel_inline17 inline
#else
# define nsel_inline17 /*inline*/
#endif

// Compiler versions:
//
// MSVC++  6.0  _MSC_VER == 1200  nsel_COMPILER_MSVC_VERSION ==  60  (Visual Studio 6.0)
// MSVC++  7.0  _MSC_VER == 1300  nsel_COMPILER_MSVC_VERSION ==  70  (Visual Studio .NET 2002)
// MSVC++  7.1  _MSC_VER == 1310  nsel_COMPILER_MSVC_VERSION ==  71  (Visual Studio .NET 2003)
// MSVC++  8.0  _MSC_VER == 1400  nsel_COMPILER_MSVC_VERSION ==  80  (Visual Studio 2005)
// MSVC++  9.0  _MSC_VER == 1500  nsel_COMPILER_MSVC_VERSION ==  90  (Visual Studio 2008)
// MSVC++ 10.0  _MSC_VER == 1600  nsel_COMPILER_MSVC_VERSION == 100  (Visual Studio 2010)
// MSVC++ 11.0  _MSC_VER == 1700  nsel_COMPILER_MSVC_VERSION == 110  (Visual Studio 2012)
// MSVC++ 12.0  _MSC_VER == 1800  nsel_COMPILER_MSVC_VERSION == 120  (Visual Studio 2013)
// MSVC++ 14.0  _MSC_VER == 1900  nsel_COMPILER_MSVC_VERSION == 140  (Visual Studio 2015)
// MSVC++ 14.1  _MSC_VER >= 1910  nsel_COMPILER_MSVC_VERSION == 141  (Visual Studio 2017)
// MSVC++ 14.2  _MSC_VER >= 1920  nsel_COMPILER_MSVC_VERSION == 142  (Visual Studio 2019)

#if defined(_MSC_VER) && !defined(__clang__)
# define nsel_COMPILER_MSVC_VER      (_MSC_VER )
# define nsel_COMPILER_MSVC_VERSION  (_MSC_VER / 10 - 10 * ( 5 + (_MSC_VER < 1900)) )
#else
# define nsel_COMPILER_MSVC_VER      0
# define nsel_COMPILER_MSVC_VERSION  0
#endif

#define nsel_COMPILER_VERSION( major, minor, patch )  ( 10 * ( 10 * (major) + (minor) ) + (patch) )

#if defined(__clang__)
# define nsel_COMPILER_CLANG_VERSION  nsel_COMPILER_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
# define nsel_COMPILER_CLANG_VERSION  0
#endif

#if defined(__GNUC__) && !defined(__clang__)
# define nsel_COMPILER_GNUC_VERSION  nsel_COMPILER_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
# define nsel_COMPILER_GNUC_VERSION  0
#endif

// half-open range [lo..hi):
//#define nsel_BETWEEN( v, lo, hi ) ( (lo) <= (v) && (v) < (hi) )

// Method enabling

#define nsel_REQUIRES_0(...) \
    template< bool B = (__VA_ARGS__), typename std::enable_if<B, int>::type = 0 >

#define nsel_REQUIRES_T(...) \
    , typename std::enable_if< (__VA_ARGS__), int >::type = 0

#define nsel_REQUIRES_R(R, ...) \
    typename std::enable_if< (__VA_ARGS__), R>::type

#define nsel_REQUIRES_A(...) \
    , typename std::enable_if< (__VA_ARGS__), void*>::type = nullptr

// Presence of language and library features:

#ifdef _HAS_CPP0X
# define nsel_HAS_CPP0X  _HAS_CPP0X
#else
# define nsel_HAS_CPP0X  0
#endif

//#define nsel_CPP11_140  (nsel_CPP11_OR_GREATER || nsel_COMPILER_MSVC_VER >= 1900)

// Clang, GNUC, MSVC warning suppression macros:

#ifdef __clang__
# pragma clang diagnostic push
#elif defined  __GNUC__
# pragma  GCC  diagnostic push
#endif // __clang__

#if nsel_COMPILER_MSVC_VERSION >= 140
# pragma warning( push )
# define nsel_DISABLE_MSVC_WARNINGS(codes)  __pragma( warning(disable: codes) )
#else
# define nsel_DISABLE_MSVC_WARNINGS(codes)
#endif

#ifdef __clang__
# define nsel_RESTORE_WARNINGS()  _Pragma("clang diagnostic pop")
#elif defined __GNUC__
# define nsel_RESTORE_WARNINGS()  _Pragma("GCC diagnostic pop")
#elif nsel_COMPILER_MSVC_VERSION >= 140
# define nsel_RESTORE_WARNINGS()  __pragma( warning( pop ) )
#else
# define nsel_RESTORE_WARNINGS()
#endif

// Suppress the following MSVC (GSL) warnings:
// - C26409: Avoid calling new and delete explicitly, use std::make_unique<T> instead (r.11)

nsel_DISABLE_MSVC_WARNINGS( 26409 )

//
// expected:
//

namespace nonstd { namespace expected_lite {

// type traits C++17:

namespace std17 {

#if nsel_CPP17_OR_GREATER

using std::conjunction;
using std::is_swappable;
using std::is_nothrow_swappable;

#else // nsel_CPP17_OR_GREATER

namespace detail {

using std::swap;

struct is_swappable
{
    template< typename T, typename = decltype( swap( std::declval<T&>(), std::declval<T&>() ) ) >
    static std::true_type test( int /* unused */);

    template< typename >
    static std::false_type test(...);
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
    static auto test( int ) -> std::integral_constant<bool, satisfies<T>()>{}

    template< typename >
    static auto test(...) -> std::false_type;
};
} // namespace detail

// is [nothrow] swappable:

template< typename T >
struct is_swappable : decltype( detail::is_swappable::test<T>(0) ){};

template< typename T >
struct is_nothrow_swappable : decltype( detail::is_nothrow_swappable::test<T>(0) ){};

// conjunction:

template< typename... > struct conjunction : std::true_type{};
template< typename B1 > struct conjunction<B1> : B1{};

template< typename B1, typename... Bn >
struct conjunction<B1, Bn...> : std::conditional<bool(B1::value), conjunction<Bn...>, B1>::type{};

#endif // nsel_CPP17_OR_GREATER

} // namespace std17

// type traits C++20:

namespace std20 {

#if defined(__cpp_lib_remove_cvref)

using std::remove_cvref;

#else

template< typename T >
struct remove_cvref
{
    typedef typename std::remove_cv< typename std::remove_reference<T>::type >::type type;
};

#endif

} // namespace std20

// forward declaration:

template< typename T, typename E >
class expected;

namespace detail {

#if nsel_P2505R >= 3
template< typename T >
struct is_expected : std::false_type {};

template< typename T, typename E >
struct is_expected< expected< T, E > > : std::true_type {};
#endif // nsel_P2505R >= 3

/// discriminated union to hold value or 'error'.

template< typename T, typename E >
class storage_t_noncopy_nonmove_impl
{
    template< typename, typename > friend class nonstd::expected_lite::expected;

public:
    using value_type = T;
    using error_type = E;

    // no-op construction
    storage_t_noncopy_nonmove_impl() {}
    ~storage_t_noncopy_nonmove_impl() {}

    explicit storage_t_noncopy_nonmove_impl( bool has_value )
        : m_has_value( has_value )
    {}

    void construct_value()
    {
        new( &m_value ) value_type();
    }

    // void construct_value( value_type const & e )
    // {
    //     new( &m_value ) value_type( e );
    // }

    // void construct_value( value_type && e )
    // {
    //     new( &m_value ) value_type( std::move( e ) );
    // }

    template< class... Args >
    void emplace_value( Args&&... args )
    {
        new( &m_value ) value_type( std::forward<Args>(args)...);
    }

    template< class U, class... Args >
    void emplace_value( std::initializer_list<U> il, Args&&... args )
    {
        new( &m_value ) value_type( il, std::forward<Args>(args)... );
    }

    void destruct_value()
    {
        m_value.~value_type();
    }

    // void construct_error( error_type const & e )
    // {
    //     // new( &m_error ) error_type( e );
    // }

    // void construct_error( error_type && e )
    // {
    //     // new( &m_error ) error_type( std::move( e ) );
    // }

    template< class... Args >
    void emplace_error( Args&&... args )
    {
        new( &m_error ) error_type( std::forward<Args>(args)...);
    }

    template< class U, class... Args >
    void emplace_error( std::initializer_list<U> il, Args&&... args )
    {
        new( &m_error ) error_type( il, std::forward<Args>(args)... );
    }

    void destruct_error()
    {
        m_error.~error_type();
    }

    constexpr value_type const & value() const &
    {
        return m_value;
    }

    value_type & value() &
    {
        return m_value;
    }

    constexpr value_type const && value() const &&
    {
        return std::move( m_value );
    }

    nsel_constexpr14 value_type && value() &&
    {
        return std::move( m_value );
    }

    value_type const * value_ptr() const
    {
        return &m_value;
    }

    value_type * value_ptr()
    {
        return &m_value;
    }

    error_type const & error() const &
    {
        return m_error;
    }

    error_type & error() &
    {
        return m_error;
    }

    constexpr error_type const && error() const &&
    {
        return std::move( m_error );
    }

    nsel_constexpr14 error_type && error() &&
    {
        return std::move( m_error );
    }

    bool has_value() const
    {
        return m_has_value;
    }

    void set_has_value( bool v )
    {
        m_has_value = v;
    }

private:
    union
    {
        value_type m_value;
        error_type m_error;
    };

    bool m_has_value = false;
};

template< typename T, typename E >
class storage_t_impl
{
    template< typename, typename > friend class nonstd::expected_lite::expected;

public:
    using value_type = T;
    using error_type = E;

    // no-op construction
    storage_t_impl() {}
    ~storage_t_impl() {}

    explicit storage_t_impl( bool has_value )
        : m_has_value( has_value )
    {}

    void construct_value()
    {
        new( &m_value ) value_type();
    }

    void construct_value( value_type const & e )
    {
        new( &m_value ) value_type( e );
    }

    void construct_value( value_type && e )
    {
        new( &m_value ) value_type( std::move( e ) );
    }

    template< class... Args >
    void emplace_value( Args&&... args )
    {
        new( &m_value ) value_type( std::forward<Args>(args)...);
    }

    template< class U, class... Args >
    void emplace_value( std::initializer_list<U> il, Args&&... args )
    {
        new( &m_value ) value_type( il, std::forward<Args>(args)... );
    }

    void destruct_value()
    {
        m_value.~value_type();
    }

    void construct_error( error_type const & e )
    {
        new( &m_error ) error_type( e );
    }

    void construct_error( error_type && e )
    {
        new( &m_error ) error_type( std::move( e ) );
    }

    template< class... Args >
    void emplace_error( Args&&... args )
    {
        new( &m_error ) error_type( std::forward<Args>(args)...);
    }

    template< class U, class... Args >
    void emplace_error( std::initializer_list<U> il, Args&&... args )
    {
        new( &m_error ) error_type( il, std::forward<Args>(args)... );
    }

    void destruct_error()
    {
        m_error.~error_type();
    }

    constexpr value_type const & value() const &
    {
        return m_value;
    }

    value_type & value() &
    {
        return m_value;
    }

    constexpr value_type const && value() const &&
    {
        return std::move( m_value );
    }

    nsel_constexpr14 value_type && value() &&
    {
        return std::move( m_value );
    }

    value_type const * value_ptr() const
    {
        return &m_value;
    }

    value_type * value_ptr()
    {
        return &m_value;
    }

    error_type const & error() const &
    {
        return m_error;
    }

    error_type & error() &
    {
        return m_error;
    }

    constexpr error_type const && error() const &&
    {
        return std::move( m_error );
    }

    nsel_constexpr14 error_type && error() &&
    {
        return std::move( m_error );
    }

    bool has_value() const
    {
        return m_has_value;
    }

    void set_has_value( bool v )
    {
        m_has_value = v;
    }

private:
    union
    {
        value_type m_value;
        error_type m_error;
    };

    bool m_has_value = false;
};

/// discriminated union to hold only 'error'.

template< typename E >
struct storage_t_impl<void, E>
{
    template< typename, typename > friend class nonstd::expected_lite::expected;

public:
    using value_type = void;
    using error_type = E;

    // no-op construction
    storage_t_impl() {}
    ~storage_t_impl() {}

    explicit storage_t_impl( bool has_value )
        : m_has_value( has_value )
    {}

    void construct_error( error_type const & e )
    {
        new( &m_error ) error_type( e );
    }

    void construct_error( error_type && e )
    {
        new( &m_error ) error_type( std::move( e ) );
    }

    template< class... Args >
    void emplace_error( Args&&... args )
    {
        new( &m_error ) error_type( std::forward<Args>(args)...);
    }

    template< class U, class... Args >
    void emplace_error( std::initializer_list<U> il, Args&&... args )
    {
        new( &m_error ) error_type( il, std::forward<Args>(args)... );
    }

    void destruct_error()
    {
        m_error.~error_type();
    }

    error_type const & error() const &
    {
        return m_error;
    }

    error_type & error() &
    {
        return m_error;
    }

    constexpr error_type const && error() const &&
    {
        return std::move( m_error );
    }

    nsel_constexpr14 error_type && error() &&
    {
        return std::move( m_error );
    }

    bool has_value() const
    {
        return m_has_value;
    }

    void set_has_value( bool v )
    {
        m_has_value = v;
    }

private:
    union
    {
        char m_dummy;
        error_type m_error;
    };

    bool m_has_value = false;
};

template< typename T, typename E, bool isConstructable, bool isMoveable >
class storage_t
{
public:
};

template< typename T, typename E >
class storage_t<T, E, false, false> : public storage_t_noncopy_nonmove_impl<T, E>
{
public:
    storage_t() = default;
    ~storage_t() = default;

    explicit storage_t( bool has_value )
        : storage_t_noncopy_nonmove_impl<T, E>( has_value )
    {}

    storage_t( storage_t const & other ) = delete;
    storage_t( storage_t &&      other ) = delete;

};

template< typename T, typename E >
class storage_t<T, E, true, true> : public storage_t_impl<T, E>
{
public:
    storage_t() = default;
    ~storage_t() = default;

    explicit storage_t( bool has_value )
        : storage_t_impl<T, E>( has_value )
    {}

    storage_t( storage_t const & other )
        : storage_t_impl<T, E>( other.has_value() )
    {
        if ( this->has_value() ) this->construct_value( other.value() );
        else                     this->construct_error( other.error() );
    }

    storage_t(storage_t && other )
        : storage_t_impl<T, E>( other.has_value() )
    {
        if ( this->has_value() ) this->construct_value( std::move( other.value() ) );
        else                     this->construct_error( std::move( other.error() ) );
    }
};

template< typename E >
class storage_t<void, E, true, true> : public storage_t_impl<void, E>
{
public:
    storage_t() = default;
    ~storage_t() = default;

    explicit storage_t( bool has_value )
        : storage_t_impl<void, E>( has_value )
    {}

    storage_t( storage_t const & other )
        : storage_t_impl<void, E>( other.has_value() )
    {
        if ( this->has_value() ) ;
        else                     this->construct_error( other.error() );
    }

    storage_t(storage_t && other )
        : storage_t_impl<void, E>( other.has_value() )
    {
        if ( this->has_value() ) ;
        else                     this->construct_error( std::move( other.error() ) );
    }
};

template< typename T, typename E >
class storage_t<T, E, true, false> : public storage_t_impl<T, E>
{
public:
    storage_t() = default;
    ~storage_t() = default;

    explicit storage_t( bool has_value )
        : storage_t_impl<T, E>( has_value )
    {}

    storage_t( storage_t const & other )
        : storage_t_impl<T, E>(other.has_value())
    {
        if ( this->has_value() ) this->construct_value( other.value() );
        else                     this->construct_error( other.error() );
    }

    storage_t( storage_t && other ) = delete;
};

template< typename E >
class storage_t<void, E, true, false> : public storage_t_impl<void, E>
{
public:
    storage_t() = default;
    ~storage_t() = default;

    explicit storage_t( bool has_value )
        : storage_t_impl<void, E>( has_value )
    {}

    storage_t( storage_t const & other )
        : storage_t_impl<void, E>(other.has_value())
    {
        if ( this->has_value() ) ;
        else                     this->construct_error( other.error() );
    }

    storage_t( storage_t && other ) = delete;
};

template< typename T, typename E >
class storage_t<T, E, false, true> : public storage_t_impl<T, E>
{
public:
    storage_t() = default;
    ~storage_t() = default;

    explicit storage_t( bool has_value )
        : storage_t_impl<T, E>( has_value )
    {}

    storage_t( storage_t const & other ) = delete;

    storage_t( storage_t && other )
        : storage_t_impl<T, E>( other.has_value() )
    {
        if ( this->has_value() ) this->construct_value( std::move( other.value() ) );
        else                     this->construct_error( std::move( other.error() ) );
    }
};

template< typename E >
class storage_t<void, E, false, true> : public storage_t_impl<void, E>
{
public:
    storage_t() = default;
    ~storage_t() = default;

    explicit storage_t( bool has_value )
        : storage_t_impl<void, E>( has_value )
    {}

    storage_t( storage_t const & other ) = delete;

    storage_t( storage_t && other )
        : storage_t_impl<void, E>( other.has_value() )
    {
        if ( this->has_value() ) ;
        else                     this->construct_error( std::move( other.error() ) );
    }
};

#if nsel_P2505R >= 3
// C++11 invoke implementation
template< typename >
struct is_reference_wrapper : std::false_type {};
template< typename T >
struct is_reference_wrapper< std::reference_wrapper< T > > : std::true_type {};

template< typename FnT, typename ClassT, typename ObjectT, typename... Args
    nsel_REQUIRES_T(
        std::is_function<FnT>::value
        && ( std::is_same< ClassT, typename std20::remove_cvref< ObjectT >::type >::value
        || std::is_base_of< ClassT, typename std20::remove_cvref< ObjectT >::type >::value )
    )
>
nsel_constexpr auto invoke_member_function_impl( FnT ClassT::* memfnptr, ObjectT && obj, Args && ... args )
        noexcept( noexcept( (std::forward< ObjectT >( obj ).*memfnptr)( std::forward< Args >( args )... ) ) )
        -> decltype( (std::forward< ObjectT >( obj ).*memfnptr)( std::forward< Args >( args )...) )
{
      return (std::forward< ObjectT >( obj ).*memfnptr)( std::forward< Args >( args )... );
}

template< typename FnT, typename ClassT, typename ObjectT, typename... Args
    nsel_REQUIRES_T(
        std::is_function<FnT>::value
        && is_reference_wrapper< typename std20::remove_cvref< ObjectT >::type >::value
    )
>
nsel_constexpr auto invoke_member_function_impl( FnT ClassT::* memfnptr, ObjectT && obj, Args && ... args )
        noexcept( noexcept( (obj.get().*memfnptr)( std::forward< Args >( args ) ... ) ) )
        -> decltype( (obj.get().*memfnptr)( std::forward< Args >( args ) ... ) )
{
    return (obj.get().*memfnptr)( std::forward< Args >( args ) ... );
}

template< typename FnT, typename ClassT, typename ObjectT, typename... Args
    nsel_REQUIRES_T(
        std::is_function<FnT>::value
        && !std::is_same< ClassT, typename std20::remove_cvref< ObjectT >::type >::value
        && !std::is_base_of< ClassT, typename std20::remove_cvref< ObjectT >::type >::value
        && !is_reference_wrapper< typename std20::remove_cvref< ObjectT >::type >::value
    )
>
nsel_constexpr auto invoke_member_function_impl( FnT ClassT::* memfnptr, ObjectT && obj, Args && ... args )
        noexcept( noexcept( ((*std::forward< ObjectT >( obj )).*memfnptr)( std::forward< Args >( args ) ... ) ) )
        -> decltype( ((*std::forward< ObjectT >( obj )).*memfnptr)( std::forward< Args >( args ) ... ) )
{
    return ((*std::forward<ObjectT>(obj)).*memfnptr)( std::forward< Args >( args ) ... );
}

template< typename MemberT, typename ClassT, typename ObjectT
    nsel_REQUIRES_T(
        std::is_same< ClassT, typename std20::remove_cvref< ObjectT >::type >::value
        || std::is_base_of< ClassT, typename std20::remove_cvref< ObjectT >::type >::value
    )
>
nsel_constexpr auto invoke_member_object_impl( MemberT ClassT::* memobjptr, ObjectT && obj )
        noexcept( noexcept( std::forward< ObjectT >( obj ).*memobjptr ) )
        -> decltype( std::forward< ObjectT >( obj ).*memobjptr )
{
    return std::forward< ObjectT >( obj ).*memobjptr;
}

template< typename MemberT, typename ClassT, typename ObjectT
    nsel_REQUIRES_T(
        is_reference_wrapper< typename std20::remove_cvref< ObjectT >::type >::value
    )
>
nsel_constexpr auto invoke_member_object_impl( MemberT ClassT::* memobjptr, ObjectT && obj )
        noexcept( noexcept( obj.get().*memobjptr ) )
        -> decltype( obj.get().*memobjptr )
{
    return obj.get().*memobjptr;
}

template< typename MemberT, typename ClassT, typename ObjectT
    nsel_REQUIRES_T(
        !std::is_same< ClassT, typename std20::remove_cvref< ObjectT >::type >::value
        && !std::is_base_of< ClassT, typename std20::remove_cvref< ObjectT >::type >::value
        && !is_reference_wrapper< typename std20::remove_cvref< ObjectT >::type >::value
    )
>
nsel_constexpr auto invoke_member_object_impl( MemberT ClassT::* memobjptr, ObjectT && obj )
        noexcept( noexcept( (*std::forward< ObjectT >( obj )).*memobjptr ) )
        -> decltype( (*std::forward< ObjectT >( obj )).*memobjptr )
{
    return (*std::forward< ObjectT >( obj )).*memobjptr;
}

template< typename F, typename... Args
    nsel_REQUIRES_T(
        std::is_member_function_pointer< typename std20::remove_cvref< F >::type >::value
    )
>
nsel_constexpr auto invoke( F && f, Args && ... args )
        noexcept( noexcept( invoke_member_function_impl( std::forward< F >( f ), std::forward< Args >( args ) ... ) ) )
        -> decltype( invoke_member_function_impl( std::forward< F >( f ), std::forward< Args >( args ) ... ) )
{
    return invoke_member_function_impl( std::forward< F >( f ), std::forward< Args >( args ) ... );
}

template< typename F, typename... Args
    nsel_REQUIRES_T(
        std::is_member_object_pointer< typename std20::remove_cvref< F >::type >::value
    )
>
nsel_constexpr auto invoke( F && f, Args && ... args )
        noexcept( noexcept( invoke_member_object_impl( std::forward< F >( f ), std::forward< Args >( args ) ... ) ) )
        -> decltype( invoke_member_object_impl( std::forward< F >( f ), std::forward< Args >( args ) ... ) )
{
    return invoke_member_object_impl( std::forward< F >( f ), std::forward< Args >( args ) ... );
}

template< typename F, typename... Args
    nsel_REQUIRES_T(
        !std::is_member_function_pointer< typename std20::remove_cvref< F >::type >::value
        && !std::is_member_object_pointer< typename std20::remove_cvref< F >::type >::value
    )
>
nsel_constexpr auto invoke( F && f, Args && ... args )
        noexcept( noexcept( std::forward< F >( f )( std::forward< Args >( args ) ... ) ) )
        -> decltype( std::forward< F >( f )( std::forward< Args >( args ) ... ) )
{
    return std::forward< F >( f )( std::forward< Args >( args ) ... );
}

template< typename F, typename ... Args >
using invoke_result_nocvref_t = typename std20::remove_cvref< decltype( invoke( std::declval< F >(), std::declval< Args >()... ) ) >::type;

#if nsel_P2505R >= 5
template< typename F, typename ... Args >
using transform_invoke_result_t = typename std::remove_cv< decltype( invoke( std::declval< F >(), std::declval< Args >()... ) ) >::type;
#else
template< typename F, typename ... Args >
using transform_invoke_result_t = invoke_result_nocvref_t
#endif // nsel_P2505R >= 5

template< typename T >
struct valid_expected_value_type : std::integral_constant< bool, std::is_destructible< T >::value && !std::is_reference< T >::value && !std::is_array< T >::value > {};

#endif // nsel_P2505R >= 3
} // namespace detail

/// x.x.5 Unexpected object type; unexpected_type; C++17 and later can also use aliased type unexpected.

#if nsel_P0323R <= 2
template< typename E = std::exception_ptr >
class unexpected_type
#else
template< typename E >
class unexpected_type
#endif // nsel_P0323R
{
public:
    using error_type = E;

    // x.x.5.2.1 Constructors

//  unexpected_type() = delete;

    constexpr unexpected_type( unexpected_type const & ) = default;
    constexpr unexpected_type( unexpected_type && ) = default;

    template< typename... Args
        nsel_REQUIRES_T(
            std::is_constructible<E, Args&&...>::value
        )
    >
    constexpr explicit unexpected_type( nonstd_lite_in_place_t(E), Args &&... args )
    : m_error( std::forward<Args>( args )...)
    {}

    template< typename U, typename... Args
        nsel_REQUIRES_T(
            std::is_constructible<E, std::initializer_list<U>, Args&&...>::value
        )
    >
    constexpr explicit unexpected_type( nonstd_lite_in_place_t(E), std::initializer_list<U> il, Args &&... args )
    : m_error( il, std::forward<Args>( args )...)
    {}

    template< typename E2
        nsel_REQUIRES_T(
            std::is_constructible<E,E2>::value
            && !std::is_same< typename std20::remove_cvref<E2>::type, nonstd_lite_in_place_t(E2) >::value
            && !std::is_same< typename std20::remove_cvref<E2>::type, unexpected_type >::value
        )
    >
    constexpr explicit unexpected_type( E2 && error )
    : m_error( std::forward<E2>( error ) )
    {}

    template< typename E2
        nsel_REQUIRES_T(
            std::is_constructible<    E, E2>::value
            && !std::is_constructible<E, unexpected_type<E2>       &   >::value
            && !std::is_constructible<E, unexpected_type<E2>           >::value
            && !std::is_constructible<E, unexpected_type<E2> const &   >::value
            && !std::is_constructible<E, unexpected_type<E2> const     >::value
            && !std::is_convertible<     unexpected_type<E2>       &, E>::value
            && !std::is_convertible<     unexpected_type<E2>        , E>::value
            && !std::is_convertible<     unexpected_type<E2> const &, E>::value
            && !std::is_convertible<     unexpected_type<E2> const  , E>::value
            && !std::is_convertible< E2 const &, E>::value /*=> explicit */
        )
    >
    constexpr explicit unexpected_type( unexpected_type<E2> const & error )
    : m_error( E{ error.value() } )
    {}

    template< typename E2
        nsel_REQUIRES_T(
            std::is_constructible<    E, E2>::value
            && !std::is_constructible<E, unexpected_type<E2>       &   >::value
            && !std::is_constructible<E, unexpected_type<E2>           >::value
            && !std::is_constructible<E, unexpected_type<E2> const &   >::value
            && !std::is_constructible<E, unexpected_type<E2> const     >::value
            && !std::is_convertible<     unexpected_type<E2>       &, E>::value
            && !std::is_convertible<     unexpected_type<E2>        , E>::value
            && !std::is_convertible<     unexpected_type<E2> const &, E>::value
            && !std::is_convertible<     unexpected_type<E2> const  , E>::value
            &&  std::is_convertible< E2 const &, E>::value /*=> explicit */
        )
    >
    constexpr /*non-explicit*/ unexpected_type( unexpected_type<E2> const & error )
    : m_error( error.value() )
    {}

    template< typename E2
        nsel_REQUIRES_T(
            std::is_constructible<    E, E2>::value
            && !std::is_constructible<E, unexpected_type<E2>       &   >::value
            && !std::is_constructible<E, unexpected_type<E2>           >::value
            && !std::is_constructible<E, unexpected_type<E2> const &   >::value
            && !std::is_constructible<E, unexpected_type<E2> const     >::value
            && !std::is_convertible<     unexpected_type<E2>       &, E>::value
            && !std::is_convertible<     unexpected_type<E2>        , E>::value
            && !std::is_convertible<     unexpected_type<E2> const &, E>::value
            && !std::is_convertible<     unexpected_type<E2> const  , E>::value
            && !std::is_convertible< E2 const &, E>::value /*=> explicit */
        )
    >
    constexpr explicit unexpected_type( unexpected_type<E2> && error )
    : m_error( E{ std::move( error.value() ) } )
    {}

    template< typename E2
        nsel_REQUIRES_T(
            std::is_constructible<    E, E2>::value
            && !std::is_constructible<E, unexpected_type<E2>       &   >::value
            && !std::is_constructible<E, unexpected_type<E2>           >::value
            && !std::is_constructible<E, unexpected_type<E2> const &   >::value
            && !std::is_constructible<E, unexpected_type<E2> const     >::value
            && !std::is_convertible<     unexpected_type<E2>       &, E>::value
            && !std::is_convertible<     unexpected_type<E2>        , E>::value
            && !std::is_convertible<     unexpected_type<E2> const &, E>::value
            && !std::is_convertible<     unexpected_type<E2> const  , E>::value
            &&  std::is_convertible< E2 const &, E>::value /*=> non-explicit */
        )
    >
    constexpr /*non-explicit*/ unexpected_type( unexpected_type<E2> && error )
    : m_error( std::move( error.value() ) )
    {}

    // x.x.5.2.2 Assignment

    nsel_constexpr14 unexpected_type& operator=( unexpected_type const & ) = default;
    nsel_constexpr14 unexpected_type& operator=( unexpected_type && ) = default;

    template< typename E2 = E >
    nsel_constexpr14 unexpected_type & operator=( unexpected_type<E2> const & other )
    {
        unexpected_type{ other.value() }.swap( *this );
        return *this;
    }

    template< typename E2 = E >
    nsel_constexpr14 unexpected_type & operator=( unexpected_type<E2> && other )
    {
        unexpected_type{ std::move( other.value() ) }.swap( *this );
        return *this;
    }

    // x.x.5.2.3 Observers

    nsel_constexpr14 E & value() & noexcept
    {
        return m_error;
    }

    constexpr E const & value() const & noexcept
    {
        return m_error;
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490

    nsel_constexpr14 E && value() && noexcept
    {
        return std::move( m_error );
    }

    constexpr E const && value() const && noexcept
    {
        return std::move( m_error );
    }

#endif

    // x.x.5.2.4 Swap

    template< typename U=E >
    nsel_REQUIRES_R( void,
        std17::is_swappable<U>::value
    )
    swap( unexpected_type & other ) noexcept (
        std17::is_nothrow_swappable<U>::value
    )
    {
        using std::swap;
        swap( m_error, other.m_error );
    }

    // TODO: ??? unexpected_type: in-class friend operator==, !=

private:
    error_type m_error;
};

#if nsel_CPP17_OR_GREATER

/// template deduction guide:

template< typename E >
unexpected_type( E ) -> unexpected_type< E >;

#endif

/// class unexpected_type, std::exception_ptr specialization (P0323R2)

#if !nsel_CONFIG_NO_EXCEPTIONS
#if  nsel_P0323R <= 2

// TODO: Should expected be specialized for particular E types such as exception_ptr and how?
//       See p0323r7 2.1. Ergonomics, http://wg21.link/p0323
template<>
class unexpected_type< std::exception_ptr >
{
public:
    using error_type = std::exception_ptr;

    unexpected_type() = delete;

    ~unexpected_type(){}

    explicit unexpected_type( std::exception_ptr const & error )
    : m_error( error )
    {}

    explicit unexpected_type(std::exception_ptr && error )
    : m_error( std::move( error ) )
    {}

    template< typename E >
    explicit unexpected_type( E error )
    : m_error( std::make_exception_ptr( error ) )
    {}

    std::exception_ptr const & value() const
    {
        return m_error;
    }

    std::exception_ptr & value()
    {
        return m_error;
    }

private:
    std::exception_ptr m_error;
};

#endif // nsel_P0323R
#endif // !nsel_CONFIG_NO_EXCEPTIONS

/// x.x.4, Unexpected equality operators

template< typename E1, typename E2 >
constexpr bool operator==( unexpected_type<E1> const & x, unexpected_type<E2> const & y )
{
    return x.value() == y.value();
}

template< typename E1, typename E2 >
constexpr bool operator!=( unexpected_type<E1> const & x, unexpected_type<E2> const & y )
{
    return ! ( x == y );
}

#if nsel_P0323R <= 2

template< typename E >
constexpr bool operator<( unexpected_type<E> const & x, unexpected_type<E> const & y )
{
    return x.value() < y.value();
}

template< typename E >
constexpr bool operator>( unexpected_type<E> const & x, unexpected_type<E> const & y )
{
    return ( y < x );
}

template< typename E >
constexpr bool operator<=( unexpected_type<E> const & x, unexpected_type<E> const & y )
{
    return ! ( y < x  );
}

template< typename E >
constexpr bool operator>=( unexpected_type<E> const & x, unexpected_type<E> const & y )
{
    return ! ( x < y );
}

#endif // nsel_P0323R

/// x.x.5 Specialized algorithms

template< typename E
    nsel_REQUIRES_T(
        std17::is_swappable<E>::value
    )
>
void swap( unexpected_type<E> & x, unexpected_type<E> & y) noexcept ( noexcept ( x.swap(y) ) )
{
    x.swap( y );
}

#if nsel_P0323R <= 2

// unexpected: relational operators for std::exception_ptr:

inline constexpr bool operator<( unexpected_type<std::exception_ptr> const & /*x*/, unexpected_type<std::exception_ptr> const & /*y*/ )
{
    return false;
}

inline constexpr bool operator>( unexpected_type<std::exception_ptr> const & /*x*/, unexpected_type<std::exception_ptr> const & /*y*/ )
{
    return false;
}

inline constexpr bool operator<=( unexpected_type<std::exception_ptr> const & x, unexpected_type<std::exception_ptr> const & y )
{
    return ( x == y );
}

inline constexpr bool operator>=( unexpected_type<std::exception_ptr> const & x, unexpected_type<std::exception_ptr> const & y )
{
    return ( x == y );
}

#endif // nsel_P0323R

// unexpected: traits

#if nsel_P0323R <= 3

template< typename E>
struct is_unexpected : std::false_type {};

template< typename E>
struct is_unexpected< unexpected_type<E> > : std::true_type {};

#endif // nsel_P0323R

// unexpected: factory

// keep make_unexpected() removed in p0323r2 for pre-C++17:

template< typename E>
nsel_constexpr14 auto
make_unexpected( E && value ) -> unexpected_type< typename std::decay<E>::type >
{
    return unexpected_type< typename std::decay<E>::type >( std::forward<E>(value) );
}

#if nsel_P0323R <= 3

/*nsel_constexpr14*/ auto inline
make_unexpected_from_current_exception() -> unexpected_type< std::exception_ptr >
{
    return unexpected_type< std::exception_ptr >( std::current_exception() );
}

#endif // nsel_P0323R

/// x.x.6, x.x.7 expected access error

template< typename E >
class bad_expected_access;

/// x.x.7 bad_expected_access<void>: expected access error

template <>
class bad_expected_access< void > : public std::exception
{
public:
    explicit bad_expected_access()
    : std::exception()
    {}
};

/// x.x.6 bad_expected_access: expected access error

#if !nsel_CONFIG_NO_EXCEPTIONS

template< typename E >
class bad_expected_access : public bad_expected_access< void >
{
public:
    using error_type = E;

    explicit bad_expected_access( error_type error )
    : m_error( error )
    {}

    virtual char const * what() const noexcept override
    {
        return "bad_expected_access";
    }

    nsel_constexpr14 error_type & error() &
    {
        return m_error;
    }

    constexpr error_type const & error() const &
    {
        return m_error;
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490

    nsel_constexpr14 error_type && error() &&
    {
        return std::move( m_error );
    }

    constexpr error_type const && error() const &&
    {
        return std::move( m_error );
    }

#endif

private:
    error_type m_error;
};

#endif // nsel_CONFIG_NO_EXCEPTIONS

/// x.x.8 unexpect tag, in_place_unexpected tag: construct an error

struct unexpect_t{};
using in_place_unexpected_t = unexpect_t;

nsel_inline17 constexpr unexpect_t unexpect{};
nsel_inline17 constexpr unexpect_t in_place_unexpected{};

/// class error_traits

#if nsel_CONFIG_NO_EXCEPTIONS

namespace detail {
    inline bool text( char const * /*text*/ ) { return true; }
}

template< typename Error >
struct error_traits
{
    static void rethrow( Error const & /*e*/ )
    {
#if nsel_CONFIG_NO_EXCEPTIONS_SEH
        RaiseException( EXCEPTION_ACCESS_VIOLATION, EXCEPTION_NONCONTINUABLE, 0, NULL );
#else
        assert( false && detail::text("throw bad_expected_access<Error>{ e };") );
#endif
    }
};

template<>
struct error_traits< std::exception_ptr >
{
    static void rethrow( std::exception_ptr const & /*e*/ )
    {
#if nsel_CONFIG_NO_EXCEPTIONS_SEH
        RaiseException( EXCEPTION_ACCESS_VIOLATION, EXCEPTION_NONCONTINUABLE, 0, NULL );
#else
        assert( false && detail::text("throw bad_expected_access<std::exception_ptr>{ e };") );
#endif
    }
};

template<>
struct error_traits< std::error_code >
{
    static void rethrow( std::error_code const & /*e*/ )
    {
#if nsel_CONFIG_NO_EXCEPTIONS_SEH
        RaiseException( EXCEPTION_ACCESS_VIOLATION, EXCEPTION_NONCONTINUABLE, 0, NULL );
#else
        assert( false && detail::text("throw std::system_error( e );") );
#endif
    }
};

#else // nsel_CONFIG_NO_EXCEPTIONS

template< typename Error >
struct error_traits
{
    static void rethrow( Error const & e )
    {
        throw bad_expected_access<Error>{ e };
    }
};

template<>
struct error_traits< std::exception_ptr >
{
    static void rethrow( std::exception_ptr const & e )
    {
        std::rethrow_exception( e );
    }
};

template<>
struct error_traits< std::error_code >
{
    static void rethrow( std::error_code const & e )
    {
        throw std::system_error( e );
    }
};

#endif // nsel_CONFIG_NO_EXCEPTIONS

#if nsel_P2505R >= 3
namespace detail {

// from https://en.cppreference.com/w/cpp/utility/expected/unexpected:
// "the type of the unexpected value. The type must not be an array type, a non-object type, a specialization of std::unexpected, or a cv-qualified type."
template< typename T >
struct valid_unexpected_type : std::integral_constant< bool,
    std::is_same< T, typename std20::remove_cvref< T >::type >::value
    && std::is_object< T >::value
    && !std::is_array< T >::value
> {};

template< typename T >
struct valid_unexpected_type< unexpected_type< T > > : std::false_type {};

} // namespace detail
#endif // nsel_P2505R >= 3

} // namespace expected_lite

// provide nonstd::unexpected_type:

using expected_lite::unexpected_type;

namespace expected_lite {

/// class expected

#if nsel_P0323R <= 2
template< typename T, typename E = std::exception_ptr >
class expected
#else
template< typename T, typename E >
class expected
#endif // nsel_P0323R
{
private:
    template< typename, typename > friend class expected;

public:
    using value_type = T;
    using error_type = E;
    using unexpected_type = nonstd::unexpected_type<E>;

    template< typename U >
    struct rebind
    {
        using type = expected<U, error_type>;
    };

    // x.x.4.1 constructors

    nsel_REQUIRES_0(
        std::is_default_constructible<T>::value
    )
    nsel_constexpr14 expected()
    : contained( true )
    {
        contained.construct_value();
    }

    nsel_constexpr14 expected( expected const & ) = default;
    nsel_constexpr14 expected( expected &&      ) = default;

    template< typename U, typename G
        nsel_REQUIRES_T(
            std::is_constructible<    T, U const &>::value
            &&  std::is_constructible<E, G const &>::value
            && !std::is_constructible<T, expected<U, G>       &    >::value
            && !std::is_constructible<T, expected<U, G>       &&   >::value
            && !std::is_constructible<T, expected<U, G> const &    >::value
            && !std::is_constructible<T, expected<U, G> const &&   >::value
            && !std::is_convertible<     expected<U, G>       & , T>::value
            && !std::is_convertible<     expected<U, G>       &&, T>::value
            && !std::is_convertible<     expected<U, G> const & , T>::value
            && !std::is_convertible<     expected<U, G> const &&, T>::value
            && (!std::is_convertible<U const &, T>::value || !std::is_convertible<G const &, E>::value ) /*=> explicit */
        )
    >
    nsel_constexpr14 explicit expected( expected<U, G> const & other )
    : contained( other.has_value() )
    {
        if ( has_value() ) contained.construct_value( T{ other.contained.value() } );
        else               contained.construct_error( E{ other.contained.error() } );
    }

    template< typename U, typename G
        nsel_REQUIRES_T(
            std::is_constructible<    T, U const &>::value
            &&  std::is_constructible<E, G const &>::value
            && !std::is_constructible<T, expected<U, G>       &    >::value
            && !std::is_constructible<T, expected<U, G>       &&   >::value
            && !std::is_constructible<T, expected<U, G> const &    >::value
            && !std::is_constructible<T, expected<U, G> const &&   >::value
            && !std::is_convertible<     expected<U, G>       & , T>::value
            && !std::is_convertible<     expected<U, G>       &&, T>::value
            && !std::is_convertible<     expected<U, G> const  &, T>::value
            && !std::is_convertible<     expected<U, G> const &&, T>::value
            && !(!std::is_convertible<U const &, T>::value || !std::is_convertible<G const &, E>::value ) /*=> non-explicit */
        )
    >
    nsel_constexpr14 /*non-explicit*/ expected( expected<U, G> const & other )
    : contained( other.has_value() )
    {
        if ( has_value() ) contained.construct_value( other.contained.value() );
        else               contained.construct_error( other.contained.error() );
    }

    template< typename U, typename G
        nsel_REQUIRES_T(
            std::is_constructible<    T, U>::value
            &&  std::is_constructible<E, G>::value
            && !std::is_constructible<T, expected<U, G>       &    >::value
            && !std::is_constructible<T, expected<U, G>       &&   >::value
            && !std::is_constructible<T, expected<U, G> const &    >::value
            && !std::is_constructible<T, expected<U, G> const &&   >::value
            && !std::is_convertible<     expected<U, G>       & , T>::value
            && !std::is_convertible<     expected<U, G>       &&, T>::value
            && !std::is_convertible<     expected<U, G> const & , T>::value
            && !std::is_convertible<     expected<U, G> const &&, T>::value
            && (!std::is_convertible<U, T>::value || !std::is_convertible<G, E>::value ) /*=> explicit */
        )
    >
    nsel_constexpr14 explicit expected( expected<U, G> && other )
    : contained( other.has_value() )
    {
        if ( has_value() ) contained.construct_value( T{ std::move( other.contained.value() ) } );
        else               contained.construct_error( E{ std::move( other.contained.error() ) } );
    }

    template< typename U, typename G
        nsel_REQUIRES_T(
            std::is_constructible<    T, U>::value
            &&  std::is_constructible<E, G>::value
            && !std::is_constructible<T, expected<U, G>      &     >::value
            && !std::is_constructible<T, expected<U, G>      &&    >::value
            && !std::is_constructible<T, expected<U, G> const &    >::value
            && !std::is_constructible<T, expected<U, G> const &&   >::value
            && !std::is_convertible<     expected<U, G>       & , T>::value
            && !std::is_convertible<     expected<U, G>       &&, T>::value
            && !std::is_convertible<     expected<U, G> const & , T>::value
            && !std::is_convertible<     expected<U, G> const &&, T>::value
            && !(!std::is_convertible<U, T>::value || !std::is_convertible<G, E>::value ) /*=> non-explicit */
        )
    >
    nsel_constexpr14 /*non-explicit*/ expected( expected<U, G> && other )
    : contained( other.has_value() )
    {
        if ( has_value() ) contained.construct_value( std::move( other.contained.value() ) );
        else               contained.construct_error( std::move( other.contained.error() ) );
    }

    template< typename U = T
        nsel_REQUIRES_T(
            std::is_copy_constructible<U>::value
        )
    >
    nsel_constexpr14 expected( value_type const & value )
    : contained( true )
    {
        contained.construct_value( value );
    }

    template< typename U = T
        nsel_REQUIRES_T(
            std::is_constructible<T,U&&>::value
            && !std::is_same<typename std20::remove_cvref<U>::type, nonstd_lite_in_place_t(U)>::value
            && !std::is_same<        expected<T,E>     , typename std20::remove_cvref<U>::type>::value
            && !std::is_same<nonstd::unexpected_type<E>, typename std20::remove_cvref<U>::type>::value
            && !std::is_convertible<U&&,T>::value /*=> explicit */
        )
    >
    nsel_constexpr14 explicit expected( U && value ) noexcept
    (
        std::is_nothrow_move_constructible<U>::value &&
        std::is_nothrow_move_constructible<E>::value
    )
    : contained( true )
    {
        contained.construct_value( T{ std::forward<U>( value ) } );
    }

    template< typename U = T
        nsel_REQUIRES_T(
            std::is_constructible<T,U&&>::value
            && !std::is_same<typename std20::remove_cvref<U>::type, nonstd_lite_in_place_t(U)>::value
            && !std::is_same<        expected<T,E>     , typename std20::remove_cvref<U>::type>::value
            && !std::is_same<nonstd::unexpected_type<E>, typename std20::remove_cvref<U>::type>::value
            &&  std::is_convertible<U&&,T>::value /*=> non-explicit */
        )
    >
    nsel_constexpr14 /*non-explicit*/ expected( U && value ) noexcept
    (
        std::is_nothrow_move_constructible<U>::value &&
        std::is_nothrow_move_constructible<E>::value
    )
    : contained( true )
    {
        contained.construct_value( std::forward<U>( value ) );
    }

    // construct error:

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_constructible<E, G const &   >::value
            && !std::is_convertible< G const &, E>::value /*=> explicit */
        )
    >
    nsel_constexpr14 explicit expected( nonstd::unexpected_type<G> const & error )
    : contained( false )
    {
        contained.construct_error( E{ error.value() } );
    }

    template< typename G = E
            nsel_REQUIRES_T(
            std::is_constructible<E, G const &   >::value
            && std::is_convertible<  G const &, E>::value /*=> non-explicit */
        )
    >
    nsel_constexpr14 /*non-explicit*/ expected( nonstd::unexpected_type<G> const & error )
    : contained( false )
    {
        contained.construct_error( error.value() );
    }

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_constructible<E, G&&   >::value
            && !std::is_convertible< G&&, E>::value /*=> explicit */
        )
    >
    nsel_constexpr14 explicit expected( nonstd::unexpected_type<G> && error )
    : contained( false )
    {
        contained.construct_error( E{ std::move( error.value() ) } );
    }

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_constructible<E, G&&   >::value
            && std::is_convertible<  G&&, E>::value /*=> non-explicit */
        )
    >
    nsel_constexpr14 /*non-explicit*/ expected( nonstd::unexpected_type<G> && error )
    : contained( false )
    {
        contained.construct_error( std::move( error.value() ) );
    }

    // in-place construction, value

    template< typename... Args
        nsel_REQUIRES_T(
            std::is_constructible<T, Args&&...>::value
        )
    >
    nsel_constexpr14 explicit expected( nonstd_lite_in_place_t(T), Args&&... args )
    : contained( true )
    {
        contained.emplace_value( std::forward<Args>( args )... );
    }

    template< typename U, typename... Args
        nsel_REQUIRES_T(
            std::is_constructible<T, std::initializer_list<U>, Args&&...>::value
        )
    >
    nsel_constexpr14 explicit expected( nonstd_lite_in_place_t(T), std::initializer_list<U> il, Args&&... args )
    : contained( true )
    {
        contained.emplace_value( il, std::forward<Args>( args )... );
    }

    // in-place construction, error

    template< typename... Args
        nsel_REQUIRES_T(
            std::is_constructible<E, Args&&...>::value
        )
    >
    nsel_constexpr14 explicit expected( unexpect_t, Args&&... args )
    : contained( false )
    {
        contained.emplace_error( std::forward<Args>( args )... );
    }

    template< typename U, typename... Args
        nsel_REQUIRES_T(
            std::is_constructible<E, std::initializer_list<U>, Args&&...>::value
        )
    >
    nsel_constexpr14 explicit expected( unexpect_t, std::initializer_list<U> il, Args&&... args )
    : contained( false )
    {
        contained.emplace_error( il, std::forward<Args>( args )... );
    }

    // x.x.4.2 destructor

    // TODO: ~expected: triviality
    // Effects: If T is not cv void and is_trivially_destructible_v<T> is false and bool(*this), calls val.~T(). If is_trivially_destructible_v<E> is false and !bool(*this), calls unexpect.~unexpected<E>().
    // Remarks: If either T is cv void or is_trivially_destructible_v<T> is true, and is_trivially_destructible_v<E> is true, then this destructor shall be a trivial destructor.

    ~expected()
    {
        if ( has_value() ) contained.destruct_value();
        else               contained.destruct_error();
    }

    // x.x.4.3 assignment

    expected & operator=( expected const & other )
    {
        expected( other ).swap( *this );
        return *this;
    }

    expected & operator=( expected && other ) noexcept
    (
        std::is_nothrow_move_constructible<   T>::value
        && std::is_nothrow_move_assignable<   T>::value
        && std::is_nothrow_move_constructible<E>::value     // added for missing
        && std::is_nothrow_move_assignable<   E>::value )   //   nothrow above
    {
        expected( std::move( other ) ).swap( *this );
        return *this;
    }

    template< typename U
        nsel_REQUIRES_T(
            !std::is_same<expected<T,E>, typename std20::remove_cvref<U>::type>::value
            && std17::conjunction<std::is_scalar<T>, std::is_same<T, std::decay<U>> >::value
            && std::is_constructible<T ,U>::value
            && std::is_assignable<   T&,U>::value
            && std::is_nothrow_move_constructible<E>::value )
    >
    expected & operator=( U && value )
    {
        expected( std::forward<U>( value ) ).swap( *this );
        return *this;
    }

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_constructible<E, G const&>::value &&
            std::is_copy_constructible<G>::value    // TODO: std::is_nothrow_copy_constructible<G>
            && std::is_copy_assignable<G>::value
        )
    >
    expected & operator=( nonstd::unexpected_type<G> const & error )
    {
        expected( unexpect, error.value() ).swap( *this );
        return *this;
    }

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_constructible<E, G&&>::value &&
            std::is_move_constructible<G>::value    // TODO: std::is_nothrow_move_constructible<G>
            && std::is_move_assignable<G>::value
        )
    >
    expected & operator=( nonstd::unexpected_type<G> && error )
    {
        expected( unexpect, std::move( error.value() ) ).swap( *this );
        return *this;
    }

    template< typename... Args
        nsel_REQUIRES_T(
            std::is_nothrow_constructible<T, Args&&...>::value
        )
    >
    value_type & emplace( Args &&... args )
    {
        expected( nonstd_lite_in_place(T), std::forward<Args>(args)... ).swap( *this );
        return value();
    }

    template< typename U, typename... Args
        nsel_REQUIRES_T(
            std::is_nothrow_constructible<T, std::initializer_list<U>&, Args&&...>::value
        )
    >
    value_type & emplace( std::initializer_list<U> il, Args &&... args )
    {
        expected( nonstd_lite_in_place(T), il, std::forward<Args>(args)... ).swap( *this );
        return value();
    }

    // x.x.4.4 swap

    template< typename U=T, typename G=E >
    nsel_REQUIRES_R( void,
        std17::is_swappable<   U>::value
        && std17::is_swappable<G>::value
        && ( std::is_move_constructible<U>::value || std::is_move_constructible<G>::value )
    )
    swap( expected & other ) noexcept
    (
        std::is_nothrow_move_constructible<T>::value && std17::is_nothrow_swappable<T&>::value &&
        std::is_nothrow_move_constructible<E>::value && std17::is_nothrow_swappable<E&>::value
    )
    {
        using std::swap;

        if      (   bool(*this) &&   bool(other) ) { swap( contained.value(), other.contained.value() ); }
        else if ( ! bool(*this) && ! bool(other) ) { swap( contained.error(), other.contained.error() ); }
        else if (   bool(*this) && ! bool(other) ) { error_type t( std::move( other.error() ) );
                                                     other.contained.destruct_error();
                                                     other.contained.construct_value( std::move( contained.value() ) );
                                                     contained.destruct_value();
                                                     contained.construct_error( std::move( t ) );
                                                     bool has_value = contained.has_value();
                                                     bool other_has_value = other.has_value();
                                                     other.contained.set_has_value(has_value);
                                                     contained.set_has_value(other_has_value);
                                                   }
        else if ( ! bool(*this) &&   bool(other) ) { other.swap( *this ); }
    }

    // x.x.4.5 observers

    constexpr value_type const * operator ->() const
    {
        return assert( has_value() ), contained.value_ptr();
    }

    value_type * operator ->()
    {
        return assert( has_value() ), contained.value_ptr();
    }

    constexpr value_type const & operator *() const &
    {
        return assert( has_value() ), contained.value();
    }

    value_type & operator *() &
    {
        return assert( has_value() ), contained.value();
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490

    constexpr value_type const && operator *() const &&
    {
        return std::move( ( assert( has_value() ), contained.value() ) );
    }

    nsel_constexpr14 value_type && operator *() &&
    {
        return std::move( ( assert( has_value() ), contained.value() ) );
    }

#endif

    constexpr explicit operator bool() const noexcept
    {
        return has_value();
    }

    constexpr bool has_value() const noexcept
    {
        return contained.has_value();
    }

    constexpr value_type const & value() const &
    {
        return has_value()
            ? ( contained.value() )
            : ( error_traits<error_type>::rethrow( contained.error() ), contained.value() );
    }

    value_type & value() &
    {
        return has_value()
            ? ( contained.value() )
            : ( error_traits<error_type>::rethrow( contained.error() ), contained.value() );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490

    constexpr value_type const && value() const &&
    {
        return std::move( has_value()
            ? ( contained.value() )
            : ( error_traits<error_type>::rethrow( contained.error() ), contained.value() ) );
    }

    nsel_constexpr14 value_type && value() &&
    {
        return std::move( has_value()
            ? ( contained.value() )
            : ( error_traits<error_type>::rethrow( contained.error() ), contained.value() ) );
    }

#endif

    constexpr error_type const & error() const &
    {
        return assert( ! has_value() ), contained.error();
    }

    error_type & error() &
    {
        return assert( ! has_value() ), contained.error();
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490

    constexpr error_type const && error() const &&
    {
        return std::move( ( assert( ! has_value() ), contained.error() ) );
    }

    error_type && error() &&
    {
        return std::move( ( assert( ! has_value() ), contained.error() ) );
    }

#endif

    constexpr unexpected_type get_unexpected() const
    {
        return make_unexpected( contained.error() );
    }

    template< typename Ex >
    bool has_exception() const
    {
        using ContainedEx = typename std::remove_reference< decltype( get_unexpected().value() ) >::type;
        return ! has_value() && std::is_base_of< Ex, ContainedEx>::value;
    }

    template< typename U
        nsel_REQUIRES_T(
            std::is_copy_constructible< T>::value
            && std::is_convertible<U&&, T>::value
        )
    >
    value_type value_or( U && v ) const &
    {
        return has_value()
            ? contained.value()
            : static_cast<T>( std::forward<U>( v ) );
    }

    template< typename U
        nsel_REQUIRES_T(
            std::is_move_constructible< T>::value
            && std::is_convertible<U&&, T>::value
        )
    >
    value_type value_or( U && v ) &&
    {
        return has_value()
            ? std::move( contained.value() )
            : static_cast<T>( std::forward<U>( v ) );
    }

#if nsel_P2505R >= 4
    template< typename G = E
        nsel_REQUIRES_T(
            std::is_copy_constructible< E >::value
            && std::is_convertible< G, E >::value
        )
    >
    nsel_constexpr error_type error_or( G && e ) const &
    {
        return has_value()
            ? static_cast< E >( std::forward< G >( e ) )
            : contained.error();
    }

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_move_constructible< E >::value
            && std::is_convertible< G, E >::value
        )
    >
    nsel_constexpr14 error_type error_or( G && e ) &&
    {
        return has_value()
            ? static_cast< E >( std::forward< G >( e ) )
            : std::move( contained.error() );
    }
#endif // nsel_P2505R >= 4

#if nsel_P2505R >= 3
    // Monadic operations (P2505)
    template< typename F
        nsel_REQUIRES_T(
            detail::is_expected < detail::invoke_result_nocvref_t< F, value_type & > > ::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F, value_type & >::error_type, error_type >::value
            && std::is_constructible< error_type, error_type & >::value
        )
    >
    nsel_constexpr14 detail::invoke_result_nocvref_t< F, value_type & > and_then( F && f ) &
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, value_type & >( detail::invoke( std::forward< F >( f ), value() ) )
            : detail::invoke_result_nocvref_t< F, value_type & >( unexpect, error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, const value_type & > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F, const value_type & >::error_type, error_type >::value
            && std::is_constructible< error_type, const error_type & >::value
        )
    >
    nsel_constexpr detail::invoke_result_nocvref_t< F, const value_type & > and_then( F && f ) const &
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, const value_type & >( detail::invoke( std::forward< F >( f ), value() ) )
            : detail::invoke_result_nocvref_t< F, const value_type & >( unexpect, error() );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490
    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, value_type && > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F, value_type && >::error_type, error_type >::value
            && std::is_constructible< error_type, error_type && >::value
        )
    >
    nsel_constexpr14 detail::invoke_result_nocvref_t< F, value_type && > and_then( F && f ) &&
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, value_type && >( detail::invoke( std::forward< F >( f ), std::move( value() ) ) )
            : detail::invoke_result_nocvref_t< F, value_type && >( unexpect, std::move( error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, const value_type && > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F, const value_type & >::error_type, error_type >::value
            && std::is_constructible< error_type, const error_type && >::value
        )
    >
    nsel_constexpr detail::invoke_result_nocvref_t< F, const value_type && > and_then( F && f ) const &&
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, const value_type && >( detail::invoke( std::forward< F >( f ), std::move( value() ) ) )
            : detail::invoke_result_nocvref_t< F, const value_type && >( unexpect, std::move( error() ) );
    }
#endif

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, error_type & > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F, error_type & >::value_type, value_type >::value
            && std::is_constructible< value_type, value_type & >::value
        )
    >
    nsel_constexpr14 detail::invoke_result_nocvref_t< F, error_type & > or_else( F && f ) &
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, error_type & >( value() )
            : detail::invoke_result_nocvref_t< F, error_type & >( detail::invoke( std::forward< F >( f ), error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, const error_type & > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F, const error_type & >::value_type, value_type >::value
            && std::is_constructible< value_type, const value_type & >::value
        )
    >
    nsel_constexpr detail::invoke_result_nocvref_t< F, const error_type & > or_else( F && f ) const &
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, const error_type & >( value() )
            : detail::invoke_result_nocvref_t< F, const error_type & >( detail::invoke( std::forward< F >( f ), error() ) );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490
    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, error_type && > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F, error_type && >::value_type, value_type >::value
            && std::is_constructible< value_type, value_type && >::value
        )
    >
    nsel_constexpr14 detail::invoke_result_nocvref_t< F, error_type && > or_else( F && f ) &&
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, error_type && >( std::move( value() ) )
            : detail::invoke_result_nocvref_t< F, error_type && >( detail::invoke( std::forward< F >( f ), std::move( error() ) ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, const error_type && > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F, const error_type && >::value_type, value_type >::value
            && std::is_constructible< value_type, const value_type && >::value
        )
    >
    nsel_constexpr detail::invoke_result_nocvref_t< F, const error_type && > or_else( F && f ) const &&
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, const error_type && >( std::move( value() ) )
            : detail::invoke_result_nocvref_t< F, const error_type && >( detail::invoke( std::forward< F >( f ), std::move( error() ) ) );
    }
#endif

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, error_type & >::value
            && !std::is_void< detail::transform_invoke_result_t< F, value_type & > >::value
            && detail::valid_expected_value_type< detail::transform_invoke_result_t< F, value_type & > >::value
        )
    >
    nsel_constexpr14 expected< detail::transform_invoke_result_t< F, value_type & >, error_type > transform( F && f ) &
    {
        return has_value()
            ? expected< detail::transform_invoke_result_t< F, value_type & >, error_type >( detail::invoke( std::forward< F >( f ), **this ) )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, error_type & >::value
            && std::is_void< detail::transform_invoke_result_t< F, value_type & > >::value
        )
    >
    nsel_constexpr14 expected< void, error_type > transform( F && f ) &
    {
        return has_value()
            ? ( detail::invoke( std::forward< F >( f ), **this ), expected< void, error_type >() )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, const error_type & >::value
            && !std::is_void< detail::transform_invoke_result_t< F, const value_type & > >::value
            && detail::valid_expected_value_type< detail::transform_invoke_result_t< F, const value_type & > >::value
        )
    >
    nsel_constexpr expected< detail::transform_invoke_result_t< F, const value_type & >, error_type > transform( F && f ) const &
    {
        return has_value()
            ? expected< detail::transform_invoke_result_t< F, const value_type & >, error_type >( detail::invoke( std::forward< F >( f ), **this ) )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, const error_type & >::value
            && std::is_void< detail::transform_invoke_result_t< F, const value_type & > >::value
        )
    >
    nsel_constexpr expected< void, error_type > transform( F && f ) const &
    {
        return has_value()
            ? ( detail::invoke( std::forward< F >( f ), **this ), expected< void, error_type >() )
            : make_unexpected( error() );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490
    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, error_type && >::value
            && !std::is_void< detail::transform_invoke_result_t< F, value_type && > >::value
            && detail::valid_expected_value_type< detail::transform_invoke_result_t< F, value_type && > >::value
        )
    >
    nsel_constexpr14 expected< detail::transform_invoke_result_t< F, value_type && >, error_type > transform( F && f ) &&
    {
        return has_value()
            ? expected< detail::transform_invoke_result_t< F, value_type && >, error_type >( detail::invoke( std::forward< F >( f ), std::move( **this ) ) )
            : make_unexpected( std::move( error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, error_type && >::value
            && std::is_void< detail::transform_invoke_result_t< F, value_type && > >::value
        )
    >
    nsel_constexpr14 expected< void, error_type > transform( F && f ) &&
    {
        return has_value()
            ? ( detail::invoke( std::forward< F >( f ), **this ), expected< void, error_type >() )
            : make_unexpected( std::move( error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, const error_type && >::value
            && !std::is_void< detail::transform_invoke_result_t< F, const value_type && > >::value
            && detail::valid_expected_value_type< detail::transform_invoke_result_t< F, const value_type && > >::value
        )
    >
    nsel_constexpr expected< detail::transform_invoke_result_t< F, const value_type && >, error_type > transform( F && f ) const &&
    {
        return has_value()
            ? expected< detail::transform_invoke_result_t< F, const value_type && >, error_type >( detail::invoke( std::forward< F >( f ), std::move( **this ) ) )
            : make_unexpected( std::move( error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, const error_type && >::value
            && std::is_void< detail::transform_invoke_result_t< F, const value_type && > >::value
        )
    >
    nsel_constexpr expected< void, error_type > transform( F && f ) const &&
    {
        return has_value()
            ? ( detail::invoke( std::forward< F >( f ), **this ), expected< void, error_type >() )
            : make_unexpected( std::move( error() ) );
    }
#endif

    template<typename F
        nsel_REQUIRES_T(
            detail::valid_unexpected_type< detail::transform_invoke_result_t< F, error_type & > >::value
            && std::is_constructible< value_type, value_type & >::value
        )
    >
    nsel_constexpr14 expected< value_type, detail::transform_invoke_result_t< F, error_type & > > transform_error( F && f ) &
    {
        return has_value()
            ? expected< value_type, detail::transform_invoke_result_t< F, error_type & > >( in_place, **this )
            : make_unexpected( detail::invoke( std::forward< F >( f ), error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::valid_unexpected_type< detail::transform_invoke_result_t< F, const error_type & > >::value
            && std::is_constructible< value_type, const value_type & >::value
        )
    >
    nsel_constexpr expected< value_type, detail::transform_invoke_result_t< F, const error_type & > > transform_error( F && f ) const &
    {
        return has_value()
            ? expected< value_type, detail::transform_invoke_result_t< F, const error_type & > >( in_place, **this )
            : make_unexpected( detail::invoke( std::forward< F >( f ), error() ) );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490
    template<typename F
        nsel_REQUIRES_T(
            detail::valid_unexpected_type< detail::transform_invoke_result_t< F, error_type && > >::value
            && std::is_constructible< value_type, value_type && >::value
        )
    >
    nsel_constexpr14 expected< value_type, detail::transform_invoke_result_t< F, error_type && > > transform_error( F && f ) &&
    {
        return has_value()
            ? expected< value_type, detail::transform_invoke_result_t< F, error_type && > >( in_place, std::move( **this ) )
            : make_unexpected( detail::invoke( std::forward< F >( f ), std::move( error() ) ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::valid_unexpected_type< detail::transform_invoke_result_t< F, const error_type && > >::value
            && std::is_constructible< value_type, const value_type && >::value
        )
    >
    nsel_constexpr expected< value_type, detail::transform_invoke_result_t< F, const error_type && > > transform_error( F && f ) const &&
    {
        return has_value()
            ? expected< value_type, detail::transform_invoke_result_t< F, const error_type && > >( in_place, std::move( **this ) )
            : make_unexpected( detail::invoke( std::forward< F >( f ), std::move( error() ) ) );
    }
#endif
#endif // nsel_P2505R >= 3
    // unwrap()

//  template <class U, class E>
//  constexpr expected<U,E> expected<expected<U,E>,E>::unwrap() const&;

//  template <class T, class E>
//  constexpr expected<T,E> expected<T,E>::unwrap() const&;

//  template <class U, class E>
//  expected<U,E> expected<expected<U,E>, E>::unwrap() &&;

//  template <class T, class E>
//  template expected<T,E> expected<T,E>::unwrap() &&;

    // factories

//  template< typename Ex, typename F>
//  expected<T,E> catch_exception(F&& f);

//  template< typename F>
//  expected<decltype(func(declval<T>())),E> map(F&& func) ;

//  template< typename F>
//  'see below' bind(F&& func);

//  template< typename F>
//  expected<T,E> catch_error(F&& f);

//  template< typename F>
//  'see below' then(F&& func);

private:
    detail::storage_t
    <
        T
        ,E
        , std::is_copy_constructible<T>::value && std::is_copy_constructible<E>::value
        , std::is_move_constructible<T>::value && std::is_move_constructible<E>::value
    >
    contained;
};

/// class expected, void specialization

template< typename E >
class expected<void, E>
{
private:
    template< typename, typename > friend class expected;

public:
    using value_type = void;
    using error_type = E;
    using unexpected_type = nonstd::unexpected_type<E>;

    // x.x.4.1 constructors

    constexpr expected() noexcept
        : contained( true )
    {}

    nsel_constexpr14 expected( expected const & other ) = default;
    nsel_constexpr14 expected( expected &&      other ) = default;

    constexpr explicit expected( nonstd_lite_in_place_t(void) )
        : contained( true )
    {}

    template< typename G = E
        nsel_REQUIRES_T(
            !std::is_convertible<G const &, E>::value /*=> explicit */
        )
    >
    nsel_constexpr14 explicit expected( nonstd::unexpected_type<G> const & error )
        : contained( false )
    {
        contained.construct_error( E{ error.value() } );
    }

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_convertible<G const &, E>::value /*=> non-explicit */
        )
    >
    nsel_constexpr14 /*non-explicit*/ expected( nonstd::unexpected_type<G> const & error )
        : contained( false )
    {
        contained.construct_error( error.value() );
    }

    template< typename G = E
        nsel_REQUIRES_T(
            !std::is_convertible<G&&, E>::value /*=> explicit */
        )
    >
    nsel_constexpr14 explicit expected( nonstd::unexpected_type<G> && error )
        : contained( false )
    {
        contained.construct_error( E{ std::move( error.value() ) } );
    }

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_convertible<G&&, E>::value /*=> non-explicit */
        )
    >
    nsel_constexpr14 /*non-explicit*/ expected( nonstd::unexpected_type<G> && error )
        : contained( false )
    {
        contained.construct_error( std::move( error.value() ) );
    }

    template< typename... Args
        nsel_REQUIRES_T(
            std::is_constructible<E, Args&&...>::value
        )
    >
    nsel_constexpr14 explicit expected( unexpect_t, Args&&... args )
        : contained( false )
    {
        contained.emplace_error( std::forward<Args>( args )... );
    }

    template< typename U, typename... Args
        nsel_REQUIRES_T(
            std::is_constructible<E, std::initializer_list<U>, Args&&...>::value
        )
    >
    nsel_constexpr14 explicit expected( unexpect_t, std::initializer_list<U> il, Args&&... args )
        : contained( false )
    {
        contained.emplace_error( il, std::forward<Args>( args )... );
    }

    // destructor

    ~expected()
    {
        if ( ! has_value() )
        {
            contained.destruct_error();
        }
    }

    // x.x.4.3 assignment

    expected & operator=( expected const & other )
    {
        expected( other ).swap( *this );
        return *this;
    }

    expected & operator=( expected && other ) noexcept
    (
        std::is_nothrow_move_assignable<E>::value &&
        std::is_nothrow_move_constructible<E>::value )
    {
        expected( std::move( other ) ).swap( *this );
        return *this;
    }

    void emplace()
    {
        expected().swap( *this );
    }

    // x.x.4.4 swap

    template< typename G = E >
    nsel_REQUIRES_R( void,
        std17::is_swappable<G>::value
        && std::is_move_constructible<G>::value
    )
    swap( expected & other ) noexcept
    (
        std::is_nothrow_move_constructible<E>::value && std17::is_nothrow_swappable<E&>::value
    )
    {
        using std::swap;

        if      ( ! bool(*this) && ! bool(other) ) { swap( contained.error(), other.contained.error() ); }
        else if (   bool(*this) && ! bool(other) ) { contained.construct_error( std::move( other.error() ) );
                                                     bool has_value = contained.has_value();
                                                     bool other_has_value = other.has_value();
                                                     other.contained.set_has_value(has_value);
                                                     contained.set_has_value(other_has_value);
                                                     }
        else if ( ! bool(*this) &&   bool(other) ) { other.swap( *this ); }
    }

    // x.x.4.5 observers

    constexpr explicit operator bool() const noexcept
    {
        return has_value();
    }

    constexpr bool has_value() const noexcept
    {
        return contained.has_value();
    }

    void value() const
    {
        if ( ! has_value() )
        {
            error_traits<error_type>::rethrow( contained.error() );
        }
    }

    constexpr error_type const & error() const &
    {
        return assert( ! has_value() ), contained.error();
    }

    error_type & error() &
    {
        return assert( ! has_value() ), contained.error();
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490

    constexpr error_type const && error() const &&
    {
        return std::move( ( assert( ! has_value() ), contained.error() ) );
    }

    error_type && error() &&
    {
        return std::move( ( assert( ! has_value() ), contained.error() ) );
    }

#endif

    constexpr unexpected_type get_unexpected() const
    {
        return make_unexpected( contained.error() );
    }

    template< typename Ex >
    bool has_exception() const
    {
        using ContainedEx = typename std::remove_reference< decltype( get_unexpected().value() ) >::type;
        return ! has_value() && std::is_base_of< Ex, ContainedEx>::value;
    }

#if nsel_P2505R >= 4
    template< typename G = E
        nsel_REQUIRES_T(
            std::is_copy_constructible< E >::value
            && std::is_convertible< G, E >::value
        )
    >
    nsel_constexpr error_type error_or( G && e ) const &
    {
        return has_value()
            ? static_cast< E >( std::forward< G >( e ) )
            : contained.error();
    }

    template< typename G = E
        nsel_REQUIRES_T(
            std::is_move_constructible< E >::value
            && std::is_convertible< G, E >::value
        )
    >
    nsel_constexpr14 error_type error_or( G && e ) &&
    {
        return has_value()
            ? static_cast< E >( std::forward< G >( e ) )
            : std::move( contained.error() );
    }
#endif // nsel_P2505R >= 4

#if nsel_P2505R >= 3
    // Monadic operations (P2505)
    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F >::error_type, error_type >::value
            && std::is_constructible< error_type, error_type & >::value
        )
    >
    nsel_constexpr14 detail::invoke_result_nocvref_t< F > and_then( F && f ) &
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F >( detail::invoke( std::forward< F >( f ) ) )
            : detail::invoke_result_nocvref_t< F >( unexpect, error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F >::error_type, error_type >::value
            && std::is_constructible< error_type, const error_type & >::value
        )
    >
    nsel_constexpr detail::invoke_result_nocvref_t< F > and_then( F && f ) const &
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F >( detail::invoke( std::forward< F >( f ) ) )
            : detail::invoke_result_nocvref_t< F >( unexpect, error() );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490
    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F >::error_type, error_type >::value
            && std::is_constructible< error_type, error_type && >::value
        )
    >
    nsel_constexpr14 detail::invoke_result_nocvref_t< F > and_then( F && f ) &&
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F >( detail::invoke( std::forward< F >( f ) ) )
            : detail::invoke_result_nocvref_t< F >( unexpect, std::move( error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F > >::value
            && std::is_same< typename detail::invoke_result_nocvref_t< F >::error_type, error_type >::value
            && std::is_constructible< error_type, const error_type && >::value
        )
    >
    nsel_constexpr detail::invoke_result_nocvref_t< F > and_then( F && f ) const &&
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F >( detail::invoke( std::forward< F >( f ) ) )
            : detail::invoke_result_nocvref_t< F >( unexpect, std::move( error() ) );
    }
#endif

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, error_type & > >::value
            && std::is_void< typename detail::invoke_result_nocvref_t< F, error_type & >::value_type >::value
        )
    >
    nsel_constexpr14 detail::invoke_result_nocvref_t< F, error_type & > or_else( F && f ) &
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, error_type & >()
            : detail::invoke_result_nocvref_t< F, error_type & >( detail::invoke( std::forward< F >( f ), error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, const error_type & > >::value
            && std::is_void< typename detail::invoke_result_nocvref_t< F, const error_type & >::value_type >::value
        )
    >
    nsel_constexpr detail::invoke_result_nocvref_t< F, const error_type & > or_else( F && f ) const &
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, const error_type & >()
            : detail::invoke_result_nocvref_t< F, const error_type & >( detail::invoke( std::forward< F >( f ), error() ) );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490
    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, error_type && > >::value
            && std::is_void< typename detail::invoke_result_nocvref_t< F, error_type && >::value_type >::value
        )
    >
    nsel_constexpr14 detail::invoke_result_nocvref_t< F, error_type && > or_else( F && f ) &&
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, error_type && >()
            : detail::invoke_result_nocvref_t< F, error_type && >( detail::invoke( std::forward< F >( f ), std::move( error() ) ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::is_expected< detail::invoke_result_nocvref_t< F, const error_type && > >::value
            && std::is_void< typename detail::invoke_result_nocvref_t< F, const error_type && >::value_type >::value
        )
    >
    nsel_constexpr detail::invoke_result_nocvref_t< F, const error_type && > or_else( F && f ) const &&
    {
        return has_value()
            ? detail::invoke_result_nocvref_t< F, const error_type && >()
            : detail::invoke_result_nocvref_t< F, const error_type && >( detail::invoke( std::forward< F >( f ), std::move( error() ) ) );
    }
#endif

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, error_type & >::value
            && !std::is_void< detail::transform_invoke_result_t< F > >::value
        )
    >
    nsel_constexpr14 expected< detail::transform_invoke_result_t< F >, error_type > transform( F && f ) &
    {
        return has_value()
            ? expected< detail::transform_invoke_result_t< F >, error_type >( detail::invoke( std::forward< F >( f ) ) )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, error_type & >::value
            && std::is_void< detail::transform_invoke_result_t< F > >::value
        )
    >
    nsel_constexpr14 expected< void, error_type > transform( F && f ) &
    {
        return has_value()
            ? ( detail::invoke( std::forward< F >( f ) ), expected< void, error_type >() )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, const error_type & >::value
            && !std::is_void< detail::transform_invoke_result_t< F > >::value
        )
    >
    nsel_constexpr expected< detail::transform_invoke_result_t< F >, error_type > transform( F && f ) const &
    {
        return has_value()
            ? expected< detail::transform_invoke_result_t< F >, error_type >( detail::invoke( std::forward< F >( f ) ) )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, const error_type & >::value
            && std::is_void< detail::transform_invoke_result_t< F > >::value
        )
    >
    nsel_constexpr expected< void, error_type > transform( F && f ) const &
    {
        return has_value()
            ? ( detail::invoke( std::forward< F >( f ) ), expected< void, error_type >() )
            : make_unexpected( error() );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490
    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, error_type && >::value
            && !std::is_void< detail::transform_invoke_result_t< F > >::value
        )
    >
    nsel_constexpr14 expected< detail::transform_invoke_result_t< F >, error_type > transform( F && f ) &&
    {
        return has_value()
            ? expected< detail::transform_invoke_result_t< F >, error_type >( detail::invoke( std::forward< F >( f ) ) )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, error_type && >::value
            && std::is_void< detail::transform_invoke_result_t< F > >::value
        )
    >
    nsel_constexpr14 expected< void, error_type > transform( F && f ) &&
    {
        return has_value()
            ? ( detail::invoke( std::forward< F >( f ) ), expected< void, error_type >() )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, const error_type && >::value
            && !std::is_void< detail::transform_invoke_result_t< F > >::value
        )
    >
    nsel_constexpr expected< detail::transform_invoke_result_t< F >, error_type > transform( F && f ) const &&
    {
        return has_value()
            ? expected< detail::transform_invoke_result_t< F >, error_type >( detail::invoke( std::forward< F >( f ) ) )
            : make_unexpected( error() );
    }

    template<typename F
        nsel_REQUIRES_T(
            std::is_constructible< error_type, const error_type && >::value
            && std::is_void< detail::transform_invoke_result_t< F > >::value
        )
    >
    nsel_constexpr expected< void, error_type > transform( F && f ) const &&
    {
        return has_value()
            ? ( detail::invoke( std::forward< F >( f ) ), expected< void, error_type >() )
            : make_unexpected( error() );
    }
#endif

    template<typename F
        nsel_REQUIRES_T(
            detail::valid_unexpected_type< detail::transform_invoke_result_t< F, error_type & > >::value
        )
    >
    nsel_constexpr14 expected< void, detail::transform_invoke_result_t< F, error_type & > > transform_error( F && f ) &
    {
        return has_value()
            ? expected< void, detail::transform_invoke_result_t< F, error_type & > >()
            : make_unexpected( detail::invoke( std::forward< F >( f ), error() ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::valid_unexpected_type< detail::transform_invoke_result_t< F, const error_type & > >::value
        )
    >
    nsel_constexpr expected< void, detail::transform_invoke_result_t< F, const error_type & > > transform_error( F && f ) const &
    {
        return has_value()
            ? expected< void, detail::transform_invoke_result_t< F, const error_type & > >()
            : make_unexpected( detail::invoke( std::forward< F >( f ), error() ) );
    }

#if !nsel_COMPILER_GNUC_VERSION || nsel_COMPILER_GNUC_VERSION >= 490
    template<typename F
        nsel_REQUIRES_T(
            detail::valid_unexpected_type< detail::transform_invoke_result_t< F, error_type && > >::value
        )
    >
    nsel_constexpr14 expected< void, detail::transform_invoke_result_t< F, error_type && > > transform_error( F && f ) &&
    {
        return has_value()
            ? expected< void, detail::transform_invoke_result_t< F, error_type && > >()
            : make_unexpected( detail::invoke( std::forward< F >( f ), std::move( error() ) ) );
    }

    template<typename F
        nsel_REQUIRES_T(
            detail::valid_unexpected_type< detail::transform_invoke_result_t< F, const error_type && > >::value
        )
    >
    nsel_constexpr expected< void, detail::transform_invoke_result_t< F, const error_type && > > transform_error( F && f ) const &&
    {
        return has_value()
            ? expected< void, detail::transform_invoke_result_t< F, const error_type && > >()
            : make_unexpected( detail::invoke( std::forward< F >( f ), std::move( error() ) ) );
    }
#endif
#endif // nsel_P2505R >= 3

//  template constexpr 'see below' unwrap() const&;
//
//  template 'see below' unwrap() &&;

    // factories

//  template< typename Ex, typename F>
//  expected<void,E> catch_exception(F&& f);
//
//  template< typename F>
//  expected<decltype(func()), E> map(F&& func) ;
//
//  template< typename F>
//  'see below' bind(F&& func) ;
//
//  template< typename F>
//  expected<void,E> catch_error(F&& f);
//
//  template< typename F>
//  'see below' then(F&& func);

private:
    detail::storage_t
    <
        void
        , E
        , std::is_copy_constructible<E>::value
        , std::is_move_constructible<E>::value
    >
    contained;
};

// x.x.4.6 expected<>: comparison operators

template< typename T1, typename E1, typename T2, typename E2
    nsel_REQUIRES_T(
        !std::is_void<T1>::value && !std::is_void<T2>::value
    )
>
constexpr bool operator==( expected<T1,E1> const & x, expected<T2,E2> const & y )
{
    return bool(x) != bool(y) ? false : bool(x) ? *x == *y : x.error() == y.error();
}

template< typename T1, typename E1, typename T2, typename E2
    nsel_REQUIRES_T(
        std::is_void<T1>::value && std::is_void<T2>::value
    )
>
constexpr bool operator==( expected<T1,E1> const & x, expected<T2,E2> const & y )
{
    return bool(x) != bool(y) ? false : bool(x) || static_cast<bool>( x.error() == y.error() );
}

template< typename T1, typename E1, typename T2, typename E2 >
constexpr bool operator!=( expected<T1,E1> const & x, expected<T2,E2> const & y )
{
    return !(x == y);
}

#if nsel_P0323R <= 2

template< typename T, typename E >
constexpr bool operator<( expected<T,E> const & x, expected<T,E> const & y )
{
    return (!y) ? false : (!x) ? true : *x < *y;
}

template< typename T, typename E >
constexpr bool operator>( expected<T,E> const & x, expected<T,E> const & y )
{
    return (y < x);
}

template< typename T, typename E >
constexpr bool operator<=( expected<T,E> const & x, expected<T,E> const & y )
{
    return !(y < x);
}

template< typename T, typename E >
constexpr bool operator>=( expected<T,E> const & x, expected<T,E> const & y )
{
    return !(x < y);
}

#endif

// x.x.4.7 expected: comparison with T

template< typename T1, typename E1, typename T2
    nsel_REQUIRES_T(
        !std::is_void<T1>::value
    )
>
constexpr bool operator==( expected<T1,E1> const & x, T2 const & v )
{
    return bool(x) ? *x == v : false;
}

template< typename T1, typename E1, typename T2
    nsel_REQUIRES_T(
        !std::is_void<T1>::value
    )
>
constexpr bool operator==(T2 const & v, expected<T1,E1> const & x )
{
    return bool(x) ? v == *x : false;
}

template< typename T1, typename E1, typename T2 >
constexpr bool operator!=( expected<T1,E1> const & x, T2 const & v )
{
    return bool(x) ? *x != v : true;
}

template< typename T1, typename E1, typename T2 >
constexpr bool operator!=( T2 const & v, expected<T1,E1> const & x )
{
    return bool(x) ? v != *x : true;
}

#if nsel_P0323R <= 2

template< typename T, typename E >
constexpr bool operator<( expected<T,E> const & x, T const & v )
{
    return bool(x) ? *x < v : true;
}

template< typename T, typename E >
constexpr bool operator<( T const & v, expected<T,E> const & x )
{
    return bool(x) ? v < *x : false;
}

template< typename T, typename E >
constexpr bool operator>( T const & v, expected<T,E> const & x )
{
    return bool(x) ? *x < v : false;
}

template< typename T, typename E >
constexpr bool operator>( expected<T,E> const & x, T const & v )
{
    return bool(x) ? v < *x : false;
}

template< typename T, typename E >
constexpr bool operator<=( T const & v, expected<T,E> const & x )
{
    return bool(x) ? ! ( *x < v ) : false;
}

template< typename T, typename E >
constexpr bool operator<=( expected<T,E> const & x, T const & v )
{
    return bool(x) ? ! ( v < *x ) : true;
}

template< typename T, typename E >
constexpr bool operator>=( expected<T,E> const & x, T const & v )
{
    return bool(x) ? ! ( *x < v ) : false;
}

template< typename T, typename E >
constexpr bool operator>=( T const & v, expected<T,E> const & x )
{
    return bool(x) ? ! ( v < *x ) : true;
}

#endif // nsel_P0323R

// x.x.4.8 expected: comparison with unexpected_type

template< typename T1, typename E1 , typename E2 >
constexpr bool operator==( expected<T1,E1> const & x, unexpected_type<E2> const & u )
{
    return (!x) ? x.get_unexpected() == u : false;
}

template< typename T1, typename E1 , typename E2 >
constexpr bool operator==( unexpected_type<E2> const & u, expected<T1,E1> const & x )
{
    return ( x == u );
}

template< typename T1, typename E1 , typename E2 >
constexpr bool operator!=( expected<T1,E1> const & x, unexpected_type<E2> const & u )
{
    return ! ( x == u );
}

template< typename T1, typename E1 , typename E2 >
constexpr bool operator!=( unexpected_type<E2> const & u, expected<T1,E1> const & x )
{
    return ! ( x == u );
}

#if nsel_P0323R <= 2

template< typename T, typename E >
constexpr bool operator<( expected<T,E> const & x, unexpected_type<E> const & u )
{
    return (!x) ? ( x.get_unexpected() < u ) : false;
}

template< typename T, typename E >
constexpr bool operator<( unexpected_type<E> const & u, expected<T,E> const & x )
{
  return (!x) ? ( u < x.get_unexpected() ) : true ;
}

template< typename T, typename E >
constexpr bool operator>( expected<T,E> const & x, unexpected_type<E> const & u )
{
    return ( u < x );
}

template< typename T, typename E >
constexpr bool operator>( unexpected_type<E> const & u, expected<T,E> const & x )
{
    return ( x < u );
}

template< typename T, typename E >
constexpr bool operator<=( expected<T,E> const & x, unexpected_type<E> const & u )
{
    return ! ( u < x );
}

template< typename T, typename E >
constexpr bool operator<=( unexpected_type<E> const & u, expected<T,E> const & x)
{
    return ! ( x < u );
}

template< typename T, typename E >
constexpr bool operator>=( expected<T,E> const & x, unexpected_type<E> const & u  )
{
    return ! ( u > x );
}

template< typename T, typename E >
constexpr bool operator>=( unexpected_type<E> const & u, expected<T,E> const & x )
{
    return ! ( x > u );
}

#endif // nsel_P0323R

/// x.x.x Specialized algorithms

template< typename T, typename E
    nsel_REQUIRES_T(
        ( std::is_void<T>::value || std::is_move_constructible<T>::value )
        && std::is_move_constructible<E>::value
        && std17::is_swappable<T>::value
        && std17::is_swappable<E>::value )
>
void swap( expected<T,E> & x, expected<T,E> & y ) noexcept ( noexcept ( x.swap(y) ) )
{
    x.swap( y );
}

#if nsel_P0323R <= 3

template< typename T >
constexpr auto make_expected( T && v ) -> expected< typename std::decay<T>::type >
{
    return expected< typename std::decay<T>::type >( std::forward<T>( v ) );
}

// expected<void> specialization:

auto inline make_expected() -> expected<void>
{
    return expected<void>( in_place );
}

template< typename T >
constexpr auto make_expected_from_current_exception() -> expected<T>
{
    return expected<T>( make_unexpected_from_current_exception() );
}

template< typename T >
auto make_expected_from_exception( std::exception_ptr v ) -> expected<T>
{
    return expected<T>( unexpected_type<std::exception_ptr>( std::forward<std::exception_ptr>( v ) ) );
}

template< typename T, typename E >
constexpr auto make_expected_from_error( E e ) -> expected<T, typename std::decay<E>::type>
{
    return expected<T, typename std::decay<E>::type>( make_unexpected( e ) );
}

template< typename F
    nsel_REQUIRES_T( ! std::is_same<typename std::result_of<F()>::type, void>::value )
>
/*nsel_constexpr14*/
auto make_expected_from_call( F f ) -> expected< typename std::result_of<F()>::type >
{
    try
    {
        return make_expected( f() );
    }
    catch (...)
    {
        return make_unexpected_from_current_exception();
    }
}

template< typename F
    nsel_REQUIRES_T( std::is_same<typename std::result_of<F()>::type, void>::value )
>
/*nsel_constexpr14*/
auto make_expected_from_call( F f ) -> expected<void>
{
    try
    {
        f();
        return make_expected();
    }
    catch (...)
    {
        return make_unexpected_from_current_exception();
    }
}

#endif // nsel_P0323R

} // namespace expected_lite

using namespace expected_lite;

// using expected_lite::expected;
// using ...

} // namespace nonstd

namespace std {

// expected: hash support

template< typename T, typename E >
struct hash< nonstd::expected<T,E> >
{
    using result_type = std::size_t;
    using argument_type = nonstd::expected<T,E>;

    constexpr result_type operator()(argument_type const & arg) const
    {
        return arg ? std::hash<T>{}(*arg) : result_type{};
    }
};

// TBD - ?? remove? see spec.
template< typename T, typename E >
struct hash< nonstd::expected<T&,E> >
{
    using result_type = std::size_t;
    using argument_type = nonstd::expected<T&,E>;

    constexpr result_type operator()(argument_type const & arg) const
    {
        return arg ? std::hash<T>{}(*arg) : result_type{};
    }
};

// TBD - implement
// bool(e), hash<expected<void,E>>()(e) shall evaluate to the hashing true;
// otherwise it evaluates to an unspecified value if E is exception_ptr or
// a combination of hashing false and hash<E>()(e.error()).

template< typename E >
struct hash< nonstd::expected<void,E> >
{
};

} // namespace std

namespace nonstd {

// void unexpected() is deprecated && removed in C++17

#if nsel_CPP17_OR_GREATER || nsel_COMPILER_MSVC_VERSION > 141
template< typename E >
using unexpected = unexpected_type<E>;
#endif

} // namespace nonstd

#undef nsel_REQUIRES
#undef nsel_REQUIRES_0
#undef nsel_REQUIRES_T

nsel_RESTORE_WARNINGS()

#endif // nsel_USES_STD_EXPECTED

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // NONSTD_EXPECTED_LITE_HPP
