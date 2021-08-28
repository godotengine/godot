// Copyright 2015-2021 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_HPP
#define VULKAN_HPP

#if defined( _MSVC_LANG )
#  define VULKAN_HPP_CPLUSPLUS _MSVC_LANG
#else
#  define VULKAN_HPP_CPLUSPLUS __cplusplus
#endif

#if 201703L < VULKAN_HPP_CPLUSPLUS
#  define VULKAN_HPP_CPP_VERSION 20
#elif 201402L < VULKAN_HPP_CPLUSPLUS
#  define VULKAN_HPP_CPP_VERSION 17
#elif 201103L < VULKAN_HPP_CPLUSPLUS
#  define VULKAN_HPP_CPP_VERSION 14
#elif 199711L < VULKAN_HPP_CPLUSPLUS
#  define VULKAN_HPP_CPP_VERSION 11
#else
#  error "vulkan.hpp needs at least c++ standard version 11"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <vulkan/vulkan.h>

#if 17 <= VULKAN_HPP_CPP_VERSION
#  include <string_view>
#endif

#if defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
#  if !defined( VULKAN_HPP_NO_SMART_HANDLE )
#    define VULKAN_HPP_NO_SMART_HANDLE
#  endif
#else
#  include <memory>
#  include <vector>
#endif

#if !defined( VULKAN_HPP_ASSERT )
#  include <cassert>
#  define VULKAN_HPP_ASSERT assert
#endif

#if !defined( VULKAN_HPP_ASSERT_ON_RESULT )
#  define VULKAN_HPP_ASSERT_ON_RESULT VULKAN_HPP_ASSERT
#endif

#if !defined( VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL )
#  define VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL 1
#endif

#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL == 1
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNXNTO__ ) || defined( __Fuchsia__ )
#    include <dlfcn.h>
#  elif defined( _WIN32 )
typedef struct HINSTANCE__ * HINSTANCE;
#    if defined( _WIN64 )
typedef int64_t( __stdcall * FARPROC )();
#    else
typedef int( __stdcall * FARPROC )();
#    endif
extern "C" __declspec( dllimport ) HINSTANCE __stdcall LoadLibraryA( char const * lpLibFileName );
extern "C" __declspec( dllimport ) int __stdcall FreeLibrary( HINSTANCE hLibModule );
extern "C" __declspec( dllimport ) FARPROC __stdcall GetProcAddress( HINSTANCE hModule, const char * lpProcName );
#  endif
#endif

#if !defined( __has_include )
#  define __has_include( x ) false
#endif

#if ( 201711 <= __cpp_impl_three_way_comparison ) && __has_include( <compare> ) && !defined( VULKAN_HPP_NO_SPACESHIP_OPERATOR )
#  define VULKAN_HPP_HAS_SPACESHIP_OPERATOR
#endif
#if defined( VULKAN_HPP_HAS_SPACESHIP_OPERATOR )
#  include <compare>
#endif

static_assert( VK_HEADER_VERSION == 182, "Wrong VK_HEADER_VERSION!" );

// 32-bit vulkan is not typesafe for handles, so don't allow copy constructors on this platform by default.
// To enable this feature on 32-bit platforms please define VULKAN_HPP_TYPESAFE_CONVERSION
#if ( VK_USE_64_BIT_PTR_DEFINES == 1 )
#  if !defined( VULKAN_HPP_TYPESAFE_CONVERSION )
#    define VULKAN_HPP_TYPESAFE_CONVERSION
#  endif
#endif

// <tuple> includes <sys/sysmacros.h> through some other header
// this results in major(x) being resolved to gnu_dev_major(x)
// which is an expression in a constructor initializer list.
#if defined( major )
#  undef major
#endif
#if defined( minor )
#  undef minor
#endif

// Windows defines MemoryBarrier which is deprecated and collides
// with the VULKAN_HPP_NAMESPACE::MemoryBarrier struct.
#if defined( MemoryBarrier )
#  undef MemoryBarrier
#endif

#if !defined( VULKAN_HPP_HAS_UNRESTRICTED_UNIONS )
#  if defined( __clang__ )
#    if __has_feature( cxx_unrestricted_unions )
#      define VULKAN_HPP_HAS_UNRESTRICTED_UNIONS
#    endif
#  elif defined( __GNUC__ )
#    define GCC_VERSION ( __GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__ )
#    if 40600 <= GCC_VERSION
#      define VULKAN_HPP_HAS_UNRESTRICTED_UNIONS
#    endif
#  elif defined( _MSC_VER )
#    if 1900 <= _MSC_VER
#      define VULKAN_HPP_HAS_UNRESTRICTED_UNIONS
#    endif
#  endif
#endif

#if !defined( VULKAN_HPP_INLINE )
#  if defined( __clang__ )
#    if __has_attribute( always_inline )
#      define VULKAN_HPP_INLINE __attribute__( ( always_inline ) ) __inline__
#    else
#      define VULKAN_HPP_INLINE inline
#    endif
#  elif defined( __GNUC__ )
#    define VULKAN_HPP_INLINE __attribute__( ( always_inline ) ) __inline__
#  elif defined( _MSC_VER )
#    define VULKAN_HPP_INLINE inline
#  else
#    define VULKAN_HPP_INLINE inline
#  endif
#endif

#if defined( VULKAN_HPP_TYPESAFE_CONVERSION )
#  define VULKAN_HPP_TYPESAFE_EXPLICIT
#else
#  define VULKAN_HPP_TYPESAFE_EXPLICIT explicit
#endif

#if defined( __cpp_constexpr )
#  define VULKAN_HPP_CONSTEXPR constexpr
#  if __cpp_constexpr >= 201304
#    define VULKAN_HPP_CONSTEXPR_14 constexpr
#  else
#    define VULKAN_HPP_CONSTEXPR_14
#  endif
#  define VULKAN_HPP_CONST_OR_CONSTEXPR constexpr
#else
#  define VULKAN_HPP_CONSTEXPR
#  define VULKAN_HPP_CONSTEXPR_14
#  define VULKAN_HPP_CONST_OR_CONSTEXPR const
#endif

#if !defined( VULKAN_HPP_NOEXCEPT )
#  if defined( _MSC_VER ) && ( _MSC_VER <= 1800 )
#    define VULKAN_HPP_NOEXCEPT
#  else
#    define VULKAN_HPP_NOEXCEPT noexcept
#    define VULKAN_HPP_HAS_NOEXCEPT 1
#    if defined( VULKAN_HPP_NO_EXCEPTIONS )
#      define VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS noexcept
#    else
#      define VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
#    endif
#  endif
#endif

#if 14 <= VULKAN_HPP_CPP_VERSION
#  define VULKAN_HPP_DEPRECATED( msg ) [[deprecated( msg )]]
#else
#  define VULKAN_HPP_DEPRECATED( msg )
#endif

#if ( 17 <= VULKAN_HPP_CPP_VERSION ) && !defined( VULKAN_HPP_NO_NODISCARD_WARNINGS )
#  define VULKAN_HPP_NODISCARD [[nodiscard]]
#  if defined( VULKAN_HPP_NO_EXCEPTIONS )
#    define VULKAN_HPP_NODISCARD_WHEN_NO_EXCEPTIONS [[nodiscard]]
#  else
#    define VULKAN_HPP_NODISCARD_WHEN_NO_EXCEPTIONS
#  endif
#else
#  define VULKAN_HPP_NODISCARD
#  define VULKAN_HPP_NODISCARD_WHEN_NO_EXCEPTIONS
#endif

#if !defined( VULKAN_HPP_NAMESPACE )
#  define VULKAN_HPP_NAMESPACE vk
#endif

#define VULKAN_HPP_STRINGIFY2( text ) #text
#define VULKAN_HPP_STRINGIFY( text ) VULKAN_HPP_STRINGIFY2( text )
#define VULKAN_HPP_NAMESPACE_STRING VULKAN_HPP_STRINGIFY( VULKAN_HPP_NAMESPACE )

namespace VULKAN_HPP_NAMESPACE
{
#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
  template <typename T>
  class ArrayProxy
  {
  public:
    VULKAN_HPP_CONSTEXPR ArrayProxy() VULKAN_HPP_NOEXCEPT
      : m_count( 0 )
      , m_ptr( nullptr )
    {}

    VULKAN_HPP_CONSTEXPR ArrayProxy( std::nullptr_t ) VULKAN_HPP_NOEXCEPT
      : m_count( 0 )
      , m_ptr( nullptr )
    {}

    ArrayProxy( T & value ) VULKAN_HPP_NOEXCEPT
      : m_count( 1 )
      , m_ptr( &value )
    {}

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( typename std::remove_const<T>::type & value ) VULKAN_HPP_NOEXCEPT
      : m_count( 1 )
      , m_ptr( &value )
    {}

    ArrayProxy( uint32_t count, T * ptr ) VULKAN_HPP_NOEXCEPT
      : m_count( count )
      , m_ptr( ptr )
    {}

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( uint32_t count, typename std::remove_const<T>::type * ptr ) VULKAN_HPP_NOEXCEPT
      : m_count( count )
      , m_ptr( ptr )
    {}

#  if __GNUC__ >= 9
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Winit-list-lifetime"
#  endif

    ArrayProxy( std::initializer_list<T> const & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {}

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( std::initializer_list<typename std::remove_const<T>::type> const & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {}

    ArrayProxy( std::initializer_list<T> & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {}

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( std::initializer_list<typename std::remove_const<T>::type> & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {}

#  if __GNUC__ >= 9
#    pragma GCC diagnostic pop
#  endif

    template <size_t N>
    ArrayProxy( std::array<T, N> const & data ) VULKAN_HPP_NOEXCEPT
      : m_count( N )
      , m_ptr( data.data() )
    {}

    template <size_t N, typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( std::array<typename std::remove_const<T>::type, N> const & data ) VULKAN_HPP_NOEXCEPT
      : m_count( N )
      , m_ptr( data.data() )
    {}

    template <size_t N>
    ArrayProxy( std::array<T, N> & data ) VULKAN_HPP_NOEXCEPT
      : m_count( N )
      , m_ptr( data.data() )
    {}

    template <size_t N, typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( std::array<typename std::remove_const<T>::type, N> & data ) VULKAN_HPP_NOEXCEPT
      : m_count( N )
      , m_ptr( data.data() )
    {}

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
    ArrayProxy( std::vector<T, Allocator> const & data ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( data.size() ) )
      , m_ptr( data.data() )
    {}

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>,
              typename B      = T,
              typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( std::vector<typename std::remove_const<T>::type, Allocator> const & data ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( data.size() ) )
      , m_ptr( data.data() )
    {}

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
    ArrayProxy( std::vector<T, Allocator> & data ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( data.size() ) )
      , m_ptr( data.data() )
    {}

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>,
              typename B      = T,
              typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( std::vector<typename std::remove_const<T>::type, Allocator> & data ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( data.size() ) )
      , m_ptr( data.data() )
    {}

    const T * begin() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr;
    }

    const T * end() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr + m_count;
    }

    const T & front() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_count && m_ptr );
      return *m_ptr;
    }

    const T & back() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_count && m_ptr );
      return *( m_ptr + m_count - 1 );
    }

    bool empty() const VULKAN_HPP_NOEXCEPT
    {
      return ( m_count == 0 );
    }

    uint32_t size() const VULKAN_HPP_NOEXCEPT
    {
      return m_count;
    }

    T * data() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr;
    }

  private:
    uint32_t m_count;
    T *      m_ptr;
  };

  template <typename T>
  class ArrayProxyNoTemporaries
  {
  public:
    VULKAN_HPP_CONSTEXPR ArrayProxyNoTemporaries() VULKAN_HPP_NOEXCEPT
      : m_count( 0 )
      , m_ptr( nullptr )
    {}

    VULKAN_HPP_CONSTEXPR ArrayProxyNoTemporaries( std::nullptr_t ) VULKAN_HPP_NOEXCEPT
      : m_count( 0 )
      , m_ptr( nullptr )
    {}

    ArrayProxyNoTemporaries( T & value ) VULKAN_HPP_NOEXCEPT
      : m_count( 1 )
      , m_ptr( &value )
    {}

    ArrayProxyNoTemporaries( T && value ) = delete;

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( typename std::remove_const<T>::type & value ) VULKAN_HPP_NOEXCEPT
      : m_count( 1 )
      , m_ptr( &value )
    {}

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( typename std::remove_const<T>::type && value ) = delete;

    ArrayProxyNoTemporaries( uint32_t count, T * ptr ) VULKAN_HPP_NOEXCEPT
      : m_count( count )
      , m_ptr( ptr )
    {}

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( uint32_t count, typename std::remove_const<T>::type * ptr ) VULKAN_HPP_NOEXCEPT
      : m_count( count )
      , m_ptr( ptr )
    {}

    ArrayProxyNoTemporaries( std::initializer_list<T> const & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {}

    ArrayProxyNoTemporaries( std::initializer_list<T> const && list ) = delete;

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::initializer_list<typename std::remove_const<T>::type> const & list )
      VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {}

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::initializer_list<typename std::remove_const<T>::type> const && list ) = delete;

    ArrayProxyNoTemporaries( std::initializer_list<T> & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {}

    ArrayProxyNoTemporaries( std::initializer_list<T> && list ) = delete;

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::initializer_list<typename std::remove_const<T>::type> & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {}

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::initializer_list<typename std::remove_const<T>::type> && list ) = delete;

    template <size_t N>
    ArrayProxyNoTemporaries( std::array<T, N> const & data ) VULKAN_HPP_NOEXCEPT
      : m_count( N )
      , m_ptr( data.data() )
    {}

    template <size_t N>
    ArrayProxyNoTemporaries( std::array<T, N> const && data ) = delete;

    template <size_t N, typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::array<typename std::remove_const<T>::type, N> const & data ) VULKAN_HPP_NOEXCEPT
      : m_count( N )
      , m_ptr( data.data() )
    {}

    template <size_t N, typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::array<typename std::remove_const<T>::type, N> const && data ) = delete;

    template <size_t N>
    ArrayProxyNoTemporaries( std::array<T, N> & data ) VULKAN_HPP_NOEXCEPT
      : m_count( N )
      , m_ptr( data.data() )
    {}

    template <size_t N>
    ArrayProxyNoTemporaries( std::array<T, N> && data ) = delete;

    template <size_t N, typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::array<typename std::remove_const<T>::type, N> & data ) VULKAN_HPP_NOEXCEPT
      : m_count( N )
      , m_ptr( data.data() )
    {}

    template <size_t N, typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::array<typename std::remove_const<T>::type, N> && data ) = delete;

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
    ArrayProxyNoTemporaries( std::vector<T, Allocator> const & data ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( data.size() ) )
      , m_ptr( data.data() )
    {}

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
    ArrayProxyNoTemporaries( std::vector<T, Allocator> const && data ) = delete;

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>,
              typename B      = T,
              typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::vector<typename std::remove_const<T>::type, Allocator> const & data )
      VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( data.size() ) )
      , m_ptr( data.data() )
    {}

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>,
              typename B      = T,
              typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::vector<typename std::remove_const<T>::type, Allocator> const && data ) = delete;

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
    ArrayProxyNoTemporaries( std::vector<T, Allocator> & data ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( data.size() ) )
      , m_ptr( data.data() )
    {}

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>>
    ArrayProxyNoTemporaries( std::vector<T, Allocator> && data ) = delete;

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>,
              typename B      = T,
              typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::vector<typename std::remove_const<T>::type, Allocator> & data ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( data.size() ) )
      , m_ptr( data.data() )
    {}

    template <class Allocator = std::allocator<typename std::remove_const<T>::type>,
              typename B      = T,
              typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::vector<typename std::remove_const<T>::type, Allocator> && data ) = delete;

    const T * begin() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr;
    }

    const T * end() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr + m_count;
    }

    const T & front() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_count && m_ptr );
      return *m_ptr;
    }

    const T & back() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_count && m_ptr );
      return *( m_ptr + m_count - 1 );
    }

    bool empty() const VULKAN_HPP_NOEXCEPT
    {
      return ( m_count == 0 );
    }

    uint32_t size() const VULKAN_HPP_NOEXCEPT
    {
      return m_count;
    }

    T * data() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr;
    }

  private:
    uint32_t m_count;
    T *      m_ptr;
  };
#endif

  template <typename T, size_t N>
  class ArrayWrapper1D : public std::array<T, N>
  {
  public:
    VULKAN_HPP_CONSTEXPR ArrayWrapper1D() VULKAN_HPP_NOEXCEPT : std::array<T, N>() {}

    VULKAN_HPP_CONSTEXPR ArrayWrapper1D( std::array<T, N> const & data ) VULKAN_HPP_NOEXCEPT : std::array<T, N>( data )
    {}

#if defined( _WIN32 ) && !defined( _WIN64 )
    VULKAN_HPP_CONSTEXPR T const & operator[]( int index ) const VULKAN_HPP_NOEXCEPT
    {
      return std::array<T, N>::operator[]( index );
    }

    T & operator[]( int index ) VULKAN_HPP_NOEXCEPT
    {
      return std::array<T, N>::operator[]( index );
    }
#endif

    operator T const *() const VULKAN_HPP_NOEXCEPT
    {
      return this->data();
    }

    operator T *() VULKAN_HPP_NOEXCEPT
    {
      return this->data();
    }

    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    operator std::string() const
    {
      return std::string( this->data() );
    }

#if 17 <= VULKAN_HPP_CPP_VERSION
    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    operator std::string_view() const
    {
      return std::string_view( this->data() );
    }
#endif

    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    bool operator<( ArrayWrapper1D<char, N> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return *static_cast<std::array<char, N> const *>( this ) < *static_cast<std::array<char, N> const *>( &rhs );
    }

    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    bool operator<=( ArrayWrapper1D<char, N> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return *static_cast<std::array<char, N> const *>( this ) <= *static_cast<std::array<char, N> const *>( &rhs );
    }

    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    bool operator>( ArrayWrapper1D<char, N> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return *static_cast<std::array<char, N> const *>( this ) > *static_cast<std::array<char, N> const *>( &rhs );
    }

    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    bool operator>=( ArrayWrapper1D<char, N> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return *static_cast<std::array<char, N> const *>( this ) >= *static_cast<std::array<char, N> const *>( &rhs );
    }

    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    bool operator==( ArrayWrapper1D<char, N> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return *static_cast<std::array<char, N> const *>( this ) == *static_cast<std::array<char, N> const *>( &rhs );
    }

    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    bool operator!=( ArrayWrapper1D<char, N> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return *static_cast<std::array<char, N> const *>( this ) != *static_cast<std::array<char, N> const *>( &rhs );
    }
  };

  // specialization of relational operators between std::string and arrays of chars
  template <size_t N>
  bool operator<( std::string const & lhs, ArrayWrapper1D<char, N> const & rhs ) VULKAN_HPP_NOEXCEPT
  {
    return lhs < rhs.data();
  }

  template <size_t N>
  bool operator<=( std::string const & lhs, ArrayWrapper1D<char, N> const & rhs ) VULKAN_HPP_NOEXCEPT
  {
    return lhs <= rhs.data();
  }

  template <size_t N>
  bool operator>( std::string const & lhs, ArrayWrapper1D<char, N> const & rhs ) VULKAN_HPP_NOEXCEPT
  {
    return lhs > rhs.data();
  }

  template <size_t N>
  bool operator>=( std::string const & lhs, ArrayWrapper1D<char, N> const & rhs ) VULKAN_HPP_NOEXCEPT
  {
    return lhs >= rhs.data();
  }

  template <size_t N>
  bool operator==( std::string const & lhs, ArrayWrapper1D<char, N> const & rhs ) VULKAN_HPP_NOEXCEPT
  {
    return lhs == rhs.data();
  }

  template <size_t N>
  bool operator!=( std::string const & lhs, ArrayWrapper1D<char, N> const & rhs ) VULKAN_HPP_NOEXCEPT
  {
    return lhs != rhs.data();
  }

  template <typename T, size_t N, size_t M>
  class ArrayWrapper2D : public std::array<ArrayWrapper1D<T, M>, N>
  {
  public:
    VULKAN_HPP_CONSTEXPR ArrayWrapper2D() VULKAN_HPP_NOEXCEPT : std::array<ArrayWrapper1D<T, M>, N>() {}

    VULKAN_HPP_CONSTEXPR ArrayWrapper2D( std::array<std::array<T, M>, N> const & data ) VULKAN_HPP_NOEXCEPT
      : std::array<ArrayWrapper1D<T, M>, N>( *reinterpret_cast<std::array<ArrayWrapper1D<T, M>, N> const *>( &data ) )
    {}
  };

  template <typename FlagBitsType>
  struct FlagTraits
  {
    enum
    {
      allFlags = 0
    };
  };

  template <typename BitType>
  class Flags
  {
  public:
    using MaskType = typename std::underlying_type<BitType>::type;

    // constructors
    VULKAN_HPP_CONSTEXPR Flags() VULKAN_HPP_NOEXCEPT : m_mask( 0 ) {}

    VULKAN_HPP_CONSTEXPR Flags( BitType bit ) VULKAN_HPP_NOEXCEPT : m_mask( static_cast<MaskType>( bit ) ) {}

    VULKAN_HPP_CONSTEXPR Flags( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT = default;

    VULKAN_HPP_CONSTEXPR explicit Flags( MaskType flags ) VULKAN_HPP_NOEXCEPT : m_mask( flags ) {}

    // relational operators
#if defined( VULKAN_HPP_HAS_SPACESHIP_OPERATOR )
    auto operator<=>( Flags<BitType> const & ) const = default;
#else
    VULKAN_HPP_CONSTEXPR bool operator<( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask < rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator<=( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask <= rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator>( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask > rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator>=( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask >= rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator==( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask == rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator!=( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask != rhs.m_mask;
    }
#endif

    // logical operator
    VULKAN_HPP_CONSTEXPR bool operator!() const VULKAN_HPP_NOEXCEPT
    {
      return !m_mask;
    }

    // bitwise operators
    VULKAN_HPP_CONSTEXPR Flags<BitType> operator&( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return Flags<BitType>( m_mask & rhs.m_mask );
    }

    VULKAN_HPP_CONSTEXPR Flags<BitType> operator|( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return Flags<BitType>( m_mask | rhs.m_mask );
    }

    VULKAN_HPP_CONSTEXPR Flags<BitType> operator^( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return Flags<BitType>( m_mask ^ rhs.m_mask );
    }

    VULKAN_HPP_CONSTEXPR Flags<BitType> operator~() const VULKAN_HPP_NOEXCEPT
    {
      return Flags<BitType>( m_mask ^ FlagTraits<BitType>::allFlags );
    }

    // assignment operators
    VULKAN_HPP_CONSTEXPR_14 Flags<BitType> & operator=( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT = default;

    VULKAN_HPP_CONSTEXPR_14 Flags<BitType> & operator|=( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT
    {
      m_mask |= rhs.m_mask;
      return *this;
    }

    VULKAN_HPP_CONSTEXPR_14 Flags<BitType> & operator&=( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT
    {
      m_mask &= rhs.m_mask;
      return *this;
    }

    VULKAN_HPP_CONSTEXPR_14 Flags<BitType> & operator^=( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT
    {
      m_mask ^= rhs.m_mask;
      return *this;
    }

    // cast operators
    explicit VULKAN_HPP_CONSTEXPR operator bool() const VULKAN_HPP_NOEXCEPT
    {
      return !!m_mask;
    }

    explicit VULKAN_HPP_CONSTEXPR operator MaskType() const VULKAN_HPP_NOEXCEPT
    {
      return m_mask;
    }

#if defined( VULKAN_HPP_FLAGS_MASK_TYPE_AS_PUBLIC )
  public:
#else
  private:
#endif
    MaskType m_mask;
  };

#if !defined( VULKAN_HPP_HAS_SPACESHIP_OPERATOR )
  // relational operators only needed for pre C++20
  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator<( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator>( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator<=( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator>=( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator>( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator<( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator>=( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator<=( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator==( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator==( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator!=( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator!=( bit );
  }
#endif

  // bitwise operators
  template <typename BitType>
  VULKAN_HPP_CONSTEXPR Flags<BitType> operator&(BitType bit, Flags<BitType> const & flags)VULKAN_HPP_NOEXCEPT
  {
    return flags.operator&( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR Flags<BitType> operator|( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator|( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR Flags<BitType> operator^( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator^( bit );
  }

  template <typename RefType>
  class Optional
  {
  public:
    Optional( RefType & reference ) VULKAN_HPP_NOEXCEPT
    {
      m_ptr = &reference;
    }
    Optional( RefType * ptr ) VULKAN_HPP_NOEXCEPT
    {
      m_ptr = ptr;
    }
    Optional( std::nullptr_t ) VULKAN_HPP_NOEXCEPT
    {
      m_ptr = nullptr;
    }

    operator RefType *() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr;
    }
    RefType const * operator->() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr;
    }
    explicit operator bool() const VULKAN_HPP_NOEXCEPT
    {
      return !!m_ptr;
    }

  private:
    RefType * m_ptr;
  };

  template <typename X, typename Y>
  struct StructExtends
  {
    enum
    {
      value = false
    };
  };

  template <typename Type, class...>
  struct IsPartOfStructureChain
  {
    static const bool valid = false;
  };

  template <typename Type, typename Head, typename... Tail>
  struct IsPartOfStructureChain<Type, Head, Tail...>
  {
    static const bool valid = std::is_same<Type, Head>::value || IsPartOfStructureChain<Type, Tail...>::valid;
  };

  template <size_t Index, typename T, typename... ChainElements>
  struct StructureChainContains
  {
    static const bool value =
      std::is_same<T, typename std::tuple_element<Index, std::tuple<ChainElements...>>::type>::value ||
      StructureChainContains<Index - 1, T, ChainElements...>::value;
  };

  template <typename T, typename... ChainElements>
  struct StructureChainContains<0, T, ChainElements...>
  {
    static const bool value =
      std::is_same<T, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value;
  };

  template <size_t Index, typename... ChainElements>
  struct StructureChainValidation
  {
    using TestType = typename std::tuple_element<Index, std::tuple<ChainElements...>>::type;
    static const bool valid =
      StructExtends<TestType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value &&
      ( TestType::allowDuplicate || !StructureChainContains<Index - 1, TestType, ChainElements...>::value ) &&
      StructureChainValidation<Index - 1, ChainElements...>::valid;
  };

  template <typename... ChainElements>
  struct StructureChainValidation<0, ChainElements...>
  {
    static const bool valid = true;
  };

  template <typename... ChainElements>
  class StructureChain : public std::tuple<ChainElements...>
  {
  public:
    StructureChain() VULKAN_HPP_NOEXCEPT
    {
      static_assert( StructureChainValidation<sizeof...( ChainElements ) - 1, ChainElements...>::valid,
                     "The structure chain is not valid!" );
      link<sizeof...( ChainElements ) - 1>();
    }

    StructureChain( StructureChain const & rhs ) VULKAN_HPP_NOEXCEPT : std::tuple<ChainElements...>( rhs )
    {
      static_assert( StructureChainValidation<sizeof...( ChainElements ) - 1, ChainElements...>::valid,
                     "The structure chain is not valid!" );
      link<sizeof...( ChainElements ) - 1>();
    }

    StructureChain( StructureChain && rhs ) VULKAN_HPP_NOEXCEPT
      : std::tuple<ChainElements...>( std::forward<std::tuple<ChainElements...>>( rhs ) )
    {
      static_assert( StructureChainValidation<sizeof...( ChainElements ) - 1, ChainElements...>::valid,
                     "The structure chain is not valid!" );
      link<sizeof...( ChainElements ) - 1>();
    }

    StructureChain( ChainElements const &... elems ) VULKAN_HPP_NOEXCEPT : std::tuple<ChainElements...>( elems... )
    {
      static_assert( StructureChainValidation<sizeof...( ChainElements ) - 1, ChainElements...>::valid,
                     "The structure chain is not valid!" );
      link<sizeof...( ChainElements ) - 1>();
    }

    StructureChain & operator=( StructureChain const & rhs ) VULKAN_HPP_NOEXCEPT
    {
      std::tuple<ChainElements...>::operator=( rhs );
      link<sizeof...( ChainElements ) - 1>();
      return *this;
    }

    StructureChain & operator=( StructureChain && rhs ) = delete;

    template <typename T = typename std::tuple_element<0, std::tuple<ChainElements...>>::type, size_t Which = 0>
    T & get() VULKAN_HPP_NOEXCEPT
    {
      return std::get<ChainElementIndex<0, T, Which, void, ChainElements...>::value>(
        static_cast<std::tuple<ChainElements...> &>( *this ) );
    }

    template <typename T = typename std::tuple_element<0, std::tuple<ChainElements...>>::type, size_t Which = 0>
    T const & get() const VULKAN_HPP_NOEXCEPT
    {
      return std::get<ChainElementIndex<0, T, Which, void, ChainElements...>::value>(
        static_cast<std::tuple<ChainElements...> const &>( *this ) );
    }

    template <typename T0, typename T1, typename... Ts>
    std::tuple<T0 &, T1 &, Ts &...> get() VULKAN_HPP_NOEXCEPT
    {
      return std::tie( get<T0>(), get<T1>(), get<Ts>()... );
    }

    template <typename T0, typename T1, typename... Ts>
    std::tuple<T0 const &, T1 const &, Ts const &...> get() const VULKAN_HPP_NOEXCEPT
    {
      return std::tie( get<T0>(), get<T1>(), get<Ts>()... );
    }

    template <typename ClassType, size_t Which = 0>
    typename std::enable_if<
      std::is_same<ClassType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value &&
        ( Which == 0 ),
      bool>::type
      isLinked() const VULKAN_HPP_NOEXCEPT
    {
      return true;
    }

    template <typename ClassType, size_t Which = 0>
    typename std::enable_if<
      !std::is_same<ClassType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value ||
        ( Which != 0 ),
      bool>::type
      isLinked() const VULKAN_HPP_NOEXCEPT
    {
      static_assert( IsPartOfStructureChain<ClassType, ChainElements...>::valid,
                     "Can't unlink Structure that's not part of this StructureChain!" );
      return isLinked( reinterpret_cast<VkBaseInStructure const *>( &get<ClassType, Which>() ) );
    }

    template <typename ClassType, size_t Which = 0>
    typename std::enable_if<
      !std::is_same<ClassType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value ||
        ( Which != 0 ),
      void>::type
      relink() VULKAN_HPP_NOEXCEPT
    {
      static_assert( IsPartOfStructureChain<ClassType, ChainElements...>::valid,
                     "Can't relink Structure that's not part of this StructureChain!" );
      auto pNext = reinterpret_cast<VkBaseInStructure *>( &get<ClassType, Which>() );
      VULKAN_HPP_ASSERT( !isLinked( pNext ) );
      auto & headElement = std::get<0>( static_cast<std::tuple<ChainElements...> &>( *this ) );
      pNext->pNext       = reinterpret_cast<VkBaseInStructure const *>( headElement.pNext );
      headElement.pNext  = pNext;
    }

    template <typename ClassType, size_t Which = 0>
    typename std::enable_if<
      !std::is_same<ClassType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value ||
        ( Which != 0 ),
      void>::type
      unlink() VULKAN_HPP_NOEXCEPT
    {
      static_assert( IsPartOfStructureChain<ClassType, ChainElements...>::valid,
                     "Can't unlink Structure that's not part of this StructureChain!" );
      unlink( reinterpret_cast<VkBaseOutStructure const *>( &get<ClassType, Which>() ) );
    }

  private:
    template <int Index, typename T, int Which, typename, class First, class... Types>
    struct ChainElementIndex : ChainElementIndex<Index + 1, T, Which, void, Types...>
    {};

    template <int Index, typename T, int Which, class First, class... Types>
    struct ChainElementIndex<Index,
                             T,
                             Which,
                             typename std::enable_if<!std::is_same<T, First>::value, void>::type,
                             First,
                             Types...> : ChainElementIndex<Index + 1, T, Which, void, Types...>
    {};

    template <int Index, typename T, int Which, class First, class... Types>
    struct ChainElementIndex<Index,
                             T,
                             Which,
                             typename std::enable_if<std::is_same<T, First>::value, void>::type,
                             First,
                             Types...> : ChainElementIndex<Index + 1, T, Which - 1, void, Types...>
    {};

    template <int Index, typename T, class First, class... Types>
    struct ChainElementIndex<Index,
                             T,
                             0,
                             typename std::enable_if<std::is_same<T, First>::value, void>::type,
                             First,
                             Types...> : std::integral_constant<int, Index>
    {};

    bool isLinked( VkBaseInStructure const * pNext ) const VULKAN_HPP_NOEXCEPT
    {
      VkBaseInStructure const * elementPtr = reinterpret_cast<VkBaseInStructure const *>(
        &std::get<0>( static_cast<std::tuple<ChainElements...> const &>( *this ) ) );
      while ( elementPtr )
      {
        if ( elementPtr->pNext == pNext )
        {
          return true;
        }
        elementPtr = elementPtr->pNext;
      }
      return false;
    }

    template <size_t Index>
    typename std::enable_if<Index != 0, void>::type link() VULKAN_HPP_NOEXCEPT
    {
      auto & x = std::get<Index - 1>( static_cast<std::tuple<ChainElements...> &>( *this ) );
      x.pNext  = &std::get<Index>( static_cast<std::tuple<ChainElements...> &>( *this ) );
      link<Index - 1>();
    }

    template <size_t Index>
    typename std::enable_if<Index == 0, void>::type link() VULKAN_HPP_NOEXCEPT
    {}

    void unlink( VkBaseOutStructure const * pNext ) VULKAN_HPP_NOEXCEPT
    {
      VkBaseOutStructure * elementPtr =
        reinterpret_cast<VkBaseOutStructure *>( &std::get<0>( static_cast<std::tuple<ChainElements...> &>( *this ) ) );
      while ( elementPtr && ( elementPtr->pNext != pNext ) )
      {
        elementPtr = elementPtr->pNext;
      }
      if ( elementPtr )
      {
        elementPtr->pNext = pNext->pNext;
      }
      else
      {
        VULKAN_HPP_ASSERT( false );  // fires, if the ClassType member has already been unlinked !
      }
    }
  };

#if !defined( VULKAN_HPP_NO_SMART_HANDLE )
  template <typename Type, typename Dispatch>
  class UniqueHandleTraits;

  template <typename Type, typename Dispatch>
  class UniqueHandle : public UniqueHandleTraits<Type, Dispatch>::deleter
  {
  private:
    using Deleter = typename UniqueHandleTraits<Type, Dispatch>::deleter;

  public:
    using element_type = Type;

    UniqueHandle() : Deleter(), m_value() {}

    explicit UniqueHandle( Type const & value, Deleter const & deleter = Deleter() ) VULKAN_HPP_NOEXCEPT
      : Deleter( deleter )
      , m_value( value )
    {}

    UniqueHandle( UniqueHandle const & ) = delete;

    UniqueHandle( UniqueHandle && other ) VULKAN_HPP_NOEXCEPT
      : Deleter( std::move( static_cast<Deleter &>( other ) ) )
      , m_value( other.release() )
    {}

    ~UniqueHandle() VULKAN_HPP_NOEXCEPT
    {
      if ( m_value )
      {
        this->destroy( m_value );
      }
    }

    UniqueHandle & operator=( UniqueHandle const & ) = delete;

    UniqueHandle & operator=( UniqueHandle && other ) VULKAN_HPP_NOEXCEPT
    {
      reset( other.release() );
      *static_cast<Deleter *>( this ) = std::move( static_cast<Deleter &>( other ) );
      return *this;
    }

    explicit operator bool() const VULKAN_HPP_NOEXCEPT
    {
      return m_value.operator bool();
    }

    Type const * operator->() const VULKAN_HPP_NOEXCEPT
    {
      return &m_value;
    }

    Type * operator->() VULKAN_HPP_NOEXCEPT
    {
      return &m_value;
    }

    Type const & operator*() const VULKAN_HPP_NOEXCEPT
    {
      return m_value;
    }

    Type & operator*() VULKAN_HPP_NOEXCEPT
    {
      return m_value;
    }

    const Type & get() const VULKAN_HPP_NOEXCEPT
    {
      return m_value;
    }

    Type & get() VULKAN_HPP_NOEXCEPT
    {
      return m_value;
    }

    void reset( Type const & value = Type() ) VULKAN_HPP_NOEXCEPT
    {
      if ( m_value != value )
      {
        if ( m_value )
        {
          this->destroy( m_value );
        }
        m_value = value;
      }
    }

    Type release() VULKAN_HPP_NOEXCEPT
    {
      Type value = m_value;
      m_value    = nullptr;
      return value;
    }

    void swap( UniqueHandle<Type, Dispatch> & rhs ) VULKAN_HPP_NOEXCEPT
    {
      std::swap( m_value, rhs.m_value );
      std::swap( static_cast<Deleter &>( *this ), static_cast<Deleter &>( rhs ) );
    }

  private:
    Type m_value;
  };

  template <typename UniqueType>
  VULKAN_HPP_INLINE std::vector<typename UniqueType::element_type>
                    uniqueToRaw( std::vector<UniqueType> const & handles )
  {
    std::vector<typename UniqueType::element_type> newBuffer( handles.size() );
    std::transform(
      handles.begin(), handles.end(), newBuffer.begin(), []( UniqueType const & handle ) { return handle.get(); } );
    return newBuffer;
  }

  template <typename Type, typename Dispatch>
  VULKAN_HPP_INLINE void swap( UniqueHandle<Type, Dispatch> & lhs,
                               UniqueHandle<Type, Dispatch> & rhs ) VULKAN_HPP_NOEXCEPT
  {
    lhs.swap( rhs );
  }
#endif

#if !defined( VK_NO_PROTOTYPES )
  class DispatchLoaderStatic
  {
  public:
    //=== VK_VERSION_1_0 ===

    VkResult vkCreateInstance( const VkInstanceCreateInfo *  pCreateInfo,
                               const VkAllocationCallbacks * pAllocator,
                               VkInstance *                  pInstance ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateInstance( pCreateInfo, pAllocator, pInstance );
    }

    void vkDestroyInstance( VkInstance instance, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyInstance( instance, pAllocator );
    }

    VkResult vkEnumeratePhysicalDevices( VkInstance         instance,
                                         uint32_t *         pPhysicalDeviceCount,
                                         VkPhysicalDevice * pPhysicalDevices ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumeratePhysicalDevices( instance, pPhysicalDeviceCount, pPhysicalDevices );
    }

    void vkGetPhysicalDeviceFeatures( VkPhysicalDevice           physicalDevice,
                                      VkPhysicalDeviceFeatures * pFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFeatures( physicalDevice, pFeatures );
    }

    void vkGetPhysicalDeviceFormatProperties( VkPhysicalDevice     physicalDevice,
                                              VkFormat             format,
                                              VkFormatProperties * pFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFormatProperties( physicalDevice, format, pFormatProperties );
    }

    VkResult vkGetPhysicalDeviceImageFormatProperties( VkPhysicalDevice          physicalDevice,
                                                       VkFormat                  format,
                                                       VkImageType               type,
                                                       VkImageTiling             tiling,
                                                       VkImageUsageFlags         usage,
                                                       VkImageCreateFlags        flags,
                                                       VkImageFormatProperties * pImageFormatProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceImageFormatProperties(
        physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties );
    }

    void vkGetPhysicalDeviceProperties( VkPhysicalDevice             physicalDevice,
                                        VkPhysicalDeviceProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceProperties( physicalDevice, pProperties );
    }

    void vkGetPhysicalDeviceQueueFamilyProperties( VkPhysicalDevice          physicalDevice,
                                                   uint32_t *                pQueueFamilyPropertyCount,
                                                   VkQueueFamilyProperties * pQueueFamilyProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties );
    }

    void vkGetPhysicalDeviceMemoryProperties(
      VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties * pMemoryProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceMemoryProperties( physicalDevice, pMemoryProperties );
    }

    PFN_vkVoidFunction vkGetInstanceProcAddr( VkInstance instance, const char * pName ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetInstanceProcAddr( instance, pName );
    }

    PFN_vkVoidFunction vkGetDeviceProcAddr( VkDevice device, const char * pName ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceProcAddr( device, pName );
    }

    VkResult vkCreateDevice( VkPhysicalDevice              physicalDevice,
                             const VkDeviceCreateInfo *    pCreateInfo,
                             const VkAllocationCallbacks * pAllocator,
                             VkDevice *                    pDevice ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDevice( physicalDevice, pCreateInfo, pAllocator, pDevice );
    }

    void vkDestroyDevice( VkDevice device, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDevice( device, pAllocator );
    }

    VkResult vkEnumerateInstanceExtensionProperties( const char *            pLayerName,
                                                     uint32_t *              pPropertyCount,
                                                     VkExtensionProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumerateInstanceExtensionProperties( pLayerName, pPropertyCount, pProperties );
    }

    VkResult vkEnumerateDeviceExtensionProperties( VkPhysicalDevice        physicalDevice,
                                                   const char *            pLayerName,
                                                   uint32_t *              pPropertyCount,
                                                   VkExtensionProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumerateDeviceExtensionProperties( physicalDevice, pLayerName, pPropertyCount, pProperties );
    }

    VkResult vkEnumerateInstanceLayerProperties( uint32_t *          pPropertyCount,
                                                 VkLayerProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumerateInstanceLayerProperties( pPropertyCount, pProperties );
    }

    VkResult vkEnumerateDeviceLayerProperties( VkPhysicalDevice    physicalDevice,
                                               uint32_t *          pPropertyCount,
                                               VkLayerProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumerateDeviceLayerProperties( physicalDevice, pPropertyCount, pProperties );
    }

    void vkGetDeviceQueue( VkDevice  device,
                           uint32_t  queueFamilyIndex,
                           uint32_t  queueIndex,
                           VkQueue * pQueue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceQueue( device, queueFamilyIndex, queueIndex, pQueue );
    }

    VkResult vkQueueSubmit( VkQueue              queue,
                            uint32_t             submitCount,
                            const VkSubmitInfo * pSubmits,
                            VkFence              fence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueSubmit( queue, submitCount, pSubmits, fence );
    }

    VkResult vkQueueWaitIdle( VkQueue queue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueWaitIdle( queue );
    }

    VkResult vkDeviceWaitIdle( VkDevice device ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDeviceWaitIdle( device );
    }

    VkResult vkAllocateMemory( VkDevice                      device,
                               const VkMemoryAllocateInfo *  pAllocateInfo,
                               const VkAllocationCallbacks * pAllocator,
                               VkDeviceMemory *              pMemory ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAllocateMemory( device, pAllocateInfo, pAllocator, pMemory );
    }

    void vkFreeMemory( VkDevice                      device,
                       VkDeviceMemory                memory,
                       const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkFreeMemory( device, memory, pAllocator );
    }

    VkResult vkMapMemory( VkDevice         device,
                          VkDeviceMemory   memory,
                          VkDeviceSize     offset,
                          VkDeviceSize     size,
                          VkMemoryMapFlags flags,
                          void **          ppData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkMapMemory( device, memory, offset, size, flags, ppData );
    }

    void vkUnmapMemory( VkDevice device, VkDeviceMemory memory ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUnmapMemory( device, memory );
    }

    VkResult vkFlushMappedMemoryRanges( VkDevice                    device,
                                        uint32_t                    memoryRangeCount,
                                        const VkMappedMemoryRange * pMemoryRanges ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkFlushMappedMemoryRanges( device, memoryRangeCount, pMemoryRanges );
    }

    VkResult vkInvalidateMappedMemoryRanges( VkDevice                    device,
                                             uint32_t                    memoryRangeCount,
                                             const VkMappedMemoryRange * pMemoryRanges ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkInvalidateMappedMemoryRanges( device, memoryRangeCount, pMemoryRanges );
    }

    void vkGetDeviceMemoryCommitment( VkDevice       device,
                                      VkDeviceMemory memory,
                                      VkDeviceSize * pCommittedMemoryInBytes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceMemoryCommitment( device, memory, pCommittedMemoryInBytes );
    }

    VkResult vkBindBufferMemory( VkDevice       device,
                                 VkBuffer       buffer,
                                 VkDeviceMemory memory,
                                 VkDeviceSize   memoryOffset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindBufferMemory( device, buffer, memory, memoryOffset );
    }

    VkResult vkBindImageMemory( VkDevice       device,
                                VkImage        image,
                                VkDeviceMemory memory,
                                VkDeviceSize   memoryOffset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindImageMemory( device, image, memory, memoryOffset );
    }

    void vkGetBufferMemoryRequirements( VkDevice               device,
                                        VkBuffer               buffer,
                                        VkMemoryRequirements * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferMemoryRequirements( device, buffer, pMemoryRequirements );
    }

    void vkGetImageMemoryRequirements( VkDevice               device,
                                       VkImage                image,
                                       VkMemoryRequirements * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageMemoryRequirements( device, image, pMemoryRequirements );
    }

    void vkGetImageSparseMemoryRequirements( VkDevice                          device,
                                             VkImage                           image,
                                             uint32_t *                        pSparseMemoryRequirementCount,
                                             VkSparseImageMemoryRequirements * pSparseMemoryRequirements ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSparseMemoryRequirements(
        device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements );
    }

    void vkGetPhysicalDeviceSparseImageFormatProperties( VkPhysicalDevice                physicalDevice,
                                                         VkFormat                        format,
                                                         VkImageType                     type,
                                                         VkSampleCountFlagBits           samples,
                                                         VkImageUsageFlags               usage,
                                                         VkImageTiling                   tiling,
                                                         uint32_t *                      pPropertyCount,
                                                         VkSparseImageFormatProperties * pProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSparseImageFormatProperties(
        physicalDevice, format, type, samples, usage, tiling, pPropertyCount, pProperties );
    }

    VkResult vkQueueBindSparse( VkQueue                  queue,
                                uint32_t                 bindInfoCount,
                                const VkBindSparseInfo * pBindInfo,
                                VkFence                  fence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueBindSparse( queue, bindInfoCount, pBindInfo, fence );
    }

    VkResult vkCreateFence( VkDevice                      device,
                            const VkFenceCreateInfo *     pCreateInfo,
                            const VkAllocationCallbacks * pAllocator,
                            VkFence *                     pFence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateFence( device, pCreateInfo, pAllocator, pFence );
    }

    void vkDestroyFence( VkDevice                      device,
                         VkFence                       fence,
                         const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyFence( device, fence, pAllocator );
    }

    VkResult vkResetFences( VkDevice device, uint32_t fenceCount, const VkFence * pFences ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetFences( device, fenceCount, pFences );
    }

    VkResult vkGetFenceStatus( VkDevice device, VkFence fence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetFenceStatus( device, fence );
    }

    VkResult vkWaitForFences( VkDevice        device,
                              uint32_t        fenceCount,
                              const VkFence * pFences,
                              VkBool32        waitAll,
                              uint64_t        timeout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWaitForFences( device, fenceCount, pFences, waitAll, timeout );
    }

    VkResult vkCreateSemaphore( VkDevice                      device,
                                const VkSemaphoreCreateInfo * pCreateInfo,
                                const VkAllocationCallbacks * pAllocator,
                                VkSemaphore *                 pSemaphore ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateSemaphore( device, pCreateInfo, pAllocator, pSemaphore );
    }

    void vkDestroySemaphore( VkDevice                      device,
                             VkSemaphore                   semaphore,
                             const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroySemaphore( device, semaphore, pAllocator );
    }

    VkResult vkCreateEvent( VkDevice                      device,
                            const VkEventCreateInfo *     pCreateInfo,
                            const VkAllocationCallbacks * pAllocator,
                            VkEvent *                     pEvent ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateEvent( device, pCreateInfo, pAllocator, pEvent );
    }

    void vkDestroyEvent( VkDevice                      device,
                         VkEvent                       event,
                         const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyEvent( device, event, pAllocator );
    }

    VkResult vkGetEventStatus( VkDevice device, VkEvent event ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetEventStatus( device, event );
    }

    VkResult vkSetEvent( VkDevice device, VkEvent event ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetEvent( device, event );
    }

    VkResult vkResetEvent( VkDevice device, VkEvent event ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetEvent( device, event );
    }

    VkResult vkCreateQueryPool( VkDevice                      device,
                                const VkQueryPoolCreateInfo * pCreateInfo,
                                const VkAllocationCallbacks * pAllocator,
                                VkQueryPool *                 pQueryPool ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateQueryPool( device, pCreateInfo, pAllocator, pQueryPool );
    }

    void vkDestroyQueryPool( VkDevice                      device,
                             VkQueryPool                   queryPool,
                             const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyQueryPool( device, queryPool, pAllocator );
    }

    VkResult vkGetQueryPoolResults( VkDevice           device,
                                    VkQueryPool        queryPool,
                                    uint32_t           firstQuery,
                                    uint32_t           queryCount,
                                    size_t             dataSize,
                                    void *             pData,
                                    VkDeviceSize       stride,
                                    VkQueryResultFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetQueryPoolResults( device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags );
    }

    VkResult vkCreateBuffer( VkDevice                      device,
                             const VkBufferCreateInfo *    pCreateInfo,
                             const VkAllocationCallbacks * pAllocator,
                             VkBuffer *                    pBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateBuffer( device, pCreateInfo, pAllocator, pBuffer );
    }

    void vkDestroyBuffer( VkDevice                      device,
                          VkBuffer                      buffer,
                          const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyBuffer( device, buffer, pAllocator );
    }

    VkResult vkCreateBufferView( VkDevice                       device,
                                 const VkBufferViewCreateInfo * pCreateInfo,
                                 const VkAllocationCallbacks *  pAllocator,
                                 VkBufferView *                 pView ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateBufferView( device, pCreateInfo, pAllocator, pView );
    }

    void vkDestroyBufferView( VkDevice                      device,
                              VkBufferView                  bufferView,
                              const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyBufferView( device, bufferView, pAllocator );
    }

    VkResult vkCreateImage( VkDevice                      device,
                            const VkImageCreateInfo *     pCreateInfo,
                            const VkAllocationCallbacks * pAllocator,
                            VkImage *                     pImage ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateImage( device, pCreateInfo, pAllocator, pImage );
    }

    void vkDestroyImage( VkDevice                      device,
                         VkImage                       image,
                         const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyImage( device, image, pAllocator );
    }

    void vkGetImageSubresourceLayout( VkDevice                   device,
                                      VkImage                    image,
                                      const VkImageSubresource * pSubresource,
                                      VkSubresourceLayout *      pLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSubresourceLayout( device, image, pSubresource, pLayout );
    }

    VkResult vkCreateImageView( VkDevice                      device,
                                const VkImageViewCreateInfo * pCreateInfo,
                                const VkAllocationCallbacks * pAllocator,
                                VkImageView *                 pView ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateImageView( device, pCreateInfo, pAllocator, pView );
    }

    void vkDestroyImageView( VkDevice                      device,
                             VkImageView                   imageView,
                             const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyImageView( device, imageView, pAllocator );
    }

    VkResult vkCreateShaderModule( VkDevice                         device,
                                   const VkShaderModuleCreateInfo * pCreateInfo,
                                   const VkAllocationCallbacks *    pAllocator,
                                   VkShaderModule *                 pShaderModule ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateShaderModule( device, pCreateInfo, pAllocator, pShaderModule );
    }

    void vkDestroyShaderModule( VkDevice                      device,
                                VkShaderModule                shaderModule,
                                const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyShaderModule( device, shaderModule, pAllocator );
    }

    VkResult vkCreatePipelineCache( VkDevice                          device,
                                    const VkPipelineCacheCreateInfo * pCreateInfo,
                                    const VkAllocationCallbacks *     pAllocator,
                                    VkPipelineCache *                 pPipelineCache ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreatePipelineCache( device, pCreateInfo, pAllocator, pPipelineCache );
    }

    void vkDestroyPipelineCache( VkDevice                      device,
                                 VkPipelineCache               pipelineCache,
                                 const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyPipelineCache( device, pipelineCache, pAllocator );
    }

    VkResult vkGetPipelineCacheData( VkDevice        device,
                                     VkPipelineCache pipelineCache,
                                     size_t *        pDataSize,
                                     void *          pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineCacheData( device, pipelineCache, pDataSize, pData );
    }

    VkResult vkMergePipelineCaches( VkDevice                device,
                                    VkPipelineCache         dstCache,
                                    uint32_t                srcCacheCount,
                                    const VkPipelineCache * pSrcCaches ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkMergePipelineCaches( device, dstCache, srcCacheCount, pSrcCaches );
    }

    VkResult vkCreateGraphicsPipelines( VkDevice                             device,
                                        VkPipelineCache                      pipelineCache,
                                        uint32_t                             createInfoCount,
                                        const VkGraphicsPipelineCreateInfo * pCreateInfos,
                                        const VkAllocationCallbacks *        pAllocator,
                                        VkPipeline *                         pPipelines ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateGraphicsPipelines(
        device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines );
    }

    VkResult vkCreateComputePipelines( VkDevice                            device,
                                       VkPipelineCache                     pipelineCache,
                                       uint32_t                            createInfoCount,
                                       const VkComputePipelineCreateInfo * pCreateInfos,
                                       const VkAllocationCallbacks *       pAllocator,
                                       VkPipeline *                        pPipelines ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateComputePipelines( device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines );
    }

    void vkDestroyPipeline( VkDevice                      device,
                            VkPipeline                    pipeline,
                            const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyPipeline( device, pipeline, pAllocator );
    }

    VkResult vkCreatePipelineLayout( VkDevice                           device,
                                     const VkPipelineLayoutCreateInfo * pCreateInfo,
                                     const VkAllocationCallbacks *      pAllocator,
                                     VkPipelineLayout *                 pPipelineLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreatePipelineLayout( device, pCreateInfo, pAllocator, pPipelineLayout );
    }

    void vkDestroyPipelineLayout( VkDevice                      device,
                                  VkPipelineLayout              pipelineLayout,
                                  const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyPipelineLayout( device, pipelineLayout, pAllocator );
    }

    VkResult vkCreateSampler( VkDevice                      device,
                              const VkSamplerCreateInfo *   pCreateInfo,
                              const VkAllocationCallbacks * pAllocator,
                              VkSampler *                   pSampler ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateSampler( device, pCreateInfo, pAllocator, pSampler );
    }

    void vkDestroySampler( VkDevice                      device,
                           VkSampler                     sampler,
                           const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroySampler( device, sampler, pAllocator );
    }

    VkResult vkCreateDescriptorSetLayout( VkDevice                                device,
                                          const VkDescriptorSetLayoutCreateInfo * pCreateInfo,
                                          const VkAllocationCallbacks *           pAllocator,
                                          VkDescriptorSetLayout *                 pSetLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDescriptorSetLayout( device, pCreateInfo, pAllocator, pSetLayout );
    }

    void vkDestroyDescriptorSetLayout( VkDevice                      device,
                                       VkDescriptorSetLayout         descriptorSetLayout,
                                       const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDescriptorSetLayout( device, descriptorSetLayout, pAllocator );
    }

    VkResult vkCreateDescriptorPool( VkDevice                           device,
                                     const VkDescriptorPoolCreateInfo * pCreateInfo,
                                     const VkAllocationCallbacks *      pAllocator,
                                     VkDescriptorPool *                 pDescriptorPool ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDescriptorPool( device, pCreateInfo, pAllocator, pDescriptorPool );
    }

    void vkDestroyDescriptorPool( VkDevice                      device,
                                  VkDescriptorPool              descriptorPool,
                                  const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDescriptorPool( device, descriptorPool, pAllocator );
    }

    VkResult vkResetDescriptorPool( VkDevice                   device,
                                    VkDescriptorPool           descriptorPool,
                                    VkDescriptorPoolResetFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetDescriptorPool( device, descriptorPool, flags );
    }

    VkResult vkAllocateDescriptorSets( VkDevice                            device,
                                       const VkDescriptorSetAllocateInfo * pAllocateInfo,
                                       VkDescriptorSet *                   pDescriptorSets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAllocateDescriptorSets( device, pAllocateInfo, pDescriptorSets );
    }

    VkResult vkFreeDescriptorSets( VkDevice                device,
                                   VkDescriptorPool        descriptorPool,
                                   uint32_t                descriptorSetCount,
                                   const VkDescriptorSet * pDescriptorSets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkFreeDescriptorSets( device, descriptorPool, descriptorSetCount, pDescriptorSets );
    }

    void vkUpdateDescriptorSets( VkDevice                     device,
                                 uint32_t                     descriptorWriteCount,
                                 const VkWriteDescriptorSet * pDescriptorWrites,
                                 uint32_t                     descriptorCopyCount,
                                 const VkCopyDescriptorSet *  pDescriptorCopies ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUpdateDescriptorSets(
        device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies );
    }

    VkResult vkCreateFramebuffer( VkDevice                        device,
                                  const VkFramebufferCreateInfo * pCreateInfo,
                                  const VkAllocationCallbacks *   pAllocator,
                                  VkFramebuffer *                 pFramebuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateFramebuffer( device, pCreateInfo, pAllocator, pFramebuffer );
    }

    void vkDestroyFramebuffer( VkDevice                      device,
                               VkFramebuffer                 framebuffer,
                               const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyFramebuffer( device, framebuffer, pAllocator );
    }

    VkResult vkCreateRenderPass( VkDevice                       device,
                                 const VkRenderPassCreateInfo * pCreateInfo,
                                 const VkAllocationCallbacks *  pAllocator,
                                 VkRenderPass *                 pRenderPass ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateRenderPass( device, pCreateInfo, pAllocator, pRenderPass );
    }

    void vkDestroyRenderPass( VkDevice                      device,
                              VkRenderPass                  renderPass,
                              const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyRenderPass( device, renderPass, pAllocator );
    }

    void vkGetRenderAreaGranularity( VkDevice     device,
                                     VkRenderPass renderPass,
                                     VkExtent2D * pGranularity ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRenderAreaGranularity( device, renderPass, pGranularity );
    }

    VkResult vkCreateCommandPool( VkDevice                        device,
                                  const VkCommandPoolCreateInfo * pCreateInfo,
                                  const VkAllocationCallbacks *   pAllocator,
                                  VkCommandPool *                 pCommandPool ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateCommandPool( device, pCreateInfo, pAllocator, pCommandPool );
    }

    void vkDestroyCommandPool( VkDevice                      device,
                               VkCommandPool                 commandPool,
                               const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyCommandPool( device, commandPool, pAllocator );
    }

    VkResult vkResetCommandPool( VkDevice                device,
                                 VkCommandPool           commandPool,
                                 VkCommandPoolResetFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetCommandPool( device, commandPool, flags );
    }

    VkResult vkAllocateCommandBuffers( VkDevice                            device,
                                       const VkCommandBufferAllocateInfo * pAllocateInfo,
                                       VkCommandBuffer *                   pCommandBuffers ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAllocateCommandBuffers( device, pAllocateInfo, pCommandBuffers );
    }

    void vkFreeCommandBuffers( VkDevice                device,
                               VkCommandPool           commandPool,
                               uint32_t                commandBufferCount,
                               const VkCommandBuffer * pCommandBuffers ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkFreeCommandBuffers( device, commandPool, commandBufferCount, pCommandBuffers );
    }

    VkResult vkBeginCommandBuffer( VkCommandBuffer                  commandBuffer,
                                   const VkCommandBufferBeginInfo * pBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBeginCommandBuffer( commandBuffer, pBeginInfo );
    }

    VkResult vkEndCommandBuffer( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEndCommandBuffer( commandBuffer );
    }

    VkResult vkResetCommandBuffer( VkCommandBuffer           commandBuffer,
                                   VkCommandBufferResetFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetCommandBuffer( commandBuffer, flags );
    }

    void vkCmdBindPipeline( VkCommandBuffer     commandBuffer,
                            VkPipelineBindPoint pipelineBindPoint,
                            VkPipeline          pipeline ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindPipeline( commandBuffer, pipelineBindPoint, pipeline );
    }

    void vkCmdSetViewport( VkCommandBuffer    commandBuffer,
                           uint32_t           firstViewport,
                           uint32_t           viewportCount,
                           const VkViewport * pViewports ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewport( commandBuffer, firstViewport, viewportCount, pViewports );
    }

    void vkCmdSetScissor( VkCommandBuffer  commandBuffer,
                          uint32_t         firstScissor,
                          uint32_t         scissorCount,
                          const VkRect2D * pScissors ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetScissor( commandBuffer, firstScissor, scissorCount, pScissors );
    }

    void vkCmdSetLineWidth( VkCommandBuffer commandBuffer, float lineWidth ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetLineWidth( commandBuffer, lineWidth );
    }

    void vkCmdSetDepthBias( VkCommandBuffer commandBuffer,
                            float           depthBiasConstantFactor,
                            float           depthBiasClamp,
                            float           depthBiasSlopeFactor ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBias( commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor );
    }

    void vkCmdSetBlendConstants( VkCommandBuffer commandBuffer,
                                 const float     blendConstants[4] ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetBlendConstants( commandBuffer, blendConstants );
    }

    void vkCmdSetDepthBounds( VkCommandBuffer commandBuffer,
                              float           minDepthBounds,
                              float           maxDepthBounds ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBounds( commandBuffer, minDepthBounds, maxDepthBounds );
    }

    void vkCmdSetStencilCompareMask( VkCommandBuffer    commandBuffer,
                                     VkStencilFaceFlags faceMask,
                                     uint32_t           compareMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilCompareMask( commandBuffer, faceMask, compareMask );
    }

    void vkCmdSetStencilWriteMask( VkCommandBuffer    commandBuffer,
                                   VkStencilFaceFlags faceMask,
                                   uint32_t           writeMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilWriteMask( commandBuffer, faceMask, writeMask );
    }

    void vkCmdSetStencilReference( VkCommandBuffer    commandBuffer,
                                   VkStencilFaceFlags faceMask,
                                   uint32_t           reference ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilReference( commandBuffer, faceMask, reference );
    }

    void vkCmdBindDescriptorSets( VkCommandBuffer         commandBuffer,
                                  VkPipelineBindPoint     pipelineBindPoint,
                                  VkPipelineLayout        layout,
                                  uint32_t                firstSet,
                                  uint32_t                descriptorSetCount,
                                  const VkDescriptorSet * pDescriptorSets,
                                  uint32_t                dynamicOffsetCount,
                                  const uint32_t *        pDynamicOffsets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindDescriptorSets( commandBuffer,
                                        pipelineBindPoint,
                                        layout,
                                        firstSet,
                                        descriptorSetCount,
                                        pDescriptorSets,
                                        dynamicOffsetCount,
                                        pDynamicOffsets );
    }

    void vkCmdBindIndexBuffer( VkCommandBuffer commandBuffer,
                               VkBuffer        buffer,
                               VkDeviceSize    offset,
                               VkIndexType     indexType ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindIndexBuffer( commandBuffer, buffer, offset, indexType );
    }

    void vkCmdBindVertexBuffers( VkCommandBuffer      commandBuffer,
                                 uint32_t             firstBinding,
                                 uint32_t             bindingCount,
                                 const VkBuffer *     pBuffers,
                                 const VkDeviceSize * pOffsets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindVertexBuffers( commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets );
    }

    void vkCmdDraw( VkCommandBuffer commandBuffer,
                    uint32_t        vertexCount,
                    uint32_t        instanceCount,
                    uint32_t        firstVertex,
                    uint32_t        firstInstance ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDraw( commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance );
    }

    void vkCmdDrawIndexed( VkCommandBuffer commandBuffer,
                           uint32_t        indexCount,
                           uint32_t        instanceCount,
                           uint32_t        firstIndex,
                           int32_t         vertexOffset,
                           uint32_t        firstInstance ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexed( commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance );
    }

    void vkCmdDrawIndirect( VkCommandBuffer commandBuffer,
                            VkBuffer        buffer,
                            VkDeviceSize    offset,
                            uint32_t        drawCount,
                            uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndirect( commandBuffer, buffer, offset, drawCount, stride );
    }

    void vkCmdDrawIndexedIndirect( VkCommandBuffer commandBuffer,
                                   VkBuffer        buffer,
                                   VkDeviceSize    offset,
                                   uint32_t        drawCount,
                                   uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexedIndirect( commandBuffer, buffer, offset, drawCount, stride );
    }

    void vkCmdDispatch( VkCommandBuffer commandBuffer,
                        uint32_t        groupCountX,
                        uint32_t        groupCountY,
                        uint32_t        groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatch( commandBuffer, groupCountX, groupCountY, groupCountZ );
    }

    void vkCmdDispatchIndirect( VkCommandBuffer commandBuffer,
                                VkBuffer        buffer,
                                VkDeviceSize    offset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatchIndirect( commandBuffer, buffer, offset );
    }

    void vkCmdCopyBuffer( VkCommandBuffer      commandBuffer,
                          VkBuffer             srcBuffer,
                          VkBuffer             dstBuffer,
                          uint32_t             regionCount,
                          const VkBufferCopy * pRegions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyBuffer( commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions );
    }

    void vkCmdCopyImage( VkCommandBuffer     commandBuffer,
                         VkImage             srcImage,
                         VkImageLayout       srcImageLayout,
                         VkImage             dstImage,
                         VkImageLayout       dstImageLayout,
                         uint32_t            regionCount,
                         const VkImageCopy * pRegions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyImage(
        commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions );
    }

    void vkCmdBlitImage( VkCommandBuffer     commandBuffer,
                         VkImage             srcImage,
                         VkImageLayout       srcImageLayout,
                         VkImage             dstImage,
                         VkImageLayout       dstImageLayout,
                         uint32_t            regionCount,
                         const VkImageBlit * pRegions,
                         VkFilter            filter ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBlitImage(
        commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter );
    }

    void vkCmdCopyBufferToImage( VkCommandBuffer           commandBuffer,
                                 VkBuffer                  srcBuffer,
                                 VkImage                   dstImage,
                                 VkImageLayout             dstImageLayout,
                                 uint32_t                  regionCount,
                                 const VkBufferImageCopy * pRegions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyBufferToImage( commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions );
    }

    void vkCmdCopyImageToBuffer( VkCommandBuffer           commandBuffer,
                                 VkImage                   srcImage,
                                 VkImageLayout             srcImageLayout,
                                 VkBuffer                  dstBuffer,
                                 uint32_t                  regionCount,
                                 const VkBufferImageCopy * pRegions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyImageToBuffer( commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions );
    }

    void vkCmdUpdateBuffer( VkCommandBuffer commandBuffer,
                            VkBuffer        dstBuffer,
                            VkDeviceSize    dstOffset,
                            VkDeviceSize    dataSize,
                            const void *    pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdUpdateBuffer( commandBuffer, dstBuffer, dstOffset, dataSize, pData );
    }

    void vkCmdFillBuffer( VkCommandBuffer commandBuffer,
                          VkBuffer        dstBuffer,
                          VkDeviceSize    dstOffset,
                          VkDeviceSize    size,
                          uint32_t        data ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdFillBuffer( commandBuffer, dstBuffer, dstOffset, size, data );
    }

    void vkCmdClearColorImage( VkCommandBuffer                 commandBuffer,
                               VkImage                         image,
                               VkImageLayout                   imageLayout,
                               const VkClearColorValue *       pColor,
                               uint32_t                        rangeCount,
                               const VkImageSubresourceRange * pRanges ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdClearColorImage( commandBuffer, image, imageLayout, pColor, rangeCount, pRanges );
    }

    void vkCmdClearDepthStencilImage( VkCommandBuffer                  commandBuffer,
                                      VkImage                          image,
                                      VkImageLayout                    imageLayout,
                                      const VkClearDepthStencilValue * pDepthStencil,
                                      uint32_t                         rangeCount,
                                      const VkImageSubresourceRange *  pRanges ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdClearDepthStencilImage( commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges );
    }

    void vkCmdClearAttachments( VkCommandBuffer           commandBuffer,
                                uint32_t                  attachmentCount,
                                const VkClearAttachment * pAttachments,
                                uint32_t                  rectCount,
                                const VkClearRect *       pRects ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdClearAttachments( commandBuffer, attachmentCount, pAttachments, rectCount, pRects );
    }

    void vkCmdResolveImage( VkCommandBuffer        commandBuffer,
                            VkImage                srcImage,
                            VkImageLayout          srcImageLayout,
                            VkImage                dstImage,
                            VkImageLayout          dstImageLayout,
                            uint32_t               regionCount,
                            const VkImageResolve * pRegions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResolveImage(
        commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions );
    }

    void vkCmdSetEvent( VkCommandBuffer      commandBuffer,
                        VkEvent              event,
                        VkPipelineStageFlags stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetEvent( commandBuffer, event, stageMask );
    }

    void vkCmdResetEvent( VkCommandBuffer      commandBuffer,
                          VkEvent              event,
                          VkPipelineStageFlags stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResetEvent( commandBuffer, event, stageMask );
    }

    void vkCmdWaitEvents( VkCommandBuffer               commandBuffer,
                          uint32_t                      eventCount,
                          const VkEvent *               pEvents,
                          VkPipelineStageFlags          srcStageMask,
                          VkPipelineStageFlags          dstStageMask,
                          uint32_t                      memoryBarrierCount,
                          const VkMemoryBarrier *       pMemoryBarriers,
                          uint32_t                      bufferMemoryBarrierCount,
                          const VkBufferMemoryBarrier * pBufferMemoryBarriers,
                          uint32_t                      imageMemoryBarrierCount,
                          const VkImageMemoryBarrier *  pImageMemoryBarriers ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWaitEvents( commandBuffer,
                                eventCount,
                                pEvents,
                                srcStageMask,
                                dstStageMask,
                                memoryBarrierCount,
                                pMemoryBarriers,
                                bufferMemoryBarrierCount,
                                pBufferMemoryBarriers,
                                imageMemoryBarrierCount,
                                pImageMemoryBarriers );
    }

    void vkCmdPipelineBarrier( VkCommandBuffer               commandBuffer,
                               VkPipelineStageFlags          srcStageMask,
                               VkPipelineStageFlags          dstStageMask,
                               VkDependencyFlags             dependencyFlags,
                               uint32_t                      memoryBarrierCount,
                               const VkMemoryBarrier *       pMemoryBarriers,
                               uint32_t                      bufferMemoryBarrierCount,
                               const VkBufferMemoryBarrier * pBufferMemoryBarriers,
                               uint32_t                      imageMemoryBarrierCount,
                               const VkImageMemoryBarrier *  pImageMemoryBarriers ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPipelineBarrier( commandBuffer,
                                     srcStageMask,
                                     dstStageMask,
                                     dependencyFlags,
                                     memoryBarrierCount,
                                     pMemoryBarriers,
                                     bufferMemoryBarrierCount,
                                     pBufferMemoryBarriers,
                                     imageMemoryBarrierCount,
                                     pImageMemoryBarriers );
    }

    void vkCmdBeginQuery( VkCommandBuffer     commandBuffer,
                          VkQueryPool         queryPool,
                          uint32_t            query,
                          VkQueryControlFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginQuery( commandBuffer, queryPool, query, flags );
    }

    void vkCmdEndQuery( VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndQuery( commandBuffer, queryPool, query );
    }

    void vkCmdResetQueryPool( VkCommandBuffer commandBuffer,
                              VkQueryPool     queryPool,
                              uint32_t        firstQuery,
                              uint32_t        queryCount ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResetQueryPool( commandBuffer, queryPool, firstQuery, queryCount );
    }

    void vkCmdWriteTimestamp( VkCommandBuffer         commandBuffer,
                              VkPipelineStageFlagBits pipelineStage,
                              VkQueryPool             queryPool,
                              uint32_t                query ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteTimestamp( commandBuffer, pipelineStage, queryPool, query );
    }

    void vkCmdCopyQueryPoolResults( VkCommandBuffer    commandBuffer,
                                    VkQueryPool        queryPool,
                                    uint32_t           firstQuery,
                                    uint32_t           queryCount,
                                    VkBuffer           dstBuffer,
                                    VkDeviceSize       dstOffset,
                                    VkDeviceSize       stride,
                                    VkQueryResultFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyQueryPoolResults(
        commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags );
    }

    void vkCmdPushConstants( VkCommandBuffer    commandBuffer,
                             VkPipelineLayout   layout,
                             VkShaderStageFlags stageFlags,
                             uint32_t           offset,
                             uint32_t           size,
                             const void *       pValues ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPushConstants( commandBuffer, layout, stageFlags, offset, size, pValues );
    }

    void vkCmdBeginRenderPass( VkCommandBuffer               commandBuffer,
                               const VkRenderPassBeginInfo * pRenderPassBegin,
                               VkSubpassContents             contents ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginRenderPass( commandBuffer, pRenderPassBegin, contents );
    }

    void vkCmdNextSubpass( VkCommandBuffer commandBuffer, VkSubpassContents contents ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdNextSubpass( commandBuffer, contents );
    }

    void vkCmdEndRenderPass( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndRenderPass( commandBuffer );
    }

    void vkCmdExecuteCommands( VkCommandBuffer         commandBuffer,
                               uint32_t                commandBufferCount,
                               const VkCommandBuffer * pCommandBuffers ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdExecuteCommands( commandBuffer, commandBufferCount, pCommandBuffers );
    }

    //=== VK_VERSION_1_1 ===

    VkResult vkEnumerateInstanceVersion( uint32_t * pApiVersion ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumerateInstanceVersion( pApiVersion );
    }

    VkResult vkBindBufferMemory2( VkDevice                       device,
                                  uint32_t                       bindInfoCount,
                                  const VkBindBufferMemoryInfo * pBindInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindBufferMemory2( device, bindInfoCount, pBindInfos );
    }

    VkResult vkBindImageMemory2( VkDevice                      device,
                                 uint32_t                      bindInfoCount,
                                 const VkBindImageMemoryInfo * pBindInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindImageMemory2( device, bindInfoCount, pBindInfos );
    }

    void vkGetDeviceGroupPeerMemoryFeatures( VkDevice                   device,
                                             uint32_t                   heapIndex,
                                             uint32_t                   localDeviceIndex,
                                             uint32_t                   remoteDeviceIndex,
                                             VkPeerMemoryFeatureFlags * pPeerMemoryFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceGroupPeerMemoryFeatures(
        device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures );
    }

    void vkCmdSetDeviceMask( VkCommandBuffer commandBuffer, uint32_t deviceMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDeviceMask( commandBuffer, deviceMask );
    }

    void vkCmdDispatchBase( VkCommandBuffer commandBuffer,
                            uint32_t        baseGroupX,
                            uint32_t        baseGroupY,
                            uint32_t        baseGroupZ,
                            uint32_t        groupCountX,
                            uint32_t        groupCountY,
                            uint32_t        groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatchBase(
        commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ );
    }

    VkResult vkEnumeratePhysicalDeviceGroups( VkInstance                        instance,
                                              uint32_t *                        pPhysicalDeviceGroupCount,
                                              VkPhysicalDeviceGroupProperties * pPhysicalDeviceGroupProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumeratePhysicalDeviceGroups( instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties );
    }

    void vkGetImageMemoryRequirements2( VkDevice                               device,
                                        const VkImageMemoryRequirementsInfo2 * pInfo,
                                        VkMemoryRequirements2 * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageMemoryRequirements2( device, pInfo, pMemoryRequirements );
    }

    void vkGetBufferMemoryRequirements2( VkDevice                                device,
                                         const VkBufferMemoryRequirementsInfo2 * pInfo,
                                         VkMemoryRequirements2 * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferMemoryRequirements2( device, pInfo, pMemoryRequirements );
    }

    void vkGetImageSparseMemoryRequirements2( VkDevice                                     device,
                                              const VkImageSparseMemoryRequirementsInfo2 * pInfo,
                                              uint32_t *                         pSparseMemoryRequirementCount,
                                              VkSparseImageMemoryRequirements2 * pSparseMemoryRequirements ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSparseMemoryRequirements2(
        device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements );
    }

    void vkGetPhysicalDeviceFeatures2( VkPhysicalDevice            physicalDevice,
                                       VkPhysicalDeviceFeatures2 * pFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFeatures2( physicalDevice, pFeatures );
    }

    void vkGetPhysicalDeviceProperties2( VkPhysicalDevice              physicalDevice,
                                         VkPhysicalDeviceProperties2 * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceProperties2( physicalDevice, pProperties );
    }

    void vkGetPhysicalDeviceFormatProperties2( VkPhysicalDevice      physicalDevice,
                                               VkFormat              format,
                                               VkFormatProperties2 * pFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFormatProperties2( physicalDevice, format, pFormatProperties );
    }

    VkResult vkGetPhysicalDeviceImageFormatProperties2( VkPhysicalDevice                         physicalDevice,
                                                        const VkPhysicalDeviceImageFormatInfo2 * pImageFormatInfo,
                                                        VkImageFormatProperties2 * pImageFormatProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceImageFormatProperties2( physicalDevice, pImageFormatInfo, pImageFormatProperties );
    }

    void vkGetPhysicalDeviceQueueFamilyProperties2( VkPhysicalDevice           physicalDevice,
                                                    uint32_t *                 pQueueFamilyPropertyCount,
                                                    VkQueueFamilyProperties2 * pQueueFamilyProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceQueueFamilyProperties2(
        physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties );
    }

    void vkGetPhysicalDeviceMemoryProperties2(
      VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2 * pMemoryProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceMemoryProperties2( physicalDevice, pMemoryProperties );
    }

    void vkGetPhysicalDeviceSparseImageFormatProperties2( VkPhysicalDevice                               physicalDevice,
                                                          const VkPhysicalDeviceSparseImageFormatInfo2 * pFormatInfo,
                                                          uint32_t *                                     pPropertyCount,
                                                          VkSparseImageFormatProperties2 * pProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSparseImageFormatProperties2(
        physicalDevice, pFormatInfo, pPropertyCount, pProperties );
    }

    void vkTrimCommandPool( VkDevice               device,
                            VkCommandPool          commandPool,
                            VkCommandPoolTrimFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkTrimCommandPool( device, commandPool, flags );
    }

    void vkGetDeviceQueue2( VkDevice                   device,
                            const VkDeviceQueueInfo2 * pQueueInfo,
                            VkQueue *                  pQueue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceQueue2( device, pQueueInfo, pQueue );
    }

    VkResult vkCreateSamplerYcbcrConversion( VkDevice                                   device,
                                             const VkSamplerYcbcrConversionCreateInfo * pCreateInfo,
                                             const VkAllocationCallbacks *              pAllocator,
                                             VkSamplerYcbcrConversion * pYcbcrConversion ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateSamplerYcbcrConversion( device, pCreateInfo, pAllocator, pYcbcrConversion );
    }

    void vkDestroySamplerYcbcrConversion( VkDevice                      device,
                                          VkSamplerYcbcrConversion      ycbcrConversion,
                                          const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroySamplerYcbcrConversion( device, ycbcrConversion, pAllocator );
    }

    VkResult vkCreateDescriptorUpdateTemplate( VkDevice                                     device,
                                               const VkDescriptorUpdateTemplateCreateInfo * pCreateInfo,
                                               const VkAllocationCallbacks *                pAllocator,
                                               VkDescriptorUpdateTemplate * pDescriptorUpdateTemplate ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDescriptorUpdateTemplate( device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate );
    }

    void vkDestroyDescriptorUpdateTemplate( VkDevice                      device,
                                            VkDescriptorUpdateTemplate    descriptorUpdateTemplate,
                                            const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDescriptorUpdateTemplate( device, descriptorUpdateTemplate, pAllocator );
    }

    void vkUpdateDescriptorSetWithTemplate( VkDevice                   device,
                                            VkDescriptorSet            descriptorSet,
                                            VkDescriptorUpdateTemplate descriptorUpdateTemplate,
                                            const void *               pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUpdateDescriptorSetWithTemplate( device, descriptorSet, descriptorUpdateTemplate, pData );
    }

    void vkGetPhysicalDeviceExternalBufferProperties( VkPhysicalDevice                           physicalDevice,
                                                      const VkPhysicalDeviceExternalBufferInfo * pExternalBufferInfo,
                                                      VkExternalBufferProperties * pExternalBufferProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalBufferProperties(
        physicalDevice, pExternalBufferInfo, pExternalBufferProperties );
    }

    void vkGetPhysicalDeviceExternalFenceProperties( VkPhysicalDevice                          physicalDevice,
                                                     const VkPhysicalDeviceExternalFenceInfo * pExternalFenceInfo,
                                                     VkExternalFenceProperties * pExternalFenceProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalFenceProperties(
        physicalDevice, pExternalFenceInfo, pExternalFenceProperties );
    }

    void vkGetPhysicalDeviceExternalSemaphoreProperties(
      VkPhysicalDevice                              physicalDevice,
      const VkPhysicalDeviceExternalSemaphoreInfo * pExternalSemaphoreInfo,
      VkExternalSemaphoreProperties *               pExternalSemaphoreProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalSemaphoreProperties(
        physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties );
    }

    void vkGetDescriptorSetLayoutSupport( VkDevice                                device,
                                          const VkDescriptorSetLayoutCreateInfo * pCreateInfo,
                                          VkDescriptorSetLayoutSupport *          pSupport ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDescriptorSetLayoutSupport( device, pCreateInfo, pSupport );
    }

    //=== VK_VERSION_1_2 ===

    void vkCmdDrawIndirectCount( VkCommandBuffer commandBuffer,
                                 VkBuffer        buffer,
                                 VkDeviceSize    offset,
                                 VkBuffer        countBuffer,
                                 VkDeviceSize    countBufferOffset,
                                 uint32_t        maxDrawCount,
                                 uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndirectCount(
        commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    void vkCmdDrawIndexedIndirectCount( VkCommandBuffer commandBuffer,
                                        VkBuffer        buffer,
                                        VkDeviceSize    offset,
                                        VkBuffer        countBuffer,
                                        VkDeviceSize    countBufferOffset,
                                        uint32_t        maxDrawCount,
                                        uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexedIndirectCount(
        commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    VkResult vkCreateRenderPass2( VkDevice                        device,
                                  const VkRenderPassCreateInfo2 * pCreateInfo,
                                  const VkAllocationCallbacks *   pAllocator,
                                  VkRenderPass *                  pRenderPass ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateRenderPass2( device, pCreateInfo, pAllocator, pRenderPass );
    }

    void vkCmdBeginRenderPass2( VkCommandBuffer               commandBuffer,
                                const VkRenderPassBeginInfo * pRenderPassBegin,
                                const VkSubpassBeginInfo *    pSubpassBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginRenderPass2( commandBuffer, pRenderPassBegin, pSubpassBeginInfo );
    }

    void vkCmdNextSubpass2( VkCommandBuffer            commandBuffer,
                            const VkSubpassBeginInfo * pSubpassBeginInfo,
                            const VkSubpassEndInfo *   pSubpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdNextSubpass2( commandBuffer, pSubpassBeginInfo, pSubpassEndInfo );
    }

    void vkCmdEndRenderPass2( VkCommandBuffer          commandBuffer,
                              const VkSubpassEndInfo * pSubpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndRenderPass2( commandBuffer, pSubpassEndInfo );
    }

    void vkResetQueryPool( VkDevice    device,
                           VkQueryPool queryPool,
                           uint32_t    firstQuery,
                           uint32_t    queryCount ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetQueryPool( device, queryPool, firstQuery, queryCount );
    }

    VkResult
      vkGetSemaphoreCounterValue( VkDevice device, VkSemaphore semaphore, uint64_t * pValue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreCounterValue( device, semaphore, pValue );
    }

    VkResult vkWaitSemaphores( VkDevice                    device,
                               const VkSemaphoreWaitInfo * pWaitInfo,
                               uint64_t                    timeout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWaitSemaphores( device, pWaitInfo, timeout );
    }

    VkResult vkSignalSemaphore( VkDevice device, const VkSemaphoreSignalInfo * pSignalInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSignalSemaphore( device, pSignalInfo );
    }

    VkDeviceAddress vkGetBufferDeviceAddress( VkDevice                          device,
                                              const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferDeviceAddress( device, pInfo );
    }

    uint64_t vkGetBufferOpaqueCaptureAddress( VkDevice                          device,
                                              const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferOpaqueCaptureAddress( device, pInfo );
    }

    uint64_t vkGetDeviceMemoryOpaqueCaptureAddress(
      VkDevice device, const VkDeviceMemoryOpaqueCaptureAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceMemoryOpaqueCaptureAddress( device, pInfo );
    }

    //=== VK_KHR_surface ===

    void vkDestroySurfaceKHR( VkInstance                    instance,
                              VkSurfaceKHR                  surface,
                              const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroySurfaceKHR( instance, surface, pAllocator );
    }

    VkResult vkGetPhysicalDeviceSurfaceSupportKHR( VkPhysicalDevice physicalDevice,
                                                   uint32_t         queueFamilyIndex,
                                                   VkSurfaceKHR     surface,
                                                   VkBool32 *       pSupported ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceSupportKHR( physicalDevice, queueFamilyIndex, surface, pSupported );
    }

    VkResult vkGetPhysicalDeviceSurfaceCapabilitiesKHR( VkPhysicalDevice           physicalDevice,
                                                        VkSurfaceKHR               surface,
                                                        VkSurfaceCapabilitiesKHR * pSurfaceCapabilities ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceCapabilitiesKHR( physicalDevice, surface, pSurfaceCapabilities );
    }

    VkResult vkGetPhysicalDeviceSurfaceFormatsKHR( VkPhysicalDevice     physicalDevice,
                                                   VkSurfaceKHR         surface,
                                                   uint32_t *           pSurfaceFormatCount,
                                                   VkSurfaceFormatKHR * pSurfaceFormats ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceFormatsKHR( physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats );
    }

    VkResult vkGetPhysicalDeviceSurfacePresentModesKHR( VkPhysicalDevice   physicalDevice,
                                                        VkSurfaceKHR       surface,
                                                        uint32_t *         pPresentModeCount,
                                                        VkPresentModeKHR * pPresentModes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfacePresentModesKHR( physicalDevice, surface, pPresentModeCount, pPresentModes );
    }

    //=== VK_KHR_swapchain ===

    VkResult vkCreateSwapchainKHR( VkDevice                         device,
                                   const VkSwapchainCreateInfoKHR * pCreateInfo,
                                   const VkAllocationCallbacks *    pAllocator,
                                   VkSwapchainKHR *                 pSwapchain ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateSwapchainKHR( device, pCreateInfo, pAllocator, pSwapchain );
    }

    void vkDestroySwapchainKHR( VkDevice                      device,
                                VkSwapchainKHR                swapchain,
                                const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroySwapchainKHR( device, swapchain, pAllocator );
    }

    VkResult vkGetSwapchainImagesKHR( VkDevice       device,
                                      VkSwapchainKHR swapchain,
                                      uint32_t *     pSwapchainImageCount,
                                      VkImage *      pSwapchainImages ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSwapchainImagesKHR( device, swapchain, pSwapchainImageCount, pSwapchainImages );
    }

    VkResult vkAcquireNextImageKHR( VkDevice       device,
                                    VkSwapchainKHR swapchain,
                                    uint64_t       timeout,
                                    VkSemaphore    semaphore,
                                    VkFence        fence,
                                    uint32_t *     pImageIndex ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireNextImageKHR( device, swapchain, timeout, semaphore, fence, pImageIndex );
    }

    VkResult vkQueuePresentKHR( VkQueue queue, const VkPresentInfoKHR * pPresentInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueuePresentKHR( queue, pPresentInfo );
    }

    VkResult vkGetDeviceGroupPresentCapabilitiesKHR(
      VkDevice device, VkDeviceGroupPresentCapabilitiesKHR * pDeviceGroupPresentCapabilities ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceGroupPresentCapabilitiesKHR( device, pDeviceGroupPresentCapabilities );
    }

    VkResult vkGetDeviceGroupSurfacePresentModesKHR(
      VkDevice device, VkSurfaceKHR surface, VkDeviceGroupPresentModeFlagsKHR * pModes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceGroupSurfacePresentModesKHR( device, surface, pModes );
    }

    VkResult vkGetPhysicalDevicePresentRectanglesKHR( VkPhysicalDevice physicalDevice,
                                                      VkSurfaceKHR     surface,
                                                      uint32_t *       pRectCount,
                                                      VkRect2D *       pRects ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDevicePresentRectanglesKHR( physicalDevice, surface, pRectCount, pRects );
    }

    VkResult vkAcquireNextImage2KHR( VkDevice                          device,
                                     const VkAcquireNextImageInfoKHR * pAcquireInfo,
                                     uint32_t *                        pImageIndex ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireNextImage2KHR( device, pAcquireInfo, pImageIndex );
    }

    //=== VK_KHR_display ===

    VkResult vkGetPhysicalDeviceDisplayPropertiesKHR( VkPhysicalDevice         physicalDevice,
                                                      uint32_t *               pPropertyCount,
                                                      VkDisplayPropertiesKHR * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceDisplayPropertiesKHR( physicalDevice, pPropertyCount, pProperties );
    }

    VkResult vkGetPhysicalDeviceDisplayPlanePropertiesKHR( VkPhysicalDevice              physicalDevice,
                                                           uint32_t *                    pPropertyCount,
                                                           VkDisplayPlanePropertiesKHR * pProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceDisplayPlanePropertiesKHR( physicalDevice, pPropertyCount, pProperties );
    }

    VkResult vkGetDisplayPlaneSupportedDisplaysKHR( VkPhysicalDevice physicalDevice,
                                                    uint32_t         planeIndex,
                                                    uint32_t *       pDisplayCount,
                                                    VkDisplayKHR *   pDisplays ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDisplayPlaneSupportedDisplaysKHR( physicalDevice, planeIndex, pDisplayCount, pDisplays );
    }

    VkResult vkGetDisplayModePropertiesKHR( VkPhysicalDevice             physicalDevice,
                                            VkDisplayKHR                 display,
                                            uint32_t *                   pPropertyCount,
                                            VkDisplayModePropertiesKHR * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDisplayModePropertiesKHR( physicalDevice, display, pPropertyCount, pProperties );
    }

    VkResult vkCreateDisplayModeKHR( VkPhysicalDevice                   physicalDevice,
                                     VkDisplayKHR                       display,
                                     const VkDisplayModeCreateInfoKHR * pCreateInfo,
                                     const VkAllocationCallbacks *      pAllocator,
                                     VkDisplayModeKHR *                 pMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDisplayModeKHR( physicalDevice, display, pCreateInfo, pAllocator, pMode );
    }

    VkResult vkGetDisplayPlaneCapabilitiesKHR( VkPhysicalDevice                physicalDevice,
                                               VkDisplayModeKHR                mode,
                                               uint32_t                        planeIndex,
                                               VkDisplayPlaneCapabilitiesKHR * pCapabilities ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDisplayPlaneCapabilitiesKHR( physicalDevice, mode, planeIndex, pCapabilities );
    }

    VkResult vkCreateDisplayPlaneSurfaceKHR( VkInstance                            instance,
                                             const VkDisplaySurfaceCreateInfoKHR * pCreateInfo,
                                             const VkAllocationCallbacks *         pAllocator,
                                             VkSurfaceKHR *                        pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDisplayPlaneSurfaceKHR( instance, pCreateInfo, pAllocator, pSurface );
    }

    //=== VK_KHR_display_swapchain ===

    VkResult vkCreateSharedSwapchainsKHR( VkDevice                         device,
                                          uint32_t                         swapchainCount,
                                          const VkSwapchainCreateInfoKHR * pCreateInfos,
                                          const VkAllocationCallbacks *    pAllocator,
                                          VkSwapchainKHR *                 pSwapchains ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateSharedSwapchainsKHR( device, swapchainCount, pCreateInfos, pAllocator, pSwapchains );
    }

#  if defined( VK_USE_PLATFORM_XLIB_KHR )
    //=== VK_KHR_xlib_surface ===

    VkResult vkCreateXlibSurfaceKHR( VkInstance                         instance,
                                     const VkXlibSurfaceCreateInfoKHR * pCreateInfo,
                                     const VkAllocationCallbacks *      pAllocator,
                                     VkSurfaceKHR *                     pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateXlibSurfaceKHR( instance, pCreateInfo, pAllocator, pSurface );
    }

    VkBool32 vkGetPhysicalDeviceXlibPresentationSupportKHR( VkPhysicalDevice physicalDevice,
                                                            uint32_t         queueFamilyIndex,
                                                            Display *        dpy,
                                                            VisualID         visualID ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceXlibPresentationSupportKHR( physicalDevice, queueFamilyIndex, dpy, visualID );
    }
#  endif /*VK_USE_PLATFORM_XLIB_KHR*/

#  if defined( VK_USE_PLATFORM_XCB_KHR )
    //=== VK_KHR_xcb_surface ===

    VkResult vkCreateXcbSurfaceKHR( VkInstance                        instance,
                                    const VkXcbSurfaceCreateInfoKHR * pCreateInfo,
                                    const VkAllocationCallbacks *     pAllocator,
                                    VkSurfaceKHR *                    pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateXcbSurfaceKHR( instance, pCreateInfo, pAllocator, pSurface );
    }

    VkBool32 vkGetPhysicalDeviceXcbPresentationSupportKHR( VkPhysicalDevice   physicalDevice,
                                                           uint32_t           queueFamilyIndex,
                                                           xcb_connection_t * connection,
                                                           xcb_visualid_t     visual_id ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceXcbPresentationSupportKHR( physicalDevice, queueFamilyIndex, connection, visual_id );
    }
#  endif /*VK_USE_PLATFORM_XCB_KHR*/

#  if defined( VK_USE_PLATFORM_WAYLAND_KHR )
    //=== VK_KHR_wayland_surface ===

    VkResult vkCreateWaylandSurfaceKHR( VkInstance                            instance,
                                        const VkWaylandSurfaceCreateInfoKHR * pCreateInfo,
                                        const VkAllocationCallbacks *         pAllocator,
                                        VkSurfaceKHR *                        pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateWaylandSurfaceKHR( instance, pCreateInfo, pAllocator, pSurface );
    }

    VkBool32 vkGetPhysicalDeviceWaylandPresentationSupportKHR( VkPhysicalDevice    physicalDevice,
                                                               uint32_t            queueFamilyIndex,
                                                               struct wl_display * display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceWaylandPresentationSupportKHR( physicalDevice, queueFamilyIndex, display );
    }
#  endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
    //=== VK_KHR_android_surface ===

    VkResult vkCreateAndroidSurfaceKHR( VkInstance                            instance,
                                        const VkAndroidSurfaceCreateInfoKHR * pCreateInfo,
                                        const VkAllocationCallbacks *         pAllocator,
                                        VkSurfaceKHR *                        pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateAndroidSurfaceKHR( instance, pCreateInfo, pAllocator, pSurface );
    }
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_win32_surface ===

    VkResult vkCreateWin32SurfaceKHR( VkInstance                          instance,
                                      const VkWin32SurfaceCreateInfoKHR * pCreateInfo,
                                      const VkAllocationCallbacks *       pAllocator,
                                      VkSurfaceKHR *                      pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateWin32SurfaceKHR( instance, pCreateInfo, pAllocator, pSurface );
    }

    VkBool32 vkGetPhysicalDeviceWin32PresentationSupportKHR( VkPhysicalDevice physicalDevice,
                                                             uint32_t queueFamilyIndex ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceWin32PresentationSupportKHR( physicalDevice, queueFamilyIndex );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_EXT_debug_report ===

    VkResult vkCreateDebugReportCallbackEXT( VkInstance                                 instance,
                                             const VkDebugReportCallbackCreateInfoEXT * pCreateInfo,
                                             const VkAllocationCallbacks *              pAllocator,
                                             VkDebugReportCallbackEXT * pCallback ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDebugReportCallbackEXT( instance, pCreateInfo, pAllocator, pCallback );
    }

    void vkDestroyDebugReportCallbackEXT( VkInstance                    instance,
                                          VkDebugReportCallbackEXT      callback,
                                          const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDebugReportCallbackEXT( instance, callback, pAllocator );
    }

    void vkDebugReportMessageEXT( VkInstance                 instance,
                                  VkDebugReportFlagsEXT      flags,
                                  VkDebugReportObjectTypeEXT objectType,
                                  uint64_t                   object,
                                  size_t                     location,
                                  int32_t                    messageCode,
                                  const char *               pLayerPrefix,
                                  const char *               pMessage ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDebugReportMessageEXT(
        instance, flags, objectType, object, location, messageCode, pLayerPrefix, pMessage );
    }

    //=== VK_EXT_debug_marker ===

    VkResult vkDebugMarkerSetObjectTagEXT( VkDevice                              device,
                                           const VkDebugMarkerObjectTagInfoEXT * pTagInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDebugMarkerSetObjectTagEXT( device, pTagInfo );
    }

    VkResult vkDebugMarkerSetObjectNameEXT( VkDevice                               device,
                                            const VkDebugMarkerObjectNameInfoEXT * pNameInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDebugMarkerSetObjectNameEXT( device, pNameInfo );
    }

    void vkCmdDebugMarkerBeginEXT( VkCommandBuffer                    commandBuffer,
                                   const VkDebugMarkerMarkerInfoEXT * pMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDebugMarkerBeginEXT( commandBuffer, pMarkerInfo );
    }

    void vkCmdDebugMarkerEndEXT( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDebugMarkerEndEXT( commandBuffer );
    }

    void vkCmdDebugMarkerInsertEXT( VkCommandBuffer                    commandBuffer,
                                    const VkDebugMarkerMarkerInfoEXT * pMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDebugMarkerInsertEXT( commandBuffer, pMarkerInfo );
    }

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_KHR_video_queue ===

    VkResult vkGetPhysicalDeviceVideoCapabilitiesKHR( VkPhysicalDevice          physicalDevice,
                                                      const VkVideoProfileKHR * pVideoProfile,
                                                      VkVideoCapabilitiesKHR * pCapabilities ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceVideoCapabilitiesKHR( physicalDevice, pVideoProfile, pCapabilities );
    }

    VkResult vkGetPhysicalDeviceVideoFormatPropertiesKHR( VkPhysicalDevice                           physicalDevice,
                                                          const VkPhysicalDeviceVideoFormatInfoKHR * pVideoFormatInfo,
                                                          uint32_t *                   pVideoFormatPropertyCount,
                                                          VkVideoFormatPropertiesKHR * pVideoFormatProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceVideoFormatPropertiesKHR(
        physicalDevice, pVideoFormatInfo, pVideoFormatPropertyCount, pVideoFormatProperties );
    }

    VkResult vkCreateVideoSessionKHR( VkDevice                            device,
                                      const VkVideoSessionCreateInfoKHR * pCreateInfo,
                                      const VkAllocationCallbacks *       pAllocator,
                                      VkVideoSessionKHR *                 pVideoSession ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateVideoSessionKHR( device, pCreateInfo, pAllocator, pVideoSession );
    }

    void vkDestroyVideoSessionKHR( VkDevice                      device,
                                   VkVideoSessionKHR             videoSession,
                                   const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyVideoSessionKHR( device, videoSession, pAllocator );
    }

    VkResult vkGetVideoSessionMemoryRequirementsKHR(
      VkDevice                        device,
      VkVideoSessionKHR               videoSession,
      uint32_t *                      pVideoSessionMemoryRequirementsCount,
      VkVideoGetMemoryPropertiesKHR * pVideoSessionMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetVideoSessionMemoryRequirementsKHR(
        device, videoSession, pVideoSessionMemoryRequirementsCount, pVideoSessionMemoryRequirements );
    }

    VkResult
      vkBindVideoSessionMemoryKHR( VkDevice                     device,
                                   VkVideoSessionKHR            videoSession,
                                   uint32_t                     videoSessionBindMemoryCount,
                                   const VkVideoBindMemoryKHR * pVideoSessionBindMemories ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindVideoSessionMemoryKHR(
        device, videoSession, videoSessionBindMemoryCount, pVideoSessionBindMemories );
    }

    VkResult vkCreateVideoSessionParametersKHR( VkDevice                                      device,
                                                const VkVideoSessionParametersCreateInfoKHR * pCreateInfo,
                                                const VkAllocationCallbacks *                 pAllocator,
                                                VkVideoSessionParametersKHR * pVideoSessionParameters ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateVideoSessionParametersKHR( device, pCreateInfo, pAllocator, pVideoSessionParameters );
    }

    VkResult vkUpdateVideoSessionParametersKHR( VkDevice                                      device,
                                                VkVideoSessionParametersKHR                   videoSessionParameters,
                                                const VkVideoSessionParametersUpdateInfoKHR * pUpdateInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkUpdateVideoSessionParametersKHR( device, videoSessionParameters, pUpdateInfo );
    }

    void vkDestroyVideoSessionParametersKHR( VkDevice                      device,
                                             VkVideoSessionParametersKHR   videoSessionParameters,
                                             const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyVideoSessionParametersKHR( device, videoSessionParameters, pAllocator );
    }

    void vkCmdBeginVideoCodingKHR( VkCommandBuffer                   commandBuffer,
                                   const VkVideoBeginCodingInfoKHR * pBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginVideoCodingKHR( commandBuffer, pBeginInfo );
    }

    void vkCmdEndVideoCodingKHR( VkCommandBuffer                 commandBuffer,
                                 const VkVideoEndCodingInfoKHR * pEndCodingInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndVideoCodingKHR( commandBuffer, pEndCodingInfo );
    }

    void vkCmdControlVideoCodingKHR( VkCommandBuffer                     commandBuffer,
                                     const VkVideoCodingControlInfoKHR * pCodingControlInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdControlVideoCodingKHR( commandBuffer, pCodingControlInfo );
    }
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_KHR_video_decode_queue ===

    void vkCmdDecodeVideoKHR( VkCommandBuffer              commandBuffer,
                              const VkVideoDecodeInfoKHR * pFrameInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDecodeVideoKHR( commandBuffer, pFrameInfo );
    }
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

    //=== VK_EXT_transform_feedback ===

    void vkCmdBindTransformFeedbackBuffersEXT( VkCommandBuffer      commandBuffer,
                                               uint32_t             firstBinding,
                                               uint32_t             bindingCount,
                                               const VkBuffer *     pBuffers,
                                               const VkDeviceSize * pOffsets,
                                               const VkDeviceSize * pSizes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindTransformFeedbackBuffersEXT(
        commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes );
    }

    void vkCmdBeginTransformFeedbackEXT( VkCommandBuffer      commandBuffer,
                                         uint32_t             firstCounterBuffer,
                                         uint32_t             counterBufferCount,
                                         const VkBuffer *     pCounterBuffers,
                                         const VkDeviceSize * pCounterBufferOffsets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginTransformFeedbackEXT(
        commandBuffer, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets );
    }

    void vkCmdEndTransformFeedbackEXT( VkCommandBuffer      commandBuffer,
                                       uint32_t             firstCounterBuffer,
                                       uint32_t             counterBufferCount,
                                       const VkBuffer *     pCounterBuffers,
                                       const VkDeviceSize * pCounterBufferOffsets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndTransformFeedbackEXT(
        commandBuffer, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets );
    }

    void vkCmdBeginQueryIndexedEXT( VkCommandBuffer     commandBuffer,
                                    VkQueryPool         queryPool,
                                    uint32_t            query,
                                    VkQueryControlFlags flags,
                                    uint32_t            index ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginQueryIndexedEXT( commandBuffer, queryPool, query, flags, index );
    }

    void vkCmdEndQueryIndexedEXT( VkCommandBuffer commandBuffer,
                                  VkQueryPool     queryPool,
                                  uint32_t        query,
                                  uint32_t        index ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndQueryIndexedEXT( commandBuffer, queryPool, query, index );
    }

    void vkCmdDrawIndirectByteCountEXT( VkCommandBuffer commandBuffer,
                                        uint32_t        instanceCount,
                                        uint32_t        firstInstance,
                                        VkBuffer        counterBuffer,
                                        VkDeviceSize    counterBufferOffset,
                                        uint32_t        counterOffset,
                                        uint32_t        vertexStride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndirectByteCountEXT(
        commandBuffer, instanceCount, firstInstance, counterBuffer, counterBufferOffset, counterOffset, vertexStride );
    }

    //=== VK_NVX_binary_import ===

    VkResult vkCreateCuModuleNVX( VkDevice                        device,
                                  const VkCuModuleCreateInfoNVX * pCreateInfo,
                                  const VkAllocationCallbacks *   pAllocator,
                                  VkCuModuleNVX *                 pModule ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateCuModuleNVX( device, pCreateInfo, pAllocator, pModule );
    }

    VkResult vkCreateCuFunctionNVX( VkDevice                          device,
                                    const VkCuFunctionCreateInfoNVX * pCreateInfo,
                                    const VkAllocationCallbacks *     pAllocator,
                                    VkCuFunctionNVX *                 pFunction ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateCuFunctionNVX( device, pCreateInfo, pAllocator, pFunction );
    }

    void vkDestroyCuModuleNVX( VkDevice                      device,
                               VkCuModuleNVX                 module,
                               const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyCuModuleNVX( device, module, pAllocator );
    }

    void vkDestroyCuFunctionNVX( VkDevice                      device,
                                 VkCuFunctionNVX               function,
                                 const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyCuFunctionNVX( device, function, pAllocator );
    }

    void vkCmdCuLaunchKernelNVX( VkCommandBuffer           commandBuffer,
                                 const VkCuLaunchInfoNVX * pLaunchInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCuLaunchKernelNVX( commandBuffer, pLaunchInfo );
    }

    //=== VK_NVX_image_view_handle ===

    uint32_t vkGetImageViewHandleNVX( VkDevice                         device,
                                      const VkImageViewHandleInfoNVX * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageViewHandleNVX( device, pInfo );
    }

    VkResult vkGetImageViewAddressNVX( VkDevice                          device,
                                       VkImageView                       imageView,
                                       VkImageViewAddressPropertiesNVX * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageViewAddressNVX( device, imageView, pProperties );
    }

    //=== VK_AMD_draw_indirect_count ===

    void vkCmdDrawIndirectCountAMD( VkCommandBuffer commandBuffer,
                                    VkBuffer        buffer,
                                    VkDeviceSize    offset,
                                    VkBuffer        countBuffer,
                                    VkDeviceSize    countBufferOffset,
                                    uint32_t        maxDrawCount,
                                    uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndirectCountAMD(
        commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    void vkCmdDrawIndexedIndirectCountAMD( VkCommandBuffer commandBuffer,
                                           VkBuffer        buffer,
                                           VkDeviceSize    offset,
                                           VkBuffer        countBuffer,
                                           VkDeviceSize    countBufferOffset,
                                           uint32_t        maxDrawCount,
                                           uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexedIndirectCountAMD(
        commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    //=== VK_AMD_shader_info ===

    VkResult vkGetShaderInfoAMD( VkDevice              device,
                                 VkPipeline            pipeline,
                                 VkShaderStageFlagBits shaderStage,
                                 VkShaderInfoTypeAMD   infoType,
                                 size_t *              pInfoSize,
                                 void *                pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetShaderInfoAMD( device, pipeline, shaderStage, infoType, pInfoSize, pInfo );
    }

#  if defined( VK_USE_PLATFORM_GGP )
    //=== VK_GGP_stream_descriptor_surface ===

    VkResult vkCreateStreamDescriptorSurfaceGGP( VkInstance                                     instance,
                                                 const VkStreamDescriptorSurfaceCreateInfoGGP * pCreateInfo,
                                                 const VkAllocationCallbacks *                  pAllocator,
                                                 VkSurfaceKHR * pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateStreamDescriptorSurfaceGGP( instance, pCreateInfo, pAllocator, pSurface );
    }
#  endif /*VK_USE_PLATFORM_GGP*/

    //=== VK_NV_external_memory_capabilities ===

    VkResult vkGetPhysicalDeviceExternalImageFormatPropertiesNV(
      VkPhysicalDevice                    physicalDevice,
      VkFormat                            format,
      VkImageType                         type,
      VkImageTiling                       tiling,
      VkImageUsageFlags                   usage,
      VkImageCreateFlags                  flags,
      VkExternalMemoryHandleTypeFlagsNV   externalHandleType,
      VkExternalImageFormatPropertiesNV * pExternalImageFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalImageFormatPropertiesNV(
        physicalDevice, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_NV_external_memory_win32 ===

    VkResult vkGetMemoryWin32HandleNV( VkDevice                          device,
                                       VkDeviceMemory                    memory,
                                       VkExternalMemoryHandleTypeFlagsNV handleType,
                                       HANDLE *                          pHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryWin32HandleNV( device, memory, handleType, pHandle );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_get_physical_device_properties2 ===

    void vkGetPhysicalDeviceFeatures2KHR( VkPhysicalDevice            physicalDevice,
                                          VkPhysicalDeviceFeatures2 * pFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFeatures2KHR( physicalDevice, pFeatures );
    }

    void vkGetPhysicalDeviceProperties2KHR( VkPhysicalDevice              physicalDevice,
                                            VkPhysicalDeviceProperties2 * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceProperties2KHR( physicalDevice, pProperties );
    }

    void vkGetPhysicalDeviceFormatProperties2KHR( VkPhysicalDevice      physicalDevice,
                                                  VkFormat              format,
                                                  VkFormatProperties2 * pFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFormatProperties2KHR( physicalDevice, format, pFormatProperties );
    }

    VkResult vkGetPhysicalDeviceImageFormatProperties2KHR( VkPhysicalDevice                         physicalDevice,
                                                           const VkPhysicalDeviceImageFormatInfo2 * pImageFormatInfo,
                                                           VkImageFormatProperties2 * pImageFormatProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceImageFormatProperties2KHR( physicalDevice, pImageFormatInfo, pImageFormatProperties );
    }

    void vkGetPhysicalDeviceQueueFamilyProperties2KHR( VkPhysicalDevice           physicalDevice,
                                                       uint32_t *                 pQueueFamilyPropertyCount,
                                                       VkQueueFamilyProperties2 * pQueueFamilyProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceQueueFamilyProperties2KHR(
        physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties );
    }

    void vkGetPhysicalDeviceMemoryProperties2KHR(
      VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2 * pMemoryProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceMemoryProperties2KHR( physicalDevice, pMemoryProperties );
    }

    void vkGetPhysicalDeviceSparseImageFormatProperties2KHR( VkPhysicalDevice physicalDevice,
                                                             const VkPhysicalDeviceSparseImageFormatInfo2 * pFormatInfo,
                                                             uint32_t *                       pPropertyCount,
                                                             VkSparseImageFormatProperties2 * pProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSparseImageFormatProperties2KHR(
        physicalDevice, pFormatInfo, pPropertyCount, pProperties );
    }

    //=== VK_KHR_device_group ===

    void
      vkGetDeviceGroupPeerMemoryFeaturesKHR( VkDevice                   device,
                                             uint32_t                   heapIndex,
                                             uint32_t                   localDeviceIndex,
                                             uint32_t                   remoteDeviceIndex,
                                             VkPeerMemoryFeatureFlags * pPeerMemoryFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceGroupPeerMemoryFeaturesKHR(
        device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures );
    }

    void vkCmdSetDeviceMaskKHR( VkCommandBuffer commandBuffer, uint32_t deviceMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDeviceMaskKHR( commandBuffer, deviceMask );
    }

    void vkCmdDispatchBaseKHR( VkCommandBuffer commandBuffer,
                               uint32_t        baseGroupX,
                               uint32_t        baseGroupY,
                               uint32_t        baseGroupZ,
                               uint32_t        groupCountX,
                               uint32_t        groupCountY,
                               uint32_t        groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatchBaseKHR(
        commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ );
    }

#  if defined( VK_USE_PLATFORM_VI_NN )
    //=== VK_NN_vi_surface ===

    VkResult vkCreateViSurfaceNN( VkInstance                      instance,
                                  const VkViSurfaceCreateInfoNN * pCreateInfo,
                                  const VkAllocationCallbacks *   pAllocator,
                                  VkSurfaceKHR *                  pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateViSurfaceNN( instance, pCreateInfo, pAllocator, pSurface );
    }
#  endif /*VK_USE_PLATFORM_VI_NN*/

    //=== VK_KHR_maintenance1 ===

    void vkTrimCommandPoolKHR( VkDevice               device,
                               VkCommandPool          commandPool,
                               VkCommandPoolTrimFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkTrimCommandPoolKHR( device, commandPool, flags );
    }

    //=== VK_KHR_device_group_creation ===

    VkResult vkEnumeratePhysicalDeviceGroupsKHR(
      VkInstance                        instance,
      uint32_t *                        pPhysicalDeviceGroupCount,
      VkPhysicalDeviceGroupProperties * pPhysicalDeviceGroupProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumeratePhysicalDeviceGroupsKHR(
        instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties );
    }

    //=== VK_KHR_external_memory_capabilities ===

    void vkGetPhysicalDeviceExternalBufferPropertiesKHR( VkPhysicalDevice                           physicalDevice,
                                                         const VkPhysicalDeviceExternalBufferInfo * pExternalBufferInfo,
                                                         VkExternalBufferProperties * pExternalBufferProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalBufferPropertiesKHR(
        physicalDevice, pExternalBufferInfo, pExternalBufferProperties );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_memory_win32 ===

    VkResult vkGetMemoryWin32HandleKHR( VkDevice                              device,
                                        const VkMemoryGetWin32HandleInfoKHR * pGetWin32HandleInfo,
                                        HANDLE *                              pHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryWin32HandleKHR( device, pGetWin32HandleInfo, pHandle );
    }

    VkResult vkGetMemoryWin32HandlePropertiesKHR(
      VkDevice                           device,
      VkExternalMemoryHandleTypeFlagBits handleType,
      HANDLE                             handle,
      VkMemoryWin32HandlePropertiesKHR * pMemoryWin32HandleProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryWin32HandlePropertiesKHR( device, handleType, handle, pMemoryWin32HandleProperties );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_memory_fd ===

    VkResult
      vkGetMemoryFdKHR( VkDevice device, const VkMemoryGetFdInfoKHR * pGetFdInfo, int * pFd ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryFdKHR( device, pGetFdInfo, pFd );
    }

    VkResult vkGetMemoryFdPropertiesKHR( VkDevice                           device,
                                         VkExternalMemoryHandleTypeFlagBits handleType,
                                         int                                fd,
                                         VkMemoryFdPropertiesKHR * pMemoryFdProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryFdPropertiesKHR( device, handleType, fd, pMemoryFdProperties );
    }

    //=== VK_KHR_external_semaphore_capabilities ===

    void vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(
      VkPhysicalDevice                              physicalDevice,
      const VkPhysicalDeviceExternalSemaphoreInfo * pExternalSemaphoreInfo,
      VkExternalSemaphoreProperties *               pExternalSemaphoreProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(
        physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_semaphore_win32 ===

    VkResult vkImportSemaphoreWin32HandleKHR(
      VkDevice                                    device,
      const VkImportSemaphoreWin32HandleInfoKHR * pImportSemaphoreWin32HandleInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportSemaphoreWin32HandleKHR( device, pImportSemaphoreWin32HandleInfo );
    }

    VkResult vkGetSemaphoreWin32HandleKHR( VkDevice                                 device,
                                           const VkSemaphoreGetWin32HandleInfoKHR * pGetWin32HandleInfo,
                                           HANDLE *                                 pHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreWin32HandleKHR( device, pGetWin32HandleInfo, pHandle );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_semaphore_fd ===

    VkResult
      vkImportSemaphoreFdKHR( VkDevice                           device,
                              const VkImportSemaphoreFdInfoKHR * pImportSemaphoreFdInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportSemaphoreFdKHR( device, pImportSemaphoreFdInfo );
    }

    VkResult vkGetSemaphoreFdKHR( VkDevice                        device,
                                  const VkSemaphoreGetFdInfoKHR * pGetFdInfo,
                                  int *                           pFd ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreFdKHR( device, pGetFdInfo, pFd );
    }

    //=== VK_KHR_push_descriptor ===

    void vkCmdPushDescriptorSetKHR( VkCommandBuffer              commandBuffer,
                                    VkPipelineBindPoint          pipelineBindPoint,
                                    VkPipelineLayout             layout,
                                    uint32_t                     set,
                                    uint32_t                     descriptorWriteCount,
                                    const VkWriteDescriptorSet * pDescriptorWrites ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPushDescriptorSetKHR(
        commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites );
    }

    void vkCmdPushDescriptorSetWithTemplateKHR( VkCommandBuffer            commandBuffer,
                                                VkDescriptorUpdateTemplate descriptorUpdateTemplate,
                                                VkPipelineLayout           layout,
                                                uint32_t                   set,
                                                const void *               pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPushDescriptorSetWithTemplateKHR( commandBuffer, descriptorUpdateTemplate, layout, set, pData );
    }

    //=== VK_EXT_conditional_rendering ===

    void vkCmdBeginConditionalRenderingEXT(
      VkCommandBuffer                            commandBuffer,
      const VkConditionalRenderingBeginInfoEXT * pConditionalRenderingBegin ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginConditionalRenderingEXT( commandBuffer, pConditionalRenderingBegin );
    }

    void vkCmdEndConditionalRenderingEXT( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndConditionalRenderingEXT( commandBuffer );
    }

    //=== VK_KHR_descriptor_update_template ===

    VkResult vkCreateDescriptorUpdateTemplateKHR( VkDevice                                     device,
                                                  const VkDescriptorUpdateTemplateCreateInfo * pCreateInfo,
                                                  const VkAllocationCallbacks *                pAllocator,
                                                  VkDescriptorUpdateTemplate * pDescriptorUpdateTemplate ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDescriptorUpdateTemplateKHR( device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate );
    }

    void vkDestroyDescriptorUpdateTemplateKHR( VkDevice                      device,
                                               VkDescriptorUpdateTemplate    descriptorUpdateTemplate,
                                               const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDescriptorUpdateTemplateKHR( device, descriptorUpdateTemplate, pAllocator );
    }

    void vkUpdateDescriptorSetWithTemplateKHR( VkDevice                   device,
                                               VkDescriptorSet            descriptorSet,
                                               VkDescriptorUpdateTemplate descriptorUpdateTemplate,
                                               const void *               pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUpdateDescriptorSetWithTemplateKHR( device, descriptorSet, descriptorUpdateTemplate, pData );
    }

    //=== VK_NV_clip_space_w_scaling ===

    void vkCmdSetViewportWScalingNV( VkCommandBuffer              commandBuffer,
                                     uint32_t                     firstViewport,
                                     uint32_t                     viewportCount,
                                     const VkViewportWScalingNV * pViewportWScalings ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewportWScalingNV( commandBuffer, firstViewport, viewportCount, pViewportWScalings );
    }

    //=== VK_EXT_direct_mode_display ===

    VkResult vkReleaseDisplayEXT( VkPhysicalDevice physicalDevice, VkDisplayKHR display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkReleaseDisplayEXT( physicalDevice, display );
    }

#  if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
    //=== VK_EXT_acquire_xlib_display ===

    VkResult vkAcquireXlibDisplayEXT( VkPhysicalDevice physicalDevice,
                                      Display *        dpy,
                                      VkDisplayKHR     display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireXlibDisplayEXT( physicalDevice, dpy, display );
    }

    VkResult vkGetRandROutputDisplayEXT( VkPhysicalDevice physicalDevice,
                                         Display *        dpy,
                                         RROutput         rrOutput,
                                         VkDisplayKHR *   pDisplay ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRandROutputDisplayEXT( physicalDevice, dpy, rrOutput, pDisplay );
    }
#  endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

    //=== VK_EXT_display_surface_counter ===

    VkResult vkGetPhysicalDeviceSurfaceCapabilities2EXT( VkPhysicalDevice            physicalDevice,
                                                         VkSurfaceKHR                surface,
                                                         VkSurfaceCapabilities2EXT * pSurfaceCapabilities ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceCapabilities2EXT( physicalDevice, surface, pSurfaceCapabilities );
    }

    //=== VK_EXT_display_control ===

    VkResult vkDisplayPowerControlEXT( VkDevice                      device,
                                       VkDisplayKHR                  display,
                                       const VkDisplayPowerInfoEXT * pDisplayPowerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDisplayPowerControlEXT( device, display, pDisplayPowerInfo );
    }

    VkResult vkRegisterDeviceEventEXT( VkDevice                      device,
                                       const VkDeviceEventInfoEXT *  pDeviceEventInfo,
                                       const VkAllocationCallbacks * pAllocator,
                                       VkFence *                     pFence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkRegisterDeviceEventEXT( device, pDeviceEventInfo, pAllocator, pFence );
    }

    VkResult vkRegisterDisplayEventEXT( VkDevice                      device,
                                        VkDisplayKHR                  display,
                                        const VkDisplayEventInfoEXT * pDisplayEventInfo,
                                        const VkAllocationCallbacks * pAllocator,
                                        VkFence *                     pFence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkRegisterDisplayEventEXT( device, display, pDisplayEventInfo, pAllocator, pFence );
    }

    VkResult vkGetSwapchainCounterEXT( VkDevice                    device,
                                       VkSwapchainKHR              swapchain,
                                       VkSurfaceCounterFlagBitsEXT counter,
                                       uint64_t *                  pCounterValue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSwapchainCounterEXT( device, swapchain, counter, pCounterValue );
    }

    //=== VK_GOOGLE_display_timing ===

    VkResult vkGetRefreshCycleDurationGOOGLE( VkDevice                       device,
                                              VkSwapchainKHR                 swapchain,
                                              VkRefreshCycleDurationGOOGLE * pDisplayTimingProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRefreshCycleDurationGOOGLE( device, swapchain, pDisplayTimingProperties );
    }

    VkResult vkGetPastPresentationTimingGOOGLE( VkDevice                         device,
                                                VkSwapchainKHR                   swapchain,
                                                uint32_t *                       pPresentationTimingCount,
                                                VkPastPresentationTimingGOOGLE * pPresentationTimings ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPastPresentationTimingGOOGLE( device, swapchain, pPresentationTimingCount, pPresentationTimings );
    }

    //=== VK_EXT_discard_rectangles ===

    void vkCmdSetDiscardRectangleEXT( VkCommandBuffer  commandBuffer,
                                      uint32_t         firstDiscardRectangle,
                                      uint32_t         discardRectangleCount,
                                      const VkRect2D * pDiscardRectangles ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDiscardRectangleEXT(
        commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles );
    }

    //=== VK_EXT_hdr_metadata ===

    void vkSetHdrMetadataEXT( VkDevice                 device,
                              uint32_t                 swapchainCount,
                              const VkSwapchainKHR *   pSwapchains,
                              const VkHdrMetadataEXT * pMetadata ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetHdrMetadataEXT( device, swapchainCount, pSwapchains, pMetadata );
    }

    //=== VK_KHR_create_renderpass2 ===

    VkResult vkCreateRenderPass2KHR( VkDevice                        device,
                                     const VkRenderPassCreateInfo2 * pCreateInfo,
                                     const VkAllocationCallbacks *   pAllocator,
                                     VkRenderPass *                  pRenderPass ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateRenderPass2KHR( device, pCreateInfo, pAllocator, pRenderPass );
    }

    void vkCmdBeginRenderPass2KHR( VkCommandBuffer               commandBuffer,
                                   const VkRenderPassBeginInfo * pRenderPassBegin,
                                   const VkSubpassBeginInfo *    pSubpassBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginRenderPass2KHR( commandBuffer, pRenderPassBegin, pSubpassBeginInfo );
    }

    void vkCmdNextSubpass2KHR( VkCommandBuffer            commandBuffer,
                               const VkSubpassBeginInfo * pSubpassBeginInfo,
                               const VkSubpassEndInfo *   pSubpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdNextSubpass2KHR( commandBuffer, pSubpassBeginInfo, pSubpassEndInfo );
    }

    void vkCmdEndRenderPass2KHR( VkCommandBuffer          commandBuffer,
                                 const VkSubpassEndInfo * pSubpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndRenderPass2KHR( commandBuffer, pSubpassEndInfo );
    }

    //=== VK_KHR_shared_presentable_image ===

    VkResult vkGetSwapchainStatusKHR( VkDevice device, VkSwapchainKHR swapchain ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSwapchainStatusKHR( device, swapchain );
    }

    //=== VK_KHR_external_fence_capabilities ===

    void vkGetPhysicalDeviceExternalFencePropertiesKHR( VkPhysicalDevice                          physicalDevice,
                                                        const VkPhysicalDeviceExternalFenceInfo * pExternalFenceInfo,
                                                        VkExternalFenceProperties * pExternalFenceProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalFencePropertiesKHR(
        physicalDevice, pExternalFenceInfo, pExternalFenceProperties );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_fence_win32 ===

    VkResult vkImportFenceWin32HandleKHR(
      VkDevice device, const VkImportFenceWin32HandleInfoKHR * pImportFenceWin32HandleInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportFenceWin32HandleKHR( device, pImportFenceWin32HandleInfo );
    }

    VkResult vkGetFenceWin32HandleKHR( VkDevice                             device,
                                       const VkFenceGetWin32HandleInfoKHR * pGetWin32HandleInfo,
                                       HANDLE *                             pHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetFenceWin32HandleKHR( device, pGetWin32HandleInfo, pHandle );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_fence_fd ===

    VkResult vkImportFenceFdKHR( VkDevice                       device,
                                 const VkImportFenceFdInfoKHR * pImportFenceFdInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportFenceFdKHR( device, pImportFenceFdInfo );
    }

    VkResult
      vkGetFenceFdKHR( VkDevice device, const VkFenceGetFdInfoKHR * pGetFdInfo, int * pFd ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetFenceFdKHR( device, pGetFdInfo, pFd );
    }

    //=== VK_KHR_performance_query ===

    VkResult vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
      VkPhysicalDevice                     physicalDevice,
      uint32_t                             queueFamilyIndex,
      uint32_t *                           pCounterCount,
      VkPerformanceCounterKHR *            pCounters,
      VkPerformanceCounterDescriptionKHR * pCounterDescriptions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
        physicalDevice, queueFamilyIndex, pCounterCount, pCounters, pCounterDescriptions );
    }

    void vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
      VkPhysicalDevice                            physicalDevice,
      const VkQueryPoolPerformanceCreateInfoKHR * pPerformanceQueryCreateInfo,
      uint32_t *                                  pNumPasses ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
        physicalDevice, pPerformanceQueryCreateInfo, pNumPasses );
    }

    VkResult vkAcquireProfilingLockKHR( VkDevice                              device,
                                        const VkAcquireProfilingLockInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireProfilingLockKHR( device, pInfo );
    }

    void vkReleaseProfilingLockKHR( VkDevice device ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkReleaseProfilingLockKHR( device );
    }

    //=== VK_KHR_get_surface_capabilities2 ===

    VkResult vkGetPhysicalDeviceSurfaceCapabilities2KHR( VkPhysicalDevice                        physicalDevice,
                                                         const VkPhysicalDeviceSurfaceInfo2KHR * pSurfaceInfo,
                                                         VkSurfaceCapabilities2KHR * pSurfaceCapabilities ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceCapabilities2KHR( physicalDevice, pSurfaceInfo, pSurfaceCapabilities );
    }

    VkResult vkGetPhysicalDeviceSurfaceFormats2KHR( VkPhysicalDevice                        physicalDevice,
                                                    const VkPhysicalDeviceSurfaceInfo2KHR * pSurfaceInfo,
                                                    uint32_t *                              pSurfaceFormatCount,
                                                    VkSurfaceFormat2KHR * pSurfaceFormats ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceFormats2KHR(
        physicalDevice, pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats );
    }

    //=== VK_KHR_get_display_properties2 ===

    VkResult vkGetPhysicalDeviceDisplayProperties2KHR( VkPhysicalDevice          physicalDevice,
                                                       uint32_t *                pPropertyCount,
                                                       VkDisplayProperties2KHR * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceDisplayProperties2KHR( physicalDevice, pPropertyCount, pProperties );
    }

    VkResult vkGetPhysicalDeviceDisplayPlaneProperties2KHR( VkPhysicalDevice               physicalDevice,
                                                            uint32_t *                     pPropertyCount,
                                                            VkDisplayPlaneProperties2KHR * pProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceDisplayPlaneProperties2KHR( physicalDevice, pPropertyCount, pProperties );
    }

    VkResult vkGetDisplayModeProperties2KHR( VkPhysicalDevice              physicalDevice,
                                             VkDisplayKHR                  display,
                                             uint32_t *                    pPropertyCount,
                                             VkDisplayModeProperties2KHR * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDisplayModeProperties2KHR( physicalDevice, display, pPropertyCount, pProperties );
    }

    VkResult
      vkGetDisplayPlaneCapabilities2KHR( VkPhysicalDevice                 physicalDevice,
                                         const VkDisplayPlaneInfo2KHR *   pDisplayPlaneInfo,
                                         VkDisplayPlaneCapabilities2KHR * pCapabilities ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDisplayPlaneCapabilities2KHR( physicalDevice, pDisplayPlaneInfo, pCapabilities );
    }

#  if defined( VK_USE_PLATFORM_IOS_MVK )
    //=== VK_MVK_ios_surface ===

    VkResult vkCreateIOSSurfaceMVK( VkInstance                        instance,
                                    const VkIOSSurfaceCreateInfoMVK * pCreateInfo,
                                    const VkAllocationCallbacks *     pAllocator,
                                    VkSurfaceKHR *                    pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateIOSSurfaceMVK( instance, pCreateInfo, pAllocator, pSurface );
    }
#  endif /*VK_USE_PLATFORM_IOS_MVK*/

#  if defined( VK_USE_PLATFORM_MACOS_MVK )
    //=== VK_MVK_macos_surface ===

    VkResult vkCreateMacOSSurfaceMVK( VkInstance                          instance,
                                      const VkMacOSSurfaceCreateInfoMVK * pCreateInfo,
                                      const VkAllocationCallbacks *       pAllocator,
                                      VkSurfaceKHR *                      pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateMacOSSurfaceMVK( instance, pCreateInfo, pAllocator, pSurface );
    }
#  endif /*VK_USE_PLATFORM_MACOS_MVK*/

    //=== VK_EXT_debug_utils ===

    VkResult vkSetDebugUtilsObjectNameEXT( VkDevice                              device,
                                           const VkDebugUtilsObjectNameInfoEXT * pNameInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetDebugUtilsObjectNameEXT( device, pNameInfo );
    }

    VkResult vkSetDebugUtilsObjectTagEXT( VkDevice                             device,
                                          const VkDebugUtilsObjectTagInfoEXT * pTagInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetDebugUtilsObjectTagEXT( device, pTagInfo );
    }

    void vkQueueBeginDebugUtilsLabelEXT( VkQueue                      queue,
                                         const VkDebugUtilsLabelEXT * pLabelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueBeginDebugUtilsLabelEXT( queue, pLabelInfo );
    }

    void vkQueueEndDebugUtilsLabelEXT( VkQueue queue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueEndDebugUtilsLabelEXT( queue );
    }

    void vkQueueInsertDebugUtilsLabelEXT( VkQueue                      queue,
                                          const VkDebugUtilsLabelEXT * pLabelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueInsertDebugUtilsLabelEXT( queue, pLabelInfo );
    }

    void vkCmdBeginDebugUtilsLabelEXT( VkCommandBuffer              commandBuffer,
                                       const VkDebugUtilsLabelEXT * pLabelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginDebugUtilsLabelEXT( commandBuffer, pLabelInfo );
    }

    void vkCmdEndDebugUtilsLabelEXT( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndDebugUtilsLabelEXT( commandBuffer );
    }

    void vkCmdInsertDebugUtilsLabelEXT( VkCommandBuffer              commandBuffer,
                                        const VkDebugUtilsLabelEXT * pLabelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdInsertDebugUtilsLabelEXT( commandBuffer, pLabelInfo );
    }

    VkResult vkCreateDebugUtilsMessengerEXT( VkInstance                                 instance,
                                             const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
                                             const VkAllocationCallbacks *              pAllocator,
                                             VkDebugUtilsMessengerEXT * pMessenger ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDebugUtilsMessengerEXT( instance, pCreateInfo, pAllocator, pMessenger );
    }

    void vkDestroyDebugUtilsMessengerEXT( VkInstance                    instance,
                                          VkDebugUtilsMessengerEXT      messenger,
                                          const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDebugUtilsMessengerEXT( instance, messenger, pAllocator );
    }

    void vkSubmitDebugUtilsMessageEXT( VkInstance                                   instance,
                                       VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                       VkDebugUtilsMessageTypeFlagsEXT              messageTypes,
                                       const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkSubmitDebugUtilsMessageEXT( instance, messageSeverity, messageTypes, pCallbackData );
    }

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
    //=== VK_ANDROID_external_memory_android_hardware_buffer ===

    VkResult vkGetAndroidHardwareBufferPropertiesANDROID( VkDevice                                   device,
                                                          const struct AHardwareBuffer *             buffer,
                                                          VkAndroidHardwareBufferPropertiesANDROID * pProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAndroidHardwareBufferPropertiesANDROID( device, buffer, pProperties );
    }

    VkResult vkGetMemoryAndroidHardwareBufferANDROID( VkDevice                                            device,
                                                      const VkMemoryGetAndroidHardwareBufferInfoANDROID * pInfo,
                                                      struct AHardwareBuffer ** pBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryAndroidHardwareBufferANDROID( device, pInfo, pBuffer );
    }
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

    //=== VK_EXT_sample_locations ===

    void vkCmdSetSampleLocationsEXT( VkCommandBuffer                  commandBuffer,
                                     const VkSampleLocationsInfoEXT * pSampleLocationsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetSampleLocationsEXT( commandBuffer, pSampleLocationsInfo );
    }

    void vkGetPhysicalDeviceMultisamplePropertiesEXT( VkPhysicalDevice             physicalDevice,
                                                      VkSampleCountFlagBits        samples,
                                                      VkMultisamplePropertiesEXT * pMultisampleProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceMultisamplePropertiesEXT( physicalDevice, samples, pMultisampleProperties );
    }

    //=== VK_KHR_get_memory_requirements2 ===

    void vkGetImageMemoryRequirements2KHR( VkDevice                               device,
                                           const VkImageMemoryRequirementsInfo2 * pInfo,
                                           VkMemoryRequirements2 * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageMemoryRequirements2KHR( device, pInfo, pMemoryRequirements );
    }

    void vkGetBufferMemoryRequirements2KHR( VkDevice                                device,
                                            const VkBufferMemoryRequirementsInfo2 * pInfo,
                                            VkMemoryRequirements2 * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferMemoryRequirements2KHR( device, pInfo, pMemoryRequirements );
    }

    void vkGetImageSparseMemoryRequirements2KHR( VkDevice                                     device,
                                                 const VkImageSparseMemoryRequirementsInfo2 * pInfo,
                                                 uint32_t *                         pSparseMemoryRequirementCount,
                                                 VkSparseImageMemoryRequirements2 * pSparseMemoryRequirements ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSparseMemoryRequirements2KHR(
        device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements );
    }

    //=== VK_KHR_acceleration_structure ===

    VkResult
      vkCreateAccelerationStructureKHR( VkDevice                                     device,
                                        const VkAccelerationStructureCreateInfoKHR * pCreateInfo,
                                        const VkAllocationCallbacks *                pAllocator,
                                        VkAccelerationStructureKHR * pAccelerationStructure ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateAccelerationStructureKHR( device, pCreateInfo, pAllocator, pAccelerationStructure );
    }

    void vkDestroyAccelerationStructureKHR( VkDevice                      device,
                                            VkAccelerationStructureKHR    accelerationStructure,
                                            const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyAccelerationStructureKHR( device, accelerationStructure, pAllocator );
    }

    void vkCmdBuildAccelerationStructuresKHR(
      VkCommandBuffer                                          commandBuffer,
      uint32_t                                                 infoCount,
      const VkAccelerationStructureBuildGeometryInfoKHR *      pInfos,
      const VkAccelerationStructureBuildRangeInfoKHR * const * ppBuildRangeInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBuildAccelerationStructuresKHR( commandBuffer, infoCount, pInfos, ppBuildRangeInfos );
    }

    void vkCmdBuildAccelerationStructuresIndirectKHR( VkCommandBuffer                                     commandBuffer,
                                                      uint32_t                                            infoCount,
                                                      const VkAccelerationStructureBuildGeometryInfoKHR * pInfos,
                                                      const VkDeviceAddress *  pIndirectDeviceAddresses,
                                                      const uint32_t *         pIndirectStrides,
                                                      const uint32_t * const * ppMaxPrimitiveCounts ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBuildAccelerationStructuresIndirectKHR(
        commandBuffer, infoCount, pInfos, pIndirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts );
    }

    VkResult vkBuildAccelerationStructuresKHR(
      VkDevice                                                 device,
      VkDeferredOperationKHR                                   deferredOperation,
      uint32_t                                                 infoCount,
      const VkAccelerationStructureBuildGeometryInfoKHR *      pInfos,
      const VkAccelerationStructureBuildRangeInfoKHR * const * ppBuildRangeInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBuildAccelerationStructuresKHR( device, deferredOperation, infoCount, pInfos, ppBuildRangeInfos );
    }

    VkResult
      vkCopyAccelerationStructureKHR( VkDevice                                   device,
                                      VkDeferredOperationKHR                     deferredOperation,
                                      const VkCopyAccelerationStructureInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyAccelerationStructureKHR( device, deferredOperation, pInfo );
    }

    VkResult vkCopyAccelerationStructureToMemoryKHR( VkDevice               device,
                                                     VkDeferredOperationKHR deferredOperation,
                                                     const VkCopyAccelerationStructureToMemoryInfoKHR * pInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyAccelerationStructureToMemoryKHR( device, deferredOperation, pInfo );
    }

    VkResult vkCopyMemoryToAccelerationStructureKHR( VkDevice               device,
                                                     VkDeferredOperationKHR deferredOperation,
                                                     const VkCopyMemoryToAccelerationStructureInfoKHR * pInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyMemoryToAccelerationStructureKHR( device, deferredOperation, pInfo );
    }

    VkResult vkWriteAccelerationStructuresPropertiesKHR( VkDevice                           device,
                                                         uint32_t                           accelerationStructureCount,
                                                         const VkAccelerationStructureKHR * pAccelerationStructures,
                                                         VkQueryType                        queryType,
                                                         size_t                             dataSize,
                                                         void *                             pData,
                                                         size_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWriteAccelerationStructuresPropertiesKHR(
        device, accelerationStructureCount, pAccelerationStructures, queryType, dataSize, pData, stride );
    }

    void vkCmdCopyAccelerationStructureKHR( VkCommandBuffer                            commandBuffer,
                                            const VkCopyAccelerationStructureInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyAccelerationStructureKHR( commandBuffer, pInfo );
    }

    void vkCmdCopyAccelerationStructureToMemoryKHR( VkCommandBuffer                                    commandBuffer,
                                                    const VkCopyAccelerationStructureToMemoryInfoKHR * pInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyAccelerationStructureToMemoryKHR( commandBuffer, pInfo );
    }

    void vkCmdCopyMemoryToAccelerationStructureKHR( VkCommandBuffer                                    commandBuffer,
                                                    const VkCopyMemoryToAccelerationStructureInfoKHR * pInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyMemoryToAccelerationStructureKHR( commandBuffer, pInfo );
    }

    VkDeviceAddress vkGetAccelerationStructureDeviceAddressKHR(
      VkDevice device, const VkAccelerationStructureDeviceAddressInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAccelerationStructureDeviceAddressKHR( device, pInfo );
    }

    void vkCmdWriteAccelerationStructuresPropertiesKHR( VkCommandBuffer                    commandBuffer,
                                                        uint32_t                           accelerationStructureCount,
                                                        const VkAccelerationStructureKHR * pAccelerationStructures,
                                                        VkQueryType                        queryType,
                                                        VkQueryPool                        queryPool,
                                                        uint32_t firstQuery ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteAccelerationStructuresPropertiesKHR(
        commandBuffer, accelerationStructureCount, pAccelerationStructures, queryType, queryPool, firstQuery );
    }

    void vkGetDeviceAccelerationStructureCompatibilityKHR(
      VkDevice                                      device,
      const VkAccelerationStructureVersionInfoKHR * pVersionInfo,
      VkAccelerationStructureCompatibilityKHR *     pCompatibility ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceAccelerationStructureCompatibilityKHR( device, pVersionInfo, pCompatibility );
    }

    void vkGetAccelerationStructureBuildSizesKHR( VkDevice                                            device,
                                                  VkAccelerationStructureBuildTypeKHR                 buildType,
                                                  const VkAccelerationStructureBuildGeometryInfoKHR * pBuildInfo,
                                                  const uint32_t *                           pMaxPrimitiveCounts,
                                                  VkAccelerationStructureBuildSizesInfoKHR * pSizeInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAccelerationStructureBuildSizesKHR( device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo );
    }

    //=== VK_KHR_sampler_ycbcr_conversion ===

    VkResult vkCreateSamplerYcbcrConversionKHR( VkDevice                                   device,
                                                const VkSamplerYcbcrConversionCreateInfo * pCreateInfo,
                                                const VkAllocationCallbacks *              pAllocator,
                                                VkSamplerYcbcrConversion * pYcbcrConversion ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateSamplerYcbcrConversionKHR( device, pCreateInfo, pAllocator, pYcbcrConversion );
    }

    void vkDestroySamplerYcbcrConversionKHR( VkDevice                      device,
                                             VkSamplerYcbcrConversion      ycbcrConversion,
                                             const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroySamplerYcbcrConversionKHR( device, ycbcrConversion, pAllocator );
    }

    //=== VK_KHR_bind_memory2 ===

    VkResult vkBindBufferMemory2KHR( VkDevice                       device,
                                     uint32_t                       bindInfoCount,
                                     const VkBindBufferMemoryInfo * pBindInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindBufferMemory2KHR( device, bindInfoCount, pBindInfos );
    }

    VkResult vkBindImageMemory2KHR( VkDevice                      device,
                                    uint32_t                      bindInfoCount,
                                    const VkBindImageMemoryInfo * pBindInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindImageMemory2KHR( device, bindInfoCount, pBindInfos );
    }

    //=== VK_EXT_image_drm_format_modifier ===

    VkResult vkGetImageDrmFormatModifierPropertiesEXT(
      VkDevice device, VkImage image, VkImageDrmFormatModifierPropertiesEXT * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageDrmFormatModifierPropertiesEXT( device, image, pProperties );
    }

    //=== VK_EXT_validation_cache ===

    VkResult vkCreateValidationCacheEXT( VkDevice                               device,
                                         const VkValidationCacheCreateInfoEXT * pCreateInfo,
                                         const VkAllocationCallbacks *          pAllocator,
                                         VkValidationCacheEXT * pValidationCache ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateValidationCacheEXT( device, pCreateInfo, pAllocator, pValidationCache );
    }

    void vkDestroyValidationCacheEXT( VkDevice                      device,
                                      VkValidationCacheEXT          validationCache,
                                      const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyValidationCacheEXT( device, validationCache, pAllocator );
    }

    VkResult vkMergeValidationCachesEXT( VkDevice                     device,
                                         VkValidationCacheEXT         dstCache,
                                         uint32_t                     srcCacheCount,
                                         const VkValidationCacheEXT * pSrcCaches ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkMergeValidationCachesEXT( device, dstCache, srcCacheCount, pSrcCaches );
    }

    VkResult vkGetValidationCacheDataEXT( VkDevice             device,
                                          VkValidationCacheEXT validationCache,
                                          size_t *             pDataSize,
                                          void *               pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetValidationCacheDataEXT( device, validationCache, pDataSize, pData );
    }

    //=== VK_NV_shading_rate_image ===

    void vkCmdBindShadingRateImageNV( VkCommandBuffer commandBuffer,
                                      VkImageView     imageView,
                                      VkImageLayout   imageLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindShadingRateImageNV( commandBuffer, imageView, imageLayout );
    }

    void vkCmdSetViewportShadingRatePaletteNV( VkCommandBuffer                commandBuffer,
                                               uint32_t                       firstViewport,
                                               uint32_t                       viewportCount,
                                               const VkShadingRatePaletteNV * pShadingRatePalettes ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewportShadingRatePaletteNV(
        commandBuffer, firstViewport, viewportCount, pShadingRatePalettes );
    }

    void
      vkCmdSetCoarseSampleOrderNV( VkCommandBuffer                     commandBuffer,
                                   VkCoarseSampleOrderTypeNV           sampleOrderType,
                                   uint32_t                            customSampleOrderCount,
                                   const VkCoarseSampleOrderCustomNV * pCustomSampleOrders ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCoarseSampleOrderNV(
        commandBuffer, sampleOrderType, customSampleOrderCount, pCustomSampleOrders );
    }

    //=== VK_NV_ray_tracing ===

    VkResult
      vkCreateAccelerationStructureNV( VkDevice                                    device,
                                       const VkAccelerationStructureCreateInfoNV * pCreateInfo,
                                       const VkAllocationCallbacks *               pAllocator,
                                       VkAccelerationStructureNV * pAccelerationStructure ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateAccelerationStructureNV( device, pCreateInfo, pAllocator, pAccelerationStructure );
    }

    void vkDestroyAccelerationStructureNV( VkDevice                      device,
                                           VkAccelerationStructureNV     accelerationStructure,
                                           const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyAccelerationStructureNV( device, accelerationStructure, pAllocator );
    }

    void vkGetAccelerationStructureMemoryRequirementsNV( VkDevice                                                device,
                                                         const VkAccelerationStructureMemoryRequirementsInfoNV * pInfo,
                                                         VkMemoryRequirements2KHR * pMemoryRequirements ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAccelerationStructureMemoryRequirementsNV( device, pInfo, pMemoryRequirements );
    }

    VkResult vkBindAccelerationStructureMemoryNV( VkDevice                                        device,
                                                  uint32_t                                        bindInfoCount,
                                                  const VkBindAccelerationStructureMemoryInfoNV * pBindInfos ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindAccelerationStructureMemoryNV( device, bindInfoCount, pBindInfos );
    }

    void vkCmdBuildAccelerationStructureNV( VkCommandBuffer                       commandBuffer,
                                            const VkAccelerationStructureInfoNV * pInfo,
                                            VkBuffer                              instanceData,
                                            VkDeviceSize                          instanceOffset,
                                            VkBool32                              update,
                                            VkAccelerationStructureNV             dst,
                                            VkAccelerationStructureNV             src,
                                            VkBuffer                              scratch,
                                            VkDeviceSize scratchOffset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBuildAccelerationStructureNV(
        commandBuffer, pInfo, instanceData, instanceOffset, update, dst, src, scratch, scratchOffset );
    }

    void vkCmdCopyAccelerationStructureNV( VkCommandBuffer                    commandBuffer,
                                           VkAccelerationStructureNV          dst,
                                           VkAccelerationStructureNV          src,
                                           VkCopyAccelerationStructureModeKHR mode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyAccelerationStructureNV( commandBuffer, dst, src, mode );
    }

    void vkCmdTraceRaysNV( VkCommandBuffer commandBuffer,
                           VkBuffer        raygenShaderBindingTableBuffer,
                           VkDeviceSize    raygenShaderBindingOffset,
                           VkBuffer        missShaderBindingTableBuffer,
                           VkDeviceSize    missShaderBindingOffset,
                           VkDeviceSize    missShaderBindingStride,
                           VkBuffer        hitShaderBindingTableBuffer,
                           VkDeviceSize    hitShaderBindingOffset,
                           VkDeviceSize    hitShaderBindingStride,
                           VkBuffer        callableShaderBindingTableBuffer,
                           VkDeviceSize    callableShaderBindingOffset,
                           VkDeviceSize    callableShaderBindingStride,
                           uint32_t        width,
                           uint32_t        height,
                           uint32_t        depth ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdTraceRaysNV( commandBuffer,
                                 raygenShaderBindingTableBuffer,
                                 raygenShaderBindingOffset,
                                 missShaderBindingTableBuffer,
                                 missShaderBindingOffset,
                                 missShaderBindingStride,
                                 hitShaderBindingTableBuffer,
                                 hitShaderBindingOffset,
                                 hitShaderBindingStride,
                                 callableShaderBindingTableBuffer,
                                 callableShaderBindingOffset,
                                 callableShaderBindingStride,
                                 width,
                                 height,
                                 depth );
    }

    VkResult vkCreateRayTracingPipelinesNV( VkDevice                                 device,
                                            VkPipelineCache                          pipelineCache,
                                            uint32_t                                 createInfoCount,
                                            const VkRayTracingPipelineCreateInfoNV * pCreateInfos,
                                            const VkAllocationCallbacks *            pAllocator,
                                            VkPipeline * pPipelines ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateRayTracingPipelinesNV(
        device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines );
    }

    VkResult vkGetRayTracingShaderGroupHandlesNV( VkDevice   device,
                                                  VkPipeline pipeline,
                                                  uint32_t   firstGroup,
                                                  uint32_t   groupCount,
                                                  size_t     dataSize,
                                                  void *     pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRayTracingShaderGroupHandlesNV( device, pipeline, firstGroup, groupCount, dataSize, pData );
    }

    VkResult vkGetAccelerationStructureHandleNV( VkDevice                  device,
                                                 VkAccelerationStructureNV accelerationStructure,
                                                 size_t                    dataSize,
                                                 void *                    pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAccelerationStructureHandleNV( device, accelerationStructure, dataSize, pData );
    }

    void vkCmdWriteAccelerationStructuresPropertiesNV( VkCommandBuffer                   commandBuffer,
                                                       uint32_t                          accelerationStructureCount,
                                                       const VkAccelerationStructureNV * pAccelerationStructures,
                                                       VkQueryType                       queryType,
                                                       VkQueryPool                       queryPool,
                                                       uint32_t firstQuery ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteAccelerationStructuresPropertiesNV(
        commandBuffer, accelerationStructureCount, pAccelerationStructures, queryType, queryPool, firstQuery );
    }

    VkResult vkCompileDeferredNV( VkDevice device, VkPipeline pipeline, uint32_t shader ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCompileDeferredNV( device, pipeline, shader );
    }

    //=== VK_KHR_maintenance3 ===

    void vkGetDescriptorSetLayoutSupportKHR( VkDevice                                device,
                                             const VkDescriptorSetLayoutCreateInfo * pCreateInfo,
                                             VkDescriptorSetLayoutSupport * pSupport ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDescriptorSetLayoutSupportKHR( device, pCreateInfo, pSupport );
    }

    //=== VK_KHR_draw_indirect_count ===

    void vkCmdDrawIndirectCountKHR( VkCommandBuffer commandBuffer,
                                    VkBuffer        buffer,
                                    VkDeviceSize    offset,
                                    VkBuffer        countBuffer,
                                    VkDeviceSize    countBufferOffset,
                                    uint32_t        maxDrawCount,
                                    uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndirectCountKHR(
        commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    void vkCmdDrawIndexedIndirectCountKHR( VkCommandBuffer commandBuffer,
                                           VkBuffer        buffer,
                                           VkDeviceSize    offset,
                                           VkBuffer        countBuffer,
                                           VkDeviceSize    countBufferOffset,
                                           uint32_t        maxDrawCount,
                                           uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexedIndirectCountKHR(
        commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    //=== VK_EXT_external_memory_host ===

    VkResult vkGetMemoryHostPointerPropertiesEXT(
      VkDevice                           device,
      VkExternalMemoryHandleTypeFlagBits handleType,
      const void *                       pHostPointer,
      VkMemoryHostPointerPropertiesEXT * pMemoryHostPointerProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryHostPointerPropertiesEXT( device, handleType, pHostPointer, pMemoryHostPointerProperties );
    }

    //=== VK_AMD_buffer_marker ===

    void vkCmdWriteBufferMarkerAMD( VkCommandBuffer         commandBuffer,
                                    VkPipelineStageFlagBits pipelineStage,
                                    VkBuffer                dstBuffer,
                                    VkDeviceSize            dstOffset,
                                    uint32_t                marker ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteBufferMarkerAMD( commandBuffer, pipelineStage, dstBuffer, dstOffset, marker );
    }

    //=== VK_EXT_calibrated_timestamps ===

    VkResult vkGetPhysicalDeviceCalibrateableTimeDomainsEXT( VkPhysicalDevice  physicalDevice,
                                                             uint32_t *        pTimeDomainCount,
                                                             VkTimeDomainEXT * pTimeDomains ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceCalibrateableTimeDomainsEXT( physicalDevice, pTimeDomainCount, pTimeDomains );
    }

    VkResult vkGetCalibratedTimestampsEXT( VkDevice                             device,
                                           uint32_t                             timestampCount,
                                           const VkCalibratedTimestampInfoEXT * pTimestampInfos,
                                           uint64_t *                           pTimestamps,
                                           uint64_t * pMaxDeviation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetCalibratedTimestampsEXT( device, timestampCount, pTimestampInfos, pTimestamps, pMaxDeviation );
    }

    //=== VK_NV_mesh_shader ===

    void vkCmdDrawMeshTasksNV( VkCommandBuffer commandBuffer,
                               uint32_t        taskCount,
                               uint32_t        firstTask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMeshTasksNV( commandBuffer, taskCount, firstTask );
    }

    void vkCmdDrawMeshTasksIndirectNV( VkCommandBuffer commandBuffer,
                                       VkBuffer        buffer,
                                       VkDeviceSize    offset,
                                       uint32_t        drawCount,
                                       uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMeshTasksIndirectNV( commandBuffer, buffer, offset, drawCount, stride );
    }

    void vkCmdDrawMeshTasksIndirectCountNV( VkCommandBuffer commandBuffer,
                                            VkBuffer        buffer,
                                            VkDeviceSize    offset,
                                            VkBuffer        countBuffer,
                                            VkDeviceSize    countBufferOffset,
                                            uint32_t        maxDrawCount,
                                            uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMeshTasksIndirectCountNV(
        commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    //=== VK_NV_scissor_exclusive ===

    void vkCmdSetExclusiveScissorNV( VkCommandBuffer  commandBuffer,
                                     uint32_t         firstExclusiveScissor,
                                     uint32_t         exclusiveScissorCount,
                                     const VkRect2D * pExclusiveScissors ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetExclusiveScissorNV(
        commandBuffer, firstExclusiveScissor, exclusiveScissorCount, pExclusiveScissors );
    }

    //=== VK_NV_device_diagnostic_checkpoints ===

    void vkCmdSetCheckpointNV( VkCommandBuffer commandBuffer, const void * pCheckpointMarker ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCheckpointNV( commandBuffer, pCheckpointMarker );
    }

    void vkGetQueueCheckpointDataNV( VkQueue              queue,
                                     uint32_t *           pCheckpointDataCount,
                                     VkCheckpointDataNV * pCheckpointData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetQueueCheckpointDataNV( queue, pCheckpointDataCount, pCheckpointData );
    }

    //=== VK_KHR_timeline_semaphore ===

    VkResult vkGetSemaphoreCounterValueKHR( VkDevice    device,
                                            VkSemaphore semaphore,
                                            uint64_t *  pValue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreCounterValueKHR( device, semaphore, pValue );
    }

    VkResult vkWaitSemaphoresKHR( VkDevice                    device,
                                  const VkSemaphoreWaitInfo * pWaitInfo,
                                  uint64_t                    timeout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWaitSemaphoresKHR( device, pWaitInfo, timeout );
    }

    VkResult vkSignalSemaphoreKHR( VkDevice                      device,
                                   const VkSemaphoreSignalInfo * pSignalInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSignalSemaphoreKHR( device, pSignalInfo );
    }

    //=== VK_INTEL_performance_query ===

    VkResult vkInitializePerformanceApiINTEL(
      VkDevice device, const VkInitializePerformanceApiInfoINTEL * pInitializeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkInitializePerformanceApiINTEL( device, pInitializeInfo );
    }

    void vkUninitializePerformanceApiINTEL( VkDevice device ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUninitializePerformanceApiINTEL( device );
    }

    VkResult
      vkCmdSetPerformanceMarkerINTEL( VkCommandBuffer                      commandBuffer,
                                      const VkPerformanceMarkerInfoINTEL * pMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPerformanceMarkerINTEL( commandBuffer, pMarkerInfo );
    }

    VkResult vkCmdSetPerformanceStreamMarkerINTEL(
      VkCommandBuffer commandBuffer, const VkPerformanceStreamMarkerInfoINTEL * pMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPerformanceStreamMarkerINTEL( commandBuffer, pMarkerInfo );
    }

    VkResult
      vkCmdSetPerformanceOverrideINTEL( VkCommandBuffer                        commandBuffer,
                                        const VkPerformanceOverrideInfoINTEL * pOverrideInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPerformanceOverrideINTEL( commandBuffer, pOverrideInfo );
    }

    VkResult vkAcquirePerformanceConfigurationINTEL( VkDevice                                           device,
                                                     const VkPerformanceConfigurationAcquireInfoINTEL * pAcquireInfo,
                                                     VkPerformanceConfigurationINTEL * pConfiguration ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquirePerformanceConfigurationINTEL( device, pAcquireInfo, pConfiguration );
    }

    VkResult
      vkReleasePerformanceConfigurationINTEL( VkDevice                        device,
                                              VkPerformanceConfigurationINTEL configuration ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkReleasePerformanceConfigurationINTEL( device, configuration );
    }

    VkResult
      vkQueueSetPerformanceConfigurationINTEL( VkQueue                         queue,
                                               VkPerformanceConfigurationINTEL configuration ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueSetPerformanceConfigurationINTEL( queue, configuration );
    }

    VkResult vkGetPerformanceParameterINTEL( VkDevice                        device,
                                             VkPerformanceParameterTypeINTEL parameter,
                                             VkPerformanceValueINTEL *       pValue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPerformanceParameterINTEL( device, parameter, pValue );
    }

    //=== VK_AMD_display_native_hdr ===

    void vkSetLocalDimmingAMD( VkDevice       device,
                               VkSwapchainKHR swapChain,
                               VkBool32       localDimmingEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetLocalDimmingAMD( device, swapChain, localDimmingEnable );
    }

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_imagepipe_surface ===

    VkResult vkCreateImagePipeSurfaceFUCHSIA( VkInstance                                  instance,
                                              const VkImagePipeSurfaceCreateInfoFUCHSIA * pCreateInfo,
                                              const VkAllocationCallbacks *               pAllocator,
                                              VkSurfaceKHR * pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateImagePipeSurfaceFUCHSIA( instance, pCreateInfo, pAllocator, pSurface );
    }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
    //=== VK_EXT_metal_surface ===

    VkResult vkCreateMetalSurfaceEXT( VkInstance                          instance,
                                      const VkMetalSurfaceCreateInfoEXT * pCreateInfo,
                                      const VkAllocationCallbacks *       pAllocator,
                                      VkSurfaceKHR *                      pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateMetalSurfaceEXT( instance, pCreateInfo, pAllocator, pSurface );
    }
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

    //=== VK_KHR_fragment_shading_rate ===

    VkResult vkGetPhysicalDeviceFragmentShadingRatesKHR(
      VkPhysicalDevice                         physicalDevice,
      uint32_t *                               pFragmentShadingRateCount,
      VkPhysicalDeviceFragmentShadingRateKHR * pFragmentShadingRates ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFragmentShadingRatesKHR(
        physicalDevice, pFragmentShadingRateCount, pFragmentShadingRates );
    }

    void vkCmdSetFragmentShadingRateKHR( VkCommandBuffer                          commandBuffer,
                                         const VkExtent2D *                       pFragmentSize,
                                         const VkFragmentShadingRateCombinerOpKHR combinerOps[2] ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetFragmentShadingRateKHR( commandBuffer, pFragmentSize, combinerOps );
    }

    //=== VK_EXT_buffer_device_address ===

    VkDeviceAddress vkGetBufferDeviceAddressEXT( VkDevice                          device,
                                                 const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferDeviceAddressEXT( device, pInfo );
    }

    //=== VK_EXT_tooling_info ===

    VkResult vkGetPhysicalDeviceToolPropertiesEXT( VkPhysicalDevice                    physicalDevice,
                                                   uint32_t *                          pToolCount,
                                                   VkPhysicalDeviceToolPropertiesEXT * pToolProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceToolPropertiesEXT( physicalDevice, pToolCount, pToolProperties );
    }

    //=== VK_NV_cooperative_matrix ===

    VkResult vkGetPhysicalDeviceCooperativeMatrixPropertiesNV( VkPhysicalDevice                  physicalDevice,
                                                               uint32_t *                        pPropertyCount,
                                                               VkCooperativeMatrixPropertiesNV * pProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceCooperativeMatrixPropertiesNV( physicalDevice, pPropertyCount, pProperties );
    }

    //=== VK_NV_coverage_reduction_mode ===

    VkResult vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
      VkPhysicalDevice                         physicalDevice,
      uint32_t *                               pCombinationCount,
      VkFramebufferMixedSamplesCombinationNV * pCombinations ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
        physicalDevice, pCombinationCount, pCombinations );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_EXT_full_screen_exclusive ===

    VkResult vkGetPhysicalDeviceSurfacePresentModes2EXT( VkPhysicalDevice                        physicalDevice,
                                                         const VkPhysicalDeviceSurfaceInfo2KHR * pSurfaceInfo,
                                                         uint32_t *                              pPresentModeCount,
                                                         VkPresentModeKHR * pPresentModes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfacePresentModes2EXT(
        physicalDevice, pSurfaceInfo, pPresentModeCount, pPresentModes );
    }

    VkResult vkAcquireFullScreenExclusiveModeEXT( VkDevice device, VkSwapchainKHR swapchain ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireFullScreenExclusiveModeEXT( device, swapchain );
    }

    VkResult vkReleaseFullScreenExclusiveModeEXT( VkDevice device, VkSwapchainKHR swapchain ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkReleaseFullScreenExclusiveModeEXT( device, swapchain );
    }

    VkResult
      vkGetDeviceGroupSurfacePresentModes2EXT( VkDevice                                device,
                                               const VkPhysicalDeviceSurfaceInfo2KHR * pSurfaceInfo,
                                               VkDeviceGroupPresentModeFlagsKHR * pModes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceGroupSurfacePresentModes2EXT( device, pSurfaceInfo, pModes );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_EXT_headless_surface ===

    VkResult vkCreateHeadlessSurfaceEXT( VkInstance                             instance,
                                         const VkHeadlessSurfaceCreateInfoEXT * pCreateInfo,
                                         const VkAllocationCallbacks *          pAllocator,
                                         VkSurfaceKHR *                         pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateHeadlessSurfaceEXT( instance, pCreateInfo, pAllocator, pSurface );
    }

    //=== VK_KHR_buffer_device_address ===

    VkDeviceAddress vkGetBufferDeviceAddressKHR( VkDevice                          device,
                                                 const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferDeviceAddressKHR( device, pInfo );
    }

    uint64_t vkGetBufferOpaqueCaptureAddressKHR( VkDevice                          device,
                                                 const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferOpaqueCaptureAddressKHR( device, pInfo );
    }

    uint64_t vkGetDeviceMemoryOpaqueCaptureAddressKHR(
      VkDevice device, const VkDeviceMemoryOpaqueCaptureAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceMemoryOpaqueCaptureAddressKHR( device, pInfo );
    }

    //=== VK_EXT_line_rasterization ===

    void vkCmdSetLineStippleEXT( VkCommandBuffer commandBuffer,
                                 uint32_t        lineStippleFactor,
                                 uint16_t        lineStipplePattern ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetLineStippleEXT( commandBuffer, lineStippleFactor, lineStipplePattern );
    }

    //=== VK_EXT_host_query_reset ===

    void vkResetQueryPoolEXT( VkDevice    device,
                              VkQueryPool queryPool,
                              uint32_t    firstQuery,
                              uint32_t    queryCount ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetQueryPoolEXT( device, queryPool, firstQuery, queryCount );
    }

    //=== VK_EXT_extended_dynamic_state ===

    void vkCmdSetCullModeEXT( VkCommandBuffer commandBuffer, VkCullModeFlags cullMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCullModeEXT( commandBuffer, cullMode );
    }

    void vkCmdSetFrontFaceEXT( VkCommandBuffer commandBuffer, VkFrontFace frontFace ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetFrontFaceEXT( commandBuffer, frontFace );
    }

    void vkCmdSetPrimitiveTopologyEXT( VkCommandBuffer     commandBuffer,
                                       VkPrimitiveTopology primitiveTopology ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPrimitiveTopologyEXT( commandBuffer, primitiveTopology );
    }

    void vkCmdSetViewportWithCountEXT( VkCommandBuffer    commandBuffer,
                                       uint32_t           viewportCount,
                                       const VkViewport * pViewports ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewportWithCountEXT( commandBuffer, viewportCount, pViewports );
    }

    void vkCmdSetScissorWithCountEXT( VkCommandBuffer  commandBuffer,
                                      uint32_t         scissorCount,
                                      const VkRect2D * pScissors ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetScissorWithCountEXT( commandBuffer, scissorCount, pScissors );
    }

    void vkCmdBindVertexBuffers2EXT( VkCommandBuffer      commandBuffer,
                                     uint32_t             firstBinding,
                                     uint32_t             bindingCount,
                                     const VkBuffer *     pBuffers,
                                     const VkDeviceSize * pOffsets,
                                     const VkDeviceSize * pSizes,
                                     const VkDeviceSize * pStrides ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindVertexBuffers2EXT(
        commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes, pStrides );
    }

    void vkCmdSetDepthTestEnableEXT( VkCommandBuffer commandBuffer, VkBool32 depthTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthTestEnableEXT( commandBuffer, depthTestEnable );
    }

    void vkCmdSetDepthWriteEnableEXT( VkCommandBuffer commandBuffer,
                                      VkBool32        depthWriteEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthWriteEnableEXT( commandBuffer, depthWriteEnable );
    }

    void vkCmdSetDepthCompareOpEXT( VkCommandBuffer commandBuffer,
                                    VkCompareOp     depthCompareOp ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthCompareOpEXT( commandBuffer, depthCompareOp );
    }

    void vkCmdSetDepthBoundsTestEnableEXT( VkCommandBuffer commandBuffer,
                                           VkBool32        depthBoundsTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBoundsTestEnableEXT( commandBuffer, depthBoundsTestEnable );
    }

    void vkCmdSetStencilTestEnableEXT( VkCommandBuffer commandBuffer,
                                       VkBool32        stencilTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilTestEnableEXT( commandBuffer, stencilTestEnable );
    }

    void vkCmdSetStencilOpEXT( VkCommandBuffer    commandBuffer,
                               VkStencilFaceFlags faceMask,
                               VkStencilOp        failOp,
                               VkStencilOp        passOp,
                               VkStencilOp        depthFailOp,
                               VkCompareOp        compareOp ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilOpEXT( commandBuffer, faceMask, failOp, passOp, depthFailOp, compareOp );
    }

    //=== VK_KHR_deferred_host_operations ===

    VkResult vkCreateDeferredOperationKHR( VkDevice                      device,
                                           const VkAllocationCallbacks * pAllocator,
                                           VkDeferredOperationKHR *      pDeferredOperation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDeferredOperationKHR( device, pAllocator, pDeferredOperation );
    }

    void vkDestroyDeferredOperationKHR( VkDevice                      device,
                                        VkDeferredOperationKHR        operation,
                                        const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDeferredOperationKHR( device, operation, pAllocator );
    }

    uint32_t vkGetDeferredOperationMaxConcurrencyKHR( VkDevice               device,
                                                      VkDeferredOperationKHR operation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeferredOperationMaxConcurrencyKHR( device, operation );
    }

    VkResult vkGetDeferredOperationResultKHR( VkDevice               device,
                                              VkDeferredOperationKHR operation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeferredOperationResultKHR( device, operation );
    }

    VkResult vkDeferredOperationJoinKHR( VkDevice device, VkDeferredOperationKHR operation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDeferredOperationJoinKHR( device, operation );
    }

    //=== VK_KHR_pipeline_executable_properties ===

    VkResult
      vkGetPipelineExecutablePropertiesKHR( VkDevice                            device,
                                            const VkPipelineInfoKHR *           pPipelineInfo,
                                            uint32_t *                          pExecutableCount,
                                            VkPipelineExecutablePropertiesKHR * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineExecutablePropertiesKHR( device, pPipelineInfo, pExecutableCount, pProperties );
    }

    VkResult
      vkGetPipelineExecutableStatisticsKHR( VkDevice                            device,
                                            const VkPipelineExecutableInfoKHR * pExecutableInfo,
                                            uint32_t *                          pStatisticCount,
                                            VkPipelineExecutableStatisticKHR *  pStatistics ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineExecutableStatisticsKHR( device, pExecutableInfo, pStatisticCount, pStatistics );
    }

    VkResult vkGetPipelineExecutableInternalRepresentationsKHR(
      VkDevice                                        device,
      const VkPipelineExecutableInfoKHR *             pExecutableInfo,
      uint32_t *                                      pInternalRepresentationCount,
      VkPipelineExecutableInternalRepresentationKHR * pInternalRepresentations ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineExecutableInternalRepresentationsKHR(
        device, pExecutableInfo, pInternalRepresentationCount, pInternalRepresentations );
    }

    //=== VK_NV_device_generated_commands ===

    void vkGetGeneratedCommandsMemoryRequirementsNV( VkDevice                                            device,
                                                     const VkGeneratedCommandsMemoryRequirementsInfoNV * pInfo,
                                                     VkMemoryRequirements2 * pMemoryRequirements ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetGeneratedCommandsMemoryRequirementsNV( device, pInfo, pMemoryRequirements );
    }

    void vkCmdPreprocessGeneratedCommandsNV( VkCommandBuffer                   commandBuffer,
                                             const VkGeneratedCommandsInfoNV * pGeneratedCommandsInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPreprocessGeneratedCommandsNV( commandBuffer, pGeneratedCommandsInfo );
    }

    void vkCmdExecuteGeneratedCommandsNV( VkCommandBuffer                   commandBuffer,
                                          VkBool32                          isPreprocessed,
                                          const VkGeneratedCommandsInfoNV * pGeneratedCommandsInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdExecuteGeneratedCommandsNV( commandBuffer, isPreprocessed, pGeneratedCommandsInfo );
    }

    void vkCmdBindPipelineShaderGroupNV( VkCommandBuffer     commandBuffer,
                                         VkPipelineBindPoint pipelineBindPoint,
                                         VkPipeline          pipeline,
                                         uint32_t            groupIndex ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindPipelineShaderGroupNV( commandBuffer, pipelineBindPoint, pipeline, groupIndex );
    }

    VkResult
      vkCreateIndirectCommandsLayoutNV( VkDevice                                     device,
                                        const VkIndirectCommandsLayoutCreateInfoNV * pCreateInfo,
                                        const VkAllocationCallbacks *                pAllocator,
                                        VkIndirectCommandsLayoutNV * pIndirectCommandsLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateIndirectCommandsLayoutNV( device, pCreateInfo, pAllocator, pIndirectCommandsLayout );
    }

    void vkDestroyIndirectCommandsLayoutNV( VkDevice                      device,
                                            VkIndirectCommandsLayoutNV    indirectCommandsLayout,
                                            const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyIndirectCommandsLayoutNV( device, indirectCommandsLayout, pAllocator );
    }

    //=== VK_EXT_acquire_drm_display ===

    VkResult vkAcquireDrmDisplayEXT( VkPhysicalDevice physicalDevice,
                                     int32_t          drmFd,
                                     VkDisplayKHR     display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireDrmDisplayEXT( physicalDevice, drmFd, display );
    }

    VkResult vkGetDrmDisplayEXT( VkPhysicalDevice physicalDevice,
                                 int32_t          drmFd,
                                 uint32_t         connectorId,
                                 VkDisplayKHR *   display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDrmDisplayEXT( physicalDevice, drmFd, connectorId, display );
    }

    //=== VK_EXT_private_data ===

    VkResult vkCreatePrivateDataSlotEXT( VkDevice                               device,
                                         const VkPrivateDataSlotCreateInfoEXT * pCreateInfo,
                                         const VkAllocationCallbacks *          pAllocator,
                                         VkPrivateDataSlotEXT * pPrivateDataSlot ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreatePrivateDataSlotEXT( device, pCreateInfo, pAllocator, pPrivateDataSlot );
    }

    void vkDestroyPrivateDataSlotEXT( VkDevice                      device,
                                      VkPrivateDataSlotEXT          privateDataSlot,
                                      const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyPrivateDataSlotEXT( device, privateDataSlot, pAllocator );
    }

    VkResult vkSetPrivateDataEXT( VkDevice             device,
                                  VkObjectType         objectType,
                                  uint64_t             objectHandle,
                                  VkPrivateDataSlotEXT privateDataSlot,
                                  uint64_t             data ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetPrivateDataEXT( device, objectType, objectHandle, privateDataSlot, data );
    }

    void vkGetPrivateDataEXT( VkDevice             device,
                              VkObjectType         objectType,
                              uint64_t             objectHandle,
                              VkPrivateDataSlotEXT privateDataSlot,
                              uint64_t *           pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPrivateDataEXT( device, objectType, objectHandle, privateDataSlot, pData );
    }

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_KHR_video_encode_queue ===

    void vkCmdEncodeVideoKHR( VkCommandBuffer              commandBuffer,
                              const VkVideoEncodeInfoKHR * pEncodeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEncodeVideoKHR( commandBuffer, pEncodeInfo );
    }
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

    //=== VK_KHR_synchronization2 ===

    void vkCmdSetEvent2KHR( VkCommandBuffer             commandBuffer,
                            VkEvent                     event,
                            const VkDependencyInfoKHR * pDependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetEvent2KHR( commandBuffer, event, pDependencyInfo );
    }

    void vkCmdResetEvent2KHR( VkCommandBuffer          commandBuffer,
                              VkEvent                  event,
                              VkPipelineStageFlags2KHR stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResetEvent2KHR( commandBuffer, event, stageMask );
    }

    void vkCmdWaitEvents2KHR( VkCommandBuffer             commandBuffer,
                              uint32_t                    eventCount,
                              const VkEvent *             pEvents,
                              const VkDependencyInfoKHR * pDependencyInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWaitEvents2KHR( commandBuffer, eventCount, pEvents, pDependencyInfos );
    }

    void vkCmdPipelineBarrier2KHR( VkCommandBuffer             commandBuffer,
                                   const VkDependencyInfoKHR * pDependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPipelineBarrier2KHR( commandBuffer, pDependencyInfo );
    }

    void vkCmdWriteTimestamp2KHR( VkCommandBuffer          commandBuffer,
                                  VkPipelineStageFlags2KHR stage,
                                  VkQueryPool              queryPool,
                                  uint32_t                 query ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteTimestamp2KHR( commandBuffer, stage, queryPool, query );
    }

    VkResult vkQueueSubmit2KHR( VkQueue                  queue,
                                uint32_t                 submitCount,
                                const VkSubmitInfo2KHR * pSubmits,
                                VkFence                  fence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueSubmit2KHR( queue, submitCount, pSubmits, fence );
    }

    void vkCmdWriteBufferMarker2AMD( VkCommandBuffer          commandBuffer,
                                     VkPipelineStageFlags2KHR stage,
                                     VkBuffer                 dstBuffer,
                                     VkDeviceSize             dstOffset,
                                     uint32_t                 marker ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteBufferMarker2AMD( commandBuffer, stage, dstBuffer, dstOffset, marker );
    }

    void vkGetQueueCheckpointData2NV( VkQueue               queue,
                                      uint32_t *            pCheckpointDataCount,
                                      VkCheckpointData2NV * pCheckpointData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetQueueCheckpointData2NV( queue, pCheckpointDataCount, pCheckpointData );
    }

    //=== VK_NV_fragment_shading_rate_enums ===

    void vkCmdSetFragmentShadingRateEnumNV( VkCommandBuffer                          commandBuffer,
                                            VkFragmentShadingRateNV                  shadingRate,
                                            const VkFragmentShadingRateCombinerOpKHR combinerOps[2] ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetFragmentShadingRateEnumNV( commandBuffer, shadingRate, combinerOps );
    }

    //=== VK_KHR_copy_commands2 ===

    void vkCmdCopyBuffer2KHR( VkCommandBuffer              commandBuffer,
                              const VkCopyBufferInfo2KHR * pCopyBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyBuffer2KHR( commandBuffer, pCopyBufferInfo );
    }

    void vkCmdCopyImage2KHR( VkCommandBuffer             commandBuffer,
                             const VkCopyImageInfo2KHR * pCopyImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyImage2KHR( commandBuffer, pCopyImageInfo );
    }

    void
      vkCmdCopyBufferToImage2KHR( VkCommandBuffer                     commandBuffer,
                                  const VkCopyBufferToImageInfo2KHR * pCopyBufferToImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyBufferToImage2KHR( commandBuffer, pCopyBufferToImageInfo );
    }

    void
      vkCmdCopyImageToBuffer2KHR( VkCommandBuffer                     commandBuffer,
                                  const VkCopyImageToBufferInfo2KHR * pCopyImageToBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyImageToBuffer2KHR( commandBuffer, pCopyImageToBufferInfo );
    }

    void vkCmdBlitImage2KHR( VkCommandBuffer             commandBuffer,
                             const VkBlitImageInfo2KHR * pBlitImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBlitImage2KHR( commandBuffer, pBlitImageInfo );
    }

    void vkCmdResolveImage2KHR( VkCommandBuffer                commandBuffer,
                                const VkResolveImageInfo2KHR * pResolveImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResolveImage2KHR( commandBuffer, pResolveImageInfo );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_NV_acquire_winrt_display ===

    VkResult vkAcquireWinrtDisplayNV( VkPhysicalDevice physicalDevice, VkDisplayKHR display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireWinrtDisplayNV( physicalDevice, display );
    }

    VkResult vkGetWinrtDisplayNV( VkPhysicalDevice physicalDevice,
                                  uint32_t         deviceRelativeId,
                                  VkDisplayKHR *   pDisplay ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetWinrtDisplayNV( physicalDevice, deviceRelativeId, pDisplay );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
    //=== VK_EXT_directfb_surface ===

    VkResult vkCreateDirectFBSurfaceEXT( VkInstance                             instance,
                                         const VkDirectFBSurfaceCreateInfoEXT * pCreateInfo,
                                         const VkAllocationCallbacks *          pAllocator,
                                         VkSurfaceKHR *                         pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateDirectFBSurfaceEXT( instance, pCreateInfo, pAllocator, pSurface );
    }

    VkBool32 vkGetPhysicalDeviceDirectFBPresentationSupportEXT( VkPhysicalDevice physicalDevice,
                                                                uint32_t         queueFamilyIndex,
                                                                IDirectFB *      dfb ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceDirectFBPresentationSupportEXT( physicalDevice, queueFamilyIndex, dfb );
    }
#  endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

    //=== VK_KHR_ray_tracing_pipeline ===

    void vkCmdTraceRaysKHR( VkCommandBuffer                         commandBuffer,
                            const VkStridedDeviceAddressRegionKHR * pRaygenShaderBindingTable,
                            const VkStridedDeviceAddressRegionKHR * pMissShaderBindingTable,
                            const VkStridedDeviceAddressRegionKHR * pHitShaderBindingTable,
                            const VkStridedDeviceAddressRegionKHR * pCallableShaderBindingTable,
                            uint32_t                                width,
                            uint32_t                                height,
                            uint32_t                                depth ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdTraceRaysKHR( commandBuffer,
                                  pRaygenShaderBindingTable,
                                  pMissShaderBindingTable,
                                  pHitShaderBindingTable,
                                  pCallableShaderBindingTable,
                                  width,
                                  height,
                                  depth );
    }

    VkResult vkCreateRayTracingPipelinesKHR( VkDevice                                  device,
                                             VkDeferredOperationKHR                    deferredOperation,
                                             VkPipelineCache                           pipelineCache,
                                             uint32_t                                  createInfoCount,
                                             const VkRayTracingPipelineCreateInfoKHR * pCreateInfos,
                                             const VkAllocationCallbacks *             pAllocator,
                                             VkPipeline * pPipelines ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateRayTracingPipelinesKHR(
        device, deferredOperation, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines );
    }

    VkResult vkGetRayTracingShaderGroupHandlesKHR( VkDevice   device,
                                                   VkPipeline pipeline,
                                                   uint32_t   firstGroup,
                                                   uint32_t   groupCount,
                                                   size_t     dataSize,
                                                   void *     pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRayTracingShaderGroupHandlesKHR( device, pipeline, firstGroup, groupCount, dataSize, pData );
    }

    VkResult vkGetRayTracingCaptureReplayShaderGroupHandlesKHR( VkDevice   device,
                                                                VkPipeline pipeline,
                                                                uint32_t   firstGroup,
                                                                uint32_t   groupCount,
                                                                size_t     dataSize,
                                                                void *     pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(
        device, pipeline, firstGroup, groupCount, dataSize, pData );
    }

    void vkCmdTraceRaysIndirectKHR( VkCommandBuffer                         commandBuffer,
                                    const VkStridedDeviceAddressRegionKHR * pRaygenShaderBindingTable,
                                    const VkStridedDeviceAddressRegionKHR * pMissShaderBindingTable,
                                    const VkStridedDeviceAddressRegionKHR * pHitShaderBindingTable,
                                    const VkStridedDeviceAddressRegionKHR * pCallableShaderBindingTable,
                                    VkDeviceAddress indirectDeviceAddress ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdTraceRaysIndirectKHR( commandBuffer,
                                          pRaygenShaderBindingTable,
                                          pMissShaderBindingTable,
                                          pHitShaderBindingTable,
                                          pCallableShaderBindingTable,
                                          indirectDeviceAddress );
    }

    VkDeviceSize vkGetRayTracingShaderGroupStackSizeKHR( VkDevice               device,
                                                         VkPipeline             pipeline,
                                                         uint32_t               group,
                                                         VkShaderGroupShaderKHR groupShader ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRayTracingShaderGroupStackSizeKHR( device, pipeline, group, groupShader );
    }

    void vkCmdSetRayTracingPipelineStackSizeKHR( VkCommandBuffer commandBuffer,
                                                 uint32_t        pipelineStackSize ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetRayTracingPipelineStackSizeKHR( commandBuffer, pipelineStackSize );
    }

    //=== VK_EXT_vertex_input_dynamic_state ===

    void vkCmdSetVertexInputEXT( VkCommandBuffer                               commandBuffer,
                                 uint32_t                                      vertexBindingDescriptionCount,
                                 const VkVertexInputBindingDescription2EXT *   pVertexBindingDescriptions,
                                 uint32_t                                      vertexAttributeDescriptionCount,
                                 const VkVertexInputAttributeDescription2EXT * pVertexAttributeDescriptions ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetVertexInputEXT( commandBuffer,
                                       vertexBindingDescriptionCount,
                                       pVertexBindingDescriptions,
                                       vertexAttributeDescriptionCount,
                                       pVertexAttributeDescriptions );
    }

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_external_memory ===

    VkResult vkGetMemoryZirconHandleFUCHSIA( VkDevice                                   device,
                                             const VkMemoryGetZirconHandleInfoFUCHSIA * pGetZirconHandleInfo,
                                             zx_handle_t * pZirconHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryZirconHandleFUCHSIA( device, pGetZirconHandleInfo, pZirconHandle );
    }

    VkResult vkGetMemoryZirconHandlePropertiesFUCHSIA(
      VkDevice                                device,
      VkExternalMemoryHandleTypeFlagBits      handleType,
      zx_handle_t                             zirconHandle,
      VkMemoryZirconHandlePropertiesFUCHSIA * pMemoryZirconHandleProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryZirconHandlePropertiesFUCHSIA(
        device, handleType, zirconHandle, pMemoryZirconHandleProperties );
    }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_external_semaphore ===

    VkResult vkImportSemaphoreZirconHandleFUCHSIA(
      VkDevice                                         device,
      const VkImportSemaphoreZirconHandleInfoFUCHSIA * pImportSemaphoreZirconHandleInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportSemaphoreZirconHandleFUCHSIA( device, pImportSemaphoreZirconHandleInfo );
    }

    VkResult vkGetSemaphoreZirconHandleFUCHSIA( VkDevice                                      device,
                                                const VkSemaphoreGetZirconHandleInfoFUCHSIA * pGetZirconHandleInfo,
                                                zx_handle_t * pZirconHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreZirconHandleFUCHSIA( device, pGetZirconHandleInfo, pZirconHandle );
    }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

    //=== VK_HUAWEI_subpass_shading ===

    VkResult vkGetSubpassShadingMaxWorkgroupSizeHUAWEI( VkRenderPass renderpass,
                                                        VkExtent2D * pMaxWorkgroupSize ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSubpassShadingMaxWorkgroupSizeHUAWEI( renderpass, pMaxWorkgroupSize );
    }

    void vkCmdSubpassShadingHUAWEI( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSubpassShadingHUAWEI( commandBuffer );
    }

    //=== VK_EXT_extended_dynamic_state2 ===

    void vkCmdSetPatchControlPointsEXT( VkCommandBuffer commandBuffer,
                                        uint32_t        patchControlPoints ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPatchControlPointsEXT( commandBuffer, patchControlPoints );
    }

    void vkCmdSetRasterizerDiscardEnableEXT( VkCommandBuffer commandBuffer,
                                             VkBool32        rasterizerDiscardEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetRasterizerDiscardEnableEXT( commandBuffer, rasterizerDiscardEnable );
    }

    void vkCmdSetDepthBiasEnableEXT( VkCommandBuffer commandBuffer, VkBool32 depthBiasEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBiasEnableEXT( commandBuffer, depthBiasEnable );
    }

    void vkCmdSetLogicOpEXT( VkCommandBuffer commandBuffer, VkLogicOp logicOp ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetLogicOpEXT( commandBuffer, logicOp );
    }

    void vkCmdSetPrimitiveRestartEnableEXT( VkCommandBuffer commandBuffer,
                                            VkBool32        primitiveRestartEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPrimitiveRestartEnableEXT( commandBuffer, primitiveRestartEnable );
    }

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
    //=== VK_QNX_screen_surface ===

    VkResult vkCreateScreenSurfaceQNX( VkInstance                           instance,
                                       const VkScreenSurfaceCreateInfoQNX * pCreateInfo,
                                       const VkAllocationCallbacks *        pAllocator,
                                       VkSurfaceKHR *                       pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateScreenSurfaceQNX( instance, pCreateInfo, pAllocator, pSurface );
    }

    VkBool32 vkGetPhysicalDeviceScreenPresentationSupportQNX( VkPhysicalDevice        physicalDevice,
                                                              uint32_t                queueFamilyIndex,
                                                              struct _screen_window * window ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceScreenPresentationSupportQNX( physicalDevice, queueFamilyIndex, window );
    }
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

    //=== VK_EXT_color_write_enable ===

    void vkCmdSetColorWriteEnableEXT( VkCommandBuffer  commandBuffer,
                                      uint32_t         attachmentCount,
                                      const VkBool32 * pColorWriteEnables ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetColorWriteEnableEXT( commandBuffer, attachmentCount, pColorWriteEnables );
    }

    //=== VK_EXT_multi_draw ===

    void vkCmdDrawMultiEXT( VkCommandBuffer            commandBuffer,
                            uint32_t                   drawCount,
                            const VkMultiDrawInfoEXT * pVertexInfo,
                            uint32_t                   instanceCount,
                            uint32_t                   firstInstance,
                            uint32_t                   stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMultiEXT( commandBuffer, drawCount, pVertexInfo, instanceCount, firstInstance, stride );
    }

    void vkCmdDrawMultiIndexedEXT( VkCommandBuffer                   commandBuffer,
                                   uint32_t                          drawCount,
                                   const VkMultiDrawIndexedInfoEXT * pIndexInfo,
                                   uint32_t                          instanceCount,
                                   uint32_t                          firstInstance,
                                   uint32_t                          stride,
                                   const int32_t *                   pVertexOffset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMultiIndexedEXT(
        commandBuffer, drawCount, pIndexInfo, instanceCount, firstInstance, stride, pVertexOffset );
    }
  };
#endif

  class DispatchLoaderDynamic;
#if !defined( VULKAN_HPP_DISPATCH_LOADER_DYNAMIC )
#  if defined( VK_NO_PROTOTYPES )
#    define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#  else
#    define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 0
#  endif
#endif

#if !defined( VULKAN_HPP_STORAGE_API )
#  if defined( VULKAN_HPP_STORAGE_SHARED )
#    if defined( _MSC_VER )
#      if defined( VULKAN_HPP_STORAGE_SHARED_EXPORT )
#        define VULKAN_HPP_STORAGE_API __declspec( dllexport )
#      else
#        define VULKAN_HPP_STORAGE_API __declspec( dllimport )
#      endif
#    elif defined( __clang__ ) || defined( __GNUC__ )
#      if defined( VULKAN_HPP_STORAGE_SHARED_EXPORT )
#        define VULKAN_HPP_STORAGE_API __attribute__( ( visibility( "default" ) ) )
#      else
#        define VULKAN_HPP_STORAGE_API
#      endif
#    else
#      define VULKAN_HPP_STORAGE_API
#      pragma warning Unknown import / export semantics
#    endif
#  else
#    define VULKAN_HPP_STORAGE_API
#  endif
#endif

#if !defined( VULKAN_HPP_DEFAULT_DISPATCHER )
#  if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
#    define VULKAN_HPP_DEFAULT_DISPATCHER ::VULKAN_HPP_NAMESPACE::defaultDispatchLoaderDynamic
#    define VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE                     \
      namespace VULKAN_HPP_NAMESPACE                                               \
      {                                                                            \
        VULKAN_HPP_STORAGE_API DispatchLoaderDynamic defaultDispatchLoaderDynamic; \
      }
  extern VULKAN_HPP_STORAGE_API DispatchLoaderDynamic defaultDispatchLoaderDynamic;
#  else
#    define VULKAN_HPP_DEFAULT_DISPATCHER ::VULKAN_HPP_NAMESPACE::DispatchLoaderStatic()
#    define VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#  endif
#endif

#if !defined( VULKAN_HPP_DEFAULT_DISPATCHER_TYPE )
#  if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
#    define VULKAN_HPP_DEFAULT_DISPATCHER_TYPE ::VULKAN_HPP_NAMESPACE::DispatchLoaderDynamic
#  else
#    define VULKAN_HPP_DEFAULT_DISPATCHER_TYPE ::VULKAN_HPP_NAMESPACE::DispatchLoaderStatic
#  endif
#endif

#if defined( VULKAN_HPP_NO_DEFAULT_DISPATCHER )
#  define VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT
#  define VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT
#  define VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT
#else
#  define VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT = {}
#  define VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT = nullptr
#  define VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT = VULKAN_HPP_DEFAULT_DISPATCHER
#endif

  struct AllocationCallbacks;

  template <typename OwnerType, typename Dispatch>
  class ObjectDestroy
  {
  public:
    ObjectDestroy() = default;

    ObjectDestroy( OwnerType                           owner,
                   Optional<const AllocationCallbacks> allocationCallbacks
                                                       VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT,
                   Dispatch const &                    dispatch = VULKAN_HPP_DEFAULT_DISPATCHER ) VULKAN_HPP_NOEXCEPT
      : m_owner( owner )
      , m_allocationCallbacks( allocationCallbacks )
      , m_dispatch( &dispatch )
    {}

    OwnerType getOwner() const VULKAN_HPP_NOEXCEPT
    {
      return m_owner;
    }
    Optional<const AllocationCallbacks> getAllocator() const VULKAN_HPP_NOEXCEPT
    {
      return m_allocationCallbacks;
    }

  protected:
    template <typename T>
    void destroy( T t ) VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_owner && m_dispatch );
      m_owner.destroy( t, m_allocationCallbacks, *m_dispatch );
    }

  private:
    OwnerType                           m_owner               = {};
    Optional<const AllocationCallbacks> m_allocationCallbacks = nullptr;
    Dispatch const *                    m_dispatch            = nullptr;
  };

  class NoParent;

  template <typename Dispatch>
  class ObjectDestroy<NoParent, Dispatch>
  {
  public:
    ObjectDestroy() = default;

    ObjectDestroy( Optional<const AllocationCallbacks> allocationCallbacks,
                   Dispatch const &                    dispatch = VULKAN_HPP_DEFAULT_DISPATCHER ) VULKAN_HPP_NOEXCEPT
      : m_allocationCallbacks( allocationCallbacks )
      , m_dispatch( &dispatch )
    {}

    Optional<const AllocationCallbacks> getAllocator() const VULKAN_HPP_NOEXCEPT
    {
      return m_allocationCallbacks;
    }

  protected:
    template <typename T>
    void destroy( T t ) VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_dispatch );
      t.destroy( m_allocationCallbacks, *m_dispatch );
    }

  private:
    Optional<const AllocationCallbacks> m_allocationCallbacks = nullptr;
    Dispatch const *                    m_dispatch            = nullptr;
  };

  template <typename OwnerType, typename Dispatch>
  class ObjectFree
  {
  public:
    ObjectFree() = default;

    ObjectFree( OwnerType                           owner,
                Optional<const AllocationCallbacks> allocationCallbacks VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT,
                Dispatch const & dispatch = VULKAN_HPP_DEFAULT_DISPATCHER ) VULKAN_HPP_NOEXCEPT
      : m_owner( owner )
      , m_allocationCallbacks( allocationCallbacks )
      , m_dispatch( &dispatch )
    {}

    OwnerType getOwner() const VULKAN_HPP_NOEXCEPT
    {
      return m_owner;
    }

    Optional<const AllocationCallbacks> getAllocator() const VULKAN_HPP_NOEXCEPT
    {
      return m_allocationCallbacks;
    }

  protected:
    template <typename T>
    void destroy( T t ) VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_owner && m_dispatch );
      m_owner.free( t, m_allocationCallbacks, *m_dispatch );
    }

  private:
    OwnerType                           m_owner               = {};
    Optional<const AllocationCallbacks> m_allocationCallbacks = nullptr;
    Dispatch const *                    m_dispatch            = nullptr;
  };

  template <typename OwnerType, typename Dispatch>
  class ObjectRelease
  {
  public:
    ObjectRelease() = default;

    ObjectRelease( OwnerType owner, Dispatch const & dispatch = VULKAN_HPP_DEFAULT_DISPATCHER ) VULKAN_HPP_NOEXCEPT
      : m_owner( owner )
      , m_dispatch( &dispatch )
    {}

    OwnerType getOwner() const VULKAN_HPP_NOEXCEPT
    {
      return m_owner;
    }

  protected:
    template <typename T>
    void destroy( T t ) VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_owner && m_dispatch );
      m_owner.release( t, *m_dispatch );
    }

  private:
    OwnerType        m_owner    = {};
    Dispatch const * m_dispatch = nullptr;
  };

  template <typename OwnerType, typename PoolType, typename Dispatch>
  class PoolFree
  {
  public:
    PoolFree() = default;

    PoolFree( OwnerType        owner,
              PoolType         pool,
              Dispatch const & dispatch = VULKAN_HPP_DEFAULT_DISPATCHER ) VULKAN_HPP_NOEXCEPT
      : m_owner( owner )
      , m_pool( pool )
      , m_dispatch( &dispatch )
    {}

    OwnerType getOwner() const VULKAN_HPP_NOEXCEPT
    {
      return m_owner;
    }
    PoolType getPool() const VULKAN_HPP_NOEXCEPT
    {
      return m_pool;
    }

  protected:
    template <typename T>
    void destroy( T t ) VULKAN_HPP_NOEXCEPT
    {
      m_owner.free( m_pool, t, *m_dispatch );
    }

  private:
    OwnerType        m_owner    = OwnerType();
    PoolType         m_pool     = PoolType();
    Dispatch const * m_dispatch = nullptr;
  };

  using Bool32        = uint32_t;
  using DeviceAddress = uint64_t;
  using DeviceSize    = uint64_t;
  using SampleMask    = uint32_t;
}  // namespace VULKAN_HPP_NAMESPACE

#include <vulkan/vulkan_enums.hpp>

#ifndef VULKAN_HPP_NO_EXCEPTIONS
namespace std
{
  template <>
  struct is_error_code_enum<VULKAN_HPP_NAMESPACE::Result> : public true_type
  {};
}  // namespace std
#endif

namespace VULKAN_HPP_NAMESPACE
{
#ifndef VULKAN_HPP_NO_EXCEPTIONS

  class ErrorCategoryImpl : public std::error_category
  {
  public:
    virtual const char * name() const VULKAN_HPP_NOEXCEPT override
    {
      return VULKAN_HPP_NAMESPACE_STRING "::Result";
    }
    virtual std::string message( int ev ) const override
    {
      return to_string( static_cast<Result>( ev ) );
    }
  };

  class Error
  {
  public:
    Error() VULKAN_HPP_NOEXCEPT                = default;
    Error( const Error & ) VULKAN_HPP_NOEXCEPT = default;
    virtual ~Error() VULKAN_HPP_NOEXCEPT       = default;

    virtual const char * what() const VULKAN_HPP_NOEXCEPT = 0;
  };

  class LogicError
    : public Error
    , public std::logic_error
  {
  public:
    explicit LogicError( const std::string & what ) : Error(), std::logic_error( what ) {}
    explicit LogicError( char const * what ) : Error(), std::logic_error( what ) {}

    virtual const char * what() const VULKAN_HPP_NOEXCEPT
    {
      return std::logic_error::what();
    }
  };

  class SystemError
    : public Error
    , public std::system_error
  {
  public:
    SystemError( std::error_code ec ) : Error(), std::system_error( ec ) {}
    SystemError( std::error_code ec, std::string const & what ) : Error(), std::system_error( ec, what ) {}
    SystemError( std::error_code ec, char const * what ) : Error(), std::system_error( ec, what ) {}
    SystemError( int ev, std::error_category const & ecat ) : Error(), std::system_error( ev, ecat ) {}
    SystemError( int ev, std::error_category const & ecat, std::string const & what )
      : Error(), std::system_error( ev, ecat, what )
    {}
    SystemError( int ev, std::error_category const & ecat, char const * what )
      : Error(), std::system_error( ev, ecat, what )
    {}

    virtual const char * what() const VULKAN_HPP_NOEXCEPT
    {
      return std::system_error::what();
    }
  };

  VULKAN_HPP_INLINE const std::error_category & errorCategory() VULKAN_HPP_NOEXCEPT
  {
    static ErrorCategoryImpl instance;
    return instance;
  }

  VULKAN_HPP_INLINE std::error_code make_error_code( Result e ) VULKAN_HPP_NOEXCEPT
  {
    return std::error_code( static_cast<int>( e ), errorCategory() );
  }

  VULKAN_HPP_INLINE std::error_condition make_error_condition( Result e ) VULKAN_HPP_NOEXCEPT
  {
    return std::error_condition( static_cast<int>( e ), errorCategory() );
  }

  class OutOfHostMemoryError : public SystemError
  {
  public:
    OutOfHostMemoryError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorOutOfHostMemory ), message )
    {}
    OutOfHostMemoryError( char const * message )
      : SystemError( make_error_code( Result::eErrorOutOfHostMemory ), message )
    {}
  };

  class OutOfDeviceMemoryError : public SystemError
  {
  public:
    OutOfDeviceMemoryError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorOutOfDeviceMemory ), message )
    {}
    OutOfDeviceMemoryError( char const * message )
      : SystemError( make_error_code( Result::eErrorOutOfDeviceMemory ), message )
    {}
  };

  class InitializationFailedError : public SystemError
  {
  public:
    InitializationFailedError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorInitializationFailed ), message )
    {}
    InitializationFailedError( char const * message )
      : SystemError( make_error_code( Result::eErrorInitializationFailed ), message )
    {}
  };

  class DeviceLostError : public SystemError
  {
  public:
    DeviceLostError( std::string const & message ) : SystemError( make_error_code( Result::eErrorDeviceLost ), message )
    {}
    DeviceLostError( char const * message ) : SystemError( make_error_code( Result::eErrorDeviceLost ), message ) {}
  };

  class MemoryMapFailedError : public SystemError
  {
  public:
    MemoryMapFailedError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorMemoryMapFailed ), message )
    {}
    MemoryMapFailedError( char const * message )
      : SystemError( make_error_code( Result::eErrorMemoryMapFailed ), message )
    {}
  };

  class LayerNotPresentError : public SystemError
  {
  public:
    LayerNotPresentError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorLayerNotPresent ), message )
    {}
    LayerNotPresentError( char const * message )
      : SystemError( make_error_code( Result::eErrorLayerNotPresent ), message )
    {}
  };

  class ExtensionNotPresentError : public SystemError
  {
  public:
    ExtensionNotPresentError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorExtensionNotPresent ), message )
    {}
    ExtensionNotPresentError( char const * message )
      : SystemError( make_error_code( Result::eErrorExtensionNotPresent ), message )
    {}
  };

  class FeatureNotPresentError : public SystemError
  {
  public:
    FeatureNotPresentError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorFeatureNotPresent ), message )
    {}
    FeatureNotPresentError( char const * message )
      : SystemError( make_error_code( Result::eErrorFeatureNotPresent ), message )
    {}
  };

  class IncompatibleDriverError : public SystemError
  {
  public:
    IncompatibleDriverError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorIncompatibleDriver ), message )
    {}
    IncompatibleDriverError( char const * message )
      : SystemError( make_error_code( Result::eErrorIncompatibleDriver ), message )
    {}
  };

  class TooManyObjectsError : public SystemError
  {
  public:
    TooManyObjectsError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorTooManyObjects ), message )
    {}
    TooManyObjectsError( char const * message )
      : SystemError( make_error_code( Result::eErrorTooManyObjects ), message )
    {}
  };

  class FormatNotSupportedError : public SystemError
  {
  public:
    FormatNotSupportedError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorFormatNotSupported ), message )
    {}
    FormatNotSupportedError( char const * message )
      : SystemError( make_error_code( Result::eErrorFormatNotSupported ), message )
    {}
  };

  class FragmentedPoolError : public SystemError
  {
  public:
    FragmentedPoolError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorFragmentedPool ), message )
    {}
    FragmentedPoolError( char const * message )
      : SystemError( make_error_code( Result::eErrorFragmentedPool ), message )
    {}
  };

  class UnknownError : public SystemError
  {
  public:
    UnknownError( std::string const & message ) : SystemError( make_error_code( Result::eErrorUnknown ), message ) {}
    UnknownError( char const * message ) : SystemError( make_error_code( Result::eErrorUnknown ), message ) {}
  };

  class OutOfPoolMemoryError : public SystemError
  {
  public:
    OutOfPoolMemoryError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorOutOfPoolMemory ), message )
    {}
    OutOfPoolMemoryError( char const * message )
      : SystemError( make_error_code( Result::eErrorOutOfPoolMemory ), message )
    {}
  };

  class InvalidExternalHandleError : public SystemError
  {
  public:
    InvalidExternalHandleError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorInvalidExternalHandle ), message )
    {}
    InvalidExternalHandleError( char const * message )
      : SystemError( make_error_code( Result::eErrorInvalidExternalHandle ), message )
    {}
  };

  class FragmentationError : public SystemError
  {
  public:
    FragmentationError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorFragmentation ), message )
    {}
    FragmentationError( char const * message ) : SystemError( make_error_code( Result::eErrorFragmentation ), message )
    {}
  };

  class InvalidOpaqueCaptureAddressError : public SystemError
  {
  public:
    InvalidOpaqueCaptureAddressError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorInvalidOpaqueCaptureAddress ), message )
    {}
    InvalidOpaqueCaptureAddressError( char const * message )
      : SystemError( make_error_code( Result::eErrorInvalidOpaqueCaptureAddress ), message )
    {}
  };

  class SurfaceLostKHRError : public SystemError
  {
  public:
    SurfaceLostKHRError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorSurfaceLostKHR ), message )
    {}
    SurfaceLostKHRError( char const * message )
      : SystemError( make_error_code( Result::eErrorSurfaceLostKHR ), message )
    {}
  };

  class NativeWindowInUseKHRError : public SystemError
  {
  public:
    NativeWindowInUseKHRError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorNativeWindowInUseKHR ), message )
    {}
    NativeWindowInUseKHRError( char const * message )
      : SystemError( make_error_code( Result::eErrorNativeWindowInUseKHR ), message )
    {}
  };

  class OutOfDateKHRError : public SystemError
  {
  public:
    OutOfDateKHRError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorOutOfDateKHR ), message )
    {}
    OutOfDateKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorOutOfDateKHR ), message ) {}
  };

  class IncompatibleDisplayKHRError : public SystemError
  {
  public:
    IncompatibleDisplayKHRError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorIncompatibleDisplayKHR ), message )
    {}
    IncompatibleDisplayKHRError( char const * message )
      : SystemError( make_error_code( Result::eErrorIncompatibleDisplayKHR ), message )
    {}
  };

  class ValidationFailedEXTError : public SystemError
  {
  public:
    ValidationFailedEXTError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorValidationFailedEXT ), message )
    {}
    ValidationFailedEXTError( char const * message )
      : SystemError( make_error_code( Result::eErrorValidationFailedEXT ), message )
    {}
  };

  class InvalidShaderNVError : public SystemError
  {
  public:
    InvalidShaderNVError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorInvalidShaderNV ), message )
    {}
    InvalidShaderNVError( char const * message )
      : SystemError( make_error_code( Result::eErrorInvalidShaderNV ), message )
    {}
  };

  class InvalidDrmFormatModifierPlaneLayoutEXTError : public SystemError
  {
  public:
    InvalidDrmFormatModifierPlaneLayoutEXTError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT ), message )
    {}
    InvalidDrmFormatModifierPlaneLayoutEXTError( char const * message )
      : SystemError( make_error_code( Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT ), message )
    {}
  };

  class NotPermittedEXTError : public SystemError
  {
  public:
    NotPermittedEXTError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorNotPermittedEXT ), message )
    {}
    NotPermittedEXTError( char const * message )
      : SystemError( make_error_code( Result::eErrorNotPermittedEXT ), message )
    {}
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  class FullScreenExclusiveModeLostEXTError : public SystemError
  {
  public:
    FullScreenExclusiveModeLostEXTError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorFullScreenExclusiveModeLostEXT ), message )
    {}
    FullScreenExclusiveModeLostEXTError( char const * message )
      : SystemError( make_error_code( Result::eErrorFullScreenExclusiveModeLostEXT ), message )
    {}
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  [[noreturn]] static void throwResultException( Result result, char const * message )
  {
    switch ( result )
    {
      case Result::eErrorOutOfHostMemory: throw OutOfHostMemoryError( message );
      case Result::eErrorOutOfDeviceMemory: throw OutOfDeviceMemoryError( message );
      case Result::eErrorInitializationFailed: throw InitializationFailedError( message );
      case Result::eErrorDeviceLost: throw DeviceLostError( message );
      case Result::eErrorMemoryMapFailed: throw MemoryMapFailedError( message );
      case Result::eErrorLayerNotPresent: throw LayerNotPresentError( message );
      case Result::eErrorExtensionNotPresent: throw ExtensionNotPresentError( message );
      case Result::eErrorFeatureNotPresent: throw FeatureNotPresentError( message );
      case Result::eErrorIncompatibleDriver: throw IncompatibleDriverError( message );
      case Result::eErrorTooManyObjects: throw TooManyObjectsError( message );
      case Result::eErrorFormatNotSupported: throw FormatNotSupportedError( message );
      case Result::eErrorFragmentedPool: throw FragmentedPoolError( message );
      case Result::eErrorUnknown: throw UnknownError( message );
      case Result::eErrorOutOfPoolMemory: throw OutOfPoolMemoryError( message );
      case Result::eErrorInvalidExternalHandle: throw InvalidExternalHandleError( message );
      case Result::eErrorFragmentation: throw FragmentationError( message );
      case Result::eErrorInvalidOpaqueCaptureAddress: throw InvalidOpaqueCaptureAddressError( message );
      case Result::eErrorSurfaceLostKHR: throw SurfaceLostKHRError( message );
      case Result::eErrorNativeWindowInUseKHR: throw NativeWindowInUseKHRError( message );
      case Result::eErrorOutOfDateKHR: throw OutOfDateKHRError( message );
      case Result::eErrorIncompatibleDisplayKHR: throw IncompatibleDisplayKHRError( message );
      case Result::eErrorValidationFailedEXT: throw ValidationFailedEXTError( message );
      case Result::eErrorInvalidShaderNV: throw InvalidShaderNVError( message );
      case Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT:
        throw InvalidDrmFormatModifierPlaneLayoutEXTError( message );
      case Result::eErrorNotPermittedEXT: throw NotPermittedEXTError( message );
#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      case Result::eErrorFullScreenExclusiveModeLostEXT: throw FullScreenExclusiveModeLostEXTError( message );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/
      default: throw SystemError( make_error_code( result ) );
    }
  }
#endif

  template <typename T>
  void ignore( T const & ) VULKAN_HPP_NOEXCEPT
  {}

  template <typename T>
  struct ResultValue
  {
#ifdef VULKAN_HPP_HAS_NOEXCEPT
    ResultValue( Result r, T & v ) VULKAN_HPP_NOEXCEPT( VULKAN_HPP_NOEXCEPT( T( v ) ) )
#else
    ResultValue( Result r, T & v )
#endif
      : result( r ), value( v )
    {}

#ifdef VULKAN_HPP_HAS_NOEXCEPT
    ResultValue( Result r, T && v ) VULKAN_HPP_NOEXCEPT( VULKAN_HPP_NOEXCEPT( T( std::move( v ) ) ) )
#else
    ResultValue( Result r, T && v )
#endif
      : result( r ), value( std::move( v ) )
    {}

    Result result;
    T      value;

    operator std::tuple<Result &, T &>() VULKAN_HPP_NOEXCEPT
    {
      return std::tuple<Result &, T &>( result, value );
    }

#if !defined( VULKAN_HPP_DISABLE_IMPLICIT_RESULT_VALUE_CAST )
    VULKAN_HPP_DEPRECATED(
      "Implicit-cast operators on vk::ResultValue are deprecated. Explicitly access the value as member of ResultValue." )
    operator T const &() const & VULKAN_HPP_NOEXCEPT
    {
      return value;
    }

    VULKAN_HPP_DEPRECATED(
      "Implicit-cast operators on vk::ResultValue are deprecated. Explicitly access the value as member of ResultValue." )
    operator T &() & VULKAN_HPP_NOEXCEPT
    {
      return value;
    }

    VULKAN_HPP_DEPRECATED(
      "Implicit-cast operators on vk::ResultValue are deprecated. Explicitly access the value as member of ResultValue." )
    operator T const &&() const && VULKAN_HPP_NOEXCEPT
    {
      return std::move( value );
    }

    VULKAN_HPP_DEPRECATED(
      "Implicit-cast operators on vk::ResultValue are deprecated. Explicitly access the value as member of ResultValue." )
    operator T &&() && VULKAN_HPP_NOEXCEPT
    {
      return std::move( value );
    }
#endif
  };

#if !defined( VULKAN_HPP_NO_SMART_HANDLE )
  template <typename Type, typename Dispatch>
  struct ResultValue<UniqueHandle<Type, Dispatch>>
  {
#  ifdef VULKAN_HPP_HAS_NOEXCEPT
    ResultValue( Result r, UniqueHandle<Type, Dispatch> && v ) VULKAN_HPP_NOEXCEPT
#  else
    ResultValue( Result r, UniqueHandle<Type, Dispatch> && v )
#  endif
      : result( r )
      , value( std::move( v ) )
    {}

    std::tuple<Result, UniqueHandle<Type, Dispatch>> asTuple()
    {
      return std::make_tuple( result, std::move( value ) );
    }

#  if !defined( VULKAN_HPP_DISABLE_IMPLICIT_RESULT_VALUE_CAST )
    VULKAN_HPP_DEPRECATED(
      "Implicit-cast operators on vk::ResultValue are deprecated. Explicitly access the value as member of ResultValue." )
    operator UniqueHandle<Type, Dispatch> &() & VULKAN_HPP_NOEXCEPT
    {
      return value;
    }

    VULKAN_HPP_DEPRECATED(
      "Implicit-cast operators on vk::ResultValue are deprecated. Explicitly access the value as member of ResultValue." )
    operator UniqueHandle<Type, Dispatch>() VULKAN_HPP_NOEXCEPT
    {
      return std::move( value );
    }
#  endif

    Result                       result;
    UniqueHandle<Type, Dispatch> value;
  };

  template <typename Type, typename Dispatch>
  struct ResultValue<std::vector<UniqueHandle<Type, Dispatch>>>
  {
#  ifdef VULKAN_HPP_HAS_NOEXCEPT
    ResultValue( Result r, std::vector<UniqueHandle<Type, Dispatch>> && v ) VULKAN_HPP_NOEXCEPT
#  else
    ResultValue( Result r, std::vector<UniqueHandle<Type, Dispatch>> && v )
#  endif
      : result( r )
      , value( std::move( v ) )
    {}

    std::tuple<Result, std::vector<UniqueHandle<Type, Dispatch>>> asTuple()
    {
      return std::make_tuple( result, std::move( value ) );
    }

    Result                                    result;
    std::vector<UniqueHandle<Type, Dispatch>> value;

#  if !defined( VULKAN_HPP_DISABLE_IMPLICIT_RESULT_VALUE_CAST )
    VULKAN_HPP_DEPRECATED(
      "Implicit-cast operators on vk::ResultValue are deprecated. Explicitly access the value as member of ResultValue." )
    operator std::tuple<Result &, std::vector<UniqueHandle<Type, Dispatch>> &>() VULKAN_HPP_NOEXCEPT
    {
      return std::tuple<Result &, std::vector<UniqueHandle<Type, Dispatch>> &>( result, value );
    }
#  endif
  };
#endif

  template <typename T>
  struct ResultValueType
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    typedef ResultValue<T> type;
#else
    typedef T    type;
#endif
  };

  template <>
  struct ResultValueType<void>
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    typedef Result type;
#else
    typedef void type;
#endif
  };

  VULKAN_HPP_INLINE ResultValueType<void>::type createResultValue( Result result, char const * message )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( message );
    VULKAN_HPP_ASSERT_ON_RESULT( result == Result::eSuccess );
    return result;
#else
    if ( result != Result::eSuccess )
    {
      throwResultException( result, message );
    }
#endif
  }

  template <typename T>
  VULKAN_HPP_INLINE typename ResultValueType<T>::type createResultValue( Result result, T & data, char const * message )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( message );
    VULKAN_HPP_ASSERT_ON_RESULT( result == Result::eSuccess );
    return ResultValue<T>( result, std::move( data ) );
#else
    if ( result != Result::eSuccess )
    {
      throwResultException( result, message );
    }
    return std::move( data );
#endif
  }

  VULKAN_HPP_INLINE Result createResultValue( Result                        result,
                                              char const *                  message,
                                              std::initializer_list<Result> successCodes )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( message );
    ignore( successCodes );  // just in case VULKAN_HPP_ASSERT_ON_RESULT is empty
    VULKAN_HPP_ASSERT_ON_RESULT( std::find( successCodes.begin(), successCodes.end(), result ) != successCodes.end() );
#else
    if ( std::find( successCodes.begin(), successCodes.end(), result ) == successCodes.end() )
    {
      throwResultException( result, message );
    }
#endif
    return result;
  }

  template <typename T>
  VULKAN_HPP_INLINE ResultValue<T>
                    createResultValue( Result result, T & data, char const * message, std::initializer_list<Result> successCodes )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( message );
    ignore( successCodes );  // just in case VULKAN_HPP_ASSERT_ON_RESULT is empty
    VULKAN_HPP_ASSERT_ON_RESULT( std::find( successCodes.begin(), successCodes.end(), result ) != successCodes.end() );
#else
    if ( std::find( successCodes.begin(), successCodes.end(), result ) == successCodes.end() )
    {
      throwResultException( result, message );
    }
#endif
    return ResultValue<T>( result, std::move( data ) );
  }

#ifndef VULKAN_HPP_NO_SMART_HANDLE
  template <typename T, typename D>
  VULKAN_HPP_INLINE typename ResultValueType<UniqueHandle<T, D>>::type createResultValue(
    Result result, T & data, char const * message, typename UniqueHandleTraits<T, D>::deleter const & deleter )
  {
#  ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( message );
    VULKAN_HPP_ASSERT_ON_RESULT( result == Result::eSuccess );
    return ResultValue<UniqueHandle<T, D>>( result, UniqueHandle<T, D>( data, deleter ) );
#  else
    if ( result != Result::eSuccess )
    {
      throwResultException( result, message );
    }
    return UniqueHandle<T, D>( data, deleter );
#  endif
  }

  template <typename T, typename D>
  VULKAN_HPP_INLINE ResultValue<UniqueHandle<T, D>>
                    createResultValue( Result                                             result,
                                       T &                                                data,
                                       char const *                                       message,
                                       std::initializer_list<Result>                      successCodes,
                                       typename UniqueHandleTraits<T, D>::deleter const & deleter )
  {
#  ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( message );
    ignore( successCodes );  // just in case VULKAN_HPP_ASSERT_ON_RESULT is empty
    VULKAN_HPP_ASSERT_ON_RESULT( std::find( successCodes.begin(), successCodes.end(), result ) != successCodes.end() );
#  else
    if ( std::find( successCodes.begin(), successCodes.end(), result ) == successCodes.end() )
    {
      throwResultException( result, message );
    }
#  endif
    return ResultValue<UniqueHandle<T, D>>( result, UniqueHandle<T, D>( data, deleter ) );
  }

  template <typename T, typename D>
  VULKAN_HPP_INLINE typename ResultValueType<std::vector<UniqueHandle<T, D>>>::type
    createResultValue( Result result, std::vector<UniqueHandle<T, D>> && data, char const * message )
  {
#  ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( message );
    VULKAN_HPP_ASSERT_ON_RESULT( result == Result::eSuccess );
    return ResultValue<std::vector<UniqueHandle<T, D>>>( result, std::move( data ) );
#  else
    if ( result != Result::eSuccess )
    {
      throwResultException( result, message );
    }
    return std::move( data );
#  endif
  }

  template <typename T, typename D>
  VULKAN_HPP_INLINE ResultValue<std::vector<UniqueHandle<T, D>>>
                    createResultValue( Result                             result,
                                       std::vector<UniqueHandle<T, D>> && data,
                                       char const *                       message,
                                       std::initializer_list<Result>      successCodes )
  {
#  ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( message );
    ignore( successCodes );  // just in case VULKAN_HPP_ASSERT_ON_RESULT is empty
    VULKAN_HPP_ASSERT_ON_RESULT( std::find( successCodes.begin(), successCodes.end(), result ) != successCodes.end() );
#  else
    if ( std::find( successCodes.begin(), successCodes.end(), result ) == successCodes.end() )
    {
      throwResultException( result, message );
    }
#  endif
    return ResultValue<std::vector<UniqueHandle<T, D>>>( result, std::move( data ) );
  }
#endif
}  // namespace VULKAN_HPP_NAMESPACE

// clang-format off
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_funcs.hpp>
// clang-format on

namespace VULKAN_HPP_NAMESPACE
{
  template <>
  struct StructExtends<AccelerationStructureGeometryMotionTrianglesDataNV,
                       AccelerationStructureGeometryTrianglesDataKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<AccelerationStructureMotionInfoNV, AccelerationStructureCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct StructExtends<AndroidHardwareBufferFormatPropertiesANDROID, AndroidHardwareBufferPropertiesANDROID>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct StructExtends<AndroidHardwareBufferUsageANDROID, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
  template <>
  struct StructExtends<AttachmentDescriptionStencilLayout, AttachmentDescription2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<AttachmentReferenceStencilLayout, AttachmentReference2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<BindBufferMemoryDeviceGroupInfo, BindBufferMemoryInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<BindImageMemoryDeviceGroupInfo, BindImageMemoryInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<BindImageMemorySwapchainInfoKHR, BindImageMemoryInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<BindImagePlaneMemoryInfo, BindImageMemoryInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<BufferDeviceAddressCreateInfoEXT, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<BufferOpaqueCaptureAddressCreateInfo, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<CommandBufferInheritanceConditionalRenderingInfoEXT, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<CommandBufferInheritanceRenderPassTransformInfoQCOM, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<CommandBufferInheritanceViewportScissorInfoNV, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<CopyCommandTransformInfoQCOM, BufferImageCopy2KHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<CopyCommandTransformInfoQCOM, ImageBlit2KHR>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<D3D12FenceSubmitInfoKHR, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
  template <>
  struct StructExtends<DebugReportCallbackCreateInfoEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DebugUtilsMessengerCreateInfoEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DedicatedAllocationBufferCreateInfoNV, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DedicatedAllocationImageCreateInfoNV, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DedicatedAllocationMemoryAllocateInfoNV, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DescriptorPoolInlineUniformBlockCreateInfoEXT, DescriptorPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DescriptorSetLayoutBindingFlagsCreateInfo, DescriptorSetLayoutCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DescriptorSetVariableDescriptorCountAllocateInfo, DescriptorSetAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DescriptorSetVariableDescriptorCountLayoutSupport, DescriptorSetLayoutSupport>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceDeviceMemoryReportCreateInfoEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceDiagnosticsConfigCreateInfoNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceGroupBindSparseInfo, BindSparseInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceGroupCommandBufferBeginInfo, CommandBufferBeginInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceGroupDeviceCreateInfo, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceGroupPresentInfoKHR, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceGroupRenderPassBeginInfo, RenderPassBeginInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceGroupSubmitInfo, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceGroupSwapchainCreateInfoKHR, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceMemoryOverallocationCreateInfoAMD, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DevicePrivateDataCreateInfoEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DeviceQueueGlobalPriorityCreateInfoEXT, DeviceQueueCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DisplayNativeHdrSurfaceCapabilitiesAMD, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DisplayPresentInfoKHR, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<DrmFormatModifierPropertiesListEXT, FormatProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ExportFenceCreateInfo, FenceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<ExportFenceWin32HandleInfoKHR, FenceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
  template <>
  struct StructExtends<ExportMemoryAllocateInfo, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ExportMemoryAllocateInfoNV, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<ExportMemoryWin32HandleInfoKHR, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<ExportMemoryWin32HandleInfoNV, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
  template <>
  struct StructExtends<ExportSemaphoreCreateInfo, SemaphoreCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<ExportSemaphoreWin32HandleInfoKHR, SemaphoreCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct StructExtends<ExternalFormatANDROID, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ExternalFormatANDROID, SamplerYcbcrConversionCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
  template <>
  struct StructExtends<ExternalImageFormatProperties, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ExternalMemoryBufferCreateInfo, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ExternalMemoryImageCreateInfo, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ExternalMemoryImageCreateInfoNV, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<FilterCubicImageViewImageFormatPropertiesEXT, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<FragmentShadingRateAttachmentInfoKHR, SubpassDescription2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<FramebufferAttachmentsCreateInfo, FramebufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<GraphicsPipelineShaderGroupsCreateInfoNV, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageDrmFormatModifierExplicitCreateInfoEXT, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageDrmFormatModifierListCreateInfoEXT, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageFormatListCreateInfo, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageFormatListCreateInfo, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageFormatListCreateInfo, PhysicalDeviceImageFormatInfo2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImagePlaneMemoryRequirementsInfo, ImageMemoryRequirementsInfo2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageStencilUsageCreateInfo, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageStencilUsageCreateInfo, PhysicalDeviceImageFormatInfo2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageSwapchainCreateInfoKHR, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageViewASTCDecodeModeEXT, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImageViewUsageCreateInfo, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct StructExtends<ImportAndroidHardwareBufferInfoANDROID, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
  template <>
  struct StructExtends<ImportMemoryFdInfoKHR, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ImportMemoryHostPointerInfoEXT, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<ImportMemoryWin32HandleInfoKHR, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<ImportMemoryWin32HandleInfoNV, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct StructExtends<ImportMemoryZirconHandleInfoFUCHSIA, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_FUCHSIA*/
  template <>
  struct StructExtends<MemoryAllocateFlagsInfo, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<MemoryBarrier2KHR, SubpassDependency2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<MemoryDedicatedAllocateInfo, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<MemoryDedicatedRequirements, MemoryRequirements2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<MemoryOpaqueCaptureAddressAllocateInfo, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<MemoryPriorityAllocateInfoEXT, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<MutableDescriptorTypeCreateInfoVALVE, DescriptorSetLayoutCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<MutableDescriptorTypeCreateInfoVALVE, DescriptorPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PerformanceQuerySubmitInfoKHR, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PerformanceQuerySubmitInfoKHR, SubmitInfo2KHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevice16BitStorageFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevice16BitStorageFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevice4444FormatsFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevice4444FormatsFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevice8BitStorageFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevice8BitStorageFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceASTCDecodeFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceASTCDecodeFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceAccelerationStructureFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceAccelerationStructureFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceAccelerationStructurePropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceBlendOperationAdvancedFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceBlendOperationAdvancedFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceBlendOperationAdvancedPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceBufferDeviceAddressFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceBufferDeviceAddressFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceBufferDeviceAddressFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceBufferDeviceAddressFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCoherentMemoryFeaturesAMD, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCoherentMemoryFeaturesAMD, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceColorWriteEnableFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceColorWriteEnableFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceComputeShaderDerivativesFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceComputeShaderDerivativesFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceConditionalRenderingFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceConditionalRenderingFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceConservativeRasterizationPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCooperativeMatrixFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCooperativeMatrixFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCooperativeMatrixPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCornerSampledImageFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCornerSampledImageFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCoverageReductionModeFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCoverageReductionModeFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCustomBorderColorFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCustomBorderColorFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceCustomBorderColorPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDepthClipEnableFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDepthClipEnableFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDepthStencilResolveProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDescriptorIndexingFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDescriptorIndexingFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDescriptorIndexingProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDeviceGeneratedCommandsFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDeviceGeneratedCommandsFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDeviceGeneratedCommandsPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDeviceMemoryReportFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDeviceMemoryReportFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDiagnosticsConfigFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDiagnosticsConfigFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDiscardRectanglePropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDriverProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceDrmPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceExclusiveScissorFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceExclusiveScissorFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceExtendedDynamicState2FeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceExtendedDynamicState2FeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceExtendedDynamicStateFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceExtendedDynamicStateFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceExternalImageFormatInfo, PhysicalDeviceImageFormatInfo2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceExternalMemoryHostPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFeatures2, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFloatControlsProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMap2FeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMap2FeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMap2PropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMapFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMapFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMapPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShaderBarycentricFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShaderBarycentricFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShaderInterlockFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShaderInterlockFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShadingRateEnumsFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShadingRateEnumsFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShadingRateEnumsPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShadingRateFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShadingRateFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceFragmentShadingRatePropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceGlobalPriorityQueryFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceHostQueryResetFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceHostQueryResetFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceIDProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceImageDrmFormatModifierInfoEXT, PhysicalDeviceImageFormatInfo2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceImageRobustnessFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceImageRobustnessFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceImageViewImageFormatInfoEXT, PhysicalDeviceImageFormatInfo2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceImagelessFramebufferFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceImagelessFramebufferFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceIndexTypeUint8FeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceIndexTypeUint8FeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceInheritedViewportScissorFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceInheritedViewportScissorFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceInlineUniformBlockFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceInlineUniformBlockFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceInlineUniformBlockPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceLineRasterizationFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceLineRasterizationFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceLineRasterizationPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMaintenance3Properties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMemoryBudgetPropertiesEXT, PhysicalDeviceMemoryProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMemoryPriorityFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMemoryPriorityFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMeshShaderFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMeshShaderFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMeshShaderPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMultiDrawFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMultiDrawFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMultiDrawPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMultiviewFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMultiviewFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMultiviewProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMutableDescriptorTypeFeaturesVALVE, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceMutableDescriptorTypeFeaturesVALVE, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePCIBusInfoPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePerformanceQueryFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePerformanceQueryFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePerformanceQueryPropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePipelineCreationCacheControlFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePipelineCreationCacheControlFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePipelineExecutablePropertiesFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePipelineExecutablePropertiesFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePointClippingProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<PhysicalDevicePortabilitySubsetFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePortabilitySubsetFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<PhysicalDevicePortabilitySubsetPropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  template <>
  struct StructExtends<PhysicalDevicePrivateDataFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePrivateDataFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceProtectedMemoryFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceProtectedMemoryFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceProtectedMemoryProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceProvokingVertexFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceProvokingVertexFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceProvokingVertexPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDevicePushDescriptorPropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRayQueryFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRayQueryFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRayTracingMotionBlurFeaturesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRayTracingMotionBlurFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRayTracingPipelineFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRayTracingPipelineFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRayTracingPipelinePropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRayTracingPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRepresentativeFragmentTestFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRepresentativeFragmentTestFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRobustness2FeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRobustness2FeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceRobustness2PropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSampleLocationsPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSamplerFilterMinmaxProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSamplerYcbcrConversionFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSamplerYcbcrConversionFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceScalarBlockLayoutFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceScalarBlockLayoutFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSeparateDepthStencilLayoutsFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSeparateDepthStencilLayoutsFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderAtomicFloatFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderAtomicFloatFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderAtomicInt64Features, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderAtomicInt64Features, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderClockFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderClockFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderCoreProperties2AMD, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderCorePropertiesAMD, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderDrawParametersFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderDrawParametersFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderFloat16Int8Features, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderFloat16Int8Features, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderImageAtomicInt64FeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderImageAtomicInt64FeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderImageFootprintFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderImageFootprintFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderSMBuiltinsFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderSMBuiltinsFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderSMBuiltinsPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderSubgroupExtendedTypesFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderSubgroupExtendedTypesFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderTerminateInvocationFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShaderTerminateInvocationFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShadingRateImageFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShadingRateImageFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceShadingRateImagePropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSubgroupProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSubgroupSizeControlFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSubgroupSizeControlFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSubgroupSizeControlPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSubpassShadingFeaturesHUAWEI, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSubpassShadingFeaturesHUAWEI, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSubpassShadingPropertiesHUAWEI, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSynchronization2FeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceSynchronization2FeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTexelBufferAlignmentFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTexelBufferAlignmentFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTexelBufferAlignmentPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTextureCompressionASTCHDRFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTextureCompressionASTCHDRFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTimelineSemaphoreFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTimelineSemaphoreFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTimelineSemaphoreProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTransformFeedbackFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTransformFeedbackFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceTransformFeedbackPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceUniformBufferStandardLayoutFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceUniformBufferStandardLayoutFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVariablePointersFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVariablePointersFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVertexAttributeDivisorFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVertexAttributeDivisorFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVertexAttributeDivisorPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVertexInputDynamicStateFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVertexInputDynamicStateFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVulkan11Features, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVulkan11Features, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVulkan11Properties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVulkan12Features, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVulkan12Features, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVulkan12Properties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVulkanMemoryModelFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceVulkanMemoryModelFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceYcbcrImageArraysFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceYcbcrImageArraysFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineColorBlendAdvancedStateCreateInfoEXT, PipelineColorBlendStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineColorWriteCreateInfoEXT, PipelineColorBlendStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCompilerControlCreateInfoAMD, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCompilerControlCreateInfoAMD, ComputePipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCoverageModulationStateCreateInfoNV, PipelineMultisampleStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCoverageReductionStateCreateInfoNV, PipelineMultisampleStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCoverageToColorStateCreateInfoNV, PipelineMultisampleStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfoEXT, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfoEXT, ComputePipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfoEXT, RayTracingPipelineCreateInfoNV>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfoEXT, RayTracingPipelineCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineDiscardRectangleStateCreateInfoEXT, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineFragmentShadingRateEnumStateCreateInfoNV, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineFragmentShadingRateStateCreateInfoKHR, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineRasterizationConservativeStateCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineRasterizationDepthClipStateCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineRasterizationLineStateCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineRasterizationProvokingVertexStateCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineRasterizationStateRasterizationOrderAMD, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineRasterizationStateStreamCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineRepresentativeFragmentTestStateCreateInfoNV, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineSampleLocationsStateCreateInfoEXT, PipelineMultisampleStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT, PipelineShaderStageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineTessellationDomainOriginStateCreateInfo, PipelineTessellationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineVertexInputDivisorStateCreateInfoEXT, PipelineVertexInputStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineViewportCoarseSampleOrderStateCreateInfoNV, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineViewportExclusiveScissorStateCreateInfoNV, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineViewportShadingRateImageStateCreateInfoNV, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineViewportSwizzleStateCreateInfoNV, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PipelineViewportWScalingStateCreateInfoNV, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_GGP )
  template <>
  struct StructExtends<PresentFrameTokenGGP, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_GGP*/
  template <>
  struct StructExtends<PresentRegionsKHR, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<PresentTimesInfoGOOGLE, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ProtectedSubmitInfo, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<QueryPoolPerformanceCreateInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<QueryPoolPerformanceQueryCreateInfoINTEL, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<QueueFamilyCheckpointProperties2NV, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<QueueFamilyCheckpointPropertiesNV, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<QueueFamilyGlobalPriorityPropertiesEXT, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<RenderPassAttachmentBeginInfo, RenderPassBeginInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<RenderPassFragmentDensityMapCreateInfoEXT, RenderPassCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<RenderPassFragmentDensityMapCreateInfoEXT, RenderPassCreateInfo2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<RenderPassInputAttachmentAspectCreateInfo, RenderPassCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<RenderPassMultiviewCreateInfo, RenderPassCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<RenderPassSampleLocationsBeginInfoEXT, RenderPassBeginInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<RenderPassTransformBeginInfoQCOM, RenderPassBeginInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SampleLocationsInfoEXT, ImageMemoryBarrier>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SampleLocationsInfoEXT, ImageMemoryBarrier2KHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SamplerCustomBorderColorCreateInfoEXT, SamplerCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SamplerReductionModeCreateInfo, SamplerCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SamplerYcbcrConversionImageFormatProperties, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SamplerYcbcrConversionInfo, SamplerCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SamplerYcbcrConversionInfo, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SemaphoreTypeCreateInfo, SemaphoreCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SemaphoreTypeCreateInfo, PhysicalDeviceExternalSemaphoreInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ShaderModuleValidationCacheCreateInfoEXT, ShaderModuleCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SharedPresentSurfaceCapabilitiesKHR, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SubpassDescriptionDepthStencilResolve, SubpassDescription2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SubpassShadingPipelineCreateInfoHUAWEI, ComputePipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<SurfaceCapabilitiesFullScreenExclusiveEXT, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<SurfaceFullScreenExclusiveInfoEXT, PhysicalDeviceSurfaceInfo2KHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SurfaceFullScreenExclusiveInfoEXT, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<SurfaceFullScreenExclusiveWin32InfoEXT, PhysicalDeviceSurfaceInfo2KHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SurfaceFullScreenExclusiveWin32InfoEXT, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
  template <>
  struct StructExtends<SurfaceProtectedCapabilitiesKHR, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SwapchainCounterCreateInfoEXT, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<SwapchainDisplayNativeHdrCreateInfoAMD, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<TextureLODGatherFormatPropertiesAMD, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<TimelineSemaphoreSubmitInfo, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<TimelineSemaphoreSubmitInfo, BindSparseInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ValidationFeaturesEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<ValidationFlagsEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH264CapabilitiesEXT, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH264DpbSlotInfoEXT, VideoReferenceSlotKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH264MvcEXT, VideoDecodeH264PictureInfoEXT>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH264PictureInfoEXT, VideoDecodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH264ProfileEXT, VideoProfileKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH264SessionCreateInfoEXT, VideoSessionCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH264SessionParametersAddInfoEXT, VideoSessionParametersUpdateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH264SessionParametersCreateInfoEXT, VideoSessionParametersCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH265CapabilitiesEXT, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH265DpbSlotInfoEXT, VideoReferenceSlotKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH265PictureInfoEXT, VideoDecodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH265ProfileEXT, VideoProfileKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH265SessionCreateInfoEXT, VideoSessionCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH265SessionParametersAddInfoEXT, VideoSessionParametersUpdateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoDecodeH265SessionParametersCreateInfoEXT, VideoSessionParametersCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoEncodeH264CapabilitiesEXT, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoEncodeH264EmitPictureParametersEXT, VideoEncodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoEncodeH264ProfileEXT, VideoProfileKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoEncodeH264SessionCreateInfoEXT, VideoSessionCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoEncodeH264SessionParametersAddInfoEXT, VideoSessionParametersUpdateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoEncodeH264SessionParametersCreateInfoEXT, VideoSessionParametersCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoEncodeH264VclFrameInfoEXT, VideoEncodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoEncodeRateControlInfoKHR, VideoCodingControlInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoProfileKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<VideoProfileKHR, FormatProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<VideoProfileKHR, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<VideoProfileKHR, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<VideoProfileKHR, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoProfilesKHR, FormatProperties2>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<VideoProfilesKHR, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<VideoProfilesKHR, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<VideoProfilesKHR, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<VideoQueueFamilyProperties2KHR, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<Win32KeyedMutexAcquireReleaseInfoKHR, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<Win32KeyedMutexAcquireReleaseInfoKHR, SubmitInfo2KHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct StructExtends<Win32KeyedMutexAcquireReleaseInfoNV, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<Win32KeyedMutexAcquireReleaseInfoNV, SubmitInfo2KHR>
  {
    enum
    {
      value = true
    };
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
  template <>
  struct StructExtends<WriteDescriptorSetAccelerationStructureKHR, WriteDescriptorSet>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<WriteDescriptorSetAccelerationStructureNV, WriteDescriptorSet>
  {
    enum
    {
      value = true
    };
  };
  template <>
  struct StructExtends<WriteDescriptorSetInlineUniformBlockEXT, WriteDescriptorSet>
  {
    enum
    {
      value = true
    };
  };

#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
  class DynamicLoader
  {
  public:
#  ifdef VULKAN_HPP_NO_EXCEPTIONS
    DynamicLoader( std::string const & vulkanLibraryName = {} ) VULKAN_HPP_NOEXCEPT
#  else
    DynamicLoader( std::string const & vulkanLibraryName = {} )
#  endif
    {
      if ( !vulkanLibraryName.empty() )
      {
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNXNTO__ ) || defined( __Fuchsia__ )
        m_library = dlopen( vulkanLibraryName.c_str(), RTLD_NOW | RTLD_LOCAL );
#  elif defined( _WIN32 )
        m_library = ::LoadLibraryA( vulkanLibraryName.c_str() );
#  else
#    error unsupported platform
#  endif
      }
      else
      {
#  if defined( __unix__ ) || defined( __QNXNTO__ ) || defined( __Fuchsia__ )
        m_library = dlopen( "libvulkan.so", RTLD_NOW | RTLD_LOCAL );
        if ( m_library == nullptr )
        {
          m_library = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
        }
#  elif defined( __APPLE__ )
        m_library = dlopen( "libvulkan.dylib", RTLD_NOW | RTLD_LOCAL );
#  elif defined( _WIN32 )
        m_library = ::LoadLibraryA( "vulkan-1.dll" );
#  else
#    error unsupported platform
#  endif
      }

#  ifndef VULKAN_HPP_NO_EXCEPTIONS
      if ( m_library == nullptr )
      {
        // NOTE there should be an InitializationFailedError, but msvc insists on the symbol does not exist within the
        // scope of this function.
        throw std::runtime_error( "Failed to load vulkan library!" );
      }
#  endif
    }

    DynamicLoader( DynamicLoader const & ) = delete;

    DynamicLoader( DynamicLoader && other ) VULKAN_HPP_NOEXCEPT : m_library( other.m_library )
    {
      other.m_library = nullptr;
    }

    DynamicLoader & operator=( DynamicLoader const & ) = delete;

    DynamicLoader & operator=( DynamicLoader && other ) VULKAN_HPP_NOEXCEPT
    {
      std::swap( m_library, other.m_library );
      return *this;
    }

    ~DynamicLoader() VULKAN_HPP_NOEXCEPT
    {
      if ( m_library )
      {
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNXNTO__ ) || defined( __Fuchsia__ )
        dlclose( m_library );
#  elif defined( _WIN32 )
        ::FreeLibrary( m_library );
#  else
#    error unsupported platform
#  endif
      }
    }

    template <typename T>
    T getProcAddress( const char * function ) const VULKAN_HPP_NOEXCEPT
    {
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNXNTO__ ) || defined( __Fuchsia__ )
      return (T)dlsym( m_library, function );
#  elif defined( _WIN32 )
      return ( T )::GetProcAddress( m_library, function );
#  else
#    error unsupported platform
#  endif
    }

    bool success() const VULKAN_HPP_NOEXCEPT
    {
      return m_library != nullptr;
    }

  private:
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNXNTO__ ) || defined( __Fuchsia__ )
    void * m_library;
#  elif defined( _WIN32 )
    ::HINSTANCE m_library;
#  else
#    error unsupported platform
#  endif
  };
#endif

  class DispatchLoaderDynamic
  {
  public:
    using PFN_dummy = void ( * )();

    PFN_vkAcquireDrmDisplayEXT vkAcquireDrmDisplayEXT = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkAcquireFullScreenExclusiveModeEXT vkAcquireFullScreenExclusiveModeEXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkAcquireFullScreenExclusiveModeEXT               = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    PFN_vkAcquireNextImage2KHR                 vkAcquireNextImage2KHR                 = 0;
    PFN_vkAcquireNextImageKHR                  vkAcquireNextImageKHR                  = 0;
    PFN_vkAcquirePerformanceConfigurationINTEL vkAcquirePerformanceConfigurationINTEL = 0;
    PFN_vkAcquireProfilingLockKHR              vkAcquireProfilingLockKHR              = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkAcquireWinrtDisplayNV vkAcquireWinrtDisplayNV = 0;
#else
    PFN_dummy placeholder_dont_call_vkAcquireWinrtDisplayNV                           = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
    PFN_vkAcquireXlibDisplayEXT vkAcquireXlibDisplayEXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkAcquireXlibDisplayEXT                           = 0;
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/
    PFN_vkAllocateCommandBuffers            vkAllocateCommandBuffers            = 0;
    PFN_vkAllocateDescriptorSets            vkAllocateDescriptorSets            = 0;
    PFN_vkAllocateMemory                    vkAllocateMemory                    = 0;
    PFN_vkBeginCommandBuffer                vkBeginCommandBuffer                = 0;
    PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemoryNV = 0;
    PFN_vkBindBufferMemory                  vkBindBufferMemory                  = 0;
    PFN_vkBindBufferMemory2                 vkBindBufferMemory2                 = 0;
    PFN_vkBindBufferMemory2KHR              vkBindBufferMemory2KHR              = 0;
    PFN_vkBindImageMemory                   vkBindImageMemory                   = 0;
    PFN_vkBindImageMemory2                  vkBindImageMemory2                  = 0;
    PFN_vkBindImageMemory2KHR               vkBindImageMemory2KHR               = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkBindVideoSessionMemoryKHR vkBindVideoSessionMemoryKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkBindVideoSessionMemoryKHR                       = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    PFN_vkBuildAccelerationStructuresKHR  vkBuildAccelerationStructuresKHR  = 0;
    PFN_vkCmdBeginConditionalRenderingEXT vkCmdBeginConditionalRenderingEXT = 0;
    PFN_vkCmdBeginDebugUtilsLabelEXT      vkCmdBeginDebugUtilsLabelEXT      = 0;
    PFN_vkCmdBeginQuery                   vkCmdBeginQuery                   = 0;
    PFN_vkCmdBeginQueryIndexedEXT         vkCmdBeginQueryIndexedEXT         = 0;
    PFN_vkCmdBeginRenderPass              vkCmdBeginRenderPass              = 0;
    PFN_vkCmdBeginRenderPass2             vkCmdBeginRenderPass2             = 0;
    PFN_vkCmdBeginRenderPass2KHR          vkCmdBeginRenderPass2KHR          = 0;
    PFN_vkCmdBeginTransformFeedbackEXT    vkCmdBeginTransformFeedbackEXT    = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkCmdBeginVideoCodingKHR vkCmdBeginVideoCodingKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCmdBeginVideoCodingKHR                          = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    PFN_vkCmdBindDescriptorSets                     vkCmdBindDescriptorSets                     = 0;
    PFN_vkCmdBindIndexBuffer                        vkCmdBindIndexBuffer                        = 0;
    PFN_vkCmdBindPipeline                           vkCmdBindPipeline                           = 0;
    PFN_vkCmdBindPipelineShaderGroupNV              vkCmdBindPipelineShaderGroupNV              = 0;
    PFN_vkCmdBindShadingRateImageNV                 vkCmdBindShadingRateImageNV                 = 0;
    PFN_vkCmdBindTransformFeedbackBuffersEXT        vkCmdBindTransformFeedbackBuffersEXT        = 0;
    PFN_vkCmdBindVertexBuffers                      vkCmdBindVertexBuffers                      = 0;
    PFN_vkCmdBindVertexBuffers2EXT                  vkCmdBindVertexBuffers2EXT                  = 0;
    PFN_vkCmdBlitImage                              vkCmdBlitImage                              = 0;
    PFN_vkCmdBlitImage2KHR                          vkCmdBlitImage2KHR                          = 0;
    PFN_vkCmdBuildAccelerationStructureNV           vkCmdBuildAccelerationStructureNV           = 0;
    PFN_vkCmdBuildAccelerationStructuresIndirectKHR vkCmdBuildAccelerationStructuresIndirectKHR = 0;
    PFN_vkCmdBuildAccelerationStructuresKHR         vkCmdBuildAccelerationStructuresKHR         = 0;
    PFN_vkCmdClearAttachments                       vkCmdClearAttachments                       = 0;
    PFN_vkCmdClearColorImage                        vkCmdClearColorImage                        = 0;
    PFN_vkCmdClearDepthStencilImage                 vkCmdClearDepthStencilImage                 = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkCmdControlVideoCodingKHR vkCmdControlVideoCodingKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCmdControlVideoCodingKHR                        = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    PFN_vkCmdCopyAccelerationStructureKHR         vkCmdCopyAccelerationStructureKHR         = 0;
    PFN_vkCmdCopyAccelerationStructureNV          vkCmdCopyAccelerationStructureNV          = 0;
    PFN_vkCmdCopyAccelerationStructureToMemoryKHR vkCmdCopyAccelerationStructureToMemoryKHR = 0;
    PFN_vkCmdCopyBuffer                           vkCmdCopyBuffer                           = 0;
    PFN_vkCmdCopyBuffer2KHR                       vkCmdCopyBuffer2KHR                       = 0;
    PFN_vkCmdCopyBufferToImage                    vkCmdCopyBufferToImage                    = 0;
    PFN_vkCmdCopyBufferToImage2KHR                vkCmdCopyBufferToImage2KHR                = 0;
    PFN_vkCmdCopyImage                            vkCmdCopyImage                            = 0;
    PFN_vkCmdCopyImage2KHR                        vkCmdCopyImage2KHR                        = 0;
    PFN_vkCmdCopyImageToBuffer                    vkCmdCopyImageToBuffer                    = 0;
    PFN_vkCmdCopyImageToBuffer2KHR                vkCmdCopyImageToBuffer2KHR                = 0;
    PFN_vkCmdCopyMemoryToAccelerationStructureKHR vkCmdCopyMemoryToAccelerationStructureKHR = 0;
    PFN_vkCmdCopyQueryPoolResults                 vkCmdCopyQueryPoolResults                 = 0;
    PFN_vkCmdCuLaunchKernelNVX                    vkCmdCuLaunchKernelNVX                    = 0;
    PFN_vkCmdDebugMarkerBeginEXT                  vkCmdDebugMarkerBeginEXT                  = 0;
    PFN_vkCmdDebugMarkerEndEXT                    vkCmdDebugMarkerEndEXT                    = 0;
    PFN_vkCmdDebugMarkerInsertEXT                 vkCmdDebugMarkerInsertEXT                 = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkCmdDecodeVideoKHR vkCmdDecodeVideoKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCmdDecodeVideoKHR                               = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    PFN_vkCmdDispatch                     vkCmdDispatch                     = 0;
    PFN_vkCmdDispatchBase                 vkCmdDispatchBase                 = 0;
    PFN_vkCmdDispatchBaseKHR              vkCmdDispatchBaseKHR              = 0;
    PFN_vkCmdDispatchIndirect             vkCmdDispatchIndirect             = 0;
    PFN_vkCmdDraw                         vkCmdDraw                         = 0;
    PFN_vkCmdDrawIndexed                  vkCmdDrawIndexed                  = 0;
    PFN_vkCmdDrawIndexedIndirect          vkCmdDrawIndexedIndirect          = 0;
    PFN_vkCmdDrawIndexedIndirectCount     vkCmdDrawIndexedIndirectCount     = 0;
    PFN_vkCmdDrawIndexedIndirectCountAMD  vkCmdDrawIndexedIndirectCountAMD  = 0;
    PFN_vkCmdDrawIndexedIndirectCountKHR  vkCmdDrawIndexedIndirectCountKHR  = 0;
    PFN_vkCmdDrawIndirect                 vkCmdDrawIndirect                 = 0;
    PFN_vkCmdDrawIndirectByteCountEXT     vkCmdDrawIndirectByteCountEXT     = 0;
    PFN_vkCmdDrawIndirectCount            vkCmdDrawIndirectCount            = 0;
    PFN_vkCmdDrawIndirectCountAMD         vkCmdDrawIndirectCountAMD         = 0;
    PFN_vkCmdDrawIndirectCountKHR         vkCmdDrawIndirectCountKHR         = 0;
    PFN_vkCmdDrawMeshTasksIndirectCountNV vkCmdDrawMeshTasksIndirectCountNV = 0;
    PFN_vkCmdDrawMeshTasksIndirectNV      vkCmdDrawMeshTasksIndirectNV      = 0;
    PFN_vkCmdDrawMeshTasksNV              vkCmdDrawMeshTasksNV              = 0;
    PFN_vkCmdDrawMultiEXT                 vkCmdDrawMultiEXT                 = 0;
    PFN_vkCmdDrawMultiIndexedEXT          vkCmdDrawMultiIndexedEXT          = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkCmdEncodeVideoKHR vkCmdEncodeVideoKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCmdEncodeVideoKHR                               = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    PFN_vkCmdEndConditionalRenderingEXT vkCmdEndConditionalRenderingEXT = 0;
    PFN_vkCmdEndDebugUtilsLabelEXT      vkCmdEndDebugUtilsLabelEXT      = 0;
    PFN_vkCmdEndQuery                   vkCmdEndQuery                   = 0;
    PFN_vkCmdEndQueryIndexedEXT         vkCmdEndQueryIndexedEXT         = 0;
    PFN_vkCmdEndRenderPass              vkCmdEndRenderPass              = 0;
    PFN_vkCmdEndRenderPass2             vkCmdEndRenderPass2             = 0;
    PFN_vkCmdEndRenderPass2KHR          vkCmdEndRenderPass2KHR          = 0;
    PFN_vkCmdEndTransformFeedbackEXT    vkCmdEndTransformFeedbackEXT    = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkCmdEndVideoCodingKHR vkCmdEndVideoCodingKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCmdEndVideoCodingKHR                            = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    PFN_vkCmdExecuteCommands                          vkCmdExecuteCommands                          = 0;
    PFN_vkCmdExecuteGeneratedCommandsNV               vkCmdExecuteGeneratedCommandsNV               = 0;
    PFN_vkCmdFillBuffer                               vkCmdFillBuffer                               = 0;
    PFN_vkCmdInsertDebugUtilsLabelEXT                 vkCmdInsertDebugUtilsLabelEXT                 = 0;
    PFN_vkCmdNextSubpass                              vkCmdNextSubpass                              = 0;
    PFN_vkCmdNextSubpass2                             vkCmdNextSubpass2                             = 0;
    PFN_vkCmdNextSubpass2KHR                          vkCmdNextSubpass2KHR                          = 0;
    PFN_vkCmdPipelineBarrier                          vkCmdPipelineBarrier                          = 0;
    PFN_vkCmdPipelineBarrier2KHR                      vkCmdPipelineBarrier2KHR                      = 0;
    PFN_vkCmdPreprocessGeneratedCommandsNV            vkCmdPreprocessGeneratedCommandsNV            = 0;
    PFN_vkCmdPushConstants                            vkCmdPushConstants                            = 0;
    PFN_vkCmdPushDescriptorSetKHR                     vkCmdPushDescriptorSetKHR                     = 0;
    PFN_vkCmdPushDescriptorSetWithTemplateKHR         vkCmdPushDescriptorSetWithTemplateKHR         = 0;
    PFN_vkCmdResetEvent                               vkCmdResetEvent                               = 0;
    PFN_vkCmdResetEvent2KHR                           vkCmdResetEvent2KHR                           = 0;
    PFN_vkCmdResetQueryPool                           vkCmdResetQueryPool                           = 0;
    PFN_vkCmdResolveImage                             vkCmdResolveImage                             = 0;
    PFN_vkCmdResolveImage2KHR                         vkCmdResolveImage2KHR                         = 0;
    PFN_vkCmdSetBlendConstants                        vkCmdSetBlendConstants                        = 0;
    PFN_vkCmdSetCheckpointNV                          vkCmdSetCheckpointNV                          = 0;
    PFN_vkCmdSetCoarseSampleOrderNV                   vkCmdSetCoarseSampleOrderNV                   = 0;
    PFN_vkCmdSetColorWriteEnableEXT                   vkCmdSetColorWriteEnableEXT                   = 0;
    PFN_vkCmdSetCullModeEXT                           vkCmdSetCullModeEXT                           = 0;
    PFN_vkCmdSetDepthBias                             vkCmdSetDepthBias                             = 0;
    PFN_vkCmdSetDepthBiasEnableEXT                    vkCmdSetDepthBiasEnableEXT                    = 0;
    PFN_vkCmdSetDepthBounds                           vkCmdSetDepthBounds                           = 0;
    PFN_vkCmdSetDepthBoundsTestEnableEXT              vkCmdSetDepthBoundsTestEnableEXT              = 0;
    PFN_vkCmdSetDepthCompareOpEXT                     vkCmdSetDepthCompareOpEXT                     = 0;
    PFN_vkCmdSetDepthTestEnableEXT                    vkCmdSetDepthTestEnableEXT                    = 0;
    PFN_vkCmdSetDepthWriteEnableEXT                   vkCmdSetDepthWriteEnableEXT                   = 0;
    PFN_vkCmdSetDeviceMask                            vkCmdSetDeviceMask                            = 0;
    PFN_vkCmdSetDeviceMaskKHR                         vkCmdSetDeviceMaskKHR                         = 0;
    PFN_vkCmdSetDiscardRectangleEXT                   vkCmdSetDiscardRectangleEXT                   = 0;
    PFN_vkCmdSetEvent                                 vkCmdSetEvent                                 = 0;
    PFN_vkCmdSetEvent2KHR                             vkCmdSetEvent2KHR                             = 0;
    PFN_vkCmdSetExclusiveScissorNV                    vkCmdSetExclusiveScissorNV                    = 0;
    PFN_vkCmdSetFragmentShadingRateEnumNV             vkCmdSetFragmentShadingRateEnumNV             = 0;
    PFN_vkCmdSetFragmentShadingRateKHR                vkCmdSetFragmentShadingRateKHR                = 0;
    PFN_vkCmdSetFrontFaceEXT                          vkCmdSetFrontFaceEXT                          = 0;
    PFN_vkCmdSetLineStippleEXT                        vkCmdSetLineStippleEXT                        = 0;
    PFN_vkCmdSetLineWidth                             vkCmdSetLineWidth                             = 0;
    PFN_vkCmdSetLogicOpEXT                            vkCmdSetLogicOpEXT                            = 0;
    PFN_vkCmdSetPatchControlPointsEXT                 vkCmdSetPatchControlPointsEXT                 = 0;
    PFN_vkCmdSetPerformanceMarkerINTEL                vkCmdSetPerformanceMarkerINTEL                = 0;
    PFN_vkCmdSetPerformanceOverrideINTEL              vkCmdSetPerformanceOverrideINTEL              = 0;
    PFN_vkCmdSetPerformanceStreamMarkerINTEL          vkCmdSetPerformanceStreamMarkerINTEL          = 0;
    PFN_vkCmdSetPrimitiveRestartEnableEXT             vkCmdSetPrimitiveRestartEnableEXT             = 0;
    PFN_vkCmdSetPrimitiveTopologyEXT                  vkCmdSetPrimitiveTopologyEXT                  = 0;
    PFN_vkCmdSetRasterizerDiscardEnableEXT            vkCmdSetRasterizerDiscardEnableEXT            = 0;
    PFN_vkCmdSetRayTracingPipelineStackSizeKHR        vkCmdSetRayTracingPipelineStackSizeKHR        = 0;
    PFN_vkCmdSetSampleLocationsEXT                    vkCmdSetSampleLocationsEXT                    = 0;
    PFN_vkCmdSetScissor                               vkCmdSetScissor                               = 0;
    PFN_vkCmdSetScissorWithCountEXT                   vkCmdSetScissorWithCountEXT                   = 0;
    PFN_vkCmdSetStencilCompareMask                    vkCmdSetStencilCompareMask                    = 0;
    PFN_vkCmdSetStencilOpEXT                          vkCmdSetStencilOpEXT                          = 0;
    PFN_vkCmdSetStencilReference                      vkCmdSetStencilReference                      = 0;
    PFN_vkCmdSetStencilTestEnableEXT                  vkCmdSetStencilTestEnableEXT                  = 0;
    PFN_vkCmdSetStencilWriteMask                      vkCmdSetStencilWriteMask                      = 0;
    PFN_vkCmdSetVertexInputEXT                        vkCmdSetVertexInputEXT                        = 0;
    PFN_vkCmdSetViewport                              vkCmdSetViewport                              = 0;
    PFN_vkCmdSetViewportShadingRatePaletteNV          vkCmdSetViewportShadingRatePaletteNV          = 0;
    PFN_vkCmdSetViewportWScalingNV                    vkCmdSetViewportWScalingNV                    = 0;
    PFN_vkCmdSetViewportWithCountEXT                  vkCmdSetViewportWithCountEXT                  = 0;
    PFN_vkCmdSubpassShadingHUAWEI                     vkCmdSubpassShadingHUAWEI                     = 0;
    PFN_vkCmdTraceRaysIndirectKHR                     vkCmdTraceRaysIndirectKHR                     = 0;
    PFN_vkCmdTraceRaysKHR                             vkCmdTraceRaysKHR                             = 0;
    PFN_vkCmdTraceRaysNV                              vkCmdTraceRaysNV                              = 0;
    PFN_vkCmdUpdateBuffer                             vkCmdUpdateBuffer                             = 0;
    PFN_vkCmdWaitEvents                               vkCmdWaitEvents                               = 0;
    PFN_vkCmdWaitEvents2KHR                           vkCmdWaitEvents2KHR                           = 0;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdWriteAccelerationStructuresPropertiesKHR = 0;
    PFN_vkCmdWriteAccelerationStructuresPropertiesNV  vkCmdWriteAccelerationStructuresPropertiesNV  = 0;
    PFN_vkCmdWriteBufferMarker2AMD                    vkCmdWriteBufferMarker2AMD                    = 0;
    PFN_vkCmdWriteBufferMarkerAMD                     vkCmdWriteBufferMarkerAMD                     = 0;
    PFN_vkCmdWriteTimestamp                           vkCmdWriteTimestamp                           = 0;
    PFN_vkCmdWriteTimestamp2KHR                       vkCmdWriteTimestamp2KHR                       = 0;
    PFN_vkCompileDeferredNV                           vkCompileDeferredNV                           = 0;
    PFN_vkCopyAccelerationStructureKHR                vkCopyAccelerationStructureKHR                = 0;
    PFN_vkCopyAccelerationStructureToMemoryKHR        vkCopyAccelerationStructureToMemoryKHR        = 0;
    PFN_vkCopyMemoryToAccelerationStructureKHR        vkCopyMemoryToAccelerationStructureKHR        = 0;
    PFN_vkCreateAccelerationStructureKHR              vkCreateAccelerationStructureKHR              = 0;
    PFN_vkCreateAccelerationStructureNV               vkCreateAccelerationStructureNV               = 0;
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    PFN_vkCreateAndroidSurfaceKHR vkCreateAndroidSurfaceKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateAndroidSurfaceKHR                         = 0;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    PFN_vkCreateBuffer                      vkCreateBuffer                      = 0;
    PFN_vkCreateBufferView                  vkCreateBufferView                  = 0;
    PFN_vkCreateCommandPool                 vkCreateCommandPool                 = 0;
    PFN_vkCreateComputePipelines            vkCreateComputePipelines            = 0;
    PFN_vkCreateCuFunctionNVX               vkCreateCuFunctionNVX               = 0;
    PFN_vkCreateCuModuleNVX                 vkCreateCuModuleNVX                 = 0;
    PFN_vkCreateDebugReportCallbackEXT      vkCreateDebugReportCallbackEXT      = 0;
    PFN_vkCreateDebugUtilsMessengerEXT      vkCreateDebugUtilsMessengerEXT      = 0;
    PFN_vkCreateDeferredOperationKHR        vkCreateDeferredOperationKHR        = 0;
    PFN_vkCreateDescriptorPool              vkCreateDescriptorPool              = 0;
    PFN_vkCreateDescriptorSetLayout         vkCreateDescriptorSetLayout         = 0;
    PFN_vkCreateDescriptorUpdateTemplate    vkCreateDescriptorUpdateTemplate    = 0;
    PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR = 0;
    PFN_vkCreateDevice                      vkCreateDevice                      = 0;
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
    PFN_vkCreateDirectFBSurfaceEXT vkCreateDirectFBSurfaceEXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateDirectFBSurfaceEXT                        = 0;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
    PFN_vkCreateDisplayModeKHR         vkCreateDisplayModeKHR         = 0;
    PFN_vkCreateDisplayPlaneSurfaceKHR vkCreateDisplayPlaneSurfaceKHR = 0;
    PFN_vkCreateEvent                  vkCreateEvent                  = 0;
    PFN_vkCreateFence                  vkCreateFence                  = 0;
    PFN_vkCreateFramebuffer            vkCreateFramebuffer            = 0;
    PFN_vkCreateGraphicsPipelines      vkCreateGraphicsPipelines      = 0;
    PFN_vkCreateHeadlessSurfaceEXT     vkCreateHeadlessSurfaceEXT     = 0;
#if defined( VK_USE_PLATFORM_IOS_MVK )
    PFN_vkCreateIOSSurfaceMVK vkCreateIOSSurfaceMVK = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateIOSSurfaceMVK                             = 0;
#endif /*VK_USE_PLATFORM_IOS_MVK*/
    PFN_vkCreateImage vkCreateImage = 0;
#if defined( VK_USE_PLATFORM_FUCHSIA )
    PFN_vkCreateImagePipeSurfaceFUCHSIA vkCreateImagePipeSurfaceFUCHSIA = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateImagePipeSurfaceFUCHSIA                   = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    PFN_vkCreateImageView                vkCreateImageView                = 0;
    PFN_vkCreateIndirectCommandsLayoutNV vkCreateIndirectCommandsLayoutNV = 0;
    PFN_vkCreateInstance                 vkCreateInstance                 = 0;
#if defined( VK_USE_PLATFORM_MACOS_MVK )
    PFN_vkCreateMacOSSurfaceMVK vkCreateMacOSSurfaceMVK = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateMacOSSurfaceMVK                           = 0;
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
#if defined( VK_USE_PLATFORM_METAL_EXT )
    PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateMetalSurfaceEXT                           = 0;
#endif /*VK_USE_PLATFORM_METAL_EXT*/
    PFN_vkCreatePipelineCache             vkCreatePipelineCache             = 0;
    PFN_vkCreatePipelineLayout            vkCreatePipelineLayout            = 0;
    PFN_vkCreatePrivateDataSlotEXT        vkCreatePrivateDataSlotEXT        = 0;
    PFN_vkCreateQueryPool                 vkCreateQueryPool                 = 0;
    PFN_vkCreateRayTracingPipelinesKHR    vkCreateRayTracingPipelinesKHR    = 0;
    PFN_vkCreateRayTracingPipelinesNV     vkCreateRayTracingPipelinesNV     = 0;
    PFN_vkCreateRenderPass                vkCreateRenderPass                = 0;
    PFN_vkCreateRenderPass2               vkCreateRenderPass2               = 0;
    PFN_vkCreateRenderPass2KHR            vkCreateRenderPass2KHR            = 0;
    PFN_vkCreateSampler                   vkCreateSampler                   = 0;
    PFN_vkCreateSamplerYcbcrConversion    vkCreateSamplerYcbcrConversion    = 0;
    PFN_vkCreateSamplerYcbcrConversionKHR vkCreateSamplerYcbcrConversionKHR = 0;
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    PFN_vkCreateScreenSurfaceQNX vkCreateScreenSurfaceQNX = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateScreenSurfaceQNX                          = 0;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
    PFN_vkCreateSemaphore           vkCreateSemaphore           = 0;
    PFN_vkCreateShaderModule        vkCreateShaderModule        = 0;
    PFN_vkCreateSharedSwapchainsKHR vkCreateSharedSwapchainsKHR = 0;
#if defined( VK_USE_PLATFORM_GGP )
    PFN_vkCreateStreamDescriptorSurfaceGGP vkCreateStreamDescriptorSurfaceGGP = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateStreamDescriptorSurfaceGGP                = 0;
#endif /*VK_USE_PLATFORM_GGP*/
    PFN_vkCreateSwapchainKHR       vkCreateSwapchainKHR       = 0;
    PFN_vkCreateValidationCacheEXT vkCreateValidationCacheEXT = 0;
#if defined( VK_USE_PLATFORM_VI_NN )
    PFN_vkCreateViSurfaceNN vkCreateViSurfaceNN = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateViSurfaceNN                               = 0;
#endif /*VK_USE_PLATFORM_VI_NN*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkCreateVideoSessionKHR vkCreateVideoSessionKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateVideoSessionKHR                           = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkCreateVideoSessionParametersKHR vkCreateVideoSessionParametersKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateVideoSessionParametersKHR                 = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
    PFN_vkCreateWaylandSurfaceKHR vkCreateWaylandSurfaceKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateWaylandSurfaceKHR                         = 0;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateWin32SurfaceKHR                           = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
    PFN_vkCreateXcbSurfaceKHR vkCreateXcbSurfaceKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateXcbSurfaceKHR                             = 0;
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_XLIB_KHR )
    PFN_vkCreateXlibSurfaceKHR vkCreateXlibSurfaceKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkCreateXlibSurfaceKHR                            = 0;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
    PFN_vkDebugMarkerSetObjectNameEXT        vkDebugMarkerSetObjectNameEXT        = 0;
    PFN_vkDebugMarkerSetObjectTagEXT         vkDebugMarkerSetObjectTagEXT         = 0;
    PFN_vkDebugReportMessageEXT              vkDebugReportMessageEXT              = 0;
    PFN_vkDeferredOperationJoinKHR           vkDeferredOperationJoinKHR           = 0;
    PFN_vkDestroyAccelerationStructureKHR    vkDestroyAccelerationStructureKHR    = 0;
    PFN_vkDestroyAccelerationStructureNV     vkDestroyAccelerationStructureNV     = 0;
    PFN_vkDestroyBuffer                      vkDestroyBuffer                      = 0;
    PFN_vkDestroyBufferView                  vkDestroyBufferView                  = 0;
    PFN_vkDestroyCommandPool                 vkDestroyCommandPool                 = 0;
    PFN_vkDestroyCuFunctionNVX               vkDestroyCuFunctionNVX               = 0;
    PFN_vkDestroyCuModuleNVX                 vkDestroyCuModuleNVX                 = 0;
    PFN_vkDestroyDebugReportCallbackEXT      vkDestroyDebugReportCallbackEXT      = 0;
    PFN_vkDestroyDebugUtilsMessengerEXT      vkDestroyDebugUtilsMessengerEXT      = 0;
    PFN_vkDestroyDeferredOperationKHR        vkDestroyDeferredOperationKHR        = 0;
    PFN_vkDestroyDescriptorPool              vkDestroyDescriptorPool              = 0;
    PFN_vkDestroyDescriptorSetLayout         vkDestroyDescriptorSetLayout         = 0;
    PFN_vkDestroyDescriptorUpdateTemplate    vkDestroyDescriptorUpdateTemplate    = 0;
    PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR = 0;
    PFN_vkDestroyDevice                      vkDestroyDevice                      = 0;
    PFN_vkDestroyEvent                       vkDestroyEvent                       = 0;
    PFN_vkDestroyFence                       vkDestroyFence                       = 0;
    PFN_vkDestroyFramebuffer                 vkDestroyFramebuffer                 = 0;
    PFN_vkDestroyImage                       vkDestroyImage                       = 0;
    PFN_vkDestroyImageView                   vkDestroyImageView                   = 0;
    PFN_vkDestroyIndirectCommandsLayoutNV    vkDestroyIndirectCommandsLayoutNV    = 0;
    PFN_vkDestroyInstance                    vkDestroyInstance                    = 0;
    PFN_vkDestroyPipeline                    vkDestroyPipeline                    = 0;
    PFN_vkDestroyPipelineCache               vkDestroyPipelineCache               = 0;
    PFN_vkDestroyPipelineLayout              vkDestroyPipelineLayout              = 0;
    PFN_vkDestroyPrivateDataSlotEXT          vkDestroyPrivateDataSlotEXT          = 0;
    PFN_vkDestroyQueryPool                   vkDestroyQueryPool                   = 0;
    PFN_vkDestroyRenderPass                  vkDestroyRenderPass                  = 0;
    PFN_vkDestroySampler                     vkDestroySampler                     = 0;
    PFN_vkDestroySamplerYcbcrConversion      vkDestroySamplerYcbcrConversion      = 0;
    PFN_vkDestroySamplerYcbcrConversionKHR   vkDestroySamplerYcbcrConversionKHR   = 0;
    PFN_vkDestroySemaphore                   vkDestroySemaphore                   = 0;
    PFN_vkDestroyShaderModule                vkDestroyShaderModule                = 0;
    PFN_vkDestroySurfaceKHR                  vkDestroySurfaceKHR                  = 0;
    PFN_vkDestroySwapchainKHR                vkDestroySwapchainKHR                = 0;
    PFN_vkDestroyValidationCacheEXT          vkDestroyValidationCacheEXT          = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkDestroyVideoSessionKHR vkDestroyVideoSessionKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkDestroyVideoSessionKHR                          = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkDestroyVideoSessionParametersKHR vkDestroyVideoSessionParametersKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkDestroyVideoSessionParametersKHR                = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    PFN_vkDeviceWaitIdle                       vkDeviceWaitIdle                       = 0;
    PFN_vkDisplayPowerControlEXT               vkDisplayPowerControlEXT               = 0;
    PFN_vkEndCommandBuffer                     vkEndCommandBuffer                     = 0;
    PFN_vkEnumerateDeviceExtensionProperties   vkEnumerateDeviceExtensionProperties   = 0;
    PFN_vkEnumerateDeviceLayerProperties       vkEnumerateDeviceLayerProperties       = 0;
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = 0;
    PFN_vkEnumerateInstanceLayerProperties     vkEnumerateInstanceLayerProperties     = 0;
    PFN_vkEnumerateInstanceVersion             vkEnumerateInstanceVersion             = 0;
    PFN_vkEnumeratePhysicalDeviceGroups        vkEnumeratePhysicalDeviceGroups        = 0;
    PFN_vkEnumeratePhysicalDeviceGroupsKHR     vkEnumeratePhysicalDeviceGroupsKHR     = 0;
    PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR
                                                       vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR = 0;
    PFN_vkEnumeratePhysicalDevices                     vkEnumeratePhysicalDevices                     = 0;
    PFN_vkFlushMappedMemoryRanges                      vkFlushMappedMemoryRanges                      = 0;
    PFN_vkFreeCommandBuffers                           vkFreeCommandBuffers                           = 0;
    PFN_vkFreeDescriptorSets                           vkFreeDescriptorSets                           = 0;
    PFN_vkFreeMemory                                   vkFreeMemory                                   = 0;
    PFN_vkGetAccelerationStructureBuildSizesKHR        vkGetAccelerationStructureBuildSizesKHR        = 0;
    PFN_vkGetAccelerationStructureDeviceAddressKHR     vkGetAccelerationStructureDeviceAddressKHR     = 0;
    PFN_vkGetAccelerationStructureHandleNV             vkGetAccelerationStructureHandleNV             = 0;
    PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV = 0;
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetAndroidHardwareBufferPropertiesANDROID       = 0;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    PFN_vkGetBufferDeviceAddress                         vkGetBufferDeviceAddress                         = 0;
    PFN_vkGetBufferDeviceAddressEXT                      vkGetBufferDeviceAddressEXT                      = 0;
    PFN_vkGetBufferDeviceAddressKHR                      vkGetBufferDeviceAddressKHR                      = 0;
    PFN_vkGetBufferMemoryRequirements                    vkGetBufferMemoryRequirements                    = 0;
    PFN_vkGetBufferMemoryRequirements2                   vkGetBufferMemoryRequirements2                   = 0;
    PFN_vkGetBufferMemoryRequirements2KHR                vkGetBufferMemoryRequirements2KHR                = 0;
    PFN_vkGetBufferOpaqueCaptureAddress                  vkGetBufferOpaqueCaptureAddress                  = 0;
    PFN_vkGetBufferOpaqueCaptureAddressKHR               vkGetBufferOpaqueCaptureAddressKHR               = 0;
    PFN_vkGetCalibratedTimestampsEXT                     vkGetCalibratedTimestampsEXT                     = 0;
    PFN_vkGetDeferredOperationMaxConcurrencyKHR          vkGetDeferredOperationMaxConcurrencyKHR          = 0;
    PFN_vkGetDeferredOperationResultKHR                  vkGetDeferredOperationResultKHR                  = 0;
    PFN_vkGetDescriptorSetLayoutSupport                  vkGetDescriptorSetLayoutSupport                  = 0;
    PFN_vkGetDescriptorSetLayoutSupportKHR               vkGetDescriptorSetLayoutSupportKHR               = 0;
    PFN_vkGetDeviceAccelerationStructureCompatibilityKHR vkGetDeviceAccelerationStructureCompatibilityKHR = 0;
    PFN_vkGetDeviceGroupPeerMemoryFeatures               vkGetDeviceGroupPeerMemoryFeatures               = 0;
    PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR            vkGetDeviceGroupPeerMemoryFeaturesKHR            = 0;
    PFN_vkGetDeviceGroupPresentCapabilitiesKHR           vkGetDeviceGroupPresentCapabilitiesKHR           = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetDeviceGroupSurfacePresentModes2EXT vkGetDeviceGroupSurfacePresentModes2EXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetDeviceGroupSurfacePresentModes2EXT           = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    PFN_vkGetDeviceGroupSurfacePresentModesKHR   vkGetDeviceGroupSurfacePresentModesKHR   = 0;
    PFN_vkGetDeviceMemoryCommitment              vkGetDeviceMemoryCommitment              = 0;
    PFN_vkGetDeviceMemoryOpaqueCaptureAddress    vkGetDeviceMemoryOpaqueCaptureAddress    = 0;
    PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR vkGetDeviceMemoryOpaqueCaptureAddressKHR = 0;
    PFN_vkGetDeviceProcAddr                      vkGetDeviceProcAddr                      = 0;
    PFN_vkGetDeviceQueue                         vkGetDeviceQueue                         = 0;
    PFN_vkGetDeviceQueue2                        vkGetDeviceQueue2                        = 0;
    PFN_vkGetDisplayModeProperties2KHR           vkGetDisplayModeProperties2KHR           = 0;
    PFN_vkGetDisplayModePropertiesKHR            vkGetDisplayModePropertiesKHR            = 0;
    PFN_vkGetDisplayPlaneCapabilities2KHR        vkGetDisplayPlaneCapabilities2KHR        = 0;
    PFN_vkGetDisplayPlaneCapabilitiesKHR         vkGetDisplayPlaneCapabilitiesKHR         = 0;
    PFN_vkGetDisplayPlaneSupportedDisplaysKHR    vkGetDisplayPlaneSupportedDisplaysKHR    = 0;
    PFN_vkGetDrmDisplayEXT                       vkGetDrmDisplayEXT                       = 0;
    PFN_vkGetEventStatus                         vkGetEventStatus                         = 0;
    PFN_vkGetFenceFdKHR                          vkGetFenceFdKHR                          = 0;
    PFN_vkGetFenceStatus                         vkGetFenceStatus                         = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetFenceWin32HandleKHR vkGetFenceWin32HandleKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetFenceWin32HandleKHR                          = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    PFN_vkGetGeneratedCommandsMemoryRequirementsNV vkGetGeneratedCommandsMemoryRequirementsNV = 0;
    PFN_vkGetImageDrmFormatModifierPropertiesEXT   vkGetImageDrmFormatModifierPropertiesEXT   = 0;
    PFN_vkGetImageMemoryRequirements               vkGetImageMemoryRequirements               = 0;
    PFN_vkGetImageMemoryRequirements2              vkGetImageMemoryRequirements2              = 0;
    PFN_vkGetImageMemoryRequirements2KHR           vkGetImageMemoryRequirements2KHR           = 0;
    PFN_vkGetImageSparseMemoryRequirements         vkGetImageSparseMemoryRequirements         = 0;
    PFN_vkGetImageSparseMemoryRequirements2        vkGetImageSparseMemoryRequirements2        = 0;
    PFN_vkGetImageSparseMemoryRequirements2KHR     vkGetImageSparseMemoryRequirements2KHR     = 0;
    PFN_vkGetImageSubresourceLayout                vkGetImageSubresourceLayout                = 0;
    PFN_vkGetImageViewAddressNVX                   vkGetImageViewAddressNVX                   = 0;
    PFN_vkGetImageViewHandleNVX                    vkGetImageViewHandleNVX                    = 0;
    PFN_vkGetInstanceProcAddr                      vkGetInstanceProcAddr                      = 0;
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    PFN_vkGetMemoryAndroidHardwareBufferANDROID vkGetMemoryAndroidHardwareBufferANDROID = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetMemoryAndroidHardwareBufferANDROID           = 0;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    PFN_vkGetMemoryFdKHR                    vkGetMemoryFdKHR                    = 0;
    PFN_vkGetMemoryFdPropertiesKHR          vkGetMemoryFdPropertiesKHR          = 0;
    PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerPropertiesEXT = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetMemoryWin32HandleKHR                         = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetMemoryWin32HandleNV vkGetMemoryWin32HandleNV = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetMemoryWin32HandleNV                          = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetMemoryWin32HandlePropertiesKHR vkGetMemoryWin32HandlePropertiesKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetMemoryWin32HandlePropertiesKHR               = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
    PFN_vkGetMemoryZirconHandleFUCHSIA vkGetMemoryZirconHandleFUCHSIA = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetMemoryZirconHandleFUCHSIA                    = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
    PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA vkGetMemoryZirconHandlePropertiesFUCHSIA = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetMemoryZirconHandlePropertiesFUCHSIA          = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    PFN_vkGetPastPresentationTimingGOOGLE                vkGetPastPresentationTimingGOOGLE                = 0;
    PFN_vkGetPerformanceParameterINTEL                   vkGetPerformanceParameterINTEL                   = 0;
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT   vkGetPhysicalDeviceCalibrateableTimeDomainsEXT   = 0;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = 0;
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
    PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT vkGetPhysicalDeviceDirectFBPresentationSupportEXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceDirectFBPresentationSupportEXT = 0;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
    PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR      vkGetPhysicalDeviceDisplayPlaneProperties2KHR      = 0;
    PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR       vkGetPhysicalDeviceDisplayPlanePropertiesKHR       = 0;
    PFN_vkGetPhysicalDeviceDisplayProperties2KHR           vkGetPhysicalDeviceDisplayProperties2KHR           = 0;
    PFN_vkGetPhysicalDeviceDisplayPropertiesKHR            vkGetPhysicalDeviceDisplayPropertiesKHR            = 0;
    PFN_vkGetPhysicalDeviceExternalBufferProperties        vkGetPhysicalDeviceExternalBufferProperties        = 0;
    PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR     vkGetPhysicalDeviceExternalBufferPropertiesKHR     = 0;
    PFN_vkGetPhysicalDeviceExternalFenceProperties         vkGetPhysicalDeviceExternalFenceProperties         = 0;
    PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR      vkGetPhysicalDeviceExternalFencePropertiesKHR      = 0;
    PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV vkGetPhysicalDeviceExternalImageFormatPropertiesNV = 0;
    PFN_vkGetPhysicalDeviceExternalSemaphoreProperties     vkGetPhysicalDeviceExternalSemaphoreProperties     = 0;
    PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR  vkGetPhysicalDeviceExternalSemaphorePropertiesKHR  = 0;
    PFN_vkGetPhysicalDeviceFeatures                        vkGetPhysicalDeviceFeatures                        = 0;
    PFN_vkGetPhysicalDeviceFeatures2                       vkGetPhysicalDeviceFeatures2                       = 0;
    PFN_vkGetPhysicalDeviceFeatures2KHR                    vkGetPhysicalDeviceFeatures2KHR                    = 0;
    PFN_vkGetPhysicalDeviceFormatProperties                vkGetPhysicalDeviceFormatProperties                = 0;
    PFN_vkGetPhysicalDeviceFormatProperties2               vkGetPhysicalDeviceFormatProperties2               = 0;
    PFN_vkGetPhysicalDeviceFormatProperties2KHR            vkGetPhysicalDeviceFormatProperties2KHR            = 0;
    PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR         vkGetPhysicalDeviceFragmentShadingRatesKHR         = 0;
    PFN_vkGetPhysicalDeviceImageFormatProperties           vkGetPhysicalDeviceImageFormatProperties           = 0;
    PFN_vkGetPhysicalDeviceImageFormatProperties2          vkGetPhysicalDeviceImageFormatProperties2          = 0;
    PFN_vkGetPhysicalDeviceImageFormatProperties2KHR       vkGetPhysicalDeviceImageFormatProperties2KHR       = 0;
    PFN_vkGetPhysicalDeviceMemoryProperties                vkGetPhysicalDeviceMemoryProperties                = 0;
    PFN_vkGetPhysicalDeviceMemoryProperties2               vkGetPhysicalDeviceMemoryProperties2               = 0;
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR            vkGetPhysicalDeviceMemoryProperties2KHR            = 0;
    PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT        vkGetPhysicalDeviceMultisamplePropertiesEXT        = 0;
    PFN_vkGetPhysicalDevicePresentRectanglesKHR            vkGetPhysicalDevicePresentRectanglesKHR            = 0;
    PFN_vkGetPhysicalDeviceProperties                      vkGetPhysicalDeviceProperties                      = 0;
    PFN_vkGetPhysicalDeviceProperties2                     vkGetPhysicalDeviceProperties2                     = 0;
    PFN_vkGetPhysicalDeviceProperties2KHR                  vkGetPhysicalDeviceProperties2KHR                  = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR
                                                     vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties     vkGetPhysicalDeviceQueueFamilyProperties                = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2    vkGetPhysicalDeviceQueueFamilyProperties2               = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR            = 0;
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX vkGetPhysicalDeviceScreenPresentationSupportQNX = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceScreenPresentationSupportQNX   = 0;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
    PFN_vkGetPhysicalDeviceSparseImageFormatProperties     vkGetPhysicalDeviceSparseImageFormatProperties     = 0;
    PFN_vkGetPhysicalDeviceSparseImageFormatProperties2    vkGetPhysicalDeviceSparseImageFormatProperties2    = 0;
    PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR = 0;
    PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV
                                                   vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV = 0;
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT vkGetPhysicalDeviceSurfaceCapabilities2EXT = 0;
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR = 0;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR  vkGetPhysicalDeviceSurfaceCapabilitiesKHR  = 0;
    PFN_vkGetPhysicalDeviceSurfaceFormats2KHR      vkGetPhysicalDeviceSurfaceFormats2KHR      = 0;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR       vkGetPhysicalDeviceSurfaceFormatsKHR       = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT vkGetPhysicalDeviceSurfacePresentModes2EXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceSurfacePresentModes2EXT        = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR = 0;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR      vkGetPhysicalDeviceSurfaceSupportKHR      = 0;
    PFN_vkGetPhysicalDeviceToolPropertiesEXT      vkGetPhysicalDeviceToolPropertiesEXT      = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR vkGetPhysicalDeviceVideoCapabilitiesKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceVideoCapabilitiesKHR           = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR vkGetPhysicalDeviceVideoFormatPropertiesKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceVideoFormatPropertiesKHR       = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
    PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR vkGetPhysicalDeviceWaylandPresentationSupportKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceWaylandPresentationSupportKHR  = 0;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR vkGetPhysicalDeviceWin32PresentationSupportKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceWin32PresentationSupportKHR    = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
    PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR vkGetPhysicalDeviceXcbPresentationSupportKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceXcbPresentationSupportKHR      = 0;
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_XLIB_KHR )
    PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR vkGetPhysicalDeviceXlibPresentationSupportKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetPhysicalDeviceXlibPresentationSupportKHR     = 0;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
    PFN_vkGetPipelineCacheData                            vkGetPipelineCacheData                            = 0;
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR vkGetPipelineExecutableInternalRepresentationsKHR = 0;
    PFN_vkGetPipelineExecutablePropertiesKHR              vkGetPipelineExecutablePropertiesKHR              = 0;
    PFN_vkGetPipelineExecutableStatisticsKHR              vkGetPipelineExecutableStatisticsKHR              = 0;
    PFN_vkGetPrivateDataEXT                               vkGetPrivateDataEXT                               = 0;
    PFN_vkGetQueryPoolResults                             vkGetQueryPoolResults                             = 0;
    PFN_vkGetQueueCheckpointData2NV                       vkGetQueueCheckpointData2NV                       = 0;
    PFN_vkGetQueueCheckpointDataNV                        vkGetQueueCheckpointDataNV                        = 0;
#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
    PFN_vkGetRandROutputDisplayEXT vkGetRandROutputDisplayEXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetRandROutputDisplayEXT                        = 0;
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/
    PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = 0;
    PFN_vkGetRayTracingShaderGroupHandlesKHR              vkGetRayTracingShaderGroupHandlesKHR              = 0;
    PFN_vkGetRayTracingShaderGroupHandlesNV               vkGetRayTracingShaderGroupHandlesNV               = 0;
    PFN_vkGetRayTracingShaderGroupStackSizeKHR            vkGetRayTracingShaderGroupStackSizeKHR            = 0;
    PFN_vkGetRefreshCycleDurationGOOGLE                   vkGetRefreshCycleDurationGOOGLE                   = 0;
    PFN_vkGetRenderAreaGranularity                        vkGetRenderAreaGranularity                        = 0;
    PFN_vkGetSemaphoreCounterValue                        vkGetSemaphoreCounterValue                        = 0;
    PFN_vkGetSemaphoreCounterValueKHR                     vkGetSemaphoreCounterValueKHR                     = 0;
    PFN_vkGetSemaphoreFdKHR                               vkGetSemaphoreFdKHR                               = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetSemaphoreWin32HandleKHR                      = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
    PFN_vkGetSemaphoreZirconHandleFUCHSIA vkGetSemaphoreZirconHandleFUCHSIA = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetSemaphoreZirconHandleFUCHSIA                 = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    PFN_vkGetShaderInfoAMD                        vkGetShaderInfoAMD                        = 0;
    PFN_vkGetSubpassShadingMaxWorkgroupSizeHUAWEI vkGetSubpassShadingMaxWorkgroupSizeHUAWEI = 0;
    PFN_vkGetSwapchainCounterEXT                  vkGetSwapchainCounterEXT                  = 0;
    PFN_vkGetSwapchainImagesKHR                   vkGetSwapchainImagesKHR                   = 0;
    PFN_vkGetSwapchainStatusKHR                   vkGetSwapchainStatusKHR                   = 0;
    PFN_vkGetValidationCacheDataEXT               vkGetValidationCacheDataEXT               = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkGetVideoSessionMemoryRequirementsKHR vkGetVideoSessionMemoryRequirementsKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetVideoSessionMemoryRequirementsKHR            = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkGetWinrtDisplayNV vkGetWinrtDisplayNV = 0;
#else
    PFN_dummy placeholder_dont_call_vkGetWinrtDisplayNV                               = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    PFN_vkImportFenceFdKHR vkImportFenceFdKHR = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkImportFenceWin32HandleKHR vkImportFenceWin32HandleKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkImportFenceWin32HandleKHR                       = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkImportSemaphoreWin32HandleKHR vkImportSemaphoreWin32HandleKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkImportSemaphoreWin32HandleKHR                   = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
    PFN_vkImportSemaphoreZirconHandleFUCHSIA vkImportSemaphoreZirconHandleFUCHSIA = 0;
#else
    PFN_dummy placeholder_dont_call_vkImportSemaphoreZirconHandleFUCHSIA              = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    PFN_vkInitializePerformanceApiINTEL         vkInitializePerformanceApiINTEL         = 0;
    PFN_vkInvalidateMappedMemoryRanges          vkInvalidateMappedMemoryRanges          = 0;
    PFN_vkMapMemory                             vkMapMemory                             = 0;
    PFN_vkMergePipelineCaches                   vkMergePipelineCaches                   = 0;
    PFN_vkMergeValidationCachesEXT              vkMergeValidationCachesEXT              = 0;
    PFN_vkQueueBeginDebugUtilsLabelEXT          vkQueueBeginDebugUtilsLabelEXT          = 0;
    PFN_vkQueueBindSparse                       vkQueueBindSparse                       = 0;
    PFN_vkQueueEndDebugUtilsLabelEXT            vkQueueEndDebugUtilsLabelEXT            = 0;
    PFN_vkQueueInsertDebugUtilsLabelEXT         vkQueueInsertDebugUtilsLabelEXT         = 0;
    PFN_vkQueuePresentKHR                       vkQueuePresentKHR                       = 0;
    PFN_vkQueueSetPerformanceConfigurationINTEL vkQueueSetPerformanceConfigurationINTEL = 0;
    PFN_vkQueueSubmit                           vkQueueSubmit                           = 0;
    PFN_vkQueueSubmit2KHR                       vkQueueSubmit2KHR                       = 0;
    PFN_vkQueueWaitIdle                         vkQueueWaitIdle                         = 0;
    PFN_vkRegisterDeviceEventEXT                vkRegisterDeviceEventEXT                = 0;
    PFN_vkRegisterDisplayEventEXT               vkRegisterDisplayEventEXT               = 0;
    PFN_vkReleaseDisplayEXT                     vkReleaseDisplayEXT                     = 0;
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    PFN_vkReleaseFullScreenExclusiveModeEXT vkReleaseFullScreenExclusiveModeEXT = 0;
#else
    PFN_dummy placeholder_dont_call_vkReleaseFullScreenExclusiveModeEXT               = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    PFN_vkReleasePerformanceConfigurationINTEL vkReleasePerformanceConfigurationINTEL = 0;
    PFN_vkReleaseProfilingLockKHR              vkReleaseProfilingLockKHR              = 0;
    PFN_vkResetCommandBuffer                   vkResetCommandBuffer                   = 0;
    PFN_vkResetCommandPool                     vkResetCommandPool                     = 0;
    PFN_vkResetDescriptorPool                  vkResetDescriptorPool                  = 0;
    PFN_vkResetEvent                           vkResetEvent                           = 0;
    PFN_vkResetFences                          vkResetFences                          = 0;
    PFN_vkResetQueryPool                       vkResetQueryPool                       = 0;
    PFN_vkResetQueryPoolEXT                    vkResetQueryPoolEXT                    = 0;
    PFN_vkSetDebugUtilsObjectNameEXT           vkSetDebugUtilsObjectNameEXT           = 0;
    PFN_vkSetDebugUtilsObjectTagEXT            vkSetDebugUtilsObjectTagEXT            = 0;
    PFN_vkSetEvent                             vkSetEvent                             = 0;
    PFN_vkSetHdrMetadataEXT                    vkSetHdrMetadataEXT                    = 0;
    PFN_vkSetLocalDimmingAMD                   vkSetLocalDimmingAMD                   = 0;
    PFN_vkSetPrivateDataEXT                    vkSetPrivateDataEXT                    = 0;
    PFN_vkSignalSemaphore                      vkSignalSemaphore                      = 0;
    PFN_vkSignalSemaphoreKHR                   vkSignalSemaphoreKHR                   = 0;
    PFN_vkSubmitDebugUtilsMessageEXT           vkSubmitDebugUtilsMessageEXT           = 0;
    PFN_vkTrimCommandPool                      vkTrimCommandPool                      = 0;
    PFN_vkTrimCommandPoolKHR                   vkTrimCommandPoolKHR                   = 0;
    PFN_vkUninitializePerformanceApiINTEL      vkUninitializePerformanceApiINTEL      = 0;
    PFN_vkUnmapMemory                          vkUnmapMemory                          = 0;
    PFN_vkUpdateDescriptorSetWithTemplate      vkUpdateDescriptorSetWithTemplate      = 0;
    PFN_vkUpdateDescriptorSetWithTemplateKHR   vkUpdateDescriptorSetWithTemplateKHR   = 0;
    PFN_vkUpdateDescriptorSets                 vkUpdateDescriptorSets                 = 0;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    PFN_vkUpdateVideoSessionParametersKHR vkUpdateVideoSessionParametersKHR = 0;
#else
    PFN_dummy placeholder_dont_call_vkUpdateVideoSessionParametersKHR                 = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    PFN_vkWaitForFences                            vkWaitForFences                            = 0;
    PFN_vkWaitSemaphores                           vkWaitSemaphores                           = 0;
    PFN_vkWaitSemaphoresKHR                        vkWaitSemaphoresKHR                        = 0;
    PFN_vkWriteAccelerationStructuresPropertiesKHR vkWriteAccelerationStructuresPropertiesKHR = 0;

  public:
    DispatchLoaderDynamic() VULKAN_HPP_NOEXCEPT                                    = default;
    DispatchLoaderDynamic( DispatchLoaderDynamic const & rhs ) VULKAN_HPP_NOEXCEPT = default;

#if !defined( VK_NO_PROTOTYPES )
    // This interface is designed to be used for per-device function pointers in combination with a linked vulkan
    // library.
    template <typename DynamicLoader>
    void init( VULKAN_HPP_NAMESPACE::Instance const & instance,
               VULKAN_HPP_NAMESPACE::Device const &   device,
               DynamicLoader const &                  dl ) VULKAN_HPP_NOEXCEPT
    {
      PFN_vkGetInstanceProcAddr getInstanceProcAddr =
        dl.template getProcAddress<PFN_vkGetInstanceProcAddr>( "vkGetInstanceProcAddr" );
      PFN_vkGetDeviceProcAddr getDeviceProcAddr =
        dl.template getProcAddress<PFN_vkGetDeviceProcAddr>( "vkGetDeviceProcAddr" );
      init( static_cast<VkInstance>( instance ),
            getInstanceProcAddr,
            static_cast<VkDevice>( device ),
            device ? getDeviceProcAddr : nullptr );
    }

    // This interface is designed to be used for per-device function pointers in combination with a linked vulkan
    // library.
    template <typename DynamicLoader
#  if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
              = VULKAN_HPP_NAMESPACE::DynamicLoader
#  endif
              >
    void init( VULKAN_HPP_NAMESPACE::Instance const & instance,
               VULKAN_HPP_NAMESPACE::Device const &   device ) VULKAN_HPP_NOEXCEPT
    {
      static DynamicLoader dl;
      init( instance, device, dl );
    }
#endif  // !defined( VK_NO_PROTOTYPES )

    DispatchLoaderDynamic( PFN_vkGetInstanceProcAddr getInstanceProcAddr ) VULKAN_HPP_NOEXCEPT
    {
      init( getInstanceProcAddr );
    }

    void init( PFN_vkGetInstanceProcAddr getInstanceProcAddr ) VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getInstanceProcAddr );

      vkGetInstanceProcAddr = getInstanceProcAddr;
      vkCreateInstance      = PFN_vkCreateInstance( vkGetInstanceProcAddr( NULL, "vkCreateInstance" ) );
      vkEnumerateInstanceExtensionProperties = PFN_vkEnumerateInstanceExtensionProperties(
        vkGetInstanceProcAddr( NULL, "vkEnumerateInstanceExtensionProperties" ) );
      vkEnumerateInstanceLayerProperties =
        PFN_vkEnumerateInstanceLayerProperties( vkGetInstanceProcAddr( NULL, "vkEnumerateInstanceLayerProperties" ) );
      vkEnumerateInstanceVersion =
        PFN_vkEnumerateInstanceVersion( vkGetInstanceProcAddr( NULL, "vkEnumerateInstanceVersion" ) );
    }

    // This interface does not require a linked vulkan library.
    DispatchLoaderDynamic( VkInstance                instance,
                           PFN_vkGetInstanceProcAddr getInstanceProcAddr,
                           VkDevice                  device            = {},
                           PFN_vkGetDeviceProcAddr   getDeviceProcAddr = nullptr ) VULKAN_HPP_NOEXCEPT
    {
      init( instance, getInstanceProcAddr, device, getDeviceProcAddr );
    }

    // This interface does not require a linked vulkan library.
    void init( VkInstance                instance,
               PFN_vkGetInstanceProcAddr getInstanceProcAddr,
               VkDevice                  device              = {},
               PFN_vkGetDeviceProcAddr /*getDeviceProcAddr*/ = nullptr ) VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( instance && getInstanceProcAddr );
      vkGetInstanceProcAddr = getInstanceProcAddr;
      init( VULKAN_HPP_NAMESPACE::Instance( instance ) );
      if ( device )
      {
        init( VULKAN_HPP_NAMESPACE::Device( device ) );
      }
    }

    void init( VULKAN_HPP_NAMESPACE::Instance instanceCpp ) VULKAN_HPP_NOEXCEPT
    {
      VkInstance instance = static_cast<VkInstance>( instanceCpp );
      vkAcquireDrmDisplayEXT =
        PFN_vkAcquireDrmDisplayEXT( vkGetInstanceProcAddr( instance, "vkAcquireDrmDisplayEXT" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkAcquireWinrtDisplayNV =
        PFN_vkAcquireWinrtDisplayNV( vkGetInstanceProcAddr( instance, "vkAcquireWinrtDisplayNV" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
      vkAcquireXlibDisplayEXT =
        PFN_vkAcquireXlibDisplayEXT( vkGetInstanceProcAddr( instance, "vkAcquireXlibDisplayEXT" ) );
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      vkCreateAndroidSurfaceKHR =
        PFN_vkCreateAndroidSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateAndroidSurfaceKHR" ) );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      vkCreateDebugReportCallbackEXT =
        PFN_vkCreateDebugReportCallbackEXT( vkGetInstanceProcAddr( instance, "vkCreateDebugReportCallbackEXT" ) );
      vkCreateDebugUtilsMessengerEXT =
        PFN_vkCreateDebugUtilsMessengerEXT( vkGetInstanceProcAddr( instance, "vkCreateDebugUtilsMessengerEXT" ) );
      vkCreateDevice = PFN_vkCreateDevice( vkGetInstanceProcAddr( instance, "vkCreateDevice" ) );
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      vkCreateDirectFBSurfaceEXT =
        PFN_vkCreateDirectFBSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateDirectFBSurfaceEXT" ) );
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
      vkCreateDisplayModeKHR =
        PFN_vkCreateDisplayModeKHR( vkGetInstanceProcAddr( instance, "vkCreateDisplayModeKHR" ) );
      vkCreateDisplayPlaneSurfaceKHR =
        PFN_vkCreateDisplayPlaneSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateDisplayPlaneSurfaceKHR" ) );
      vkCreateHeadlessSurfaceEXT =
        PFN_vkCreateHeadlessSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateHeadlessSurfaceEXT" ) );
#if defined( VK_USE_PLATFORM_IOS_MVK )
      vkCreateIOSSurfaceMVK = PFN_vkCreateIOSSurfaceMVK( vkGetInstanceProcAddr( instance, "vkCreateIOSSurfaceMVK" ) );
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkCreateImagePipeSurfaceFUCHSIA =
        PFN_vkCreateImagePipeSurfaceFUCHSIA( vkGetInstanceProcAddr( instance, "vkCreateImagePipeSurfaceFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
      vkCreateMacOSSurfaceMVK =
        PFN_vkCreateMacOSSurfaceMVK( vkGetInstanceProcAddr( instance, "vkCreateMacOSSurfaceMVK" ) );
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
#if defined( VK_USE_PLATFORM_METAL_EXT )
      vkCreateMetalSurfaceEXT =
        PFN_vkCreateMetalSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateMetalSurfaceEXT" ) );
#endif /*VK_USE_PLATFORM_METAL_EXT*/
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      vkCreateScreenSurfaceQNX =
        PFN_vkCreateScreenSurfaceQNX( vkGetInstanceProcAddr( instance, "vkCreateScreenSurfaceQNX" ) );
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
#if defined( VK_USE_PLATFORM_GGP )
      vkCreateStreamDescriptorSurfaceGGP = PFN_vkCreateStreamDescriptorSurfaceGGP(
        vkGetInstanceProcAddr( instance, "vkCreateStreamDescriptorSurfaceGGP" ) );
#endif /*VK_USE_PLATFORM_GGP*/
#if defined( VK_USE_PLATFORM_VI_NN )
      vkCreateViSurfaceNN = PFN_vkCreateViSurfaceNN( vkGetInstanceProcAddr( instance, "vkCreateViSurfaceNN" ) );
#endif /*VK_USE_PLATFORM_VI_NN*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      vkCreateWaylandSurfaceKHR =
        PFN_vkCreateWaylandSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateWaylandSurfaceKHR" ) );
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkCreateWin32SurfaceKHR =
        PFN_vkCreateWin32SurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateWin32SurfaceKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
      vkCreateXcbSurfaceKHR = PFN_vkCreateXcbSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateXcbSurfaceKHR" ) );
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_XLIB_KHR )
      vkCreateXlibSurfaceKHR =
        PFN_vkCreateXlibSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateXlibSurfaceKHR" ) );
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
      vkDebugReportMessageEXT =
        PFN_vkDebugReportMessageEXT( vkGetInstanceProcAddr( instance, "vkDebugReportMessageEXT" ) );
      vkDestroyDebugReportCallbackEXT =
        PFN_vkDestroyDebugReportCallbackEXT( vkGetInstanceProcAddr( instance, "vkDestroyDebugReportCallbackEXT" ) );
      vkDestroyDebugUtilsMessengerEXT =
        PFN_vkDestroyDebugUtilsMessengerEXT( vkGetInstanceProcAddr( instance, "vkDestroyDebugUtilsMessengerEXT" ) );
      vkDestroyInstance   = PFN_vkDestroyInstance( vkGetInstanceProcAddr( instance, "vkDestroyInstance" ) );
      vkDestroySurfaceKHR = PFN_vkDestroySurfaceKHR( vkGetInstanceProcAddr( instance, "vkDestroySurfaceKHR" ) );
      vkEnumerateDeviceExtensionProperties = PFN_vkEnumerateDeviceExtensionProperties(
        vkGetInstanceProcAddr( instance, "vkEnumerateDeviceExtensionProperties" ) );
      vkEnumerateDeviceLayerProperties =
        PFN_vkEnumerateDeviceLayerProperties( vkGetInstanceProcAddr( instance, "vkEnumerateDeviceLayerProperties" ) );
      vkEnumeratePhysicalDeviceGroups =
        PFN_vkEnumeratePhysicalDeviceGroups( vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceGroups" ) );
      vkEnumeratePhysicalDeviceGroupsKHR = PFN_vkEnumeratePhysicalDeviceGroupsKHR(
        vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceGroupsKHR" ) );
      if ( !vkEnumeratePhysicalDeviceGroups )
        vkEnumeratePhysicalDeviceGroups = vkEnumeratePhysicalDeviceGroupsKHR;
      vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR =
        PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
          vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR" ) );
      vkEnumeratePhysicalDevices =
        PFN_vkEnumeratePhysicalDevices( vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDevices" ) );
      vkGetDisplayModeProperties2KHR =
        PFN_vkGetDisplayModeProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetDisplayModeProperties2KHR" ) );
      vkGetDisplayModePropertiesKHR =
        PFN_vkGetDisplayModePropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetDisplayModePropertiesKHR" ) );
      vkGetDisplayPlaneCapabilities2KHR =
        PFN_vkGetDisplayPlaneCapabilities2KHR( vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneCapabilities2KHR" ) );
      vkGetDisplayPlaneCapabilitiesKHR =
        PFN_vkGetDisplayPlaneCapabilitiesKHR( vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneCapabilitiesKHR" ) );
      vkGetDisplayPlaneSupportedDisplaysKHR = PFN_vkGetDisplayPlaneSupportedDisplaysKHR(
        vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneSupportedDisplaysKHR" ) );
      vkGetDrmDisplayEXT    = PFN_vkGetDrmDisplayEXT( vkGetInstanceProcAddr( instance, "vkGetDrmDisplayEXT" ) );
      vkGetInstanceProcAddr = PFN_vkGetInstanceProcAddr( vkGetInstanceProcAddr( instance, "vkGetInstanceProcAddr" ) );
      vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT" ) );
      vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV" ) );
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      vkGetPhysicalDeviceDirectFBPresentationSupportEXT = PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDirectFBPresentationSupportEXT" ) );
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
      vkGetPhysicalDeviceDisplayPlaneProperties2KHR = PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPlaneProperties2KHR" ) );
      vkGetPhysicalDeviceDisplayPlanePropertiesKHR = PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR" ) );
      vkGetPhysicalDeviceDisplayProperties2KHR = PFN_vkGetPhysicalDeviceDisplayProperties2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayProperties2KHR" ) );
      vkGetPhysicalDeviceDisplayPropertiesKHR = PFN_vkGetPhysicalDeviceDisplayPropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPropertiesKHR" ) );
      vkGetPhysicalDeviceExternalBufferProperties = PFN_vkGetPhysicalDeviceExternalBufferProperties(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalBufferProperties" ) );
      vkGetPhysicalDeviceExternalBufferPropertiesKHR = PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalBufferPropertiesKHR" ) );
      if ( !vkGetPhysicalDeviceExternalBufferProperties )
        vkGetPhysicalDeviceExternalBufferProperties = vkGetPhysicalDeviceExternalBufferPropertiesKHR;
      vkGetPhysicalDeviceExternalFenceProperties = PFN_vkGetPhysicalDeviceExternalFenceProperties(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalFenceProperties" ) );
      vkGetPhysicalDeviceExternalFencePropertiesKHR = PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalFencePropertiesKHR" ) );
      if ( !vkGetPhysicalDeviceExternalFenceProperties )
        vkGetPhysicalDeviceExternalFenceProperties = vkGetPhysicalDeviceExternalFencePropertiesKHR;
      vkGetPhysicalDeviceExternalImageFormatPropertiesNV = PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalImageFormatPropertiesNV" ) );
      vkGetPhysicalDeviceExternalSemaphoreProperties = PFN_vkGetPhysicalDeviceExternalSemaphoreProperties(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalSemaphoreProperties" ) );
      vkGetPhysicalDeviceExternalSemaphorePropertiesKHR = PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR" ) );
      if ( !vkGetPhysicalDeviceExternalSemaphoreProperties )
        vkGetPhysicalDeviceExternalSemaphoreProperties = vkGetPhysicalDeviceExternalSemaphorePropertiesKHR;
      vkGetPhysicalDeviceFeatures =
        PFN_vkGetPhysicalDeviceFeatures( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures" ) );
      vkGetPhysicalDeviceFeatures2 =
        PFN_vkGetPhysicalDeviceFeatures2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures2" ) );
      vkGetPhysicalDeviceFeatures2KHR =
        PFN_vkGetPhysicalDeviceFeatures2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures2KHR" ) );
      if ( !vkGetPhysicalDeviceFeatures2 )
        vkGetPhysicalDeviceFeatures2 = vkGetPhysicalDeviceFeatures2KHR;
      vkGetPhysicalDeviceFormatProperties = PFN_vkGetPhysicalDeviceFormatProperties(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties" ) );
      vkGetPhysicalDeviceFormatProperties2 = PFN_vkGetPhysicalDeviceFormatProperties2(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties2" ) );
      vkGetPhysicalDeviceFormatProperties2KHR = PFN_vkGetPhysicalDeviceFormatProperties2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceFormatProperties2 )
        vkGetPhysicalDeviceFormatProperties2 = vkGetPhysicalDeviceFormatProperties2KHR;
      vkGetPhysicalDeviceFragmentShadingRatesKHR = PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFragmentShadingRatesKHR" ) );
      vkGetPhysicalDeviceImageFormatProperties = PFN_vkGetPhysicalDeviceImageFormatProperties(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties" ) );
      vkGetPhysicalDeviceImageFormatProperties2 = PFN_vkGetPhysicalDeviceImageFormatProperties2(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties2" ) );
      vkGetPhysicalDeviceImageFormatProperties2KHR = PFN_vkGetPhysicalDeviceImageFormatProperties2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceImageFormatProperties2 )
        vkGetPhysicalDeviceImageFormatProperties2 = vkGetPhysicalDeviceImageFormatProperties2KHR;
      vkGetPhysicalDeviceMemoryProperties = PFN_vkGetPhysicalDeviceMemoryProperties(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties" ) );
      vkGetPhysicalDeviceMemoryProperties2 = PFN_vkGetPhysicalDeviceMemoryProperties2(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties2" ) );
      vkGetPhysicalDeviceMemoryProperties2KHR = PFN_vkGetPhysicalDeviceMemoryProperties2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceMemoryProperties2 )
        vkGetPhysicalDeviceMemoryProperties2 = vkGetPhysicalDeviceMemoryProperties2KHR;
      vkGetPhysicalDeviceMultisamplePropertiesEXT = PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMultisamplePropertiesEXT" ) );
      vkGetPhysicalDevicePresentRectanglesKHR = PFN_vkGetPhysicalDevicePresentRectanglesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDevicePresentRectanglesKHR" ) );
      vkGetPhysicalDeviceProperties =
        PFN_vkGetPhysicalDeviceProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties" ) );
      vkGetPhysicalDeviceProperties2 =
        PFN_vkGetPhysicalDeviceProperties2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties2" ) );
      vkGetPhysicalDeviceProperties2KHR =
        PFN_vkGetPhysicalDeviceProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceProperties2 )
        vkGetPhysicalDeviceProperties2 = vkGetPhysicalDeviceProperties2KHR;
      vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR =
        PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR" ) );
      vkGetPhysicalDeviceQueueFamilyProperties = PFN_vkGetPhysicalDeviceQueueFamilyProperties(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties" ) );
      vkGetPhysicalDeviceQueueFamilyProperties2 = PFN_vkGetPhysicalDeviceQueueFamilyProperties2(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties2" ) );
      vkGetPhysicalDeviceQueueFamilyProperties2KHR = PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceQueueFamilyProperties2 )
        vkGetPhysicalDeviceQueueFamilyProperties2 = vkGetPhysicalDeviceQueueFamilyProperties2KHR;
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      vkGetPhysicalDeviceScreenPresentationSupportQNX = PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceScreenPresentationSupportQNX" ) );
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      vkGetPhysicalDeviceSparseImageFormatProperties = PFN_vkGetPhysicalDeviceSparseImageFormatProperties(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties" ) );
      vkGetPhysicalDeviceSparseImageFormatProperties2 = PFN_vkGetPhysicalDeviceSparseImageFormatProperties2(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties2" ) );
      vkGetPhysicalDeviceSparseImageFormatProperties2KHR = PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceSparseImageFormatProperties2 )
        vkGetPhysicalDeviceSparseImageFormatProperties2 = vkGetPhysicalDeviceSparseImageFormatProperties2KHR;
      vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV =
        PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV" ) );
      vkGetPhysicalDeviceSurfaceCapabilities2EXT = PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilities2EXT" ) );
      vkGetPhysicalDeviceSurfaceCapabilities2KHR = PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR" ) );
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR = PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR" ) );
      vkGetPhysicalDeviceSurfaceFormats2KHR = PFN_vkGetPhysicalDeviceSurfaceFormats2KHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceFormats2KHR" ) );
      vkGetPhysicalDeviceSurfaceFormatsKHR = PFN_vkGetPhysicalDeviceSurfaceFormatsKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceFormatsKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetPhysicalDeviceSurfacePresentModes2EXT = PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfacePresentModes2EXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkGetPhysicalDeviceSurfacePresentModesKHR = PFN_vkGetPhysicalDeviceSurfacePresentModesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfacePresentModesKHR" ) );
      vkGetPhysicalDeviceSurfaceSupportKHR = PFN_vkGetPhysicalDeviceSurfaceSupportKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceSupportKHR" ) );
      vkGetPhysicalDeviceToolPropertiesEXT = PFN_vkGetPhysicalDeviceToolPropertiesEXT(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceToolPropertiesEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkGetPhysicalDeviceVideoCapabilitiesKHR = PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkGetPhysicalDeviceVideoFormatPropertiesKHR = PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceVideoFormatPropertiesKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      vkGetPhysicalDeviceWaylandPresentationSupportKHR = PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceWaylandPresentationSupportKHR" ) );
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetPhysicalDeviceWin32PresentationSupportKHR = PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceWin32PresentationSupportKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
      vkGetPhysicalDeviceXcbPresentationSupportKHR = PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceXcbPresentationSupportKHR" ) );
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_XLIB_KHR )
      vkGetPhysicalDeviceXlibPresentationSupportKHR = PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceXlibPresentationSupportKHR" ) );
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
      vkGetRandROutputDisplayEXT =
        PFN_vkGetRandROutputDisplayEXT( vkGetInstanceProcAddr( instance, "vkGetRandROutputDisplayEXT" ) );
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetWinrtDisplayNV = PFN_vkGetWinrtDisplayNV( vkGetInstanceProcAddr( instance, "vkGetWinrtDisplayNV" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkReleaseDisplayEXT = PFN_vkReleaseDisplayEXT( vkGetInstanceProcAddr( instance, "vkReleaseDisplayEXT" ) );
      vkSubmitDebugUtilsMessageEXT =
        PFN_vkSubmitDebugUtilsMessageEXT( vkGetInstanceProcAddr( instance, "vkSubmitDebugUtilsMessageEXT" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkAcquireFullScreenExclusiveModeEXT = PFN_vkAcquireFullScreenExclusiveModeEXT(
        vkGetInstanceProcAddr( instance, "vkAcquireFullScreenExclusiveModeEXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkAcquireNextImage2KHR =
        PFN_vkAcquireNextImage2KHR( vkGetInstanceProcAddr( instance, "vkAcquireNextImage2KHR" ) );
      vkAcquireNextImageKHR = PFN_vkAcquireNextImageKHR( vkGetInstanceProcAddr( instance, "vkAcquireNextImageKHR" ) );
      vkAcquirePerformanceConfigurationINTEL = PFN_vkAcquirePerformanceConfigurationINTEL(
        vkGetInstanceProcAddr( instance, "vkAcquirePerformanceConfigurationINTEL" ) );
      vkAcquireProfilingLockKHR =
        PFN_vkAcquireProfilingLockKHR( vkGetInstanceProcAddr( instance, "vkAcquireProfilingLockKHR" ) );
      vkAllocateCommandBuffers =
        PFN_vkAllocateCommandBuffers( vkGetInstanceProcAddr( instance, "vkAllocateCommandBuffers" ) );
      vkAllocateDescriptorSets =
        PFN_vkAllocateDescriptorSets( vkGetInstanceProcAddr( instance, "vkAllocateDescriptorSets" ) );
      vkAllocateMemory     = PFN_vkAllocateMemory( vkGetInstanceProcAddr( instance, "vkAllocateMemory" ) );
      vkBeginCommandBuffer = PFN_vkBeginCommandBuffer( vkGetInstanceProcAddr( instance, "vkBeginCommandBuffer" ) );
      vkBindAccelerationStructureMemoryNV = PFN_vkBindAccelerationStructureMemoryNV(
        vkGetInstanceProcAddr( instance, "vkBindAccelerationStructureMemoryNV" ) );
      vkBindBufferMemory  = PFN_vkBindBufferMemory( vkGetInstanceProcAddr( instance, "vkBindBufferMemory" ) );
      vkBindBufferMemory2 = PFN_vkBindBufferMemory2( vkGetInstanceProcAddr( instance, "vkBindBufferMemory2" ) );
      vkBindBufferMemory2KHR =
        PFN_vkBindBufferMemory2KHR( vkGetInstanceProcAddr( instance, "vkBindBufferMemory2KHR" ) );
      if ( !vkBindBufferMemory2 )
        vkBindBufferMemory2 = vkBindBufferMemory2KHR;
      vkBindImageMemory     = PFN_vkBindImageMemory( vkGetInstanceProcAddr( instance, "vkBindImageMemory" ) );
      vkBindImageMemory2    = PFN_vkBindImageMemory2( vkGetInstanceProcAddr( instance, "vkBindImageMemory2" ) );
      vkBindImageMemory2KHR = PFN_vkBindImageMemory2KHR( vkGetInstanceProcAddr( instance, "vkBindImageMemory2KHR" ) );
      if ( !vkBindImageMemory2 )
        vkBindImageMemory2 = vkBindImageMemory2KHR;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkBindVideoSessionMemoryKHR =
        PFN_vkBindVideoSessionMemoryKHR( vkGetInstanceProcAddr( instance, "vkBindVideoSessionMemoryKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkBuildAccelerationStructuresKHR =
        PFN_vkBuildAccelerationStructuresKHR( vkGetInstanceProcAddr( instance, "vkBuildAccelerationStructuresKHR" ) );
      vkCmdBeginConditionalRenderingEXT =
        PFN_vkCmdBeginConditionalRenderingEXT( vkGetInstanceProcAddr( instance, "vkCmdBeginConditionalRenderingEXT" ) );
      vkCmdBeginDebugUtilsLabelEXT =
        PFN_vkCmdBeginDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkCmdBeginDebugUtilsLabelEXT" ) );
      vkCmdBeginQuery = PFN_vkCmdBeginQuery( vkGetInstanceProcAddr( instance, "vkCmdBeginQuery" ) );
      vkCmdBeginQueryIndexedEXT =
        PFN_vkCmdBeginQueryIndexedEXT( vkGetInstanceProcAddr( instance, "vkCmdBeginQueryIndexedEXT" ) );
      vkCmdBeginRenderPass  = PFN_vkCmdBeginRenderPass( vkGetInstanceProcAddr( instance, "vkCmdBeginRenderPass" ) );
      vkCmdBeginRenderPass2 = PFN_vkCmdBeginRenderPass2( vkGetInstanceProcAddr( instance, "vkCmdBeginRenderPass2" ) );
      vkCmdBeginRenderPass2KHR =
        PFN_vkCmdBeginRenderPass2KHR( vkGetInstanceProcAddr( instance, "vkCmdBeginRenderPass2KHR" ) );
      if ( !vkCmdBeginRenderPass2 )
        vkCmdBeginRenderPass2 = vkCmdBeginRenderPass2KHR;
      vkCmdBeginTransformFeedbackEXT =
        PFN_vkCmdBeginTransformFeedbackEXT( vkGetInstanceProcAddr( instance, "vkCmdBeginTransformFeedbackEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdBeginVideoCodingKHR =
        PFN_vkCmdBeginVideoCodingKHR( vkGetInstanceProcAddr( instance, "vkCmdBeginVideoCodingKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdBindDescriptorSets =
        PFN_vkCmdBindDescriptorSets( vkGetInstanceProcAddr( instance, "vkCmdBindDescriptorSets" ) );
      vkCmdBindIndexBuffer = PFN_vkCmdBindIndexBuffer( vkGetInstanceProcAddr( instance, "vkCmdBindIndexBuffer" ) );
      vkCmdBindPipeline    = PFN_vkCmdBindPipeline( vkGetInstanceProcAddr( instance, "vkCmdBindPipeline" ) );
      vkCmdBindPipelineShaderGroupNV =
        PFN_vkCmdBindPipelineShaderGroupNV( vkGetInstanceProcAddr( instance, "vkCmdBindPipelineShaderGroupNV" ) );
      vkCmdBindShadingRateImageNV =
        PFN_vkCmdBindShadingRateImageNV( vkGetInstanceProcAddr( instance, "vkCmdBindShadingRateImageNV" ) );
      vkCmdBindTransformFeedbackBuffersEXT = PFN_vkCmdBindTransformFeedbackBuffersEXT(
        vkGetInstanceProcAddr( instance, "vkCmdBindTransformFeedbackBuffersEXT" ) );
      vkCmdBindVertexBuffers =
        PFN_vkCmdBindVertexBuffers( vkGetInstanceProcAddr( instance, "vkCmdBindVertexBuffers" ) );
      vkCmdBindVertexBuffers2EXT =
        PFN_vkCmdBindVertexBuffers2EXT( vkGetInstanceProcAddr( instance, "vkCmdBindVertexBuffers2EXT" ) );
      vkCmdBlitImage     = PFN_vkCmdBlitImage( vkGetInstanceProcAddr( instance, "vkCmdBlitImage" ) );
      vkCmdBlitImage2KHR = PFN_vkCmdBlitImage2KHR( vkGetInstanceProcAddr( instance, "vkCmdBlitImage2KHR" ) );
      vkCmdBuildAccelerationStructureNV =
        PFN_vkCmdBuildAccelerationStructureNV( vkGetInstanceProcAddr( instance, "vkCmdBuildAccelerationStructureNV" ) );
      vkCmdBuildAccelerationStructuresIndirectKHR = PFN_vkCmdBuildAccelerationStructuresIndirectKHR(
        vkGetInstanceProcAddr( instance, "vkCmdBuildAccelerationStructuresIndirectKHR" ) );
      vkCmdBuildAccelerationStructuresKHR = PFN_vkCmdBuildAccelerationStructuresKHR(
        vkGetInstanceProcAddr( instance, "vkCmdBuildAccelerationStructuresKHR" ) );
      vkCmdClearAttachments = PFN_vkCmdClearAttachments( vkGetInstanceProcAddr( instance, "vkCmdClearAttachments" ) );
      vkCmdClearColorImage  = PFN_vkCmdClearColorImage( vkGetInstanceProcAddr( instance, "vkCmdClearColorImage" ) );
      vkCmdClearDepthStencilImage =
        PFN_vkCmdClearDepthStencilImage( vkGetInstanceProcAddr( instance, "vkCmdClearDepthStencilImage" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdControlVideoCodingKHR =
        PFN_vkCmdControlVideoCodingKHR( vkGetInstanceProcAddr( instance, "vkCmdControlVideoCodingKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdCopyAccelerationStructureKHR =
        PFN_vkCmdCopyAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkCmdCopyAccelerationStructureKHR" ) );
      vkCmdCopyAccelerationStructureNV =
        PFN_vkCmdCopyAccelerationStructureNV( vkGetInstanceProcAddr( instance, "vkCmdCopyAccelerationStructureNV" ) );
      vkCmdCopyAccelerationStructureToMemoryKHR = PFN_vkCmdCopyAccelerationStructureToMemoryKHR(
        vkGetInstanceProcAddr( instance, "vkCmdCopyAccelerationStructureToMemoryKHR" ) );
      vkCmdCopyBuffer     = PFN_vkCmdCopyBuffer( vkGetInstanceProcAddr( instance, "vkCmdCopyBuffer" ) );
      vkCmdCopyBuffer2KHR = PFN_vkCmdCopyBuffer2KHR( vkGetInstanceProcAddr( instance, "vkCmdCopyBuffer2KHR" ) );
      vkCmdCopyBufferToImage =
        PFN_vkCmdCopyBufferToImage( vkGetInstanceProcAddr( instance, "vkCmdCopyBufferToImage" ) );
      vkCmdCopyBufferToImage2KHR =
        PFN_vkCmdCopyBufferToImage2KHR( vkGetInstanceProcAddr( instance, "vkCmdCopyBufferToImage2KHR" ) );
      vkCmdCopyImage     = PFN_vkCmdCopyImage( vkGetInstanceProcAddr( instance, "vkCmdCopyImage" ) );
      vkCmdCopyImage2KHR = PFN_vkCmdCopyImage2KHR( vkGetInstanceProcAddr( instance, "vkCmdCopyImage2KHR" ) );
      vkCmdCopyImageToBuffer =
        PFN_vkCmdCopyImageToBuffer( vkGetInstanceProcAddr( instance, "vkCmdCopyImageToBuffer" ) );
      vkCmdCopyImageToBuffer2KHR =
        PFN_vkCmdCopyImageToBuffer2KHR( vkGetInstanceProcAddr( instance, "vkCmdCopyImageToBuffer2KHR" ) );
      vkCmdCopyMemoryToAccelerationStructureKHR = PFN_vkCmdCopyMemoryToAccelerationStructureKHR(
        vkGetInstanceProcAddr( instance, "vkCmdCopyMemoryToAccelerationStructureKHR" ) );
      vkCmdCopyQueryPoolResults =
        PFN_vkCmdCopyQueryPoolResults( vkGetInstanceProcAddr( instance, "vkCmdCopyQueryPoolResults" ) );
      vkCmdCuLaunchKernelNVX =
        PFN_vkCmdCuLaunchKernelNVX( vkGetInstanceProcAddr( instance, "vkCmdCuLaunchKernelNVX" ) );
      vkCmdDebugMarkerBeginEXT =
        PFN_vkCmdDebugMarkerBeginEXT( vkGetInstanceProcAddr( instance, "vkCmdDebugMarkerBeginEXT" ) );
      vkCmdDebugMarkerEndEXT =
        PFN_vkCmdDebugMarkerEndEXT( vkGetInstanceProcAddr( instance, "vkCmdDebugMarkerEndEXT" ) );
      vkCmdDebugMarkerInsertEXT =
        PFN_vkCmdDebugMarkerInsertEXT( vkGetInstanceProcAddr( instance, "vkCmdDebugMarkerInsertEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdDecodeVideoKHR = PFN_vkCmdDecodeVideoKHR( vkGetInstanceProcAddr( instance, "vkCmdDecodeVideoKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdDispatch        = PFN_vkCmdDispatch( vkGetInstanceProcAddr( instance, "vkCmdDispatch" ) );
      vkCmdDispatchBase    = PFN_vkCmdDispatchBase( vkGetInstanceProcAddr( instance, "vkCmdDispatchBase" ) );
      vkCmdDispatchBaseKHR = PFN_vkCmdDispatchBaseKHR( vkGetInstanceProcAddr( instance, "vkCmdDispatchBaseKHR" ) );
      if ( !vkCmdDispatchBase )
        vkCmdDispatchBase = vkCmdDispatchBaseKHR;
      vkCmdDispatchIndirect = PFN_vkCmdDispatchIndirect( vkGetInstanceProcAddr( instance, "vkCmdDispatchIndirect" ) );
      vkCmdDraw             = PFN_vkCmdDraw( vkGetInstanceProcAddr( instance, "vkCmdDraw" ) );
      vkCmdDrawIndexed      = PFN_vkCmdDrawIndexed( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexed" ) );
      vkCmdDrawIndexedIndirect =
        PFN_vkCmdDrawIndexedIndirect( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexedIndirect" ) );
      vkCmdDrawIndexedIndirectCount =
        PFN_vkCmdDrawIndexedIndirectCount( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexedIndirectCount" ) );
      vkCmdDrawIndexedIndirectCountAMD =
        PFN_vkCmdDrawIndexedIndirectCountAMD( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexedIndirectCountAMD" ) );
      if ( !vkCmdDrawIndexedIndirectCount )
        vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountAMD;
      vkCmdDrawIndexedIndirectCountKHR =
        PFN_vkCmdDrawIndexedIndirectCountKHR( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexedIndirectCountKHR" ) );
      if ( !vkCmdDrawIndexedIndirectCount )
        vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountKHR;
      vkCmdDrawIndirect = PFN_vkCmdDrawIndirect( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirect" ) );
      vkCmdDrawIndirectByteCountEXT =
        PFN_vkCmdDrawIndirectByteCountEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirectByteCountEXT" ) );
      vkCmdDrawIndirectCount =
        PFN_vkCmdDrawIndirectCount( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirectCount" ) );
      vkCmdDrawIndirectCountAMD =
        PFN_vkCmdDrawIndirectCountAMD( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirectCountAMD" ) );
      if ( !vkCmdDrawIndirectCount )
        vkCmdDrawIndirectCount = vkCmdDrawIndirectCountAMD;
      vkCmdDrawIndirectCountKHR =
        PFN_vkCmdDrawIndirectCountKHR( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirectCountKHR" ) );
      if ( !vkCmdDrawIndirectCount )
        vkCmdDrawIndirectCount = vkCmdDrawIndirectCountKHR;
      vkCmdDrawMeshTasksIndirectCountNV =
        PFN_vkCmdDrawMeshTasksIndirectCountNV( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksIndirectCountNV" ) );
      vkCmdDrawMeshTasksIndirectNV =
        PFN_vkCmdDrawMeshTasksIndirectNV( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksIndirectNV" ) );
      vkCmdDrawMeshTasksNV = PFN_vkCmdDrawMeshTasksNV( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksNV" ) );
      vkCmdDrawMultiEXT    = PFN_vkCmdDrawMultiEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawMultiEXT" ) );
      vkCmdDrawMultiIndexedEXT =
        PFN_vkCmdDrawMultiIndexedEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawMultiIndexedEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdEncodeVideoKHR = PFN_vkCmdEncodeVideoKHR( vkGetInstanceProcAddr( instance, "vkCmdEncodeVideoKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdEndConditionalRenderingEXT =
        PFN_vkCmdEndConditionalRenderingEXT( vkGetInstanceProcAddr( instance, "vkCmdEndConditionalRenderingEXT" ) );
      vkCmdEndDebugUtilsLabelEXT =
        PFN_vkCmdEndDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkCmdEndDebugUtilsLabelEXT" ) );
      vkCmdEndQuery = PFN_vkCmdEndQuery( vkGetInstanceProcAddr( instance, "vkCmdEndQuery" ) );
      vkCmdEndQueryIndexedEXT =
        PFN_vkCmdEndQueryIndexedEXT( vkGetInstanceProcAddr( instance, "vkCmdEndQueryIndexedEXT" ) );
      vkCmdEndRenderPass  = PFN_vkCmdEndRenderPass( vkGetInstanceProcAddr( instance, "vkCmdEndRenderPass" ) );
      vkCmdEndRenderPass2 = PFN_vkCmdEndRenderPass2( vkGetInstanceProcAddr( instance, "vkCmdEndRenderPass2" ) );
      vkCmdEndRenderPass2KHR =
        PFN_vkCmdEndRenderPass2KHR( vkGetInstanceProcAddr( instance, "vkCmdEndRenderPass2KHR" ) );
      if ( !vkCmdEndRenderPass2 )
        vkCmdEndRenderPass2 = vkCmdEndRenderPass2KHR;
      vkCmdEndTransformFeedbackEXT =
        PFN_vkCmdEndTransformFeedbackEXT( vkGetInstanceProcAddr( instance, "vkCmdEndTransformFeedbackEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdEndVideoCodingKHR =
        PFN_vkCmdEndVideoCodingKHR( vkGetInstanceProcAddr( instance, "vkCmdEndVideoCodingKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdExecuteCommands = PFN_vkCmdExecuteCommands( vkGetInstanceProcAddr( instance, "vkCmdExecuteCommands" ) );
      vkCmdExecuteGeneratedCommandsNV =
        PFN_vkCmdExecuteGeneratedCommandsNV( vkGetInstanceProcAddr( instance, "vkCmdExecuteGeneratedCommandsNV" ) );
      vkCmdFillBuffer = PFN_vkCmdFillBuffer( vkGetInstanceProcAddr( instance, "vkCmdFillBuffer" ) );
      vkCmdInsertDebugUtilsLabelEXT =
        PFN_vkCmdInsertDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkCmdInsertDebugUtilsLabelEXT" ) );
      vkCmdNextSubpass     = PFN_vkCmdNextSubpass( vkGetInstanceProcAddr( instance, "vkCmdNextSubpass" ) );
      vkCmdNextSubpass2    = PFN_vkCmdNextSubpass2( vkGetInstanceProcAddr( instance, "vkCmdNextSubpass2" ) );
      vkCmdNextSubpass2KHR = PFN_vkCmdNextSubpass2KHR( vkGetInstanceProcAddr( instance, "vkCmdNextSubpass2KHR" ) );
      if ( !vkCmdNextSubpass2 )
        vkCmdNextSubpass2 = vkCmdNextSubpass2KHR;
      vkCmdPipelineBarrier = PFN_vkCmdPipelineBarrier( vkGetInstanceProcAddr( instance, "vkCmdPipelineBarrier" ) );
      vkCmdPipelineBarrier2KHR =
        PFN_vkCmdPipelineBarrier2KHR( vkGetInstanceProcAddr( instance, "vkCmdPipelineBarrier2KHR" ) );
      vkCmdPreprocessGeneratedCommandsNV = PFN_vkCmdPreprocessGeneratedCommandsNV(
        vkGetInstanceProcAddr( instance, "vkCmdPreprocessGeneratedCommandsNV" ) );
      vkCmdPushConstants = PFN_vkCmdPushConstants( vkGetInstanceProcAddr( instance, "vkCmdPushConstants" ) );
      vkCmdPushDescriptorSetKHR =
        PFN_vkCmdPushDescriptorSetKHR( vkGetInstanceProcAddr( instance, "vkCmdPushDescriptorSetKHR" ) );
      vkCmdPushDescriptorSetWithTemplateKHR = PFN_vkCmdPushDescriptorSetWithTemplateKHR(
        vkGetInstanceProcAddr( instance, "vkCmdPushDescriptorSetWithTemplateKHR" ) );
      vkCmdResetEvent       = PFN_vkCmdResetEvent( vkGetInstanceProcAddr( instance, "vkCmdResetEvent" ) );
      vkCmdResetEvent2KHR   = PFN_vkCmdResetEvent2KHR( vkGetInstanceProcAddr( instance, "vkCmdResetEvent2KHR" ) );
      vkCmdResetQueryPool   = PFN_vkCmdResetQueryPool( vkGetInstanceProcAddr( instance, "vkCmdResetQueryPool" ) );
      vkCmdResolveImage     = PFN_vkCmdResolveImage( vkGetInstanceProcAddr( instance, "vkCmdResolveImage" ) );
      vkCmdResolveImage2KHR = PFN_vkCmdResolveImage2KHR( vkGetInstanceProcAddr( instance, "vkCmdResolveImage2KHR" ) );
      vkCmdSetBlendConstants =
        PFN_vkCmdSetBlendConstants( vkGetInstanceProcAddr( instance, "vkCmdSetBlendConstants" ) );
      vkCmdSetCheckpointNV = PFN_vkCmdSetCheckpointNV( vkGetInstanceProcAddr( instance, "vkCmdSetCheckpointNV" ) );
      vkCmdSetCoarseSampleOrderNV =
        PFN_vkCmdSetCoarseSampleOrderNV( vkGetInstanceProcAddr( instance, "vkCmdSetCoarseSampleOrderNV" ) );
      vkCmdSetColorWriteEnableEXT =
        PFN_vkCmdSetColorWriteEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetColorWriteEnableEXT" ) );
      vkCmdSetCullModeEXT = PFN_vkCmdSetCullModeEXT( vkGetInstanceProcAddr( instance, "vkCmdSetCullModeEXT" ) );
      vkCmdSetDepthBias   = PFN_vkCmdSetDepthBias( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBias" ) );
      vkCmdSetDepthBiasEnableEXT =
        PFN_vkCmdSetDepthBiasEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBiasEnableEXT" ) );
      vkCmdSetDepthBounds = PFN_vkCmdSetDepthBounds( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBounds" ) );
      vkCmdSetDepthBoundsTestEnableEXT =
        PFN_vkCmdSetDepthBoundsTestEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBoundsTestEnableEXT" ) );
      vkCmdSetDepthCompareOpEXT =
        PFN_vkCmdSetDepthCompareOpEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthCompareOpEXT" ) );
      vkCmdSetDepthTestEnableEXT =
        PFN_vkCmdSetDepthTestEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthTestEnableEXT" ) );
      vkCmdSetDepthWriteEnableEXT =
        PFN_vkCmdSetDepthWriteEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthWriteEnableEXT" ) );
      vkCmdSetDeviceMask    = PFN_vkCmdSetDeviceMask( vkGetInstanceProcAddr( instance, "vkCmdSetDeviceMask" ) );
      vkCmdSetDeviceMaskKHR = PFN_vkCmdSetDeviceMaskKHR( vkGetInstanceProcAddr( instance, "vkCmdSetDeviceMaskKHR" ) );
      if ( !vkCmdSetDeviceMask )
        vkCmdSetDeviceMask = vkCmdSetDeviceMaskKHR;
      vkCmdSetDiscardRectangleEXT =
        PFN_vkCmdSetDiscardRectangleEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDiscardRectangleEXT" ) );
      vkCmdSetEvent     = PFN_vkCmdSetEvent( vkGetInstanceProcAddr( instance, "vkCmdSetEvent" ) );
      vkCmdSetEvent2KHR = PFN_vkCmdSetEvent2KHR( vkGetInstanceProcAddr( instance, "vkCmdSetEvent2KHR" ) );
      vkCmdSetExclusiveScissorNV =
        PFN_vkCmdSetExclusiveScissorNV( vkGetInstanceProcAddr( instance, "vkCmdSetExclusiveScissorNV" ) );
      vkCmdSetFragmentShadingRateEnumNV =
        PFN_vkCmdSetFragmentShadingRateEnumNV( vkGetInstanceProcAddr( instance, "vkCmdSetFragmentShadingRateEnumNV" ) );
      vkCmdSetFragmentShadingRateKHR =
        PFN_vkCmdSetFragmentShadingRateKHR( vkGetInstanceProcAddr( instance, "vkCmdSetFragmentShadingRateKHR" ) );
      vkCmdSetFrontFaceEXT = PFN_vkCmdSetFrontFaceEXT( vkGetInstanceProcAddr( instance, "vkCmdSetFrontFaceEXT" ) );
      vkCmdSetLineStippleEXT =
        PFN_vkCmdSetLineStippleEXT( vkGetInstanceProcAddr( instance, "vkCmdSetLineStippleEXT" ) );
      vkCmdSetLineWidth  = PFN_vkCmdSetLineWidth( vkGetInstanceProcAddr( instance, "vkCmdSetLineWidth" ) );
      vkCmdSetLogicOpEXT = PFN_vkCmdSetLogicOpEXT( vkGetInstanceProcAddr( instance, "vkCmdSetLogicOpEXT" ) );
      vkCmdSetPatchControlPointsEXT =
        PFN_vkCmdSetPatchControlPointsEXT( vkGetInstanceProcAddr( instance, "vkCmdSetPatchControlPointsEXT" ) );
      vkCmdSetPerformanceMarkerINTEL =
        PFN_vkCmdSetPerformanceMarkerINTEL( vkGetInstanceProcAddr( instance, "vkCmdSetPerformanceMarkerINTEL" ) );
      vkCmdSetPerformanceOverrideINTEL =
        PFN_vkCmdSetPerformanceOverrideINTEL( vkGetInstanceProcAddr( instance, "vkCmdSetPerformanceOverrideINTEL" ) );
      vkCmdSetPerformanceStreamMarkerINTEL = PFN_vkCmdSetPerformanceStreamMarkerINTEL(
        vkGetInstanceProcAddr( instance, "vkCmdSetPerformanceStreamMarkerINTEL" ) );
      vkCmdSetPrimitiveRestartEnableEXT =
        PFN_vkCmdSetPrimitiveRestartEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetPrimitiveRestartEnableEXT" ) );
      vkCmdSetPrimitiveTopologyEXT =
        PFN_vkCmdSetPrimitiveTopologyEXT( vkGetInstanceProcAddr( instance, "vkCmdSetPrimitiveTopologyEXT" ) );
      vkCmdSetRasterizerDiscardEnableEXT = PFN_vkCmdSetRasterizerDiscardEnableEXT(
        vkGetInstanceProcAddr( instance, "vkCmdSetRasterizerDiscardEnableEXT" ) );
      vkCmdSetRayTracingPipelineStackSizeKHR = PFN_vkCmdSetRayTracingPipelineStackSizeKHR(
        vkGetInstanceProcAddr( instance, "vkCmdSetRayTracingPipelineStackSizeKHR" ) );
      vkCmdSetSampleLocationsEXT =
        PFN_vkCmdSetSampleLocationsEXT( vkGetInstanceProcAddr( instance, "vkCmdSetSampleLocationsEXT" ) );
      vkCmdSetScissor = PFN_vkCmdSetScissor( vkGetInstanceProcAddr( instance, "vkCmdSetScissor" ) );
      vkCmdSetScissorWithCountEXT =
        PFN_vkCmdSetScissorWithCountEXT( vkGetInstanceProcAddr( instance, "vkCmdSetScissorWithCountEXT" ) );
      vkCmdSetStencilCompareMask =
        PFN_vkCmdSetStencilCompareMask( vkGetInstanceProcAddr( instance, "vkCmdSetStencilCompareMask" ) );
      vkCmdSetStencilOpEXT = PFN_vkCmdSetStencilOpEXT( vkGetInstanceProcAddr( instance, "vkCmdSetStencilOpEXT" ) );
      vkCmdSetStencilReference =
        PFN_vkCmdSetStencilReference( vkGetInstanceProcAddr( instance, "vkCmdSetStencilReference" ) );
      vkCmdSetStencilTestEnableEXT =
        PFN_vkCmdSetStencilTestEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetStencilTestEnableEXT" ) );
      vkCmdSetStencilWriteMask =
        PFN_vkCmdSetStencilWriteMask( vkGetInstanceProcAddr( instance, "vkCmdSetStencilWriteMask" ) );
      vkCmdSetVertexInputEXT =
        PFN_vkCmdSetVertexInputEXT( vkGetInstanceProcAddr( instance, "vkCmdSetVertexInputEXT" ) );
      vkCmdSetViewport = PFN_vkCmdSetViewport( vkGetInstanceProcAddr( instance, "vkCmdSetViewport" ) );
      vkCmdSetViewportShadingRatePaletteNV = PFN_vkCmdSetViewportShadingRatePaletteNV(
        vkGetInstanceProcAddr( instance, "vkCmdSetViewportShadingRatePaletteNV" ) );
      vkCmdSetViewportWScalingNV =
        PFN_vkCmdSetViewportWScalingNV( vkGetInstanceProcAddr( instance, "vkCmdSetViewportWScalingNV" ) );
      vkCmdSetViewportWithCountEXT =
        PFN_vkCmdSetViewportWithCountEXT( vkGetInstanceProcAddr( instance, "vkCmdSetViewportWithCountEXT" ) );
      vkCmdSubpassShadingHUAWEI =
        PFN_vkCmdSubpassShadingHUAWEI( vkGetInstanceProcAddr( instance, "vkCmdSubpassShadingHUAWEI" ) );
      vkCmdTraceRaysIndirectKHR =
        PFN_vkCmdTraceRaysIndirectKHR( vkGetInstanceProcAddr( instance, "vkCmdTraceRaysIndirectKHR" ) );
      vkCmdTraceRaysKHR   = PFN_vkCmdTraceRaysKHR( vkGetInstanceProcAddr( instance, "vkCmdTraceRaysKHR" ) );
      vkCmdTraceRaysNV    = PFN_vkCmdTraceRaysNV( vkGetInstanceProcAddr( instance, "vkCmdTraceRaysNV" ) );
      vkCmdUpdateBuffer   = PFN_vkCmdUpdateBuffer( vkGetInstanceProcAddr( instance, "vkCmdUpdateBuffer" ) );
      vkCmdWaitEvents     = PFN_vkCmdWaitEvents( vkGetInstanceProcAddr( instance, "vkCmdWaitEvents" ) );
      vkCmdWaitEvents2KHR = PFN_vkCmdWaitEvents2KHR( vkGetInstanceProcAddr( instance, "vkCmdWaitEvents2KHR" ) );
      vkCmdWriteAccelerationStructuresPropertiesKHR = PFN_vkCmdWriteAccelerationStructuresPropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkCmdWriteAccelerationStructuresPropertiesKHR" ) );
      vkCmdWriteAccelerationStructuresPropertiesNV = PFN_vkCmdWriteAccelerationStructuresPropertiesNV(
        vkGetInstanceProcAddr( instance, "vkCmdWriteAccelerationStructuresPropertiesNV" ) );
      vkCmdWriteBufferMarker2AMD =
        PFN_vkCmdWriteBufferMarker2AMD( vkGetInstanceProcAddr( instance, "vkCmdWriteBufferMarker2AMD" ) );
      vkCmdWriteBufferMarkerAMD =
        PFN_vkCmdWriteBufferMarkerAMD( vkGetInstanceProcAddr( instance, "vkCmdWriteBufferMarkerAMD" ) );
      vkCmdWriteTimestamp = PFN_vkCmdWriteTimestamp( vkGetInstanceProcAddr( instance, "vkCmdWriteTimestamp" ) );
      vkCmdWriteTimestamp2KHR =
        PFN_vkCmdWriteTimestamp2KHR( vkGetInstanceProcAddr( instance, "vkCmdWriteTimestamp2KHR" ) );
      vkCompileDeferredNV = PFN_vkCompileDeferredNV( vkGetInstanceProcAddr( instance, "vkCompileDeferredNV" ) );
      vkCopyAccelerationStructureKHR =
        PFN_vkCopyAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkCopyAccelerationStructureKHR" ) );
      vkCopyAccelerationStructureToMemoryKHR = PFN_vkCopyAccelerationStructureToMemoryKHR(
        vkGetInstanceProcAddr( instance, "vkCopyAccelerationStructureToMemoryKHR" ) );
      vkCopyMemoryToAccelerationStructureKHR = PFN_vkCopyMemoryToAccelerationStructureKHR(
        vkGetInstanceProcAddr( instance, "vkCopyMemoryToAccelerationStructureKHR" ) );
      vkCreateAccelerationStructureKHR =
        PFN_vkCreateAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkCreateAccelerationStructureKHR" ) );
      vkCreateAccelerationStructureNV =
        PFN_vkCreateAccelerationStructureNV( vkGetInstanceProcAddr( instance, "vkCreateAccelerationStructureNV" ) );
      vkCreateBuffer      = PFN_vkCreateBuffer( vkGetInstanceProcAddr( instance, "vkCreateBuffer" ) );
      vkCreateBufferView  = PFN_vkCreateBufferView( vkGetInstanceProcAddr( instance, "vkCreateBufferView" ) );
      vkCreateCommandPool = PFN_vkCreateCommandPool( vkGetInstanceProcAddr( instance, "vkCreateCommandPool" ) );
      vkCreateComputePipelines =
        PFN_vkCreateComputePipelines( vkGetInstanceProcAddr( instance, "vkCreateComputePipelines" ) );
      vkCreateCuFunctionNVX = PFN_vkCreateCuFunctionNVX( vkGetInstanceProcAddr( instance, "vkCreateCuFunctionNVX" ) );
      vkCreateCuModuleNVX   = PFN_vkCreateCuModuleNVX( vkGetInstanceProcAddr( instance, "vkCreateCuModuleNVX" ) );
      vkCreateDeferredOperationKHR =
        PFN_vkCreateDeferredOperationKHR( vkGetInstanceProcAddr( instance, "vkCreateDeferredOperationKHR" ) );
      vkCreateDescriptorPool =
        PFN_vkCreateDescriptorPool( vkGetInstanceProcAddr( instance, "vkCreateDescriptorPool" ) );
      vkCreateDescriptorSetLayout =
        PFN_vkCreateDescriptorSetLayout( vkGetInstanceProcAddr( instance, "vkCreateDescriptorSetLayout" ) );
      vkCreateDescriptorUpdateTemplate =
        PFN_vkCreateDescriptorUpdateTemplate( vkGetInstanceProcAddr( instance, "vkCreateDescriptorUpdateTemplate" ) );
      vkCreateDescriptorUpdateTemplateKHR = PFN_vkCreateDescriptorUpdateTemplateKHR(
        vkGetInstanceProcAddr( instance, "vkCreateDescriptorUpdateTemplateKHR" ) );
      if ( !vkCreateDescriptorUpdateTemplate )
        vkCreateDescriptorUpdateTemplate = vkCreateDescriptorUpdateTemplateKHR;
      vkCreateEvent       = PFN_vkCreateEvent( vkGetInstanceProcAddr( instance, "vkCreateEvent" ) );
      vkCreateFence       = PFN_vkCreateFence( vkGetInstanceProcAddr( instance, "vkCreateFence" ) );
      vkCreateFramebuffer = PFN_vkCreateFramebuffer( vkGetInstanceProcAddr( instance, "vkCreateFramebuffer" ) );
      vkCreateGraphicsPipelines =
        PFN_vkCreateGraphicsPipelines( vkGetInstanceProcAddr( instance, "vkCreateGraphicsPipelines" ) );
      vkCreateImage     = PFN_vkCreateImage( vkGetInstanceProcAddr( instance, "vkCreateImage" ) );
      vkCreateImageView = PFN_vkCreateImageView( vkGetInstanceProcAddr( instance, "vkCreateImageView" ) );
      vkCreateIndirectCommandsLayoutNV =
        PFN_vkCreateIndirectCommandsLayoutNV( vkGetInstanceProcAddr( instance, "vkCreateIndirectCommandsLayoutNV" ) );
      vkCreatePipelineCache = PFN_vkCreatePipelineCache( vkGetInstanceProcAddr( instance, "vkCreatePipelineCache" ) );
      vkCreatePipelineLayout =
        PFN_vkCreatePipelineLayout( vkGetInstanceProcAddr( instance, "vkCreatePipelineLayout" ) );
      vkCreatePrivateDataSlotEXT =
        PFN_vkCreatePrivateDataSlotEXT( vkGetInstanceProcAddr( instance, "vkCreatePrivateDataSlotEXT" ) );
      vkCreateQueryPool = PFN_vkCreateQueryPool( vkGetInstanceProcAddr( instance, "vkCreateQueryPool" ) );
      vkCreateRayTracingPipelinesKHR =
        PFN_vkCreateRayTracingPipelinesKHR( vkGetInstanceProcAddr( instance, "vkCreateRayTracingPipelinesKHR" ) );
      vkCreateRayTracingPipelinesNV =
        PFN_vkCreateRayTracingPipelinesNV( vkGetInstanceProcAddr( instance, "vkCreateRayTracingPipelinesNV" ) );
      vkCreateRenderPass  = PFN_vkCreateRenderPass( vkGetInstanceProcAddr( instance, "vkCreateRenderPass" ) );
      vkCreateRenderPass2 = PFN_vkCreateRenderPass2( vkGetInstanceProcAddr( instance, "vkCreateRenderPass2" ) );
      vkCreateRenderPass2KHR =
        PFN_vkCreateRenderPass2KHR( vkGetInstanceProcAddr( instance, "vkCreateRenderPass2KHR" ) );
      if ( !vkCreateRenderPass2 )
        vkCreateRenderPass2 = vkCreateRenderPass2KHR;
      vkCreateSampler = PFN_vkCreateSampler( vkGetInstanceProcAddr( instance, "vkCreateSampler" ) );
      vkCreateSamplerYcbcrConversion =
        PFN_vkCreateSamplerYcbcrConversion( vkGetInstanceProcAddr( instance, "vkCreateSamplerYcbcrConversion" ) );
      vkCreateSamplerYcbcrConversionKHR =
        PFN_vkCreateSamplerYcbcrConversionKHR( vkGetInstanceProcAddr( instance, "vkCreateSamplerYcbcrConversionKHR" ) );
      if ( !vkCreateSamplerYcbcrConversion )
        vkCreateSamplerYcbcrConversion = vkCreateSamplerYcbcrConversionKHR;
      vkCreateSemaphore    = PFN_vkCreateSemaphore( vkGetInstanceProcAddr( instance, "vkCreateSemaphore" ) );
      vkCreateShaderModule = PFN_vkCreateShaderModule( vkGetInstanceProcAddr( instance, "vkCreateShaderModule" ) );
      vkCreateSharedSwapchainsKHR =
        PFN_vkCreateSharedSwapchainsKHR( vkGetInstanceProcAddr( instance, "vkCreateSharedSwapchainsKHR" ) );
      vkCreateSwapchainKHR = PFN_vkCreateSwapchainKHR( vkGetInstanceProcAddr( instance, "vkCreateSwapchainKHR" ) );
      vkCreateValidationCacheEXT =
        PFN_vkCreateValidationCacheEXT( vkGetInstanceProcAddr( instance, "vkCreateValidationCacheEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCreateVideoSessionKHR =
        PFN_vkCreateVideoSessionKHR( vkGetInstanceProcAddr( instance, "vkCreateVideoSessionKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCreateVideoSessionParametersKHR =
        PFN_vkCreateVideoSessionParametersKHR( vkGetInstanceProcAddr( instance, "vkCreateVideoSessionParametersKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkDebugMarkerSetObjectNameEXT =
        PFN_vkDebugMarkerSetObjectNameEXT( vkGetInstanceProcAddr( instance, "vkDebugMarkerSetObjectNameEXT" ) );
      vkDebugMarkerSetObjectTagEXT =
        PFN_vkDebugMarkerSetObjectTagEXT( vkGetInstanceProcAddr( instance, "vkDebugMarkerSetObjectTagEXT" ) );
      vkDeferredOperationJoinKHR =
        PFN_vkDeferredOperationJoinKHR( vkGetInstanceProcAddr( instance, "vkDeferredOperationJoinKHR" ) );
      vkDestroyAccelerationStructureKHR =
        PFN_vkDestroyAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkDestroyAccelerationStructureKHR" ) );
      vkDestroyAccelerationStructureNV =
        PFN_vkDestroyAccelerationStructureNV( vkGetInstanceProcAddr( instance, "vkDestroyAccelerationStructureNV" ) );
      vkDestroyBuffer      = PFN_vkDestroyBuffer( vkGetInstanceProcAddr( instance, "vkDestroyBuffer" ) );
      vkDestroyBufferView  = PFN_vkDestroyBufferView( vkGetInstanceProcAddr( instance, "vkDestroyBufferView" ) );
      vkDestroyCommandPool = PFN_vkDestroyCommandPool( vkGetInstanceProcAddr( instance, "vkDestroyCommandPool" ) );
      vkDestroyCuFunctionNVX =
        PFN_vkDestroyCuFunctionNVX( vkGetInstanceProcAddr( instance, "vkDestroyCuFunctionNVX" ) );
      vkDestroyCuModuleNVX = PFN_vkDestroyCuModuleNVX( vkGetInstanceProcAddr( instance, "vkDestroyCuModuleNVX" ) );
      vkDestroyDeferredOperationKHR =
        PFN_vkDestroyDeferredOperationKHR( vkGetInstanceProcAddr( instance, "vkDestroyDeferredOperationKHR" ) );
      vkDestroyDescriptorPool =
        PFN_vkDestroyDescriptorPool( vkGetInstanceProcAddr( instance, "vkDestroyDescriptorPool" ) );
      vkDestroyDescriptorSetLayout =
        PFN_vkDestroyDescriptorSetLayout( vkGetInstanceProcAddr( instance, "vkDestroyDescriptorSetLayout" ) );
      vkDestroyDescriptorUpdateTemplate =
        PFN_vkDestroyDescriptorUpdateTemplate( vkGetInstanceProcAddr( instance, "vkDestroyDescriptorUpdateTemplate" ) );
      vkDestroyDescriptorUpdateTemplateKHR = PFN_vkDestroyDescriptorUpdateTemplateKHR(
        vkGetInstanceProcAddr( instance, "vkDestroyDescriptorUpdateTemplateKHR" ) );
      if ( !vkDestroyDescriptorUpdateTemplate )
        vkDestroyDescriptorUpdateTemplate = vkDestroyDescriptorUpdateTemplateKHR;
      vkDestroyDevice      = PFN_vkDestroyDevice( vkGetInstanceProcAddr( instance, "vkDestroyDevice" ) );
      vkDestroyEvent       = PFN_vkDestroyEvent( vkGetInstanceProcAddr( instance, "vkDestroyEvent" ) );
      vkDestroyFence       = PFN_vkDestroyFence( vkGetInstanceProcAddr( instance, "vkDestroyFence" ) );
      vkDestroyFramebuffer = PFN_vkDestroyFramebuffer( vkGetInstanceProcAddr( instance, "vkDestroyFramebuffer" ) );
      vkDestroyImage       = PFN_vkDestroyImage( vkGetInstanceProcAddr( instance, "vkDestroyImage" ) );
      vkDestroyImageView   = PFN_vkDestroyImageView( vkGetInstanceProcAddr( instance, "vkDestroyImageView" ) );
      vkDestroyIndirectCommandsLayoutNV =
        PFN_vkDestroyIndirectCommandsLayoutNV( vkGetInstanceProcAddr( instance, "vkDestroyIndirectCommandsLayoutNV" ) );
      vkDestroyPipeline = PFN_vkDestroyPipeline( vkGetInstanceProcAddr( instance, "vkDestroyPipeline" ) );
      vkDestroyPipelineCache =
        PFN_vkDestroyPipelineCache( vkGetInstanceProcAddr( instance, "vkDestroyPipelineCache" ) );
      vkDestroyPipelineLayout =
        PFN_vkDestroyPipelineLayout( vkGetInstanceProcAddr( instance, "vkDestroyPipelineLayout" ) );
      vkDestroyPrivateDataSlotEXT =
        PFN_vkDestroyPrivateDataSlotEXT( vkGetInstanceProcAddr( instance, "vkDestroyPrivateDataSlotEXT" ) );
      vkDestroyQueryPool  = PFN_vkDestroyQueryPool( vkGetInstanceProcAddr( instance, "vkDestroyQueryPool" ) );
      vkDestroyRenderPass = PFN_vkDestroyRenderPass( vkGetInstanceProcAddr( instance, "vkDestroyRenderPass" ) );
      vkDestroySampler    = PFN_vkDestroySampler( vkGetInstanceProcAddr( instance, "vkDestroySampler" ) );
      vkDestroySamplerYcbcrConversion =
        PFN_vkDestroySamplerYcbcrConversion( vkGetInstanceProcAddr( instance, "vkDestroySamplerYcbcrConversion" ) );
      vkDestroySamplerYcbcrConversionKHR = PFN_vkDestroySamplerYcbcrConversionKHR(
        vkGetInstanceProcAddr( instance, "vkDestroySamplerYcbcrConversionKHR" ) );
      if ( !vkDestroySamplerYcbcrConversion )
        vkDestroySamplerYcbcrConversion = vkDestroySamplerYcbcrConversionKHR;
      vkDestroySemaphore    = PFN_vkDestroySemaphore( vkGetInstanceProcAddr( instance, "vkDestroySemaphore" ) );
      vkDestroyShaderModule = PFN_vkDestroyShaderModule( vkGetInstanceProcAddr( instance, "vkDestroyShaderModule" ) );
      vkDestroySwapchainKHR = PFN_vkDestroySwapchainKHR( vkGetInstanceProcAddr( instance, "vkDestroySwapchainKHR" ) );
      vkDestroyValidationCacheEXT =
        PFN_vkDestroyValidationCacheEXT( vkGetInstanceProcAddr( instance, "vkDestroyValidationCacheEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkDestroyVideoSessionKHR =
        PFN_vkDestroyVideoSessionKHR( vkGetInstanceProcAddr( instance, "vkDestroyVideoSessionKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkDestroyVideoSessionParametersKHR = PFN_vkDestroyVideoSessionParametersKHR(
        vkGetInstanceProcAddr( instance, "vkDestroyVideoSessionParametersKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkDeviceWaitIdle = PFN_vkDeviceWaitIdle( vkGetInstanceProcAddr( instance, "vkDeviceWaitIdle" ) );
      vkDisplayPowerControlEXT =
        PFN_vkDisplayPowerControlEXT( vkGetInstanceProcAddr( instance, "vkDisplayPowerControlEXT" ) );
      vkEndCommandBuffer = PFN_vkEndCommandBuffer( vkGetInstanceProcAddr( instance, "vkEndCommandBuffer" ) );
      vkFlushMappedMemoryRanges =
        PFN_vkFlushMappedMemoryRanges( vkGetInstanceProcAddr( instance, "vkFlushMappedMemoryRanges" ) );
      vkFreeCommandBuffers = PFN_vkFreeCommandBuffers( vkGetInstanceProcAddr( instance, "vkFreeCommandBuffers" ) );
      vkFreeDescriptorSets = PFN_vkFreeDescriptorSets( vkGetInstanceProcAddr( instance, "vkFreeDescriptorSets" ) );
      vkFreeMemory         = PFN_vkFreeMemory( vkGetInstanceProcAddr( instance, "vkFreeMemory" ) );
      vkGetAccelerationStructureBuildSizesKHR = PFN_vkGetAccelerationStructureBuildSizesKHR(
        vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureBuildSizesKHR" ) );
      vkGetAccelerationStructureDeviceAddressKHR = PFN_vkGetAccelerationStructureDeviceAddressKHR(
        vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureDeviceAddressKHR" ) );
      vkGetAccelerationStructureHandleNV = PFN_vkGetAccelerationStructureHandleNV(
        vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureHandleNV" ) );
      vkGetAccelerationStructureMemoryRequirementsNV = PFN_vkGetAccelerationStructureMemoryRequirementsNV(
        vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureMemoryRequirementsNV" ) );
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      vkGetAndroidHardwareBufferPropertiesANDROID = PFN_vkGetAndroidHardwareBufferPropertiesANDROID(
        vkGetInstanceProcAddr( instance, "vkGetAndroidHardwareBufferPropertiesANDROID" ) );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      vkGetBufferDeviceAddress =
        PFN_vkGetBufferDeviceAddress( vkGetInstanceProcAddr( instance, "vkGetBufferDeviceAddress" ) );
      vkGetBufferDeviceAddressEXT =
        PFN_vkGetBufferDeviceAddressEXT( vkGetInstanceProcAddr( instance, "vkGetBufferDeviceAddressEXT" ) );
      if ( !vkGetBufferDeviceAddress )
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressEXT;
      vkGetBufferDeviceAddressKHR =
        PFN_vkGetBufferDeviceAddressKHR( vkGetInstanceProcAddr( instance, "vkGetBufferDeviceAddressKHR" ) );
      if ( !vkGetBufferDeviceAddress )
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressKHR;
      vkGetBufferMemoryRequirements =
        PFN_vkGetBufferMemoryRequirements( vkGetInstanceProcAddr( instance, "vkGetBufferMemoryRequirements" ) );
      vkGetBufferMemoryRequirements2 =
        PFN_vkGetBufferMemoryRequirements2( vkGetInstanceProcAddr( instance, "vkGetBufferMemoryRequirements2" ) );
      vkGetBufferMemoryRequirements2KHR =
        PFN_vkGetBufferMemoryRequirements2KHR( vkGetInstanceProcAddr( instance, "vkGetBufferMemoryRequirements2KHR" ) );
      if ( !vkGetBufferMemoryRequirements2 )
        vkGetBufferMemoryRequirements2 = vkGetBufferMemoryRequirements2KHR;
      vkGetBufferOpaqueCaptureAddress =
        PFN_vkGetBufferOpaqueCaptureAddress( vkGetInstanceProcAddr( instance, "vkGetBufferOpaqueCaptureAddress" ) );
      vkGetBufferOpaqueCaptureAddressKHR = PFN_vkGetBufferOpaqueCaptureAddressKHR(
        vkGetInstanceProcAddr( instance, "vkGetBufferOpaqueCaptureAddressKHR" ) );
      if ( !vkGetBufferOpaqueCaptureAddress )
        vkGetBufferOpaqueCaptureAddress = vkGetBufferOpaqueCaptureAddressKHR;
      vkGetCalibratedTimestampsEXT =
        PFN_vkGetCalibratedTimestampsEXT( vkGetInstanceProcAddr( instance, "vkGetCalibratedTimestampsEXT" ) );
      vkGetDeferredOperationMaxConcurrencyKHR = PFN_vkGetDeferredOperationMaxConcurrencyKHR(
        vkGetInstanceProcAddr( instance, "vkGetDeferredOperationMaxConcurrencyKHR" ) );
      vkGetDeferredOperationResultKHR =
        PFN_vkGetDeferredOperationResultKHR( vkGetInstanceProcAddr( instance, "vkGetDeferredOperationResultKHR" ) );
      vkGetDescriptorSetLayoutSupport =
        PFN_vkGetDescriptorSetLayoutSupport( vkGetInstanceProcAddr( instance, "vkGetDescriptorSetLayoutSupport" ) );
      vkGetDescriptorSetLayoutSupportKHR = PFN_vkGetDescriptorSetLayoutSupportKHR(
        vkGetInstanceProcAddr( instance, "vkGetDescriptorSetLayoutSupportKHR" ) );
      if ( !vkGetDescriptorSetLayoutSupport )
        vkGetDescriptorSetLayoutSupport = vkGetDescriptorSetLayoutSupportKHR;
      vkGetDeviceAccelerationStructureCompatibilityKHR = PFN_vkGetDeviceAccelerationStructureCompatibilityKHR(
        vkGetInstanceProcAddr( instance, "vkGetDeviceAccelerationStructureCompatibilityKHR" ) );
      vkGetDeviceGroupPeerMemoryFeatures = PFN_vkGetDeviceGroupPeerMemoryFeatures(
        vkGetInstanceProcAddr( instance, "vkGetDeviceGroupPeerMemoryFeatures" ) );
      vkGetDeviceGroupPeerMemoryFeaturesKHR = PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR(
        vkGetInstanceProcAddr( instance, "vkGetDeviceGroupPeerMemoryFeaturesKHR" ) );
      if ( !vkGetDeviceGroupPeerMemoryFeatures )
        vkGetDeviceGroupPeerMemoryFeatures = vkGetDeviceGroupPeerMemoryFeaturesKHR;
      vkGetDeviceGroupPresentCapabilitiesKHR = PFN_vkGetDeviceGroupPresentCapabilitiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetDeviceGroupPresentCapabilitiesKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetDeviceGroupSurfacePresentModes2EXT = PFN_vkGetDeviceGroupSurfacePresentModes2EXT(
        vkGetInstanceProcAddr( instance, "vkGetDeviceGroupSurfacePresentModes2EXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkGetDeviceGroupSurfacePresentModesKHR = PFN_vkGetDeviceGroupSurfacePresentModesKHR(
        vkGetInstanceProcAddr( instance, "vkGetDeviceGroupSurfacePresentModesKHR" ) );
      vkGetDeviceMemoryCommitment =
        PFN_vkGetDeviceMemoryCommitment( vkGetInstanceProcAddr( instance, "vkGetDeviceMemoryCommitment" ) );
      vkGetDeviceMemoryOpaqueCaptureAddress = PFN_vkGetDeviceMemoryOpaqueCaptureAddress(
        vkGetInstanceProcAddr( instance, "vkGetDeviceMemoryOpaqueCaptureAddress" ) );
      vkGetDeviceMemoryOpaqueCaptureAddressKHR = PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR(
        vkGetInstanceProcAddr( instance, "vkGetDeviceMemoryOpaqueCaptureAddressKHR" ) );
      if ( !vkGetDeviceMemoryOpaqueCaptureAddress )
        vkGetDeviceMemoryOpaqueCaptureAddress = vkGetDeviceMemoryOpaqueCaptureAddressKHR;
      vkGetDeviceProcAddr = PFN_vkGetDeviceProcAddr( vkGetInstanceProcAddr( instance, "vkGetDeviceProcAddr" ) );
      vkGetDeviceQueue    = PFN_vkGetDeviceQueue( vkGetInstanceProcAddr( instance, "vkGetDeviceQueue" ) );
      vkGetDeviceQueue2   = PFN_vkGetDeviceQueue2( vkGetInstanceProcAddr( instance, "vkGetDeviceQueue2" ) );
      vkGetEventStatus    = PFN_vkGetEventStatus( vkGetInstanceProcAddr( instance, "vkGetEventStatus" ) );
      vkGetFenceFdKHR     = PFN_vkGetFenceFdKHR( vkGetInstanceProcAddr( instance, "vkGetFenceFdKHR" ) );
      vkGetFenceStatus    = PFN_vkGetFenceStatus( vkGetInstanceProcAddr( instance, "vkGetFenceStatus" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetFenceWin32HandleKHR =
        PFN_vkGetFenceWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkGetFenceWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkGetGeneratedCommandsMemoryRequirementsNV = PFN_vkGetGeneratedCommandsMemoryRequirementsNV(
        vkGetInstanceProcAddr( instance, "vkGetGeneratedCommandsMemoryRequirementsNV" ) );
      vkGetImageDrmFormatModifierPropertiesEXT = PFN_vkGetImageDrmFormatModifierPropertiesEXT(
        vkGetInstanceProcAddr( instance, "vkGetImageDrmFormatModifierPropertiesEXT" ) );
      vkGetImageMemoryRequirements =
        PFN_vkGetImageMemoryRequirements( vkGetInstanceProcAddr( instance, "vkGetImageMemoryRequirements" ) );
      vkGetImageMemoryRequirements2 =
        PFN_vkGetImageMemoryRequirements2( vkGetInstanceProcAddr( instance, "vkGetImageMemoryRequirements2" ) );
      vkGetImageMemoryRequirements2KHR =
        PFN_vkGetImageMemoryRequirements2KHR( vkGetInstanceProcAddr( instance, "vkGetImageMemoryRequirements2KHR" ) );
      if ( !vkGetImageMemoryRequirements2 )
        vkGetImageMemoryRequirements2 = vkGetImageMemoryRequirements2KHR;
      vkGetImageSparseMemoryRequirements = PFN_vkGetImageSparseMemoryRequirements(
        vkGetInstanceProcAddr( instance, "vkGetImageSparseMemoryRequirements" ) );
      vkGetImageSparseMemoryRequirements2 = PFN_vkGetImageSparseMemoryRequirements2(
        vkGetInstanceProcAddr( instance, "vkGetImageSparseMemoryRequirements2" ) );
      vkGetImageSparseMemoryRequirements2KHR = PFN_vkGetImageSparseMemoryRequirements2KHR(
        vkGetInstanceProcAddr( instance, "vkGetImageSparseMemoryRequirements2KHR" ) );
      if ( !vkGetImageSparseMemoryRequirements2 )
        vkGetImageSparseMemoryRequirements2 = vkGetImageSparseMemoryRequirements2KHR;
      vkGetImageSubresourceLayout =
        PFN_vkGetImageSubresourceLayout( vkGetInstanceProcAddr( instance, "vkGetImageSubresourceLayout" ) );
      vkGetImageViewAddressNVX =
        PFN_vkGetImageViewAddressNVX( vkGetInstanceProcAddr( instance, "vkGetImageViewAddressNVX" ) );
      vkGetImageViewHandleNVX =
        PFN_vkGetImageViewHandleNVX( vkGetInstanceProcAddr( instance, "vkGetImageViewHandleNVX" ) );
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      vkGetMemoryAndroidHardwareBufferANDROID = PFN_vkGetMemoryAndroidHardwareBufferANDROID(
        vkGetInstanceProcAddr( instance, "vkGetMemoryAndroidHardwareBufferANDROID" ) );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      vkGetMemoryFdKHR = PFN_vkGetMemoryFdKHR( vkGetInstanceProcAddr( instance, "vkGetMemoryFdKHR" ) );
      vkGetMemoryFdPropertiesKHR =
        PFN_vkGetMemoryFdPropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetMemoryFdPropertiesKHR" ) );
      vkGetMemoryHostPointerPropertiesEXT = PFN_vkGetMemoryHostPointerPropertiesEXT(
        vkGetInstanceProcAddr( instance, "vkGetMemoryHostPointerPropertiesEXT" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetMemoryWin32HandleKHR =
        PFN_vkGetMemoryWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkGetMemoryWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetMemoryWin32HandleNV =
        PFN_vkGetMemoryWin32HandleNV( vkGetInstanceProcAddr( instance, "vkGetMemoryWin32HandleNV" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetMemoryWin32HandlePropertiesKHR = PFN_vkGetMemoryWin32HandlePropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetMemoryWin32HandlePropertiesKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkGetMemoryZirconHandleFUCHSIA =
        PFN_vkGetMemoryZirconHandleFUCHSIA( vkGetInstanceProcAddr( instance, "vkGetMemoryZirconHandleFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkGetMemoryZirconHandlePropertiesFUCHSIA = PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA(
        vkGetInstanceProcAddr( instance, "vkGetMemoryZirconHandlePropertiesFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      vkGetPastPresentationTimingGOOGLE =
        PFN_vkGetPastPresentationTimingGOOGLE( vkGetInstanceProcAddr( instance, "vkGetPastPresentationTimingGOOGLE" ) );
      vkGetPerformanceParameterINTEL =
        PFN_vkGetPerformanceParameterINTEL( vkGetInstanceProcAddr( instance, "vkGetPerformanceParameterINTEL" ) );
      vkGetPipelineCacheData =
        PFN_vkGetPipelineCacheData( vkGetInstanceProcAddr( instance, "vkGetPipelineCacheData" ) );
      vkGetPipelineExecutableInternalRepresentationsKHR = PFN_vkGetPipelineExecutableInternalRepresentationsKHR(
        vkGetInstanceProcAddr( instance, "vkGetPipelineExecutableInternalRepresentationsKHR" ) );
      vkGetPipelineExecutablePropertiesKHR = PFN_vkGetPipelineExecutablePropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPipelineExecutablePropertiesKHR" ) );
      vkGetPipelineExecutableStatisticsKHR = PFN_vkGetPipelineExecutableStatisticsKHR(
        vkGetInstanceProcAddr( instance, "vkGetPipelineExecutableStatisticsKHR" ) );
      vkGetPrivateDataEXT   = PFN_vkGetPrivateDataEXT( vkGetInstanceProcAddr( instance, "vkGetPrivateDataEXT" ) );
      vkGetQueryPoolResults = PFN_vkGetQueryPoolResults( vkGetInstanceProcAddr( instance, "vkGetQueryPoolResults" ) );
      vkGetQueueCheckpointData2NV =
        PFN_vkGetQueueCheckpointData2NV( vkGetInstanceProcAddr( instance, "vkGetQueueCheckpointData2NV" ) );
      vkGetQueueCheckpointDataNV =
        PFN_vkGetQueueCheckpointDataNV( vkGetInstanceProcAddr( instance, "vkGetQueueCheckpointDataNV" ) );
      vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(
        vkGetInstanceProcAddr( instance, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR" ) );
      vkGetRayTracingShaderGroupHandlesKHR = PFN_vkGetRayTracingShaderGroupHandlesKHR(
        vkGetInstanceProcAddr( instance, "vkGetRayTracingShaderGroupHandlesKHR" ) );
      vkGetRayTracingShaderGroupHandlesNV = PFN_vkGetRayTracingShaderGroupHandlesNV(
        vkGetInstanceProcAddr( instance, "vkGetRayTracingShaderGroupHandlesNV" ) );
      if ( !vkGetRayTracingShaderGroupHandlesKHR )
        vkGetRayTracingShaderGroupHandlesKHR = vkGetRayTracingShaderGroupHandlesNV;
      vkGetRayTracingShaderGroupStackSizeKHR = PFN_vkGetRayTracingShaderGroupStackSizeKHR(
        vkGetInstanceProcAddr( instance, "vkGetRayTracingShaderGroupStackSizeKHR" ) );
      vkGetRefreshCycleDurationGOOGLE =
        PFN_vkGetRefreshCycleDurationGOOGLE( vkGetInstanceProcAddr( instance, "vkGetRefreshCycleDurationGOOGLE" ) );
      vkGetRenderAreaGranularity =
        PFN_vkGetRenderAreaGranularity( vkGetInstanceProcAddr( instance, "vkGetRenderAreaGranularity" ) );
      vkGetSemaphoreCounterValue =
        PFN_vkGetSemaphoreCounterValue( vkGetInstanceProcAddr( instance, "vkGetSemaphoreCounterValue" ) );
      vkGetSemaphoreCounterValueKHR =
        PFN_vkGetSemaphoreCounterValueKHR( vkGetInstanceProcAddr( instance, "vkGetSemaphoreCounterValueKHR" ) );
      if ( !vkGetSemaphoreCounterValue )
        vkGetSemaphoreCounterValue = vkGetSemaphoreCounterValueKHR;
      vkGetSemaphoreFdKHR = PFN_vkGetSemaphoreFdKHR( vkGetInstanceProcAddr( instance, "vkGetSemaphoreFdKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetSemaphoreWin32HandleKHR =
        PFN_vkGetSemaphoreWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkGetSemaphoreWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkGetSemaphoreZirconHandleFUCHSIA =
        PFN_vkGetSemaphoreZirconHandleFUCHSIA( vkGetInstanceProcAddr( instance, "vkGetSemaphoreZirconHandleFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      vkGetShaderInfoAMD = PFN_vkGetShaderInfoAMD( vkGetInstanceProcAddr( instance, "vkGetShaderInfoAMD" ) );
      vkGetSubpassShadingMaxWorkgroupSizeHUAWEI = PFN_vkGetSubpassShadingMaxWorkgroupSizeHUAWEI(
        vkGetInstanceProcAddr( instance, "vkGetSubpassShadingMaxWorkgroupSizeHUAWEI" ) );
      vkGetSwapchainCounterEXT =
        PFN_vkGetSwapchainCounterEXT( vkGetInstanceProcAddr( instance, "vkGetSwapchainCounterEXT" ) );
      vkGetSwapchainImagesKHR =
        PFN_vkGetSwapchainImagesKHR( vkGetInstanceProcAddr( instance, "vkGetSwapchainImagesKHR" ) );
      vkGetSwapchainStatusKHR =
        PFN_vkGetSwapchainStatusKHR( vkGetInstanceProcAddr( instance, "vkGetSwapchainStatusKHR" ) );
      vkGetValidationCacheDataEXT =
        PFN_vkGetValidationCacheDataEXT( vkGetInstanceProcAddr( instance, "vkGetValidationCacheDataEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkGetVideoSessionMemoryRequirementsKHR = PFN_vkGetVideoSessionMemoryRequirementsKHR(
        vkGetInstanceProcAddr( instance, "vkGetVideoSessionMemoryRequirementsKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkImportFenceFdKHR = PFN_vkImportFenceFdKHR( vkGetInstanceProcAddr( instance, "vkImportFenceFdKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkImportFenceWin32HandleKHR =
        PFN_vkImportFenceWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkImportFenceWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkImportSemaphoreFdKHR =
        PFN_vkImportSemaphoreFdKHR( vkGetInstanceProcAddr( instance, "vkImportSemaphoreFdKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkImportSemaphoreWin32HandleKHR =
        PFN_vkImportSemaphoreWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkImportSemaphoreWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkImportSemaphoreZirconHandleFUCHSIA = PFN_vkImportSemaphoreZirconHandleFUCHSIA(
        vkGetInstanceProcAddr( instance, "vkImportSemaphoreZirconHandleFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      vkInitializePerformanceApiINTEL =
        PFN_vkInitializePerformanceApiINTEL( vkGetInstanceProcAddr( instance, "vkInitializePerformanceApiINTEL" ) );
      vkInvalidateMappedMemoryRanges =
        PFN_vkInvalidateMappedMemoryRanges( vkGetInstanceProcAddr( instance, "vkInvalidateMappedMemoryRanges" ) );
      vkMapMemory           = PFN_vkMapMemory( vkGetInstanceProcAddr( instance, "vkMapMemory" ) );
      vkMergePipelineCaches = PFN_vkMergePipelineCaches( vkGetInstanceProcAddr( instance, "vkMergePipelineCaches" ) );
      vkMergeValidationCachesEXT =
        PFN_vkMergeValidationCachesEXT( vkGetInstanceProcAddr( instance, "vkMergeValidationCachesEXT" ) );
      vkQueueBeginDebugUtilsLabelEXT =
        PFN_vkQueueBeginDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkQueueBeginDebugUtilsLabelEXT" ) );
      vkQueueBindSparse = PFN_vkQueueBindSparse( vkGetInstanceProcAddr( instance, "vkQueueBindSparse" ) );
      vkQueueEndDebugUtilsLabelEXT =
        PFN_vkQueueEndDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkQueueEndDebugUtilsLabelEXT" ) );
      vkQueueInsertDebugUtilsLabelEXT =
        PFN_vkQueueInsertDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkQueueInsertDebugUtilsLabelEXT" ) );
      vkQueuePresentKHR = PFN_vkQueuePresentKHR( vkGetInstanceProcAddr( instance, "vkQueuePresentKHR" ) );
      vkQueueSetPerformanceConfigurationINTEL = PFN_vkQueueSetPerformanceConfigurationINTEL(
        vkGetInstanceProcAddr( instance, "vkQueueSetPerformanceConfigurationINTEL" ) );
      vkQueueSubmit     = PFN_vkQueueSubmit( vkGetInstanceProcAddr( instance, "vkQueueSubmit" ) );
      vkQueueSubmit2KHR = PFN_vkQueueSubmit2KHR( vkGetInstanceProcAddr( instance, "vkQueueSubmit2KHR" ) );
      vkQueueWaitIdle   = PFN_vkQueueWaitIdle( vkGetInstanceProcAddr( instance, "vkQueueWaitIdle" ) );
      vkRegisterDeviceEventEXT =
        PFN_vkRegisterDeviceEventEXT( vkGetInstanceProcAddr( instance, "vkRegisterDeviceEventEXT" ) );
      vkRegisterDisplayEventEXT =
        PFN_vkRegisterDisplayEventEXT( vkGetInstanceProcAddr( instance, "vkRegisterDisplayEventEXT" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkReleaseFullScreenExclusiveModeEXT = PFN_vkReleaseFullScreenExclusiveModeEXT(
        vkGetInstanceProcAddr( instance, "vkReleaseFullScreenExclusiveModeEXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkReleasePerformanceConfigurationINTEL = PFN_vkReleasePerformanceConfigurationINTEL(
        vkGetInstanceProcAddr( instance, "vkReleasePerformanceConfigurationINTEL" ) );
      vkReleaseProfilingLockKHR =
        PFN_vkReleaseProfilingLockKHR( vkGetInstanceProcAddr( instance, "vkReleaseProfilingLockKHR" ) );
      vkResetCommandBuffer  = PFN_vkResetCommandBuffer( vkGetInstanceProcAddr( instance, "vkResetCommandBuffer" ) );
      vkResetCommandPool    = PFN_vkResetCommandPool( vkGetInstanceProcAddr( instance, "vkResetCommandPool" ) );
      vkResetDescriptorPool = PFN_vkResetDescriptorPool( vkGetInstanceProcAddr( instance, "vkResetDescriptorPool" ) );
      vkResetEvent          = PFN_vkResetEvent( vkGetInstanceProcAddr( instance, "vkResetEvent" ) );
      vkResetFences         = PFN_vkResetFences( vkGetInstanceProcAddr( instance, "vkResetFences" ) );
      vkResetQueryPool      = PFN_vkResetQueryPool( vkGetInstanceProcAddr( instance, "vkResetQueryPool" ) );
      vkResetQueryPoolEXT   = PFN_vkResetQueryPoolEXT( vkGetInstanceProcAddr( instance, "vkResetQueryPoolEXT" ) );
      if ( !vkResetQueryPool )
        vkResetQueryPool = vkResetQueryPoolEXT;
      vkSetDebugUtilsObjectNameEXT =
        PFN_vkSetDebugUtilsObjectNameEXT( vkGetInstanceProcAddr( instance, "vkSetDebugUtilsObjectNameEXT" ) );
      vkSetDebugUtilsObjectTagEXT =
        PFN_vkSetDebugUtilsObjectTagEXT( vkGetInstanceProcAddr( instance, "vkSetDebugUtilsObjectTagEXT" ) );
      vkSetEvent           = PFN_vkSetEvent( vkGetInstanceProcAddr( instance, "vkSetEvent" ) );
      vkSetHdrMetadataEXT  = PFN_vkSetHdrMetadataEXT( vkGetInstanceProcAddr( instance, "vkSetHdrMetadataEXT" ) );
      vkSetLocalDimmingAMD = PFN_vkSetLocalDimmingAMD( vkGetInstanceProcAddr( instance, "vkSetLocalDimmingAMD" ) );
      vkSetPrivateDataEXT  = PFN_vkSetPrivateDataEXT( vkGetInstanceProcAddr( instance, "vkSetPrivateDataEXT" ) );
      vkSignalSemaphore    = PFN_vkSignalSemaphore( vkGetInstanceProcAddr( instance, "vkSignalSemaphore" ) );
      vkSignalSemaphoreKHR = PFN_vkSignalSemaphoreKHR( vkGetInstanceProcAddr( instance, "vkSignalSemaphoreKHR" ) );
      if ( !vkSignalSemaphore )
        vkSignalSemaphore = vkSignalSemaphoreKHR;
      vkTrimCommandPool    = PFN_vkTrimCommandPool( vkGetInstanceProcAddr( instance, "vkTrimCommandPool" ) );
      vkTrimCommandPoolKHR = PFN_vkTrimCommandPoolKHR( vkGetInstanceProcAddr( instance, "vkTrimCommandPoolKHR" ) );
      if ( !vkTrimCommandPool )
        vkTrimCommandPool = vkTrimCommandPoolKHR;
      vkUninitializePerformanceApiINTEL =
        PFN_vkUninitializePerformanceApiINTEL( vkGetInstanceProcAddr( instance, "vkUninitializePerformanceApiINTEL" ) );
      vkUnmapMemory = PFN_vkUnmapMemory( vkGetInstanceProcAddr( instance, "vkUnmapMemory" ) );
      vkUpdateDescriptorSetWithTemplate =
        PFN_vkUpdateDescriptorSetWithTemplate( vkGetInstanceProcAddr( instance, "vkUpdateDescriptorSetWithTemplate" ) );
      vkUpdateDescriptorSetWithTemplateKHR = PFN_vkUpdateDescriptorSetWithTemplateKHR(
        vkGetInstanceProcAddr( instance, "vkUpdateDescriptorSetWithTemplateKHR" ) );
      if ( !vkUpdateDescriptorSetWithTemplate )
        vkUpdateDescriptorSetWithTemplate = vkUpdateDescriptorSetWithTemplateKHR;
      vkUpdateDescriptorSets =
        PFN_vkUpdateDescriptorSets( vkGetInstanceProcAddr( instance, "vkUpdateDescriptorSets" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkUpdateVideoSessionParametersKHR =
        PFN_vkUpdateVideoSessionParametersKHR( vkGetInstanceProcAddr( instance, "vkUpdateVideoSessionParametersKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkWaitForFences     = PFN_vkWaitForFences( vkGetInstanceProcAddr( instance, "vkWaitForFences" ) );
      vkWaitSemaphores    = PFN_vkWaitSemaphores( vkGetInstanceProcAddr( instance, "vkWaitSemaphores" ) );
      vkWaitSemaphoresKHR = PFN_vkWaitSemaphoresKHR( vkGetInstanceProcAddr( instance, "vkWaitSemaphoresKHR" ) );
      if ( !vkWaitSemaphores )
        vkWaitSemaphores = vkWaitSemaphoresKHR;
      vkWriteAccelerationStructuresPropertiesKHR = PFN_vkWriteAccelerationStructuresPropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkWriteAccelerationStructuresPropertiesKHR" ) );
    }

    void init( VULKAN_HPP_NAMESPACE::Device deviceCpp ) VULKAN_HPP_NOEXCEPT
    {
      VkDevice device = static_cast<VkDevice>( deviceCpp );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkAcquireFullScreenExclusiveModeEXT =
        PFN_vkAcquireFullScreenExclusiveModeEXT( vkGetDeviceProcAddr( device, "vkAcquireFullScreenExclusiveModeEXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkAcquireNextImage2KHR = PFN_vkAcquireNextImage2KHR( vkGetDeviceProcAddr( device, "vkAcquireNextImage2KHR" ) );
      vkAcquireNextImageKHR  = PFN_vkAcquireNextImageKHR( vkGetDeviceProcAddr( device, "vkAcquireNextImageKHR" ) );
      vkAcquirePerformanceConfigurationINTEL = PFN_vkAcquirePerformanceConfigurationINTEL(
        vkGetDeviceProcAddr( device, "vkAcquirePerformanceConfigurationINTEL" ) );
      vkAcquireProfilingLockKHR =
        PFN_vkAcquireProfilingLockKHR( vkGetDeviceProcAddr( device, "vkAcquireProfilingLockKHR" ) );
      vkAllocateCommandBuffers =
        PFN_vkAllocateCommandBuffers( vkGetDeviceProcAddr( device, "vkAllocateCommandBuffers" ) );
      vkAllocateDescriptorSets =
        PFN_vkAllocateDescriptorSets( vkGetDeviceProcAddr( device, "vkAllocateDescriptorSets" ) );
      vkAllocateMemory     = PFN_vkAllocateMemory( vkGetDeviceProcAddr( device, "vkAllocateMemory" ) );
      vkBeginCommandBuffer = PFN_vkBeginCommandBuffer( vkGetDeviceProcAddr( device, "vkBeginCommandBuffer" ) );
      vkBindAccelerationStructureMemoryNV =
        PFN_vkBindAccelerationStructureMemoryNV( vkGetDeviceProcAddr( device, "vkBindAccelerationStructureMemoryNV" ) );
      vkBindBufferMemory     = PFN_vkBindBufferMemory( vkGetDeviceProcAddr( device, "vkBindBufferMemory" ) );
      vkBindBufferMemory2    = PFN_vkBindBufferMemory2( vkGetDeviceProcAddr( device, "vkBindBufferMemory2" ) );
      vkBindBufferMemory2KHR = PFN_vkBindBufferMemory2KHR( vkGetDeviceProcAddr( device, "vkBindBufferMemory2KHR" ) );
      if ( !vkBindBufferMemory2 )
        vkBindBufferMemory2 = vkBindBufferMemory2KHR;
      vkBindImageMemory     = PFN_vkBindImageMemory( vkGetDeviceProcAddr( device, "vkBindImageMemory" ) );
      vkBindImageMemory2    = PFN_vkBindImageMemory2( vkGetDeviceProcAddr( device, "vkBindImageMemory2" ) );
      vkBindImageMemory2KHR = PFN_vkBindImageMemory2KHR( vkGetDeviceProcAddr( device, "vkBindImageMemory2KHR" ) );
      if ( !vkBindImageMemory2 )
        vkBindImageMemory2 = vkBindImageMemory2KHR;
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkBindVideoSessionMemoryKHR =
        PFN_vkBindVideoSessionMemoryKHR( vkGetDeviceProcAddr( device, "vkBindVideoSessionMemoryKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkBuildAccelerationStructuresKHR =
        PFN_vkBuildAccelerationStructuresKHR( vkGetDeviceProcAddr( device, "vkBuildAccelerationStructuresKHR" ) );
      vkCmdBeginConditionalRenderingEXT =
        PFN_vkCmdBeginConditionalRenderingEXT( vkGetDeviceProcAddr( device, "vkCmdBeginConditionalRenderingEXT" ) );
      vkCmdBeginDebugUtilsLabelEXT =
        PFN_vkCmdBeginDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdBeginDebugUtilsLabelEXT" ) );
      vkCmdBeginQuery = PFN_vkCmdBeginQuery( vkGetDeviceProcAddr( device, "vkCmdBeginQuery" ) );
      vkCmdBeginQueryIndexedEXT =
        PFN_vkCmdBeginQueryIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdBeginQueryIndexedEXT" ) );
      vkCmdBeginRenderPass  = PFN_vkCmdBeginRenderPass( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass" ) );
      vkCmdBeginRenderPass2 = PFN_vkCmdBeginRenderPass2( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass2" ) );
      vkCmdBeginRenderPass2KHR =
        PFN_vkCmdBeginRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass2KHR" ) );
      if ( !vkCmdBeginRenderPass2 )
        vkCmdBeginRenderPass2 = vkCmdBeginRenderPass2KHR;
      vkCmdBeginTransformFeedbackEXT =
        PFN_vkCmdBeginTransformFeedbackEXT( vkGetDeviceProcAddr( device, "vkCmdBeginTransformFeedbackEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdBeginVideoCodingKHR =
        PFN_vkCmdBeginVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdBeginVideoCodingKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdBindDescriptorSets = PFN_vkCmdBindDescriptorSets( vkGetDeviceProcAddr( device, "vkCmdBindDescriptorSets" ) );
      vkCmdBindIndexBuffer    = PFN_vkCmdBindIndexBuffer( vkGetDeviceProcAddr( device, "vkCmdBindIndexBuffer" ) );
      vkCmdBindPipeline       = PFN_vkCmdBindPipeline( vkGetDeviceProcAddr( device, "vkCmdBindPipeline" ) );
      vkCmdBindPipelineShaderGroupNV =
        PFN_vkCmdBindPipelineShaderGroupNV( vkGetDeviceProcAddr( device, "vkCmdBindPipelineShaderGroupNV" ) );
      vkCmdBindShadingRateImageNV =
        PFN_vkCmdBindShadingRateImageNV( vkGetDeviceProcAddr( device, "vkCmdBindShadingRateImageNV" ) );
      vkCmdBindTransformFeedbackBuffersEXT = PFN_vkCmdBindTransformFeedbackBuffersEXT(
        vkGetDeviceProcAddr( device, "vkCmdBindTransformFeedbackBuffersEXT" ) );
      vkCmdBindVertexBuffers = PFN_vkCmdBindVertexBuffers( vkGetDeviceProcAddr( device, "vkCmdBindVertexBuffers" ) );
      vkCmdBindVertexBuffers2EXT =
        PFN_vkCmdBindVertexBuffers2EXT( vkGetDeviceProcAddr( device, "vkCmdBindVertexBuffers2EXT" ) );
      vkCmdBlitImage     = PFN_vkCmdBlitImage( vkGetDeviceProcAddr( device, "vkCmdBlitImage" ) );
      vkCmdBlitImage2KHR = PFN_vkCmdBlitImage2KHR( vkGetDeviceProcAddr( device, "vkCmdBlitImage2KHR" ) );
      vkCmdBuildAccelerationStructureNV =
        PFN_vkCmdBuildAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructureNV" ) );
      vkCmdBuildAccelerationStructuresIndirectKHR = PFN_vkCmdBuildAccelerationStructuresIndirectKHR(
        vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructuresIndirectKHR" ) );
      vkCmdBuildAccelerationStructuresKHR =
        PFN_vkCmdBuildAccelerationStructuresKHR( vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructuresKHR" ) );
      vkCmdClearAttachments = PFN_vkCmdClearAttachments( vkGetDeviceProcAddr( device, "vkCmdClearAttachments" ) );
      vkCmdClearColorImage  = PFN_vkCmdClearColorImage( vkGetDeviceProcAddr( device, "vkCmdClearColorImage" ) );
      vkCmdClearDepthStencilImage =
        PFN_vkCmdClearDepthStencilImage( vkGetDeviceProcAddr( device, "vkCmdClearDepthStencilImage" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdControlVideoCodingKHR =
        PFN_vkCmdControlVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdControlVideoCodingKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdCopyAccelerationStructureKHR =
        PFN_vkCmdCopyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureKHR" ) );
      vkCmdCopyAccelerationStructureNV =
        PFN_vkCmdCopyAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureNV" ) );
      vkCmdCopyAccelerationStructureToMemoryKHR = PFN_vkCmdCopyAccelerationStructureToMemoryKHR(
        vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureToMemoryKHR" ) );
      vkCmdCopyBuffer        = PFN_vkCmdCopyBuffer( vkGetDeviceProcAddr( device, "vkCmdCopyBuffer" ) );
      vkCmdCopyBuffer2KHR    = PFN_vkCmdCopyBuffer2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyBuffer2KHR" ) );
      vkCmdCopyBufferToImage = PFN_vkCmdCopyBufferToImage( vkGetDeviceProcAddr( device, "vkCmdCopyBufferToImage" ) );
      vkCmdCopyBufferToImage2KHR =
        PFN_vkCmdCopyBufferToImage2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyBufferToImage2KHR" ) );
      vkCmdCopyImage         = PFN_vkCmdCopyImage( vkGetDeviceProcAddr( device, "vkCmdCopyImage" ) );
      vkCmdCopyImage2KHR     = PFN_vkCmdCopyImage2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyImage2KHR" ) );
      vkCmdCopyImageToBuffer = PFN_vkCmdCopyImageToBuffer( vkGetDeviceProcAddr( device, "vkCmdCopyImageToBuffer" ) );
      vkCmdCopyImageToBuffer2KHR =
        PFN_vkCmdCopyImageToBuffer2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyImageToBuffer2KHR" ) );
      vkCmdCopyMemoryToAccelerationStructureKHR = PFN_vkCmdCopyMemoryToAccelerationStructureKHR(
        vkGetDeviceProcAddr( device, "vkCmdCopyMemoryToAccelerationStructureKHR" ) );
      vkCmdCopyQueryPoolResults =
        PFN_vkCmdCopyQueryPoolResults( vkGetDeviceProcAddr( device, "vkCmdCopyQueryPoolResults" ) );
      vkCmdCuLaunchKernelNVX = PFN_vkCmdCuLaunchKernelNVX( vkGetDeviceProcAddr( device, "vkCmdCuLaunchKernelNVX" ) );
      vkCmdDebugMarkerBeginEXT =
        PFN_vkCmdDebugMarkerBeginEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerBeginEXT" ) );
      vkCmdDebugMarkerEndEXT = PFN_vkCmdDebugMarkerEndEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerEndEXT" ) );
      vkCmdDebugMarkerInsertEXT =
        PFN_vkCmdDebugMarkerInsertEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerInsertEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdDecodeVideoKHR = PFN_vkCmdDecodeVideoKHR( vkGetDeviceProcAddr( device, "vkCmdDecodeVideoKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdDispatch        = PFN_vkCmdDispatch( vkGetDeviceProcAddr( device, "vkCmdDispatch" ) );
      vkCmdDispatchBase    = PFN_vkCmdDispatchBase( vkGetDeviceProcAddr( device, "vkCmdDispatchBase" ) );
      vkCmdDispatchBaseKHR = PFN_vkCmdDispatchBaseKHR( vkGetDeviceProcAddr( device, "vkCmdDispatchBaseKHR" ) );
      if ( !vkCmdDispatchBase )
        vkCmdDispatchBase = vkCmdDispatchBaseKHR;
      vkCmdDispatchIndirect = PFN_vkCmdDispatchIndirect( vkGetDeviceProcAddr( device, "vkCmdDispatchIndirect" ) );
      vkCmdDraw             = PFN_vkCmdDraw( vkGetDeviceProcAddr( device, "vkCmdDraw" ) );
      vkCmdDrawIndexed      = PFN_vkCmdDrawIndexed( vkGetDeviceProcAddr( device, "vkCmdDrawIndexed" ) );
      vkCmdDrawIndexedIndirect =
        PFN_vkCmdDrawIndexedIndirect( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirect" ) );
      vkCmdDrawIndexedIndirectCount =
        PFN_vkCmdDrawIndexedIndirectCount( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCount" ) );
      vkCmdDrawIndexedIndirectCountAMD =
        PFN_vkCmdDrawIndexedIndirectCountAMD( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCountAMD" ) );
      if ( !vkCmdDrawIndexedIndirectCount )
        vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountAMD;
      vkCmdDrawIndexedIndirectCountKHR =
        PFN_vkCmdDrawIndexedIndirectCountKHR( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCountKHR" ) );
      if ( !vkCmdDrawIndexedIndirectCount )
        vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountKHR;
      vkCmdDrawIndirect = PFN_vkCmdDrawIndirect( vkGetDeviceProcAddr( device, "vkCmdDrawIndirect" ) );
      vkCmdDrawIndirectByteCountEXT =
        PFN_vkCmdDrawIndirectByteCountEXT( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectByteCountEXT" ) );
      vkCmdDrawIndirectCount = PFN_vkCmdDrawIndirectCount( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCount" ) );
      vkCmdDrawIndirectCountAMD =
        PFN_vkCmdDrawIndirectCountAMD( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCountAMD" ) );
      if ( !vkCmdDrawIndirectCount )
        vkCmdDrawIndirectCount = vkCmdDrawIndirectCountAMD;
      vkCmdDrawIndirectCountKHR =
        PFN_vkCmdDrawIndirectCountKHR( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCountKHR" ) );
      if ( !vkCmdDrawIndirectCount )
        vkCmdDrawIndirectCount = vkCmdDrawIndirectCountKHR;
      vkCmdDrawMeshTasksIndirectCountNV =
        PFN_vkCmdDrawMeshTasksIndirectCountNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksIndirectCountNV" ) );
      vkCmdDrawMeshTasksIndirectNV =
        PFN_vkCmdDrawMeshTasksIndirectNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksIndirectNV" ) );
      vkCmdDrawMeshTasksNV = PFN_vkCmdDrawMeshTasksNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksNV" ) );
      vkCmdDrawMultiEXT    = PFN_vkCmdDrawMultiEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMultiEXT" ) );
      vkCmdDrawMultiIndexedEXT =
        PFN_vkCmdDrawMultiIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMultiIndexedEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdEncodeVideoKHR = PFN_vkCmdEncodeVideoKHR( vkGetDeviceProcAddr( device, "vkCmdEncodeVideoKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdEndConditionalRenderingEXT =
        PFN_vkCmdEndConditionalRenderingEXT( vkGetDeviceProcAddr( device, "vkCmdEndConditionalRenderingEXT" ) );
      vkCmdEndDebugUtilsLabelEXT =
        PFN_vkCmdEndDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdEndDebugUtilsLabelEXT" ) );
      vkCmdEndQuery           = PFN_vkCmdEndQuery( vkGetDeviceProcAddr( device, "vkCmdEndQuery" ) );
      vkCmdEndQueryIndexedEXT = PFN_vkCmdEndQueryIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdEndQueryIndexedEXT" ) );
      vkCmdEndRenderPass      = PFN_vkCmdEndRenderPass( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass" ) );
      vkCmdEndRenderPass2     = PFN_vkCmdEndRenderPass2( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass2" ) );
      vkCmdEndRenderPass2KHR  = PFN_vkCmdEndRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass2KHR" ) );
      if ( !vkCmdEndRenderPass2 )
        vkCmdEndRenderPass2 = vkCmdEndRenderPass2KHR;
      vkCmdEndTransformFeedbackEXT =
        PFN_vkCmdEndTransformFeedbackEXT( vkGetDeviceProcAddr( device, "vkCmdEndTransformFeedbackEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCmdEndVideoCodingKHR = PFN_vkCmdEndVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdEndVideoCodingKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkCmdExecuteCommands = PFN_vkCmdExecuteCommands( vkGetDeviceProcAddr( device, "vkCmdExecuteCommands" ) );
      vkCmdExecuteGeneratedCommandsNV =
        PFN_vkCmdExecuteGeneratedCommandsNV( vkGetDeviceProcAddr( device, "vkCmdExecuteGeneratedCommandsNV" ) );
      vkCmdFillBuffer = PFN_vkCmdFillBuffer( vkGetDeviceProcAddr( device, "vkCmdFillBuffer" ) );
      vkCmdInsertDebugUtilsLabelEXT =
        PFN_vkCmdInsertDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdInsertDebugUtilsLabelEXT" ) );
      vkCmdNextSubpass     = PFN_vkCmdNextSubpass( vkGetDeviceProcAddr( device, "vkCmdNextSubpass" ) );
      vkCmdNextSubpass2    = PFN_vkCmdNextSubpass2( vkGetDeviceProcAddr( device, "vkCmdNextSubpass2" ) );
      vkCmdNextSubpass2KHR = PFN_vkCmdNextSubpass2KHR( vkGetDeviceProcAddr( device, "vkCmdNextSubpass2KHR" ) );
      if ( !vkCmdNextSubpass2 )
        vkCmdNextSubpass2 = vkCmdNextSubpass2KHR;
      vkCmdPipelineBarrier = PFN_vkCmdPipelineBarrier( vkGetDeviceProcAddr( device, "vkCmdPipelineBarrier" ) );
      vkCmdPipelineBarrier2KHR =
        PFN_vkCmdPipelineBarrier2KHR( vkGetDeviceProcAddr( device, "vkCmdPipelineBarrier2KHR" ) );
      vkCmdPreprocessGeneratedCommandsNV =
        PFN_vkCmdPreprocessGeneratedCommandsNV( vkGetDeviceProcAddr( device, "vkCmdPreprocessGeneratedCommandsNV" ) );
      vkCmdPushConstants = PFN_vkCmdPushConstants( vkGetDeviceProcAddr( device, "vkCmdPushConstants" ) );
      vkCmdPushDescriptorSetKHR =
        PFN_vkCmdPushDescriptorSetKHR( vkGetDeviceProcAddr( device, "vkCmdPushDescriptorSetKHR" ) );
      vkCmdPushDescriptorSetWithTemplateKHR = PFN_vkCmdPushDescriptorSetWithTemplateKHR(
        vkGetDeviceProcAddr( device, "vkCmdPushDescriptorSetWithTemplateKHR" ) );
      vkCmdResetEvent        = PFN_vkCmdResetEvent( vkGetDeviceProcAddr( device, "vkCmdResetEvent" ) );
      vkCmdResetEvent2KHR    = PFN_vkCmdResetEvent2KHR( vkGetDeviceProcAddr( device, "vkCmdResetEvent2KHR" ) );
      vkCmdResetQueryPool    = PFN_vkCmdResetQueryPool( vkGetDeviceProcAddr( device, "vkCmdResetQueryPool" ) );
      vkCmdResolveImage      = PFN_vkCmdResolveImage( vkGetDeviceProcAddr( device, "vkCmdResolveImage" ) );
      vkCmdResolveImage2KHR  = PFN_vkCmdResolveImage2KHR( vkGetDeviceProcAddr( device, "vkCmdResolveImage2KHR" ) );
      vkCmdSetBlendConstants = PFN_vkCmdSetBlendConstants( vkGetDeviceProcAddr( device, "vkCmdSetBlendConstants" ) );
      vkCmdSetCheckpointNV   = PFN_vkCmdSetCheckpointNV( vkGetDeviceProcAddr( device, "vkCmdSetCheckpointNV" ) );
      vkCmdSetCoarseSampleOrderNV =
        PFN_vkCmdSetCoarseSampleOrderNV( vkGetDeviceProcAddr( device, "vkCmdSetCoarseSampleOrderNV" ) );
      vkCmdSetColorWriteEnableEXT =
        PFN_vkCmdSetColorWriteEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetColorWriteEnableEXT" ) );
      vkCmdSetCullModeEXT = PFN_vkCmdSetCullModeEXT( vkGetDeviceProcAddr( device, "vkCmdSetCullModeEXT" ) );
      vkCmdSetDepthBias   = PFN_vkCmdSetDepthBias( vkGetDeviceProcAddr( device, "vkCmdSetDepthBias" ) );
      vkCmdSetDepthBiasEnableEXT =
        PFN_vkCmdSetDepthBiasEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthBiasEnableEXT" ) );
      vkCmdSetDepthBounds = PFN_vkCmdSetDepthBounds( vkGetDeviceProcAddr( device, "vkCmdSetDepthBounds" ) );
      vkCmdSetDepthBoundsTestEnableEXT =
        PFN_vkCmdSetDepthBoundsTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthBoundsTestEnableEXT" ) );
      vkCmdSetDepthCompareOpEXT =
        PFN_vkCmdSetDepthCompareOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthCompareOpEXT" ) );
      vkCmdSetDepthTestEnableEXT =
        PFN_vkCmdSetDepthTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthTestEnableEXT" ) );
      vkCmdSetDepthWriteEnableEXT =
        PFN_vkCmdSetDepthWriteEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthWriteEnableEXT" ) );
      vkCmdSetDeviceMask    = PFN_vkCmdSetDeviceMask( vkGetDeviceProcAddr( device, "vkCmdSetDeviceMask" ) );
      vkCmdSetDeviceMaskKHR = PFN_vkCmdSetDeviceMaskKHR( vkGetDeviceProcAddr( device, "vkCmdSetDeviceMaskKHR" ) );
      if ( !vkCmdSetDeviceMask )
        vkCmdSetDeviceMask = vkCmdSetDeviceMaskKHR;
      vkCmdSetDiscardRectangleEXT =
        PFN_vkCmdSetDiscardRectangleEXT( vkGetDeviceProcAddr( device, "vkCmdSetDiscardRectangleEXT" ) );
      vkCmdSetEvent     = PFN_vkCmdSetEvent( vkGetDeviceProcAddr( device, "vkCmdSetEvent" ) );
      vkCmdSetEvent2KHR = PFN_vkCmdSetEvent2KHR( vkGetDeviceProcAddr( device, "vkCmdSetEvent2KHR" ) );
      vkCmdSetExclusiveScissorNV =
        PFN_vkCmdSetExclusiveScissorNV( vkGetDeviceProcAddr( device, "vkCmdSetExclusiveScissorNV" ) );
      vkCmdSetFragmentShadingRateEnumNV =
        PFN_vkCmdSetFragmentShadingRateEnumNV( vkGetDeviceProcAddr( device, "vkCmdSetFragmentShadingRateEnumNV" ) );
      vkCmdSetFragmentShadingRateKHR =
        PFN_vkCmdSetFragmentShadingRateKHR( vkGetDeviceProcAddr( device, "vkCmdSetFragmentShadingRateKHR" ) );
      vkCmdSetFrontFaceEXT   = PFN_vkCmdSetFrontFaceEXT( vkGetDeviceProcAddr( device, "vkCmdSetFrontFaceEXT" ) );
      vkCmdSetLineStippleEXT = PFN_vkCmdSetLineStippleEXT( vkGetDeviceProcAddr( device, "vkCmdSetLineStippleEXT" ) );
      vkCmdSetLineWidth      = PFN_vkCmdSetLineWidth( vkGetDeviceProcAddr( device, "vkCmdSetLineWidth" ) );
      vkCmdSetLogicOpEXT     = PFN_vkCmdSetLogicOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetLogicOpEXT" ) );
      vkCmdSetPatchControlPointsEXT =
        PFN_vkCmdSetPatchControlPointsEXT( vkGetDeviceProcAddr( device, "vkCmdSetPatchControlPointsEXT" ) );
      vkCmdSetPerformanceMarkerINTEL =
        PFN_vkCmdSetPerformanceMarkerINTEL( vkGetDeviceProcAddr( device, "vkCmdSetPerformanceMarkerINTEL" ) );
      vkCmdSetPerformanceOverrideINTEL =
        PFN_vkCmdSetPerformanceOverrideINTEL( vkGetDeviceProcAddr( device, "vkCmdSetPerformanceOverrideINTEL" ) );
      vkCmdSetPerformanceStreamMarkerINTEL = PFN_vkCmdSetPerformanceStreamMarkerINTEL(
        vkGetDeviceProcAddr( device, "vkCmdSetPerformanceStreamMarkerINTEL" ) );
      vkCmdSetPrimitiveRestartEnableEXT =
        PFN_vkCmdSetPrimitiveRestartEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetPrimitiveRestartEnableEXT" ) );
      vkCmdSetPrimitiveTopologyEXT =
        PFN_vkCmdSetPrimitiveTopologyEXT( vkGetDeviceProcAddr( device, "vkCmdSetPrimitiveTopologyEXT" ) );
      vkCmdSetRasterizerDiscardEnableEXT =
        PFN_vkCmdSetRasterizerDiscardEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetRasterizerDiscardEnableEXT" ) );
      vkCmdSetRayTracingPipelineStackSizeKHR = PFN_vkCmdSetRayTracingPipelineStackSizeKHR(
        vkGetDeviceProcAddr( device, "vkCmdSetRayTracingPipelineStackSizeKHR" ) );
      vkCmdSetSampleLocationsEXT =
        PFN_vkCmdSetSampleLocationsEXT( vkGetDeviceProcAddr( device, "vkCmdSetSampleLocationsEXT" ) );
      vkCmdSetScissor = PFN_vkCmdSetScissor( vkGetDeviceProcAddr( device, "vkCmdSetScissor" ) );
      vkCmdSetScissorWithCountEXT =
        PFN_vkCmdSetScissorWithCountEXT( vkGetDeviceProcAddr( device, "vkCmdSetScissorWithCountEXT" ) );
      vkCmdSetStencilCompareMask =
        PFN_vkCmdSetStencilCompareMask( vkGetDeviceProcAddr( device, "vkCmdSetStencilCompareMask" ) );
      vkCmdSetStencilOpEXT = PFN_vkCmdSetStencilOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetStencilOpEXT" ) );
      vkCmdSetStencilReference =
        PFN_vkCmdSetStencilReference( vkGetDeviceProcAddr( device, "vkCmdSetStencilReference" ) );
      vkCmdSetStencilTestEnableEXT =
        PFN_vkCmdSetStencilTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetStencilTestEnableEXT" ) );
      vkCmdSetStencilWriteMask =
        PFN_vkCmdSetStencilWriteMask( vkGetDeviceProcAddr( device, "vkCmdSetStencilWriteMask" ) );
      vkCmdSetVertexInputEXT = PFN_vkCmdSetVertexInputEXT( vkGetDeviceProcAddr( device, "vkCmdSetVertexInputEXT" ) );
      vkCmdSetViewport       = PFN_vkCmdSetViewport( vkGetDeviceProcAddr( device, "vkCmdSetViewport" ) );
      vkCmdSetViewportShadingRatePaletteNV = PFN_vkCmdSetViewportShadingRatePaletteNV(
        vkGetDeviceProcAddr( device, "vkCmdSetViewportShadingRatePaletteNV" ) );
      vkCmdSetViewportWScalingNV =
        PFN_vkCmdSetViewportWScalingNV( vkGetDeviceProcAddr( device, "vkCmdSetViewportWScalingNV" ) );
      vkCmdSetViewportWithCountEXT =
        PFN_vkCmdSetViewportWithCountEXT( vkGetDeviceProcAddr( device, "vkCmdSetViewportWithCountEXT" ) );
      vkCmdSubpassShadingHUAWEI =
        PFN_vkCmdSubpassShadingHUAWEI( vkGetDeviceProcAddr( device, "vkCmdSubpassShadingHUAWEI" ) );
      vkCmdTraceRaysIndirectKHR =
        PFN_vkCmdTraceRaysIndirectKHR( vkGetDeviceProcAddr( device, "vkCmdTraceRaysIndirectKHR" ) );
      vkCmdTraceRaysKHR   = PFN_vkCmdTraceRaysKHR( vkGetDeviceProcAddr( device, "vkCmdTraceRaysKHR" ) );
      vkCmdTraceRaysNV    = PFN_vkCmdTraceRaysNV( vkGetDeviceProcAddr( device, "vkCmdTraceRaysNV" ) );
      vkCmdUpdateBuffer   = PFN_vkCmdUpdateBuffer( vkGetDeviceProcAddr( device, "vkCmdUpdateBuffer" ) );
      vkCmdWaitEvents     = PFN_vkCmdWaitEvents( vkGetDeviceProcAddr( device, "vkCmdWaitEvents" ) );
      vkCmdWaitEvents2KHR = PFN_vkCmdWaitEvents2KHR( vkGetDeviceProcAddr( device, "vkCmdWaitEvents2KHR" ) );
      vkCmdWriteAccelerationStructuresPropertiesKHR = PFN_vkCmdWriteAccelerationStructuresPropertiesKHR(
        vkGetDeviceProcAddr( device, "vkCmdWriteAccelerationStructuresPropertiesKHR" ) );
      vkCmdWriteAccelerationStructuresPropertiesNV = PFN_vkCmdWriteAccelerationStructuresPropertiesNV(
        vkGetDeviceProcAddr( device, "vkCmdWriteAccelerationStructuresPropertiesNV" ) );
      vkCmdWriteBufferMarker2AMD =
        PFN_vkCmdWriteBufferMarker2AMD( vkGetDeviceProcAddr( device, "vkCmdWriteBufferMarker2AMD" ) );
      vkCmdWriteBufferMarkerAMD =
        PFN_vkCmdWriteBufferMarkerAMD( vkGetDeviceProcAddr( device, "vkCmdWriteBufferMarkerAMD" ) );
      vkCmdWriteTimestamp     = PFN_vkCmdWriteTimestamp( vkGetDeviceProcAddr( device, "vkCmdWriteTimestamp" ) );
      vkCmdWriteTimestamp2KHR = PFN_vkCmdWriteTimestamp2KHR( vkGetDeviceProcAddr( device, "vkCmdWriteTimestamp2KHR" ) );
      vkCompileDeferredNV     = PFN_vkCompileDeferredNV( vkGetDeviceProcAddr( device, "vkCompileDeferredNV" ) );
      vkCopyAccelerationStructureKHR =
        PFN_vkCopyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCopyAccelerationStructureKHR" ) );
      vkCopyAccelerationStructureToMemoryKHR = PFN_vkCopyAccelerationStructureToMemoryKHR(
        vkGetDeviceProcAddr( device, "vkCopyAccelerationStructureToMemoryKHR" ) );
      vkCopyMemoryToAccelerationStructureKHR = PFN_vkCopyMemoryToAccelerationStructureKHR(
        vkGetDeviceProcAddr( device, "vkCopyMemoryToAccelerationStructureKHR" ) );
      vkCreateAccelerationStructureKHR =
        PFN_vkCreateAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCreateAccelerationStructureKHR" ) );
      vkCreateAccelerationStructureNV =
        PFN_vkCreateAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCreateAccelerationStructureNV" ) );
      vkCreateBuffer      = PFN_vkCreateBuffer( vkGetDeviceProcAddr( device, "vkCreateBuffer" ) );
      vkCreateBufferView  = PFN_vkCreateBufferView( vkGetDeviceProcAddr( device, "vkCreateBufferView" ) );
      vkCreateCommandPool = PFN_vkCreateCommandPool( vkGetDeviceProcAddr( device, "vkCreateCommandPool" ) );
      vkCreateComputePipelines =
        PFN_vkCreateComputePipelines( vkGetDeviceProcAddr( device, "vkCreateComputePipelines" ) );
      vkCreateCuFunctionNVX = PFN_vkCreateCuFunctionNVX( vkGetDeviceProcAddr( device, "vkCreateCuFunctionNVX" ) );
      vkCreateCuModuleNVX   = PFN_vkCreateCuModuleNVX( vkGetDeviceProcAddr( device, "vkCreateCuModuleNVX" ) );
      vkCreateDeferredOperationKHR =
        PFN_vkCreateDeferredOperationKHR( vkGetDeviceProcAddr( device, "vkCreateDeferredOperationKHR" ) );
      vkCreateDescriptorPool = PFN_vkCreateDescriptorPool( vkGetDeviceProcAddr( device, "vkCreateDescriptorPool" ) );
      vkCreateDescriptorSetLayout =
        PFN_vkCreateDescriptorSetLayout( vkGetDeviceProcAddr( device, "vkCreateDescriptorSetLayout" ) );
      vkCreateDescriptorUpdateTemplate =
        PFN_vkCreateDescriptorUpdateTemplate( vkGetDeviceProcAddr( device, "vkCreateDescriptorUpdateTemplate" ) );
      vkCreateDescriptorUpdateTemplateKHR =
        PFN_vkCreateDescriptorUpdateTemplateKHR( vkGetDeviceProcAddr( device, "vkCreateDescriptorUpdateTemplateKHR" ) );
      if ( !vkCreateDescriptorUpdateTemplate )
        vkCreateDescriptorUpdateTemplate = vkCreateDescriptorUpdateTemplateKHR;
      vkCreateEvent       = PFN_vkCreateEvent( vkGetDeviceProcAddr( device, "vkCreateEvent" ) );
      vkCreateFence       = PFN_vkCreateFence( vkGetDeviceProcAddr( device, "vkCreateFence" ) );
      vkCreateFramebuffer = PFN_vkCreateFramebuffer( vkGetDeviceProcAddr( device, "vkCreateFramebuffer" ) );
      vkCreateGraphicsPipelines =
        PFN_vkCreateGraphicsPipelines( vkGetDeviceProcAddr( device, "vkCreateGraphicsPipelines" ) );
      vkCreateImage     = PFN_vkCreateImage( vkGetDeviceProcAddr( device, "vkCreateImage" ) );
      vkCreateImageView = PFN_vkCreateImageView( vkGetDeviceProcAddr( device, "vkCreateImageView" ) );
      vkCreateIndirectCommandsLayoutNV =
        PFN_vkCreateIndirectCommandsLayoutNV( vkGetDeviceProcAddr( device, "vkCreateIndirectCommandsLayoutNV" ) );
      vkCreatePipelineCache  = PFN_vkCreatePipelineCache( vkGetDeviceProcAddr( device, "vkCreatePipelineCache" ) );
      vkCreatePipelineLayout = PFN_vkCreatePipelineLayout( vkGetDeviceProcAddr( device, "vkCreatePipelineLayout" ) );
      vkCreatePrivateDataSlotEXT =
        PFN_vkCreatePrivateDataSlotEXT( vkGetDeviceProcAddr( device, "vkCreatePrivateDataSlotEXT" ) );
      vkCreateQueryPool = PFN_vkCreateQueryPool( vkGetDeviceProcAddr( device, "vkCreateQueryPool" ) );
      vkCreateRayTracingPipelinesKHR =
        PFN_vkCreateRayTracingPipelinesKHR( vkGetDeviceProcAddr( device, "vkCreateRayTracingPipelinesKHR" ) );
      vkCreateRayTracingPipelinesNV =
        PFN_vkCreateRayTracingPipelinesNV( vkGetDeviceProcAddr( device, "vkCreateRayTracingPipelinesNV" ) );
      vkCreateRenderPass     = PFN_vkCreateRenderPass( vkGetDeviceProcAddr( device, "vkCreateRenderPass" ) );
      vkCreateRenderPass2    = PFN_vkCreateRenderPass2( vkGetDeviceProcAddr( device, "vkCreateRenderPass2" ) );
      vkCreateRenderPass2KHR = PFN_vkCreateRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCreateRenderPass2KHR" ) );
      if ( !vkCreateRenderPass2 )
        vkCreateRenderPass2 = vkCreateRenderPass2KHR;
      vkCreateSampler = PFN_vkCreateSampler( vkGetDeviceProcAddr( device, "vkCreateSampler" ) );
      vkCreateSamplerYcbcrConversion =
        PFN_vkCreateSamplerYcbcrConversion( vkGetDeviceProcAddr( device, "vkCreateSamplerYcbcrConversion" ) );
      vkCreateSamplerYcbcrConversionKHR =
        PFN_vkCreateSamplerYcbcrConversionKHR( vkGetDeviceProcAddr( device, "vkCreateSamplerYcbcrConversionKHR" ) );
      if ( !vkCreateSamplerYcbcrConversion )
        vkCreateSamplerYcbcrConversion = vkCreateSamplerYcbcrConversionKHR;
      vkCreateSemaphore    = PFN_vkCreateSemaphore( vkGetDeviceProcAddr( device, "vkCreateSemaphore" ) );
      vkCreateShaderModule = PFN_vkCreateShaderModule( vkGetDeviceProcAddr( device, "vkCreateShaderModule" ) );
      vkCreateSharedSwapchainsKHR =
        PFN_vkCreateSharedSwapchainsKHR( vkGetDeviceProcAddr( device, "vkCreateSharedSwapchainsKHR" ) );
      vkCreateSwapchainKHR = PFN_vkCreateSwapchainKHR( vkGetDeviceProcAddr( device, "vkCreateSwapchainKHR" ) );
      vkCreateValidationCacheEXT =
        PFN_vkCreateValidationCacheEXT( vkGetDeviceProcAddr( device, "vkCreateValidationCacheEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCreateVideoSessionKHR = PFN_vkCreateVideoSessionKHR( vkGetDeviceProcAddr( device, "vkCreateVideoSessionKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkCreateVideoSessionParametersKHR =
        PFN_vkCreateVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkCreateVideoSessionParametersKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkDebugMarkerSetObjectNameEXT =
        PFN_vkDebugMarkerSetObjectNameEXT( vkGetDeviceProcAddr( device, "vkDebugMarkerSetObjectNameEXT" ) );
      vkDebugMarkerSetObjectTagEXT =
        PFN_vkDebugMarkerSetObjectTagEXT( vkGetDeviceProcAddr( device, "vkDebugMarkerSetObjectTagEXT" ) );
      vkDeferredOperationJoinKHR =
        PFN_vkDeferredOperationJoinKHR( vkGetDeviceProcAddr( device, "vkDeferredOperationJoinKHR" ) );
      vkDestroyAccelerationStructureKHR =
        PFN_vkDestroyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkDestroyAccelerationStructureKHR" ) );
      vkDestroyAccelerationStructureNV =
        PFN_vkDestroyAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkDestroyAccelerationStructureNV" ) );
      vkDestroyBuffer        = PFN_vkDestroyBuffer( vkGetDeviceProcAddr( device, "vkDestroyBuffer" ) );
      vkDestroyBufferView    = PFN_vkDestroyBufferView( vkGetDeviceProcAddr( device, "vkDestroyBufferView" ) );
      vkDestroyCommandPool   = PFN_vkDestroyCommandPool( vkGetDeviceProcAddr( device, "vkDestroyCommandPool" ) );
      vkDestroyCuFunctionNVX = PFN_vkDestroyCuFunctionNVX( vkGetDeviceProcAddr( device, "vkDestroyCuFunctionNVX" ) );
      vkDestroyCuModuleNVX   = PFN_vkDestroyCuModuleNVX( vkGetDeviceProcAddr( device, "vkDestroyCuModuleNVX" ) );
      vkDestroyDeferredOperationKHR =
        PFN_vkDestroyDeferredOperationKHR( vkGetDeviceProcAddr( device, "vkDestroyDeferredOperationKHR" ) );
      vkDestroyDescriptorPool = PFN_vkDestroyDescriptorPool( vkGetDeviceProcAddr( device, "vkDestroyDescriptorPool" ) );
      vkDestroyDescriptorSetLayout =
        PFN_vkDestroyDescriptorSetLayout( vkGetDeviceProcAddr( device, "vkDestroyDescriptorSetLayout" ) );
      vkDestroyDescriptorUpdateTemplate =
        PFN_vkDestroyDescriptorUpdateTemplate( vkGetDeviceProcAddr( device, "vkDestroyDescriptorUpdateTemplate" ) );
      vkDestroyDescriptorUpdateTemplateKHR = PFN_vkDestroyDescriptorUpdateTemplateKHR(
        vkGetDeviceProcAddr( device, "vkDestroyDescriptorUpdateTemplateKHR" ) );
      if ( !vkDestroyDescriptorUpdateTemplate )
        vkDestroyDescriptorUpdateTemplate = vkDestroyDescriptorUpdateTemplateKHR;
      vkDestroyDevice      = PFN_vkDestroyDevice( vkGetDeviceProcAddr( device, "vkDestroyDevice" ) );
      vkDestroyEvent       = PFN_vkDestroyEvent( vkGetDeviceProcAddr( device, "vkDestroyEvent" ) );
      vkDestroyFence       = PFN_vkDestroyFence( vkGetDeviceProcAddr( device, "vkDestroyFence" ) );
      vkDestroyFramebuffer = PFN_vkDestroyFramebuffer( vkGetDeviceProcAddr( device, "vkDestroyFramebuffer" ) );
      vkDestroyImage       = PFN_vkDestroyImage( vkGetDeviceProcAddr( device, "vkDestroyImage" ) );
      vkDestroyImageView   = PFN_vkDestroyImageView( vkGetDeviceProcAddr( device, "vkDestroyImageView" ) );
      vkDestroyIndirectCommandsLayoutNV =
        PFN_vkDestroyIndirectCommandsLayoutNV( vkGetDeviceProcAddr( device, "vkDestroyIndirectCommandsLayoutNV" ) );
      vkDestroyPipeline       = PFN_vkDestroyPipeline( vkGetDeviceProcAddr( device, "vkDestroyPipeline" ) );
      vkDestroyPipelineCache  = PFN_vkDestroyPipelineCache( vkGetDeviceProcAddr( device, "vkDestroyPipelineCache" ) );
      vkDestroyPipelineLayout = PFN_vkDestroyPipelineLayout( vkGetDeviceProcAddr( device, "vkDestroyPipelineLayout" ) );
      vkDestroyPrivateDataSlotEXT =
        PFN_vkDestroyPrivateDataSlotEXT( vkGetDeviceProcAddr( device, "vkDestroyPrivateDataSlotEXT" ) );
      vkDestroyQueryPool  = PFN_vkDestroyQueryPool( vkGetDeviceProcAddr( device, "vkDestroyQueryPool" ) );
      vkDestroyRenderPass = PFN_vkDestroyRenderPass( vkGetDeviceProcAddr( device, "vkDestroyRenderPass" ) );
      vkDestroySampler    = PFN_vkDestroySampler( vkGetDeviceProcAddr( device, "vkDestroySampler" ) );
      vkDestroySamplerYcbcrConversion =
        PFN_vkDestroySamplerYcbcrConversion( vkGetDeviceProcAddr( device, "vkDestroySamplerYcbcrConversion" ) );
      vkDestroySamplerYcbcrConversionKHR =
        PFN_vkDestroySamplerYcbcrConversionKHR( vkGetDeviceProcAddr( device, "vkDestroySamplerYcbcrConversionKHR" ) );
      if ( !vkDestroySamplerYcbcrConversion )
        vkDestroySamplerYcbcrConversion = vkDestroySamplerYcbcrConversionKHR;
      vkDestroySemaphore    = PFN_vkDestroySemaphore( vkGetDeviceProcAddr( device, "vkDestroySemaphore" ) );
      vkDestroyShaderModule = PFN_vkDestroyShaderModule( vkGetDeviceProcAddr( device, "vkDestroyShaderModule" ) );
      vkDestroySwapchainKHR = PFN_vkDestroySwapchainKHR( vkGetDeviceProcAddr( device, "vkDestroySwapchainKHR" ) );
      vkDestroyValidationCacheEXT =
        PFN_vkDestroyValidationCacheEXT( vkGetDeviceProcAddr( device, "vkDestroyValidationCacheEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkDestroyVideoSessionKHR =
        PFN_vkDestroyVideoSessionKHR( vkGetDeviceProcAddr( device, "vkDestroyVideoSessionKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkDestroyVideoSessionParametersKHR =
        PFN_vkDestroyVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkDestroyVideoSessionParametersKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkDeviceWaitIdle = PFN_vkDeviceWaitIdle( vkGetDeviceProcAddr( device, "vkDeviceWaitIdle" ) );
      vkDisplayPowerControlEXT =
        PFN_vkDisplayPowerControlEXT( vkGetDeviceProcAddr( device, "vkDisplayPowerControlEXT" ) );
      vkEndCommandBuffer = PFN_vkEndCommandBuffer( vkGetDeviceProcAddr( device, "vkEndCommandBuffer" ) );
      vkFlushMappedMemoryRanges =
        PFN_vkFlushMappedMemoryRanges( vkGetDeviceProcAddr( device, "vkFlushMappedMemoryRanges" ) );
      vkFreeCommandBuffers = PFN_vkFreeCommandBuffers( vkGetDeviceProcAddr( device, "vkFreeCommandBuffers" ) );
      vkFreeDescriptorSets = PFN_vkFreeDescriptorSets( vkGetDeviceProcAddr( device, "vkFreeDescriptorSets" ) );
      vkFreeMemory         = PFN_vkFreeMemory( vkGetDeviceProcAddr( device, "vkFreeMemory" ) );
      vkGetAccelerationStructureBuildSizesKHR = PFN_vkGetAccelerationStructureBuildSizesKHR(
        vkGetDeviceProcAddr( device, "vkGetAccelerationStructureBuildSizesKHR" ) );
      vkGetAccelerationStructureDeviceAddressKHR = PFN_vkGetAccelerationStructureDeviceAddressKHR(
        vkGetDeviceProcAddr( device, "vkGetAccelerationStructureDeviceAddressKHR" ) );
      vkGetAccelerationStructureHandleNV =
        PFN_vkGetAccelerationStructureHandleNV( vkGetDeviceProcAddr( device, "vkGetAccelerationStructureHandleNV" ) );
      vkGetAccelerationStructureMemoryRequirementsNV = PFN_vkGetAccelerationStructureMemoryRequirementsNV(
        vkGetDeviceProcAddr( device, "vkGetAccelerationStructureMemoryRequirementsNV" ) );
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      vkGetAndroidHardwareBufferPropertiesANDROID = PFN_vkGetAndroidHardwareBufferPropertiesANDROID(
        vkGetDeviceProcAddr( device, "vkGetAndroidHardwareBufferPropertiesANDROID" ) );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      vkGetBufferDeviceAddress =
        PFN_vkGetBufferDeviceAddress( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddress" ) );
      vkGetBufferDeviceAddressEXT =
        PFN_vkGetBufferDeviceAddressEXT( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddressEXT" ) );
      if ( !vkGetBufferDeviceAddress )
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressEXT;
      vkGetBufferDeviceAddressKHR =
        PFN_vkGetBufferDeviceAddressKHR( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddressKHR" ) );
      if ( !vkGetBufferDeviceAddress )
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressKHR;
      vkGetBufferMemoryRequirements =
        PFN_vkGetBufferMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements" ) );
      vkGetBufferMemoryRequirements2 =
        PFN_vkGetBufferMemoryRequirements2( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements2" ) );
      vkGetBufferMemoryRequirements2KHR =
        PFN_vkGetBufferMemoryRequirements2KHR( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements2KHR" ) );
      if ( !vkGetBufferMemoryRequirements2 )
        vkGetBufferMemoryRequirements2 = vkGetBufferMemoryRequirements2KHR;
      vkGetBufferOpaqueCaptureAddress =
        PFN_vkGetBufferOpaqueCaptureAddress( vkGetDeviceProcAddr( device, "vkGetBufferOpaqueCaptureAddress" ) );
      vkGetBufferOpaqueCaptureAddressKHR =
        PFN_vkGetBufferOpaqueCaptureAddressKHR( vkGetDeviceProcAddr( device, "vkGetBufferOpaqueCaptureAddressKHR" ) );
      if ( !vkGetBufferOpaqueCaptureAddress )
        vkGetBufferOpaqueCaptureAddress = vkGetBufferOpaqueCaptureAddressKHR;
      vkGetCalibratedTimestampsEXT =
        PFN_vkGetCalibratedTimestampsEXT( vkGetDeviceProcAddr( device, "vkGetCalibratedTimestampsEXT" ) );
      vkGetDeferredOperationMaxConcurrencyKHR = PFN_vkGetDeferredOperationMaxConcurrencyKHR(
        vkGetDeviceProcAddr( device, "vkGetDeferredOperationMaxConcurrencyKHR" ) );
      vkGetDeferredOperationResultKHR =
        PFN_vkGetDeferredOperationResultKHR( vkGetDeviceProcAddr( device, "vkGetDeferredOperationResultKHR" ) );
      vkGetDescriptorSetLayoutSupport =
        PFN_vkGetDescriptorSetLayoutSupport( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutSupport" ) );
      vkGetDescriptorSetLayoutSupportKHR =
        PFN_vkGetDescriptorSetLayoutSupportKHR( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutSupportKHR" ) );
      if ( !vkGetDescriptorSetLayoutSupport )
        vkGetDescriptorSetLayoutSupport = vkGetDescriptorSetLayoutSupportKHR;
      vkGetDeviceAccelerationStructureCompatibilityKHR = PFN_vkGetDeviceAccelerationStructureCompatibilityKHR(
        vkGetDeviceProcAddr( device, "vkGetDeviceAccelerationStructureCompatibilityKHR" ) );
      vkGetDeviceGroupPeerMemoryFeatures =
        PFN_vkGetDeviceGroupPeerMemoryFeatures( vkGetDeviceProcAddr( device, "vkGetDeviceGroupPeerMemoryFeatures" ) );
      vkGetDeviceGroupPeerMemoryFeaturesKHR = PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR(
        vkGetDeviceProcAddr( device, "vkGetDeviceGroupPeerMemoryFeaturesKHR" ) );
      if ( !vkGetDeviceGroupPeerMemoryFeatures )
        vkGetDeviceGroupPeerMemoryFeatures = vkGetDeviceGroupPeerMemoryFeaturesKHR;
      vkGetDeviceGroupPresentCapabilitiesKHR = PFN_vkGetDeviceGroupPresentCapabilitiesKHR(
        vkGetDeviceProcAddr( device, "vkGetDeviceGroupPresentCapabilitiesKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetDeviceGroupSurfacePresentModes2EXT = PFN_vkGetDeviceGroupSurfacePresentModes2EXT(
        vkGetDeviceProcAddr( device, "vkGetDeviceGroupSurfacePresentModes2EXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkGetDeviceGroupSurfacePresentModesKHR = PFN_vkGetDeviceGroupSurfacePresentModesKHR(
        vkGetDeviceProcAddr( device, "vkGetDeviceGroupSurfacePresentModesKHR" ) );
      vkGetDeviceMemoryCommitment =
        PFN_vkGetDeviceMemoryCommitment( vkGetDeviceProcAddr( device, "vkGetDeviceMemoryCommitment" ) );
      vkGetDeviceMemoryOpaqueCaptureAddress = PFN_vkGetDeviceMemoryOpaqueCaptureAddress(
        vkGetDeviceProcAddr( device, "vkGetDeviceMemoryOpaqueCaptureAddress" ) );
      vkGetDeviceMemoryOpaqueCaptureAddressKHR = PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR(
        vkGetDeviceProcAddr( device, "vkGetDeviceMemoryOpaqueCaptureAddressKHR" ) );
      if ( !vkGetDeviceMemoryOpaqueCaptureAddress )
        vkGetDeviceMemoryOpaqueCaptureAddress = vkGetDeviceMemoryOpaqueCaptureAddressKHR;
      vkGetDeviceProcAddr = PFN_vkGetDeviceProcAddr( vkGetDeviceProcAddr( device, "vkGetDeviceProcAddr" ) );
      vkGetDeviceQueue    = PFN_vkGetDeviceQueue( vkGetDeviceProcAddr( device, "vkGetDeviceQueue" ) );
      vkGetDeviceQueue2   = PFN_vkGetDeviceQueue2( vkGetDeviceProcAddr( device, "vkGetDeviceQueue2" ) );
      vkGetEventStatus    = PFN_vkGetEventStatus( vkGetDeviceProcAddr( device, "vkGetEventStatus" ) );
      vkGetFenceFdKHR     = PFN_vkGetFenceFdKHR( vkGetDeviceProcAddr( device, "vkGetFenceFdKHR" ) );
      vkGetFenceStatus    = PFN_vkGetFenceStatus( vkGetDeviceProcAddr( device, "vkGetFenceStatus" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetFenceWin32HandleKHR =
        PFN_vkGetFenceWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetFenceWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkGetGeneratedCommandsMemoryRequirementsNV = PFN_vkGetGeneratedCommandsMemoryRequirementsNV(
        vkGetDeviceProcAddr( device, "vkGetGeneratedCommandsMemoryRequirementsNV" ) );
      vkGetImageDrmFormatModifierPropertiesEXT = PFN_vkGetImageDrmFormatModifierPropertiesEXT(
        vkGetDeviceProcAddr( device, "vkGetImageDrmFormatModifierPropertiesEXT" ) );
      vkGetImageMemoryRequirements =
        PFN_vkGetImageMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements" ) );
      vkGetImageMemoryRequirements2 =
        PFN_vkGetImageMemoryRequirements2( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements2" ) );
      vkGetImageMemoryRequirements2KHR =
        PFN_vkGetImageMemoryRequirements2KHR( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements2KHR" ) );
      if ( !vkGetImageMemoryRequirements2 )
        vkGetImageMemoryRequirements2 = vkGetImageMemoryRequirements2KHR;
      vkGetImageSparseMemoryRequirements =
        PFN_vkGetImageSparseMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements" ) );
      vkGetImageSparseMemoryRequirements2 =
        PFN_vkGetImageSparseMemoryRequirements2( vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements2" ) );
      vkGetImageSparseMemoryRequirements2KHR = PFN_vkGetImageSparseMemoryRequirements2KHR(
        vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements2KHR" ) );
      if ( !vkGetImageSparseMemoryRequirements2 )
        vkGetImageSparseMemoryRequirements2 = vkGetImageSparseMemoryRequirements2KHR;
      vkGetImageSubresourceLayout =
        PFN_vkGetImageSubresourceLayout( vkGetDeviceProcAddr( device, "vkGetImageSubresourceLayout" ) );
      vkGetImageViewAddressNVX =
        PFN_vkGetImageViewAddressNVX( vkGetDeviceProcAddr( device, "vkGetImageViewAddressNVX" ) );
      vkGetImageViewHandleNVX = PFN_vkGetImageViewHandleNVX( vkGetDeviceProcAddr( device, "vkGetImageViewHandleNVX" ) );
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      vkGetMemoryAndroidHardwareBufferANDROID = PFN_vkGetMemoryAndroidHardwareBufferANDROID(
        vkGetDeviceProcAddr( device, "vkGetMemoryAndroidHardwareBufferANDROID" ) );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      vkGetMemoryFdKHR = PFN_vkGetMemoryFdKHR( vkGetDeviceProcAddr( device, "vkGetMemoryFdKHR" ) );
      vkGetMemoryFdPropertiesKHR =
        PFN_vkGetMemoryFdPropertiesKHR( vkGetDeviceProcAddr( device, "vkGetMemoryFdPropertiesKHR" ) );
      vkGetMemoryHostPointerPropertiesEXT =
        PFN_vkGetMemoryHostPointerPropertiesEXT( vkGetDeviceProcAddr( device, "vkGetMemoryHostPointerPropertiesEXT" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetMemoryWin32HandleKHR =
        PFN_vkGetMemoryWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetMemoryWin32HandleNV =
        PFN_vkGetMemoryWin32HandleNV( vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandleNV" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetMemoryWin32HandlePropertiesKHR =
        PFN_vkGetMemoryWin32HandlePropertiesKHR( vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandlePropertiesKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkGetMemoryZirconHandleFUCHSIA =
        PFN_vkGetMemoryZirconHandleFUCHSIA( vkGetDeviceProcAddr( device, "vkGetMemoryZirconHandleFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkGetMemoryZirconHandlePropertiesFUCHSIA = PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA(
        vkGetDeviceProcAddr( device, "vkGetMemoryZirconHandlePropertiesFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      vkGetPastPresentationTimingGOOGLE =
        PFN_vkGetPastPresentationTimingGOOGLE( vkGetDeviceProcAddr( device, "vkGetPastPresentationTimingGOOGLE" ) );
      vkGetPerformanceParameterINTEL =
        PFN_vkGetPerformanceParameterINTEL( vkGetDeviceProcAddr( device, "vkGetPerformanceParameterINTEL" ) );
      vkGetPipelineCacheData = PFN_vkGetPipelineCacheData( vkGetDeviceProcAddr( device, "vkGetPipelineCacheData" ) );
      vkGetPipelineExecutableInternalRepresentationsKHR = PFN_vkGetPipelineExecutableInternalRepresentationsKHR(
        vkGetDeviceProcAddr( device, "vkGetPipelineExecutableInternalRepresentationsKHR" ) );
      vkGetPipelineExecutablePropertiesKHR = PFN_vkGetPipelineExecutablePropertiesKHR(
        vkGetDeviceProcAddr( device, "vkGetPipelineExecutablePropertiesKHR" ) );
      vkGetPipelineExecutableStatisticsKHR = PFN_vkGetPipelineExecutableStatisticsKHR(
        vkGetDeviceProcAddr( device, "vkGetPipelineExecutableStatisticsKHR" ) );
      vkGetPrivateDataEXT   = PFN_vkGetPrivateDataEXT( vkGetDeviceProcAddr( device, "vkGetPrivateDataEXT" ) );
      vkGetQueryPoolResults = PFN_vkGetQueryPoolResults( vkGetDeviceProcAddr( device, "vkGetQueryPoolResults" ) );
      vkGetQueueCheckpointData2NV =
        PFN_vkGetQueueCheckpointData2NV( vkGetDeviceProcAddr( device, "vkGetQueueCheckpointData2NV" ) );
      vkGetQueueCheckpointDataNV =
        PFN_vkGetQueueCheckpointDataNV( vkGetDeviceProcAddr( device, "vkGetQueueCheckpointDataNV" ) );
      vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(
        vkGetDeviceProcAddr( device, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR" ) );
      vkGetRayTracingShaderGroupHandlesKHR = PFN_vkGetRayTracingShaderGroupHandlesKHR(
        vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupHandlesKHR" ) );
      vkGetRayTracingShaderGroupHandlesNV =
        PFN_vkGetRayTracingShaderGroupHandlesNV( vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupHandlesNV" ) );
      if ( !vkGetRayTracingShaderGroupHandlesKHR )
        vkGetRayTracingShaderGroupHandlesKHR = vkGetRayTracingShaderGroupHandlesNV;
      vkGetRayTracingShaderGroupStackSizeKHR = PFN_vkGetRayTracingShaderGroupStackSizeKHR(
        vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupStackSizeKHR" ) );
      vkGetRefreshCycleDurationGOOGLE =
        PFN_vkGetRefreshCycleDurationGOOGLE( vkGetDeviceProcAddr( device, "vkGetRefreshCycleDurationGOOGLE" ) );
      vkGetRenderAreaGranularity =
        PFN_vkGetRenderAreaGranularity( vkGetDeviceProcAddr( device, "vkGetRenderAreaGranularity" ) );
      vkGetSemaphoreCounterValue =
        PFN_vkGetSemaphoreCounterValue( vkGetDeviceProcAddr( device, "vkGetSemaphoreCounterValue" ) );
      vkGetSemaphoreCounterValueKHR =
        PFN_vkGetSemaphoreCounterValueKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreCounterValueKHR" ) );
      if ( !vkGetSemaphoreCounterValue )
        vkGetSemaphoreCounterValue = vkGetSemaphoreCounterValueKHR;
      vkGetSemaphoreFdKHR = PFN_vkGetSemaphoreFdKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreFdKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkGetSemaphoreWin32HandleKHR =
        PFN_vkGetSemaphoreWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkGetSemaphoreZirconHandleFUCHSIA =
        PFN_vkGetSemaphoreZirconHandleFUCHSIA( vkGetDeviceProcAddr( device, "vkGetSemaphoreZirconHandleFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      vkGetShaderInfoAMD = PFN_vkGetShaderInfoAMD( vkGetDeviceProcAddr( device, "vkGetShaderInfoAMD" ) );
      vkGetSubpassShadingMaxWorkgroupSizeHUAWEI = PFN_vkGetSubpassShadingMaxWorkgroupSizeHUAWEI(
        vkGetDeviceProcAddr( device, "vkGetSubpassShadingMaxWorkgroupSizeHUAWEI" ) );
      vkGetSwapchainCounterEXT =
        PFN_vkGetSwapchainCounterEXT( vkGetDeviceProcAddr( device, "vkGetSwapchainCounterEXT" ) );
      vkGetSwapchainImagesKHR = PFN_vkGetSwapchainImagesKHR( vkGetDeviceProcAddr( device, "vkGetSwapchainImagesKHR" ) );
      vkGetSwapchainStatusKHR = PFN_vkGetSwapchainStatusKHR( vkGetDeviceProcAddr( device, "vkGetSwapchainStatusKHR" ) );
      vkGetValidationCacheDataEXT =
        PFN_vkGetValidationCacheDataEXT( vkGetDeviceProcAddr( device, "vkGetValidationCacheDataEXT" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkGetVideoSessionMemoryRequirementsKHR = PFN_vkGetVideoSessionMemoryRequirementsKHR(
        vkGetDeviceProcAddr( device, "vkGetVideoSessionMemoryRequirementsKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkImportFenceFdKHR = PFN_vkImportFenceFdKHR( vkGetDeviceProcAddr( device, "vkImportFenceFdKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkImportFenceWin32HandleKHR =
        PFN_vkImportFenceWin32HandleKHR( vkGetDeviceProcAddr( device, "vkImportFenceWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkImportSemaphoreFdKHR = PFN_vkImportSemaphoreFdKHR( vkGetDeviceProcAddr( device, "vkImportSemaphoreFdKHR" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkImportSemaphoreWin32HandleKHR =
        PFN_vkImportSemaphoreWin32HandleKHR( vkGetDeviceProcAddr( device, "vkImportSemaphoreWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      vkImportSemaphoreZirconHandleFUCHSIA = PFN_vkImportSemaphoreZirconHandleFUCHSIA(
        vkGetDeviceProcAddr( device, "vkImportSemaphoreZirconHandleFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      vkInitializePerformanceApiINTEL =
        PFN_vkInitializePerformanceApiINTEL( vkGetDeviceProcAddr( device, "vkInitializePerformanceApiINTEL" ) );
      vkInvalidateMappedMemoryRanges =
        PFN_vkInvalidateMappedMemoryRanges( vkGetDeviceProcAddr( device, "vkInvalidateMappedMemoryRanges" ) );
      vkMapMemory           = PFN_vkMapMemory( vkGetDeviceProcAddr( device, "vkMapMemory" ) );
      vkMergePipelineCaches = PFN_vkMergePipelineCaches( vkGetDeviceProcAddr( device, "vkMergePipelineCaches" ) );
      vkMergeValidationCachesEXT =
        PFN_vkMergeValidationCachesEXT( vkGetDeviceProcAddr( device, "vkMergeValidationCachesEXT" ) );
      vkQueueBeginDebugUtilsLabelEXT =
        PFN_vkQueueBeginDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueBeginDebugUtilsLabelEXT" ) );
      vkQueueBindSparse = PFN_vkQueueBindSparse( vkGetDeviceProcAddr( device, "vkQueueBindSparse" ) );
      vkQueueEndDebugUtilsLabelEXT =
        PFN_vkQueueEndDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueEndDebugUtilsLabelEXT" ) );
      vkQueueInsertDebugUtilsLabelEXT =
        PFN_vkQueueInsertDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueInsertDebugUtilsLabelEXT" ) );
      vkQueuePresentKHR = PFN_vkQueuePresentKHR( vkGetDeviceProcAddr( device, "vkQueuePresentKHR" ) );
      vkQueueSetPerformanceConfigurationINTEL = PFN_vkQueueSetPerformanceConfigurationINTEL(
        vkGetDeviceProcAddr( device, "vkQueueSetPerformanceConfigurationINTEL" ) );
      vkQueueSubmit     = PFN_vkQueueSubmit( vkGetDeviceProcAddr( device, "vkQueueSubmit" ) );
      vkQueueSubmit2KHR = PFN_vkQueueSubmit2KHR( vkGetDeviceProcAddr( device, "vkQueueSubmit2KHR" ) );
      vkQueueWaitIdle   = PFN_vkQueueWaitIdle( vkGetDeviceProcAddr( device, "vkQueueWaitIdle" ) );
      vkRegisterDeviceEventEXT =
        PFN_vkRegisterDeviceEventEXT( vkGetDeviceProcAddr( device, "vkRegisterDeviceEventEXT" ) );
      vkRegisterDisplayEventEXT =
        PFN_vkRegisterDisplayEventEXT( vkGetDeviceProcAddr( device, "vkRegisterDisplayEventEXT" ) );
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      vkReleaseFullScreenExclusiveModeEXT =
        PFN_vkReleaseFullScreenExclusiveModeEXT( vkGetDeviceProcAddr( device, "vkReleaseFullScreenExclusiveModeEXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      vkReleasePerformanceConfigurationINTEL = PFN_vkReleasePerformanceConfigurationINTEL(
        vkGetDeviceProcAddr( device, "vkReleasePerformanceConfigurationINTEL" ) );
      vkReleaseProfilingLockKHR =
        PFN_vkReleaseProfilingLockKHR( vkGetDeviceProcAddr( device, "vkReleaseProfilingLockKHR" ) );
      vkResetCommandBuffer  = PFN_vkResetCommandBuffer( vkGetDeviceProcAddr( device, "vkResetCommandBuffer" ) );
      vkResetCommandPool    = PFN_vkResetCommandPool( vkGetDeviceProcAddr( device, "vkResetCommandPool" ) );
      vkResetDescriptorPool = PFN_vkResetDescriptorPool( vkGetDeviceProcAddr( device, "vkResetDescriptorPool" ) );
      vkResetEvent          = PFN_vkResetEvent( vkGetDeviceProcAddr( device, "vkResetEvent" ) );
      vkResetFences         = PFN_vkResetFences( vkGetDeviceProcAddr( device, "vkResetFences" ) );
      vkResetQueryPool      = PFN_vkResetQueryPool( vkGetDeviceProcAddr( device, "vkResetQueryPool" ) );
      vkResetQueryPoolEXT   = PFN_vkResetQueryPoolEXT( vkGetDeviceProcAddr( device, "vkResetQueryPoolEXT" ) );
      if ( !vkResetQueryPool )
        vkResetQueryPool = vkResetQueryPoolEXT;
      vkSetDebugUtilsObjectNameEXT =
        PFN_vkSetDebugUtilsObjectNameEXT( vkGetDeviceProcAddr( device, "vkSetDebugUtilsObjectNameEXT" ) );
      vkSetDebugUtilsObjectTagEXT =
        PFN_vkSetDebugUtilsObjectTagEXT( vkGetDeviceProcAddr( device, "vkSetDebugUtilsObjectTagEXT" ) );
      vkSetEvent           = PFN_vkSetEvent( vkGetDeviceProcAddr( device, "vkSetEvent" ) );
      vkSetHdrMetadataEXT  = PFN_vkSetHdrMetadataEXT( vkGetDeviceProcAddr( device, "vkSetHdrMetadataEXT" ) );
      vkSetLocalDimmingAMD = PFN_vkSetLocalDimmingAMD( vkGetDeviceProcAddr( device, "vkSetLocalDimmingAMD" ) );
      vkSetPrivateDataEXT  = PFN_vkSetPrivateDataEXT( vkGetDeviceProcAddr( device, "vkSetPrivateDataEXT" ) );
      vkSignalSemaphore    = PFN_vkSignalSemaphore( vkGetDeviceProcAddr( device, "vkSignalSemaphore" ) );
      vkSignalSemaphoreKHR = PFN_vkSignalSemaphoreKHR( vkGetDeviceProcAddr( device, "vkSignalSemaphoreKHR" ) );
      if ( !vkSignalSemaphore )
        vkSignalSemaphore = vkSignalSemaphoreKHR;
      vkTrimCommandPool    = PFN_vkTrimCommandPool( vkGetDeviceProcAddr( device, "vkTrimCommandPool" ) );
      vkTrimCommandPoolKHR = PFN_vkTrimCommandPoolKHR( vkGetDeviceProcAddr( device, "vkTrimCommandPoolKHR" ) );
      if ( !vkTrimCommandPool )
        vkTrimCommandPool = vkTrimCommandPoolKHR;
      vkUninitializePerformanceApiINTEL =
        PFN_vkUninitializePerformanceApiINTEL( vkGetDeviceProcAddr( device, "vkUninitializePerformanceApiINTEL" ) );
      vkUnmapMemory = PFN_vkUnmapMemory( vkGetDeviceProcAddr( device, "vkUnmapMemory" ) );
      vkUpdateDescriptorSetWithTemplate =
        PFN_vkUpdateDescriptorSetWithTemplate( vkGetDeviceProcAddr( device, "vkUpdateDescriptorSetWithTemplate" ) );
      vkUpdateDescriptorSetWithTemplateKHR = PFN_vkUpdateDescriptorSetWithTemplateKHR(
        vkGetDeviceProcAddr( device, "vkUpdateDescriptorSetWithTemplateKHR" ) );
      if ( !vkUpdateDescriptorSetWithTemplate )
        vkUpdateDescriptorSetWithTemplate = vkUpdateDescriptorSetWithTemplateKHR;
      vkUpdateDescriptorSets = PFN_vkUpdateDescriptorSets( vkGetDeviceProcAddr( device, "vkUpdateDescriptorSets" ) );
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      vkUpdateVideoSessionParametersKHR =
        PFN_vkUpdateVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkUpdateVideoSessionParametersKHR" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      vkWaitForFences     = PFN_vkWaitForFences( vkGetDeviceProcAddr( device, "vkWaitForFences" ) );
      vkWaitSemaphores    = PFN_vkWaitSemaphores( vkGetDeviceProcAddr( device, "vkWaitSemaphores" ) );
      vkWaitSemaphoresKHR = PFN_vkWaitSemaphoresKHR( vkGetDeviceProcAddr( device, "vkWaitSemaphoresKHR" ) );
      if ( !vkWaitSemaphores )
        vkWaitSemaphores = vkWaitSemaphoresKHR;
      vkWriteAccelerationStructuresPropertiesKHR = PFN_vkWriteAccelerationStructuresPropertiesKHR(
        vkGetDeviceProcAddr( device, "vkWriteAccelerationStructuresPropertiesKHR" ) );
    }
  };

}  // namespace VULKAN_HPP_NAMESPACE

namespace std
{
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureKHR const & accelerationStructureKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkAccelerationStructureKHR>{}(
        static_cast<VkAccelerationStructureKHR>( accelerationStructureKHR ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureNV const & accelerationStructureNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkAccelerationStructureNV>{}(
        static_cast<VkAccelerationStructureNV>( accelerationStructureNV ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Buffer>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Buffer const & buffer ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkBuffer>{}( static_cast<VkBuffer>( buffer ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferView>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferView const & bufferView ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkBufferView>{}( static_cast<VkBufferView>( bufferView ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBuffer>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandBuffer const & commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkCommandBuffer>{}( static_cast<VkCommandBuffer>( commandBuffer ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandPool>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandPool const & commandPool ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkCommandPool>{}( static_cast<VkCommandPool>( commandPool ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuFunctionNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CuFunctionNVX const & cuFunctionNVX ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkCuFunctionNVX>{}( static_cast<VkCuFunctionNVX>( cuFunctionNVX ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuModuleNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CuModuleNVX const & cuModuleNVX ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkCuModuleNVX>{}( static_cast<VkCuModuleNVX>( cuModuleNVX ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT const & debugReportCallbackEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDebugReportCallbackEXT>{}( static_cast<VkDebugReportCallbackEXT>( debugReportCallbackEXT ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT const & debugUtilsMessengerEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDebugUtilsMessengerEXT>{}( static_cast<VkDebugUtilsMessengerEXT>( debugUtilsMessengerEXT ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeferredOperationKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DeferredOperationKHR const & deferredOperationKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDeferredOperationKHR>{}( static_cast<VkDeferredOperationKHR>( deferredOperationKHR ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPool>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorPool const & descriptorPool ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDescriptorPool>{}( static_cast<VkDescriptorPool>( descriptorPool ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSet>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSet const & descriptorSet ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDescriptorSet>{}( static_cast<VkDescriptorSet>( descriptorSet ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayout>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DescriptorSetLayout const & descriptorSetLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDescriptorSetLayout>{}( static_cast<VkDescriptorSetLayout>( descriptorSetLayout ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate const & descriptorUpdateTemplate ) const
      VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDescriptorUpdateTemplate>{}(
        static_cast<VkDescriptorUpdateTemplate>( descriptorUpdateTemplate ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Device>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Device const & device ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDevice>{}( static_cast<VkDevice>( device ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemory>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceMemory const & deviceMemory ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDeviceMemory>{}( static_cast<VkDeviceMemory>( deviceMemory ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayKHR const & displayKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDisplayKHR>{}( static_cast<VkDisplayKHR>( displayKHR ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayModeKHR const & displayModeKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDisplayModeKHR>{}( static_cast<VkDisplayModeKHR>( displayModeKHR ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Event>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Event const & event ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkEvent>{}( static_cast<VkEvent>( event ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Fence>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Fence const & fence ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkFence>{}( static_cast<VkFence>( fence ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Framebuffer>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Framebuffer const & framebuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkFramebuffer>{}( static_cast<VkFramebuffer>( framebuffer ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Image>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Image const & image ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkImage>{}( static_cast<VkImage>( image ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageView>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageView const & imageView ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkImageView>{}( static_cast<VkImageView>( imageView ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV const & indirectCommandsLayoutNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkIndirectCommandsLayoutNV>{}(
        static_cast<VkIndirectCommandsLayoutNV>( indirectCommandsLayoutNV ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Instance>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Instance const & instance ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkInstance>{}( static_cast<VkInstance>( instance ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL const & performanceConfigurationINTEL )
      const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPerformanceConfigurationINTEL>{}(
        static_cast<VkPerformanceConfigurationINTEL>( performanceConfigurationINTEL ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevice const & physicalDevice ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPhysicalDevice>{}( static_cast<VkPhysicalDevice>( physicalDevice ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Pipeline>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Pipeline const & pipeline ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPipeline>{}( static_cast<VkPipeline>( pipeline ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCache>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCache const & pipelineCache ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPipelineCache>{}( static_cast<VkPipelineCache>( pipelineCache ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineLayout>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineLayout const & pipelineLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPipelineLayout>{}( static_cast<VkPipelineLayout>( pipelineLayout ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PrivateDataSlotEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PrivateDataSlotEXT const & privateDataSlotEXT ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPrivateDataSlotEXT>{}( static_cast<VkPrivateDataSlotEXT>( privateDataSlotEXT ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPool>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueryPool const & queryPool ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkQueryPool>{}( static_cast<VkQueryPool>( queryPool ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Queue>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Queue const & queue ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkQueue>{}( static_cast<VkQueue>( queue ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPass>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPass const & renderPass ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkRenderPass>{}( static_cast<VkRenderPass>( renderPass ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Sampler>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Sampler const & sampler ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSampler>{}( static_cast<VkSampler>( sampler ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion const & samplerYcbcrConversion ) const
      VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSamplerYcbcrConversion>{}( static_cast<VkSamplerYcbcrConversion>( samplerYcbcrConversion ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Semaphore>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Semaphore const & semaphore ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSemaphore>{}( static_cast<VkSemaphore>( semaphore ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModule>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ShaderModule const & shaderModule ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkShaderModule>{}( static_cast<VkShaderModule>( shaderModule ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceKHR const & surfaceKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSurfaceKHR>{}( static_cast<VkSurfaceKHR>( surfaceKHR ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SwapchainKHR const & swapchainKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSwapchainKHR>{}( static_cast<VkSwapchainKHR>( swapchainKHR ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationCacheEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::ValidationCacheEXT const & validationCacheEXT ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkValidationCacheEXT>{}( static_cast<VkValidationCacheEXT>( validationCacheEXT ) );
    }
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoSessionKHR const & videoSessionKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkVideoSessionKHR>{}( static_cast<VkVideoSessionKHR>( videoSessionKHR ) );
    }
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR const & videoSessionParametersKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkVideoSessionParametersKHR>{}(
        static_cast<VkVideoSessionParametersKHR>( videoSessionParametersKHR ) );
    }
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
}  // namespace std
#endif
