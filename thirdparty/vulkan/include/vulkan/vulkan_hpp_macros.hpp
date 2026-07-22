// Copyright 2015-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_HPP_MACROS_HPP
#define VULKAN_HPP_MACROS_HPP

#if defined( _MSVC_LANG )
#  define VULKAN_HPP_CPLUSPLUS _MSVC_LANG
#else
#  define VULKAN_HPP_CPLUSPLUS __cplusplus
#endif

#if 202002L < VULKAN_HPP_CPLUSPLUS
#  define VULKAN_HPP_CPP_VERSION 23
#elif 201703L < VULKAN_HPP_CPLUSPLUS
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

// include headers holding feature-test macros
#if 20 <= VULKAN_HPP_CPP_VERSION
#  include <version>
#else
#  include <ciso646>
#endif

#define VULKAN_HPP_STRINGIFY2( text ) #text
#define VULKAN_HPP_STRINGIFY( text )  VULKAN_HPP_STRINGIFY2( text )
#define VULKAN_HPP_NAMESPACE_STRING   VULKAN_HPP_STRINGIFY( VULKAN_HPP_NAMESPACE )

#if defined( __clang__ ) || defined( __GNUC__ ) || defined( __GNUG__ )
#  define VULKAN_HPP_COMPILE_WARNING( text ) _Pragma( VULKAN_HPP_STRINGIFY( GCC warning text ) )
#elif defined( _MSC_VER )
#  define VULKAN_HPP_COMPILE_WARNING( text ) _Pragma( VULKAN_HPP_STRINGIFY( message( __FILE__ "(" VULKAN_HPP_STRINGIFY( __LINE__ ) "): warning: " text ) ) )
#else
#  define VULKAN_HPP_COMPILE_WARNING( text )
#endif

#if defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
#  if !defined( VULKAN_HPP_NO_SMART_HANDLE )
#    define VULKAN_HPP_NO_SMART_HANDLE
#  endif
#endif

#if defined( VULKAN_HPP_NO_CONSTRUCTORS )
#  if !defined( VULKAN_HPP_NO_STRUCT_CONSTRUCTORS )
#    define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#  endif
#  if !defined( VULKAN_HPP_NO_UNION_CONSTRUCTORS )
#    define VULKAN_HPP_NO_UNION_CONSTRUCTORS
#  endif
#endif

#if defined( VULKAN_HPP_NO_SETTERS )
#  if !defined( VULKAN_HPP_NO_STRUCT_SETTERS )
#    define VULKAN_HPP_NO_STRUCT_SETTERS
#  endif
#  if !defined( VULKAN_HPP_NO_UNION_SETTERS )
#    define VULKAN_HPP_NO_UNION_SETTERS
#  endif
#endif

#if !defined( VULKAN_HPP_ASSERT )
#  define VULKAN_HPP_ASSERT assert
#endif

#if !defined( VULKAN_HPP_ASSERT_ON_RESULT )
#  define VULKAN_HPP_ASSERT_ON_RESULT VULKAN_HPP_ASSERT
#endif

#if !defined( VULKAN_HPP_STATIC_ASSERT )
#  define VULKAN_HPP_STATIC_ASSERT static_assert
#endif

#if !defined( VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL )
#  define VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL 1
#endif

#if !defined( __has_include )
#  define __has_include( x ) false
#endif

#if defined( __cpp_lib_three_way_comparison ) && ( 201907 <= __cpp_lib_three_way_comparison ) && __has_include( <compare> ) && !defined( VULKAN_HPP_NO_SPACESHIP_OPERATOR )
#  define VULKAN_HPP_HAS_SPACESHIP_OPERATOR
#endif

#if defined( __cpp_lib_span ) && ( 201803 <= __cpp_lib_span )
#  define VULKAN_HPP_SUPPORT_SPAN
#endif

#if defined( VULKAN_HPP_CXX_MODULE ) && !( defined( __cpp_modules ) && defined( __cpp_lib_modules ) )
VULKAN_HPP_COMPILE_WARNING( "This is a non-conforming implementation of C++ named modules and the standard library module." )
#endif

#ifndef VK_USE_64_BIT_PTR_DEFINES
#  if defined( __LP64__ ) || defined( _WIN64 ) || ( defined( __x86_64__ ) && !defined( __ILP32__ ) ) || defined( _M_X64 ) || defined( __ia64 ) || \
    defined( _M_IA64 ) || defined( __aarch64__ ) || defined( __powerpc64__ ) || ( defined( __riscv ) && __riscv_xlen == 64 )
#    define VK_USE_64_BIT_PTR_DEFINES 1
#  else
#    define VK_USE_64_BIT_PTR_DEFINES 0
#  endif
#endif

// 32-bit vulkan is not typesafe for non-dispatchable handles, so don't allow copy constructors on this platform by default.
// To enable this feature on 32-bit platforms please #define VULKAN_HPP_TYPESAFE_CONVERSION 1
// To disable this feature on 64-bit platforms please #define VULKAN_HPP_TYPESAFE_CONVERSION 0
#if ( VK_USE_64_BIT_PTR_DEFINES == 1 )
#  if !defined( VULKAN_HPP_TYPESAFE_CONVERSION )
#    define VULKAN_HPP_TYPESAFE_CONVERSION 1
#  endif
#endif

#if defined( __GNUC__ )
#  define GCC_VERSION ( __GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__ )
#endif

#if !defined( VULKAN_HPP_HAS_UNRESTRICTED_UNIONS )
#  if defined( __clang__ )
#    if __has_feature( cxx_unrestricted_unions )
#      define VULKAN_HPP_HAS_UNRESTRICTED_UNIONS
#    endif
#  elif defined( __GNUC__ )
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

#if ( VULKAN_HPP_TYPESAFE_CONVERSION == 1 )
#  define VULKAN_HPP_TYPESAFE_EXPLICIT
#else
#  define VULKAN_HPP_TYPESAFE_EXPLICIT explicit
#endif

#if defined( __cpp_constexpr )
#  define VULKAN_HPP_CONSTEXPR constexpr
#  if 201304 <= __cpp_constexpr
#    define VULKAN_HPP_CONSTEXPR_14 constexpr
#  else
#    define VULKAN_HPP_CONSTEXPR_14
#  endif
#  if 201603 <= __cpp_constexpr
#    define VULKAN_HPP_CONSTEXPR_17 constexpr
#  else
#    define VULKAN_HPP_CONSTEXPR_17
#  endif
#  if ( 201907 <= __cpp_constexpr ) && ( !defined( __GNUC__ ) || ( 110400 < GCC_VERSION ) )
#    define VULKAN_HPP_CONSTEXPR_20 constexpr
#  else
#    define VULKAN_HPP_CONSTEXPR_20
#  endif
#  define VULKAN_HPP_CONST_OR_CONSTEXPR constexpr
#else
#  define VULKAN_HPP_CONSTEXPR
#  define VULKAN_HPP_CONSTEXPR_14
#  define VULKAN_HPP_CONST_OR_CONSTEXPR const
#endif

#if !defined( VULKAN_HPP_CONSTEXPR_INLINE )
#  if 201606L <= __cpp_inline_variables
#    define VULKAN_HPP_CONSTEXPR_INLINE VULKAN_HPP_CONSTEXPR inline
#  else
#    define VULKAN_HPP_CONSTEXPR_INLINE VULKAN_HPP_CONSTEXPR
#  endif
#endif

#if !defined( VULKAN_HPP_NOEXCEPT )
#  if defined( _MSC_VER ) && ( _MSC_VER <= 1800 )
#    define VULKAN_HPP_NOEXCEPT
#  else
#    define VULKAN_HPP_NOEXCEPT     noexcept
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

#if 17 <= VULKAN_HPP_CPP_VERSION
#  define VULKAN_HPP_DEPRECATED_17( msg ) [[deprecated( msg )]]
#else
#  define VULKAN_HPP_DEPRECATED_17( msg )
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

namespace VULKAN_HPP_NAMESPACE
{
  namespace detail
  {
    class DispatchLoaderDynamic;

#if !defined( VULKAN_HPP_DEFAULT_DISPATCHER )
#  if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
    extern VULKAN_HPP_STORAGE_API DispatchLoaderDynamic defaultDispatchLoaderDynamic;
#  endif
#endif
  }  // namespace detail
}  // namespace VULKAN_HPP_NAMESPACE

#if !defined( VULKAN_HPP_DISPATCH_LOADER_DYNAMIC_TYPE )
#  define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC_TYPE VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic
#endif
#if !defined( VULKAN_HPP_DISPATCH_LOADER_STATIC_TYPE )
#  define VULKAN_HPP_DISPATCH_LOADER_STATIC_TYPE VULKAN_HPP_NAMESPACE::detail::DispatchLoaderStatic
#endif

#if !defined( VULKAN_HPP_DEFAULT_DISPATCHER_TYPE )
#  if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
#    define VULKAN_HPP_DEFAULT_DISPATCHER_TYPE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC_TYPE
#  else
#    define VULKAN_HPP_DEFAULT_DISPATCHER_TYPE VULKAN_HPP_DISPATCH_LOADER_STATIC_TYPE
#  endif
#endif

#if !defined( VULKAN_HPP_DEFAULT_DISPATCHER )
#  if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
#    define VULKAN_HPP_DEFAULT_DISPATCHER ::VULKAN_HPP_NAMESPACE::detail::defaultDispatchLoaderDynamic
#    define VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE                       \
      namespace VULKAN_HPP_NAMESPACE                                                 \
      {                                                                              \
        namespace detail                                                             \
        {                                                                            \
          VULKAN_HPP_STORAGE_API DispatchLoaderDynamic defaultDispatchLoaderDynamic; \
        }                                                                            \
      }
#  else
#    define VULKAN_HPP_DEFAULT_DISPATCHER ::VULKAN_HPP_NAMESPACE::detail::getDispatchLoaderStatic()
#    define VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#  endif
#endif

#if defined( VULKAN_HPP_NO_DEFAULT_DISPATCHER )
#  define VULKAN_HPP_DEFAULT_ASSIGNMENT( assignment )
#else
#  define VULKAN_HPP_DEFAULT_ASSIGNMENT( assignment ) = assignment
#endif
#define VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT VULKAN_HPP_DEFAULT_ASSIGNMENT( VULKAN_HPP_DEFAULT_DISPATCHER )

#if !defined( VULKAN_HPP_RAII_NAMESPACE )
#  define VULKAN_HPP_RAII_NAMESPACE        raii
#  define VULKAN_HPP_RAII_NAMESPACE_STRING VULKAN_HPP_STRINGIFY( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE )
#endif

#endif
