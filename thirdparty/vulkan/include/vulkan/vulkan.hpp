// Copyright 2015-2024 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_HPP
#define VULKAN_HPP

#include <algorithm>
#include <array>     // ArrayWrapperND
#include <string.h>  // strnlen
#include <string>    // std::string
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_hpp_macros.hpp>

#if 17 <= VULKAN_HPP_CPP_VERSION
#  include <string_view>
#endif

#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
#  include <tuple>   // std::tie
#  include <vector>  // std::vector
#endif

#if !defined( VULKAN_HPP_NO_EXCEPTIONS )
#  include <system_error>  // std::is_error_code_enum
#endif

#if ( VULKAN_HPP_ASSERT == assert )
#  include <cassert>
#endif

#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL == 1
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNX__ ) || defined( __Fuchsia__ )
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

#if defined( VULKAN_HPP_HAS_SPACESHIP_OPERATOR )
#  include <compare>
#endif

#if defined( VULKAN_HPP_SUPPORT_SPAN )
#  include <span>
#endif

static_assert( VK_HEADER_VERSION == 275, "Wrong VK_HEADER_VERSION!" );

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

// XLib.h defines True/False, which collides with our vk::True/vk::False
// ->  undef them and provide some namepace-secure constexpr
#if defined( True )
#  undef True
constexpr int True = 1;
#endif
#if defined( False )
#  undef False
constexpr int False = 0;
#endif

namespace VULKAN_HPP_NAMESPACE
{
  template <typename T, size_t N>
  class ArrayWrapper1D : public std::array<T, N>
  {
  public:
    VULKAN_HPP_CONSTEXPR ArrayWrapper1D() VULKAN_HPP_NOEXCEPT : std::array<T, N>() {}

    VULKAN_HPP_CONSTEXPR ArrayWrapper1D( std::array<T, N> const & data ) VULKAN_HPP_NOEXCEPT : std::array<T, N>( data ) {}

    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    VULKAN_HPP_CONSTEXPR_14 ArrayWrapper1D( std::string const & data ) VULKAN_HPP_NOEXCEPT
    {
      copy( data.data(), data.length() );
    }

#if 17 <= VULKAN_HPP_CPP_VERSION
    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    VULKAN_HPP_CONSTEXPR_14 ArrayWrapper1D( std::string_view data ) VULKAN_HPP_NOEXCEPT
    {
      copy( data.data(), data.length() );
    }
#endif

#if ( VK_USE_64_BIT_PTR_DEFINES == 0 )
    // on 32 bit compiles, needs overloads on index type int to resolve ambiguities
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
      return std::string( this->data(), strnlen( this->data(), N ) );
    }

#if 17 <= VULKAN_HPP_CPP_VERSION
    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    operator std::string_view() const
    {
      return std::string_view( this->data(), strnlen( this->data(), N ) );
    }
#endif

#if defined( VULKAN_HPP_HAS_SPACESHIP_OPERATOR )
    template <typename B = T, typename std::enable_if<std::is_same<B, char>::value, int>::type = 0>
    std::strong_ordering operator<=>( ArrayWrapper1D<char, N> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return *static_cast<std::array<char, N> const *>( this ) <=> *static_cast<std::array<char, N> const *>( &rhs );
    }
#else
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
#endif

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

  private:
    VULKAN_HPP_CONSTEXPR_14 void copy( char const * data, size_t len ) VULKAN_HPP_NOEXCEPT
    {
      size_t n = std::min( N, len );
      for ( size_t i = 0; i < n; ++i )
      {
        ( *this )[i] = data[i];
      }
      for ( size_t i = n; i < N; ++i )
      {
        ( *this )[i] = 0;
      }
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
    {
    }
  };

#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
  template <typename T>
  class ArrayProxy
  {
  public:
    VULKAN_HPP_CONSTEXPR ArrayProxy() VULKAN_HPP_NOEXCEPT
      : m_count( 0 )
      , m_ptr( nullptr )
    {
    }

    VULKAN_HPP_CONSTEXPR ArrayProxy( std::nullptr_t ) VULKAN_HPP_NOEXCEPT
      : m_count( 0 )
      , m_ptr( nullptr )
    {
    }

    ArrayProxy( T const & value ) VULKAN_HPP_NOEXCEPT
      : m_count( 1 )
      , m_ptr( &value )
    {
    }

    ArrayProxy( uint32_t count, T const * ptr ) VULKAN_HPP_NOEXCEPT
      : m_count( count )
      , m_ptr( ptr )
    {
    }

    template <std::size_t C>
    ArrayProxy( T const ( &ptr )[C] ) VULKAN_HPP_NOEXCEPT
      : m_count( C )
      , m_ptr( ptr )
    {
    }

#  if __GNUC__ >= 9
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Winit-list-lifetime"
#  endif

    ArrayProxy( std::initializer_list<T> const & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {
    }

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy( std::initializer_list<typename std::remove_const<T>::type> const & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {
    }

#  if __GNUC__ >= 9
#    pragma GCC diagnostic pop
#  endif

    // Any type with a .data() return type implicitly convertible to T*, and a .size() return type implicitly
    // convertible to size_t. The const version can capture temporaries, with lifetime ending at end of statement.
    template <typename V,
              typename std::enable_if<std::is_convertible<decltype( std::declval<V>().data() ), T *>::value &&
                                      std::is_convertible<decltype( std::declval<V>().size() ), std::size_t>::value>::type * = nullptr>
    ArrayProxy( V const & v ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( v.size() ) )
      , m_ptr( v.data() )
    {
    }

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

    T const * data() const VULKAN_HPP_NOEXCEPT
    {
      return m_ptr;
    }

  private:
    uint32_t  m_count;
    T const * m_ptr;
  };

  template <typename T>
  class ArrayProxyNoTemporaries
  {
  public:
    VULKAN_HPP_CONSTEXPR ArrayProxyNoTemporaries() VULKAN_HPP_NOEXCEPT
      : m_count( 0 )
      , m_ptr( nullptr )
    {
    }

    VULKAN_HPP_CONSTEXPR ArrayProxyNoTemporaries( std::nullptr_t ) VULKAN_HPP_NOEXCEPT
      : m_count( 0 )
      , m_ptr( nullptr )
    {
    }

    ArrayProxyNoTemporaries( T & value ) VULKAN_HPP_NOEXCEPT
      : m_count( 1 )
      , m_ptr( &value )
    {
    }

    template <typename V>
    ArrayProxyNoTemporaries( V && value ) = delete;

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( typename std::remove_const<T>::type & value ) VULKAN_HPP_NOEXCEPT
      : m_count( 1 )
      , m_ptr( &value )
    {
    }

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( typename std::remove_const<T>::type && value ) = delete;

    ArrayProxyNoTemporaries( uint32_t count, T * ptr ) VULKAN_HPP_NOEXCEPT
      : m_count( count )
      , m_ptr( ptr )
    {
    }

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( uint32_t count, typename std::remove_const<T>::type * ptr ) VULKAN_HPP_NOEXCEPT
      : m_count( count )
      , m_ptr( ptr )
    {
    }

    template <std::size_t C>
    ArrayProxyNoTemporaries( T ( &ptr )[C] ) VULKAN_HPP_NOEXCEPT
      : m_count( C )
      , m_ptr( ptr )
    {
    }

    template <std::size_t C>
    ArrayProxyNoTemporaries( T ( &&ptr )[C] ) = delete;

    template <std::size_t C, typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( typename std::remove_const<T>::type ( &ptr )[C] ) VULKAN_HPP_NOEXCEPT
      : m_count( C )
      , m_ptr( ptr )
    {
    }

    template <std::size_t C, typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( typename std::remove_const<T>::type ( &&ptr )[C] ) = delete;

    ArrayProxyNoTemporaries( std::initializer_list<T> const & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {
    }

    ArrayProxyNoTemporaries( std::initializer_list<T> const && list ) = delete;

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::initializer_list<typename std::remove_const<T>::type> const & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {
    }

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::initializer_list<typename std::remove_const<T>::type> const && list ) = delete;

    ArrayProxyNoTemporaries( std::initializer_list<T> & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {
    }

    ArrayProxyNoTemporaries( std::initializer_list<T> && list ) = delete;

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::initializer_list<typename std::remove_const<T>::type> & list ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( list.size() ) )
      , m_ptr( list.begin() )
    {
    }

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries( std::initializer_list<typename std::remove_const<T>::type> && list ) = delete;

    // Any type with a .data() return type implicitly convertible to T*, and a .size() return type implicitly convertible to size_t.
    template <typename V,
              typename std::enable_if<std::is_convertible<decltype( std::declval<V>().data() ), T *>::value &&
                                      std::is_convertible<decltype( std::declval<V>().size() ), std::size_t>::value>::type * = nullptr>
    ArrayProxyNoTemporaries( V & v ) VULKAN_HPP_NOEXCEPT
      : m_count( static_cast<uint32_t>( v.size() ) )
      , m_ptr( v.data() )
    {
    }

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
  class StridedArrayProxy : protected ArrayProxy<T>
  {
  public:
    using ArrayProxy<T>::ArrayProxy;

    StridedArrayProxy( uint32_t count, T const * ptr, uint32_t stride ) VULKAN_HPP_NOEXCEPT
      : ArrayProxy<T>( count, ptr )
      , m_stride( stride )
    {
      VULKAN_HPP_ASSERT( sizeof( T ) <= stride );
    }

    using ArrayProxy<T>::begin;

    const T * end() const VULKAN_HPP_NOEXCEPT
    {
      return reinterpret_cast<T const *>( static_cast<uint8_t const *>( begin() ) + size() * m_stride );
    }

    using ArrayProxy<T>::front;

    const T & back() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( begin() && size() );
      return *reinterpret_cast<T const *>( static_cast<uint8_t const *>( begin() ) + ( size() - 1 ) * m_stride );
    }

    using ArrayProxy<T>::empty;
    using ArrayProxy<T>::size;
    using ArrayProxy<T>::data;

    uint32_t stride() const
    {
      return m_stride;
    }

  private:
    uint32_t m_stride = sizeof( T );
  };

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
    static const bool value = std::is_same<T, typename std::tuple_element<Index, std::tuple<ChainElements...>>::type>::value ||
                              StructureChainContains<Index - 1, T, ChainElements...>::value;
  };

  template <typename T, typename... ChainElements>
  struct StructureChainContains<0, T, ChainElements...>
  {
    static const bool value = std::is_same<T, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value;
  };

  template <size_t Index, typename... ChainElements>
  struct StructureChainValidation
  {
    using TestType          = typename std::tuple_element<Index, std::tuple<ChainElements...>>::type;
    static const bool valid = StructExtends<TestType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value &&
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
      static_assert( StructureChainValidation<sizeof...( ChainElements ) - 1, ChainElements...>::valid, "The structure chain is not valid!" );
      link<sizeof...( ChainElements ) - 1>();
    }

    StructureChain( StructureChain const & rhs ) VULKAN_HPP_NOEXCEPT : std::tuple<ChainElements...>( rhs )
    {
      static_assert( StructureChainValidation<sizeof...( ChainElements ) - 1, ChainElements...>::valid, "The structure chain is not valid!" );
      link( &std::get<0>( *this ),
            &std::get<0>( rhs ),
            reinterpret_cast<VkBaseOutStructure *>( &std::get<0>( *this ) ),
            reinterpret_cast<VkBaseInStructure const *>( &std::get<0>( rhs ) ) );
    }

    StructureChain( StructureChain && rhs ) VULKAN_HPP_NOEXCEPT : std::tuple<ChainElements...>( std::forward<std::tuple<ChainElements...>>( rhs ) )
    {
      static_assert( StructureChainValidation<sizeof...( ChainElements ) - 1, ChainElements...>::valid, "The structure chain is not valid!" );
      link( &std::get<0>( *this ),
            &std::get<0>( rhs ),
            reinterpret_cast<VkBaseOutStructure *>( &std::get<0>( *this ) ),
            reinterpret_cast<VkBaseInStructure const *>( &std::get<0>( rhs ) ) );
    }

    StructureChain( ChainElements const &... elems ) VULKAN_HPP_NOEXCEPT : std::tuple<ChainElements...>( elems... )
    {
      static_assert( StructureChainValidation<sizeof...( ChainElements ) - 1, ChainElements...>::valid, "The structure chain is not valid!" );
      link<sizeof...( ChainElements ) - 1>();
    }

    StructureChain & operator=( StructureChain const & rhs ) VULKAN_HPP_NOEXCEPT
    {
      std::tuple<ChainElements...>::operator=( rhs );
      link( &std::get<0>( *this ),
            &std::get<0>( rhs ),
            reinterpret_cast<VkBaseOutStructure *>( &std::get<0>( *this ) ),
            reinterpret_cast<VkBaseInStructure const *>( &std::get<0>( rhs ) ) );
      return *this;
    }

    StructureChain & operator=( StructureChain && rhs ) = delete;

    template <typename T = typename std::tuple_element<0, std::tuple<ChainElements...>>::type, size_t Which = 0>
    T & get() VULKAN_HPP_NOEXCEPT
    {
      return std::get<ChainElementIndex<0, T, Which, void, ChainElements...>::value>( static_cast<std::tuple<ChainElements...> &>( *this ) );
    }

    template <typename T = typename std::tuple_element<0, std::tuple<ChainElements...>>::type, size_t Which = 0>
    T const & get() const VULKAN_HPP_NOEXCEPT
    {
      return std::get<ChainElementIndex<0, T, Which, void, ChainElements...>::value>( static_cast<std::tuple<ChainElements...> const &>( *this ) );
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

    // assign a complete structure to the StructureChain without modifying the chaining
    template <typename T = typename std::tuple_element<0, std::tuple<ChainElements...>>::type, size_t Which = 0>
    StructureChain & assign( const T & rhs ) VULKAN_HPP_NOEXCEPT
    {
      T &    lhs   = get<T, Which>();
      void * pNext = lhs.pNext;
      lhs          = rhs;
      lhs.pNext    = pNext;
      return *this;
    }

    template <typename ClassType, size_t Which = 0>
    typename std::enable_if<std::is_same<ClassType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value && ( Which == 0 ), bool>::type
      isLinked() const VULKAN_HPP_NOEXCEPT
    {
      return true;
    }

    template <typename ClassType, size_t Which = 0>
    typename std::enable_if<!std::is_same<ClassType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value || ( Which != 0 ), bool>::type
      isLinked() const VULKAN_HPP_NOEXCEPT
    {
      static_assert( IsPartOfStructureChain<ClassType, ChainElements...>::valid, "Can't unlink Structure that's not part of this StructureChain!" );
      return isLinked( reinterpret_cast<VkBaseInStructure const *>( &get<ClassType, Which>() ) );
    }

    template <typename ClassType, size_t Which = 0>
    typename std::enable_if<!std::is_same<ClassType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value || ( Which != 0 ), void>::type
      relink() VULKAN_HPP_NOEXCEPT
    {
      static_assert( IsPartOfStructureChain<ClassType, ChainElements...>::valid, "Can't relink Structure that's not part of this StructureChain!" );
      auto pNext = reinterpret_cast<VkBaseInStructure *>( &get<ClassType, Which>() );
      VULKAN_HPP_ASSERT( !isLinked( pNext ) );
      auto & headElement = std::get<0>( static_cast<std::tuple<ChainElements...> &>( *this ) );
      pNext->pNext       = reinterpret_cast<VkBaseInStructure const *>( headElement.pNext );
      headElement.pNext  = pNext;
    }

    template <typename ClassType, size_t Which = 0>
    typename std::enable_if<!std::is_same<ClassType, typename std::tuple_element<0, std::tuple<ChainElements...>>::type>::value || ( Which != 0 ), void>::type
      unlink() VULKAN_HPP_NOEXCEPT
    {
      static_assert( IsPartOfStructureChain<ClassType, ChainElements...>::valid, "Can't unlink Structure that's not part of this StructureChain!" );
      unlink( reinterpret_cast<VkBaseOutStructure const *>( &get<ClassType, Which>() ) );
    }

  private:
    template <int Index, typename T, int Which, typename, class First, class... Types>
    struct ChainElementIndex : ChainElementIndex<Index + 1, T, Which, void, Types...>
    {
    };

    template <int Index, typename T, int Which, class First, class... Types>
    struct ChainElementIndex<Index, T, Which, typename std::enable_if<!std::is_same<T, First>::value, void>::type, First, Types...>
      : ChainElementIndex<Index + 1, T, Which, void, Types...>
    {
    };

    template <int Index, typename T, int Which, class First, class... Types>
    struct ChainElementIndex<Index, T, Which, typename std::enable_if<std::is_same<T, First>::value, void>::type, First, Types...>
      : ChainElementIndex<Index + 1, T, Which - 1, void, Types...>
    {
    };

    template <int Index, typename T, class First, class... Types>
    struct ChainElementIndex<Index, T, 0, typename std::enable_if<std::is_same<T, First>::value, void>::type, First, Types...>
      : std::integral_constant<int, Index>
    {
    };

    bool isLinked( VkBaseInStructure const * pNext ) const VULKAN_HPP_NOEXCEPT
    {
      VkBaseInStructure const * elementPtr =
        reinterpret_cast<VkBaseInStructure const *>( &std::get<0>( static_cast<std::tuple<ChainElements...> const &>( *this ) ) );
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
    {
    }

    void link( void * dstBase, void const * srcBase, VkBaseOutStructure * dst, VkBaseInStructure const * src )
    {
      while ( src->pNext )
      {
        std::ptrdiff_t offset = reinterpret_cast<char const *>( src->pNext ) - reinterpret_cast<char const *>( srcBase );
        dst->pNext            = reinterpret_cast<VkBaseOutStructure *>( reinterpret_cast<char *>( dstBase ) + offset );
        dst                   = dst->pNext;
        src                   = src->pNext;
      }
      dst->pNext = nullptr;
    }

    void unlink( VkBaseOutStructure const * pNext ) VULKAN_HPP_NOEXCEPT
    {
      VkBaseOutStructure * elementPtr = reinterpret_cast<VkBaseOutStructure *>( &std::get<0>( static_cast<std::tuple<ChainElements...> &>( *this ) ) );
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

  // interupt the VULKAN_HPP_NAMESPACE for a moment to add specializations of std::tuple_size and std::tuple_element for the StructureChain!
}

namespace std
{
  template <typename... Elements>
  struct tuple_size<VULKAN_HPP_NAMESPACE::StructureChain<Elements...>>
  {
    static constexpr size_t value = std::tuple_size<std::tuple<Elements...>>::value;
  };

  template <std::size_t Index, typename... Elements>
  struct tuple_element<Index, VULKAN_HPP_NAMESPACE::StructureChain<Elements...>>
  {
    using type = typename std::tuple_element<Index, std::tuple<Elements...>>::type;
  };
}  // namespace std

namespace VULKAN_HPP_NAMESPACE
{

#  if !defined( VULKAN_HPP_NO_SMART_HANDLE )
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
    {
    }

    UniqueHandle( UniqueHandle const & ) = delete;

    UniqueHandle( UniqueHandle && other ) VULKAN_HPP_NOEXCEPT
      : Deleter( std::move( static_cast<Deleter &>( other ) ) )
      , m_value( other.release() )
    {
    }

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
  VULKAN_HPP_INLINE std::vector<typename UniqueType::element_type> uniqueToRaw( std::vector<UniqueType> const & handles )
  {
    std::vector<typename UniqueType::element_type> newBuffer( handles.size() );
    std::transform( handles.begin(), handles.end(), newBuffer.begin(), []( UniqueType const & handle ) { return handle.get(); } );
    return newBuffer;
  }

  template <typename Type, typename Dispatch>
  VULKAN_HPP_INLINE void swap( UniqueHandle<Type, Dispatch> & lhs, UniqueHandle<Type, Dispatch> & rhs ) VULKAN_HPP_NOEXCEPT
  {
    lhs.swap( rhs );
  }
#  endif
#endif  // VULKAN_HPP_DISABLE_ENHANCED_MODE

  class DispatchLoaderBase
  {
  public:
    DispatchLoaderBase() = default;
    DispatchLoaderBase( std::nullptr_t )
#if !defined( NDEBUG )
      : m_valid( false )
#endif
    {
    }

#if !defined( NDEBUG )
    size_t getVkHeaderVersion() const
    {
      VULKAN_HPP_ASSERT( m_valid );
      return vkHeaderVersion;
    }

  private:
    size_t vkHeaderVersion = VK_HEADER_VERSION;
    bool   m_valid         = true;
#endif
  };

#if !defined( VK_NO_PROTOTYPES )
  class DispatchLoaderStatic : public DispatchLoaderBase
  {
  public:
    //=== VK_VERSION_1_0 ===

    VkResult
      vkCreateInstance( const VkInstanceCreateInfo * pCreateInfo, const VkAllocationCallbacks * pAllocator, VkInstance * pInstance ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateInstance( pCreateInfo, pAllocator, pInstance );
    }

    void vkDestroyInstance( VkInstance instance, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyInstance( instance, pAllocator );
    }

    VkResult vkEnumeratePhysicalDevices( VkInstance instance, uint32_t * pPhysicalDeviceCount, VkPhysicalDevice * pPhysicalDevices ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumeratePhysicalDevices( instance, pPhysicalDeviceCount, pPhysicalDevices );
    }

    void vkGetPhysicalDeviceFeatures( VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures * pFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFeatures( physicalDevice, pFeatures );
    }

    void
      vkGetPhysicalDeviceFormatProperties( VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties * pFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFormatProperties( physicalDevice, format, pFormatProperties );
    }

    VkResult vkGetPhysicalDeviceImageFormatProperties( VkPhysicalDevice          physicalDevice,
                                                       VkFormat                  format,
                                                       VkImageType               type,
                                                       VkImageTiling             tiling,
                                                       VkImageUsageFlags         usage,
                                                       VkImageCreateFlags        flags,
                                                       VkImageFormatProperties * pImageFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceImageFormatProperties( physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties );
    }

    void vkGetPhysicalDeviceProperties( VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceProperties( physicalDevice, pProperties );
    }

    void vkGetPhysicalDeviceQueueFamilyProperties( VkPhysicalDevice          physicalDevice,
                                                   uint32_t *                pQueueFamilyPropertyCount,
                                                   VkQueueFamilyProperties * pQueueFamilyProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceQueueFamilyProperties( physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties );
    }

    void vkGetPhysicalDeviceMemoryProperties( VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties * pMemoryProperties ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkEnumerateInstanceLayerProperties( uint32_t * pPropertyCount, VkLayerProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumerateInstanceLayerProperties( pPropertyCount, pProperties );
    }

    VkResult
      vkEnumerateDeviceLayerProperties( VkPhysicalDevice physicalDevice, uint32_t * pPropertyCount, VkLayerProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumerateDeviceLayerProperties( physicalDevice, pPropertyCount, pProperties );
    }

    void vkGetDeviceQueue( VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue * pQueue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceQueue( device, queueFamilyIndex, queueIndex, pQueue );
    }

    VkResult vkQueueSubmit( VkQueue queue, uint32_t submitCount, const VkSubmitInfo * pSubmits, VkFence fence ) const VULKAN_HPP_NOEXCEPT
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

    void vkFreeMemory( VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkFreeMemory( device, memory, pAllocator );
    }

    VkResult vkMapMemory( VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void ** ppData ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkMapMemory( device, memory, offset, size, flags, ppData );
    }

    void vkUnmapMemory( VkDevice device, VkDeviceMemory memory ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUnmapMemory( device, memory );
    }

    VkResult vkFlushMappedMemoryRanges( VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange * pMemoryRanges ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkFlushMappedMemoryRanges( device, memoryRangeCount, pMemoryRanges );
    }

    VkResult vkInvalidateMappedMemoryRanges( VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange * pMemoryRanges ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkInvalidateMappedMemoryRanges( device, memoryRangeCount, pMemoryRanges );
    }

    void vkGetDeviceMemoryCommitment( VkDevice device, VkDeviceMemory memory, VkDeviceSize * pCommittedMemoryInBytes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceMemoryCommitment( device, memory, pCommittedMemoryInBytes );
    }

    VkResult vkBindBufferMemory( VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindBufferMemory( device, buffer, memory, memoryOffset );
    }

    VkResult vkBindImageMemory( VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindImageMemory( device, image, memory, memoryOffset );
    }

    void vkGetBufferMemoryRequirements( VkDevice device, VkBuffer buffer, VkMemoryRequirements * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferMemoryRequirements( device, buffer, pMemoryRequirements );
    }

    void vkGetImageMemoryRequirements( VkDevice device, VkImage image, VkMemoryRequirements * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageMemoryRequirements( device, image, pMemoryRequirements );
    }

    void vkGetImageSparseMemoryRequirements( VkDevice                          device,
                                             VkImage                           image,
                                             uint32_t *                        pSparseMemoryRequirementCount,
                                             VkSparseImageMemoryRequirements * pSparseMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSparseMemoryRequirements( device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements );
    }

    void vkGetPhysicalDeviceSparseImageFormatProperties( VkPhysicalDevice                physicalDevice,
                                                         VkFormat                        format,
                                                         VkImageType                     type,
                                                         VkSampleCountFlagBits           samples,
                                                         VkImageUsageFlags               usage,
                                                         VkImageTiling                   tiling,
                                                         uint32_t *                      pPropertyCount,
                                                         VkSparseImageFormatProperties * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSparseImageFormatProperties( physicalDevice, format, type, samples, usage, tiling, pPropertyCount, pProperties );
    }

    VkResult vkQueueBindSparse( VkQueue queue, uint32_t bindInfoCount, const VkBindSparseInfo * pBindInfo, VkFence fence ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyFence( VkDevice device, VkFence fence, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkWaitForFences( VkDevice device, uint32_t fenceCount, const VkFence * pFences, VkBool32 waitAll, uint64_t timeout ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroySemaphore( VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyEvent( VkDevice device, VkEvent event, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyQueryPool( VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyBuffer( VkDevice device, VkBuffer buffer, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyBufferView( VkDevice device, VkBufferView bufferView, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyImage( VkDevice device, VkImage image, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyImageView( VkDevice device, VkImageView imageView, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyShaderModule( VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyPipelineCache( VkDevice device, VkPipelineCache pipelineCache, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyPipelineCache( device, pipelineCache, pAllocator );
    }

    VkResult vkGetPipelineCacheData( VkDevice device, VkPipelineCache pipelineCache, size_t * pDataSize, void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineCacheData( device, pipelineCache, pDataSize, pData );
    }

    VkResult
      vkMergePipelineCaches( VkDevice device, VkPipelineCache dstCache, uint32_t srcCacheCount, const VkPipelineCache * pSrcCaches ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCreateGraphicsPipelines( device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines );
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

    void vkDestroyPipeline( VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyPipelineLayout( VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroySampler( VkDevice device, VkSampler sampler, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyDescriptorPool( VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDescriptorPool( device, descriptorPool, pAllocator );
    }

    VkResult vkResetDescriptorPool( VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorPoolResetFlags flags ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkUpdateDescriptorSets( device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies );
    }

    VkResult vkCreateFramebuffer( VkDevice                        device,
                                  const VkFramebufferCreateInfo * pCreateInfo,
                                  const VkAllocationCallbacks *   pAllocator,
                                  VkFramebuffer *                 pFramebuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateFramebuffer( device, pCreateInfo, pAllocator, pFramebuffer );
    }

    void vkDestroyFramebuffer( VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyRenderPass( VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyRenderPass( device, renderPass, pAllocator );
    }

    void vkGetRenderAreaGranularity( VkDevice device, VkRenderPass renderPass, VkExtent2D * pGranularity ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyCommandPool( VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyCommandPool( device, commandPool, pAllocator );
    }

    VkResult vkResetCommandPool( VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkBeginCommandBuffer( VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo * pBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBeginCommandBuffer( commandBuffer, pBeginInfo );
    }

    VkResult vkEndCommandBuffer( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEndCommandBuffer( commandBuffer );
    }

    VkResult vkResetCommandBuffer( VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetCommandBuffer( commandBuffer, flags );
    }

    void vkCmdBindPipeline( VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindPipeline( commandBuffer, pipelineBindPoint, pipeline );
    }

    void
      vkCmdSetViewport( VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount, const VkViewport * pViewports ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewport( commandBuffer, firstViewport, viewportCount, pViewports );
    }

    void vkCmdSetScissor( VkCommandBuffer commandBuffer, uint32_t firstScissor, uint32_t scissorCount, const VkRect2D * pScissors ) const VULKAN_HPP_NOEXCEPT
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

    void vkCmdSetBlendConstants( VkCommandBuffer commandBuffer, const float blendConstants[4] ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetBlendConstants( commandBuffer, blendConstants );
    }

    void vkCmdSetDepthBounds( VkCommandBuffer commandBuffer, float minDepthBounds, float maxDepthBounds ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBounds( commandBuffer, minDepthBounds, maxDepthBounds );
    }

    void vkCmdSetStencilCompareMask( VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t compareMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilCompareMask( commandBuffer, faceMask, compareMask );
    }

    void vkCmdSetStencilWriteMask( VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t writeMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilWriteMask( commandBuffer, faceMask, writeMask );
    }

    void vkCmdSetStencilReference( VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t reference ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdBindDescriptorSets(
        commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets );
    }

    void vkCmdBindIndexBuffer( VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkIndexType indexType ) const VULKAN_HPP_NOEXCEPT
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

    void vkCmdDraw( VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance ) const
      VULKAN_HPP_NOEXCEPT
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

    void vkCmdDrawIndirect( VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndirect( commandBuffer, buffer, offset, drawCount, stride );
    }

    void vkCmdDrawIndexedIndirect( VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexedIndirect( commandBuffer, buffer, offset, drawCount, stride );
    }

    void vkCmdDispatch( VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatch( commandBuffer, groupCountX, groupCountY, groupCountZ );
    }

    void vkCmdDispatchIndirect( VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatchIndirect( commandBuffer, buffer, offset );
    }

    void vkCmdCopyBuffer( VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy * pRegions ) const
      VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdCopyImage( commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions );
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
      return ::vkCmdBlitImage( commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter );
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

    void vkCmdUpdateBuffer( VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize, const void * pData ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdUpdateBuffer( commandBuffer, dstBuffer, dstOffset, dataSize, pData );
    }

    void
      vkCmdFillBuffer( VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdResolveImage( commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions );
    }

    void vkCmdSetEvent( VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetEvent( commandBuffer, event, stageMask );
    }

    void vkCmdResetEvent( VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask ) const VULKAN_HPP_NOEXCEPT
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

    void vkCmdBeginQuery( VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginQuery( commandBuffer, queryPool, query, flags );
    }

    void vkCmdEndQuery( VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndQuery( commandBuffer, queryPool, query );
    }

    void vkCmdResetQueryPool( VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdCopyQueryPoolResults( commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags );
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

    void vkCmdExecuteCommands( VkCommandBuffer commandBuffer, uint32_t commandBufferCount, const VkCommandBuffer * pCommandBuffers ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdExecuteCommands( commandBuffer, commandBufferCount, pCommandBuffers );
    }

    //=== VK_VERSION_1_1 ===

    VkResult vkEnumerateInstanceVersion( uint32_t * pApiVersion ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumerateInstanceVersion( pApiVersion );
    }

    VkResult vkBindBufferMemory2( VkDevice device, uint32_t bindInfoCount, const VkBindBufferMemoryInfo * pBindInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindBufferMemory2( device, bindInfoCount, pBindInfos );
    }

    VkResult vkBindImageMemory2( VkDevice device, uint32_t bindInfoCount, const VkBindImageMemoryInfo * pBindInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindImageMemory2( device, bindInfoCount, pBindInfos );
    }

    void vkGetDeviceGroupPeerMemoryFeatures( VkDevice                   device,
                                             uint32_t                   heapIndex,
                                             uint32_t                   localDeviceIndex,
                                             uint32_t                   remoteDeviceIndex,
                                             VkPeerMemoryFeatureFlags * pPeerMemoryFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceGroupPeerMemoryFeatures( device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures );
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
      return ::vkCmdDispatchBase( commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ );
    }

    VkResult vkEnumeratePhysicalDeviceGroups( VkInstance                        instance,
                                              uint32_t *                        pPhysicalDeviceGroupCount,
                                              VkPhysicalDeviceGroupProperties * pPhysicalDeviceGroupProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumeratePhysicalDeviceGroups( instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties );
    }

    void vkGetImageMemoryRequirements2( VkDevice                               device,
                                        const VkImageMemoryRequirementsInfo2 * pInfo,
                                        VkMemoryRequirements2 *                pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageMemoryRequirements2( device, pInfo, pMemoryRequirements );
    }

    void vkGetBufferMemoryRequirements2( VkDevice                                device,
                                         const VkBufferMemoryRequirementsInfo2 * pInfo,
                                         VkMemoryRequirements2 *                 pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferMemoryRequirements2( device, pInfo, pMemoryRequirements );
    }

    void vkGetImageSparseMemoryRequirements2( VkDevice                                     device,
                                              const VkImageSparseMemoryRequirementsInfo2 * pInfo,
                                              uint32_t *                                   pSparseMemoryRequirementCount,
                                              VkSparseImageMemoryRequirements2 *           pSparseMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSparseMemoryRequirements2( device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements );
    }

    void vkGetPhysicalDeviceFeatures2( VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 * pFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFeatures2( physicalDevice, pFeatures );
    }

    void vkGetPhysicalDeviceProperties2( VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 * pProperties ) const VULKAN_HPP_NOEXCEPT
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
                                                        VkImageFormatProperties2 *               pImageFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceImageFormatProperties2( physicalDevice, pImageFormatInfo, pImageFormatProperties );
    }

    void vkGetPhysicalDeviceQueueFamilyProperties2( VkPhysicalDevice           physicalDevice,
                                                    uint32_t *                 pQueueFamilyPropertyCount,
                                                    VkQueueFamilyProperties2 * pQueueFamilyProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceQueueFamilyProperties2( physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties );
    }

    void vkGetPhysicalDeviceMemoryProperties2( VkPhysicalDevice                    physicalDevice,
                                               VkPhysicalDeviceMemoryProperties2 * pMemoryProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceMemoryProperties2( physicalDevice, pMemoryProperties );
    }

    void vkGetPhysicalDeviceSparseImageFormatProperties2( VkPhysicalDevice                               physicalDevice,
                                                          const VkPhysicalDeviceSparseImageFormatInfo2 * pFormatInfo,
                                                          uint32_t *                                     pPropertyCount,
                                                          VkSparseImageFormatProperties2 *               pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSparseImageFormatProperties2( physicalDevice, pFormatInfo, pPropertyCount, pProperties );
    }

    void vkTrimCommandPool( VkDevice device, VkCommandPool commandPool, VkCommandPoolTrimFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkTrimCommandPool( device, commandPool, flags );
    }

    void vkGetDeviceQueue2( VkDevice device, const VkDeviceQueueInfo2 * pQueueInfo, VkQueue * pQueue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceQueue2( device, pQueueInfo, pQueue );
    }

    VkResult vkCreateSamplerYcbcrConversion( VkDevice                                   device,
                                             const VkSamplerYcbcrConversionCreateInfo * pCreateInfo,
                                             const VkAllocationCallbacks *              pAllocator,
                                             VkSamplerYcbcrConversion *                 pYcbcrConversion ) const VULKAN_HPP_NOEXCEPT
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
                                               VkDescriptorUpdateTemplate *                 pDescriptorUpdateTemplate ) const VULKAN_HPP_NOEXCEPT
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
                                                      VkExternalBufferProperties *               pExternalBufferProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalBufferProperties( physicalDevice, pExternalBufferInfo, pExternalBufferProperties );
    }

    void vkGetPhysicalDeviceExternalFenceProperties( VkPhysicalDevice                          physicalDevice,
                                                     const VkPhysicalDeviceExternalFenceInfo * pExternalFenceInfo,
                                                     VkExternalFenceProperties *               pExternalFenceProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalFenceProperties( physicalDevice, pExternalFenceInfo, pExternalFenceProperties );
    }

    void vkGetPhysicalDeviceExternalSemaphoreProperties( VkPhysicalDevice                              physicalDevice,
                                                         const VkPhysicalDeviceExternalSemaphoreInfo * pExternalSemaphoreInfo,
                                                         VkExternalSemaphoreProperties *               pExternalSemaphoreProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalSemaphoreProperties( physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties );
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
      return ::vkCmdDrawIndirectCount( commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    void vkCmdDrawIndexedIndirectCount( VkCommandBuffer commandBuffer,
                                        VkBuffer        buffer,
                                        VkDeviceSize    offset,
                                        VkBuffer        countBuffer,
                                        VkDeviceSize    countBufferOffset,
                                        uint32_t        maxDrawCount,
                                        uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexedIndirectCount( commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
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

    void vkCmdEndRenderPass2( VkCommandBuffer commandBuffer, const VkSubpassEndInfo * pSubpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndRenderPass2( commandBuffer, pSubpassEndInfo );
    }

    void vkResetQueryPool( VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkResetQueryPool( device, queryPool, firstQuery, queryCount );
    }

    VkResult vkGetSemaphoreCounterValue( VkDevice device, VkSemaphore semaphore, uint64_t * pValue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreCounterValue( device, semaphore, pValue );
    }

    VkResult vkWaitSemaphores( VkDevice device, const VkSemaphoreWaitInfo * pWaitInfo, uint64_t timeout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWaitSemaphores( device, pWaitInfo, timeout );
    }

    VkResult vkSignalSemaphore( VkDevice device, const VkSemaphoreSignalInfo * pSignalInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSignalSemaphore( device, pSignalInfo );
    }

    VkDeviceAddress vkGetBufferDeviceAddress( VkDevice device, const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferDeviceAddress( device, pInfo );
    }

    uint64_t vkGetBufferOpaqueCaptureAddress( VkDevice device, const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferOpaqueCaptureAddress( device, pInfo );
    }

    uint64_t vkGetDeviceMemoryOpaqueCaptureAddress( VkDevice device, const VkDeviceMemoryOpaqueCaptureAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceMemoryOpaqueCaptureAddress( device, pInfo );
    }

    //=== VK_VERSION_1_3 ===

    VkResult vkGetPhysicalDeviceToolProperties( VkPhysicalDevice                 physicalDevice,
                                                uint32_t *                       pToolCount,
                                                VkPhysicalDeviceToolProperties * pToolProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceToolProperties( physicalDevice, pToolCount, pToolProperties );
    }

    VkResult vkCreatePrivateDataSlot( VkDevice                            device,
                                      const VkPrivateDataSlotCreateInfo * pCreateInfo,
                                      const VkAllocationCallbacks *       pAllocator,
                                      VkPrivateDataSlot *                 pPrivateDataSlot ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreatePrivateDataSlot( device, pCreateInfo, pAllocator, pPrivateDataSlot );
    }

    void vkDestroyPrivateDataSlot( VkDevice device, VkPrivateDataSlot privateDataSlot, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyPrivateDataSlot( device, privateDataSlot, pAllocator );
    }

    VkResult vkSetPrivateData( VkDevice device, VkObjectType objectType, uint64_t objectHandle, VkPrivateDataSlot privateDataSlot, uint64_t data ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetPrivateData( device, objectType, objectHandle, privateDataSlot, data );
    }

    void vkGetPrivateData( VkDevice device, VkObjectType objectType, uint64_t objectHandle, VkPrivateDataSlot privateDataSlot, uint64_t * pData ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPrivateData( device, objectType, objectHandle, privateDataSlot, pData );
    }

    void vkCmdSetEvent2( VkCommandBuffer commandBuffer, VkEvent event, const VkDependencyInfo * pDependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetEvent2( commandBuffer, event, pDependencyInfo );
    }

    void vkCmdResetEvent2( VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags2 stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResetEvent2( commandBuffer, event, stageMask );
    }

    void vkCmdWaitEvents2( VkCommandBuffer          commandBuffer,
                           uint32_t                 eventCount,
                           const VkEvent *          pEvents,
                           const VkDependencyInfo * pDependencyInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWaitEvents2( commandBuffer, eventCount, pEvents, pDependencyInfos );
    }

    void vkCmdPipelineBarrier2( VkCommandBuffer commandBuffer, const VkDependencyInfo * pDependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPipelineBarrier2( commandBuffer, pDependencyInfo );
    }

    void vkCmdWriteTimestamp2( VkCommandBuffer commandBuffer, VkPipelineStageFlags2 stage, VkQueryPool queryPool, uint32_t query ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteTimestamp2( commandBuffer, stage, queryPool, query );
    }

    VkResult vkQueueSubmit2( VkQueue queue, uint32_t submitCount, const VkSubmitInfo2 * pSubmits, VkFence fence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueSubmit2( queue, submitCount, pSubmits, fence );
    }

    void vkCmdCopyBuffer2( VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 * pCopyBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyBuffer2( commandBuffer, pCopyBufferInfo );
    }

    void vkCmdCopyImage2( VkCommandBuffer commandBuffer, const VkCopyImageInfo2 * pCopyImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyImage2( commandBuffer, pCopyImageInfo );
    }

    void vkCmdCopyBufferToImage2( VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 * pCopyBufferToImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyBufferToImage2( commandBuffer, pCopyBufferToImageInfo );
    }

    void vkCmdCopyImageToBuffer2( VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2 * pCopyImageToBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyImageToBuffer2( commandBuffer, pCopyImageToBufferInfo );
    }

    void vkCmdBlitImage2( VkCommandBuffer commandBuffer, const VkBlitImageInfo2 * pBlitImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBlitImage2( commandBuffer, pBlitImageInfo );
    }

    void vkCmdResolveImage2( VkCommandBuffer commandBuffer, const VkResolveImageInfo2 * pResolveImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResolveImage2( commandBuffer, pResolveImageInfo );
    }

    void vkCmdBeginRendering( VkCommandBuffer commandBuffer, const VkRenderingInfo * pRenderingInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginRendering( commandBuffer, pRenderingInfo );
    }

    void vkCmdEndRendering( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndRendering( commandBuffer );
    }

    void vkCmdSetCullMode( VkCommandBuffer commandBuffer, VkCullModeFlags cullMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCullMode( commandBuffer, cullMode );
    }

    void vkCmdSetFrontFace( VkCommandBuffer commandBuffer, VkFrontFace frontFace ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetFrontFace( commandBuffer, frontFace );
    }

    void vkCmdSetPrimitiveTopology( VkCommandBuffer commandBuffer, VkPrimitiveTopology primitiveTopology ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPrimitiveTopology( commandBuffer, primitiveTopology );
    }

    void vkCmdSetViewportWithCount( VkCommandBuffer commandBuffer, uint32_t viewportCount, const VkViewport * pViewports ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewportWithCount( commandBuffer, viewportCount, pViewports );
    }

    void vkCmdSetScissorWithCount( VkCommandBuffer commandBuffer, uint32_t scissorCount, const VkRect2D * pScissors ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetScissorWithCount( commandBuffer, scissorCount, pScissors );
    }

    void vkCmdBindVertexBuffers2( VkCommandBuffer      commandBuffer,
                                  uint32_t             firstBinding,
                                  uint32_t             bindingCount,
                                  const VkBuffer *     pBuffers,
                                  const VkDeviceSize * pOffsets,
                                  const VkDeviceSize * pSizes,
                                  const VkDeviceSize * pStrides ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindVertexBuffers2( commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes, pStrides );
    }

    void vkCmdSetDepthTestEnable( VkCommandBuffer commandBuffer, VkBool32 depthTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthTestEnable( commandBuffer, depthTestEnable );
    }

    void vkCmdSetDepthWriteEnable( VkCommandBuffer commandBuffer, VkBool32 depthWriteEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthWriteEnable( commandBuffer, depthWriteEnable );
    }

    void vkCmdSetDepthCompareOp( VkCommandBuffer commandBuffer, VkCompareOp depthCompareOp ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthCompareOp( commandBuffer, depthCompareOp );
    }

    void vkCmdSetDepthBoundsTestEnable( VkCommandBuffer commandBuffer, VkBool32 depthBoundsTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBoundsTestEnable( commandBuffer, depthBoundsTestEnable );
    }

    void vkCmdSetStencilTestEnable( VkCommandBuffer commandBuffer, VkBool32 stencilTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilTestEnable( commandBuffer, stencilTestEnable );
    }

    void vkCmdSetStencilOp( VkCommandBuffer    commandBuffer,
                            VkStencilFaceFlags faceMask,
                            VkStencilOp        failOp,
                            VkStencilOp        passOp,
                            VkStencilOp        depthFailOp,
                            VkCompareOp        compareOp ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetStencilOp( commandBuffer, faceMask, failOp, passOp, depthFailOp, compareOp );
    }

    void vkCmdSetRasterizerDiscardEnable( VkCommandBuffer commandBuffer, VkBool32 rasterizerDiscardEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetRasterizerDiscardEnable( commandBuffer, rasterizerDiscardEnable );
    }

    void vkCmdSetDepthBiasEnable( VkCommandBuffer commandBuffer, VkBool32 depthBiasEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBiasEnable( commandBuffer, depthBiasEnable );
    }

    void vkCmdSetPrimitiveRestartEnable( VkCommandBuffer commandBuffer, VkBool32 primitiveRestartEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPrimitiveRestartEnable( commandBuffer, primitiveRestartEnable );
    }

    void vkGetDeviceBufferMemoryRequirements( VkDevice                                 device,
                                              const VkDeviceBufferMemoryRequirements * pInfo,
                                              VkMemoryRequirements2 *                  pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceBufferMemoryRequirements( device, pInfo, pMemoryRequirements );
    }

    void vkGetDeviceImageMemoryRequirements( VkDevice                                device,
                                             const VkDeviceImageMemoryRequirements * pInfo,
                                             VkMemoryRequirements2 *                 pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceImageMemoryRequirements( device, pInfo, pMemoryRequirements );
    }

    void vkGetDeviceImageSparseMemoryRequirements( VkDevice                                device,
                                                   const VkDeviceImageMemoryRequirements * pInfo,
                                                   uint32_t *                              pSparseMemoryRequirementCount,
                                                   VkSparseImageMemoryRequirements2 *      pSparseMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceImageSparseMemoryRequirements( device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements );
    }

    //=== VK_KHR_surface ===

    void vkDestroySurfaceKHR( VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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
                                                        VkSurfaceCapabilitiesKHR * pSurfaceCapabilities ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroySwapchainKHR( VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkAcquireNextImageKHR(
      VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t * pImageIndex ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireNextImageKHR( device, swapchain, timeout, semaphore, fence, pImageIndex );
    }

    VkResult vkQueuePresentKHR( VkQueue queue, const VkPresentInfoKHR * pPresentInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueuePresentKHR( queue, pPresentInfo );
    }

    VkResult vkGetDeviceGroupPresentCapabilitiesKHR( VkDevice                              device,
                                                     VkDeviceGroupPresentCapabilitiesKHR * pDeviceGroupPresentCapabilities ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceGroupPresentCapabilitiesKHR( device, pDeviceGroupPresentCapabilities );
    }

    VkResult
      vkGetDeviceGroupSurfacePresentModesKHR( VkDevice device, VkSurfaceKHR surface, VkDeviceGroupPresentModeFlagsKHR * pModes ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkAcquireNextImage2KHR( VkDevice device, const VkAcquireNextImageInfoKHR * pAcquireInfo, uint32_t * pImageIndex ) const VULKAN_HPP_NOEXCEPT
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
                                                           VkDisplayPlanePropertiesKHR * pProperties ) const VULKAN_HPP_NOEXCEPT
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

    VkBool32 vkGetPhysicalDeviceWin32PresentationSupportKHR( VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceWin32PresentationSupportKHR( physicalDevice, queueFamilyIndex );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_EXT_debug_report ===

    VkResult vkCreateDebugReportCallbackEXT( VkInstance                                 instance,
                                             const VkDebugReportCallbackCreateInfoEXT * pCreateInfo,
                                             const VkAllocationCallbacks *              pAllocator,
                                             VkDebugReportCallbackEXT *                 pCallback ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkDebugReportMessageEXT( instance, flags, objectType, object, location, messageCode, pLayerPrefix, pMessage );
    }

    //=== VK_EXT_debug_marker ===

    VkResult vkDebugMarkerSetObjectTagEXT( VkDevice device, const VkDebugMarkerObjectTagInfoEXT * pTagInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDebugMarkerSetObjectTagEXT( device, pTagInfo );
    }

    VkResult vkDebugMarkerSetObjectNameEXT( VkDevice device, const VkDebugMarkerObjectNameInfoEXT * pNameInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDebugMarkerSetObjectNameEXT( device, pNameInfo );
    }

    void vkCmdDebugMarkerBeginEXT( VkCommandBuffer commandBuffer, const VkDebugMarkerMarkerInfoEXT * pMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDebugMarkerBeginEXT( commandBuffer, pMarkerInfo );
    }

    void vkCmdDebugMarkerEndEXT( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDebugMarkerEndEXT( commandBuffer );
    }

    void vkCmdDebugMarkerInsertEXT( VkCommandBuffer commandBuffer, const VkDebugMarkerMarkerInfoEXT * pMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDebugMarkerInsertEXT( commandBuffer, pMarkerInfo );
    }

    //=== VK_KHR_video_queue ===

    VkResult vkGetPhysicalDeviceVideoCapabilitiesKHR( VkPhysicalDevice              physicalDevice,
                                                      const VkVideoProfileInfoKHR * pVideoProfile,
                                                      VkVideoCapabilitiesKHR *      pCapabilities ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceVideoCapabilitiesKHR( physicalDevice, pVideoProfile, pCapabilities );
    }

    VkResult vkGetPhysicalDeviceVideoFormatPropertiesKHR( VkPhysicalDevice                           physicalDevice,
                                                          const VkPhysicalDeviceVideoFormatInfoKHR * pVideoFormatInfo,
                                                          uint32_t *                                 pVideoFormatPropertyCount,
                                                          VkVideoFormatPropertiesKHR *               pVideoFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceVideoFormatPropertiesKHR( physicalDevice, pVideoFormatInfo, pVideoFormatPropertyCount, pVideoFormatProperties );
    }

    VkResult vkCreateVideoSessionKHR( VkDevice                            device,
                                      const VkVideoSessionCreateInfoKHR * pCreateInfo,
                                      const VkAllocationCallbacks *       pAllocator,
                                      VkVideoSessionKHR *                 pVideoSession ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateVideoSessionKHR( device, pCreateInfo, pAllocator, pVideoSession );
    }

    void vkDestroyVideoSessionKHR( VkDevice device, VkVideoSessionKHR videoSession, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyVideoSessionKHR( device, videoSession, pAllocator );
    }

    VkResult vkGetVideoSessionMemoryRequirementsKHR( VkDevice                              device,
                                                     VkVideoSessionKHR                     videoSession,
                                                     uint32_t *                            pMemoryRequirementsCount,
                                                     VkVideoSessionMemoryRequirementsKHR * pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetVideoSessionMemoryRequirementsKHR( device, videoSession, pMemoryRequirementsCount, pMemoryRequirements );
    }

    VkResult vkBindVideoSessionMemoryKHR( VkDevice                                device,
                                          VkVideoSessionKHR                       videoSession,
                                          uint32_t                                bindSessionMemoryInfoCount,
                                          const VkBindVideoSessionMemoryInfoKHR * pBindSessionMemoryInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindVideoSessionMemoryKHR( device, videoSession, bindSessionMemoryInfoCount, pBindSessionMemoryInfos );
    }

    VkResult vkCreateVideoSessionParametersKHR( VkDevice                                      device,
                                                const VkVideoSessionParametersCreateInfoKHR * pCreateInfo,
                                                const VkAllocationCallbacks *                 pAllocator,
                                                VkVideoSessionParametersKHR *                 pVideoSessionParameters ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateVideoSessionParametersKHR( device, pCreateInfo, pAllocator, pVideoSessionParameters );
    }

    VkResult vkUpdateVideoSessionParametersKHR( VkDevice                                      device,
                                                VkVideoSessionParametersKHR                   videoSessionParameters,
                                                const VkVideoSessionParametersUpdateInfoKHR * pUpdateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUpdateVideoSessionParametersKHR( device, videoSessionParameters, pUpdateInfo );
    }

    void vkDestroyVideoSessionParametersKHR( VkDevice                      device,
                                             VkVideoSessionParametersKHR   videoSessionParameters,
                                             const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyVideoSessionParametersKHR( device, videoSessionParameters, pAllocator );
    }

    void vkCmdBeginVideoCodingKHR( VkCommandBuffer commandBuffer, const VkVideoBeginCodingInfoKHR * pBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginVideoCodingKHR( commandBuffer, pBeginInfo );
    }

    void vkCmdEndVideoCodingKHR( VkCommandBuffer commandBuffer, const VkVideoEndCodingInfoKHR * pEndCodingInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndVideoCodingKHR( commandBuffer, pEndCodingInfo );
    }

    void vkCmdControlVideoCodingKHR( VkCommandBuffer commandBuffer, const VkVideoCodingControlInfoKHR * pCodingControlInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdControlVideoCodingKHR( commandBuffer, pCodingControlInfo );
    }

    //=== VK_KHR_video_decode_queue ===

    void vkCmdDecodeVideoKHR( VkCommandBuffer commandBuffer, const VkVideoDecodeInfoKHR * pDecodeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDecodeVideoKHR( commandBuffer, pDecodeInfo );
    }

    //=== VK_EXT_transform_feedback ===

    void vkCmdBindTransformFeedbackBuffersEXT( VkCommandBuffer      commandBuffer,
                                               uint32_t             firstBinding,
                                               uint32_t             bindingCount,
                                               const VkBuffer *     pBuffers,
                                               const VkDeviceSize * pOffsets,
                                               const VkDeviceSize * pSizes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindTransformFeedbackBuffersEXT( commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes );
    }

    void vkCmdBeginTransformFeedbackEXT( VkCommandBuffer      commandBuffer,
                                         uint32_t             firstCounterBuffer,
                                         uint32_t             counterBufferCount,
                                         const VkBuffer *     pCounterBuffers,
                                         const VkDeviceSize * pCounterBufferOffsets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginTransformFeedbackEXT( commandBuffer, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets );
    }

    void vkCmdEndTransformFeedbackEXT( VkCommandBuffer      commandBuffer,
                                       uint32_t             firstCounterBuffer,
                                       uint32_t             counterBufferCount,
                                       const VkBuffer *     pCounterBuffers,
                                       const VkDeviceSize * pCounterBufferOffsets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndTransformFeedbackEXT( commandBuffer, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets );
    }

    void vkCmdBeginQueryIndexedEXT( VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags, uint32_t index ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginQueryIndexedEXT( commandBuffer, queryPool, query, flags, index );
    }

    void vkCmdEndQueryIndexedEXT( VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, uint32_t index ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdDrawIndirectByteCountEXT( commandBuffer, instanceCount, firstInstance, counterBuffer, counterBufferOffset, counterOffset, vertexStride );
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

    void vkDestroyCuModuleNVX( VkDevice device, VkCuModuleNVX module, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyCuModuleNVX( device, module, pAllocator );
    }

    void vkDestroyCuFunctionNVX( VkDevice device, VkCuFunctionNVX function, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyCuFunctionNVX( device, function, pAllocator );
    }

    void vkCmdCuLaunchKernelNVX( VkCommandBuffer commandBuffer, const VkCuLaunchInfoNVX * pLaunchInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCuLaunchKernelNVX( commandBuffer, pLaunchInfo );
    }

    //=== VK_NVX_image_view_handle ===

    uint32_t vkGetImageViewHandleNVX( VkDevice device, const VkImageViewHandleInfoNVX * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageViewHandleNVX( device, pInfo );
    }

    VkResult vkGetImageViewAddressNVX( VkDevice device, VkImageView imageView, VkImageViewAddressPropertiesNVX * pProperties ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdDrawIndirectCountAMD( commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    void vkCmdDrawIndexedIndirectCountAMD( VkCommandBuffer commandBuffer,
                                           VkBuffer        buffer,
                                           VkDeviceSize    offset,
                                           VkBuffer        countBuffer,
                                           VkDeviceSize    countBufferOffset,
                                           uint32_t        maxDrawCount,
                                           uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexedIndirectCountAMD( commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
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

    //=== VK_KHR_dynamic_rendering ===

    void vkCmdBeginRenderingKHR( VkCommandBuffer commandBuffer, const VkRenderingInfo * pRenderingInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginRenderingKHR( commandBuffer, pRenderingInfo );
    }

    void vkCmdEndRenderingKHR( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndRenderingKHR( commandBuffer );
    }

#  if defined( VK_USE_PLATFORM_GGP )
    //=== VK_GGP_stream_descriptor_surface ===

    VkResult vkCreateStreamDescriptorSurfaceGGP( VkInstance                                     instance,
                                                 const VkStreamDescriptorSurfaceCreateInfoGGP * pCreateInfo,
                                                 const VkAllocationCallbacks *                  pAllocator,
                                                 VkSurfaceKHR *                                 pSurface ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateStreamDescriptorSurfaceGGP( instance, pCreateInfo, pAllocator, pSurface );
    }
#  endif /*VK_USE_PLATFORM_GGP*/

    //=== VK_NV_external_memory_capabilities ===

    VkResult vkGetPhysicalDeviceExternalImageFormatPropertiesNV( VkPhysicalDevice                    physicalDevice,
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

    void vkGetPhysicalDeviceFeatures2KHR( VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 * pFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFeatures2KHR( physicalDevice, pFeatures );
    }

    void vkGetPhysicalDeviceProperties2KHR( VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 * pProperties ) const VULKAN_HPP_NOEXCEPT
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
                                                           VkImageFormatProperties2 *               pImageFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceImageFormatProperties2KHR( physicalDevice, pImageFormatInfo, pImageFormatProperties );
    }

    void vkGetPhysicalDeviceQueueFamilyProperties2KHR( VkPhysicalDevice           physicalDevice,
                                                       uint32_t *                 pQueueFamilyPropertyCount,
                                                       VkQueueFamilyProperties2 * pQueueFamilyProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceQueueFamilyProperties2KHR( physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties );
    }

    void vkGetPhysicalDeviceMemoryProperties2KHR( VkPhysicalDevice                    physicalDevice,
                                                  VkPhysicalDeviceMemoryProperties2 * pMemoryProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceMemoryProperties2KHR( physicalDevice, pMemoryProperties );
    }

    void vkGetPhysicalDeviceSparseImageFormatProperties2KHR( VkPhysicalDevice                               physicalDevice,
                                                             const VkPhysicalDeviceSparseImageFormatInfo2 * pFormatInfo,
                                                             uint32_t *                                     pPropertyCount,
                                                             VkSparseImageFormatProperties2 *               pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSparseImageFormatProperties2KHR( physicalDevice, pFormatInfo, pPropertyCount, pProperties );
    }

    //=== VK_KHR_device_group ===

    void vkGetDeviceGroupPeerMemoryFeaturesKHR( VkDevice                   device,
                                                uint32_t                   heapIndex,
                                                uint32_t                   localDeviceIndex,
                                                uint32_t                   remoteDeviceIndex,
                                                VkPeerMemoryFeatureFlags * pPeerMemoryFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceGroupPeerMemoryFeaturesKHR( device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures );
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
      return ::vkCmdDispatchBaseKHR( commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ );
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

    void vkTrimCommandPoolKHR( VkDevice device, VkCommandPool commandPool, VkCommandPoolTrimFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkTrimCommandPoolKHR( device, commandPool, flags );
    }

    //=== VK_KHR_device_group_creation ===

    VkResult vkEnumeratePhysicalDeviceGroupsKHR( VkInstance                        instance,
                                                 uint32_t *                        pPhysicalDeviceGroupCount,
                                                 VkPhysicalDeviceGroupProperties * pPhysicalDeviceGroupProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumeratePhysicalDeviceGroupsKHR( instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties );
    }

    //=== VK_KHR_external_memory_capabilities ===

    void vkGetPhysicalDeviceExternalBufferPropertiesKHR( VkPhysicalDevice                           physicalDevice,
                                                         const VkPhysicalDeviceExternalBufferInfo * pExternalBufferInfo,
                                                         VkExternalBufferProperties *               pExternalBufferProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalBufferPropertiesKHR( physicalDevice, pExternalBufferInfo, pExternalBufferProperties );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_memory_win32 ===

    VkResult vkGetMemoryWin32HandleKHR( VkDevice device, const VkMemoryGetWin32HandleInfoKHR * pGetWin32HandleInfo, HANDLE * pHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryWin32HandleKHR( device, pGetWin32HandleInfo, pHandle );
    }

    VkResult vkGetMemoryWin32HandlePropertiesKHR( VkDevice                           device,
                                                  VkExternalMemoryHandleTypeFlagBits handleType,
                                                  HANDLE                             handle,
                                                  VkMemoryWin32HandlePropertiesKHR * pMemoryWin32HandleProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryWin32HandlePropertiesKHR( device, handleType, handle, pMemoryWin32HandleProperties );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_memory_fd ===

    VkResult vkGetMemoryFdKHR( VkDevice device, const VkMemoryGetFdInfoKHR * pGetFdInfo, int * pFd ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryFdKHR( device, pGetFdInfo, pFd );
    }

    VkResult vkGetMemoryFdPropertiesKHR( VkDevice                           device,
                                         VkExternalMemoryHandleTypeFlagBits handleType,
                                         int                                fd,
                                         VkMemoryFdPropertiesKHR *          pMemoryFdProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryFdPropertiesKHR( device, handleType, fd, pMemoryFdProperties );
    }

    //=== VK_KHR_external_semaphore_capabilities ===

    void vkGetPhysicalDeviceExternalSemaphorePropertiesKHR( VkPhysicalDevice                              physicalDevice,
                                                            const VkPhysicalDeviceExternalSemaphoreInfo * pExternalSemaphoreInfo,
                                                            VkExternalSemaphoreProperties * pExternalSemaphoreProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalSemaphorePropertiesKHR( physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_semaphore_win32 ===

    VkResult vkImportSemaphoreWin32HandleKHR( VkDevice                                    device,
                                              const VkImportSemaphoreWin32HandleInfoKHR * pImportSemaphoreWin32HandleInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportSemaphoreWin32HandleKHR( device, pImportSemaphoreWin32HandleInfo );
    }

    VkResult
      vkGetSemaphoreWin32HandleKHR( VkDevice device, const VkSemaphoreGetWin32HandleInfoKHR * pGetWin32HandleInfo, HANDLE * pHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreWin32HandleKHR( device, pGetWin32HandleInfo, pHandle );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_semaphore_fd ===

    VkResult vkImportSemaphoreFdKHR( VkDevice device, const VkImportSemaphoreFdInfoKHR * pImportSemaphoreFdInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportSemaphoreFdKHR( device, pImportSemaphoreFdInfo );
    }

    VkResult vkGetSemaphoreFdKHR( VkDevice device, const VkSemaphoreGetFdInfoKHR * pGetFdInfo, int * pFd ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdPushDescriptorSetKHR( commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites );
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

    void vkCmdBeginConditionalRenderingEXT( VkCommandBuffer                            commandBuffer,
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
                                                  VkDescriptorUpdateTemplate *                 pDescriptorUpdateTemplate ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkAcquireXlibDisplayEXT( VkPhysicalDevice physicalDevice, Display * dpy, VkDisplayKHR display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireXlibDisplayEXT( physicalDevice, dpy, display );
    }

    VkResult vkGetRandROutputDisplayEXT( VkPhysicalDevice physicalDevice, Display * dpy, RROutput rrOutput, VkDisplayKHR * pDisplay ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRandROutputDisplayEXT( physicalDevice, dpy, rrOutput, pDisplay );
    }
#  endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

    //=== VK_EXT_display_surface_counter ===

    VkResult vkGetPhysicalDeviceSurfaceCapabilities2EXT( VkPhysicalDevice            physicalDevice,
                                                         VkSurfaceKHR                surface,
                                                         VkSurfaceCapabilities2EXT * pSurfaceCapabilities ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceCapabilities2EXT( physicalDevice, surface, pSurfaceCapabilities );
    }

    //=== VK_EXT_display_control ===

    VkResult vkDisplayPowerControlEXT( VkDevice device, VkDisplayKHR display, const VkDisplayPowerInfoEXT * pDisplayPowerInfo ) const VULKAN_HPP_NOEXCEPT
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
                                              VkRefreshCycleDurationGOOGLE * pDisplayTimingProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRefreshCycleDurationGOOGLE( device, swapchain, pDisplayTimingProperties );
    }

    VkResult vkGetPastPresentationTimingGOOGLE( VkDevice                         device,
                                                VkSwapchainKHR                   swapchain,
                                                uint32_t *                       pPresentationTimingCount,
                                                VkPastPresentationTimingGOOGLE * pPresentationTimings ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPastPresentationTimingGOOGLE( device, swapchain, pPresentationTimingCount, pPresentationTimings );
    }

    //=== VK_EXT_discard_rectangles ===

    void vkCmdSetDiscardRectangleEXT( VkCommandBuffer  commandBuffer,
                                      uint32_t         firstDiscardRectangle,
                                      uint32_t         discardRectangleCount,
                                      const VkRect2D * pDiscardRectangles ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDiscardRectangleEXT( commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles );
    }

    void vkCmdSetDiscardRectangleEnableEXT( VkCommandBuffer commandBuffer, VkBool32 discardRectangleEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDiscardRectangleEnableEXT( commandBuffer, discardRectangleEnable );
    }

    void vkCmdSetDiscardRectangleModeEXT( VkCommandBuffer commandBuffer, VkDiscardRectangleModeEXT discardRectangleMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDiscardRectangleModeEXT( commandBuffer, discardRectangleMode );
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

    void vkCmdEndRenderPass2KHR( VkCommandBuffer commandBuffer, const VkSubpassEndInfo * pSubpassEndInfo ) const VULKAN_HPP_NOEXCEPT
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
                                                        VkExternalFenceProperties *               pExternalFenceProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceExternalFencePropertiesKHR( physicalDevice, pExternalFenceInfo, pExternalFenceProperties );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_fence_win32 ===

    VkResult vkImportFenceWin32HandleKHR( VkDevice device, const VkImportFenceWin32HandleInfoKHR * pImportFenceWin32HandleInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportFenceWin32HandleKHR( device, pImportFenceWin32HandleInfo );
    }

    VkResult vkGetFenceWin32HandleKHR( VkDevice device, const VkFenceGetWin32HandleInfoKHR * pGetWin32HandleInfo, HANDLE * pHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetFenceWin32HandleKHR( device, pGetWin32HandleInfo, pHandle );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_fence_fd ===

    VkResult vkImportFenceFdKHR( VkDevice device, const VkImportFenceFdInfoKHR * pImportFenceFdInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportFenceFdKHR( device, pImportFenceFdInfo );
    }

    VkResult vkGetFenceFdKHR( VkDevice device, const VkFenceGetFdInfoKHR * pGetFdInfo, int * pFd ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetFenceFdKHR( device, pGetFdInfo, pFd );
    }

    //=== VK_KHR_performance_query ===

    VkResult
      vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR( VkPhysicalDevice                     physicalDevice,
                                                                       uint32_t                             queueFamilyIndex,
                                                                       uint32_t *                           pCounterCount,
                                                                       VkPerformanceCounterKHR *            pCounters,
                                                                       VkPerformanceCounterDescriptionKHR * pCounterDescriptions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
        physicalDevice, queueFamilyIndex, pCounterCount, pCounters, pCounterDescriptions );
    }

    void vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR( VkPhysicalDevice                            physicalDevice,
                                                                  const VkQueryPoolPerformanceCreateInfoKHR * pPerformanceQueryCreateInfo,
                                                                  uint32_t *                                  pNumPasses ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR( physicalDevice, pPerformanceQueryCreateInfo, pNumPasses );
    }

    VkResult vkAcquireProfilingLockKHR( VkDevice device, const VkAcquireProfilingLockInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
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
                                                         VkSurfaceCapabilities2KHR *             pSurfaceCapabilities ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceCapabilities2KHR( physicalDevice, pSurfaceInfo, pSurfaceCapabilities );
    }

    VkResult vkGetPhysicalDeviceSurfaceFormats2KHR( VkPhysicalDevice                        physicalDevice,
                                                    const VkPhysicalDeviceSurfaceInfo2KHR * pSurfaceInfo,
                                                    uint32_t *                              pSurfaceFormatCount,
                                                    VkSurfaceFormat2KHR *                   pSurfaceFormats ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfaceFormats2KHR( physicalDevice, pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats );
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
                                                            VkDisplayPlaneProperties2KHR * pProperties ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkGetDisplayPlaneCapabilities2KHR( VkPhysicalDevice                 physicalDevice,
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

    VkResult vkSetDebugUtilsObjectNameEXT( VkDevice device, const VkDebugUtilsObjectNameInfoEXT * pNameInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetDebugUtilsObjectNameEXT( device, pNameInfo );
    }

    VkResult vkSetDebugUtilsObjectTagEXT( VkDevice device, const VkDebugUtilsObjectTagInfoEXT * pTagInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetDebugUtilsObjectTagEXT( device, pTagInfo );
    }

    void vkQueueBeginDebugUtilsLabelEXT( VkQueue queue, const VkDebugUtilsLabelEXT * pLabelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueBeginDebugUtilsLabelEXT( queue, pLabelInfo );
    }

    void vkQueueEndDebugUtilsLabelEXT( VkQueue queue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueEndDebugUtilsLabelEXT( queue );
    }

    void vkQueueInsertDebugUtilsLabelEXT( VkQueue queue, const VkDebugUtilsLabelEXT * pLabelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueInsertDebugUtilsLabelEXT( queue, pLabelInfo );
    }

    void vkCmdBeginDebugUtilsLabelEXT( VkCommandBuffer commandBuffer, const VkDebugUtilsLabelEXT * pLabelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBeginDebugUtilsLabelEXT( commandBuffer, pLabelInfo );
    }

    void vkCmdEndDebugUtilsLabelEXT( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEndDebugUtilsLabelEXT( commandBuffer );
    }

    void vkCmdInsertDebugUtilsLabelEXT( VkCommandBuffer commandBuffer, const VkDebugUtilsLabelEXT * pLabelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdInsertDebugUtilsLabelEXT( commandBuffer, pLabelInfo );
    }

    VkResult vkCreateDebugUtilsMessengerEXT( VkInstance                                 instance,
                                             const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
                                             const VkAllocationCallbacks *              pAllocator,
                                             VkDebugUtilsMessengerEXT *                 pMessenger ) const VULKAN_HPP_NOEXCEPT
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
                                       const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSubmitDebugUtilsMessageEXT( instance, messageSeverity, messageTypes, pCallbackData );
    }

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
    //=== VK_ANDROID_external_memory_android_hardware_buffer ===

    VkResult vkGetAndroidHardwareBufferPropertiesANDROID( VkDevice                                   device,
                                                          const struct AHardwareBuffer *             buffer,
                                                          VkAndroidHardwareBufferPropertiesANDROID * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAndroidHardwareBufferPropertiesANDROID( device, buffer, pProperties );
    }

    VkResult vkGetMemoryAndroidHardwareBufferANDROID( VkDevice                                            device,
                                                      const VkMemoryGetAndroidHardwareBufferInfoANDROID * pInfo,
                                                      struct AHardwareBuffer **                           pBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryAndroidHardwareBufferANDROID( device, pInfo, pBuffer );
    }
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_AMDX_shader_enqueue ===

    VkResult vkCreateExecutionGraphPipelinesAMDX( VkDevice                                       device,
                                                  VkPipelineCache                                pipelineCache,
                                                  uint32_t                                       createInfoCount,
                                                  const VkExecutionGraphPipelineCreateInfoAMDX * pCreateInfos,
                                                  const VkAllocationCallbacks *                  pAllocator,
                                                  VkPipeline *                                   pPipelines ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateExecutionGraphPipelinesAMDX( device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines );
    }

    VkResult vkGetExecutionGraphPipelineScratchSizeAMDX( VkDevice                                  device,
                                                         VkPipeline                                executionGraph,
                                                         VkExecutionGraphPipelineScratchSizeAMDX * pSizeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetExecutionGraphPipelineScratchSizeAMDX( device, executionGraph, pSizeInfo );
    }

    VkResult vkGetExecutionGraphPipelineNodeIndexAMDX( VkDevice                                        device,
                                                       VkPipeline                                      executionGraph,
                                                       const VkPipelineShaderStageNodeCreateInfoAMDX * pNodeInfo,
                                                       uint32_t *                                      pNodeIndex ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetExecutionGraphPipelineNodeIndexAMDX( device, executionGraph, pNodeInfo, pNodeIndex );
    }

    void vkCmdInitializeGraphScratchMemoryAMDX( VkCommandBuffer commandBuffer, VkDeviceAddress scratch ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdInitializeGraphScratchMemoryAMDX( commandBuffer, scratch );
    }

    void vkCmdDispatchGraphAMDX( VkCommandBuffer                      commandBuffer,
                                 VkDeviceAddress                      scratch,
                                 const VkDispatchGraphCountInfoAMDX * pCountInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatchGraphAMDX( commandBuffer, scratch, pCountInfo );
    }

    void vkCmdDispatchGraphIndirectAMDX( VkCommandBuffer                      commandBuffer,
                                         VkDeviceAddress                      scratch,
                                         const VkDispatchGraphCountInfoAMDX * pCountInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatchGraphIndirectAMDX( commandBuffer, scratch, pCountInfo );
    }

    void vkCmdDispatchGraphIndirectCountAMDX( VkCommandBuffer commandBuffer, VkDeviceAddress scratch, VkDeviceAddress countInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDispatchGraphIndirectCountAMDX( commandBuffer, scratch, countInfo );
    }
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

    //=== VK_EXT_sample_locations ===

    void vkCmdSetSampleLocationsEXT( VkCommandBuffer commandBuffer, const VkSampleLocationsInfoEXT * pSampleLocationsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetSampleLocationsEXT( commandBuffer, pSampleLocationsInfo );
    }

    void vkGetPhysicalDeviceMultisamplePropertiesEXT( VkPhysicalDevice             physicalDevice,
                                                      VkSampleCountFlagBits        samples,
                                                      VkMultisamplePropertiesEXT * pMultisampleProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceMultisamplePropertiesEXT( physicalDevice, samples, pMultisampleProperties );
    }

    //=== VK_KHR_get_memory_requirements2 ===

    void vkGetImageMemoryRequirements2KHR( VkDevice                               device,
                                           const VkImageMemoryRequirementsInfo2 * pInfo,
                                           VkMemoryRequirements2 *                pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageMemoryRequirements2KHR( device, pInfo, pMemoryRequirements );
    }

    void vkGetBufferMemoryRequirements2KHR( VkDevice                                device,
                                            const VkBufferMemoryRequirementsInfo2 * pInfo,
                                            VkMemoryRequirements2 *                 pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferMemoryRequirements2KHR( device, pInfo, pMemoryRequirements );
    }

    void vkGetImageSparseMemoryRequirements2KHR( VkDevice                                     device,
                                                 const VkImageSparseMemoryRequirementsInfo2 * pInfo,
                                                 uint32_t *                                   pSparseMemoryRequirementCount,
                                                 VkSparseImageMemoryRequirements2 *           pSparseMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSparseMemoryRequirements2KHR( device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements );
    }

    //=== VK_KHR_acceleration_structure ===

    VkResult vkCreateAccelerationStructureKHR( VkDevice                                     device,
                                               const VkAccelerationStructureCreateInfoKHR * pCreateInfo,
                                               const VkAllocationCallbacks *                pAllocator,
                                               VkAccelerationStructureKHR *                 pAccelerationStructure ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateAccelerationStructureKHR( device, pCreateInfo, pAllocator, pAccelerationStructure );
    }

    void vkDestroyAccelerationStructureKHR( VkDevice                      device,
                                            VkAccelerationStructureKHR    accelerationStructure,
                                            const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyAccelerationStructureKHR( device, accelerationStructure, pAllocator );
    }

    void vkCmdBuildAccelerationStructuresKHR( VkCommandBuffer                                          commandBuffer,
                                              uint32_t                                                 infoCount,
                                              const VkAccelerationStructureBuildGeometryInfoKHR *      pInfos,
                                              const VkAccelerationStructureBuildRangeInfoKHR * const * ppBuildRangeInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBuildAccelerationStructuresKHR( commandBuffer, infoCount, pInfos, ppBuildRangeInfos );
    }

    void vkCmdBuildAccelerationStructuresIndirectKHR( VkCommandBuffer                                     commandBuffer,
                                                      uint32_t                                            infoCount,
                                                      const VkAccelerationStructureBuildGeometryInfoKHR * pInfos,
                                                      const VkDeviceAddress *                             pIndirectDeviceAddresses,
                                                      const uint32_t *                                    pIndirectStrides,
                                                      const uint32_t * const *                            ppMaxPrimitiveCounts ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBuildAccelerationStructuresIndirectKHR(
        commandBuffer, infoCount, pInfos, pIndirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts );
    }

    VkResult vkBuildAccelerationStructuresKHR( VkDevice                                                 device,
                                               VkDeferredOperationKHR                                   deferredOperation,
                                               uint32_t                                                 infoCount,
                                               const VkAccelerationStructureBuildGeometryInfoKHR *      pInfos,
                                               const VkAccelerationStructureBuildRangeInfoKHR * const * ppBuildRangeInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBuildAccelerationStructuresKHR( device, deferredOperation, infoCount, pInfos, ppBuildRangeInfos );
    }

    VkResult vkCopyAccelerationStructureKHR( VkDevice                                   device,
                                             VkDeferredOperationKHR                     deferredOperation,
                                             const VkCopyAccelerationStructureInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyAccelerationStructureKHR( device, deferredOperation, pInfo );
    }

    VkResult vkCopyAccelerationStructureToMemoryKHR( VkDevice                                           device,
                                                     VkDeferredOperationKHR                             deferredOperation,
                                                     const VkCopyAccelerationStructureToMemoryInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyAccelerationStructureToMemoryKHR( device, deferredOperation, pInfo );
    }

    VkResult vkCopyMemoryToAccelerationStructureKHR( VkDevice                                           device,
                                                     VkDeferredOperationKHR                             deferredOperation,
                                                     const VkCopyMemoryToAccelerationStructureInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyMemoryToAccelerationStructureKHR( device, deferredOperation, pInfo );
    }

    VkResult vkWriteAccelerationStructuresPropertiesKHR( VkDevice                           device,
                                                         uint32_t                           accelerationStructureCount,
                                                         const VkAccelerationStructureKHR * pAccelerationStructures,
                                                         VkQueryType                        queryType,
                                                         size_t                             dataSize,
                                                         void *                             pData,
                                                         size_t                             stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWriteAccelerationStructuresPropertiesKHR( device, accelerationStructureCount, pAccelerationStructures, queryType, dataSize, pData, stride );
    }

    void vkCmdCopyAccelerationStructureKHR( VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyAccelerationStructureKHR( commandBuffer, pInfo );
    }

    void vkCmdCopyAccelerationStructureToMemoryKHR( VkCommandBuffer                                    commandBuffer,
                                                    const VkCopyAccelerationStructureToMemoryInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyAccelerationStructureToMemoryKHR( commandBuffer, pInfo );
    }

    void vkCmdCopyMemoryToAccelerationStructureKHR( VkCommandBuffer                                    commandBuffer,
                                                    const VkCopyMemoryToAccelerationStructureInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyMemoryToAccelerationStructureKHR( commandBuffer, pInfo );
    }

    VkDeviceAddress vkGetAccelerationStructureDeviceAddressKHR( VkDevice                                            device,
                                                                const VkAccelerationStructureDeviceAddressInfoKHR * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAccelerationStructureDeviceAddressKHR( device, pInfo );
    }

    void vkCmdWriteAccelerationStructuresPropertiesKHR( VkCommandBuffer                    commandBuffer,
                                                        uint32_t                           accelerationStructureCount,
                                                        const VkAccelerationStructureKHR * pAccelerationStructures,
                                                        VkQueryType                        queryType,
                                                        VkQueryPool                        queryPool,
                                                        uint32_t                           firstQuery ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteAccelerationStructuresPropertiesKHR(
        commandBuffer, accelerationStructureCount, pAccelerationStructures, queryType, queryPool, firstQuery );
    }

    void vkGetDeviceAccelerationStructureCompatibilityKHR( VkDevice                                      device,
                                                           const VkAccelerationStructureVersionInfoKHR * pVersionInfo,
                                                           VkAccelerationStructureCompatibilityKHR *     pCompatibility ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceAccelerationStructureCompatibilityKHR( device, pVersionInfo, pCompatibility );
    }

    void vkGetAccelerationStructureBuildSizesKHR( VkDevice                                            device,
                                                  VkAccelerationStructureBuildTypeKHR                 buildType,
                                                  const VkAccelerationStructureBuildGeometryInfoKHR * pBuildInfo,
                                                  const uint32_t *                                    pMaxPrimitiveCounts,
                                                  VkAccelerationStructureBuildSizesInfoKHR *          pSizeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAccelerationStructureBuildSizesKHR( device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo );
    }

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
      return ::vkCmdTraceRaysKHR(
        commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable, width, height, depth );
    }

    VkResult vkCreateRayTracingPipelinesKHR( VkDevice                                  device,
                                             VkDeferredOperationKHR                    deferredOperation,
                                             VkPipelineCache                           pipelineCache,
                                             uint32_t                                  createInfoCount,
                                             const VkRayTracingPipelineCreateInfoKHR * pCreateInfos,
                                             const VkAllocationCallbacks *             pAllocator,
                                             VkPipeline *                              pPipelines ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateRayTracingPipelinesKHR( device, deferredOperation, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines );
    }

    VkResult vkGetRayTracingShaderGroupHandlesKHR(
      VkDevice device, VkPipeline pipeline, uint32_t firstGroup, uint32_t groupCount, size_t dataSize, void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRayTracingShaderGroupHandlesKHR( device, pipeline, firstGroup, groupCount, dataSize, pData );
    }

    VkResult vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(
      VkDevice device, VkPipeline pipeline, uint32_t firstGroup, uint32_t groupCount, size_t dataSize, void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRayTracingCaptureReplayShaderGroupHandlesKHR( device, pipeline, firstGroup, groupCount, dataSize, pData );
    }

    void vkCmdTraceRaysIndirectKHR( VkCommandBuffer                         commandBuffer,
                                    const VkStridedDeviceAddressRegionKHR * pRaygenShaderBindingTable,
                                    const VkStridedDeviceAddressRegionKHR * pMissShaderBindingTable,
                                    const VkStridedDeviceAddressRegionKHR * pHitShaderBindingTable,
                                    const VkStridedDeviceAddressRegionKHR * pCallableShaderBindingTable,
                                    VkDeviceAddress                         indirectDeviceAddress ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdTraceRaysIndirectKHR(
        commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable, indirectDeviceAddress );
    }

    VkDeviceSize vkGetRayTracingShaderGroupStackSizeKHR( VkDevice               device,
                                                         VkPipeline             pipeline,
                                                         uint32_t               group,
                                                         VkShaderGroupShaderKHR groupShader ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRayTracingShaderGroupStackSizeKHR( device, pipeline, group, groupShader );
    }

    void vkCmdSetRayTracingPipelineStackSizeKHR( VkCommandBuffer commandBuffer, uint32_t pipelineStackSize ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetRayTracingPipelineStackSizeKHR( commandBuffer, pipelineStackSize );
    }

    //=== VK_KHR_sampler_ycbcr_conversion ===

    VkResult vkCreateSamplerYcbcrConversionKHR( VkDevice                                   device,
                                                const VkSamplerYcbcrConversionCreateInfo * pCreateInfo,
                                                const VkAllocationCallbacks *              pAllocator,
                                                VkSamplerYcbcrConversion *                 pYcbcrConversion ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkBindBufferMemory2KHR( VkDevice device, uint32_t bindInfoCount, const VkBindBufferMemoryInfo * pBindInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindBufferMemory2KHR( device, bindInfoCount, pBindInfos );
    }

    VkResult vkBindImageMemory2KHR( VkDevice device, uint32_t bindInfoCount, const VkBindImageMemoryInfo * pBindInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindImageMemory2KHR( device, bindInfoCount, pBindInfos );
    }

    //=== VK_EXT_image_drm_format_modifier ===

    VkResult
      vkGetImageDrmFormatModifierPropertiesEXT( VkDevice device, VkImage image, VkImageDrmFormatModifierPropertiesEXT * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageDrmFormatModifierPropertiesEXT( device, image, pProperties );
    }

    //=== VK_EXT_validation_cache ===

    VkResult vkCreateValidationCacheEXT( VkDevice                               device,
                                         const VkValidationCacheCreateInfoEXT * pCreateInfo,
                                         const VkAllocationCallbacks *          pAllocator,
                                         VkValidationCacheEXT *                 pValidationCache ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateValidationCacheEXT( device, pCreateInfo, pAllocator, pValidationCache );
    }

    void
      vkDestroyValidationCacheEXT( VkDevice device, VkValidationCacheEXT validationCache, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkGetValidationCacheDataEXT( VkDevice device, VkValidationCacheEXT validationCache, size_t * pDataSize, void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetValidationCacheDataEXT( device, validationCache, pDataSize, pData );
    }

    //=== VK_NV_shading_rate_image ===

    void vkCmdBindShadingRateImageNV( VkCommandBuffer commandBuffer, VkImageView imageView, VkImageLayout imageLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindShadingRateImageNV( commandBuffer, imageView, imageLayout );
    }

    void vkCmdSetViewportShadingRatePaletteNV( VkCommandBuffer                commandBuffer,
                                               uint32_t                       firstViewport,
                                               uint32_t                       viewportCount,
                                               const VkShadingRatePaletteNV * pShadingRatePalettes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewportShadingRatePaletteNV( commandBuffer, firstViewport, viewportCount, pShadingRatePalettes );
    }

    void vkCmdSetCoarseSampleOrderNV( VkCommandBuffer                     commandBuffer,
                                      VkCoarseSampleOrderTypeNV           sampleOrderType,
                                      uint32_t                            customSampleOrderCount,
                                      const VkCoarseSampleOrderCustomNV * pCustomSampleOrders ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCoarseSampleOrderNV( commandBuffer, sampleOrderType, customSampleOrderCount, pCustomSampleOrders );
    }

    //=== VK_NV_ray_tracing ===

    VkResult vkCreateAccelerationStructureNV( VkDevice                                    device,
                                              const VkAccelerationStructureCreateInfoNV * pCreateInfo,
                                              const VkAllocationCallbacks *               pAllocator,
                                              VkAccelerationStructureNV *                 pAccelerationStructure ) const VULKAN_HPP_NOEXCEPT
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
                                                         VkMemoryRequirements2KHR *                              pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAccelerationStructureMemoryRequirementsNV( device, pInfo, pMemoryRequirements );
    }

    VkResult vkBindAccelerationStructureMemoryNV( VkDevice                                        device,
                                                  uint32_t                                        bindInfoCount,
                                                  const VkBindAccelerationStructureMemoryInfoNV * pBindInfos ) const VULKAN_HPP_NOEXCEPT
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
                                            VkDeviceSize                          scratchOffset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBuildAccelerationStructureNV( commandBuffer, pInfo, instanceData, instanceOffset, update, dst, src, scratch, scratchOffset );
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
                                            VkPipeline *                             pPipelines ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateRayTracingPipelinesNV( device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines );
    }

    VkResult vkGetRayTracingShaderGroupHandlesNV(
      VkDevice device, VkPipeline pipeline, uint32_t firstGroup, uint32_t groupCount, size_t dataSize, void * pData ) const VULKAN_HPP_NOEXCEPT
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
                                                       uint32_t                          firstQuery ) const VULKAN_HPP_NOEXCEPT
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
                                             VkDescriptorSetLayoutSupport *          pSupport ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdDrawIndirectCountKHR( commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    void vkCmdDrawIndexedIndirectCountKHR( VkCommandBuffer commandBuffer,
                                           VkBuffer        buffer,
                                           VkDeviceSize    offset,
                                           VkBuffer        countBuffer,
                                           VkDeviceSize    countBufferOffset,
                                           uint32_t        maxDrawCount,
                                           uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawIndexedIndirectCountKHR( commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    //=== VK_EXT_external_memory_host ===

    VkResult vkGetMemoryHostPointerPropertiesEXT( VkDevice                           device,
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
                                                             VkTimeDomainKHR * pTimeDomains ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceCalibrateableTimeDomainsEXT( physicalDevice, pTimeDomainCount, pTimeDomains );
    }

    VkResult vkGetCalibratedTimestampsEXT( VkDevice                             device,
                                           uint32_t                             timestampCount,
                                           const VkCalibratedTimestampInfoKHR * pTimestampInfos,
                                           uint64_t *                           pTimestamps,
                                           uint64_t *                           pMaxDeviation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetCalibratedTimestampsEXT( device, timestampCount, pTimestampInfos, pTimestamps, pMaxDeviation );
    }

    //=== VK_NV_mesh_shader ===

    void vkCmdDrawMeshTasksNV( VkCommandBuffer commandBuffer, uint32_t taskCount, uint32_t firstTask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMeshTasksNV( commandBuffer, taskCount, firstTask );
    }

    void vkCmdDrawMeshTasksIndirectNV( VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride ) const
      VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdDrawMeshTasksIndirectCountNV( commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    //=== VK_NV_scissor_exclusive ===

    void vkCmdSetExclusiveScissorEnableNV( VkCommandBuffer  commandBuffer,
                                           uint32_t         firstExclusiveScissor,
                                           uint32_t         exclusiveScissorCount,
                                           const VkBool32 * pExclusiveScissorEnables ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetExclusiveScissorEnableNV( commandBuffer, firstExclusiveScissor, exclusiveScissorCount, pExclusiveScissorEnables );
    }

    void vkCmdSetExclusiveScissorNV( VkCommandBuffer  commandBuffer,
                                     uint32_t         firstExclusiveScissor,
                                     uint32_t         exclusiveScissorCount,
                                     const VkRect2D * pExclusiveScissors ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetExclusiveScissorNV( commandBuffer, firstExclusiveScissor, exclusiveScissorCount, pExclusiveScissors );
    }

    //=== VK_NV_device_diagnostic_checkpoints ===

    void vkCmdSetCheckpointNV( VkCommandBuffer commandBuffer, const void * pCheckpointMarker ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCheckpointNV( commandBuffer, pCheckpointMarker );
    }

    void vkGetQueueCheckpointDataNV( VkQueue queue, uint32_t * pCheckpointDataCount, VkCheckpointDataNV * pCheckpointData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetQueueCheckpointDataNV( queue, pCheckpointDataCount, pCheckpointData );
    }

    //=== VK_KHR_timeline_semaphore ===

    VkResult vkGetSemaphoreCounterValueKHR( VkDevice device, VkSemaphore semaphore, uint64_t * pValue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreCounterValueKHR( device, semaphore, pValue );
    }

    VkResult vkWaitSemaphoresKHR( VkDevice device, const VkSemaphoreWaitInfo * pWaitInfo, uint64_t timeout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWaitSemaphoresKHR( device, pWaitInfo, timeout );
    }

    VkResult vkSignalSemaphoreKHR( VkDevice device, const VkSemaphoreSignalInfo * pSignalInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSignalSemaphoreKHR( device, pSignalInfo );
    }

    //=== VK_INTEL_performance_query ===

    VkResult vkInitializePerformanceApiINTEL( VkDevice device, const VkInitializePerformanceApiInfoINTEL * pInitializeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkInitializePerformanceApiINTEL( device, pInitializeInfo );
    }

    void vkUninitializePerformanceApiINTEL( VkDevice device ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUninitializePerformanceApiINTEL( device );
    }

    VkResult vkCmdSetPerformanceMarkerINTEL( VkCommandBuffer commandBuffer, const VkPerformanceMarkerInfoINTEL * pMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPerformanceMarkerINTEL( commandBuffer, pMarkerInfo );
    }

    VkResult vkCmdSetPerformanceStreamMarkerINTEL( VkCommandBuffer                            commandBuffer,
                                                   const VkPerformanceStreamMarkerInfoINTEL * pMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPerformanceStreamMarkerINTEL( commandBuffer, pMarkerInfo );
    }

    VkResult vkCmdSetPerformanceOverrideINTEL( VkCommandBuffer commandBuffer, const VkPerformanceOverrideInfoINTEL * pOverrideInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPerformanceOverrideINTEL( commandBuffer, pOverrideInfo );
    }

    VkResult vkAcquirePerformanceConfigurationINTEL( VkDevice                                           device,
                                                     const VkPerformanceConfigurationAcquireInfoINTEL * pAcquireInfo,
                                                     VkPerformanceConfigurationINTEL *                  pConfiguration ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquirePerformanceConfigurationINTEL( device, pAcquireInfo, pConfiguration );
    }

    VkResult vkReleasePerformanceConfigurationINTEL( VkDevice device, VkPerformanceConfigurationINTEL configuration ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkReleasePerformanceConfigurationINTEL( device, configuration );
    }

    VkResult vkQueueSetPerformanceConfigurationINTEL( VkQueue queue, VkPerformanceConfigurationINTEL configuration ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueSetPerformanceConfigurationINTEL( queue, configuration );
    }

    VkResult
      vkGetPerformanceParameterINTEL( VkDevice device, VkPerformanceParameterTypeINTEL parameter, VkPerformanceValueINTEL * pValue ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPerformanceParameterINTEL( device, parameter, pValue );
    }

    //=== VK_AMD_display_native_hdr ===

    void vkSetLocalDimmingAMD( VkDevice device, VkSwapchainKHR swapChain, VkBool32 localDimmingEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetLocalDimmingAMD( device, swapChain, localDimmingEnable );
    }

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_imagepipe_surface ===

    VkResult vkCreateImagePipeSurfaceFUCHSIA( VkInstance                                  instance,
                                              const VkImagePipeSurfaceCreateInfoFUCHSIA * pCreateInfo,
                                              const VkAllocationCallbacks *               pAllocator,
                                              VkSurfaceKHR *                              pSurface ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkGetPhysicalDeviceFragmentShadingRatesKHR( VkPhysicalDevice                         physicalDevice,
                                                         uint32_t *                               pFragmentShadingRateCount,
                                                         VkPhysicalDeviceFragmentShadingRateKHR * pFragmentShadingRates ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceFragmentShadingRatesKHR( physicalDevice, pFragmentShadingRateCount, pFragmentShadingRates );
    }

    void vkCmdSetFragmentShadingRateKHR( VkCommandBuffer                          commandBuffer,
                                         const VkExtent2D *                       pFragmentSize,
                                         const VkFragmentShadingRateCombinerOpKHR combinerOps[2] ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetFragmentShadingRateKHR( commandBuffer, pFragmentSize, combinerOps );
    }

    //=== VK_EXT_buffer_device_address ===

    VkDeviceAddress vkGetBufferDeviceAddressEXT( VkDevice device, const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferDeviceAddressEXT( device, pInfo );
    }

    //=== VK_EXT_tooling_info ===

    VkResult vkGetPhysicalDeviceToolPropertiesEXT( VkPhysicalDevice                 physicalDevice,
                                                   uint32_t *                       pToolCount,
                                                   VkPhysicalDeviceToolProperties * pToolProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceToolPropertiesEXT( physicalDevice, pToolCount, pToolProperties );
    }

    //=== VK_KHR_present_wait ===

    VkResult vkWaitForPresentKHR( VkDevice device, VkSwapchainKHR swapchain, uint64_t presentId, uint64_t timeout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWaitForPresentKHR( device, swapchain, presentId, timeout );
    }

    //=== VK_NV_cooperative_matrix ===

    VkResult vkGetPhysicalDeviceCooperativeMatrixPropertiesNV( VkPhysicalDevice                  physicalDevice,
                                                               uint32_t *                        pPropertyCount,
                                                               VkCooperativeMatrixPropertiesNV * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceCooperativeMatrixPropertiesNV( physicalDevice, pPropertyCount, pProperties );
    }

    //=== VK_NV_coverage_reduction_mode ===

    VkResult vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
      VkPhysicalDevice physicalDevice, uint32_t * pCombinationCount, VkFramebufferMixedSamplesCombinationNV * pCombinations ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV( physicalDevice, pCombinationCount, pCombinations );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_EXT_full_screen_exclusive ===

    VkResult vkGetPhysicalDeviceSurfacePresentModes2EXT( VkPhysicalDevice                        physicalDevice,
                                                         const VkPhysicalDeviceSurfaceInfo2KHR * pSurfaceInfo,
                                                         uint32_t *                              pPresentModeCount,
                                                         VkPresentModeKHR *                      pPresentModes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceSurfacePresentModes2EXT( physicalDevice, pSurfaceInfo, pPresentModeCount, pPresentModes );
    }

    VkResult vkAcquireFullScreenExclusiveModeEXT( VkDevice device, VkSwapchainKHR swapchain ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireFullScreenExclusiveModeEXT( device, swapchain );
    }

    VkResult vkReleaseFullScreenExclusiveModeEXT( VkDevice device, VkSwapchainKHR swapchain ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkReleaseFullScreenExclusiveModeEXT( device, swapchain );
    }

    VkResult vkGetDeviceGroupSurfacePresentModes2EXT( VkDevice                                device,
                                                      const VkPhysicalDeviceSurfaceInfo2KHR * pSurfaceInfo,
                                                      VkDeviceGroupPresentModeFlagsKHR *      pModes ) const VULKAN_HPP_NOEXCEPT
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

    VkDeviceAddress vkGetBufferDeviceAddressKHR( VkDevice device, const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferDeviceAddressKHR( device, pInfo );
    }

    uint64_t vkGetBufferOpaqueCaptureAddressKHR( VkDevice device, const VkBufferDeviceAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferOpaqueCaptureAddressKHR( device, pInfo );
    }

    uint64_t vkGetDeviceMemoryOpaqueCaptureAddressKHR( VkDevice device, const VkDeviceMemoryOpaqueCaptureAddressInfo * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceMemoryOpaqueCaptureAddressKHR( device, pInfo );
    }

    //=== VK_EXT_line_rasterization ===

    void vkCmdSetLineStippleEXT( VkCommandBuffer commandBuffer, uint32_t lineStippleFactor, uint16_t lineStipplePattern ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetLineStippleEXT( commandBuffer, lineStippleFactor, lineStipplePattern );
    }

    //=== VK_EXT_host_query_reset ===

    void vkResetQueryPoolEXT( VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount ) const VULKAN_HPP_NOEXCEPT
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

    void vkCmdSetPrimitiveTopologyEXT( VkCommandBuffer commandBuffer, VkPrimitiveTopology primitiveTopology ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPrimitiveTopologyEXT( commandBuffer, primitiveTopology );
    }

    void vkCmdSetViewportWithCountEXT( VkCommandBuffer commandBuffer, uint32_t viewportCount, const VkViewport * pViewports ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewportWithCountEXT( commandBuffer, viewportCount, pViewports );
    }

    void vkCmdSetScissorWithCountEXT( VkCommandBuffer commandBuffer, uint32_t scissorCount, const VkRect2D * pScissors ) const VULKAN_HPP_NOEXCEPT
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
      return ::vkCmdBindVertexBuffers2EXT( commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes, pStrides );
    }

    void vkCmdSetDepthTestEnableEXT( VkCommandBuffer commandBuffer, VkBool32 depthTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthTestEnableEXT( commandBuffer, depthTestEnable );
    }

    void vkCmdSetDepthWriteEnableEXT( VkCommandBuffer commandBuffer, VkBool32 depthWriteEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthWriteEnableEXT( commandBuffer, depthWriteEnable );
    }

    void vkCmdSetDepthCompareOpEXT( VkCommandBuffer commandBuffer, VkCompareOp depthCompareOp ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthCompareOpEXT( commandBuffer, depthCompareOp );
    }

    void vkCmdSetDepthBoundsTestEnableEXT( VkCommandBuffer commandBuffer, VkBool32 depthBoundsTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBoundsTestEnableEXT( commandBuffer, depthBoundsTestEnable );
    }

    void vkCmdSetStencilTestEnableEXT( VkCommandBuffer commandBuffer, VkBool32 stencilTestEnable ) const VULKAN_HPP_NOEXCEPT
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

    void vkDestroyDeferredOperationKHR( VkDevice device, VkDeferredOperationKHR operation, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyDeferredOperationKHR( device, operation, pAllocator );
    }

    uint32_t vkGetDeferredOperationMaxConcurrencyKHR( VkDevice device, VkDeferredOperationKHR operation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeferredOperationMaxConcurrencyKHR( device, operation );
    }

    VkResult vkGetDeferredOperationResultKHR( VkDevice device, VkDeferredOperationKHR operation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeferredOperationResultKHR( device, operation );
    }

    VkResult vkDeferredOperationJoinKHR( VkDevice device, VkDeferredOperationKHR operation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDeferredOperationJoinKHR( device, operation );
    }

    //=== VK_KHR_pipeline_executable_properties ===

    VkResult vkGetPipelineExecutablePropertiesKHR( VkDevice                            device,
                                                   const VkPipelineInfoKHR *           pPipelineInfo,
                                                   uint32_t *                          pExecutableCount,
                                                   VkPipelineExecutablePropertiesKHR * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineExecutablePropertiesKHR( device, pPipelineInfo, pExecutableCount, pProperties );
    }

    VkResult vkGetPipelineExecutableStatisticsKHR( VkDevice                            device,
                                                   const VkPipelineExecutableInfoKHR * pExecutableInfo,
                                                   uint32_t *                          pStatisticCount,
                                                   VkPipelineExecutableStatisticKHR *  pStatistics ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineExecutableStatisticsKHR( device, pExecutableInfo, pStatisticCount, pStatistics );
    }

    VkResult
      vkGetPipelineExecutableInternalRepresentationsKHR( VkDevice                                        device,
                                                         const VkPipelineExecutableInfoKHR *             pExecutableInfo,
                                                         uint32_t *                                      pInternalRepresentationCount,
                                                         VkPipelineExecutableInternalRepresentationKHR * pInternalRepresentations ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineExecutableInternalRepresentationsKHR( device, pExecutableInfo, pInternalRepresentationCount, pInternalRepresentations );
    }

    //=== VK_EXT_host_image_copy ===

    VkResult vkCopyMemoryToImageEXT( VkDevice device, const VkCopyMemoryToImageInfoEXT * pCopyMemoryToImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyMemoryToImageEXT( device, pCopyMemoryToImageInfo );
    }

    VkResult vkCopyImageToMemoryEXT( VkDevice device, const VkCopyImageToMemoryInfoEXT * pCopyImageToMemoryInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyImageToMemoryEXT( device, pCopyImageToMemoryInfo );
    }

    VkResult vkCopyImageToImageEXT( VkDevice device, const VkCopyImageToImageInfoEXT * pCopyImageToImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyImageToImageEXT( device, pCopyImageToImageInfo );
    }

    VkResult
      vkTransitionImageLayoutEXT( VkDevice device, uint32_t transitionCount, const VkHostImageLayoutTransitionInfoEXT * pTransitions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkTransitionImageLayoutEXT( device, transitionCount, pTransitions );
    }

    void vkGetImageSubresourceLayout2EXT( VkDevice                       device,
                                          VkImage                        image,
                                          const VkImageSubresource2KHR * pSubresource,
                                          VkSubresourceLayout2KHR *      pLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSubresourceLayout2EXT( device, image, pSubresource, pLayout );
    }

    //=== VK_KHR_map_memory2 ===

    VkResult vkMapMemory2KHR( VkDevice device, const VkMemoryMapInfoKHR * pMemoryMapInfo, void ** ppData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkMapMemory2KHR( device, pMemoryMapInfo, ppData );
    }

    VkResult vkUnmapMemory2KHR( VkDevice device, const VkMemoryUnmapInfoKHR * pMemoryUnmapInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkUnmapMemory2KHR( device, pMemoryUnmapInfo );
    }

    //=== VK_EXT_swapchain_maintenance1 ===

    VkResult vkReleaseSwapchainImagesEXT( VkDevice device, const VkReleaseSwapchainImagesInfoEXT * pReleaseInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkReleaseSwapchainImagesEXT( device, pReleaseInfo );
    }

    //=== VK_NV_device_generated_commands ===

    void vkGetGeneratedCommandsMemoryRequirementsNV( VkDevice                                            device,
                                                     const VkGeneratedCommandsMemoryRequirementsInfoNV * pInfo,
                                                     VkMemoryRequirements2 *                             pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetGeneratedCommandsMemoryRequirementsNV( device, pInfo, pMemoryRequirements );
    }

    void vkCmdPreprocessGeneratedCommandsNV( VkCommandBuffer commandBuffer, const VkGeneratedCommandsInfoNV * pGeneratedCommandsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPreprocessGeneratedCommandsNV( commandBuffer, pGeneratedCommandsInfo );
    }

    void vkCmdExecuteGeneratedCommandsNV( VkCommandBuffer                   commandBuffer,
                                          VkBool32                          isPreprocessed,
                                          const VkGeneratedCommandsInfoNV * pGeneratedCommandsInfo ) const VULKAN_HPP_NOEXCEPT
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

    VkResult vkCreateIndirectCommandsLayoutNV( VkDevice                                     device,
                                               const VkIndirectCommandsLayoutCreateInfoNV * pCreateInfo,
                                               const VkAllocationCallbacks *                pAllocator,
                                               VkIndirectCommandsLayoutNV *                 pIndirectCommandsLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateIndirectCommandsLayoutNV( device, pCreateInfo, pAllocator, pIndirectCommandsLayout );
    }

    void vkDestroyIndirectCommandsLayoutNV( VkDevice                      device,
                                            VkIndirectCommandsLayoutNV    indirectCommandsLayout,
                                            const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyIndirectCommandsLayoutNV( device, indirectCommandsLayout, pAllocator );
    }

    //=== VK_EXT_depth_bias_control ===

    void vkCmdSetDepthBias2EXT( VkCommandBuffer commandBuffer, const VkDepthBiasInfoEXT * pDepthBiasInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthBias2EXT( commandBuffer, pDepthBiasInfo );
    }

    //=== VK_EXT_acquire_drm_display ===

    VkResult vkAcquireDrmDisplayEXT( VkPhysicalDevice physicalDevice, int32_t drmFd, VkDisplayKHR display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireDrmDisplayEXT( physicalDevice, drmFd, display );
    }

    VkResult vkGetDrmDisplayEXT( VkPhysicalDevice physicalDevice, int32_t drmFd, uint32_t connectorId, VkDisplayKHR * display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDrmDisplayEXT( physicalDevice, drmFd, connectorId, display );
    }

    //=== VK_EXT_private_data ===

    VkResult vkCreatePrivateDataSlotEXT( VkDevice                            device,
                                         const VkPrivateDataSlotCreateInfo * pCreateInfo,
                                         const VkAllocationCallbacks *       pAllocator,
                                         VkPrivateDataSlot *                 pPrivateDataSlot ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreatePrivateDataSlotEXT( device, pCreateInfo, pAllocator, pPrivateDataSlot );
    }

    void vkDestroyPrivateDataSlotEXT( VkDevice device, VkPrivateDataSlot privateDataSlot, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyPrivateDataSlotEXT( device, privateDataSlot, pAllocator );
    }

    VkResult vkSetPrivateDataEXT( VkDevice device, VkObjectType objectType, uint64_t objectHandle, VkPrivateDataSlot privateDataSlot, uint64_t data ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetPrivateDataEXT( device, objectType, objectHandle, privateDataSlot, data );
    }

    void vkGetPrivateDataEXT( VkDevice device, VkObjectType objectType, uint64_t objectHandle, VkPrivateDataSlot privateDataSlot, uint64_t * pData ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPrivateDataEXT( device, objectType, objectHandle, privateDataSlot, pData );
    }

    //=== VK_KHR_video_encode_queue ===

    VkResult
      vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR( VkPhysicalDevice                                       physicalDevice,
                                                               const VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR * pQualityLevelInfo,
                                                               VkVideoEncodeQualityLevelPropertiesKHR * pQualityLevelProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR( physicalDevice, pQualityLevelInfo, pQualityLevelProperties );
    }

    VkResult vkGetEncodedVideoSessionParametersKHR( VkDevice                                         device,
                                                    const VkVideoEncodeSessionParametersGetInfoKHR * pVideoSessionParametersInfo,
                                                    VkVideoEncodeSessionParametersFeedbackInfoKHR *  pFeedbackInfo,
                                                    size_t *                                         pDataSize,
                                                    void *                                           pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetEncodedVideoSessionParametersKHR( device, pVideoSessionParametersInfo, pFeedbackInfo, pDataSize, pData );
    }

    void vkCmdEncodeVideoKHR( VkCommandBuffer commandBuffer, const VkVideoEncodeInfoKHR * pEncodeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdEncodeVideoKHR( commandBuffer, pEncodeInfo );
    }

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_NV_cuda_kernel_launch ===

    VkResult vkCreateCudaModuleNV( VkDevice                         device,
                                   const VkCudaModuleCreateInfoNV * pCreateInfo,
                                   const VkAllocationCallbacks *    pAllocator,
                                   VkCudaModuleNV *                 pModule ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateCudaModuleNV( device, pCreateInfo, pAllocator, pModule );
    }

    VkResult vkGetCudaModuleCacheNV( VkDevice device, VkCudaModuleNV module, size_t * pCacheSize, void * pCacheData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetCudaModuleCacheNV( device, module, pCacheSize, pCacheData );
    }

    VkResult vkCreateCudaFunctionNV( VkDevice                           device,
                                     const VkCudaFunctionCreateInfoNV * pCreateInfo,
                                     const VkAllocationCallbacks *      pAllocator,
                                     VkCudaFunctionNV *                 pFunction ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateCudaFunctionNV( device, pCreateInfo, pAllocator, pFunction );
    }

    void vkDestroyCudaModuleNV( VkDevice device, VkCudaModuleNV module, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyCudaModuleNV( device, module, pAllocator );
    }

    void vkDestroyCudaFunctionNV( VkDevice device, VkCudaFunctionNV function, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyCudaFunctionNV( device, function, pAllocator );
    }

    void vkCmdCudaLaunchKernelNV( VkCommandBuffer commandBuffer, const VkCudaLaunchInfoNV * pLaunchInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCudaLaunchKernelNV( commandBuffer, pLaunchInfo );
    }
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
    //=== VK_EXT_metal_objects ===

    void vkExportMetalObjectsEXT( VkDevice device, VkExportMetalObjectsInfoEXT * pMetalObjectsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkExportMetalObjectsEXT( device, pMetalObjectsInfo );
    }
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

    //=== VK_KHR_synchronization2 ===

    void vkCmdSetEvent2KHR( VkCommandBuffer commandBuffer, VkEvent event, const VkDependencyInfo * pDependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetEvent2KHR( commandBuffer, event, pDependencyInfo );
    }

    void vkCmdResetEvent2KHR( VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags2 stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResetEvent2KHR( commandBuffer, event, stageMask );
    }

    void vkCmdWaitEvents2KHR( VkCommandBuffer          commandBuffer,
                              uint32_t                 eventCount,
                              const VkEvent *          pEvents,
                              const VkDependencyInfo * pDependencyInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWaitEvents2KHR( commandBuffer, eventCount, pEvents, pDependencyInfos );
    }

    void vkCmdPipelineBarrier2KHR( VkCommandBuffer commandBuffer, const VkDependencyInfo * pDependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPipelineBarrier2KHR( commandBuffer, pDependencyInfo );
    }

    void vkCmdWriteTimestamp2KHR( VkCommandBuffer commandBuffer, VkPipelineStageFlags2 stage, VkQueryPool queryPool, uint32_t query ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteTimestamp2KHR( commandBuffer, stage, queryPool, query );
    }

    VkResult vkQueueSubmit2KHR( VkQueue queue, uint32_t submitCount, const VkSubmitInfo2 * pSubmits, VkFence fence ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueSubmit2KHR( queue, submitCount, pSubmits, fence );
    }

    void vkCmdWriteBufferMarker2AMD(
      VkCommandBuffer commandBuffer, VkPipelineStageFlags2 stage, VkBuffer dstBuffer, VkDeviceSize dstOffset, uint32_t marker ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteBufferMarker2AMD( commandBuffer, stage, dstBuffer, dstOffset, marker );
    }

    void vkGetQueueCheckpointData2NV( VkQueue queue, uint32_t * pCheckpointDataCount, VkCheckpointData2NV * pCheckpointData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetQueueCheckpointData2NV( queue, pCheckpointDataCount, pCheckpointData );
    }

    //=== VK_EXT_descriptor_buffer ===

    void vkGetDescriptorSetLayoutSizeEXT( VkDevice device, VkDescriptorSetLayout layout, VkDeviceSize * pLayoutSizeInBytes ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDescriptorSetLayoutSizeEXT( device, layout, pLayoutSizeInBytes );
    }

    void vkGetDescriptorSetLayoutBindingOffsetEXT( VkDevice              device,
                                                   VkDescriptorSetLayout layout,
                                                   uint32_t              binding,
                                                   VkDeviceSize *        pOffset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDescriptorSetLayoutBindingOffsetEXT( device, layout, binding, pOffset );
    }

    void vkGetDescriptorEXT( VkDevice device, const VkDescriptorGetInfoEXT * pDescriptorInfo, size_t dataSize, void * pDescriptor ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDescriptorEXT( device, pDescriptorInfo, dataSize, pDescriptor );
    }

    void vkCmdBindDescriptorBuffersEXT( VkCommandBuffer                          commandBuffer,
                                        uint32_t                                 bufferCount,
                                        const VkDescriptorBufferBindingInfoEXT * pBindingInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindDescriptorBuffersEXT( commandBuffer, bufferCount, pBindingInfos );
    }

    void vkCmdSetDescriptorBufferOffsetsEXT( VkCommandBuffer      commandBuffer,
                                             VkPipelineBindPoint  pipelineBindPoint,
                                             VkPipelineLayout     layout,
                                             uint32_t             firstSet,
                                             uint32_t             setCount,
                                             const uint32_t *     pBufferIndices,
                                             const VkDeviceSize * pOffsets ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDescriptorBufferOffsetsEXT( commandBuffer, pipelineBindPoint, layout, firstSet, setCount, pBufferIndices, pOffsets );
    }

    void vkCmdBindDescriptorBufferEmbeddedSamplersEXT( VkCommandBuffer     commandBuffer,
                                                       VkPipelineBindPoint pipelineBindPoint,
                                                       VkPipelineLayout    layout,
                                                       uint32_t            set ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindDescriptorBufferEmbeddedSamplersEXT( commandBuffer, pipelineBindPoint, layout, set );
    }

    VkResult
      vkGetBufferOpaqueCaptureDescriptorDataEXT( VkDevice device, const VkBufferCaptureDescriptorDataInfoEXT * pInfo, void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferOpaqueCaptureDescriptorDataEXT( device, pInfo, pData );
    }

    VkResult
      vkGetImageOpaqueCaptureDescriptorDataEXT( VkDevice device, const VkImageCaptureDescriptorDataInfoEXT * pInfo, void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageOpaqueCaptureDescriptorDataEXT( device, pInfo, pData );
    }

    VkResult vkGetImageViewOpaqueCaptureDescriptorDataEXT( VkDevice                                        device,
                                                           const VkImageViewCaptureDescriptorDataInfoEXT * pInfo,
                                                           void *                                          pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageViewOpaqueCaptureDescriptorDataEXT( device, pInfo, pData );
    }

    VkResult
      vkGetSamplerOpaqueCaptureDescriptorDataEXT( VkDevice device, const VkSamplerCaptureDescriptorDataInfoEXT * pInfo, void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSamplerOpaqueCaptureDescriptorDataEXT( device, pInfo, pData );
    }

    VkResult vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT( VkDevice                                                    device,
                                                                       const VkAccelerationStructureCaptureDescriptorDataInfoEXT * pInfo,
                                                                       void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT( device, pInfo, pData );
    }

    //=== VK_NV_fragment_shading_rate_enums ===

    void vkCmdSetFragmentShadingRateEnumNV( VkCommandBuffer                          commandBuffer,
                                            VkFragmentShadingRateNV                  shadingRate,
                                            const VkFragmentShadingRateCombinerOpKHR combinerOps[2] ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetFragmentShadingRateEnumNV( commandBuffer, shadingRate, combinerOps );
    }

    //=== VK_EXT_mesh_shader ===

    void vkCmdDrawMeshTasksEXT( VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMeshTasksEXT( commandBuffer, groupCountX, groupCountY, groupCountZ );
    }

    void vkCmdDrawMeshTasksIndirectEXT( VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMeshTasksIndirectEXT( commandBuffer, buffer, offset, drawCount, stride );
    }

    void vkCmdDrawMeshTasksIndirectCountEXT( VkCommandBuffer commandBuffer,
                                             VkBuffer        buffer,
                                             VkDeviceSize    offset,
                                             VkBuffer        countBuffer,
                                             VkDeviceSize    countBufferOffset,
                                             uint32_t        maxDrawCount,
                                             uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawMeshTasksIndirectCountEXT( commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride );
    }

    //=== VK_KHR_copy_commands2 ===

    void vkCmdCopyBuffer2KHR( VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 * pCopyBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyBuffer2KHR( commandBuffer, pCopyBufferInfo );
    }

    void vkCmdCopyImage2KHR( VkCommandBuffer commandBuffer, const VkCopyImageInfo2 * pCopyImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyImage2KHR( commandBuffer, pCopyImageInfo );
    }

    void vkCmdCopyBufferToImage2KHR( VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 * pCopyBufferToImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyBufferToImage2KHR( commandBuffer, pCopyBufferToImageInfo );
    }

    void vkCmdCopyImageToBuffer2KHR( VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2 * pCopyImageToBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyImageToBuffer2KHR( commandBuffer, pCopyImageToBufferInfo );
    }

    void vkCmdBlitImage2KHR( VkCommandBuffer commandBuffer, const VkBlitImageInfo2 * pBlitImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBlitImage2KHR( commandBuffer, pBlitImageInfo );
    }

    void vkCmdResolveImage2KHR( VkCommandBuffer commandBuffer, const VkResolveImageInfo2 * pResolveImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdResolveImage2KHR( commandBuffer, pResolveImageInfo );
    }

    //=== VK_EXT_device_fault ===

    VkResult vkGetDeviceFaultInfoEXT( VkDevice device, VkDeviceFaultCountsEXT * pFaultCounts, VkDeviceFaultInfoEXT * pFaultInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceFaultInfoEXT( device, pFaultCounts, pFaultInfo );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_NV_acquire_winrt_display ===

    VkResult vkAcquireWinrtDisplayNV( VkPhysicalDevice physicalDevice, VkDisplayKHR display ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkAcquireWinrtDisplayNV( physicalDevice, display );
    }

    VkResult vkGetWinrtDisplayNV( VkPhysicalDevice physicalDevice, uint32_t deviceRelativeId, VkDisplayKHR * pDisplay ) const VULKAN_HPP_NOEXCEPT
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

    VkBool32
      vkGetPhysicalDeviceDirectFBPresentationSupportEXT( VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, IDirectFB * dfb ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceDirectFBPresentationSupportEXT( physicalDevice, queueFamilyIndex, dfb );
    }
#  endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

    //=== VK_EXT_vertex_input_dynamic_state ===

    void vkCmdSetVertexInputEXT( VkCommandBuffer                               commandBuffer,
                                 uint32_t                                      vertexBindingDescriptionCount,
                                 const VkVertexInputBindingDescription2EXT *   pVertexBindingDescriptions,
                                 uint32_t                                      vertexAttributeDescriptionCount,
                                 const VkVertexInputAttributeDescription2EXT * pVertexAttributeDescriptions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetVertexInputEXT(
        commandBuffer, vertexBindingDescriptionCount, pVertexBindingDescriptions, vertexAttributeDescriptionCount, pVertexAttributeDescriptions );
    }

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_external_memory ===

    VkResult vkGetMemoryZirconHandleFUCHSIA( VkDevice                                   device,
                                             const VkMemoryGetZirconHandleInfoFUCHSIA * pGetZirconHandleInfo,
                                             zx_handle_t *                              pZirconHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryZirconHandleFUCHSIA( device, pGetZirconHandleInfo, pZirconHandle );
    }

    VkResult vkGetMemoryZirconHandlePropertiesFUCHSIA( VkDevice                                device,
                                                       VkExternalMemoryHandleTypeFlagBits      handleType,
                                                       zx_handle_t                             zirconHandle,
                                                       VkMemoryZirconHandlePropertiesFUCHSIA * pMemoryZirconHandleProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryZirconHandlePropertiesFUCHSIA( device, handleType, zirconHandle, pMemoryZirconHandleProperties );
    }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_external_semaphore ===

    VkResult vkImportSemaphoreZirconHandleFUCHSIA( VkDevice                                         device,
                                                   const VkImportSemaphoreZirconHandleInfoFUCHSIA * pImportSemaphoreZirconHandleInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkImportSemaphoreZirconHandleFUCHSIA( device, pImportSemaphoreZirconHandleInfo );
    }

    VkResult vkGetSemaphoreZirconHandleFUCHSIA( VkDevice                                      device,
                                                const VkSemaphoreGetZirconHandleInfoFUCHSIA * pGetZirconHandleInfo,
                                                zx_handle_t *                                 pZirconHandle ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetSemaphoreZirconHandleFUCHSIA( device, pGetZirconHandleInfo, pZirconHandle );
    }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_buffer_collection ===

    VkResult vkCreateBufferCollectionFUCHSIA( VkDevice                                    device,
                                              const VkBufferCollectionCreateInfoFUCHSIA * pCreateInfo,
                                              const VkAllocationCallbacks *               pAllocator,
                                              VkBufferCollectionFUCHSIA *                 pCollection ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateBufferCollectionFUCHSIA( device, pCreateInfo, pAllocator, pCollection );
    }

    VkResult vkSetBufferCollectionImageConstraintsFUCHSIA( VkDevice                              device,
                                                           VkBufferCollectionFUCHSIA             collection,
                                                           const VkImageConstraintsInfoFUCHSIA * pImageConstraintsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetBufferCollectionImageConstraintsFUCHSIA( device, collection, pImageConstraintsInfo );
    }

    VkResult vkSetBufferCollectionBufferConstraintsFUCHSIA( VkDevice                               device,
                                                            VkBufferCollectionFUCHSIA              collection,
                                                            const VkBufferConstraintsInfoFUCHSIA * pBufferConstraintsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetBufferCollectionBufferConstraintsFUCHSIA( device, collection, pBufferConstraintsInfo );
    }

    void vkDestroyBufferCollectionFUCHSIA( VkDevice                      device,
                                           VkBufferCollectionFUCHSIA     collection,
                                           const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyBufferCollectionFUCHSIA( device, collection, pAllocator );
    }

    VkResult vkGetBufferCollectionPropertiesFUCHSIA( VkDevice                              device,
                                                     VkBufferCollectionFUCHSIA             collection,
                                                     VkBufferCollectionPropertiesFUCHSIA * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetBufferCollectionPropertiesFUCHSIA( device, collection, pProperties );
    }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

    //=== VK_HUAWEI_subpass_shading ===

    VkResult
      vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI( VkDevice device, VkRenderPass renderpass, VkExtent2D * pMaxWorkgroupSize ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI( device, renderpass, pMaxWorkgroupSize );
    }

    void vkCmdSubpassShadingHUAWEI( VkCommandBuffer commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSubpassShadingHUAWEI( commandBuffer );
    }

    //=== VK_HUAWEI_invocation_mask ===

    void vkCmdBindInvocationMaskHUAWEI( VkCommandBuffer commandBuffer, VkImageView imageView, VkImageLayout imageLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindInvocationMaskHUAWEI( commandBuffer, imageView, imageLayout );
    }

    //=== VK_NV_external_memory_rdma ===

    VkResult vkGetMemoryRemoteAddressNV( VkDevice                               device,
                                         const VkMemoryGetRemoteAddressInfoNV * pMemoryGetRemoteAddressInfo,
                                         VkRemoteAddressNV *                    pAddress ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMemoryRemoteAddressNV( device, pMemoryGetRemoteAddressInfo, pAddress );
    }

    //=== VK_EXT_pipeline_properties ===

    VkResult
      vkGetPipelinePropertiesEXT( VkDevice device, const VkPipelineInfoEXT * pPipelineInfo, VkBaseOutStructure * pPipelineProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelinePropertiesEXT( device, pPipelineInfo, pPipelineProperties );
    }

    //=== VK_EXT_extended_dynamic_state2 ===

    void vkCmdSetPatchControlPointsEXT( VkCommandBuffer commandBuffer, uint32_t patchControlPoints ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPatchControlPointsEXT( commandBuffer, patchControlPoints );
    }

    void vkCmdSetRasterizerDiscardEnableEXT( VkCommandBuffer commandBuffer, VkBool32 rasterizerDiscardEnable ) const VULKAN_HPP_NOEXCEPT
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

    void vkCmdSetPrimitiveRestartEnableEXT( VkCommandBuffer commandBuffer, VkBool32 primitiveRestartEnable ) const VULKAN_HPP_NOEXCEPT
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

    void vkCmdSetColorWriteEnableEXT( VkCommandBuffer commandBuffer, uint32_t attachmentCount, const VkBool32 * pColorWriteEnables ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetColorWriteEnableEXT( commandBuffer, attachmentCount, pColorWriteEnables );
    }

    //=== VK_KHR_ray_tracing_maintenance1 ===

    void vkCmdTraceRaysIndirect2KHR( VkCommandBuffer commandBuffer, VkDeviceAddress indirectDeviceAddress ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdTraceRaysIndirect2KHR( commandBuffer, indirectDeviceAddress );
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
      return ::vkCmdDrawMultiIndexedEXT( commandBuffer, drawCount, pIndexInfo, instanceCount, firstInstance, stride, pVertexOffset );
    }

    //=== VK_EXT_opacity_micromap ===

    VkResult vkCreateMicromapEXT( VkDevice                        device,
                                  const VkMicromapCreateInfoEXT * pCreateInfo,
                                  const VkAllocationCallbacks *   pAllocator,
                                  VkMicromapEXT *                 pMicromap ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateMicromapEXT( device, pCreateInfo, pAllocator, pMicromap );
    }

    void vkDestroyMicromapEXT( VkDevice device, VkMicromapEXT micromap, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyMicromapEXT( device, micromap, pAllocator );
    }

    void vkCmdBuildMicromapsEXT( VkCommandBuffer commandBuffer, uint32_t infoCount, const VkMicromapBuildInfoEXT * pInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBuildMicromapsEXT( commandBuffer, infoCount, pInfos );
    }

    VkResult vkBuildMicromapsEXT( VkDevice                       device,
                                  VkDeferredOperationKHR         deferredOperation,
                                  uint32_t                       infoCount,
                                  const VkMicromapBuildInfoEXT * pInfos ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBuildMicromapsEXT( device, deferredOperation, infoCount, pInfos );
    }

    VkResult vkCopyMicromapEXT( VkDevice device, VkDeferredOperationKHR deferredOperation, const VkCopyMicromapInfoEXT * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyMicromapEXT( device, deferredOperation, pInfo );
    }

    VkResult vkCopyMicromapToMemoryEXT( VkDevice                              device,
                                        VkDeferredOperationKHR                deferredOperation,
                                        const VkCopyMicromapToMemoryInfoEXT * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyMicromapToMemoryEXT( device, deferredOperation, pInfo );
    }

    VkResult vkCopyMemoryToMicromapEXT( VkDevice                              device,
                                        VkDeferredOperationKHR                deferredOperation,
                                        const VkCopyMemoryToMicromapInfoEXT * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCopyMemoryToMicromapEXT( device, deferredOperation, pInfo );
    }

    VkResult vkWriteMicromapsPropertiesEXT( VkDevice              device,
                                            uint32_t              micromapCount,
                                            const VkMicromapEXT * pMicromaps,
                                            VkQueryType           queryType,
                                            size_t                dataSize,
                                            void *                pData,
                                            size_t                stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkWriteMicromapsPropertiesEXT( device, micromapCount, pMicromaps, queryType, dataSize, pData, stride );
    }

    void vkCmdCopyMicromapEXT( VkCommandBuffer commandBuffer, const VkCopyMicromapInfoEXT * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyMicromapEXT( commandBuffer, pInfo );
    }

    void vkCmdCopyMicromapToMemoryEXT( VkCommandBuffer commandBuffer, const VkCopyMicromapToMemoryInfoEXT * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyMicromapToMemoryEXT( commandBuffer, pInfo );
    }

    void vkCmdCopyMemoryToMicromapEXT( VkCommandBuffer commandBuffer, const VkCopyMemoryToMicromapInfoEXT * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyMemoryToMicromapEXT( commandBuffer, pInfo );
    }

    void vkCmdWriteMicromapsPropertiesEXT( VkCommandBuffer       commandBuffer,
                                           uint32_t              micromapCount,
                                           const VkMicromapEXT * pMicromaps,
                                           VkQueryType           queryType,
                                           VkQueryPool           queryPool,
                                           uint32_t              firstQuery ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdWriteMicromapsPropertiesEXT( commandBuffer, micromapCount, pMicromaps, queryType, queryPool, firstQuery );
    }

    void vkGetDeviceMicromapCompatibilityEXT( VkDevice                                  device,
                                              const VkMicromapVersionInfoEXT *          pVersionInfo,
                                              VkAccelerationStructureCompatibilityKHR * pCompatibility ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceMicromapCompatibilityEXT( device, pVersionInfo, pCompatibility );
    }

    void vkGetMicromapBuildSizesEXT( VkDevice                            device,
                                     VkAccelerationStructureBuildTypeKHR buildType,
                                     const VkMicromapBuildInfoEXT *      pBuildInfo,
                                     VkMicromapBuildSizesInfoEXT *       pSizeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetMicromapBuildSizesEXT( device, buildType, pBuildInfo, pSizeInfo );
    }

    //=== VK_HUAWEI_cluster_culling_shader ===

    void vkCmdDrawClusterHUAWEI( VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawClusterHUAWEI( commandBuffer, groupCountX, groupCountY, groupCountZ );
    }

    void vkCmdDrawClusterIndirectHUAWEI( VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDrawClusterIndirectHUAWEI( commandBuffer, buffer, offset );
    }

    //=== VK_EXT_pageable_device_local_memory ===

    void vkSetDeviceMemoryPriorityEXT( VkDevice device, VkDeviceMemory memory, float priority ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetDeviceMemoryPriorityEXT( device, memory, priority );
    }

    //=== VK_KHR_maintenance4 ===

    void vkGetDeviceBufferMemoryRequirementsKHR( VkDevice                                 device,
                                                 const VkDeviceBufferMemoryRequirements * pInfo,
                                                 VkMemoryRequirements2 *                  pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceBufferMemoryRequirementsKHR( device, pInfo, pMemoryRequirements );
    }

    void vkGetDeviceImageMemoryRequirementsKHR( VkDevice                                device,
                                                const VkDeviceImageMemoryRequirements * pInfo,
                                                VkMemoryRequirements2 *                 pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceImageMemoryRequirementsKHR( device, pInfo, pMemoryRequirements );
    }

    void vkGetDeviceImageSparseMemoryRequirementsKHR( VkDevice                                device,
                                                      const VkDeviceImageMemoryRequirements * pInfo,
                                                      uint32_t *                              pSparseMemoryRequirementCount,
                                                      VkSparseImageMemoryRequirements2 *      pSparseMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceImageSparseMemoryRequirementsKHR( device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements );
    }

    //=== VK_VALVE_descriptor_set_host_mapping ===

    void vkGetDescriptorSetLayoutHostMappingInfoVALVE( VkDevice                                     device,
                                                       const VkDescriptorSetBindingReferenceVALVE * pBindingReference,
                                                       VkDescriptorSetLayoutHostMappingInfoVALVE *  pHostMapping ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDescriptorSetLayoutHostMappingInfoVALVE( device, pBindingReference, pHostMapping );
    }

    void vkGetDescriptorSetHostMappingVALVE( VkDevice device, VkDescriptorSet descriptorSet, void ** ppData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDescriptorSetHostMappingVALVE( device, descriptorSet, ppData );
    }

    //=== VK_NV_copy_memory_indirect ===

    void vkCmdCopyMemoryIndirectNV( VkCommandBuffer commandBuffer,
                                    VkDeviceAddress copyBufferAddress,
                                    uint32_t        copyCount,
                                    uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyMemoryIndirectNV( commandBuffer, copyBufferAddress, copyCount, stride );
    }

    void vkCmdCopyMemoryToImageIndirectNV( VkCommandBuffer                  commandBuffer,
                                           VkDeviceAddress                  copyBufferAddress,
                                           uint32_t                         copyCount,
                                           uint32_t                         stride,
                                           VkImage                          dstImage,
                                           VkImageLayout                    dstImageLayout,
                                           const VkImageSubresourceLayers * pImageSubresources ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdCopyMemoryToImageIndirectNV( commandBuffer, copyBufferAddress, copyCount, stride, dstImage, dstImageLayout, pImageSubresources );
    }

    //=== VK_NV_memory_decompression ===

    void vkCmdDecompressMemoryNV( VkCommandBuffer                    commandBuffer,
                                  uint32_t                           decompressRegionCount,
                                  const VkDecompressMemoryRegionNV * pDecompressMemoryRegions ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDecompressMemoryNV( commandBuffer, decompressRegionCount, pDecompressMemoryRegions );
    }

    void vkCmdDecompressMemoryIndirectCountNV( VkCommandBuffer commandBuffer,
                                               VkDeviceAddress indirectCommandsAddress,
                                               VkDeviceAddress indirectCommandsCountAddress,
                                               uint32_t        stride ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdDecompressMemoryIndirectCountNV( commandBuffer, indirectCommandsAddress, indirectCommandsCountAddress, stride );
    }

    //=== VK_NV_device_generated_commands_compute ===

    void vkGetPipelineIndirectMemoryRequirementsNV( VkDevice                            device,
                                                    const VkComputePipelineCreateInfo * pCreateInfo,
                                                    VkMemoryRequirements2 *             pMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineIndirectMemoryRequirementsNV( device, pCreateInfo, pMemoryRequirements );
    }

    void
      vkCmdUpdatePipelineIndirectBufferNV( VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdUpdatePipelineIndirectBufferNV( commandBuffer, pipelineBindPoint, pipeline );
    }

    VkDeviceAddress vkGetPipelineIndirectDeviceAddressNV( VkDevice device, const VkPipelineIndirectDeviceAddressInfoNV * pInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPipelineIndirectDeviceAddressNV( device, pInfo );
    }

    //=== VK_EXT_extended_dynamic_state3 ===

    void vkCmdSetTessellationDomainOriginEXT( VkCommandBuffer commandBuffer, VkTessellationDomainOrigin domainOrigin ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetTessellationDomainOriginEXT( commandBuffer, domainOrigin );
    }

    void vkCmdSetDepthClampEnableEXT( VkCommandBuffer commandBuffer, VkBool32 depthClampEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthClampEnableEXT( commandBuffer, depthClampEnable );
    }

    void vkCmdSetPolygonModeEXT( VkCommandBuffer commandBuffer, VkPolygonMode polygonMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetPolygonModeEXT( commandBuffer, polygonMode );
    }

    void vkCmdSetRasterizationSamplesEXT( VkCommandBuffer commandBuffer, VkSampleCountFlagBits rasterizationSamples ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetRasterizationSamplesEXT( commandBuffer, rasterizationSamples );
    }

    void vkCmdSetSampleMaskEXT( VkCommandBuffer commandBuffer, VkSampleCountFlagBits samples, const VkSampleMask * pSampleMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetSampleMaskEXT( commandBuffer, samples, pSampleMask );
    }

    void vkCmdSetAlphaToCoverageEnableEXT( VkCommandBuffer commandBuffer, VkBool32 alphaToCoverageEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetAlphaToCoverageEnableEXT( commandBuffer, alphaToCoverageEnable );
    }

    void vkCmdSetAlphaToOneEnableEXT( VkCommandBuffer commandBuffer, VkBool32 alphaToOneEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetAlphaToOneEnableEXT( commandBuffer, alphaToOneEnable );
    }

    void vkCmdSetLogicOpEnableEXT( VkCommandBuffer commandBuffer, VkBool32 logicOpEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetLogicOpEnableEXT( commandBuffer, logicOpEnable );
    }

    void vkCmdSetColorBlendEnableEXT( VkCommandBuffer  commandBuffer,
                                      uint32_t         firstAttachment,
                                      uint32_t         attachmentCount,
                                      const VkBool32 * pColorBlendEnables ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetColorBlendEnableEXT( commandBuffer, firstAttachment, attachmentCount, pColorBlendEnables );
    }

    void vkCmdSetColorBlendEquationEXT( VkCommandBuffer                 commandBuffer,
                                        uint32_t                        firstAttachment,
                                        uint32_t                        attachmentCount,
                                        const VkColorBlendEquationEXT * pColorBlendEquations ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetColorBlendEquationEXT( commandBuffer, firstAttachment, attachmentCount, pColorBlendEquations );
    }

    void vkCmdSetColorWriteMaskEXT( VkCommandBuffer               commandBuffer,
                                    uint32_t                      firstAttachment,
                                    uint32_t                      attachmentCount,
                                    const VkColorComponentFlags * pColorWriteMasks ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetColorWriteMaskEXT( commandBuffer, firstAttachment, attachmentCount, pColorWriteMasks );
    }

    void vkCmdSetRasterizationStreamEXT( VkCommandBuffer commandBuffer, uint32_t rasterizationStream ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetRasterizationStreamEXT( commandBuffer, rasterizationStream );
    }

    void vkCmdSetConservativeRasterizationModeEXT( VkCommandBuffer                    commandBuffer,
                                                   VkConservativeRasterizationModeEXT conservativeRasterizationMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetConservativeRasterizationModeEXT( commandBuffer, conservativeRasterizationMode );
    }

    void vkCmdSetExtraPrimitiveOverestimationSizeEXT( VkCommandBuffer commandBuffer, float extraPrimitiveOverestimationSize ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetExtraPrimitiveOverestimationSizeEXT( commandBuffer, extraPrimitiveOverestimationSize );
    }

    void vkCmdSetDepthClipEnableEXT( VkCommandBuffer commandBuffer, VkBool32 depthClipEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthClipEnableEXT( commandBuffer, depthClipEnable );
    }

    void vkCmdSetSampleLocationsEnableEXT( VkCommandBuffer commandBuffer, VkBool32 sampleLocationsEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetSampleLocationsEnableEXT( commandBuffer, sampleLocationsEnable );
    }

    void vkCmdSetColorBlendAdvancedEXT( VkCommandBuffer                 commandBuffer,
                                        uint32_t                        firstAttachment,
                                        uint32_t                        attachmentCount,
                                        const VkColorBlendAdvancedEXT * pColorBlendAdvanced ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetColorBlendAdvancedEXT( commandBuffer, firstAttachment, attachmentCount, pColorBlendAdvanced );
    }

    void vkCmdSetProvokingVertexModeEXT( VkCommandBuffer commandBuffer, VkProvokingVertexModeEXT provokingVertexMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetProvokingVertexModeEXT( commandBuffer, provokingVertexMode );
    }

    void vkCmdSetLineRasterizationModeEXT( VkCommandBuffer commandBuffer, VkLineRasterizationModeEXT lineRasterizationMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetLineRasterizationModeEXT( commandBuffer, lineRasterizationMode );
    }

    void vkCmdSetLineStippleEnableEXT( VkCommandBuffer commandBuffer, VkBool32 stippledLineEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetLineStippleEnableEXT( commandBuffer, stippledLineEnable );
    }

    void vkCmdSetDepthClipNegativeOneToOneEXT( VkCommandBuffer commandBuffer, VkBool32 negativeOneToOne ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDepthClipNegativeOneToOneEXT( commandBuffer, negativeOneToOne );
    }

    void vkCmdSetViewportWScalingEnableNV( VkCommandBuffer commandBuffer, VkBool32 viewportWScalingEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewportWScalingEnableNV( commandBuffer, viewportWScalingEnable );
    }

    void vkCmdSetViewportSwizzleNV( VkCommandBuffer             commandBuffer,
                                    uint32_t                    firstViewport,
                                    uint32_t                    viewportCount,
                                    const VkViewportSwizzleNV * pViewportSwizzles ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetViewportSwizzleNV( commandBuffer, firstViewport, viewportCount, pViewportSwizzles );
    }

    void vkCmdSetCoverageToColorEnableNV( VkCommandBuffer commandBuffer, VkBool32 coverageToColorEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCoverageToColorEnableNV( commandBuffer, coverageToColorEnable );
    }

    void vkCmdSetCoverageToColorLocationNV( VkCommandBuffer commandBuffer, uint32_t coverageToColorLocation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCoverageToColorLocationNV( commandBuffer, coverageToColorLocation );
    }

    void vkCmdSetCoverageModulationModeNV( VkCommandBuffer commandBuffer, VkCoverageModulationModeNV coverageModulationMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCoverageModulationModeNV( commandBuffer, coverageModulationMode );
    }

    void vkCmdSetCoverageModulationTableEnableNV( VkCommandBuffer commandBuffer, VkBool32 coverageModulationTableEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCoverageModulationTableEnableNV( commandBuffer, coverageModulationTableEnable );
    }

    void vkCmdSetCoverageModulationTableNV( VkCommandBuffer commandBuffer,
                                            uint32_t        coverageModulationTableCount,
                                            const float *   pCoverageModulationTable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCoverageModulationTableNV( commandBuffer, coverageModulationTableCount, pCoverageModulationTable );
    }

    void vkCmdSetShadingRateImageEnableNV( VkCommandBuffer commandBuffer, VkBool32 shadingRateImageEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetShadingRateImageEnableNV( commandBuffer, shadingRateImageEnable );
    }

    void vkCmdSetRepresentativeFragmentTestEnableNV( VkCommandBuffer commandBuffer, VkBool32 representativeFragmentTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetRepresentativeFragmentTestEnableNV( commandBuffer, representativeFragmentTestEnable );
    }

    void vkCmdSetCoverageReductionModeNV( VkCommandBuffer commandBuffer, VkCoverageReductionModeNV coverageReductionMode ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetCoverageReductionModeNV( commandBuffer, coverageReductionMode );
    }

    //=== VK_EXT_shader_module_identifier ===

    void vkGetShaderModuleIdentifierEXT( VkDevice device, VkShaderModule shaderModule, VkShaderModuleIdentifierEXT * pIdentifier ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetShaderModuleIdentifierEXT( device, shaderModule, pIdentifier );
    }

    void vkGetShaderModuleCreateInfoIdentifierEXT( VkDevice                         device,
                                                   const VkShaderModuleCreateInfo * pCreateInfo,
                                                   VkShaderModuleIdentifierEXT *    pIdentifier ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetShaderModuleCreateInfoIdentifierEXT( device, pCreateInfo, pIdentifier );
    }

    //=== VK_NV_optical_flow ===

    VkResult vkGetPhysicalDeviceOpticalFlowImageFormatsNV( VkPhysicalDevice                       physicalDevice,
                                                           const VkOpticalFlowImageFormatInfoNV * pOpticalFlowImageFormatInfo,
                                                           uint32_t *                             pFormatCount,
                                                           VkOpticalFlowImageFormatPropertiesNV * pImageFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceOpticalFlowImageFormatsNV( physicalDevice, pOpticalFlowImageFormatInfo, pFormatCount, pImageFormatProperties );
    }

    VkResult vkCreateOpticalFlowSessionNV( VkDevice                                 device,
                                           const VkOpticalFlowSessionCreateInfoNV * pCreateInfo,
                                           const VkAllocationCallbacks *            pAllocator,
                                           VkOpticalFlowSessionNV *                 pSession ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateOpticalFlowSessionNV( device, pCreateInfo, pAllocator, pSession );
    }

    void vkDestroyOpticalFlowSessionNV( VkDevice device, VkOpticalFlowSessionNV session, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyOpticalFlowSessionNV( device, session, pAllocator );
    }

    VkResult vkBindOpticalFlowSessionImageNV( VkDevice                           device,
                                              VkOpticalFlowSessionNV             session,
                                              VkOpticalFlowSessionBindingPointNV bindingPoint,
                                              VkImageView                        view,
                                              VkImageLayout                      layout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkBindOpticalFlowSessionImageNV( device, session, bindingPoint, view, layout );
    }

    void vkCmdOpticalFlowExecuteNV( VkCommandBuffer                    commandBuffer,
                                    VkOpticalFlowSessionNV             session,
                                    const VkOpticalFlowExecuteInfoNV * pExecuteInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdOpticalFlowExecuteNV( commandBuffer, session, pExecuteInfo );
    }

    //=== VK_KHR_maintenance5 ===

    void vkCmdBindIndexBuffer2KHR( VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, VkIndexType indexType ) const
      VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindIndexBuffer2KHR( commandBuffer, buffer, offset, size, indexType );
    }

    void vkGetRenderingAreaGranularityKHR( VkDevice                       device,
                                           const VkRenderingAreaInfoKHR * pRenderingAreaInfo,
                                           VkExtent2D *                   pGranularity ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetRenderingAreaGranularityKHR( device, pRenderingAreaInfo, pGranularity );
    }

    void vkGetDeviceImageSubresourceLayoutKHR( VkDevice                                device,
                                               const VkDeviceImageSubresourceInfoKHR * pInfo,
                                               VkSubresourceLayout2KHR *               pLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDeviceImageSubresourceLayoutKHR( device, pInfo, pLayout );
    }

    void vkGetImageSubresourceLayout2KHR( VkDevice                       device,
                                          VkImage                        image,
                                          const VkImageSubresource2KHR * pSubresource,
                                          VkSubresourceLayout2KHR *      pLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetImageSubresourceLayout2KHR( device, image, pSubresource, pLayout );
    }

    //=== VK_EXT_shader_object ===

    VkResult vkCreateShadersEXT( VkDevice                      device,
                                 uint32_t                      createInfoCount,
                                 const VkShaderCreateInfoEXT * pCreateInfos,
                                 const VkAllocationCallbacks * pAllocator,
                                 VkShaderEXT *                 pShaders ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCreateShadersEXT( device, createInfoCount, pCreateInfos, pAllocator, pShaders );
    }

    void vkDestroyShaderEXT( VkDevice device, VkShaderEXT shader, const VkAllocationCallbacks * pAllocator ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkDestroyShaderEXT( device, shader, pAllocator );
    }

    VkResult vkGetShaderBinaryDataEXT( VkDevice device, VkShaderEXT shader, size_t * pDataSize, void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetShaderBinaryDataEXT( device, shader, pDataSize, pData );
    }

    void vkCmdBindShadersEXT( VkCommandBuffer               commandBuffer,
                              uint32_t                      stageCount,
                              const VkShaderStageFlagBits * pStages,
                              const VkShaderEXT *           pShaders ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindShadersEXT( commandBuffer, stageCount, pStages, pShaders );
    }

    //=== VK_QCOM_tile_properties ===

    VkResult vkGetFramebufferTilePropertiesQCOM( VkDevice               device,
                                                 VkFramebuffer          framebuffer,
                                                 uint32_t *             pPropertiesCount,
                                                 VkTilePropertiesQCOM * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetFramebufferTilePropertiesQCOM( device, framebuffer, pPropertiesCount, pProperties );
    }

    VkResult vkGetDynamicRenderingTilePropertiesQCOM( VkDevice                device,
                                                      const VkRenderingInfo * pRenderingInfo,
                                                      VkTilePropertiesQCOM *  pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetDynamicRenderingTilePropertiesQCOM( device, pRenderingInfo, pProperties );
    }

    //=== VK_NV_low_latency2 ===

    VkResult vkSetLatencySleepModeNV( VkDevice device, VkSwapchainKHR swapchain, const VkLatencySleepModeInfoNV * pSleepModeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetLatencySleepModeNV( device, swapchain, pSleepModeInfo );
    }

    VkResult vkLatencySleepNV( VkDevice device, VkSwapchainKHR swapchain, const VkLatencySleepInfoNV * pSleepInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkLatencySleepNV( device, swapchain, pSleepInfo );
    }

    void vkSetLatencyMarkerNV( VkDevice device, VkSwapchainKHR swapchain, const VkSetLatencyMarkerInfoNV * pLatencyMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkSetLatencyMarkerNV( device, swapchain, pLatencyMarkerInfo );
    }

    void vkGetLatencyTimingsNV( VkDevice device, VkSwapchainKHR swapchain, VkGetLatencyMarkerInfoNV * pLatencyMarkerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetLatencyTimingsNV( device, swapchain, pLatencyMarkerInfo );
    }

    void vkQueueNotifyOutOfBandNV( VkQueue queue, const VkOutOfBandQueueTypeInfoNV * pQueueTypeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkQueueNotifyOutOfBandNV( queue, pQueueTypeInfo );
    }

    //=== VK_KHR_cooperative_matrix ===

    VkResult vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR( VkPhysicalDevice                   physicalDevice,
                                                                uint32_t *                         pPropertyCount,
                                                                VkCooperativeMatrixPropertiesKHR * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR( physicalDevice, pPropertyCount, pProperties );
    }

    //=== VK_EXT_attachment_feedback_loop_dynamic_state ===

    void vkCmdSetAttachmentFeedbackLoopEnableEXT( VkCommandBuffer commandBuffer, VkImageAspectFlags aspectMask ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetAttachmentFeedbackLoopEnableEXT( commandBuffer, aspectMask );
    }

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
    //=== VK_QNX_external_memory_screen_buffer ===

    VkResult vkGetScreenBufferPropertiesQNX( VkDevice                      device,
                                             const struct _screen_buffer * buffer,
                                             VkScreenBufferPropertiesQNX * pProperties ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetScreenBufferPropertiesQNX( device, buffer, pProperties );
    }
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

    //=== VK_KHR_calibrated_timestamps ===

    VkResult vkGetPhysicalDeviceCalibrateableTimeDomainsKHR( VkPhysicalDevice  physicalDevice,
                                                             uint32_t *        pTimeDomainCount,
                                                             VkTimeDomainKHR * pTimeDomains ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetPhysicalDeviceCalibrateableTimeDomainsKHR( physicalDevice, pTimeDomainCount, pTimeDomains );
    }

    VkResult vkGetCalibratedTimestampsKHR( VkDevice                             device,
                                           uint32_t                             timestampCount,
                                           const VkCalibratedTimestampInfoKHR * pTimestampInfos,
                                           uint64_t *                           pTimestamps,
                                           uint64_t *                           pMaxDeviation ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkGetCalibratedTimestampsKHR( device, timestampCount, pTimestampInfos, pTimestamps, pMaxDeviation );
    }

    //=== VK_KHR_maintenance6 ===

    void vkCmdBindDescriptorSets2KHR( VkCommandBuffer commandBuffer, const VkBindDescriptorSetsInfoKHR * pBindDescriptorSetsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindDescriptorSets2KHR( commandBuffer, pBindDescriptorSetsInfo );
    }

    void vkCmdPushConstants2KHR( VkCommandBuffer commandBuffer, const VkPushConstantsInfoKHR * pPushConstantsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPushConstants2KHR( commandBuffer, pPushConstantsInfo );
    }

    void vkCmdPushDescriptorSet2KHR( VkCommandBuffer commandBuffer, const VkPushDescriptorSetInfoKHR * pPushDescriptorSetInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPushDescriptorSet2KHR( commandBuffer, pPushDescriptorSetInfo );
    }

    void vkCmdPushDescriptorSetWithTemplate2KHR( VkCommandBuffer                                commandBuffer,
                                                 const VkPushDescriptorSetWithTemplateInfoKHR * pPushDescriptorSetWithTemplateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdPushDescriptorSetWithTemplate2KHR( commandBuffer, pPushDescriptorSetWithTemplateInfo );
    }

    void vkCmdSetDescriptorBufferOffsets2EXT( VkCommandBuffer                             commandBuffer,
                                              const VkSetDescriptorBufferOffsetsInfoEXT * pSetDescriptorBufferOffsetsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdSetDescriptorBufferOffsets2EXT( commandBuffer, pSetDescriptorBufferOffsetsInfo );
    }

    void vkCmdBindDescriptorBufferEmbeddedSamplers2EXT(
      VkCommandBuffer commandBuffer, const VkBindDescriptorBufferEmbeddedSamplersInfoEXT * pBindDescriptorBufferEmbeddedSamplersInfo ) const VULKAN_HPP_NOEXCEPT
    {
      return ::vkCmdBindDescriptorBufferEmbeddedSamplers2EXT( commandBuffer, pBindDescriptorBufferEmbeddedSamplersInfo );
    }
  };

  inline ::VULKAN_HPP_NAMESPACE::DispatchLoaderStatic & getDispatchLoaderStatic()
  {
    static ::VULKAN_HPP_NAMESPACE::DispatchLoaderStatic dls;
    return dls;
  }
#endif

#if !defined( VULKAN_HPP_NO_SMART_HANDLE )
  struct AllocationCallbacks;

  template <typename OwnerType, typename Dispatch>
  class ObjectDestroy
  {
  public:
    ObjectDestroy() = default;

    ObjectDestroy( OwnerType                                               owner,
                   Optional<const AllocationCallbacks> allocationCallbacks VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT,
                   Dispatch const & dispatch                               VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT ) VULKAN_HPP_NOEXCEPT
      : m_owner( owner )
      , m_allocationCallbacks( allocationCallbacks )
      , m_dispatch( &dispatch )
    {
    }

    OwnerType getOwner() const VULKAN_HPP_NOEXCEPT
    {
      return m_owner;
    }

    Optional<const AllocationCallbacks> getAllocator() const VULKAN_HPP_NOEXCEPT
    {
      return m_allocationCallbacks;
    }

    Dispatch const & getDispatch() const VULKAN_HPP_NOEXCEPT
    {
      return *m_dispatch;
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
                   Dispatch const & dispatch           VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT ) VULKAN_HPP_NOEXCEPT
      : m_allocationCallbacks( allocationCallbacks )
      , m_dispatch( &dispatch )
    {
    }

    Optional<const AllocationCallbacks> getAllocator() const VULKAN_HPP_NOEXCEPT
    {
      return m_allocationCallbacks;
    }

    Dispatch const & getDispatch() const VULKAN_HPP_NOEXCEPT
    {
      return *m_dispatch;
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

    ObjectFree( OwnerType                                               owner,
                Optional<const AllocationCallbacks> allocationCallbacks VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT,
                Dispatch const & dispatch                               VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT ) VULKAN_HPP_NOEXCEPT
      : m_owner( owner )
      , m_allocationCallbacks( allocationCallbacks )
      , m_dispatch( &dispatch )
    {
    }

    OwnerType getOwner() const VULKAN_HPP_NOEXCEPT
    {
      return m_owner;
    }

    Optional<const AllocationCallbacks> getAllocator() const VULKAN_HPP_NOEXCEPT
    {
      return m_allocationCallbacks;
    }

    Dispatch const & getDispatch() const VULKAN_HPP_NOEXCEPT
    {
      return *m_dispatch;
    }

  protected:
    template <typename T>
    void destroy( T t ) VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( m_owner && m_dispatch );
      ( m_owner.free )( t, m_allocationCallbacks, *m_dispatch );
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

    ObjectRelease( OwnerType owner, Dispatch const & dispatch VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT ) VULKAN_HPP_NOEXCEPT
      : m_owner( owner )
      , m_dispatch( &dispatch )
    {
    }

    OwnerType getOwner() const VULKAN_HPP_NOEXCEPT
    {
      return m_owner;
    }

    Dispatch const & getDispatch() const VULKAN_HPP_NOEXCEPT
    {
      return *m_dispatch;
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

    PoolFree( OwnerType owner, PoolType pool, Dispatch const & dispatch VULKAN_HPP_DEFAULT_DISPATCHER_ASSIGNMENT ) VULKAN_HPP_NOEXCEPT
      : m_owner( owner )
      , m_pool( pool )
      , m_dispatch( &dispatch )
    {
    }

    OwnerType getOwner() const VULKAN_HPP_NOEXCEPT
    {
      return m_owner;
    }

    PoolType getPool() const VULKAN_HPP_NOEXCEPT
    {
      return m_pool;
    }

    Dispatch const & getDispatch() const VULKAN_HPP_NOEXCEPT
    {
      return *m_dispatch;
    }

  protected:
    template <typename T>
    void destroy( T t ) VULKAN_HPP_NOEXCEPT
    {
      ( m_owner.free )( m_pool, t, *m_dispatch );
    }

  private:
    OwnerType        m_owner    = OwnerType();
    PoolType         m_pool     = PoolType();
    Dispatch const * m_dispatch = nullptr;
  };

#endif  // !VULKAN_HPP_NO_SMART_HANDLE

  //==================
  //=== BASE TYPEs ===
  //==================

  using Bool32          = uint32_t;
  using DeviceAddress   = uint64_t;
  using DeviceSize      = uint64_t;
  using RemoteAddressNV = void *;
  using SampleMask      = uint32_t;

}  // namespace VULKAN_HPP_NAMESPACE

#include <vulkan/vulkan_enums.hpp>
#if !defined( VULKAN_HPP_NO_TO_STRING )
#  include <vulkan/vulkan_to_string.hpp>
#endif

#ifndef VULKAN_HPP_NO_EXCEPTIONS
namespace std
{
  template <>
  struct is_error_code_enum<VULKAN_HPP_NAMESPACE::Result> : public true_type
  {
  };
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
#  if defined( VULKAN_HPP_NO_TO_STRING )
      return std::to_string( ev );
#  else
      return VULKAN_HPP_NAMESPACE::to_string( static_cast<VULKAN_HPP_NAMESPACE::Result>( ev ) );
#  endif
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

    SystemError( int ev, std::error_category const & ecat, std::string const & what ) : Error(), std::system_error( ev, ecat, what ) {}

    SystemError( int ev, std::error_category const & ecat, char const * what ) : Error(), std::system_error( ev, ecat, what ) {}

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
    OutOfHostMemoryError( std::string const & message ) : SystemError( make_error_code( Result::eErrorOutOfHostMemory ), message ) {}

    OutOfHostMemoryError( char const * message ) : SystemError( make_error_code( Result::eErrorOutOfHostMemory ), message ) {}
  };

  class OutOfDeviceMemoryError : public SystemError
  {
  public:
    OutOfDeviceMemoryError( std::string const & message ) : SystemError( make_error_code( Result::eErrorOutOfDeviceMemory ), message ) {}

    OutOfDeviceMemoryError( char const * message ) : SystemError( make_error_code( Result::eErrorOutOfDeviceMemory ), message ) {}
  };

  class InitializationFailedError : public SystemError
  {
  public:
    InitializationFailedError( std::string const & message ) : SystemError( make_error_code( Result::eErrorInitializationFailed ), message ) {}

    InitializationFailedError( char const * message ) : SystemError( make_error_code( Result::eErrorInitializationFailed ), message ) {}
  };

  class DeviceLostError : public SystemError
  {
  public:
    DeviceLostError( std::string const & message ) : SystemError( make_error_code( Result::eErrorDeviceLost ), message ) {}

    DeviceLostError( char const * message ) : SystemError( make_error_code( Result::eErrorDeviceLost ), message ) {}
  };

  class MemoryMapFailedError : public SystemError
  {
  public:
    MemoryMapFailedError( std::string const & message ) : SystemError( make_error_code( Result::eErrorMemoryMapFailed ), message ) {}

    MemoryMapFailedError( char const * message ) : SystemError( make_error_code( Result::eErrorMemoryMapFailed ), message ) {}
  };

  class LayerNotPresentError : public SystemError
  {
  public:
    LayerNotPresentError( std::string const & message ) : SystemError( make_error_code( Result::eErrorLayerNotPresent ), message ) {}

    LayerNotPresentError( char const * message ) : SystemError( make_error_code( Result::eErrorLayerNotPresent ), message ) {}
  };

  class ExtensionNotPresentError : public SystemError
  {
  public:
    ExtensionNotPresentError( std::string const & message ) : SystemError( make_error_code( Result::eErrorExtensionNotPresent ), message ) {}

    ExtensionNotPresentError( char const * message ) : SystemError( make_error_code( Result::eErrorExtensionNotPresent ), message ) {}
  };

  class FeatureNotPresentError : public SystemError
  {
  public:
    FeatureNotPresentError( std::string const & message ) : SystemError( make_error_code( Result::eErrorFeatureNotPresent ), message ) {}

    FeatureNotPresentError( char const * message ) : SystemError( make_error_code( Result::eErrorFeatureNotPresent ), message ) {}
  };

  class IncompatibleDriverError : public SystemError
  {
  public:
    IncompatibleDriverError( std::string const & message ) : SystemError( make_error_code( Result::eErrorIncompatibleDriver ), message ) {}

    IncompatibleDriverError( char const * message ) : SystemError( make_error_code( Result::eErrorIncompatibleDriver ), message ) {}
  };

  class TooManyObjectsError : public SystemError
  {
  public:
    TooManyObjectsError( std::string const & message ) : SystemError( make_error_code( Result::eErrorTooManyObjects ), message ) {}

    TooManyObjectsError( char const * message ) : SystemError( make_error_code( Result::eErrorTooManyObjects ), message ) {}
  };

  class FormatNotSupportedError : public SystemError
  {
  public:
    FormatNotSupportedError( std::string const & message ) : SystemError( make_error_code( Result::eErrorFormatNotSupported ), message ) {}

    FormatNotSupportedError( char const * message ) : SystemError( make_error_code( Result::eErrorFormatNotSupported ), message ) {}
  };

  class FragmentedPoolError : public SystemError
  {
  public:
    FragmentedPoolError( std::string const & message ) : SystemError( make_error_code( Result::eErrorFragmentedPool ), message ) {}

    FragmentedPoolError( char const * message ) : SystemError( make_error_code( Result::eErrorFragmentedPool ), message ) {}
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
    OutOfPoolMemoryError( std::string const & message ) : SystemError( make_error_code( Result::eErrorOutOfPoolMemory ), message ) {}

    OutOfPoolMemoryError( char const * message ) : SystemError( make_error_code( Result::eErrorOutOfPoolMemory ), message ) {}
  };

  class InvalidExternalHandleError : public SystemError
  {
  public:
    InvalidExternalHandleError( std::string const & message ) : SystemError( make_error_code( Result::eErrorInvalidExternalHandle ), message ) {}

    InvalidExternalHandleError( char const * message ) : SystemError( make_error_code( Result::eErrorInvalidExternalHandle ), message ) {}
  };

  class FragmentationError : public SystemError
  {
  public:
    FragmentationError( std::string const & message ) : SystemError( make_error_code( Result::eErrorFragmentation ), message ) {}

    FragmentationError( char const * message ) : SystemError( make_error_code( Result::eErrorFragmentation ), message ) {}
  };

  class InvalidOpaqueCaptureAddressError : public SystemError
  {
  public:
    InvalidOpaqueCaptureAddressError( std::string const & message ) : SystemError( make_error_code( Result::eErrorInvalidOpaqueCaptureAddress ), message ) {}

    InvalidOpaqueCaptureAddressError( char const * message ) : SystemError( make_error_code( Result::eErrorInvalidOpaqueCaptureAddress ), message ) {}
  };

  class SurfaceLostKHRError : public SystemError
  {
  public:
    SurfaceLostKHRError( std::string const & message ) : SystemError( make_error_code( Result::eErrorSurfaceLostKHR ), message ) {}

    SurfaceLostKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorSurfaceLostKHR ), message ) {}
  };

  class NativeWindowInUseKHRError : public SystemError
  {
  public:
    NativeWindowInUseKHRError( std::string const & message ) : SystemError( make_error_code( Result::eErrorNativeWindowInUseKHR ), message ) {}

    NativeWindowInUseKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorNativeWindowInUseKHR ), message ) {}
  };

  class OutOfDateKHRError : public SystemError
  {
  public:
    OutOfDateKHRError( std::string const & message ) : SystemError( make_error_code( Result::eErrorOutOfDateKHR ), message ) {}

    OutOfDateKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorOutOfDateKHR ), message ) {}
  };

  class IncompatibleDisplayKHRError : public SystemError
  {
  public:
    IncompatibleDisplayKHRError( std::string const & message ) : SystemError( make_error_code( Result::eErrorIncompatibleDisplayKHR ), message ) {}

    IncompatibleDisplayKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorIncompatibleDisplayKHR ), message ) {}
  };

  class ValidationFailedEXTError : public SystemError
  {
  public:
    ValidationFailedEXTError( std::string const & message ) : SystemError( make_error_code( Result::eErrorValidationFailedEXT ), message ) {}

    ValidationFailedEXTError( char const * message ) : SystemError( make_error_code( Result::eErrorValidationFailedEXT ), message ) {}
  };

  class InvalidShaderNVError : public SystemError
  {
  public:
    InvalidShaderNVError( std::string const & message ) : SystemError( make_error_code( Result::eErrorInvalidShaderNV ), message ) {}

    InvalidShaderNVError( char const * message ) : SystemError( make_error_code( Result::eErrorInvalidShaderNV ), message ) {}
  };

  class ImageUsageNotSupportedKHRError : public SystemError
  {
  public:
    ImageUsageNotSupportedKHRError( std::string const & message ) : SystemError( make_error_code( Result::eErrorImageUsageNotSupportedKHR ), message ) {}

    ImageUsageNotSupportedKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorImageUsageNotSupportedKHR ), message ) {}
  };

  class VideoPictureLayoutNotSupportedKHRError : public SystemError
  {
  public:
    VideoPictureLayoutNotSupportedKHRError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorVideoPictureLayoutNotSupportedKHR ), message )
    {
    }

    VideoPictureLayoutNotSupportedKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorVideoPictureLayoutNotSupportedKHR ), message )
    {
    }
  };

  class VideoProfileOperationNotSupportedKHRError : public SystemError
  {
  public:
    VideoProfileOperationNotSupportedKHRError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorVideoProfileOperationNotSupportedKHR ), message )
    {
    }

    VideoProfileOperationNotSupportedKHRError( char const * message )
      : SystemError( make_error_code( Result::eErrorVideoProfileOperationNotSupportedKHR ), message )
    {
    }
  };

  class VideoProfileFormatNotSupportedKHRError : public SystemError
  {
  public:
    VideoProfileFormatNotSupportedKHRError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorVideoProfileFormatNotSupportedKHR ), message )
    {
    }

    VideoProfileFormatNotSupportedKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorVideoProfileFormatNotSupportedKHR ), message )
    {
    }
  };

  class VideoProfileCodecNotSupportedKHRError : public SystemError
  {
  public:
    VideoProfileCodecNotSupportedKHRError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorVideoProfileCodecNotSupportedKHR ), message )
    {
    }

    VideoProfileCodecNotSupportedKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorVideoProfileCodecNotSupportedKHR ), message ) {}
  };

  class VideoStdVersionNotSupportedKHRError : public SystemError
  {
  public:
    VideoStdVersionNotSupportedKHRError( std::string const & message ) : SystemError( make_error_code( Result::eErrorVideoStdVersionNotSupportedKHR ), message )
    {
    }

    VideoStdVersionNotSupportedKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorVideoStdVersionNotSupportedKHR ), message ) {}
  };

  class InvalidDrmFormatModifierPlaneLayoutEXTError : public SystemError
  {
  public:
    InvalidDrmFormatModifierPlaneLayoutEXTError( std::string const & message )
      : SystemError( make_error_code( Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT ), message )
    {
    }

    InvalidDrmFormatModifierPlaneLayoutEXTError( char const * message )
      : SystemError( make_error_code( Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT ), message )
    {
    }
  };

  class NotPermittedKHRError : public SystemError
  {
  public:
    NotPermittedKHRError( std::string const & message ) : SystemError( make_error_code( Result::eErrorNotPermittedKHR ), message ) {}

    NotPermittedKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorNotPermittedKHR ), message ) {}
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  class FullScreenExclusiveModeLostEXTError : public SystemError
  {
  public:
    FullScreenExclusiveModeLostEXTError( std::string const & message ) : SystemError( make_error_code( Result::eErrorFullScreenExclusiveModeLostEXT ), message )
    {
    }

    FullScreenExclusiveModeLostEXTError( char const * message ) : SystemError( make_error_code( Result::eErrorFullScreenExclusiveModeLostEXT ), message ) {}
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  class InvalidVideoStdParametersKHRError : public SystemError
  {
  public:
    InvalidVideoStdParametersKHRError( std::string const & message ) : SystemError( make_error_code( Result::eErrorInvalidVideoStdParametersKHR ), message ) {}

    InvalidVideoStdParametersKHRError( char const * message ) : SystemError( make_error_code( Result::eErrorInvalidVideoStdParametersKHR ), message ) {}
  };

  class CompressionExhaustedEXTError : public SystemError
  {
  public:
    CompressionExhaustedEXTError( std::string const & message ) : SystemError( make_error_code( Result::eErrorCompressionExhaustedEXT ), message ) {}

    CompressionExhaustedEXTError( char const * message ) : SystemError( make_error_code( Result::eErrorCompressionExhaustedEXT ), message ) {}
  };

  class IncompatibleShaderBinaryEXTError : public SystemError
  {
  public:
    IncompatibleShaderBinaryEXTError( std::string const & message ) : SystemError( make_error_code( Result::eErrorIncompatibleShaderBinaryEXT ), message ) {}

    IncompatibleShaderBinaryEXTError( char const * message ) : SystemError( make_error_code( Result::eErrorIncompatibleShaderBinaryEXT ), message ) {}
  };

  namespace detail
  {
    [[noreturn]] VULKAN_HPP_INLINE void throwResultException( Result result, char const * message )
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
        case Result::eErrorImageUsageNotSupportedKHR: throw ImageUsageNotSupportedKHRError( message );
        case Result::eErrorVideoPictureLayoutNotSupportedKHR: throw VideoPictureLayoutNotSupportedKHRError( message );
        case Result::eErrorVideoProfileOperationNotSupportedKHR: throw VideoProfileOperationNotSupportedKHRError( message );
        case Result::eErrorVideoProfileFormatNotSupportedKHR: throw VideoProfileFormatNotSupportedKHRError( message );
        case Result::eErrorVideoProfileCodecNotSupportedKHR: throw VideoProfileCodecNotSupportedKHRError( message );
        case Result::eErrorVideoStdVersionNotSupportedKHR: throw VideoStdVersionNotSupportedKHRError( message );
        case Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT: throw InvalidDrmFormatModifierPlaneLayoutEXTError( message );
        case Result::eErrorNotPermittedKHR: throw NotPermittedKHRError( message );
#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        case Result::eErrorFullScreenExclusiveModeLostEXT: throw FullScreenExclusiveModeLostEXTError( message );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/
        case Result::eErrorInvalidVideoStdParametersKHR: throw InvalidVideoStdParametersKHRError( message );
        case Result::eErrorCompressionExhaustedEXT: throw CompressionExhaustedEXTError( message );
        case Result::eErrorIncompatibleShaderBinaryEXT: throw IncompatibleShaderBinaryEXTError( message );
        default: throw SystemError( make_error_code( result ), message );
      }
    }
  }  // namespace detail
#endif

  template <typename T>
  void ignore( T const & ) VULKAN_HPP_NOEXCEPT
  {
  }

  template <typename T>
  struct ResultValue
  {
#ifdef VULKAN_HPP_HAS_NOEXCEPT
    ResultValue( Result r, T & v ) VULKAN_HPP_NOEXCEPT( VULKAN_HPP_NOEXCEPT( T( v ) ) )
#else
    ResultValue( Result r, T & v )
#endif
      : result( r ), value( v )
    {
    }

#ifdef VULKAN_HPP_HAS_NOEXCEPT
    ResultValue( Result r, T && v ) VULKAN_HPP_NOEXCEPT( VULKAN_HPP_NOEXCEPT( T( std::move( v ) ) ) )
#else
    ResultValue( Result r, T && v )
#endif
      : result( r ), value( std::move( v ) )
    {
    }

    Result result;
    T      value;

    operator std::tuple<Result &, T &>() VULKAN_HPP_NOEXCEPT
    {
      return std::tuple<Result &, T &>( result, value );
    }
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
    {
    }

    VULKAN_HPP_DEPRECATED(
      "asTuple() on an l-value is deprecated, as it implicitly moves the UniqueHandle out of the ResultValue. Use asTuple() on an r-value instead, requiring to explicitly move the UniqueHandle." )

    std::tuple<Result, UniqueHandle<Type, Dispatch>> asTuple() &
    {
      return std::make_tuple( result, std::move( value ) );
    }

    std::tuple<Result, UniqueHandle<Type, Dispatch>> asTuple() &&
    {
      return std::make_tuple( result, std::move( value ) );
    }

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
    {
    }

    VULKAN_HPP_DEPRECATED(
      "asTuple() on an l-value is deprecated, as it implicitly moves the UniqueHandle out of the ResultValue. Use asTuple() on an r-value instead, requiring to explicitly move the UniqueHandle." )

    std::tuple<Result, std::vector<UniqueHandle<Type, Dispatch>>> asTuple() &
    {
      return std::make_tuple( result, std::move( value ) );
    }

    std::tuple<Result, std::vector<UniqueHandle<Type, Dispatch>>> asTuple() &&
    {
      return std::make_tuple( result, std::move( value ) );
    }

    Result                                    result;
    std::vector<UniqueHandle<Type, Dispatch>> value;
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

  VULKAN_HPP_INLINE typename ResultValueType<void>::type createResultValueType( Result result )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    return result;
#else
    ignore( result );
#endif
  }

  template <typename T>
  VULKAN_HPP_INLINE typename ResultValueType<T>::type createResultValueType( Result result, T & data )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    return ResultValue<T>( result, data );
#else
    ignore( result );
    return data;
#endif
  }

  template <typename T>
  VULKAN_HPP_INLINE typename ResultValueType<T>::type createResultValueType( Result result, T && data )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    return ResultValue<T>( result, std::move( data ) );
#else
    ignore( result );
    return std::move( data );
#endif
  }

  VULKAN_HPP_INLINE void resultCheck( Result result, char const * message )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( result );  // just in case VULKAN_HPP_ASSERT_ON_RESULT is empty
    ignore( message );
    VULKAN_HPP_ASSERT_ON_RESULT( result == Result::eSuccess );
#else
    if ( result != Result::eSuccess )
    {
      detail::throwResultException( result, message );
    }
#endif
  }

  VULKAN_HPP_INLINE void resultCheck( Result result, char const * message, std::initializer_list<Result> successCodes )
  {
#ifdef VULKAN_HPP_NO_EXCEPTIONS
    ignore( result );  // just in case VULKAN_HPP_ASSERT_ON_RESULT is empty
    ignore( message );
    ignore( successCodes );  // just in case VULKAN_HPP_ASSERT_ON_RESULT is empty
    VULKAN_HPP_ASSERT_ON_RESULT( std::find( successCodes.begin(), successCodes.end(), result ) != successCodes.end() );
#else
    if ( std::find( successCodes.begin(), successCodes.end(), result ) == successCodes.end() )
    {
      detail::throwResultException( result, message );
    }
#endif
  }

  //===========================
  //=== CONSTEXPR CONSTANTs ===
  //===========================

  //=== VK_VERSION_1_0 ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t AttachmentUnused          = VK_ATTACHMENT_UNUSED;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t False                     = VK_FALSE;
  VULKAN_HPP_CONSTEXPR_INLINE float    LodClampNone              = VK_LOD_CLAMP_NONE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t QueueFamilyIgnored        = VK_QUEUE_FAMILY_IGNORED;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t RemainingArrayLayers      = VK_REMAINING_ARRAY_LAYERS;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t RemainingMipLevels        = VK_REMAINING_MIP_LEVELS;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t SubpassExternal           = VK_SUBPASS_EXTERNAL;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t True                      = VK_TRUE;
  VULKAN_HPP_CONSTEXPR_INLINE uint64_t WholeSize                 = VK_WHOLE_SIZE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxMemoryTypes            = VK_MAX_MEMORY_TYPES;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxPhysicalDeviceNameSize = VK_MAX_PHYSICAL_DEVICE_NAME_SIZE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t UuidSize                  = VK_UUID_SIZE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxExtensionNameSize      = VK_MAX_EXTENSION_NAME_SIZE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxDescriptionSize        = VK_MAX_DESCRIPTION_SIZE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxMemoryHeaps            = VK_MAX_MEMORY_HEAPS;

  //=== VK_VERSION_1_1 ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxDeviceGroupSize  = VK_MAX_DEVICE_GROUP_SIZE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t LuidSize            = VK_LUID_SIZE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t QueueFamilyExternal = VK_QUEUE_FAMILY_EXTERNAL;

  //=== VK_VERSION_1_2 ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxDriverNameSize = VK_MAX_DRIVER_NAME_SIZE;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxDriverInfoSize = VK_MAX_DRIVER_INFO_SIZE;

  //=== VK_KHR_device_group_creation ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxDeviceGroupSizeKHR = VK_MAX_DEVICE_GROUP_SIZE_KHR;

  //=== VK_KHR_external_memory_capabilities ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t LuidSizeKHR = VK_LUID_SIZE_KHR;

  //=== VK_KHR_external_memory ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t QueueFamilyExternalKHR = VK_QUEUE_FAMILY_EXTERNAL_KHR;

  //=== VK_EXT_queue_family_foreign ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t QueueFamilyForeignEXT = VK_QUEUE_FAMILY_FOREIGN_EXT;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_shader_enqueue ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t ShaderIndexUnusedAMDX = VK_SHADER_INDEX_UNUSED_AMDX;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_KHR_ray_tracing_pipeline ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t ShaderUnusedKHR = VK_SHADER_UNUSED_KHR;

  //=== VK_NV_ray_tracing ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t ShaderUnusedNV = VK_SHADER_UNUSED_NV;

  //=== VK_KHR_global_priority ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxGlobalPrioritySizeKHR = VK_MAX_GLOBAL_PRIORITY_SIZE_KHR;

  //=== VK_KHR_driver_properties ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxDriverNameSizeKHR = VK_MAX_DRIVER_NAME_SIZE_KHR;
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxDriverInfoSizeKHR = VK_MAX_DRIVER_INFO_SIZE_KHR;

  //=== VK_EXT_global_priority_query ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxGlobalPrioritySizeEXT = VK_MAX_GLOBAL_PRIORITY_SIZE_EXT;

  //=== VK_EXT_image_sliced_view_of_3d ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t Remaining3DSlicesEXT = VK_REMAINING_3D_SLICES_EXT;

  //=== VK_EXT_shader_module_identifier ===
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t MaxShaderModuleIdentifierSizeEXT = VK_MAX_SHADER_MODULE_IDENTIFIER_SIZE_EXT;

  //========================
  //=== CONSTEXPR VALUEs ===
  //========================
  VULKAN_HPP_CONSTEXPR_INLINE uint32_t HeaderVersion = VK_HEADER_VERSION;

  //=========================
  //=== CONSTEXPR CALLEEs ===
  //=========================
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_CONSTEXPR uint32_t apiVersionMajor( T const version )
  {
    return ( ( (uint32_t)( version ) >> 22U ) & 0x7FU );
  }

  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_CONSTEXPR uint32_t apiVersionMinor( T const version )
  {
    return ( ( (uint32_t)( version ) >> 12U ) & 0x3FFU );
  }

  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_CONSTEXPR uint32_t apiVersionPatch( T const version )
  {
    return ( (uint32_t)(version)&0xFFFU );
  }

  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_CONSTEXPR uint32_t apiVersionVariant( T const version )
  {
    return ( (uint32_t)( version ) >> 29U );
  }

  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_CONSTEXPR uint32_t makeApiVersion( T const variant, T const major, T const minor, T const patch )
  {
    return ( ( ( (uint32_t)( variant ) ) << 29U ) | ( ( (uint32_t)( major ) ) << 22U ) | ( ( (uint32_t)( minor ) ) << 12U ) | ( (uint32_t)( patch ) ) );
  }

  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_DEPRECATED( "This define is deprecated. VK_MAKE_API_VERSION should be used instead." )
  VULKAN_HPP_CONSTEXPR uint32_t makeVersion( T const major, T const minor, T const patch )
  {
    return ( ( ( (uint32_t)( major ) ) << 22U ) | ( ( (uint32_t)( minor ) ) << 12U ) | ( (uint32_t)( patch ) ) );
  }

  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_DEPRECATED( "This define is deprecated. VK_API_VERSION_MAJOR should be used instead." )
  VULKAN_HPP_CONSTEXPR uint32_t versionMajor( T const version )
  {
    return ( (uint32_t)( version ) >> 22U );
  }

  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_DEPRECATED( "This define is deprecated. VK_API_VERSION_MINOR should be used instead." )
  VULKAN_HPP_CONSTEXPR uint32_t versionMinor( T const version )
  {
    return ( ( (uint32_t)( version ) >> 12U ) & 0x3FFU );
  }

  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  VULKAN_HPP_DEPRECATED( "This define is deprecated. VK_API_VERSION_PATCH should be used instead." )
  VULKAN_HPP_CONSTEXPR uint32_t versionPatch( T const version )
  {
    return ( (uint32_t)(version)&0xFFFU );
  }

  //=========================
  //=== CONSTEXPR CALLERs ===
  //=========================
  VULKAN_HPP_CONSTEXPR_INLINE auto ApiVersion            = makeApiVersion( 0, 1, 0, 0 );
  VULKAN_HPP_CONSTEXPR_INLINE auto ApiVersion10          = makeApiVersion( 0, 1, 0, 0 );
  VULKAN_HPP_CONSTEXPR_INLINE auto ApiVersion11          = makeApiVersion( 0, 1, 1, 0 );
  VULKAN_HPP_CONSTEXPR_INLINE auto ApiVersion12          = makeApiVersion( 0, 1, 2, 0 );
  VULKAN_HPP_CONSTEXPR_INLINE auto ApiVersion13          = makeApiVersion( 0, 1, 3, 0 );
  VULKAN_HPP_CONSTEXPR_INLINE auto HeaderVersionComplete = makeApiVersion( 0, 1, 3, VK_HEADER_VERSION );

  //=================================
  //=== CONSTEXPR EXTENSION NAMEs ===
  //=================================

  //=== VK_KHR_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSurfaceExtensionName = VK_KHR_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSurfaceSpecVersion   = VK_KHR_SURFACE_SPEC_VERSION;

  //=== VK_KHR_swapchain ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSwapchainExtensionName = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSwapchainSpecVersion   = VK_KHR_SWAPCHAIN_SPEC_VERSION;

  //=== VK_KHR_display ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDisplayExtensionName = VK_KHR_DISPLAY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDisplaySpecVersion   = VK_KHR_DISPLAY_SPEC_VERSION;

  //=== VK_KHR_display_swapchain ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDisplaySwapchainExtensionName = VK_KHR_DISPLAY_SWAPCHAIN_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDisplaySwapchainSpecVersion   = VK_KHR_DISPLAY_SWAPCHAIN_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRXlibSurfaceExtensionName = VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRXlibSurfaceSpecVersion   = VK_KHR_XLIB_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRXcbSurfaceExtensionName = VK_KHR_XCB_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRXcbSurfaceSpecVersion   = VK_KHR_XCB_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRWaylandSurfaceExtensionName = VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRWaylandSurfaceSpecVersion   = VK_KHR_WAYLAND_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRAndroidSurfaceExtensionName = VK_KHR_ANDROID_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRAndroidSurfaceSpecVersion   = VK_KHR_ANDROID_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRWin32SurfaceExtensionName = VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRWin32SurfaceSpecVersion   = VK_KHR_WIN32_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_debug_report extension has been deprecated by VK_EXT_debug_utils." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDebugReportExtensionName = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_debug_report extension has been deprecated by VK_EXT_debug_utils." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDebugReportSpecVersion = VK_EXT_DEBUG_REPORT_SPEC_VERSION;

  //=== VK_NV_glsl_shader ===
  VULKAN_HPP_DEPRECATED( "The VK_NV_glsl_shader extension has been deprecated." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVGlslShaderExtensionName = VK_NV_GLSL_SHADER_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_NV_glsl_shader extension has been deprecated." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVGlslShaderSpecVersion = VK_NV_GLSL_SHADER_SPEC_VERSION;

  //=== VK_EXT_depth_range_unrestricted ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthRangeUnrestrictedExtensionName = VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthRangeUnrestrictedSpecVersion   = VK_EXT_DEPTH_RANGE_UNRESTRICTED_SPEC_VERSION;

  //=== VK_KHR_sampler_mirror_clamp_to_edge ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_sampler_mirror_clamp_to_edge extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSamplerMirrorClampToEdgeExtensionName = VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_sampler_mirror_clamp_to_edge extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSamplerMirrorClampToEdgeSpecVersion = VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_SPEC_VERSION;

  //=== VK_IMG_filter_cubic ===
  VULKAN_HPP_CONSTEXPR_INLINE auto IMGFilterCubicExtensionName = VK_IMG_FILTER_CUBIC_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto IMGFilterCubicSpecVersion   = VK_IMG_FILTER_CUBIC_SPEC_VERSION;

  //=== VK_AMD_rasterization_order ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDRasterizationOrderExtensionName = VK_AMD_RASTERIZATION_ORDER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDRasterizationOrderSpecVersion   = VK_AMD_RASTERIZATION_ORDER_SPEC_VERSION;

  //=== VK_AMD_shader_trinary_minmax ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderTrinaryMinmaxExtensionName = VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderTrinaryMinmaxSpecVersion   = VK_AMD_SHADER_TRINARY_MINMAX_SPEC_VERSION;

  //=== VK_AMD_shader_explicit_vertex_parameter ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderExplicitVertexParameterExtensionName = VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderExplicitVertexParameterSpecVersion   = VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_SPEC_VERSION;

  //=== VK_EXT_debug_marker ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_debug_marker extension has been promoted to VK_EXT_debug_utils." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDebugMarkerExtensionName = VK_EXT_DEBUG_MARKER_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_debug_marker extension has been promoted to VK_EXT_debug_utils." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDebugMarkerSpecVersion = VK_EXT_DEBUG_MARKER_SPEC_VERSION;

  //=== VK_KHR_video_queue ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoQueueExtensionName = VK_KHR_VIDEO_QUEUE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoQueueSpecVersion   = VK_KHR_VIDEO_QUEUE_SPEC_VERSION;

  //=== VK_KHR_video_decode_queue ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoDecodeQueueExtensionName = VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoDecodeQueueSpecVersion   = VK_KHR_VIDEO_DECODE_QUEUE_SPEC_VERSION;

  //=== VK_AMD_gcn_shader ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDGcnShaderExtensionName = VK_AMD_GCN_SHADER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDGcnShaderSpecVersion   = VK_AMD_GCN_SHADER_SPEC_VERSION;

  //=== VK_NV_dedicated_allocation ===
  VULKAN_HPP_DEPRECATED( "The VK_NV_dedicated_allocation extension has been deprecated by VK_KHR_dedicated_allocation." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDedicatedAllocationExtensionName = VK_NV_DEDICATED_ALLOCATION_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_NV_dedicated_allocation extension has been deprecated by VK_KHR_dedicated_allocation." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDedicatedAllocationSpecVersion = VK_NV_DEDICATED_ALLOCATION_SPEC_VERSION;

  //=== VK_EXT_transform_feedback ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTTransformFeedbackExtensionName = VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTTransformFeedbackSpecVersion   = VK_EXT_TRANSFORM_FEEDBACK_SPEC_VERSION;

  //=== VK_NVX_binary_import ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVXBinaryImportExtensionName = VK_NVX_BINARY_IMPORT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVXBinaryImportSpecVersion   = VK_NVX_BINARY_IMPORT_SPEC_VERSION;

  //=== VK_NVX_image_view_handle ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVXImageViewHandleExtensionName = VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVXImageViewHandleSpecVersion   = VK_NVX_IMAGE_VIEW_HANDLE_SPEC_VERSION;

  //=== VK_AMD_draw_indirect_count ===
  VULKAN_HPP_DEPRECATED( "The VK_AMD_draw_indirect_count extension has been promoted to VK_KHR_draw_indirect_count." )
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDDrawIndirectCountExtensionName = VK_AMD_DRAW_INDIRECT_COUNT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_AMD_draw_indirect_count extension has been promoted to VK_KHR_draw_indirect_count." )
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDDrawIndirectCountSpecVersion = VK_AMD_DRAW_INDIRECT_COUNT_SPEC_VERSION;

  //=== VK_AMD_negative_viewport_height ===
  VULKAN_HPP_DEPRECATED( "The VK_AMD_negative_viewport_height extension has been obsoleted by VK_KHR_maintenance1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDNegativeViewportHeightExtensionName = VK_AMD_NEGATIVE_VIEWPORT_HEIGHT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_AMD_negative_viewport_height extension has been obsoleted by VK_KHR_maintenance1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDNegativeViewportHeightSpecVersion = VK_AMD_NEGATIVE_VIEWPORT_HEIGHT_SPEC_VERSION;

  //=== VK_AMD_gpu_shader_half_float ===
  VULKAN_HPP_DEPRECATED( "The VK_AMD_gpu_shader_half_float extension has been deprecated by VK_KHR_shader_float16_int8." )
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDGpuShaderHalfFloatExtensionName = VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_AMD_gpu_shader_half_float extension has been deprecated by VK_KHR_shader_float16_int8." )
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDGpuShaderHalfFloatSpecVersion = VK_AMD_GPU_SHADER_HALF_FLOAT_SPEC_VERSION;

  //=== VK_AMD_shader_ballot ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderBallotExtensionName = VK_AMD_SHADER_BALLOT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderBallotSpecVersion   = VK_AMD_SHADER_BALLOT_SPEC_VERSION;

  //=== VK_KHR_video_encode_h264 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoEncodeH264ExtensionName = VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoEncodeH264SpecVersion   = VK_KHR_VIDEO_ENCODE_H264_SPEC_VERSION;

  //=== VK_KHR_video_encode_h265 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoEncodeH265ExtensionName = VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoEncodeH265SpecVersion   = VK_KHR_VIDEO_ENCODE_H265_SPEC_VERSION;

  //=== VK_KHR_video_decode_h264 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoDecodeH264ExtensionName = VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoDecodeH264SpecVersion   = VK_KHR_VIDEO_DECODE_H264_SPEC_VERSION;

  //=== VK_AMD_texture_gather_bias_lod ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDTextureGatherBiasLodExtensionName = VK_AMD_TEXTURE_GATHER_BIAS_LOD_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDTextureGatherBiasLodSpecVersion   = VK_AMD_TEXTURE_GATHER_BIAS_LOD_SPEC_VERSION;

  //=== VK_AMD_shader_info ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderInfoExtensionName = VK_AMD_SHADER_INFO_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderInfoSpecVersion   = VK_AMD_SHADER_INFO_SPEC_VERSION;

  //=== VK_KHR_dynamic_rendering ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_dynamic_rendering extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDynamicRenderingExtensionName = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_dynamic_rendering extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDynamicRenderingSpecVersion = VK_KHR_DYNAMIC_RENDERING_SPEC_VERSION;

  //=== VK_AMD_shader_image_load_store_lod ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderImageLoadStoreLodExtensionName = VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderImageLoadStoreLodSpecVersion   = VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto GGPStreamDescriptorSurfaceExtensionName = VK_GGP_STREAM_DESCRIPTOR_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto GGPStreamDescriptorSurfaceSpecVersion   = VK_GGP_STREAM_DESCRIPTOR_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_corner_sampled_image ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCornerSampledImageExtensionName = VK_NV_CORNER_SAMPLED_IMAGE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCornerSampledImageSpecVersion   = VK_NV_CORNER_SAMPLED_IMAGE_SPEC_VERSION;

  //=== VK_KHR_multiview ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_multiview extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMultiviewExtensionName = VK_KHR_MULTIVIEW_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_multiview extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMultiviewSpecVersion = VK_KHR_MULTIVIEW_SPEC_VERSION;

  //=== VK_IMG_format_pvrtc ===
  VULKAN_HPP_DEPRECATED( "The VK_IMG_format_pvrtc extension has been deprecated." )
  VULKAN_HPP_CONSTEXPR_INLINE auto IMGFormatPvrtcExtensionName = VK_IMG_FORMAT_PVRTC_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_IMG_format_pvrtc extension has been deprecated." )
  VULKAN_HPP_CONSTEXPR_INLINE auto IMGFormatPvrtcSpecVersion = VK_IMG_FORMAT_PVRTC_SPEC_VERSION;

  //=== VK_NV_external_memory_capabilities ===
  VULKAN_HPP_DEPRECATED( "The VK_NV_external_memory_capabilities extension has been deprecated by VK_KHR_external_memory_capabilities." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExternalMemoryCapabilitiesExtensionName = VK_NV_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_NV_external_memory_capabilities extension has been deprecated by VK_KHR_external_memory_capabilities." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExternalMemoryCapabilitiesSpecVersion = VK_NV_EXTERNAL_MEMORY_CAPABILITIES_SPEC_VERSION;

  //=== VK_NV_external_memory ===
  VULKAN_HPP_DEPRECATED( "The VK_NV_external_memory extension has been deprecated by VK_KHR_external_memory." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExternalMemoryExtensionName = VK_NV_EXTERNAL_MEMORY_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_NV_external_memory extension has been deprecated by VK_KHR_external_memory." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExternalMemorySpecVersion = VK_NV_EXTERNAL_MEMORY_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_external_memory_win32 ===
  VULKAN_HPP_DEPRECATED( "The VK_NV_external_memory_win32 extension has been deprecated by VK_KHR_external_memory_win32." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExternalMemoryWin32ExtensionName = VK_NV_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_NV_external_memory_win32 extension has been deprecated by VK_KHR_external_memory_win32." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExternalMemoryWin32SpecVersion = VK_NV_EXTERNAL_MEMORY_WIN32_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_win32_keyed_mutex ===
  VULKAN_HPP_DEPRECATED( "The VK_NV_win32_keyed_mutex extension has been promoted to VK_KHR_win32_keyed_mutex." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVWin32KeyedMutexExtensionName = VK_NV_WIN32_KEYED_MUTEX_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_NV_win32_keyed_mutex extension has been promoted to VK_KHR_win32_keyed_mutex." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVWin32KeyedMutexSpecVersion = VK_NV_WIN32_KEYED_MUTEX_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_get_physical_device_properties2 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_get_physical_device_properties2 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGetPhysicalDeviceProperties2ExtensionName = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_get_physical_device_properties2 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGetPhysicalDeviceProperties2SpecVersion = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_SPEC_VERSION;

  //=== VK_KHR_device_group ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_device_group extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDeviceGroupExtensionName = VK_KHR_DEVICE_GROUP_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_device_group extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDeviceGroupSpecVersion = VK_KHR_DEVICE_GROUP_SPEC_VERSION;

  //=== VK_EXT_validation_flags ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_validation_flags extension has been deprecated by VK_EXT_layer_settings." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTValidationFlagsExtensionName = VK_EXT_VALIDATION_FLAGS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_validation_flags extension has been deprecated by VK_EXT_layer_settings." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTValidationFlagsSpecVersion = VK_EXT_VALIDATION_FLAGS_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NNViSurfaceExtensionName = VK_NN_VI_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NNViSurfaceSpecVersion   = VK_NN_VI_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_KHR_shader_draw_parameters ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_draw_parameters extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderDrawParametersExtensionName = VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_draw_parameters extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderDrawParametersSpecVersion = VK_KHR_SHADER_DRAW_PARAMETERS_SPEC_VERSION;

  //=== VK_EXT_shader_subgroup_ballot ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_shader_subgroup_ballot extension has been deprecated by VK_VERSION_1_2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderSubgroupBallotExtensionName = VK_EXT_SHADER_SUBGROUP_BALLOT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_shader_subgroup_ballot extension has been deprecated by VK_VERSION_1_2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderSubgroupBallotSpecVersion = VK_EXT_SHADER_SUBGROUP_BALLOT_SPEC_VERSION;

  //=== VK_EXT_shader_subgroup_vote ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_shader_subgroup_vote extension has been deprecated by VK_VERSION_1_1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderSubgroupVoteExtensionName = VK_EXT_SHADER_SUBGROUP_VOTE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_shader_subgroup_vote extension has been deprecated by VK_VERSION_1_1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderSubgroupVoteSpecVersion = VK_EXT_SHADER_SUBGROUP_VOTE_SPEC_VERSION;

  //=== VK_EXT_texture_compression_astc_hdr ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_texture_compression_astc_hdr extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTTextureCompressionAstcHdrExtensionName = VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_texture_compression_astc_hdr extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTTextureCompressionAstcHdrSpecVersion = VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_SPEC_VERSION;

  //=== VK_EXT_astc_decode_mode ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAstcDecodeModeExtensionName = VK_EXT_ASTC_DECODE_MODE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAstcDecodeModeSpecVersion   = VK_EXT_ASTC_DECODE_MODE_SPEC_VERSION;

  //=== VK_EXT_pipeline_robustness ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineRobustnessExtensionName = VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineRobustnessSpecVersion   = VK_EXT_PIPELINE_ROBUSTNESS_SPEC_VERSION;

  //=== VK_KHR_maintenance1 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_maintenance1 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance1ExtensionName = VK_KHR_MAINTENANCE_1_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_maintenance1 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance1SpecVersion = VK_KHR_MAINTENANCE_1_SPEC_VERSION;

  //=== VK_KHR_device_group_creation ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_device_group_creation extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDeviceGroupCreationExtensionName = VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_device_group_creation extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDeviceGroupCreationSpecVersion = VK_KHR_DEVICE_GROUP_CREATION_SPEC_VERSION;

  //=== VK_KHR_external_memory_capabilities ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_memory_capabilities extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalMemoryCapabilitiesExtensionName = VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_memory_capabilities extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalMemoryCapabilitiesSpecVersion = VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_SPEC_VERSION;

  //=== VK_KHR_external_memory ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_memory extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalMemoryExtensionName = VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_memory extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalMemorySpecVersion = VK_KHR_EXTERNAL_MEMORY_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_memory_win32 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalMemoryWin32ExtensionName = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalMemoryWin32SpecVersion   = VK_KHR_EXTERNAL_MEMORY_WIN32_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_memory_fd ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalMemoryFdExtensionName = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalMemoryFdSpecVersion   = VK_KHR_EXTERNAL_MEMORY_FD_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_keyed_mutex ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRWin32KeyedMutexExtensionName = VK_KHR_WIN32_KEYED_MUTEX_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRWin32KeyedMutexSpecVersion   = VK_KHR_WIN32_KEYED_MUTEX_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_semaphore_capabilities ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_semaphore_capabilities extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalSemaphoreCapabilitiesExtensionName = VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_semaphore_capabilities extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalSemaphoreCapabilitiesSpecVersion = VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_SPEC_VERSION;

  //=== VK_KHR_external_semaphore ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_semaphore extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalSemaphoreExtensionName = VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_semaphore extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalSemaphoreSpecVersion = VK_KHR_EXTERNAL_SEMAPHORE_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_semaphore_win32 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalSemaphoreWin32ExtensionName = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalSemaphoreWin32SpecVersion   = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_semaphore_fd ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalSemaphoreFdExtensionName = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalSemaphoreFdSpecVersion   = VK_KHR_EXTERNAL_SEMAPHORE_FD_SPEC_VERSION;

  //=== VK_KHR_push_descriptor ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPushDescriptorExtensionName = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPushDescriptorSpecVersion   = VK_KHR_PUSH_DESCRIPTOR_SPEC_VERSION;

  //=== VK_EXT_conditional_rendering ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTConditionalRenderingExtensionName = VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTConditionalRenderingSpecVersion   = VK_EXT_CONDITIONAL_RENDERING_SPEC_VERSION;

  //=== VK_KHR_shader_float16_int8 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_float16_int8 extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderFloat16Int8ExtensionName = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_float16_int8 extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderFloat16Int8SpecVersion = VK_KHR_SHADER_FLOAT16_INT8_SPEC_VERSION;

  //=== VK_KHR_16bit_storage ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_16bit_storage extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHR16BitStorageExtensionName = VK_KHR_16BIT_STORAGE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_16bit_storage extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHR16BitStorageSpecVersion = VK_KHR_16BIT_STORAGE_SPEC_VERSION;

  //=== VK_KHR_incremental_present ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRIncrementalPresentExtensionName = VK_KHR_INCREMENTAL_PRESENT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRIncrementalPresentSpecVersion   = VK_KHR_INCREMENTAL_PRESENT_SPEC_VERSION;

  //=== VK_KHR_descriptor_update_template ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_descriptor_update_template extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDescriptorUpdateTemplateExtensionName = VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_descriptor_update_template extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDescriptorUpdateTemplateSpecVersion = VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_SPEC_VERSION;

  //=== VK_NV_clip_space_w_scaling ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVClipSpaceWScalingExtensionName = VK_NV_CLIP_SPACE_W_SCALING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVClipSpaceWScalingSpecVersion   = VK_NV_CLIP_SPACE_W_SCALING_SPEC_VERSION;

  //=== VK_EXT_direct_mode_display ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDirectModeDisplayExtensionName = VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDirectModeDisplaySpecVersion   = VK_EXT_DIRECT_MODE_DISPLAY_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
  //=== VK_EXT_acquire_xlib_display ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAcquireXlibDisplayExtensionName = VK_EXT_ACQUIRE_XLIB_DISPLAY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAcquireXlibDisplaySpecVersion   = VK_EXT_ACQUIRE_XLIB_DISPLAY_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

  //=== VK_EXT_display_surface_counter ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDisplaySurfaceCounterExtensionName = VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDisplaySurfaceCounterSpecVersion   = VK_EXT_DISPLAY_SURFACE_COUNTER_SPEC_VERSION;

  //=== VK_EXT_display_control ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDisplayControlExtensionName = VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDisplayControlSpecVersion   = VK_EXT_DISPLAY_CONTROL_SPEC_VERSION;

  //=== VK_GOOGLE_display_timing ===
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLEDisplayTimingExtensionName = VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLEDisplayTimingSpecVersion   = VK_GOOGLE_DISPLAY_TIMING_SPEC_VERSION;

  //=== VK_NV_sample_mask_override_coverage ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVSampleMaskOverrideCoverageExtensionName = VK_NV_SAMPLE_MASK_OVERRIDE_COVERAGE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVSampleMaskOverrideCoverageSpecVersion   = VK_NV_SAMPLE_MASK_OVERRIDE_COVERAGE_SPEC_VERSION;

  //=== VK_NV_geometry_shader_passthrough ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVGeometryShaderPassthroughExtensionName = VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVGeometryShaderPassthroughSpecVersion   = VK_NV_GEOMETRY_SHADER_PASSTHROUGH_SPEC_VERSION;

  //=== VK_NV_viewport_array2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVViewportArray2ExtensionName = VK_NV_VIEWPORT_ARRAY_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVViewportArray2SpecVersion   = VK_NV_VIEWPORT_ARRAY_2_SPEC_VERSION;

  //=== VK_NVX_multiview_per_view_attributes ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVXMultiviewPerViewAttributesExtensionName = VK_NVX_MULTIVIEW_PER_VIEW_ATTRIBUTES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVXMultiviewPerViewAttributesSpecVersion   = VK_NVX_MULTIVIEW_PER_VIEW_ATTRIBUTES_SPEC_VERSION;

  //=== VK_NV_viewport_swizzle ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVViewportSwizzleExtensionName = VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVViewportSwizzleSpecVersion   = VK_NV_VIEWPORT_SWIZZLE_SPEC_VERSION;

  //=== VK_EXT_discard_rectangles ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDiscardRectanglesExtensionName = VK_EXT_DISCARD_RECTANGLES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDiscardRectanglesSpecVersion   = VK_EXT_DISCARD_RECTANGLES_SPEC_VERSION;

  //=== VK_EXT_conservative_rasterization ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTConservativeRasterizationExtensionName = VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTConservativeRasterizationSpecVersion   = VK_EXT_CONSERVATIVE_RASTERIZATION_SPEC_VERSION;

  //=== VK_EXT_depth_clip_enable ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthClipEnableExtensionName = VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthClipEnableSpecVersion   = VK_EXT_DEPTH_CLIP_ENABLE_SPEC_VERSION;

  //=== VK_EXT_swapchain_colorspace ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSwapchainColorSpaceExtensionName = VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSwapchainColorSpaceSpecVersion   = VK_EXT_SWAPCHAIN_COLOR_SPACE_SPEC_VERSION;

  //=== VK_EXT_hdr_metadata ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTHdrMetadataExtensionName = VK_EXT_HDR_METADATA_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTHdrMetadataSpecVersion   = VK_EXT_HDR_METADATA_SPEC_VERSION;

  //=== VK_KHR_imageless_framebuffer ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_imageless_framebuffer extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRImagelessFramebufferExtensionName = VK_KHR_IMAGELESS_FRAMEBUFFER_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_imageless_framebuffer extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRImagelessFramebufferSpecVersion = VK_KHR_IMAGELESS_FRAMEBUFFER_SPEC_VERSION;

  //=== VK_KHR_create_renderpass2 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_create_renderpass2 extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRCreateRenderpass2ExtensionName = VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_create_renderpass2 extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRCreateRenderpass2SpecVersion = VK_KHR_CREATE_RENDERPASS_2_SPEC_VERSION;

  //=== VK_IMG_relaxed_line_rasterization ===
  VULKAN_HPP_CONSTEXPR_INLINE auto IMGRelaxedLineRasterizationExtensionName = VK_IMG_RELAXED_LINE_RASTERIZATION_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto IMGRelaxedLineRasterizationSpecVersion   = VK_IMG_RELAXED_LINE_RASTERIZATION_SPEC_VERSION;

  //=== VK_KHR_shared_presentable_image ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSharedPresentableImageExtensionName = VK_KHR_SHARED_PRESENTABLE_IMAGE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSharedPresentableImageSpecVersion   = VK_KHR_SHARED_PRESENTABLE_IMAGE_SPEC_VERSION;

  //=== VK_KHR_external_fence_capabilities ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_fence_capabilities extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalFenceCapabilitiesExtensionName = VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_fence_capabilities extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalFenceCapabilitiesSpecVersion = VK_KHR_EXTERNAL_FENCE_CAPABILITIES_SPEC_VERSION;

  //=== VK_KHR_external_fence ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_fence extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalFenceExtensionName = VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_external_fence extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalFenceSpecVersion = VK_KHR_EXTERNAL_FENCE_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_fence_win32 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalFenceWin32ExtensionName = VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalFenceWin32SpecVersion   = VK_KHR_EXTERNAL_FENCE_WIN32_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_fence_fd ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalFenceFdExtensionName = VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRExternalFenceFdSpecVersion   = VK_KHR_EXTERNAL_FENCE_FD_SPEC_VERSION;

  //=== VK_KHR_performance_query ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPerformanceQueryExtensionName = VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPerformanceQuerySpecVersion   = VK_KHR_PERFORMANCE_QUERY_SPEC_VERSION;

  //=== VK_KHR_maintenance2 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_maintenance2 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance2ExtensionName = VK_KHR_MAINTENANCE_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_maintenance2 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance2SpecVersion = VK_KHR_MAINTENANCE_2_SPEC_VERSION;

  //=== VK_KHR_get_surface_capabilities2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGetSurfaceCapabilities2ExtensionName = VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGetSurfaceCapabilities2SpecVersion   = VK_KHR_GET_SURFACE_CAPABILITIES_2_SPEC_VERSION;

  //=== VK_KHR_variable_pointers ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_variable_pointers extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVariablePointersExtensionName = VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_variable_pointers extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVariablePointersSpecVersion = VK_KHR_VARIABLE_POINTERS_SPEC_VERSION;

  //=== VK_KHR_get_display_properties2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGetDisplayProperties2ExtensionName = VK_KHR_GET_DISPLAY_PROPERTIES_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGetDisplayProperties2SpecVersion   = VK_KHR_GET_DISPLAY_PROPERTIES_2_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===
  VULKAN_HPP_DEPRECATED( "The VK_MVK_ios_surface extension has been deprecated by VK_EXT_metal_surface." )
  VULKAN_HPP_CONSTEXPR_INLINE auto MVKIosSurfaceExtensionName = VK_MVK_IOS_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_MVK_ios_surface extension has been deprecated by VK_EXT_metal_surface." )
  VULKAN_HPP_CONSTEXPR_INLINE auto MVKIosSurfaceSpecVersion = VK_MVK_IOS_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===
  VULKAN_HPP_DEPRECATED( "The VK_MVK_macos_surface extension has been deprecated by VK_EXT_metal_surface." )
  VULKAN_HPP_CONSTEXPR_INLINE auto MVKMacosSurfaceExtensionName = VK_MVK_MACOS_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_MVK_macos_surface extension has been deprecated by VK_EXT_metal_surface." )
  VULKAN_HPP_CONSTEXPR_INLINE auto MVKMacosSurfaceSpecVersion = VK_MVK_MACOS_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_external_memory_dma_buf ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExternalMemoryDmaBufExtensionName = VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExternalMemoryDmaBufSpecVersion   = VK_EXT_EXTERNAL_MEMORY_DMA_BUF_SPEC_VERSION;

  //=== VK_EXT_queue_family_foreign ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTQueueFamilyForeignExtensionName = VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTQueueFamilyForeignSpecVersion   = VK_EXT_QUEUE_FAMILY_FOREIGN_SPEC_VERSION;

  //=== VK_KHR_dedicated_allocation ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_dedicated_allocation extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDedicatedAllocationExtensionName = VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_dedicated_allocation extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDedicatedAllocationSpecVersion = VK_KHR_DEDICATED_ALLOCATION_SPEC_VERSION;

  //=== VK_EXT_debug_utils ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDebugUtilsExtensionName = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDebugUtilsSpecVersion   = VK_EXT_DEBUG_UTILS_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_memory_android_hardware_buffer ===
  VULKAN_HPP_CONSTEXPR_INLINE auto ANDROIDExternalMemoryAndroidHardwareBufferExtensionName = VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto ANDROIDExternalMemoryAndroidHardwareBufferSpecVersion   = VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  //=== VK_EXT_sampler_filter_minmax ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_sampler_filter_minmax extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSamplerFilterMinmaxExtensionName = VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_sampler_filter_minmax extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSamplerFilterMinmaxSpecVersion = VK_EXT_SAMPLER_FILTER_MINMAX_SPEC_VERSION;

  //=== VK_KHR_storage_buffer_storage_class ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_storage_buffer_storage_class extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRStorageBufferStorageClassExtensionName = VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_storage_buffer_storage_class extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRStorageBufferStorageClassSpecVersion = VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_SPEC_VERSION;

  //=== VK_AMD_gpu_shader_int16 ===
  VULKAN_HPP_DEPRECATED( "The VK_AMD_gpu_shader_int16 extension has been deprecated by VK_KHR_shader_float16_int8." )
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDGpuShaderInt16ExtensionName = VK_AMD_GPU_SHADER_INT16_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_AMD_gpu_shader_int16 extension has been deprecated by VK_KHR_shader_float16_int8." )
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDGpuShaderInt16SpecVersion = VK_AMD_GPU_SHADER_INT16_SPEC_VERSION;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_shader_enqueue ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDXShaderEnqueueExtensionName = VK_AMDX_SHADER_ENQUEUE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDXShaderEnqueueSpecVersion   = VK_AMDX_SHADER_ENQUEUE_SPEC_VERSION;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_AMD_mixed_attachment_samples ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDMixedAttachmentSamplesExtensionName = VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDMixedAttachmentSamplesSpecVersion   = VK_AMD_MIXED_ATTACHMENT_SAMPLES_SPEC_VERSION;

  //=== VK_AMD_shader_fragment_mask ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderFragmentMaskExtensionName = VK_AMD_SHADER_FRAGMENT_MASK_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderFragmentMaskSpecVersion   = VK_AMD_SHADER_FRAGMENT_MASK_SPEC_VERSION;

  //=== VK_EXT_inline_uniform_block ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_inline_uniform_block extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTInlineUniformBlockExtensionName = VK_EXT_INLINE_UNIFORM_BLOCK_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_inline_uniform_block extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTInlineUniformBlockSpecVersion = VK_EXT_INLINE_UNIFORM_BLOCK_SPEC_VERSION;

  //=== VK_EXT_shader_stencil_export ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderStencilExportExtensionName = VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderStencilExportSpecVersion   = VK_EXT_SHADER_STENCIL_EXPORT_SPEC_VERSION;

  //=== VK_EXT_sample_locations ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSampleLocationsExtensionName = VK_EXT_SAMPLE_LOCATIONS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSampleLocationsSpecVersion   = VK_EXT_SAMPLE_LOCATIONS_SPEC_VERSION;

  //=== VK_KHR_relaxed_block_layout ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_relaxed_block_layout extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRelaxedBlockLayoutExtensionName = VK_KHR_RELAXED_BLOCK_LAYOUT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_relaxed_block_layout extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRelaxedBlockLayoutSpecVersion = VK_KHR_RELAXED_BLOCK_LAYOUT_SPEC_VERSION;

  //=== VK_KHR_get_memory_requirements2 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_get_memory_requirements2 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGetMemoryRequirements2ExtensionName = VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_get_memory_requirements2 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGetMemoryRequirements2SpecVersion = VK_KHR_GET_MEMORY_REQUIREMENTS_2_SPEC_VERSION;

  //=== VK_KHR_image_format_list ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_image_format_list extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRImageFormatListExtensionName = VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_image_format_list extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRImageFormatListSpecVersion = VK_KHR_IMAGE_FORMAT_LIST_SPEC_VERSION;

  //=== VK_EXT_blend_operation_advanced ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTBlendOperationAdvancedExtensionName = VK_EXT_BLEND_OPERATION_ADVANCED_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTBlendOperationAdvancedSpecVersion   = VK_EXT_BLEND_OPERATION_ADVANCED_SPEC_VERSION;

  //=== VK_NV_fragment_coverage_to_color ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFragmentCoverageToColorExtensionName = VK_NV_FRAGMENT_COVERAGE_TO_COLOR_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFragmentCoverageToColorSpecVersion   = VK_NV_FRAGMENT_COVERAGE_TO_COLOR_SPEC_VERSION;

  //=== VK_KHR_acceleration_structure ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRAccelerationStructureExtensionName = VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRAccelerationStructureSpecVersion   = VK_KHR_ACCELERATION_STRUCTURE_SPEC_VERSION;

  //=== VK_KHR_ray_tracing_pipeline ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRayTracingPipelineExtensionName = VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRayTracingPipelineSpecVersion   = VK_KHR_RAY_TRACING_PIPELINE_SPEC_VERSION;

  //=== VK_KHR_ray_query ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRayQueryExtensionName = VK_KHR_RAY_QUERY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRayQuerySpecVersion   = VK_KHR_RAY_QUERY_SPEC_VERSION;

  //=== VK_NV_framebuffer_mixed_samples ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFramebufferMixedSamplesExtensionName = VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFramebufferMixedSamplesSpecVersion   = VK_NV_FRAMEBUFFER_MIXED_SAMPLES_SPEC_VERSION;

  //=== VK_NV_fill_rectangle ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFillRectangleExtensionName = VK_NV_FILL_RECTANGLE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFillRectangleSpecVersion   = VK_NV_FILL_RECTANGLE_SPEC_VERSION;

  //=== VK_NV_shader_sm_builtins ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVShaderSmBuiltinsExtensionName = VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVShaderSmBuiltinsSpecVersion   = VK_NV_SHADER_SM_BUILTINS_SPEC_VERSION;

  //=== VK_EXT_post_depth_coverage ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPostDepthCoverageExtensionName = VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPostDepthCoverageSpecVersion   = VK_EXT_POST_DEPTH_COVERAGE_SPEC_VERSION;

  //=== VK_KHR_sampler_ycbcr_conversion ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_sampler_ycbcr_conversion extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSamplerYcbcrConversionExtensionName = VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_sampler_ycbcr_conversion extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSamplerYcbcrConversionSpecVersion = VK_KHR_SAMPLER_YCBCR_CONVERSION_SPEC_VERSION;

  //=== VK_KHR_bind_memory2 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_bind_memory2 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRBindMemory2ExtensionName = VK_KHR_BIND_MEMORY_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_bind_memory2 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRBindMemory2SpecVersion = VK_KHR_BIND_MEMORY_2_SPEC_VERSION;

  //=== VK_EXT_image_drm_format_modifier ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageDrmFormatModifierExtensionName = VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageDrmFormatModifierSpecVersion   = VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_SPEC_VERSION;

  //=== VK_EXT_validation_cache ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTValidationCacheExtensionName = VK_EXT_VALIDATION_CACHE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTValidationCacheSpecVersion   = VK_EXT_VALIDATION_CACHE_SPEC_VERSION;

  //=== VK_EXT_descriptor_indexing ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_descriptor_indexing extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDescriptorIndexingExtensionName = VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_descriptor_indexing extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDescriptorIndexingSpecVersion = VK_EXT_DESCRIPTOR_INDEXING_SPEC_VERSION;

  //=== VK_EXT_shader_viewport_index_layer ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_shader_viewport_index_layer extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderViewportIndexLayerExtensionName = VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_shader_viewport_index_layer extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderViewportIndexLayerSpecVersion = VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_SPEC_VERSION;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_portability_subset ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPortabilitySubsetExtensionName = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPortabilitySubsetSpecVersion   = VK_KHR_PORTABILITY_SUBSET_SPEC_VERSION;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_shading_rate_image ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVShadingRateImageExtensionName = VK_NV_SHADING_RATE_IMAGE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVShadingRateImageSpecVersion   = VK_NV_SHADING_RATE_IMAGE_SPEC_VERSION;

  //=== VK_NV_ray_tracing ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVRayTracingExtensionName = VK_NV_RAY_TRACING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVRayTracingSpecVersion   = VK_NV_RAY_TRACING_SPEC_VERSION;

  //=== VK_NV_representative_fragment_test ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVRepresentativeFragmentTestExtensionName = VK_NV_REPRESENTATIVE_FRAGMENT_TEST_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVRepresentativeFragmentTestSpecVersion   = VK_NV_REPRESENTATIVE_FRAGMENT_TEST_SPEC_VERSION;

  //=== VK_KHR_maintenance3 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_maintenance3 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance3ExtensionName = VK_KHR_MAINTENANCE_3_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_maintenance3 extension has been promoted to core in version 1.1." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance3SpecVersion = VK_KHR_MAINTENANCE_3_SPEC_VERSION;

  //=== VK_KHR_draw_indirect_count ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_draw_indirect_count extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDrawIndirectCountExtensionName = VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_draw_indirect_count extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDrawIndirectCountSpecVersion = VK_KHR_DRAW_INDIRECT_COUNT_SPEC_VERSION;

  //=== VK_EXT_filter_cubic ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFilterCubicExtensionName = VK_EXT_FILTER_CUBIC_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFilterCubicSpecVersion   = VK_EXT_FILTER_CUBIC_SPEC_VERSION;

  //=== VK_QCOM_render_pass_shader_resolve ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMRenderPassShaderResolveExtensionName = VK_QCOM_RENDER_PASS_SHADER_RESOLVE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMRenderPassShaderResolveSpecVersion   = VK_QCOM_RENDER_PASS_SHADER_RESOLVE_SPEC_VERSION;

  //=== VK_EXT_global_priority ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_global_priority extension has been promoted to VK_KHR_global_priority." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTGlobalPriorityExtensionName = VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_global_priority extension has been promoted to VK_KHR_global_priority." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTGlobalPrioritySpecVersion = VK_EXT_GLOBAL_PRIORITY_SPEC_VERSION;

  //=== VK_KHR_shader_subgroup_extended_types ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_subgroup_extended_types extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderSubgroupExtendedTypesExtensionName = VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_subgroup_extended_types extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderSubgroupExtendedTypesSpecVersion = VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_SPEC_VERSION;

  //=== VK_KHR_8bit_storage ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_8bit_storage extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHR8BitStorageExtensionName = VK_KHR_8BIT_STORAGE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_8bit_storage extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHR8BitStorageSpecVersion = VK_KHR_8BIT_STORAGE_SPEC_VERSION;

  //=== VK_EXT_external_memory_host ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExternalMemoryHostExtensionName = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExternalMemoryHostSpecVersion   = VK_EXT_EXTERNAL_MEMORY_HOST_SPEC_VERSION;

  //=== VK_AMD_buffer_marker ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDBufferMarkerExtensionName = VK_AMD_BUFFER_MARKER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDBufferMarkerSpecVersion   = VK_AMD_BUFFER_MARKER_SPEC_VERSION;

  //=== VK_KHR_shader_atomic_int64 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_atomic_int64 extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderAtomicInt64ExtensionName = VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_atomic_int64 extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderAtomicInt64SpecVersion = VK_KHR_SHADER_ATOMIC_INT64_SPEC_VERSION;

  //=== VK_KHR_shader_clock ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderClockExtensionName = VK_KHR_SHADER_CLOCK_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderClockSpecVersion   = VK_KHR_SHADER_CLOCK_SPEC_VERSION;

  //=== VK_AMD_pipeline_compiler_control ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDPipelineCompilerControlExtensionName = VK_AMD_PIPELINE_COMPILER_CONTROL_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDPipelineCompilerControlSpecVersion   = VK_AMD_PIPELINE_COMPILER_CONTROL_SPEC_VERSION;

  //=== VK_EXT_calibrated_timestamps ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_calibrated_timestamps extension has been promoted to VK_KHR_calibrated_timestamps." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTCalibratedTimestampsExtensionName = VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_calibrated_timestamps extension has been promoted to VK_KHR_calibrated_timestamps." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTCalibratedTimestampsSpecVersion = VK_EXT_CALIBRATED_TIMESTAMPS_SPEC_VERSION;

  //=== VK_AMD_shader_core_properties ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderCorePropertiesExtensionName = VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderCorePropertiesSpecVersion   = VK_AMD_SHADER_CORE_PROPERTIES_SPEC_VERSION;

  //=== VK_KHR_video_decode_h265 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoDecodeH265ExtensionName = VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoDecodeH265SpecVersion   = VK_KHR_VIDEO_DECODE_H265_SPEC_VERSION;

  //=== VK_KHR_global_priority ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGlobalPriorityExtensionName = VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRGlobalPrioritySpecVersion   = VK_KHR_GLOBAL_PRIORITY_SPEC_VERSION;

  //=== VK_AMD_memory_overallocation_behavior ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDMemoryOverallocationBehaviorExtensionName = VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDMemoryOverallocationBehaviorSpecVersion   = VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR_SPEC_VERSION;

  //=== VK_EXT_vertex_attribute_divisor ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_vertex_attribute_divisor extension has been promoted to VK_KHR_vertex_attribute_divisor." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTVertexAttributeDivisorExtensionName = VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_vertex_attribute_divisor extension has been promoted to VK_KHR_vertex_attribute_divisor." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTVertexAttributeDivisorSpecVersion = VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_frame_token ===
  VULKAN_HPP_CONSTEXPR_INLINE auto GGPFrameTokenExtensionName = VK_GGP_FRAME_TOKEN_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto GGPFrameTokenSpecVersion   = VK_GGP_FRAME_TOKEN_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_EXT_pipeline_creation_feedback ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_pipeline_creation_feedback extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineCreationFeedbackExtensionName = VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_pipeline_creation_feedback extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineCreationFeedbackSpecVersion = VK_EXT_PIPELINE_CREATION_FEEDBACK_SPEC_VERSION;

  //=== VK_KHR_driver_properties ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_driver_properties extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDriverPropertiesExtensionName = VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_driver_properties extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDriverPropertiesSpecVersion = VK_KHR_DRIVER_PROPERTIES_SPEC_VERSION;

  //=== VK_KHR_shader_float_controls ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_float_controls extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderFloatControlsExtensionName = VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_float_controls extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderFloatControlsSpecVersion = VK_KHR_SHADER_FLOAT_CONTROLS_SPEC_VERSION;

  //=== VK_NV_shader_subgroup_partitioned ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVShaderSubgroupPartitionedExtensionName = VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVShaderSubgroupPartitionedSpecVersion   = VK_NV_SHADER_SUBGROUP_PARTITIONED_SPEC_VERSION;

  //=== VK_KHR_depth_stencil_resolve ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_depth_stencil_resolve extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDepthStencilResolveExtensionName = VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_depth_stencil_resolve extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDepthStencilResolveSpecVersion = VK_KHR_DEPTH_STENCIL_RESOLVE_SPEC_VERSION;

  //=== VK_KHR_swapchain_mutable_format ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSwapchainMutableFormatExtensionName = VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSwapchainMutableFormatSpecVersion   = VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_SPEC_VERSION;

  //=== VK_NV_compute_shader_derivatives ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVComputeShaderDerivativesExtensionName = VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVComputeShaderDerivativesSpecVersion   = VK_NV_COMPUTE_SHADER_DERIVATIVES_SPEC_VERSION;

  //=== VK_NV_mesh_shader ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVMeshShaderExtensionName = VK_NV_MESH_SHADER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVMeshShaderSpecVersion   = VK_NV_MESH_SHADER_SPEC_VERSION;

  //=== VK_NV_fragment_shader_barycentric ===
  VULKAN_HPP_DEPRECATED( "The VK_NV_fragment_shader_barycentric extension has been promoted to VK_KHR_fragment_shader_barycentric." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFragmentShaderBarycentricExtensionName = VK_NV_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_NV_fragment_shader_barycentric extension has been promoted to VK_KHR_fragment_shader_barycentric." )
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFragmentShaderBarycentricSpecVersion = VK_NV_FRAGMENT_SHADER_BARYCENTRIC_SPEC_VERSION;

  //=== VK_NV_shader_image_footprint ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVShaderImageFootprintExtensionName = VK_NV_SHADER_IMAGE_FOOTPRINT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVShaderImageFootprintSpecVersion   = VK_NV_SHADER_IMAGE_FOOTPRINT_SPEC_VERSION;

  //=== VK_NV_scissor_exclusive ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVScissorExclusiveExtensionName = VK_NV_SCISSOR_EXCLUSIVE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVScissorExclusiveSpecVersion   = VK_NV_SCISSOR_EXCLUSIVE_SPEC_VERSION;

  //=== VK_NV_device_diagnostic_checkpoints ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDeviceDiagnosticCheckpointsExtensionName = VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDeviceDiagnosticCheckpointsSpecVersion   = VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_SPEC_VERSION;

  //=== VK_KHR_timeline_semaphore ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_timeline_semaphore extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRTimelineSemaphoreExtensionName = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_timeline_semaphore extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRTimelineSemaphoreSpecVersion = VK_KHR_TIMELINE_SEMAPHORE_SPEC_VERSION;

  //=== VK_INTEL_shader_integer_functions2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto INTELShaderIntegerFunctions2ExtensionName = VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto INTELShaderIntegerFunctions2SpecVersion   = VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_SPEC_VERSION;

  //=== VK_INTEL_performance_query ===
  VULKAN_HPP_CONSTEXPR_INLINE auto INTELPerformanceQueryExtensionName = VK_INTEL_PERFORMANCE_QUERY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto INTELPerformanceQuerySpecVersion   = VK_INTEL_PERFORMANCE_QUERY_SPEC_VERSION;

  //=== VK_KHR_vulkan_memory_model ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_vulkan_memory_model extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVulkanMemoryModelExtensionName = VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_vulkan_memory_model extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVulkanMemoryModelSpecVersion = VK_KHR_VULKAN_MEMORY_MODEL_SPEC_VERSION;

  //=== VK_EXT_pci_bus_info ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPciBusInfoExtensionName = VK_EXT_PCI_BUS_INFO_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPciBusInfoSpecVersion   = VK_EXT_PCI_BUS_INFO_SPEC_VERSION;

  //=== VK_AMD_display_native_hdr ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDDisplayNativeHdrExtensionName = VK_AMD_DISPLAY_NATIVE_HDR_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDDisplayNativeHdrSpecVersion   = VK_AMD_DISPLAY_NATIVE_HDR_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto FUCHSIAImagepipeSurfaceExtensionName = VK_FUCHSIA_IMAGEPIPE_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto FUCHSIAImagepipeSurfaceSpecVersion   = VK_FUCHSIA_IMAGEPIPE_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_KHR_shader_terminate_invocation ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_terminate_invocation extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderTerminateInvocationExtensionName = VK_KHR_SHADER_TERMINATE_INVOCATION_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_terminate_invocation extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderTerminateInvocationSpecVersion = VK_KHR_SHADER_TERMINATE_INVOCATION_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMetalSurfaceExtensionName = VK_EXT_METAL_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMetalSurfaceSpecVersion   = VK_EXT_METAL_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_fragment_density_map ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFragmentDensityMapExtensionName = VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFragmentDensityMapSpecVersion   = VK_EXT_FRAGMENT_DENSITY_MAP_SPEC_VERSION;

  //=== VK_EXT_scalar_block_layout ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_scalar_block_layout extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTScalarBlockLayoutExtensionName = VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_scalar_block_layout extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTScalarBlockLayoutSpecVersion = VK_EXT_SCALAR_BLOCK_LAYOUT_SPEC_VERSION;

  //=== VK_GOOGLE_hlsl_functionality1 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLEHlslFunctionality1ExtensionName = VK_GOOGLE_HLSL_FUNCTIONALITY_1_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLEHlslFunctionality1SpecVersion   = VK_GOOGLE_HLSL_FUNCTIONALITY_1_SPEC_VERSION;

  //=== VK_GOOGLE_decorate_string ===
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLEDecorateStringExtensionName = VK_GOOGLE_DECORATE_STRING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLEDecorateStringSpecVersion   = VK_GOOGLE_DECORATE_STRING_SPEC_VERSION;

  //=== VK_EXT_subgroup_size_control ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_subgroup_size_control extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSubgroupSizeControlExtensionName = VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_subgroup_size_control extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSubgroupSizeControlSpecVersion = VK_EXT_SUBGROUP_SIZE_CONTROL_SPEC_VERSION;

  //=== VK_KHR_fragment_shading_rate ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRFragmentShadingRateExtensionName = VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRFragmentShadingRateSpecVersion   = VK_KHR_FRAGMENT_SHADING_RATE_SPEC_VERSION;

  //=== VK_AMD_shader_core_properties2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderCoreProperties2ExtensionName = VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderCoreProperties2SpecVersion   = VK_AMD_SHADER_CORE_PROPERTIES_2_SPEC_VERSION;

  //=== VK_AMD_device_coherent_memory ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDDeviceCoherentMemoryExtensionName = VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDDeviceCoherentMemorySpecVersion   = VK_AMD_DEVICE_COHERENT_MEMORY_SPEC_VERSION;

  //=== VK_EXT_shader_image_atomic_int64 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderImageAtomicInt64ExtensionName = VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderImageAtomicInt64SpecVersion   = VK_EXT_SHADER_IMAGE_ATOMIC_INT64_SPEC_VERSION;

  //=== VK_KHR_spirv_1_4 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_spirv_1_4 extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSpirv14ExtensionName = VK_KHR_SPIRV_1_4_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_spirv_1_4 extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSpirv14SpecVersion = VK_KHR_SPIRV_1_4_SPEC_VERSION;

  //=== VK_EXT_memory_budget ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMemoryBudgetExtensionName = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMemoryBudgetSpecVersion   = VK_EXT_MEMORY_BUDGET_SPEC_VERSION;

  //=== VK_EXT_memory_priority ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMemoryPriorityExtensionName = VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMemoryPrioritySpecVersion   = VK_EXT_MEMORY_PRIORITY_SPEC_VERSION;

  //=== VK_KHR_surface_protected_capabilities ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSurfaceProtectedCapabilitiesExtensionName = VK_KHR_SURFACE_PROTECTED_CAPABILITIES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSurfaceProtectedCapabilitiesSpecVersion   = VK_KHR_SURFACE_PROTECTED_CAPABILITIES_SPEC_VERSION;

  //=== VK_NV_dedicated_allocation_image_aliasing ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDedicatedAllocationImageAliasingExtensionName = VK_NV_DEDICATED_ALLOCATION_IMAGE_ALIASING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDedicatedAllocationImageAliasingSpecVersion   = VK_NV_DEDICATED_ALLOCATION_IMAGE_ALIASING_SPEC_VERSION;

  //=== VK_KHR_separate_depth_stencil_layouts ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_separate_depth_stencil_layouts extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSeparateDepthStencilLayoutsExtensionName = VK_KHR_SEPARATE_DEPTH_STENCIL_LAYOUTS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_separate_depth_stencil_layouts extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSeparateDepthStencilLayoutsSpecVersion = VK_KHR_SEPARATE_DEPTH_STENCIL_LAYOUTS_SPEC_VERSION;

  //=== VK_EXT_buffer_device_address ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_buffer_device_address extension has been deprecated by VK_KHR_buffer_device_address." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTBufferDeviceAddressExtensionName = VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_buffer_device_address extension has been deprecated by VK_KHR_buffer_device_address." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTBufferDeviceAddressSpecVersion = VK_EXT_BUFFER_DEVICE_ADDRESS_SPEC_VERSION;

  //=== VK_EXT_tooling_info ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_tooling_info extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTToolingInfoExtensionName = VK_EXT_TOOLING_INFO_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_tooling_info extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTToolingInfoSpecVersion = VK_EXT_TOOLING_INFO_SPEC_VERSION;

  //=== VK_EXT_separate_stencil_usage ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_separate_stencil_usage extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSeparateStencilUsageExtensionName = VK_EXT_SEPARATE_STENCIL_USAGE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_separate_stencil_usage extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSeparateStencilUsageSpecVersion = VK_EXT_SEPARATE_STENCIL_USAGE_SPEC_VERSION;

  //=== VK_EXT_validation_features ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_validation_features extension has been deprecated by VK_EXT_layer_settings." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTValidationFeaturesExtensionName = VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_validation_features extension has been deprecated by VK_EXT_layer_settings." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTValidationFeaturesSpecVersion = VK_EXT_VALIDATION_FEATURES_SPEC_VERSION;

  //=== VK_KHR_present_wait ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPresentWaitExtensionName = VK_KHR_PRESENT_WAIT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPresentWaitSpecVersion   = VK_KHR_PRESENT_WAIT_SPEC_VERSION;

  //=== VK_NV_cooperative_matrix ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCooperativeMatrixExtensionName = VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCooperativeMatrixSpecVersion   = VK_NV_COOPERATIVE_MATRIX_SPEC_VERSION;

  //=== VK_NV_coverage_reduction_mode ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCoverageReductionModeExtensionName = VK_NV_COVERAGE_REDUCTION_MODE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCoverageReductionModeSpecVersion   = VK_NV_COVERAGE_REDUCTION_MODE_SPEC_VERSION;

  //=== VK_EXT_fragment_shader_interlock ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFragmentShaderInterlockExtensionName = VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFragmentShaderInterlockSpecVersion   = VK_EXT_FRAGMENT_SHADER_INTERLOCK_SPEC_VERSION;

  //=== VK_EXT_ycbcr_image_arrays ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTYcbcrImageArraysExtensionName = VK_EXT_YCBCR_IMAGE_ARRAYS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTYcbcrImageArraysSpecVersion   = VK_EXT_YCBCR_IMAGE_ARRAYS_SPEC_VERSION;

  //=== VK_KHR_uniform_buffer_standard_layout ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_uniform_buffer_standard_layout extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRUniformBufferStandardLayoutExtensionName = VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_uniform_buffer_standard_layout extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRUniformBufferStandardLayoutSpecVersion = VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_SPEC_VERSION;

  //=== VK_EXT_provoking_vertex ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTProvokingVertexExtensionName = VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTProvokingVertexSpecVersion   = VK_EXT_PROVOKING_VERTEX_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFullScreenExclusiveExtensionName = VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFullScreenExclusiveSpecVersion   = VK_EXT_FULL_SCREEN_EXCLUSIVE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_headless_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTHeadlessSurfaceExtensionName = VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTHeadlessSurfaceSpecVersion   = VK_EXT_HEADLESS_SURFACE_SPEC_VERSION;

  //=== VK_KHR_buffer_device_address ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_buffer_device_address extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRBufferDeviceAddressExtensionName = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_buffer_device_address extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRBufferDeviceAddressSpecVersion = VK_KHR_BUFFER_DEVICE_ADDRESS_SPEC_VERSION;

  //=== VK_EXT_line_rasterization ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTLineRasterizationExtensionName = VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTLineRasterizationSpecVersion   = VK_EXT_LINE_RASTERIZATION_SPEC_VERSION;

  //=== VK_EXT_shader_atomic_float ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderAtomicFloatExtensionName = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderAtomicFloatSpecVersion   = VK_EXT_SHADER_ATOMIC_FLOAT_SPEC_VERSION;

  //=== VK_EXT_host_query_reset ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_host_query_reset extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTHostQueryResetExtensionName = VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_host_query_reset extension has been promoted to core in version 1.2." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTHostQueryResetSpecVersion = VK_EXT_HOST_QUERY_RESET_SPEC_VERSION;

  //=== VK_EXT_index_type_uint8 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTIndexTypeUint8ExtensionName = VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTIndexTypeUint8SpecVersion   = VK_EXT_INDEX_TYPE_UINT8_SPEC_VERSION;

  //=== VK_EXT_extended_dynamic_state ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_extended_dynamic_state extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExtendedDynamicStateExtensionName = VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_extended_dynamic_state extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExtendedDynamicStateSpecVersion = VK_EXT_EXTENDED_DYNAMIC_STATE_SPEC_VERSION;

  //=== VK_KHR_deferred_host_operations ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDeferredHostOperationsExtensionName = VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRDeferredHostOperationsSpecVersion   = VK_KHR_DEFERRED_HOST_OPERATIONS_SPEC_VERSION;

  //=== VK_KHR_pipeline_executable_properties ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPipelineExecutablePropertiesExtensionName = VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPipelineExecutablePropertiesSpecVersion   = VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_SPEC_VERSION;

  //=== VK_EXT_host_image_copy ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTHostImageCopyExtensionName = VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTHostImageCopySpecVersion   = VK_EXT_HOST_IMAGE_COPY_SPEC_VERSION;

  //=== VK_KHR_map_memory2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMapMemory2ExtensionName = VK_KHR_MAP_MEMORY_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMapMemory2SpecVersion   = VK_KHR_MAP_MEMORY_2_SPEC_VERSION;

  //=== VK_EXT_shader_atomic_float2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderAtomicFloat2ExtensionName = VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderAtomicFloat2SpecVersion   = VK_EXT_SHADER_ATOMIC_FLOAT_2_SPEC_VERSION;

  //=== VK_EXT_surface_maintenance1 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSurfaceMaintenance1ExtensionName = VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSurfaceMaintenance1SpecVersion   = VK_EXT_SURFACE_MAINTENANCE_1_SPEC_VERSION;

  //=== VK_EXT_swapchain_maintenance1 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSwapchainMaintenance1ExtensionName = VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSwapchainMaintenance1SpecVersion   = VK_EXT_SWAPCHAIN_MAINTENANCE_1_SPEC_VERSION;

  //=== VK_EXT_shader_demote_to_helper_invocation ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_shader_demote_to_helper_invocation extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderDemoteToHelperInvocationExtensionName = VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_shader_demote_to_helper_invocation extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderDemoteToHelperInvocationSpecVersion = VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_SPEC_VERSION;

  //=== VK_NV_device_generated_commands ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDeviceGeneratedCommandsExtensionName = VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDeviceGeneratedCommandsSpecVersion   = VK_NV_DEVICE_GENERATED_COMMANDS_SPEC_VERSION;

  //=== VK_NV_inherited_viewport_scissor ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVInheritedViewportScissorExtensionName = VK_NV_INHERITED_VIEWPORT_SCISSOR_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVInheritedViewportScissorSpecVersion   = VK_NV_INHERITED_VIEWPORT_SCISSOR_SPEC_VERSION;

  //=== VK_KHR_shader_integer_dot_product ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_integer_dot_product extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderIntegerDotProductExtensionName = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_integer_dot_product extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderIntegerDotProductSpecVersion = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_SPEC_VERSION;

  //=== VK_EXT_texel_buffer_alignment ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_texel_buffer_alignment extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTTexelBufferAlignmentExtensionName = VK_EXT_TEXEL_BUFFER_ALIGNMENT_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_texel_buffer_alignment extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTTexelBufferAlignmentSpecVersion = VK_EXT_TEXEL_BUFFER_ALIGNMENT_SPEC_VERSION;

  //=== VK_QCOM_render_pass_transform ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMRenderPassTransformExtensionName = VK_QCOM_RENDER_PASS_TRANSFORM_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMRenderPassTransformSpecVersion   = VK_QCOM_RENDER_PASS_TRANSFORM_SPEC_VERSION;

  //=== VK_EXT_depth_bias_control ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthBiasControlExtensionName = VK_EXT_DEPTH_BIAS_CONTROL_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthBiasControlSpecVersion   = VK_EXT_DEPTH_BIAS_CONTROL_SPEC_VERSION;

  //=== VK_EXT_device_memory_report ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDeviceMemoryReportExtensionName = VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDeviceMemoryReportSpecVersion   = VK_EXT_DEVICE_MEMORY_REPORT_SPEC_VERSION;

  //=== VK_EXT_acquire_drm_display ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAcquireDrmDisplayExtensionName = VK_EXT_ACQUIRE_DRM_DISPLAY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAcquireDrmDisplaySpecVersion   = VK_EXT_ACQUIRE_DRM_DISPLAY_SPEC_VERSION;

  //=== VK_EXT_robustness2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTRobustness2ExtensionName = VK_EXT_ROBUSTNESS_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTRobustness2SpecVersion   = VK_EXT_ROBUSTNESS_2_SPEC_VERSION;

  //=== VK_EXT_custom_border_color ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTCustomBorderColorExtensionName = VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTCustomBorderColorSpecVersion   = VK_EXT_CUSTOM_BORDER_COLOR_SPEC_VERSION;

  //=== VK_GOOGLE_user_type ===
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLEUserTypeExtensionName = VK_GOOGLE_USER_TYPE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLEUserTypeSpecVersion   = VK_GOOGLE_USER_TYPE_SPEC_VERSION;

  //=== VK_KHR_pipeline_library ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPipelineLibraryExtensionName = VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPipelineLibrarySpecVersion   = VK_KHR_PIPELINE_LIBRARY_SPEC_VERSION;

  //=== VK_NV_present_barrier ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVPresentBarrierExtensionName = VK_NV_PRESENT_BARRIER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVPresentBarrierSpecVersion   = VK_NV_PRESENT_BARRIER_SPEC_VERSION;

  //=== VK_KHR_shader_non_semantic_info ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_non_semantic_info extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderNonSemanticInfoExtensionName = VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_shader_non_semantic_info extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderNonSemanticInfoSpecVersion = VK_KHR_SHADER_NON_SEMANTIC_INFO_SPEC_VERSION;

  //=== VK_KHR_present_id ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPresentIdExtensionName = VK_KHR_PRESENT_ID_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPresentIdSpecVersion   = VK_KHR_PRESENT_ID_SPEC_VERSION;

  //=== VK_EXT_private_data ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_private_data extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPrivateDataExtensionName = VK_EXT_PRIVATE_DATA_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_private_data extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPrivateDataSpecVersion = VK_EXT_PRIVATE_DATA_SPEC_VERSION;

  //=== VK_EXT_pipeline_creation_cache_control ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_pipeline_creation_cache_control extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineCreationCacheControlExtensionName = VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_pipeline_creation_cache_control extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineCreationCacheControlSpecVersion = VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_SPEC_VERSION;

  //=== VK_KHR_video_encode_queue ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoEncodeQueueExtensionName = VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoEncodeQueueSpecVersion   = VK_KHR_VIDEO_ENCODE_QUEUE_SPEC_VERSION;

  //=== VK_NV_device_diagnostics_config ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDeviceDiagnosticsConfigExtensionName = VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDeviceDiagnosticsConfigSpecVersion   = VK_NV_DEVICE_DIAGNOSTICS_CONFIG_SPEC_VERSION;

  //=== VK_QCOM_render_pass_store_ops ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMRenderPassStoreOpsExtensionName = VK_QCOM_RENDER_PASS_STORE_OPS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMRenderPassStoreOpsSpecVersion   = VK_QCOM_RENDER_PASS_STORE_OPS_SPEC_VERSION;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCudaKernelLaunchExtensionName = VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCudaKernelLaunchSpecVersion   = VK_NV_CUDA_KERNEL_LAUNCH_SPEC_VERSION;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_low_latency ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVLowLatencyExtensionName = VK_NV_LOW_LATENCY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVLowLatencySpecVersion   = VK_NV_LOW_LATENCY_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMetalObjectsExtensionName = VK_EXT_METAL_OBJECTS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMetalObjectsSpecVersion   = VK_EXT_METAL_OBJECTS_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_synchronization2 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_synchronization2 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSynchronization2ExtensionName = VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_synchronization2 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRSynchronization2SpecVersion = VK_KHR_SYNCHRONIZATION_2_SPEC_VERSION;

  //=== VK_EXT_descriptor_buffer ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDescriptorBufferExtensionName = VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDescriptorBufferSpecVersion   = VK_EXT_DESCRIPTOR_BUFFER_SPEC_VERSION;

  //=== VK_EXT_graphics_pipeline_library ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTGraphicsPipelineLibraryExtensionName = VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTGraphicsPipelineLibrarySpecVersion   = VK_EXT_GRAPHICS_PIPELINE_LIBRARY_SPEC_VERSION;

  //=== VK_AMD_shader_early_and_late_fragment_tests ===
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderEarlyAndLateFragmentTestsExtensionName = VK_AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto AMDShaderEarlyAndLateFragmentTestsSpecVersion   = VK_AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_SPEC_VERSION;

  //=== VK_KHR_fragment_shader_barycentric ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRFragmentShaderBarycentricExtensionName = VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRFragmentShaderBarycentricSpecVersion   = VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_SPEC_VERSION;

  //=== VK_KHR_shader_subgroup_uniform_control_flow ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderSubgroupUniformControlFlowExtensionName = VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRShaderSubgroupUniformControlFlowSpecVersion   = VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_SPEC_VERSION;

  //=== VK_KHR_zero_initialize_workgroup_memory ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_zero_initialize_workgroup_memory extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRZeroInitializeWorkgroupMemoryExtensionName = VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_zero_initialize_workgroup_memory extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRZeroInitializeWorkgroupMemorySpecVersion = VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_SPEC_VERSION;

  //=== VK_NV_fragment_shading_rate_enums ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFragmentShadingRateEnumsExtensionName = VK_NV_FRAGMENT_SHADING_RATE_ENUMS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVFragmentShadingRateEnumsSpecVersion   = VK_NV_FRAGMENT_SHADING_RATE_ENUMS_SPEC_VERSION;

  //=== VK_NV_ray_tracing_motion_blur ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVRayTracingMotionBlurExtensionName = VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVRayTracingMotionBlurSpecVersion   = VK_NV_RAY_TRACING_MOTION_BLUR_SPEC_VERSION;

  //=== VK_EXT_mesh_shader ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMeshShaderExtensionName = VK_EXT_MESH_SHADER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMeshShaderSpecVersion   = VK_EXT_MESH_SHADER_SPEC_VERSION;

  //=== VK_EXT_ycbcr_2plane_444_formats ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_ycbcr_2plane_444_formats extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTYcbcr2Plane444FormatsExtensionName = VK_EXT_YCBCR_2PLANE_444_FORMATS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_ycbcr_2plane_444_formats extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTYcbcr2Plane444FormatsSpecVersion = VK_EXT_YCBCR_2PLANE_444_FORMATS_SPEC_VERSION;

  //=== VK_EXT_fragment_density_map2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFragmentDensityMap2ExtensionName = VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFragmentDensityMap2SpecVersion   = VK_EXT_FRAGMENT_DENSITY_MAP_2_SPEC_VERSION;

  //=== VK_QCOM_rotated_copy_commands ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMRotatedCopyCommandsExtensionName = VK_QCOM_ROTATED_COPY_COMMANDS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMRotatedCopyCommandsSpecVersion   = VK_QCOM_ROTATED_COPY_COMMANDS_SPEC_VERSION;

  //=== VK_EXT_image_robustness ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_image_robustness extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageRobustnessExtensionName = VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_image_robustness extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageRobustnessSpecVersion = VK_EXT_IMAGE_ROBUSTNESS_SPEC_VERSION;

  //=== VK_KHR_workgroup_memory_explicit_layout ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRWorkgroupMemoryExplicitLayoutExtensionName = VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRWorkgroupMemoryExplicitLayoutSpecVersion   = VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_SPEC_VERSION;

  //=== VK_KHR_copy_commands2 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_copy_commands2 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRCopyCommands2ExtensionName = VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_copy_commands2 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRCopyCommands2SpecVersion = VK_KHR_COPY_COMMANDS_2_SPEC_VERSION;

  //=== VK_EXT_image_compression_control ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageCompressionControlExtensionName = VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageCompressionControlSpecVersion   = VK_EXT_IMAGE_COMPRESSION_CONTROL_SPEC_VERSION;

  //=== VK_EXT_attachment_feedback_loop_layout ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAttachmentFeedbackLoopLayoutExtensionName = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAttachmentFeedbackLoopLayoutSpecVersion   = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_SPEC_VERSION;

  //=== VK_EXT_4444_formats ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_4444_formats extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXT4444FormatsExtensionName = VK_EXT_4444_FORMATS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_4444_formats extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXT4444FormatsSpecVersion = VK_EXT_4444_FORMATS_SPEC_VERSION;

  //=== VK_EXT_device_fault ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDeviceFaultExtensionName = VK_EXT_DEVICE_FAULT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDeviceFaultSpecVersion   = VK_EXT_DEVICE_FAULT_SPEC_VERSION;

  //=== VK_ARM_rasterization_order_attachment_access ===
  VULKAN_HPP_DEPRECATED( "The VK_ARM_rasterization_order_attachment_access extension has been promoted to VK_EXT_rasterization_order_attachment_access." )
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMRasterizationOrderAttachmentAccessExtensionName = VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_ARM_rasterization_order_attachment_access extension has been promoted to VK_EXT_rasterization_order_attachment_access." )
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMRasterizationOrderAttachmentAccessSpecVersion = VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_SPEC_VERSION;

  //=== VK_EXT_rgba10x6_formats ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTRgba10X6FormatsExtensionName = VK_EXT_RGBA10X6_FORMATS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTRgba10X6FormatsSpecVersion   = VK_EXT_RGBA10X6_FORMATS_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_acquire_winrt_display ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVAcquireWinrtDisplayExtensionName = VK_NV_ACQUIRE_WINRT_DISPLAY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVAcquireWinrtDisplaySpecVersion   = VK_NV_ACQUIRE_WINRT_DISPLAY_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDirectfbSurfaceExtensionName = VK_EXT_DIRECTFB_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDirectfbSurfaceSpecVersion   = VK_EXT_DIRECTFB_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_VALVE_mutable_descriptor_type ===
  VULKAN_HPP_DEPRECATED( "The VK_VALVE_mutable_descriptor_type extension has been promoted to VK_EXT_mutable_descriptor_type." )
  VULKAN_HPP_CONSTEXPR_INLINE auto VALVEMutableDescriptorTypeExtensionName = VK_VALVE_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_VALVE_mutable_descriptor_type extension has been promoted to VK_EXT_mutable_descriptor_type." )
  VULKAN_HPP_CONSTEXPR_INLINE auto VALVEMutableDescriptorTypeSpecVersion = VK_VALVE_MUTABLE_DESCRIPTOR_TYPE_SPEC_VERSION;

  //=== VK_EXT_vertex_input_dynamic_state ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTVertexInputDynamicStateExtensionName = VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTVertexInputDynamicStateSpecVersion   = VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_SPEC_VERSION;

  //=== VK_EXT_physical_device_drm ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPhysicalDeviceDrmExtensionName = VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPhysicalDeviceDrmSpecVersion   = VK_EXT_PHYSICAL_DEVICE_DRM_SPEC_VERSION;

  //=== VK_EXT_device_address_binding_report ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDeviceAddressBindingReportExtensionName = VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDeviceAddressBindingReportSpecVersion   = VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_SPEC_VERSION;

  //=== VK_EXT_depth_clip_control ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthClipControlExtensionName = VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthClipControlSpecVersion   = VK_EXT_DEPTH_CLIP_CONTROL_SPEC_VERSION;

  //=== VK_EXT_primitive_topology_list_restart ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPrimitiveTopologyListRestartExtensionName = VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPrimitiveTopologyListRestartSpecVersion   = VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_SPEC_VERSION;

  //=== VK_KHR_format_feature_flags2 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_format_feature_flags2 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRFormatFeatureFlags2ExtensionName = VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_format_feature_flags2 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRFormatFeatureFlags2SpecVersion = VK_KHR_FORMAT_FEATURE_FLAGS_2_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_memory ===
  VULKAN_HPP_CONSTEXPR_INLINE auto FUCHSIAExternalMemoryExtensionName = VK_FUCHSIA_EXTERNAL_MEMORY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto FUCHSIAExternalMemorySpecVersion   = VK_FUCHSIA_EXTERNAL_MEMORY_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_semaphore ===
  VULKAN_HPP_CONSTEXPR_INLINE auto FUCHSIAExternalSemaphoreExtensionName = VK_FUCHSIA_EXTERNAL_SEMAPHORE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto FUCHSIAExternalSemaphoreSpecVersion   = VK_FUCHSIA_EXTERNAL_SEMAPHORE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  VULKAN_HPP_CONSTEXPR_INLINE auto FUCHSIABufferCollectionExtensionName = VK_FUCHSIA_BUFFER_COLLECTION_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto FUCHSIABufferCollectionSpecVersion   = VK_FUCHSIA_BUFFER_COLLECTION_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_HUAWEI_subpass_shading ===
  VULKAN_HPP_CONSTEXPR_INLINE auto HUAWEISubpassShadingExtensionName = VK_HUAWEI_SUBPASS_SHADING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto HUAWEISubpassShadingSpecVersion   = VK_HUAWEI_SUBPASS_SHADING_SPEC_VERSION;

  //=== VK_HUAWEI_invocation_mask ===
  VULKAN_HPP_CONSTEXPR_INLINE auto HUAWEIInvocationMaskExtensionName = VK_HUAWEI_INVOCATION_MASK_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto HUAWEIInvocationMaskSpecVersion   = VK_HUAWEI_INVOCATION_MASK_SPEC_VERSION;

  //=== VK_NV_external_memory_rdma ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExternalMemoryRdmaExtensionName = VK_NV_EXTERNAL_MEMORY_RDMA_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExternalMemoryRdmaSpecVersion   = VK_NV_EXTERNAL_MEMORY_RDMA_SPEC_VERSION;

  //=== VK_EXT_pipeline_properties ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelinePropertiesExtensionName = VK_EXT_PIPELINE_PROPERTIES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelinePropertiesSpecVersion   = VK_EXT_PIPELINE_PROPERTIES_SPEC_VERSION;

  //=== VK_EXT_frame_boundary ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFrameBoundaryExtensionName = VK_EXT_FRAME_BOUNDARY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTFrameBoundarySpecVersion   = VK_EXT_FRAME_BOUNDARY_SPEC_VERSION;

  //=== VK_EXT_multisampled_render_to_single_sampled ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMultisampledRenderToSingleSampledExtensionName = VK_EXT_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMultisampledRenderToSingleSampledSpecVersion   = VK_EXT_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_SPEC_VERSION;

  //=== VK_EXT_extended_dynamic_state2 ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_extended_dynamic_state2 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExtendedDynamicState2ExtensionName = VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_extended_dynamic_state2 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExtendedDynamicState2SpecVersion = VK_EXT_EXTENDED_DYNAMIC_STATE_2_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QNXScreenSurfaceExtensionName = VK_QNX_SCREEN_SURFACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QNXScreenSurfaceSpecVersion   = VK_QNX_SCREEN_SURFACE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_color_write_enable ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTColorWriteEnableExtensionName = VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTColorWriteEnableSpecVersion   = VK_EXT_COLOR_WRITE_ENABLE_SPEC_VERSION;

  //=== VK_EXT_primitives_generated_query ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPrimitivesGeneratedQueryExtensionName = VK_EXT_PRIMITIVES_GENERATED_QUERY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPrimitivesGeneratedQuerySpecVersion   = VK_EXT_PRIMITIVES_GENERATED_QUERY_SPEC_VERSION;

  //=== VK_KHR_ray_tracing_maintenance1 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRayTracingMaintenance1ExtensionName = VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRayTracingMaintenance1SpecVersion   = VK_KHR_RAY_TRACING_MAINTENANCE_1_SPEC_VERSION;

  //=== VK_EXT_global_priority_query ===
  VULKAN_HPP_DEPRECATED( "The VK_EXT_global_priority_query extension has been promoted to VK_KHR_global_priority." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTGlobalPriorityQueryExtensionName = VK_EXT_GLOBAL_PRIORITY_QUERY_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_EXT_global_priority_query extension has been promoted to VK_KHR_global_priority." )
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTGlobalPriorityQuerySpecVersion = VK_EXT_GLOBAL_PRIORITY_QUERY_SPEC_VERSION;

  //=== VK_EXT_image_view_min_lod ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageViewMinLodExtensionName = VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageViewMinLodSpecVersion   = VK_EXT_IMAGE_VIEW_MIN_LOD_SPEC_VERSION;

  //=== VK_EXT_multi_draw ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMultiDrawExtensionName = VK_EXT_MULTI_DRAW_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMultiDrawSpecVersion   = VK_EXT_MULTI_DRAW_SPEC_VERSION;

  //=== VK_EXT_image_2d_view_of_3d ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImage2DViewOf3DExtensionName = VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImage2DViewOf3DSpecVersion   = VK_EXT_IMAGE_2D_VIEW_OF_3D_SPEC_VERSION;

  //=== VK_KHR_portability_enumeration ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPortabilityEnumerationExtensionName = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRPortabilityEnumerationSpecVersion   = VK_KHR_PORTABILITY_ENUMERATION_SPEC_VERSION;

  //=== VK_EXT_shader_tile_image ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderTileImageExtensionName = VK_EXT_SHADER_TILE_IMAGE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderTileImageSpecVersion   = VK_EXT_SHADER_TILE_IMAGE_SPEC_VERSION;

  //=== VK_EXT_opacity_micromap ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTOpacityMicromapExtensionName = VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTOpacityMicromapSpecVersion   = VK_EXT_OPACITY_MICROMAP_SPEC_VERSION;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDisplacementMicromapExtensionName = VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDisplacementMicromapSpecVersion   = VK_NV_DISPLACEMENT_MICROMAP_SPEC_VERSION;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_load_store_op_none ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTLoadStoreOpNoneExtensionName = VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTLoadStoreOpNoneSpecVersion   = VK_EXT_LOAD_STORE_OP_NONE_SPEC_VERSION;

  //=== VK_HUAWEI_cluster_culling_shader ===
  VULKAN_HPP_CONSTEXPR_INLINE auto HUAWEIClusterCullingShaderExtensionName = VK_HUAWEI_CLUSTER_CULLING_SHADER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto HUAWEIClusterCullingShaderSpecVersion   = VK_HUAWEI_CLUSTER_CULLING_SHADER_SPEC_VERSION;

  //=== VK_EXT_border_color_swizzle ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTBorderColorSwizzleExtensionName = VK_EXT_BORDER_COLOR_SWIZZLE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTBorderColorSwizzleSpecVersion   = VK_EXT_BORDER_COLOR_SWIZZLE_SPEC_VERSION;

  //=== VK_EXT_pageable_device_local_memory ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPageableDeviceLocalMemoryExtensionName = VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPageableDeviceLocalMemorySpecVersion   = VK_EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_SPEC_VERSION;

  //=== VK_KHR_maintenance4 ===
  VULKAN_HPP_DEPRECATED( "The VK_KHR_maintenance4 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance4ExtensionName = VK_KHR_MAINTENANCE_4_EXTENSION_NAME;
  VULKAN_HPP_DEPRECATED( "The VK_KHR_maintenance4 extension has been promoted to core in version 1.3." )
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance4SpecVersion = VK_KHR_MAINTENANCE_4_SPEC_VERSION;

  //=== VK_ARM_shader_core_properties ===
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMShaderCorePropertiesExtensionName = VK_ARM_SHADER_CORE_PROPERTIES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMShaderCorePropertiesSpecVersion   = VK_ARM_SHADER_CORE_PROPERTIES_SPEC_VERSION;

  //=== VK_ARM_scheduling_controls ===
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMSchedulingControlsExtensionName = VK_ARM_SCHEDULING_CONTROLS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMSchedulingControlsSpecVersion   = VK_ARM_SCHEDULING_CONTROLS_SPEC_VERSION;

  //=== VK_EXT_image_sliced_view_of_3d ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageSlicedViewOf3DExtensionName = VK_EXT_IMAGE_SLICED_VIEW_OF_3D_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageSlicedViewOf3DSpecVersion   = VK_EXT_IMAGE_SLICED_VIEW_OF_3D_SPEC_VERSION;

  //=== VK_VALVE_descriptor_set_host_mapping ===
  VULKAN_HPP_CONSTEXPR_INLINE auto VALVEDescriptorSetHostMappingExtensionName = VK_VALVE_DESCRIPTOR_SET_HOST_MAPPING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto VALVEDescriptorSetHostMappingSpecVersion   = VK_VALVE_DESCRIPTOR_SET_HOST_MAPPING_SPEC_VERSION;

  //=== VK_EXT_depth_clamp_zero_one ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthClampZeroOneExtensionName = VK_EXT_DEPTH_CLAMP_ZERO_ONE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDepthClampZeroOneSpecVersion   = VK_EXT_DEPTH_CLAMP_ZERO_ONE_SPEC_VERSION;

  //=== VK_EXT_non_seamless_cube_map ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTNonSeamlessCubeMapExtensionName = VK_EXT_NON_SEAMLESS_CUBE_MAP_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTNonSeamlessCubeMapSpecVersion   = VK_EXT_NON_SEAMLESS_CUBE_MAP_SPEC_VERSION;

  //=== VK_ARM_render_pass_striped ===
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMRenderPassStripedExtensionName = VK_ARM_RENDER_PASS_STRIPED_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMRenderPassStripedSpecVersion   = VK_ARM_RENDER_PASS_STRIPED_SPEC_VERSION;

  //=== VK_QCOM_fragment_density_map_offset ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMFragmentDensityMapOffsetExtensionName = VK_QCOM_FRAGMENT_DENSITY_MAP_OFFSET_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMFragmentDensityMapOffsetSpecVersion   = VK_QCOM_FRAGMENT_DENSITY_MAP_OFFSET_SPEC_VERSION;

  //=== VK_NV_copy_memory_indirect ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCopyMemoryIndirectExtensionName = VK_NV_COPY_MEMORY_INDIRECT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVCopyMemoryIndirectSpecVersion   = VK_NV_COPY_MEMORY_INDIRECT_SPEC_VERSION;

  //=== VK_NV_memory_decompression ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVMemoryDecompressionExtensionName = VK_NV_MEMORY_DECOMPRESSION_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVMemoryDecompressionSpecVersion   = VK_NV_MEMORY_DECOMPRESSION_SPEC_VERSION;

  //=== VK_NV_device_generated_commands_compute ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDeviceGeneratedCommandsComputeExtensionName = VK_NV_DEVICE_GENERATED_COMMANDS_COMPUTE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDeviceGeneratedCommandsComputeSpecVersion   = VK_NV_DEVICE_GENERATED_COMMANDS_COMPUTE_SPEC_VERSION;

  //=== VK_NV_linear_color_attachment ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVLinearColorAttachmentExtensionName = VK_NV_LINEAR_COLOR_ATTACHMENT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVLinearColorAttachmentSpecVersion   = VK_NV_LINEAR_COLOR_ATTACHMENT_SPEC_VERSION;

  //=== VK_GOOGLE_surfaceless_query ===
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLESurfacelessQueryExtensionName = VK_GOOGLE_SURFACELESS_QUERY_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto GOOGLESurfacelessQuerySpecVersion   = VK_GOOGLE_SURFACELESS_QUERY_SPEC_VERSION;

  //=== VK_EXT_image_compression_control_swapchain ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageCompressionControlSwapchainExtensionName = VK_EXT_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTImageCompressionControlSwapchainSpecVersion   = VK_EXT_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_SPEC_VERSION;

  //=== VK_QCOM_image_processing ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMImageProcessingExtensionName = VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMImageProcessingSpecVersion   = VK_QCOM_IMAGE_PROCESSING_SPEC_VERSION;

  //=== VK_EXT_nested_command_buffer ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTNestedCommandBufferExtensionName = VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTNestedCommandBufferSpecVersion   = VK_EXT_NESTED_COMMAND_BUFFER_SPEC_VERSION;

  //=== VK_EXT_external_memory_acquire_unmodified ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExternalMemoryAcquireUnmodifiedExtensionName = VK_EXT_EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExternalMemoryAcquireUnmodifiedSpecVersion   = VK_EXT_EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED_SPEC_VERSION;

  //=== VK_EXT_extended_dynamic_state3 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExtendedDynamicState3ExtensionName = VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTExtendedDynamicState3SpecVersion   = VK_EXT_EXTENDED_DYNAMIC_STATE_3_SPEC_VERSION;

  //=== VK_EXT_subpass_merge_feedback ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSubpassMergeFeedbackExtensionName = VK_EXT_SUBPASS_MERGE_FEEDBACK_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTSubpassMergeFeedbackSpecVersion   = VK_EXT_SUBPASS_MERGE_FEEDBACK_SPEC_VERSION;

  //=== VK_LUNARG_direct_driver_loading ===
  VULKAN_HPP_CONSTEXPR_INLINE auto LUNARGDirectDriverLoadingExtensionName = VK_LUNARG_DIRECT_DRIVER_LOADING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto LUNARGDirectDriverLoadingSpecVersion   = VK_LUNARG_DIRECT_DRIVER_LOADING_SPEC_VERSION;

  //=== VK_EXT_shader_module_identifier ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderModuleIdentifierExtensionName = VK_EXT_SHADER_MODULE_IDENTIFIER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderModuleIdentifierSpecVersion   = VK_EXT_SHADER_MODULE_IDENTIFIER_SPEC_VERSION;

  //=== VK_EXT_rasterization_order_attachment_access ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTRasterizationOrderAttachmentAccessExtensionName = VK_EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTRasterizationOrderAttachmentAccessSpecVersion   = VK_EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_SPEC_VERSION;

  //=== VK_NV_optical_flow ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVOpticalFlowExtensionName = VK_NV_OPTICAL_FLOW_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVOpticalFlowSpecVersion   = VK_NV_OPTICAL_FLOW_SPEC_VERSION;

  //=== VK_EXT_legacy_dithering ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTLegacyDitheringExtensionName = VK_EXT_LEGACY_DITHERING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTLegacyDitheringSpecVersion   = VK_EXT_LEGACY_DITHERING_SPEC_VERSION;

  //=== VK_EXT_pipeline_protected_access ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineProtectedAccessExtensionName = VK_EXT_PIPELINE_PROTECTED_ACCESS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineProtectedAccessSpecVersion   = VK_EXT_PIPELINE_PROTECTED_ACCESS_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_format_resolve ===
  VULKAN_HPP_CONSTEXPR_INLINE auto ANDROIDExternalFormatResolveExtensionName = VK_ANDROID_EXTERNAL_FORMAT_RESOLVE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto ANDROIDExternalFormatResolveSpecVersion   = VK_ANDROID_EXTERNAL_FORMAT_RESOLVE_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  //=== VK_KHR_maintenance5 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance5ExtensionName = VK_KHR_MAINTENANCE_5_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance5SpecVersion   = VK_KHR_MAINTENANCE_5_SPEC_VERSION;

  //=== VK_KHR_ray_tracing_position_fetch ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRayTracingPositionFetchExtensionName = VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRRayTracingPositionFetchSpecVersion   = VK_KHR_RAY_TRACING_POSITION_FETCH_SPEC_VERSION;

  //=== VK_EXT_shader_object ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderObjectExtensionName = VK_EXT_SHADER_OBJECT_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTShaderObjectSpecVersion   = VK_EXT_SHADER_OBJECT_SPEC_VERSION;

  //=== VK_QCOM_tile_properties ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMTilePropertiesExtensionName = VK_QCOM_TILE_PROPERTIES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMTilePropertiesSpecVersion   = VK_QCOM_TILE_PROPERTIES_SPEC_VERSION;

  //=== VK_SEC_amigo_profiling ===
  VULKAN_HPP_CONSTEXPR_INLINE auto SECAmigoProfilingExtensionName = VK_SEC_AMIGO_PROFILING_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto SECAmigoProfilingSpecVersion   = VK_SEC_AMIGO_PROFILING_SPEC_VERSION;

  //=== VK_QCOM_multiview_per_view_viewports ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMMultiviewPerViewViewportsExtensionName = VK_QCOM_MULTIVIEW_PER_VIEW_VIEWPORTS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMMultiviewPerViewViewportsSpecVersion   = VK_QCOM_MULTIVIEW_PER_VIEW_VIEWPORTS_SPEC_VERSION;

  //=== VK_NV_ray_tracing_invocation_reorder ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVRayTracingInvocationReorderExtensionName = VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVRayTracingInvocationReorderSpecVersion   = VK_NV_RAY_TRACING_INVOCATION_REORDER_SPEC_VERSION;

  //=== VK_NV_extended_sparse_address_space ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExtendedSparseAddressSpaceExtensionName = VK_NV_EXTENDED_SPARSE_ADDRESS_SPACE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVExtendedSparseAddressSpaceSpecVersion   = VK_NV_EXTENDED_SPARSE_ADDRESS_SPACE_SPEC_VERSION;

  //=== VK_EXT_mutable_descriptor_type ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMutableDescriptorTypeExtensionName = VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTMutableDescriptorTypeSpecVersion   = VK_EXT_MUTABLE_DESCRIPTOR_TYPE_SPEC_VERSION;

  //=== VK_EXT_layer_settings ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTLayerSettingsExtensionName = VK_EXT_LAYER_SETTINGS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTLayerSettingsSpecVersion   = VK_EXT_LAYER_SETTINGS_SPEC_VERSION;

  //=== VK_ARM_shader_core_builtins ===
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMShaderCoreBuiltinsExtensionName = VK_ARM_SHADER_CORE_BUILTINS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto ARMShaderCoreBuiltinsSpecVersion   = VK_ARM_SHADER_CORE_BUILTINS_SPEC_VERSION;

  //=== VK_EXT_pipeline_library_group_handles ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineLibraryGroupHandlesExtensionName = VK_EXT_PIPELINE_LIBRARY_GROUP_HANDLES_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTPipelineLibraryGroupHandlesSpecVersion   = VK_EXT_PIPELINE_LIBRARY_GROUP_HANDLES_SPEC_VERSION;

  //=== VK_EXT_dynamic_rendering_unused_attachments ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDynamicRenderingUnusedAttachmentsExtensionName = VK_EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTDynamicRenderingUnusedAttachmentsSpecVersion   = VK_EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_SPEC_VERSION;

  //=== VK_NV_low_latency2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVLowLatency2ExtensionName = VK_NV_LOW_LATENCY_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVLowLatency2SpecVersion   = VK_NV_LOW_LATENCY_2_SPEC_VERSION;

  //=== VK_KHR_cooperative_matrix ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRCooperativeMatrixExtensionName = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRCooperativeMatrixSpecVersion   = VK_KHR_COOPERATIVE_MATRIX_SPEC_VERSION;

  //=== VK_QCOM_multiview_per_view_render_areas ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMMultiviewPerViewRenderAreasExtensionName = VK_QCOM_MULTIVIEW_PER_VIEW_RENDER_AREAS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMMultiviewPerViewRenderAreasSpecVersion   = VK_QCOM_MULTIVIEW_PER_VIEW_RENDER_AREAS_SPEC_VERSION;

  //=== VK_KHR_video_maintenance1 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoMaintenance1ExtensionName = VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVideoMaintenance1SpecVersion   = VK_KHR_VIDEO_MAINTENANCE_1_SPEC_VERSION;

  //=== VK_NV_per_stage_descriptor_set ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVPerStageDescriptorSetExtensionName = VK_NV_PER_STAGE_DESCRIPTOR_SET_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVPerStageDescriptorSetSpecVersion   = VK_NV_PER_STAGE_DESCRIPTOR_SET_SPEC_VERSION;

  //=== VK_QCOM_image_processing2 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMImageProcessing2ExtensionName = VK_QCOM_IMAGE_PROCESSING_2_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMImageProcessing2SpecVersion   = VK_QCOM_IMAGE_PROCESSING_2_SPEC_VERSION;

  //=== VK_QCOM_filter_cubic_weights ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMFilterCubicWeightsExtensionName = VK_QCOM_FILTER_CUBIC_WEIGHTS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMFilterCubicWeightsSpecVersion   = VK_QCOM_FILTER_CUBIC_WEIGHTS_SPEC_VERSION;

  //=== VK_QCOM_ycbcr_degamma ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMYcbcrDegammaExtensionName = VK_QCOM_YCBCR_DEGAMMA_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMYcbcrDegammaSpecVersion   = VK_QCOM_YCBCR_DEGAMMA_SPEC_VERSION;

  //=== VK_QCOM_filter_cubic_clamp ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMFilterCubicClampExtensionName = VK_QCOM_FILTER_CUBIC_CLAMP_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QCOMFilterCubicClampSpecVersion   = VK_QCOM_FILTER_CUBIC_CLAMP_SPEC_VERSION;

  //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAttachmentFeedbackLoopDynamicStateExtensionName = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto EXTAttachmentFeedbackLoopDynamicStateSpecVersion   = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_SPEC_VERSION;

  //=== VK_KHR_vertex_attribute_divisor ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVertexAttributeDivisorExtensionName = VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRVertexAttributeDivisorSpecVersion   = VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_SPEC_VERSION;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_external_memory_screen_buffer ===
  VULKAN_HPP_CONSTEXPR_INLINE auto QNXExternalMemoryScreenBufferExtensionName = VK_QNX_EXTERNAL_MEMORY_SCREEN_BUFFER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto QNXExternalMemoryScreenBufferSpecVersion   = VK_QNX_EXTERNAL_MEMORY_SCREEN_BUFFER_SPEC_VERSION;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_MSFT_layered_driver ===
  VULKAN_HPP_CONSTEXPR_INLINE auto MSFTLayeredDriverExtensionName = VK_MSFT_LAYERED_DRIVER_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto MSFTLayeredDriverSpecVersion   = VK_MSFT_LAYERED_DRIVER_SPEC_VERSION;

  //=== VK_KHR_calibrated_timestamps ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRCalibratedTimestampsExtensionName = VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRCalibratedTimestampsSpecVersion   = VK_KHR_CALIBRATED_TIMESTAMPS_SPEC_VERSION;

  //=== VK_KHR_maintenance6 ===
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance6ExtensionName = VK_KHR_MAINTENANCE_6_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto KHRMaintenance6SpecVersion   = VK_KHR_MAINTENANCE_6_SPEC_VERSION;

  //=== VK_NV_descriptor_pool_overallocation ===
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDescriptorPoolOverallocationExtensionName = VK_NV_DESCRIPTOR_POOL_OVERALLOCATION_EXTENSION_NAME;
  VULKAN_HPP_CONSTEXPR_INLINE auto NVDescriptorPoolOverallocationSpecVersion   = VK_NV_DESCRIPTOR_POOL_OVERALLOCATION_SPEC_VERSION;

}  // namespace VULKAN_HPP_NAMESPACE

// clang-format off
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_funcs.hpp>

// clang-format on

namespace VULKAN_HPP_NAMESPACE
{
#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )

  //=======================
  //=== STRUCTS EXTENDS ===
  //=======================

  //=== VK_VERSION_1_0 ===
  template <>
  struct StructExtends<ShaderModuleCreateInfo, PipelineShaderStageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineLayoutCreateInfo, BindDescriptorSetsInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineLayoutCreateInfo, PushConstantsInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineLayoutCreateInfo, PushDescriptorSetInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineLayoutCreateInfo, PushDescriptorSetWithTemplateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineLayoutCreateInfo, SetDescriptorBufferOffsetsInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineLayoutCreateInfo, BindDescriptorBufferEmbeddedSamplersInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_VERSION_1_1 ===
  template <>
  struct StructExtends<PhysicalDeviceSubgroupProperties, PhysicalDeviceProperties2>
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
  struct StructExtends<MemoryDedicatedRequirements, MemoryRequirements2>
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
  struct StructExtends<MemoryAllocateFlagsInfo, MemoryAllocateInfo>
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
  struct StructExtends<DeviceGroupRenderPassBeginInfo, RenderingInfo>
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
  struct StructExtends<DeviceGroupSubmitInfo, SubmitInfo>
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
  struct StructExtends<DeviceGroupDeviceCreateInfo, DeviceCreateInfo>
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
  struct StructExtends<PhysicalDevicePointClippingProperties, PhysicalDeviceProperties2>
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
  struct StructExtends<ImageViewUsageCreateInfo, ImageViewCreateInfo>
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
  struct StructExtends<RenderPassMultiviewCreateInfo, RenderPassCreateInfo>
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
  struct StructExtends<PhysicalDeviceMultiviewProperties, PhysicalDeviceProperties2>
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
  struct StructExtends<ProtectedSubmitInfo, SubmitInfo>
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
  struct StructExtends<BindImagePlaneMemoryInfo, BindImageMemoryInfo>
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
  struct StructExtends<SamplerYcbcrConversionImageFormatProperties, ImageFormatProperties2>
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
  struct StructExtends<ExternalImageFormatProperties, ImageFormatProperties2>
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
  struct StructExtends<ExternalMemoryImageCreateInfo, ImageCreateInfo>
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
  struct StructExtends<ExportMemoryAllocateInfo, MemoryAllocateInfo>
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

  template <>
  struct StructExtends<ExportSemaphoreCreateInfo, SemaphoreCreateInfo>
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

  //=== VK_VERSION_1_2 ===
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
  struct StructExtends<PhysicalDeviceDriverProperties, PhysicalDeviceProperties2>
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
  struct StructExtends<PhysicalDeviceFloatControlsProperties, PhysicalDeviceProperties2>
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
  struct StructExtends<SubpassDescriptionDepthStencilResolve, SubpassDescription2>
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
  struct StructExtends<SamplerReductionModeCreateInfo, SamplerCreateInfo>
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
  struct StructExtends<FramebufferAttachmentsCreateInfo, FramebufferCreateInfo>
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
  struct StructExtends<AttachmentReferenceStencilLayout, AttachmentReference2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AttachmentDescriptionStencilLayout, AttachmentDescription2>
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
  struct StructExtends<BufferOpaqueCaptureAddressCreateInfo, BufferCreateInfo>
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

  //=== VK_VERSION_1_3 ===
  template <>
  struct StructExtends<PhysicalDeviceVulkan13Features, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceVulkan13Features, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceVulkan13Properties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfo, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfo, ComputePipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfo, RayTracingPipelineCreateInfoNV>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfo, RayTracingPipelineCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<PipelineCreationFeedbackCreateInfo, ExecutionGraphPipelineCreateInfoAMDX>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
  template <>
  struct StructExtends<PhysicalDeviceShaderTerminateInvocationFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderTerminateInvocationFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderDemoteToHelperInvocationFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderDemoteToHelperInvocationFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePrivateDataFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePrivateDataFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<DevicePrivateDataCreateInfo, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePipelineCreationCacheControlFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePipelineCreationCacheControlFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MemoryBarrier2, SubpassDependency2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSynchronization2Features, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSynchronization2Features, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageRobustnessFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageRobustnessFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSubgroupSizeControlFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSubgroupSizeControlFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSubgroupSizeControlProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineShaderStageRequiredSubgroupSizeCreateInfo, PipelineShaderStageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineShaderStageRequiredSubgroupSizeCreateInfo, ShaderCreateInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceInlineUniformBlockFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceInlineUniformBlockFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceInlineUniformBlockProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<WriteDescriptorSetInlineUniformBlock, WriteDescriptorSet>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<DescriptorPoolInlineUniformBlockCreateInfo, DescriptorPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceTextureCompressionASTCHDRFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceTextureCompressionASTCHDRFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineRenderingCreateInfo, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDynamicRenderingFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDynamicRenderingFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<CommandBufferInheritanceRenderingInfo, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderIntegerDotProductFeatures, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderIntegerDotProductFeatures, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderIntegerDotProductProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceTexelBufferAlignmentProperties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<FormatProperties3, FormatProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMaintenance4Features, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMaintenance4Features, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMaintenance4Properties, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_swapchain ===
  template <>
  struct StructExtends<ImageSwapchainCreateInfoKHR, ImageCreateInfo>
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
  struct StructExtends<DeviceGroupPresentInfoKHR, PresentInfoKHR>
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

  //=== VK_KHR_display_swapchain ===
  template <>
  struct StructExtends<DisplayPresentInfoKHR, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_debug_report ===
  template <>
  struct StructExtends<DebugReportCallbackCreateInfoEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_AMD_rasterization_order ===
  template <>
  struct StructExtends<PipelineRasterizationStateRasterizationOrderAMD, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_video_queue ===
  template <>
  struct StructExtends<QueueFamilyQueryResultStatusPropertiesKHR, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<QueueFamilyVideoPropertiesKHR, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoProfileInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoProfileListInfoKHR, PhysicalDeviceImageFormatInfo2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoProfileListInfoKHR, PhysicalDeviceVideoFormatInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoProfileListInfoKHR, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoProfileListInfoKHR, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_video_decode_queue ===
  template <>
  struct StructExtends<VideoDecodeCapabilitiesKHR, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeUsageInfoKHR, VideoProfileInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeUsageInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_dedicated_allocation ===
  template <>
  struct StructExtends<DedicatedAllocationImageCreateInfoNV, ImageCreateInfo>
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
  struct StructExtends<DedicatedAllocationMemoryAllocateInfoNV, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_transform_feedback ===
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
  struct StructExtends<PipelineRasterizationStateStreamCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_video_encode_h264 ===
  template <>
  struct StructExtends<VideoEncodeH264CapabilitiesKHR, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264QualityLevelPropertiesKHR, VideoEncodeQualityLevelPropertiesKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264SessionCreateInfoKHR, VideoSessionCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264SessionParametersCreateInfoKHR, VideoSessionParametersCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264SessionParametersAddInfoKHR, VideoSessionParametersUpdateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264SessionParametersGetInfoKHR, VideoEncodeSessionParametersGetInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264SessionParametersFeedbackInfoKHR, VideoEncodeSessionParametersFeedbackInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264PictureInfoKHR, VideoEncodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264DpbSlotInfoKHR, VideoReferenceSlotInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264ProfileInfoKHR, VideoProfileInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264ProfileInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264RateControlInfoKHR, VideoCodingControlInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264RateControlInfoKHR, VideoBeginCodingInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264RateControlLayerInfoKHR, VideoEncodeRateControlLayerInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH264GopRemainingFrameInfoKHR, VideoBeginCodingInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_video_encode_h265 ===
  template <>
  struct StructExtends<VideoEncodeH265CapabilitiesKHR, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265SessionCreateInfoKHR, VideoSessionCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265QualityLevelPropertiesKHR, VideoEncodeQualityLevelPropertiesKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265SessionParametersCreateInfoKHR, VideoSessionParametersCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265SessionParametersAddInfoKHR, VideoSessionParametersUpdateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265SessionParametersGetInfoKHR, VideoEncodeSessionParametersGetInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265SessionParametersFeedbackInfoKHR, VideoEncodeSessionParametersFeedbackInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265PictureInfoKHR, VideoEncodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265DpbSlotInfoKHR, VideoReferenceSlotInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265ProfileInfoKHR, VideoProfileInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265ProfileInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265RateControlInfoKHR, VideoCodingControlInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265RateControlInfoKHR, VideoBeginCodingInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265RateControlLayerInfoKHR, VideoEncodeRateControlLayerInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeH265GopRemainingFrameInfoKHR, VideoBeginCodingInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_video_decode_h264 ===
  template <>
  struct StructExtends<VideoDecodeH264ProfileInfoKHR, VideoProfileInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH264ProfileInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH264CapabilitiesKHR, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH264SessionParametersCreateInfoKHR, VideoSessionParametersCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH264SessionParametersAddInfoKHR, VideoSessionParametersUpdateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH264PictureInfoKHR, VideoDecodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH264DpbSlotInfoKHR, VideoReferenceSlotInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_AMD_texture_gather_bias_lod ===
  template <>
  struct StructExtends<TextureLODGatherFormatPropertiesAMD, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_dynamic_rendering ===
  template <>
  struct StructExtends<RenderingFragmentShadingRateAttachmentInfoKHR, RenderingInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<RenderingFragmentDensityMapAttachmentInfoEXT, RenderingInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AttachmentSampleCountInfoAMD, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AttachmentSampleCountInfoAMD, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MultiviewPerViewAttributesInfoNVX, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MultiviewPerViewAttributesInfoNVX, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MultiviewPerViewAttributesInfoNVX, RenderingInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_corner_sampled_image ===
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

  //=== VK_NV_external_memory ===
  template <>
  struct StructExtends<ExternalMemoryImageCreateInfoNV, ImageCreateInfo>
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

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_external_memory_win32 ===
  template <>
  struct StructExtends<ImportMemoryWin32HandleInfoNV, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMemoryWin32HandleInfoNV, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_win32_keyed_mutex ===
  template <>
  struct StructExtends<Win32KeyedMutexAcquireReleaseInfoNV, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<Win32KeyedMutexAcquireReleaseInfoNV, SubmitInfo2>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_validation_flags ===
  template <>
  struct StructExtends<ValidationFlagsEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_astc_decode_mode ===
  template <>
  struct StructExtends<ImageViewASTCDecodeModeEXT, ImageViewCreateInfo>
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

  //=== VK_EXT_pipeline_robustness ===
  template <>
  struct StructExtends<PhysicalDevicePipelineRobustnessFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePipelineRobustnessFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePipelineRobustnessPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineRobustnessCreateInfoEXT, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineRobustnessCreateInfoEXT, ComputePipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineRobustnessCreateInfoEXT, PipelineShaderStageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineRobustnessCreateInfoEXT, RayTracingPipelineCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_memory_win32 ===
  template <>
  struct StructExtends<ImportMemoryWin32HandleInfoKHR, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMemoryWin32HandleInfoKHR, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_memory_fd ===
  template <>
  struct StructExtends<ImportMemoryFdInfoKHR, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_keyed_mutex ===
  template <>
  struct StructExtends<Win32KeyedMutexAcquireReleaseInfoKHR, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<Win32KeyedMutexAcquireReleaseInfoKHR, SubmitInfo2>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_semaphore_win32 ===
  template <>
  struct StructExtends<ExportSemaphoreWin32HandleInfoKHR, SemaphoreCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<D3D12FenceSubmitInfoKHR, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_push_descriptor ===
  template <>
  struct StructExtends<PhysicalDevicePushDescriptorPropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_conditional_rendering ===
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
  struct StructExtends<CommandBufferInheritanceConditionalRenderingInfoEXT, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_incremental_present ===
  template <>
  struct StructExtends<PresentRegionsKHR, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_clip_space_w_scaling ===
  template <>
  struct StructExtends<PipelineViewportWScalingStateCreateInfoNV, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_display_control ===
  template <>
  struct StructExtends<SwapchainCounterCreateInfoEXT, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_GOOGLE_display_timing ===
  template <>
  struct StructExtends<PresentTimesInfoGOOGLE, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NVX_multiview_per_view_attributes ===
  template <>
  struct StructExtends<PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_viewport_swizzle ===
  template <>
  struct StructExtends<PipelineViewportSwizzleStateCreateInfoNV, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_discard_rectangles ===
  template <>
  struct StructExtends<PhysicalDeviceDiscardRectanglePropertiesEXT, PhysicalDeviceProperties2>
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

  //=== VK_EXT_conservative_rasterization ===
  template <>
  struct StructExtends<PhysicalDeviceConservativeRasterizationPropertiesEXT, PhysicalDeviceProperties2>
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

  //=== VK_EXT_depth_clip_enable ===
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
  struct StructExtends<PipelineRasterizationDepthClipStateCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_IMG_relaxed_line_rasterization ===
  template <>
  struct StructExtends<PhysicalDeviceRelaxedLineRasterizationFeaturesIMG, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRelaxedLineRasterizationFeaturesIMG, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_shared_presentable_image ===
  template <>
  struct StructExtends<SharedPresentSurfaceCapabilitiesKHR, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_fence_win32 ===
  template <>
  struct StructExtends<ExportFenceWin32HandleInfoKHR, FenceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_performance_query ===
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
  struct StructExtends<QueryPoolPerformanceCreateInfoKHR, QueryPoolCreateInfo>
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
  struct StructExtends<PerformanceQuerySubmitInfoKHR, SubmitInfo2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_debug_utils ===
  template <>
  struct StructExtends<DebugUtilsMessengerCreateInfoEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<DebugUtilsObjectNameInfoEXT, PipelineShaderStageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_memory_android_hardware_buffer ===
  template <>
  struct StructExtends<AndroidHardwareBufferUsageANDROID, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AndroidHardwareBufferFormatPropertiesANDROID, AndroidHardwareBufferPropertiesANDROID>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImportAndroidHardwareBufferInfoANDROID, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

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

  template <>
  struct StructExtends<ExternalFormatANDROID, AttachmentDescription2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExternalFormatANDROID, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExternalFormatANDROID, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AndroidHardwareBufferFormatProperties2ANDROID, AndroidHardwareBufferPropertiesANDROID>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_shader_enqueue ===
  template <>
  struct StructExtends<PhysicalDeviceShaderEnqueueFeaturesAMDX, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderEnqueueFeaturesAMDX, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderEnqueuePropertiesAMDX, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineShaderStageNodeCreateInfoAMDX, PipelineShaderStageCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_sample_locations ===
  template <>
  struct StructExtends<SampleLocationsInfoEXT, ImageMemoryBarrier>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SampleLocationsInfoEXT, ImageMemoryBarrier2>
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
  struct StructExtends<PipelineSampleLocationsStateCreateInfoEXT, PipelineMultisampleStateCreateInfo>
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

  //=== VK_EXT_blend_operation_advanced ===
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
  struct StructExtends<PipelineColorBlendAdvancedStateCreateInfoEXT, PipelineColorBlendStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_fragment_coverage_to_color ===
  template <>
  struct StructExtends<PipelineCoverageToColorStateCreateInfoNV, PipelineMultisampleStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_acceleration_structure ===
  template <>
  struct StructExtends<WriteDescriptorSetAccelerationStructureKHR, WriteDescriptorSet>
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

  //=== VK_KHR_ray_tracing_pipeline ===
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

  //=== VK_KHR_ray_query ===
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

  //=== VK_NV_framebuffer_mixed_samples ===
  template <>
  struct StructExtends<PipelineCoverageModulationStateCreateInfoNV, PipelineMultisampleStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_shader_sm_builtins ===
  template <>
  struct StructExtends<PhysicalDeviceShaderSMBuiltinsPropertiesNV, PhysicalDeviceProperties2>
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

  //=== VK_EXT_image_drm_format_modifier ===
  template <>
  struct StructExtends<DrmFormatModifierPropertiesListEXT, FormatProperties2>
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
  struct StructExtends<ImageDrmFormatModifierListCreateInfoEXT, ImageCreateInfo>
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
  struct StructExtends<DrmFormatModifierPropertiesList2EXT, FormatProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_validation_cache ===
  template <>
  struct StructExtends<ShaderModuleValidationCacheCreateInfoEXT, ShaderModuleCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ShaderModuleValidationCacheCreateInfoEXT, PipelineShaderStageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_portability_subset ===
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

  template <>
  struct StructExtends<PhysicalDevicePortabilitySubsetPropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_shading_rate_image ===
  template <>
  struct StructExtends<PipelineViewportShadingRateImageStateCreateInfoNV, PipelineViewportStateCreateInfo>
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
  struct StructExtends<PipelineViewportCoarseSampleOrderStateCreateInfoNV, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_ray_tracing ===
  template <>
  struct StructExtends<WriteDescriptorSetAccelerationStructureNV, WriteDescriptorSet>
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

  //=== VK_NV_representative_fragment_test ===
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
  struct StructExtends<PipelineRepresentativeFragmentTestStateCreateInfoNV, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_filter_cubic ===
  template <>
  struct StructExtends<PhysicalDeviceImageViewImageFormatInfoEXT, PhysicalDeviceImageFormatInfo2>
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

  //=== VK_EXT_external_memory_host ===
  template <>
  struct StructExtends<ImportMemoryHostPointerInfoEXT, MemoryAllocateInfo>
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

  //=== VK_KHR_shader_clock ===
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

  //=== VK_AMD_pipeline_compiler_control ===
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
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct StructExtends<PipelineCompilerControlCreateInfoAMD, ExecutionGraphPipelineCreateInfoAMDX>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_AMD_shader_core_properties ===
  template <>
  struct StructExtends<PhysicalDeviceShaderCorePropertiesAMD, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_video_decode_h265 ===
  template <>
  struct StructExtends<VideoDecodeH265ProfileInfoKHR, VideoProfileInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH265ProfileInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH265CapabilitiesKHR, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH265SessionParametersCreateInfoKHR, VideoSessionParametersCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH265SessionParametersAddInfoKHR, VideoSessionParametersUpdateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH265PictureInfoKHR, VideoDecodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoDecodeH265DpbSlotInfoKHR, VideoReferenceSlotInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_global_priority ===
  template <>
  struct StructExtends<DeviceQueueGlobalPriorityCreateInfoKHR, DeviceQueueCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceGlobalPriorityQueryFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceGlobalPriorityQueryFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<QueueFamilyGlobalPriorityPropertiesKHR, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_AMD_memory_overallocation_behavior ===
  template <>
  struct StructExtends<DeviceMemoryOverallocationCreateInfoAMD, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_vertex_attribute_divisor ===
  template <>
  struct StructExtends<PhysicalDeviceVertexAttributeDivisorPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_frame_token ===
  template <>
  struct StructExtends<PresentFrameTokenGGP, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_compute_shader_derivatives ===
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

  //=== VK_NV_mesh_shader ===
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

  //=== VK_NV_shader_image_footprint ===
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

  //=== VK_NV_scissor_exclusive ===
  template <>
  struct StructExtends<PipelineViewportExclusiveScissorStateCreateInfoNV, PipelineViewportStateCreateInfo>
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

  //=== VK_NV_device_diagnostic_checkpoints ===
  template <>
  struct StructExtends<QueueFamilyCheckpointPropertiesNV, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_INTEL_shader_integer_functions2 ===
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

  //=== VK_INTEL_performance_query ===
  template <>
  struct StructExtends<QueryPoolPerformanceQueryCreateInfoINTEL, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_pci_bus_info ===
  template <>
  struct StructExtends<PhysicalDevicePCIBusInfoPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_AMD_display_native_hdr ===
  template <>
  struct StructExtends<DisplayNativeHdrSurfaceCapabilitiesAMD, SurfaceCapabilities2KHR>
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

  //=== VK_EXT_fragment_density_map ===
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

  //=== VK_KHR_fragment_shading_rate ===
  template <>
  struct StructExtends<FragmentShadingRateAttachmentInfoKHR, SubpassDescription2>
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

  //=== VK_AMD_shader_core_properties2 ===
  template <>
  struct StructExtends<PhysicalDeviceShaderCoreProperties2AMD, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_AMD_device_coherent_memory ===
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

  //=== VK_EXT_shader_image_atomic_int64 ===
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

  //=== VK_EXT_memory_budget ===
  template <>
  struct StructExtends<PhysicalDeviceMemoryBudgetPropertiesEXT, PhysicalDeviceMemoryProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_memory_priority ===
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
  struct StructExtends<MemoryPriorityAllocateInfoEXT, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_surface_protected_capabilities ===
  template <>
  struct StructExtends<SurfaceProtectedCapabilitiesKHR, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_dedicated_allocation_image_aliasing ===
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

  //=== VK_EXT_buffer_device_address ===
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
  struct StructExtends<BufferDeviceAddressCreateInfoEXT, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_validation_features ===
  template <>
  struct StructExtends<ValidationFeaturesEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_present_wait ===
  template <>
  struct StructExtends<PhysicalDevicePresentWaitFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePresentWaitFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_cooperative_matrix ===
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

  //=== VK_NV_coverage_reduction_mode ===
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
  struct StructExtends<PipelineCoverageReductionStateCreateInfoNV, PipelineMultisampleStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_fragment_shader_interlock ===
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

  //=== VK_EXT_ycbcr_image_arrays ===
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

  //=== VK_EXT_provoking_vertex ===
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
  struct StructExtends<PipelineRasterizationProvokingVertexStateCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===
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

  template <>
  struct StructExtends<SurfaceCapabilitiesFullScreenExclusiveEXT, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };

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
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_line_rasterization ===
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
  struct StructExtends<PipelineRasterizationLineStateCreateInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_shader_atomic_float ===
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

  //=== VK_EXT_index_type_uint8 ===
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

  //=== VK_EXT_extended_dynamic_state ===
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

  //=== VK_KHR_pipeline_executable_properties ===
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

  //=== VK_EXT_host_image_copy ===
  template <>
  struct StructExtends<PhysicalDeviceHostImageCopyFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceHostImageCopyFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceHostImageCopyPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SubresourceHostMemcpySizeEXT, SubresourceLayout2KHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<HostImageCopyDevicePerformanceQueryEXT, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_shader_atomic_float2 ===
  template <>
  struct StructExtends<PhysicalDeviceShaderAtomicFloat2FeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderAtomicFloat2FeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_surface_maintenance1 ===
  template <>
  struct StructExtends<SurfacePresentModeEXT, PhysicalDeviceSurfaceInfo2KHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SurfacePresentScalingCapabilitiesEXT, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SurfacePresentModeCompatibilityEXT, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_swapchain_maintenance1 ===
  template <>
  struct StructExtends<PhysicalDeviceSwapchainMaintenance1FeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSwapchainMaintenance1FeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SwapchainPresentFenceInfoEXT, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SwapchainPresentModesCreateInfoEXT, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SwapchainPresentModeInfoEXT, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SwapchainPresentScalingCreateInfoEXT, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_device_generated_commands ===
  template <>
  struct StructExtends<PhysicalDeviceDeviceGeneratedCommandsPropertiesNV, PhysicalDeviceProperties2>
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
  struct StructExtends<GraphicsPipelineShaderGroupsCreateInfoNV, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_inherited_viewport_scissor ===
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
  struct StructExtends<CommandBufferInheritanceViewportScissorInfoNV, CommandBufferInheritanceInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_texel_buffer_alignment ===
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

  //=== VK_QCOM_render_pass_transform ===
  template <>
  struct StructExtends<RenderPassTransformBeginInfoQCOM, RenderPassBeginInfo>
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

  //=== VK_EXT_depth_bias_control ===
  template <>
  struct StructExtends<PhysicalDeviceDepthBiasControlFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDepthBiasControlFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<DepthBiasRepresentationInfoEXT, DepthBiasInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<DepthBiasRepresentationInfoEXT, PipelineRasterizationStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_device_memory_report ===
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
  struct StructExtends<DeviceDeviceMemoryReportCreateInfoEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_robustness2 ===
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

  //=== VK_EXT_custom_border_color ===
  template <>
  struct StructExtends<SamplerCustomBorderColorCreateInfoEXT, SamplerCreateInfo>
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

  //=== VK_KHR_pipeline_library ===
  template <>
  struct StructExtends<PipelineLibraryCreateInfoKHR, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_present_barrier ===
  template <>
  struct StructExtends<PhysicalDevicePresentBarrierFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePresentBarrierFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SurfaceCapabilitiesPresentBarrierNV, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SwapchainPresentBarrierCreateInfoNV, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_present_id ===
  template <>
  struct StructExtends<PresentIdKHR, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePresentIdFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePresentIdFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_video_encode_queue ===
  template <>
  struct StructExtends<VideoEncodeCapabilitiesKHR, VideoCapabilitiesKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<QueryPoolVideoEncodeFeedbackCreateInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeUsageInfoKHR, VideoProfileInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeUsageInfoKHR, QueryPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeRateControlInfoKHR, VideoCodingControlInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeRateControlInfoKHR, VideoBeginCodingInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeQualityLevelInfoKHR, VideoCodingControlInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoEncodeQualityLevelInfoKHR, VideoSessionParametersCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_device_diagnostics_config ===
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
  struct StructExtends<DeviceDiagnosticsConfigCreateInfoNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  template <>
  struct StructExtends<PhysicalDeviceCudaKernelLaunchFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceCudaKernelLaunchFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceCudaKernelLaunchPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_low_latency ===
  template <>
  struct StructExtends<QueryLowLatencySupportNV, SemaphoreCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===
  template <>
  struct StructExtends<ExportMetalObjectCreateInfoEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalObjectCreateInfoEXT, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalObjectCreateInfoEXT, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalObjectCreateInfoEXT, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalObjectCreateInfoEXT, BufferViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalObjectCreateInfoEXT, SemaphoreCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalObjectCreateInfoEXT, EventCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalDeviceInfoEXT, ExportMetalObjectsInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalCommandQueueInfoEXT, ExportMetalObjectsInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalBufferInfoEXT, ExportMetalObjectsInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImportMetalBufferInfoEXT, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalTextureInfoEXT, ExportMetalObjectsInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImportMetalTextureInfoEXT, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalIOSurfaceInfoEXT, ExportMetalObjectsInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImportMetalIOSurfaceInfoEXT, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExportMetalSharedEventInfoEXT, ExportMetalObjectsInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImportMetalSharedEventInfoEXT, SemaphoreCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImportMetalSharedEventInfoEXT, EventCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_synchronization2 ===
  template <>
  struct StructExtends<QueueFamilyCheckpointProperties2NV, QueueFamilyProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_descriptor_buffer ===
  template <>
  struct StructExtends<PhysicalDeviceDescriptorBufferPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDescriptorBufferDensityMapPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDescriptorBufferFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDescriptorBufferFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<DescriptorBufferBindingPushDescriptorBufferHandleEXT, DescriptorBufferBindingInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpaqueCaptureDescriptorDataCreateInfoEXT, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpaqueCaptureDescriptorDataCreateInfoEXT, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpaqueCaptureDescriptorDataCreateInfoEXT, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpaqueCaptureDescriptorDataCreateInfoEXT, SamplerCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpaqueCaptureDescriptorDataCreateInfoEXT, AccelerationStructureCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpaqueCaptureDescriptorDataCreateInfoEXT, AccelerationStructureCreateInfoNV>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_graphics_pipeline_library ===
  template <>
  struct StructExtends<PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<GraphicsPipelineLibraryCreateInfoEXT, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_AMD_shader_early_and_late_fragment_tests ===
  template <>
  struct StructExtends<PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_fragment_shader_barycentric ===
  template <>
  struct StructExtends<PhysicalDeviceFragmentShaderBarycentricFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceFragmentShaderBarycentricFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceFragmentShaderBarycentricPropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_shader_subgroup_uniform_control_flow ===
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

  //=== VK_NV_fragment_shading_rate_enums ===
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
  struct StructExtends<PipelineFragmentShadingRateEnumStateCreateInfoNV, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_ray_tracing_motion_blur ===
  template <>
  struct StructExtends<AccelerationStructureGeometryMotionTrianglesDataNV, AccelerationStructureGeometryTrianglesDataKHR>
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

  template <>
  struct StructExtends<PhysicalDeviceRayTracingMotionBlurFeaturesNV, PhysicalDeviceFeatures2>
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

  //=== VK_EXT_mesh_shader ===
  template <>
  struct StructExtends<PhysicalDeviceMeshShaderFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMeshShaderFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMeshShaderPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_ycbcr_2plane_444_formats ===
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

  //=== VK_EXT_fragment_density_map2 ===
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

  //=== VK_QCOM_rotated_copy_commands ===
  template <>
  struct StructExtends<CopyCommandTransformInfoQCOM, BufferImageCopy2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<CopyCommandTransformInfoQCOM, ImageBlit2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_workgroup_memory_explicit_layout ===
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

  //=== VK_EXT_image_compression_control ===
  template <>
  struct StructExtends<PhysicalDeviceImageCompressionControlFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageCompressionControlFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImageCompressionControlEXT, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImageCompressionControlEXT, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImageCompressionControlEXT, PhysicalDeviceImageFormatInfo2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImageCompressionPropertiesEXT, ImageFormatProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImageCompressionPropertiesEXT, SurfaceFormat2KHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImageCompressionPropertiesEXT, SubresourceLayout2KHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_attachment_feedback_loop_layout ===
  template <>
  struct StructExtends<PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_4444_formats ===
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

  //=== VK_EXT_device_fault ===
  template <>
  struct StructExtends<PhysicalDeviceFaultFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceFaultFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_rgba10x6_formats ===
  template <>
  struct StructExtends<PhysicalDeviceRGBA10X6FormatsFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRGBA10X6FormatsFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_vertex_input_dynamic_state ===
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

  //=== VK_EXT_physical_device_drm ===
  template <>
  struct StructExtends<PhysicalDeviceDrmPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_device_address_binding_report ===
  template <>
  struct StructExtends<PhysicalDeviceAddressBindingReportFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceAddressBindingReportFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<DeviceAddressBindingCallbackDataEXT, DebugUtilsMessengerCallbackDataEXT>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_depth_clip_control ===
  template <>
  struct StructExtends<PhysicalDeviceDepthClipControlFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDepthClipControlFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineViewportDepthClipControlCreateInfoEXT, PipelineViewportStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_primitive_topology_list_restart ===
  template <>
  struct StructExtends<PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_memory ===
  template <>
  struct StructExtends<ImportMemoryZirconHandleInfoFUCHSIA, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  template <>
  struct StructExtends<ImportMemoryBufferCollectionFUCHSIA, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BufferCollectionImageCreateInfoFUCHSIA, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BufferCollectionBufferCreateInfoFUCHSIA, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_HUAWEI_subpass_shading ===
  template <>
  struct StructExtends<SubpassShadingPipelineCreateInfoHUAWEI, ComputePipelineCreateInfo>
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

  //=== VK_HUAWEI_invocation_mask ===
  template <>
  struct StructExtends<PhysicalDeviceInvocationMaskFeaturesHUAWEI, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceInvocationMaskFeaturesHUAWEI, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_external_memory_rdma ===
  template <>
  struct StructExtends<PhysicalDeviceExternalMemoryRDMAFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExternalMemoryRDMAFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_pipeline_properties ===
  template <>
  struct StructExtends<PhysicalDevicePipelinePropertiesFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePipelinePropertiesFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_frame_boundary ===
  template <>
  struct StructExtends<PhysicalDeviceFrameBoundaryFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceFrameBoundaryFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<FrameBoundaryEXT, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<FrameBoundaryEXT, SubmitInfo2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<FrameBoundaryEXT, PresentInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<FrameBoundaryEXT, BindSparseInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_multisampled_render_to_single_sampled ===
  template <>
  struct StructExtends<PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SubpassResolvePerformanceQueryEXT, FormatProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MultisampledRenderToSingleSampledInfoEXT, SubpassDescription2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MultisampledRenderToSingleSampledInfoEXT, RenderingInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_extended_dynamic_state2 ===
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

  //=== VK_EXT_color_write_enable ===
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
  struct StructExtends<PipelineColorWriteCreateInfoEXT, PipelineColorBlendStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_primitives_generated_query ===
  template <>
  struct StructExtends<PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_ray_tracing_maintenance1 ===
  template <>
  struct StructExtends<PhysicalDeviceRayTracingMaintenance1FeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRayTracingMaintenance1FeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_image_view_min_lod ===
  template <>
  struct StructExtends<PhysicalDeviceImageViewMinLodFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageViewMinLodFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImageViewMinLodCreateInfoEXT, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_multi_draw ===
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

  //=== VK_EXT_image_2d_view_of_3d ===
  template <>
  struct StructExtends<PhysicalDeviceImage2DViewOf3DFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImage2DViewOf3DFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_shader_tile_image ===
  template <>
  struct StructExtends<PhysicalDeviceShaderTileImageFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderTileImageFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderTileImagePropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_opacity_micromap ===
  template <>
  struct StructExtends<PhysicalDeviceOpacityMicromapFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceOpacityMicromapFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceOpacityMicromapPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AccelerationStructureTrianglesOpacityMicromapEXT, AccelerationStructureGeometryTrianglesDataKHR>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===
  template <>
  struct StructExtends<PhysicalDeviceDisplacementMicromapFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDisplacementMicromapFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDisplacementMicromapPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AccelerationStructureTrianglesDisplacementMicromapNV, AccelerationStructureGeometryTrianglesDataKHR>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_HUAWEI_cluster_culling_shader ===
  template <>
  struct StructExtends<PhysicalDeviceClusterCullingShaderFeaturesHUAWEI, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceClusterCullingShaderFeaturesHUAWEI, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceClusterCullingShaderPropertiesHUAWEI, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI, PhysicalDeviceClusterCullingShaderFeaturesHUAWEI>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_border_color_swizzle ===
  template <>
  struct StructExtends<PhysicalDeviceBorderColorSwizzleFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceBorderColorSwizzleFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SamplerBorderColorComponentMappingCreateInfoEXT, SamplerCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_pageable_device_local_memory ===
  template <>
  struct StructExtends<PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_ARM_shader_core_properties ===
  template <>
  struct StructExtends<PhysicalDeviceShaderCorePropertiesARM, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_ARM_scheduling_controls ===
  template <>
  struct StructExtends<DeviceQueueShaderCoreControlCreateInfoARM, DeviceQueueCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<DeviceQueueShaderCoreControlCreateInfoARM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSchedulingControlsFeaturesARM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSchedulingControlsFeaturesARM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSchedulingControlsPropertiesARM, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_image_sliced_view_of_3d ===
  template <>
  struct StructExtends<PhysicalDeviceImageSlicedViewOf3DFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageSlicedViewOf3DFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImageViewSlicedCreateInfoEXT, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_VALVE_descriptor_set_host_mapping ===
  template <>
  struct StructExtends<PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_depth_clamp_zero_one ===
  template <>
  struct StructExtends<PhysicalDeviceDepthClampZeroOneFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDepthClampZeroOneFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_non_seamless_cube_map ===
  template <>
  struct StructExtends<PhysicalDeviceNonSeamlessCubeMapFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceNonSeamlessCubeMapFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_ARM_render_pass_striped ===
  template <>
  struct StructExtends<PhysicalDeviceRenderPassStripedFeaturesARM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRenderPassStripedFeaturesARM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRenderPassStripedPropertiesARM, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<RenderPassStripeBeginInfoARM, RenderingInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<RenderPassStripeBeginInfoARM, RenderPassBeginInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<RenderPassStripeSubmitInfoARM, CommandBufferSubmitInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_fragment_density_map_offset ===
  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SubpassFragmentDensityMapOffsetEndInfoQCOM, SubpassEndInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_copy_memory_indirect ===
  template <>
  struct StructExtends<PhysicalDeviceCopyMemoryIndirectFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceCopyMemoryIndirectFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceCopyMemoryIndirectPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_memory_decompression ===
  template <>
  struct StructExtends<PhysicalDeviceMemoryDecompressionFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMemoryDecompressionFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMemoryDecompressionPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_device_generated_commands_compute ===
  template <>
  struct StructExtends<PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_linear_color_attachment ===
  template <>
  struct StructExtends<PhysicalDeviceLinearColorAttachmentFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceLinearColorAttachmentFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_image_compression_control_swapchain ===
  template <>
  struct StructExtends<PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_image_processing ===
  template <>
  struct StructExtends<ImageViewSampleWeightCreateInfoQCOM, ImageViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageProcessingFeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageProcessingFeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageProcessingPropertiesQCOM, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_nested_command_buffer ===
  template <>
  struct StructExtends<PhysicalDeviceNestedCommandBufferFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceNestedCommandBufferFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceNestedCommandBufferPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_external_memory_acquire_unmodified ===
  template <>
  struct StructExtends<ExternalMemoryAcquireUnmodifiedEXT, BufferMemoryBarrier>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExternalMemoryAcquireUnmodifiedEXT, BufferMemoryBarrier2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExternalMemoryAcquireUnmodifiedEXT, ImageMemoryBarrier>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExternalMemoryAcquireUnmodifiedEXT, ImageMemoryBarrier2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_extended_dynamic_state3 ===
  template <>
  struct StructExtends<PhysicalDeviceExtendedDynamicState3FeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExtendedDynamicState3FeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExtendedDynamicState3PropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_subpass_merge_feedback ===
  template <>
  struct StructExtends<PhysicalDeviceSubpassMergeFeedbackFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceSubpassMergeFeedbackFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<RenderPassCreationControlEXT, RenderPassCreateInfo2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<RenderPassCreationControlEXT, SubpassDescription2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<RenderPassCreationFeedbackCreateInfoEXT, RenderPassCreateInfo2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<RenderPassSubpassFeedbackCreateInfoEXT, SubpassDescription2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_LUNARG_direct_driver_loading ===
  template <>
  struct StructExtends<DirectDriverLoadingListLUNARG, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_shader_module_identifier ===
  template <>
  struct StructExtends<PhysicalDeviceShaderModuleIdentifierFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderModuleIdentifierFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderModuleIdentifierPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineShaderStageModuleIdentifierCreateInfoEXT, PipelineShaderStageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_rasterization_order_attachment_access ===
  template <>
  struct StructExtends<PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_optical_flow ===
  template <>
  struct StructExtends<PhysicalDeviceOpticalFlowFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceOpticalFlowFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceOpticalFlowPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpticalFlowImageFormatInfoNV, PhysicalDeviceImageFormatInfo2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpticalFlowImageFormatInfoNV, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<OpticalFlowSessionCreatePrivateDataInfoNV, OpticalFlowSessionCreateInfoNV>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_legacy_dithering ===
  template <>
  struct StructExtends<PhysicalDeviceLegacyDitheringFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceLegacyDitheringFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_pipeline_protected_access ===
  template <>
  struct StructExtends<PhysicalDevicePipelineProtectedAccessFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePipelineProtectedAccessFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_format_resolve ===
  template <>
  struct StructExtends<PhysicalDeviceExternalFormatResolveFeaturesANDROID, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExternalFormatResolveFeaturesANDROID, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExternalFormatResolvePropertiesANDROID, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AndroidHardwareBufferFormatResolvePropertiesANDROID, AndroidHardwareBufferPropertiesANDROID>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  //=== VK_KHR_maintenance5 ===
  template <>
  struct StructExtends<PhysicalDeviceMaintenance5FeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMaintenance5FeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMaintenance5PropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineCreateFlags2CreateInfoKHR, ComputePipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineCreateFlags2CreateInfoKHR, GraphicsPipelineCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineCreateFlags2CreateInfoKHR, RayTracingPipelineCreateInfoNV>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineCreateFlags2CreateInfoKHR, RayTracingPipelineCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BufferUsageFlags2CreateInfoKHR, BufferViewCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BufferUsageFlags2CreateInfoKHR, BufferCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BufferUsageFlags2CreateInfoKHR, PhysicalDeviceExternalBufferInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BufferUsageFlags2CreateInfoKHR, DescriptorBufferBindingInfoEXT>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_ray_tracing_position_fetch ===
  template <>
  struct StructExtends<PhysicalDeviceRayTracingPositionFetchFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRayTracingPositionFetchFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_shader_object ===
  template <>
  struct StructExtends<PhysicalDeviceShaderObjectFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderObjectFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderObjectPropertiesEXT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_tile_properties ===
  template <>
  struct StructExtends<PhysicalDeviceTilePropertiesFeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceTilePropertiesFeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_SEC_amigo_profiling ===
  template <>
  struct StructExtends<PhysicalDeviceAmigoProfilingFeaturesSEC, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceAmigoProfilingFeaturesSEC, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<AmigoProfilingSubmitInfoSEC, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_multiview_per_view_viewports ===
  template <>
  struct StructExtends<PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_ray_tracing_invocation_reorder ===
  template <>
  struct StructExtends<PhysicalDeviceRayTracingInvocationReorderPropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRayTracingInvocationReorderFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceRayTracingInvocationReorderFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_extended_sparse_address_space ===
  template <>
  struct StructExtends<PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExtendedSparseAddressSpacePropertiesNV, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_mutable_descriptor_type ===
  template <>
  struct StructExtends<PhysicalDeviceMutableDescriptorTypeFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMutableDescriptorTypeFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MutableDescriptorTypeCreateInfoEXT, DescriptorSetLayoutCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MutableDescriptorTypeCreateInfoEXT, DescriptorPoolCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_layer_settings ===
  template <>
  struct StructExtends<LayerSettingsCreateInfoEXT, InstanceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_ARM_shader_core_builtins ===
  template <>
  struct StructExtends<PhysicalDeviceShaderCoreBuiltinsFeaturesARM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderCoreBuiltinsFeaturesARM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceShaderCoreBuiltinsPropertiesARM, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_pipeline_library_group_handles ===
  template <>
  struct StructExtends<PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_dynamic_rendering_unused_attachments ===
  template <>
  struct StructExtends<PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_low_latency2 ===
  template <>
  struct StructExtends<LatencySubmissionPresentIdNV, SubmitInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<LatencySubmissionPresentIdNV, SubmitInfo2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SwapchainLatencyCreateInfoNV, SwapchainCreateInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<LatencySurfaceCapabilitiesNV, SurfaceCapabilities2KHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_cooperative_matrix ===
  template <>
  struct StructExtends<PhysicalDeviceCooperativeMatrixFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceCooperativeMatrixFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceCooperativeMatrixPropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_multiview_per_view_render_areas ===
  template <>
  struct StructExtends<PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM, RenderPassBeginInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM, RenderingInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_video_maintenance1 ===
  template <>
  struct StructExtends<PhysicalDeviceVideoMaintenance1FeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceVideoMaintenance1FeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoInlineQueryInfoKHR, VideoDecodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<VideoInlineQueryInfoKHR, VideoEncodeInfoKHR>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_per_stage_descriptor_set ===
  template <>
  struct StructExtends<PhysicalDevicePerStageDescriptorSetFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDevicePerStageDescriptorSetFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_image_processing2 ===
  template <>
  struct StructExtends<PhysicalDeviceImageProcessing2FeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageProcessing2FeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceImageProcessing2PropertiesQCOM, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SamplerBlockMatchWindowCreateInfoQCOM, SamplerCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_filter_cubic_weights ===
  template <>
  struct StructExtends<PhysicalDeviceCubicWeightsFeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceCubicWeightsFeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SamplerCubicWeightsCreateInfoQCOM, SamplerCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BlitImageCubicWeightsInfoQCOM, BlitImageInfo2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_ycbcr_degamma ===
  template <>
  struct StructExtends<PhysicalDeviceYcbcrDegammaFeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceYcbcrDegammaFeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<SamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM, SamplerYcbcrConversionCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_QCOM_filter_cubic_clamp ===
  template <>
  struct StructExtends<PhysicalDeviceCubicClampFeaturesQCOM, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceCubicClampFeaturesQCOM, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
  template <>
  struct StructExtends<PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_vertex_attribute_divisor ===
  template <>
  struct StructExtends<PhysicalDeviceVertexAttributeDivisorPropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PipelineVertexInputDivisorStateCreateInfoKHR, PipelineVertexInputStateCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceVertexAttributeDivisorFeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceVertexAttributeDivisorFeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_external_memory_screen_buffer ===
  template <>
  struct StructExtends<ScreenBufferFormatPropertiesQNX, ScreenBufferPropertiesQNX>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ImportScreenBufferInfoQNX, MemoryAllocateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExternalFormatQNX, ImageCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<ExternalFormatQNX, SamplerYcbcrConversionCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_MSFT_layered_driver ===
  template <>
  struct StructExtends<PhysicalDeviceLayeredDriverPropertiesMSFT, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_KHR_maintenance6 ===
  template <>
  struct StructExtends<PhysicalDeviceMaintenance6FeaturesKHR, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMaintenance6FeaturesKHR, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceMaintenance6PropertiesKHR, PhysicalDeviceProperties2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BindMemoryStatusKHR, BindBufferMemoryInfo>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<BindMemoryStatusKHR, BindImageMemoryInfo>
  {
    enum
    {
      value = true
    };
  };

  //=== VK_NV_descriptor_pool_overallocation ===
  template <>
  struct StructExtends<PhysicalDeviceDescriptorPoolOverallocationFeaturesNV, PhysicalDeviceFeatures2>
  {
    enum
    {
      value = true
    };
  };

  template <>
  struct StructExtends<PhysicalDeviceDescriptorPoolOverallocationFeaturesNV, DeviceCreateInfo>
  {
    enum
    {
      value = true
    };
  };

#endif  // VULKAN_HPP_DISABLE_ENHANCED_MODE

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
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNX__ ) || defined( __Fuchsia__ )
        m_library = dlopen( vulkanLibraryName.c_str(), RTLD_NOW | RTLD_LOCAL );
#  elif defined( _WIN32 )
        m_library = ::LoadLibraryA( vulkanLibraryName.c_str() );
#  else
#    error unsupported platform
#  endif
      }
      else
      {
#  if defined( __unix__ ) || defined( __QNX__ ) || defined( __Fuchsia__ )
        m_library = dlopen( "libvulkan.so", RTLD_NOW | RTLD_LOCAL );
        if ( m_library == nullptr )
        {
          m_library = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
        }
#  elif defined( __APPLE__ )
        m_library = dlopen( "libvulkan.dylib", RTLD_NOW | RTLD_LOCAL );
        if ( m_library == nullptr )
        {
          m_library = dlopen( "libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL );
        }
#  elif defined( _WIN32 )
        m_library = ::LoadLibraryA( "vulkan-1.dll" );
#  else
#    error unsupported platform
#  endif
      }

#  ifndef VULKAN_HPP_NO_EXCEPTIONS
      if ( m_library == nullptr )
      {
        // NOTE there should be an InitializationFailedError, but msvc insists on the symbol does not exist within the scope of this function.
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
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNX__ ) || defined( __Fuchsia__ )
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
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNX__ ) || defined( __Fuchsia__ )
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
#  if defined( __unix__ ) || defined( __APPLE__ ) || defined( __QNX__ ) || defined( __Fuchsia__ )
    void * m_library;
#  elif defined( _WIN32 )
    ::HINSTANCE m_library;
#  else
#    error unsupported platform
#  endif
  };
#endif

  using PFN_dummy = void ( * )();

  class DispatchLoaderDynamic : public DispatchLoaderBase
  {
  public:
    //=== VK_VERSION_1_0 ===
    PFN_vkCreateInstance                               vkCreateInstance                               = 0;
    PFN_vkDestroyInstance                              vkDestroyInstance                              = 0;
    PFN_vkEnumeratePhysicalDevices                     vkEnumeratePhysicalDevices                     = 0;
    PFN_vkGetPhysicalDeviceFeatures                    vkGetPhysicalDeviceFeatures                    = 0;
    PFN_vkGetPhysicalDeviceFormatProperties            vkGetPhysicalDeviceFormatProperties            = 0;
    PFN_vkGetPhysicalDeviceImageFormatProperties       vkGetPhysicalDeviceImageFormatProperties       = 0;
    PFN_vkGetPhysicalDeviceProperties                  vkGetPhysicalDeviceProperties                  = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties       vkGetPhysicalDeviceQueueFamilyProperties       = 0;
    PFN_vkGetPhysicalDeviceMemoryProperties            vkGetPhysicalDeviceMemoryProperties            = 0;
    PFN_vkGetInstanceProcAddr                          vkGetInstanceProcAddr                          = 0;
    PFN_vkGetDeviceProcAddr                            vkGetDeviceProcAddr                            = 0;
    PFN_vkCreateDevice                                 vkCreateDevice                                 = 0;
    PFN_vkDestroyDevice                                vkDestroyDevice                                = 0;
    PFN_vkEnumerateInstanceExtensionProperties         vkEnumerateInstanceExtensionProperties         = 0;
    PFN_vkEnumerateDeviceExtensionProperties           vkEnumerateDeviceExtensionProperties           = 0;
    PFN_vkEnumerateInstanceLayerProperties             vkEnumerateInstanceLayerProperties             = 0;
    PFN_vkEnumerateDeviceLayerProperties               vkEnumerateDeviceLayerProperties               = 0;
    PFN_vkGetDeviceQueue                               vkGetDeviceQueue                               = 0;
    PFN_vkQueueSubmit                                  vkQueueSubmit                                  = 0;
    PFN_vkQueueWaitIdle                                vkQueueWaitIdle                                = 0;
    PFN_vkDeviceWaitIdle                               vkDeviceWaitIdle                               = 0;
    PFN_vkAllocateMemory                               vkAllocateMemory                               = 0;
    PFN_vkFreeMemory                                   vkFreeMemory                                   = 0;
    PFN_vkMapMemory                                    vkMapMemory                                    = 0;
    PFN_vkUnmapMemory                                  vkUnmapMemory                                  = 0;
    PFN_vkFlushMappedMemoryRanges                      vkFlushMappedMemoryRanges                      = 0;
    PFN_vkInvalidateMappedMemoryRanges                 vkInvalidateMappedMemoryRanges                 = 0;
    PFN_vkGetDeviceMemoryCommitment                    vkGetDeviceMemoryCommitment                    = 0;
    PFN_vkBindBufferMemory                             vkBindBufferMemory                             = 0;
    PFN_vkBindImageMemory                              vkBindImageMemory                              = 0;
    PFN_vkGetBufferMemoryRequirements                  vkGetBufferMemoryRequirements                  = 0;
    PFN_vkGetImageMemoryRequirements                   vkGetImageMemoryRequirements                   = 0;
    PFN_vkGetImageSparseMemoryRequirements             vkGetImageSparseMemoryRequirements             = 0;
    PFN_vkGetPhysicalDeviceSparseImageFormatProperties vkGetPhysicalDeviceSparseImageFormatProperties = 0;
    PFN_vkQueueBindSparse                              vkQueueBindSparse                              = 0;
    PFN_vkCreateFence                                  vkCreateFence                                  = 0;
    PFN_vkDestroyFence                                 vkDestroyFence                                 = 0;
    PFN_vkResetFences                                  vkResetFences                                  = 0;
    PFN_vkGetFenceStatus                               vkGetFenceStatus                               = 0;
    PFN_vkWaitForFences                                vkWaitForFences                                = 0;
    PFN_vkCreateSemaphore                              vkCreateSemaphore                              = 0;
    PFN_vkDestroySemaphore                             vkDestroySemaphore                             = 0;
    PFN_vkCreateEvent                                  vkCreateEvent                                  = 0;
    PFN_vkDestroyEvent                                 vkDestroyEvent                                 = 0;
    PFN_vkGetEventStatus                               vkGetEventStatus                               = 0;
    PFN_vkSetEvent                                     vkSetEvent                                     = 0;
    PFN_vkResetEvent                                   vkResetEvent                                   = 0;
    PFN_vkCreateQueryPool                              vkCreateQueryPool                              = 0;
    PFN_vkDestroyQueryPool                             vkDestroyQueryPool                             = 0;
    PFN_vkGetQueryPoolResults                          vkGetQueryPoolResults                          = 0;
    PFN_vkCreateBuffer                                 vkCreateBuffer                                 = 0;
    PFN_vkDestroyBuffer                                vkDestroyBuffer                                = 0;
    PFN_vkCreateBufferView                             vkCreateBufferView                             = 0;
    PFN_vkDestroyBufferView                            vkDestroyBufferView                            = 0;
    PFN_vkCreateImage                                  vkCreateImage                                  = 0;
    PFN_vkDestroyImage                                 vkDestroyImage                                 = 0;
    PFN_vkGetImageSubresourceLayout                    vkGetImageSubresourceLayout                    = 0;
    PFN_vkCreateImageView                              vkCreateImageView                              = 0;
    PFN_vkDestroyImageView                             vkDestroyImageView                             = 0;
    PFN_vkCreateShaderModule                           vkCreateShaderModule                           = 0;
    PFN_vkDestroyShaderModule                          vkDestroyShaderModule                          = 0;
    PFN_vkCreatePipelineCache                          vkCreatePipelineCache                          = 0;
    PFN_vkDestroyPipelineCache                         vkDestroyPipelineCache                         = 0;
    PFN_vkGetPipelineCacheData                         vkGetPipelineCacheData                         = 0;
    PFN_vkMergePipelineCaches                          vkMergePipelineCaches                          = 0;
    PFN_vkCreateGraphicsPipelines                      vkCreateGraphicsPipelines                      = 0;
    PFN_vkCreateComputePipelines                       vkCreateComputePipelines                       = 0;
    PFN_vkDestroyPipeline                              vkDestroyPipeline                              = 0;
    PFN_vkCreatePipelineLayout                         vkCreatePipelineLayout                         = 0;
    PFN_vkDestroyPipelineLayout                        vkDestroyPipelineLayout                        = 0;
    PFN_vkCreateSampler                                vkCreateSampler                                = 0;
    PFN_vkDestroySampler                               vkDestroySampler                               = 0;
    PFN_vkCreateDescriptorSetLayout                    vkCreateDescriptorSetLayout                    = 0;
    PFN_vkDestroyDescriptorSetLayout                   vkDestroyDescriptorSetLayout                   = 0;
    PFN_vkCreateDescriptorPool                         vkCreateDescriptorPool                         = 0;
    PFN_vkDestroyDescriptorPool                        vkDestroyDescriptorPool                        = 0;
    PFN_vkResetDescriptorPool                          vkResetDescriptorPool                          = 0;
    PFN_vkAllocateDescriptorSets                       vkAllocateDescriptorSets                       = 0;
    PFN_vkFreeDescriptorSets                           vkFreeDescriptorSets                           = 0;
    PFN_vkUpdateDescriptorSets                         vkUpdateDescriptorSets                         = 0;
    PFN_vkCreateFramebuffer                            vkCreateFramebuffer                            = 0;
    PFN_vkDestroyFramebuffer                           vkDestroyFramebuffer                           = 0;
    PFN_vkCreateRenderPass                             vkCreateRenderPass                             = 0;
    PFN_vkDestroyRenderPass                            vkDestroyRenderPass                            = 0;
    PFN_vkGetRenderAreaGranularity                     vkGetRenderAreaGranularity                     = 0;
    PFN_vkCreateCommandPool                            vkCreateCommandPool                            = 0;
    PFN_vkDestroyCommandPool                           vkDestroyCommandPool                           = 0;
    PFN_vkResetCommandPool                             vkResetCommandPool                             = 0;
    PFN_vkAllocateCommandBuffers                       vkAllocateCommandBuffers                       = 0;
    PFN_vkFreeCommandBuffers                           vkFreeCommandBuffers                           = 0;
    PFN_vkBeginCommandBuffer                           vkBeginCommandBuffer                           = 0;
    PFN_vkEndCommandBuffer                             vkEndCommandBuffer                             = 0;
    PFN_vkResetCommandBuffer                           vkResetCommandBuffer                           = 0;
    PFN_vkCmdBindPipeline                              vkCmdBindPipeline                              = 0;
    PFN_vkCmdSetViewport                               vkCmdSetViewport                               = 0;
    PFN_vkCmdSetScissor                                vkCmdSetScissor                                = 0;
    PFN_vkCmdSetLineWidth                              vkCmdSetLineWidth                              = 0;
    PFN_vkCmdSetDepthBias                              vkCmdSetDepthBias                              = 0;
    PFN_vkCmdSetBlendConstants                         vkCmdSetBlendConstants                         = 0;
    PFN_vkCmdSetDepthBounds                            vkCmdSetDepthBounds                            = 0;
    PFN_vkCmdSetStencilCompareMask                     vkCmdSetStencilCompareMask                     = 0;
    PFN_vkCmdSetStencilWriteMask                       vkCmdSetStencilWriteMask                       = 0;
    PFN_vkCmdSetStencilReference                       vkCmdSetStencilReference                       = 0;
    PFN_vkCmdBindDescriptorSets                        vkCmdBindDescriptorSets                        = 0;
    PFN_vkCmdBindIndexBuffer                           vkCmdBindIndexBuffer                           = 0;
    PFN_vkCmdBindVertexBuffers                         vkCmdBindVertexBuffers                         = 0;
    PFN_vkCmdDraw                                      vkCmdDraw                                      = 0;
    PFN_vkCmdDrawIndexed                               vkCmdDrawIndexed                               = 0;
    PFN_vkCmdDrawIndirect                              vkCmdDrawIndirect                              = 0;
    PFN_vkCmdDrawIndexedIndirect                       vkCmdDrawIndexedIndirect                       = 0;
    PFN_vkCmdDispatch                                  vkCmdDispatch                                  = 0;
    PFN_vkCmdDispatchIndirect                          vkCmdDispatchIndirect                          = 0;
    PFN_vkCmdCopyBuffer                                vkCmdCopyBuffer                                = 0;
    PFN_vkCmdCopyImage                                 vkCmdCopyImage                                 = 0;
    PFN_vkCmdBlitImage                                 vkCmdBlitImage                                 = 0;
    PFN_vkCmdCopyBufferToImage                         vkCmdCopyBufferToImage                         = 0;
    PFN_vkCmdCopyImageToBuffer                         vkCmdCopyImageToBuffer                         = 0;
    PFN_vkCmdUpdateBuffer                              vkCmdUpdateBuffer                              = 0;
    PFN_vkCmdFillBuffer                                vkCmdFillBuffer                                = 0;
    PFN_vkCmdClearColorImage                           vkCmdClearColorImage                           = 0;
    PFN_vkCmdClearDepthStencilImage                    vkCmdClearDepthStencilImage                    = 0;
    PFN_vkCmdClearAttachments                          vkCmdClearAttachments                          = 0;
    PFN_vkCmdResolveImage                              vkCmdResolveImage                              = 0;
    PFN_vkCmdSetEvent                                  vkCmdSetEvent                                  = 0;
    PFN_vkCmdResetEvent                                vkCmdResetEvent                                = 0;
    PFN_vkCmdWaitEvents                                vkCmdWaitEvents                                = 0;
    PFN_vkCmdPipelineBarrier                           vkCmdPipelineBarrier                           = 0;
    PFN_vkCmdBeginQuery                                vkCmdBeginQuery                                = 0;
    PFN_vkCmdEndQuery                                  vkCmdEndQuery                                  = 0;
    PFN_vkCmdResetQueryPool                            vkCmdResetQueryPool                            = 0;
    PFN_vkCmdWriteTimestamp                            vkCmdWriteTimestamp                            = 0;
    PFN_vkCmdCopyQueryPoolResults                      vkCmdCopyQueryPoolResults                      = 0;
    PFN_vkCmdPushConstants                             vkCmdPushConstants                             = 0;
    PFN_vkCmdBeginRenderPass                           vkCmdBeginRenderPass                           = 0;
    PFN_vkCmdNextSubpass                               vkCmdNextSubpass                               = 0;
    PFN_vkCmdEndRenderPass                             vkCmdEndRenderPass                             = 0;
    PFN_vkCmdExecuteCommands                           vkCmdExecuteCommands                           = 0;

    //=== VK_VERSION_1_1 ===
    PFN_vkEnumerateInstanceVersion                      vkEnumerateInstanceVersion                      = 0;
    PFN_vkBindBufferMemory2                             vkBindBufferMemory2                             = 0;
    PFN_vkBindImageMemory2                              vkBindImageMemory2                              = 0;
    PFN_vkGetDeviceGroupPeerMemoryFeatures              vkGetDeviceGroupPeerMemoryFeatures              = 0;
    PFN_vkCmdSetDeviceMask                              vkCmdSetDeviceMask                              = 0;
    PFN_vkCmdDispatchBase                               vkCmdDispatchBase                               = 0;
    PFN_vkEnumeratePhysicalDeviceGroups                 vkEnumeratePhysicalDeviceGroups                 = 0;
    PFN_vkGetImageMemoryRequirements2                   vkGetImageMemoryRequirements2                   = 0;
    PFN_vkGetBufferMemoryRequirements2                  vkGetBufferMemoryRequirements2                  = 0;
    PFN_vkGetImageSparseMemoryRequirements2             vkGetImageSparseMemoryRequirements2             = 0;
    PFN_vkGetPhysicalDeviceFeatures2                    vkGetPhysicalDeviceFeatures2                    = 0;
    PFN_vkGetPhysicalDeviceProperties2                  vkGetPhysicalDeviceProperties2                  = 0;
    PFN_vkGetPhysicalDeviceFormatProperties2            vkGetPhysicalDeviceFormatProperties2            = 0;
    PFN_vkGetPhysicalDeviceImageFormatProperties2       vkGetPhysicalDeviceImageFormatProperties2       = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2       vkGetPhysicalDeviceQueueFamilyProperties2       = 0;
    PFN_vkGetPhysicalDeviceMemoryProperties2            vkGetPhysicalDeviceMemoryProperties2            = 0;
    PFN_vkGetPhysicalDeviceSparseImageFormatProperties2 vkGetPhysicalDeviceSparseImageFormatProperties2 = 0;
    PFN_vkTrimCommandPool                               vkTrimCommandPool                               = 0;
    PFN_vkGetDeviceQueue2                               vkGetDeviceQueue2                               = 0;
    PFN_vkCreateSamplerYcbcrConversion                  vkCreateSamplerYcbcrConversion                  = 0;
    PFN_vkDestroySamplerYcbcrConversion                 vkDestroySamplerYcbcrConversion                 = 0;
    PFN_vkCreateDescriptorUpdateTemplate                vkCreateDescriptorUpdateTemplate                = 0;
    PFN_vkDestroyDescriptorUpdateTemplate               vkDestroyDescriptorUpdateTemplate               = 0;
    PFN_vkUpdateDescriptorSetWithTemplate               vkUpdateDescriptorSetWithTemplate               = 0;
    PFN_vkGetPhysicalDeviceExternalBufferProperties     vkGetPhysicalDeviceExternalBufferProperties     = 0;
    PFN_vkGetPhysicalDeviceExternalFenceProperties      vkGetPhysicalDeviceExternalFenceProperties      = 0;
    PFN_vkGetPhysicalDeviceExternalSemaphoreProperties  vkGetPhysicalDeviceExternalSemaphoreProperties  = 0;
    PFN_vkGetDescriptorSetLayoutSupport                 vkGetDescriptorSetLayoutSupport                 = 0;

    //=== VK_VERSION_1_2 ===
    PFN_vkCmdDrawIndirectCount                vkCmdDrawIndirectCount                = 0;
    PFN_vkCmdDrawIndexedIndirectCount         vkCmdDrawIndexedIndirectCount         = 0;
    PFN_vkCreateRenderPass2                   vkCreateRenderPass2                   = 0;
    PFN_vkCmdBeginRenderPass2                 vkCmdBeginRenderPass2                 = 0;
    PFN_vkCmdNextSubpass2                     vkCmdNextSubpass2                     = 0;
    PFN_vkCmdEndRenderPass2                   vkCmdEndRenderPass2                   = 0;
    PFN_vkResetQueryPool                      vkResetQueryPool                      = 0;
    PFN_vkGetSemaphoreCounterValue            vkGetSemaphoreCounterValue            = 0;
    PFN_vkWaitSemaphores                      vkWaitSemaphores                      = 0;
    PFN_vkSignalSemaphore                     vkSignalSemaphore                     = 0;
    PFN_vkGetBufferDeviceAddress              vkGetBufferDeviceAddress              = 0;
    PFN_vkGetBufferOpaqueCaptureAddress       vkGetBufferOpaqueCaptureAddress       = 0;
    PFN_vkGetDeviceMemoryOpaqueCaptureAddress vkGetDeviceMemoryOpaqueCaptureAddress = 0;

    //=== VK_VERSION_1_3 ===
    PFN_vkGetPhysicalDeviceToolProperties        vkGetPhysicalDeviceToolProperties        = 0;
    PFN_vkCreatePrivateDataSlot                  vkCreatePrivateDataSlot                  = 0;
    PFN_vkDestroyPrivateDataSlot                 vkDestroyPrivateDataSlot                 = 0;
    PFN_vkSetPrivateData                         vkSetPrivateData                         = 0;
    PFN_vkGetPrivateData                         vkGetPrivateData                         = 0;
    PFN_vkCmdSetEvent2                           vkCmdSetEvent2                           = 0;
    PFN_vkCmdResetEvent2                         vkCmdResetEvent2                         = 0;
    PFN_vkCmdWaitEvents2                         vkCmdWaitEvents2                         = 0;
    PFN_vkCmdPipelineBarrier2                    vkCmdPipelineBarrier2                    = 0;
    PFN_vkCmdWriteTimestamp2                     vkCmdWriteTimestamp2                     = 0;
    PFN_vkQueueSubmit2                           vkQueueSubmit2                           = 0;
    PFN_vkCmdCopyBuffer2                         vkCmdCopyBuffer2                         = 0;
    PFN_vkCmdCopyImage2                          vkCmdCopyImage2                          = 0;
    PFN_vkCmdCopyBufferToImage2                  vkCmdCopyBufferToImage2                  = 0;
    PFN_vkCmdCopyImageToBuffer2                  vkCmdCopyImageToBuffer2                  = 0;
    PFN_vkCmdBlitImage2                          vkCmdBlitImage2                          = 0;
    PFN_vkCmdResolveImage2                       vkCmdResolveImage2                       = 0;
    PFN_vkCmdBeginRendering                      vkCmdBeginRendering                      = 0;
    PFN_vkCmdEndRendering                        vkCmdEndRendering                        = 0;
    PFN_vkCmdSetCullMode                         vkCmdSetCullMode                         = 0;
    PFN_vkCmdSetFrontFace                        vkCmdSetFrontFace                        = 0;
    PFN_vkCmdSetPrimitiveTopology                vkCmdSetPrimitiveTopology                = 0;
    PFN_vkCmdSetViewportWithCount                vkCmdSetViewportWithCount                = 0;
    PFN_vkCmdSetScissorWithCount                 vkCmdSetScissorWithCount                 = 0;
    PFN_vkCmdBindVertexBuffers2                  vkCmdBindVertexBuffers2                  = 0;
    PFN_vkCmdSetDepthTestEnable                  vkCmdSetDepthTestEnable                  = 0;
    PFN_vkCmdSetDepthWriteEnable                 vkCmdSetDepthWriteEnable                 = 0;
    PFN_vkCmdSetDepthCompareOp                   vkCmdSetDepthCompareOp                   = 0;
    PFN_vkCmdSetDepthBoundsTestEnable            vkCmdSetDepthBoundsTestEnable            = 0;
    PFN_vkCmdSetStencilTestEnable                vkCmdSetStencilTestEnable                = 0;
    PFN_vkCmdSetStencilOp                        vkCmdSetStencilOp                        = 0;
    PFN_vkCmdSetRasterizerDiscardEnable          vkCmdSetRasterizerDiscardEnable          = 0;
    PFN_vkCmdSetDepthBiasEnable                  vkCmdSetDepthBiasEnable                  = 0;
    PFN_vkCmdSetPrimitiveRestartEnable           vkCmdSetPrimitiveRestartEnable           = 0;
    PFN_vkGetDeviceBufferMemoryRequirements      vkGetDeviceBufferMemoryRequirements      = 0;
    PFN_vkGetDeviceImageMemoryRequirements       vkGetDeviceImageMemoryRequirements       = 0;
    PFN_vkGetDeviceImageSparseMemoryRequirements vkGetDeviceImageSparseMemoryRequirements = 0;

    //=== VK_KHR_surface ===
    PFN_vkDestroySurfaceKHR                       vkDestroySurfaceKHR                       = 0;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR      vkGetPhysicalDeviceSurfaceSupportKHR      = 0;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR = 0;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR      vkGetPhysicalDeviceSurfaceFormatsKHR      = 0;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR = 0;

    //=== VK_KHR_swapchain ===
    PFN_vkCreateSwapchainKHR                    vkCreateSwapchainKHR                    = 0;
    PFN_vkDestroySwapchainKHR                   vkDestroySwapchainKHR                   = 0;
    PFN_vkGetSwapchainImagesKHR                 vkGetSwapchainImagesKHR                 = 0;
    PFN_vkAcquireNextImageKHR                   vkAcquireNextImageKHR                   = 0;
    PFN_vkQueuePresentKHR                       vkQueuePresentKHR                       = 0;
    PFN_vkGetDeviceGroupPresentCapabilitiesKHR  vkGetDeviceGroupPresentCapabilitiesKHR  = 0;
    PFN_vkGetDeviceGroupSurfacePresentModesKHR  vkGetDeviceGroupSurfacePresentModesKHR  = 0;
    PFN_vkGetPhysicalDevicePresentRectanglesKHR vkGetPhysicalDevicePresentRectanglesKHR = 0;
    PFN_vkAcquireNextImage2KHR                  vkAcquireNextImage2KHR                  = 0;

    //=== VK_KHR_display ===
    PFN_vkGetPhysicalDeviceDisplayPropertiesKHR      vkGetPhysicalDeviceDisplayPropertiesKHR      = 0;
    PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR vkGetPhysicalDeviceDisplayPlanePropertiesKHR = 0;
    PFN_vkGetDisplayPlaneSupportedDisplaysKHR        vkGetDisplayPlaneSupportedDisplaysKHR        = 0;
    PFN_vkGetDisplayModePropertiesKHR                vkGetDisplayModePropertiesKHR                = 0;
    PFN_vkCreateDisplayModeKHR                       vkCreateDisplayModeKHR                       = 0;
    PFN_vkGetDisplayPlaneCapabilitiesKHR             vkGetDisplayPlaneCapabilitiesKHR             = 0;
    PFN_vkCreateDisplayPlaneSurfaceKHR               vkCreateDisplayPlaneSurfaceKHR               = 0;

    //=== VK_KHR_display_swapchain ===
    PFN_vkCreateSharedSwapchainsKHR vkCreateSharedSwapchainsKHR = 0;

#if defined( VK_USE_PLATFORM_XLIB_KHR )
    //=== VK_KHR_xlib_surface ===
    PFN_vkCreateXlibSurfaceKHR                        vkCreateXlibSurfaceKHR                        = 0;
    PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR vkGetPhysicalDeviceXlibPresentationSupportKHR = 0;
#else
    PFN_dummy vkCreateXlibSurfaceKHR_placeholder                            = 0;
    PFN_dummy vkGetPhysicalDeviceXlibPresentationSupportKHR_placeholder     = 0;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
    //=== VK_KHR_xcb_surface ===
    PFN_vkCreateXcbSurfaceKHR                        vkCreateXcbSurfaceKHR                        = 0;
    PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR vkGetPhysicalDeviceXcbPresentationSupportKHR = 0;
#else
    PFN_dummy vkCreateXcbSurfaceKHR_placeholder                             = 0;
    PFN_dummy vkGetPhysicalDeviceXcbPresentationSupportKHR_placeholder      = 0;
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
    //=== VK_KHR_wayland_surface ===
    PFN_vkCreateWaylandSurfaceKHR                        vkCreateWaylandSurfaceKHR                        = 0;
    PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR vkGetPhysicalDeviceWaylandPresentationSupportKHR = 0;
#else
    PFN_dummy vkCreateWaylandSurfaceKHR_placeholder                         = 0;
    PFN_dummy vkGetPhysicalDeviceWaylandPresentationSupportKHR_placeholder  = 0;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    //=== VK_KHR_android_surface ===
    PFN_vkCreateAndroidSurfaceKHR vkCreateAndroidSurfaceKHR = 0;
#else
    PFN_dummy vkCreateAndroidSurfaceKHR_placeholder                         = 0;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_win32_surface ===
    PFN_vkCreateWin32SurfaceKHR                        vkCreateWin32SurfaceKHR                        = 0;
    PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR vkGetPhysicalDeviceWin32PresentationSupportKHR = 0;
#else
    PFN_dummy vkCreateWin32SurfaceKHR_placeholder                           = 0;
    PFN_dummy vkGetPhysicalDeviceWin32PresentationSupportKHR_placeholder    = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_EXT_debug_report ===
    PFN_vkCreateDebugReportCallbackEXT  vkCreateDebugReportCallbackEXT  = 0;
    PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT = 0;
    PFN_vkDebugReportMessageEXT         vkDebugReportMessageEXT         = 0;

    //=== VK_EXT_debug_marker ===
    PFN_vkDebugMarkerSetObjectTagEXT  vkDebugMarkerSetObjectTagEXT  = 0;
    PFN_vkDebugMarkerSetObjectNameEXT vkDebugMarkerSetObjectNameEXT = 0;
    PFN_vkCmdDebugMarkerBeginEXT      vkCmdDebugMarkerBeginEXT      = 0;
    PFN_vkCmdDebugMarkerEndEXT        vkCmdDebugMarkerEndEXT        = 0;
    PFN_vkCmdDebugMarkerInsertEXT     vkCmdDebugMarkerInsertEXT     = 0;

    //=== VK_KHR_video_queue ===
    PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR     vkGetPhysicalDeviceVideoCapabilitiesKHR     = 0;
    PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR vkGetPhysicalDeviceVideoFormatPropertiesKHR = 0;
    PFN_vkCreateVideoSessionKHR                     vkCreateVideoSessionKHR                     = 0;
    PFN_vkDestroyVideoSessionKHR                    vkDestroyVideoSessionKHR                    = 0;
    PFN_vkGetVideoSessionMemoryRequirementsKHR      vkGetVideoSessionMemoryRequirementsKHR      = 0;
    PFN_vkBindVideoSessionMemoryKHR                 vkBindVideoSessionMemoryKHR                 = 0;
    PFN_vkCreateVideoSessionParametersKHR           vkCreateVideoSessionParametersKHR           = 0;
    PFN_vkUpdateVideoSessionParametersKHR           vkUpdateVideoSessionParametersKHR           = 0;
    PFN_vkDestroyVideoSessionParametersKHR          vkDestroyVideoSessionParametersKHR          = 0;
    PFN_vkCmdBeginVideoCodingKHR                    vkCmdBeginVideoCodingKHR                    = 0;
    PFN_vkCmdEndVideoCodingKHR                      vkCmdEndVideoCodingKHR                      = 0;
    PFN_vkCmdControlVideoCodingKHR                  vkCmdControlVideoCodingKHR                  = 0;

    //=== VK_KHR_video_decode_queue ===
    PFN_vkCmdDecodeVideoKHR vkCmdDecodeVideoKHR = 0;

    //=== VK_EXT_transform_feedback ===
    PFN_vkCmdBindTransformFeedbackBuffersEXT vkCmdBindTransformFeedbackBuffersEXT = 0;
    PFN_vkCmdBeginTransformFeedbackEXT       vkCmdBeginTransformFeedbackEXT       = 0;
    PFN_vkCmdEndTransformFeedbackEXT         vkCmdEndTransformFeedbackEXT         = 0;
    PFN_vkCmdBeginQueryIndexedEXT            vkCmdBeginQueryIndexedEXT            = 0;
    PFN_vkCmdEndQueryIndexedEXT              vkCmdEndQueryIndexedEXT              = 0;
    PFN_vkCmdDrawIndirectByteCountEXT        vkCmdDrawIndirectByteCountEXT        = 0;

    //=== VK_NVX_binary_import ===
    PFN_vkCreateCuModuleNVX    vkCreateCuModuleNVX    = 0;
    PFN_vkCreateCuFunctionNVX  vkCreateCuFunctionNVX  = 0;
    PFN_vkDestroyCuModuleNVX   vkDestroyCuModuleNVX   = 0;
    PFN_vkDestroyCuFunctionNVX vkDestroyCuFunctionNVX = 0;
    PFN_vkCmdCuLaunchKernelNVX vkCmdCuLaunchKernelNVX = 0;

    //=== VK_NVX_image_view_handle ===
    PFN_vkGetImageViewHandleNVX  vkGetImageViewHandleNVX  = 0;
    PFN_vkGetImageViewAddressNVX vkGetImageViewAddressNVX = 0;

    //=== VK_AMD_draw_indirect_count ===
    PFN_vkCmdDrawIndirectCountAMD        vkCmdDrawIndirectCountAMD        = 0;
    PFN_vkCmdDrawIndexedIndirectCountAMD vkCmdDrawIndexedIndirectCountAMD = 0;

    //=== VK_AMD_shader_info ===
    PFN_vkGetShaderInfoAMD vkGetShaderInfoAMD = 0;

    //=== VK_KHR_dynamic_rendering ===
    PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR = 0;
    PFN_vkCmdEndRenderingKHR   vkCmdEndRenderingKHR   = 0;

#if defined( VK_USE_PLATFORM_GGP )
    //=== VK_GGP_stream_descriptor_surface ===
    PFN_vkCreateStreamDescriptorSurfaceGGP vkCreateStreamDescriptorSurfaceGGP = 0;
#else
    PFN_dummy vkCreateStreamDescriptorSurfaceGGP_placeholder                = 0;
#endif /*VK_USE_PLATFORM_GGP*/

    //=== VK_NV_external_memory_capabilities ===
    PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV vkGetPhysicalDeviceExternalImageFormatPropertiesNV = 0;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_NV_external_memory_win32 ===
    PFN_vkGetMemoryWin32HandleNV vkGetMemoryWin32HandleNV = 0;
#else
    PFN_dummy vkGetMemoryWin32HandleNV_placeholder                          = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_get_physical_device_properties2 ===
    PFN_vkGetPhysicalDeviceFeatures2KHR                    vkGetPhysicalDeviceFeatures2KHR                    = 0;
    PFN_vkGetPhysicalDeviceProperties2KHR                  vkGetPhysicalDeviceProperties2KHR                  = 0;
    PFN_vkGetPhysicalDeviceFormatProperties2KHR            vkGetPhysicalDeviceFormatProperties2KHR            = 0;
    PFN_vkGetPhysicalDeviceImageFormatProperties2KHR       vkGetPhysicalDeviceImageFormatProperties2KHR       = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR       vkGetPhysicalDeviceQueueFamilyProperties2KHR       = 0;
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR            vkGetPhysicalDeviceMemoryProperties2KHR            = 0;
    PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR = 0;

    //=== VK_KHR_device_group ===
    PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR vkGetDeviceGroupPeerMemoryFeaturesKHR = 0;
    PFN_vkCmdSetDeviceMaskKHR                 vkCmdSetDeviceMaskKHR                 = 0;
    PFN_vkCmdDispatchBaseKHR                  vkCmdDispatchBaseKHR                  = 0;

#if defined( VK_USE_PLATFORM_VI_NN )
    //=== VK_NN_vi_surface ===
    PFN_vkCreateViSurfaceNN vkCreateViSurfaceNN = 0;
#else
    PFN_dummy vkCreateViSurfaceNN_placeholder                               = 0;
#endif /*VK_USE_PLATFORM_VI_NN*/

    //=== VK_KHR_maintenance1 ===
    PFN_vkTrimCommandPoolKHR vkTrimCommandPoolKHR = 0;

    //=== VK_KHR_device_group_creation ===
    PFN_vkEnumeratePhysicalDeviceGroupsKHR vkEnumeratePhysicalDeviceGroupsKHR = 0;

    //=== VK_KHR_external_memory_capabilities ===
    PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR vkGetPhysicalDeviceExternalBufferPropertiesKHR = 0;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_memory_win32 ===
    PFN_vkGetMemoryWin32HandleKHR           vkGetMemoryWin32HandleKHR           = 0;
    PFN_vkGetMemoryWin32HandlePropertiesKHR vkGetMemoryWin32HandlePropertiesKHR = 0;
#else
    PFN_dummy vkGetMemoryWin32HandleKHR_placeholder                         = 0;
    PFN_dummy vkGetMemoryWin32HandlePropertiesKHR_placeholder               = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_memory_fd ===
    PFN_vkGetMemoryFdKHR           vkGetMemoryFdKHR           = 0;
    PFN_vkGetMemoryFdPropertiesKHR vkGetMemoryFdPropertiesKHR = 0;

    //=== VK_KHR_external_semaphore_capabilities ===
    PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR vkGetPhysicalDeviceExternalSemaphorePropertiesKHR = 0;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_semaphore_win32 ===
    PFN_vkImportSemaphoreWin32HandleKHR vkImportSemaphoreWin32HandleKHR = 0;
    PFN_vkGetSemaphoreWin32HandleKHR    vkGetSemaphoreWin32HandleKHR    = 0;
#else
    PFN_dummy vkImportSemaphoreWin32HandleKHR_placeholder                   = 0;
    PFN_dummy vkGetSemaphoreWin32HandleKHR_placeholder                      = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_semaphore_fd ===
    PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR = 0;
    PFN_vkGetSemaphoreFdKHR    vkGetSemaphoreFdKHR    = 0;

    //=== VK_KHR_push_descriptor ===
    PFN_vkCmdPushDescriptorSetKHR             vkCmdPushDescriptorSetKHR             = 0;
    PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR = 0;

    //=== VK_EXT_conditional_rendering ===
    PFN_vkCmdBeginConditionalRenderingEXT vkCmdBeginConditionalRenderingEXT = 0;
    PFN_vkCmdEndConditionalRenderingEXT   vkCmdEndConditionalRenderingEXT   = 0;

    //=== VK_KHR_descriptor_update_template ===
    PFN_vkCreateDescriptorUpdateTemplateKHR  vkCreateDescriptorUpdateTemplateKHR  = 0;
    PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR = 0;
    PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR = 0;

    //=== VK_NV_clip_space_w_scaling ===
    PFN_vkCmdSetViewportWScalingNV vkCmdSetViewportWScalingNV = 0;

    //=== VK_EXT_direct_mode_display ===
    PFN_vkReleaseDisplayEXT vkReleaseDisplayEXT = 0;

#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
    //=== VK_EXT_acquire_xlib_display ===
    PFN_vkAcquireXlibDisplayEXT    vkAcquireXlibDisplayEXT    = 0;
    PFN_vkGetRandROutputDisplayEXT vkGetRandROutputDisplayEXT = 0;
#else
    PFN_dummy vkAcquireXlibDisplayEXT_placeholder                           = 0;
    PFN_dummy vkGetRandROutputDisplayEXT_placeholder                        = 0;
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

    //=== VK_EXT_display_surface_counter ===
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT vkGetPhysicalDeviceSurfaceCapabilities2EXT = 0;

    //=== VK_EXT_display_control ===
    PFN_vkDisplayPowerControlEXT  vkDisplayPowerControlEXT  = 0;
    PFN_vkRegisterDeviceEventEXT  vkRegisterDeviceEventEXT  = 0;
    PFN_vkRegisterDisplayEventEXT vkRegisterDisplayEventEXT = 0;
    PFN_vkGetSwapchainCounterEXT  vkGetSwapchainCounterEXT  = 0;

    //=== VK_GOOGLE_display_timing ===
    PFN_vkGetRefreshCycleDurationGOOGLE   vkGetRefreshCycleDurationGOOGLE   = 0;
    PFN_vkGetPastPresentationTimingGOOGLE vkGetPastPresentationTimingGOOGLE = 0;

    //=== VK_EXT_discard_rectangles ===
    PFN_vkCmdSetDiscardRectangleEXT       vkCmdSetDiscardRectangleEXT       = 0;
    PFN_vkCmdSetDiscardRectangleEnableEXT vkCmdSetDiscardRectangleEnableEXT = 0;
    PFN_vkCmdSetDiscardRectangleModeEXT   vkCmdSetDiscardRectangleModeEXT   = 0;

    //=== VK_EXT_hdr_metadata ===
    PFN_vkSetHdrMetadataEXT vkSetHdrMetadataEXT = 0;

    //=== VK_KHR_create_renderpass2 ===
    PFN_vkCreateRenderPass2KHR   vkCreateRenderPass2KHR   = 0;
    PFN_vkCmdBeginRenderPass2KHR vkCmdBeginRenderPass2KHR = 0;
    PFN_vkCmdNextSubpass2KHR     vkCmdNextSubpass2KHR     = 0;
    PFN_vkCmdEndRenderPass2KHR   vkCmdEndRenderPass2KHR   = 0;

    //=== VK_KHR_shared_presentable_image ===
    PFN_vkGetSwapchainStatusKHR vkGetSwapchainStatusKHR = 0;

    //=== VK_KHR_external_fence_capabilities ===
    PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR vkGetPhysicalDeviceExternalFencePropertiesKHR = 0;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_fence_win32 ===
    PFN_vkImportFenceWin32HandleKHR vkImportFenceWin32HandleKHR = 0;
    PFN_vkGetFenceWin32HandleKHR    vkGetFenceWin32HandleKHR    = 0;
#else
    PFN_dummy vkImportFenceWin32HandleKHR_placeholder                       = 0;
    PFN_dummy vkGetFenceWin32HandleKHR_placeholder                          = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_fence_fd ===
    PFN_vkImportFenceFdKHR vkImportFenceFdKHR = 0;
    PFN_vkGetFenceFdKHR    vkGetFenceFdKHR    = 0;

    //=== VK_KHR_performance_query ===
    PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR         vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR         = 0;
    PFN_vkAcquireProfilingLockKHR                                       vkAcquireProfilingLockKHR                                       = 0;
    PFN_vkReleaseProfilingLockKHR                                       vkReleaseProfilingLockKHR                                       = 0;

    //=== VK_KHR_get_surface_capabilities2 ===
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR = 0;
    PFN_vkGetPhysicalDeviceSurfaceFormats2KHR      vkGetPhysicalDeviceSurfaceFormats2KHR      = 0;

    //=== VK_KHR_get_display_properties2 ===
    PFN_vkGetPhysicalDeviceDisplayProperties2KHR      vkGetPhysicalDeviceDisplayProperties2KHR      = 0;
    PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR vkGetPhysicalDeviceDisplayPlaneProperties2KHR = 0;
    PFN_vkGetDisplayModeProperties2KHR                vkGetDisplayModeProperties2KHR                = 0;
    PFN_vkGetDisplayPlaneCapabilities2KHR             vkGetDisplayPlaneCapabilities2KHR             = 0;

#if defined( VK_USE_PLATFORM_IOS_MVK )
    //=== VK_MVK_ios_surface ===
    PFN_vkCreateIOSSurfaceMVK vkCreateIOSSurfaceMVK = 0;
#else
    PFN_dummy vkCreateIOSSurfaceMVK_placeholder                             = 0;
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
    //=== VK_MVK_macos_surface ===
    PFN_vkCreateMacOSSurfaceMVK vkCreateMacOSSurfaceMVK = 0;
#else
    PFN_dummy vkCreateMacOSSurfaceMVK_placeholder                           = 0;
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

    //=== VK_EXT_debug_utils ===
    PFN_vkSetDebugUtilsObjectNameEXT    vkSetDebugUtilsObjectNameEXT    = 0;
    PFN_vkSetDebugUtilsObjectTagEXT     vkSetDebugUtilsObjectTagEXT     = 0;
    PFN_vkQueueBeginDebugUtilsLabelEXT  vkQueueBeginDebugUtilsLabelEXT  = 0;
    PFN_vkQueueEndDebugUtilsLabelEXT    vkQueueEndDebugUtilsLabelEXT    = 0;
    PFN_vkQueueInsertDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT = 0;
    PFN_vkCmdBeginDebugUtilsLabelEXT    vkCmdBeginDebugUtilsLabelEXT    = 0;
    PFN_vkCmdEndDebugUtilsLabelEXT      vkCmdEndDebugUtilsLabelEXT      = 0;
    PFN_vkCmdInsertDebugUtilsLabelEXT   vkCmdInsertDebugUtilsLabelEXT   = 0;
    PFN_vkCreateDebugUtilsMessengerEXT  vkCreateDebugUtilsMessengerEXT  = 0;
    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = 0;
    PFN_vkSubmitDebugUtilsMessageEXT    vkSubmitDebugUtilsMessageEXT    = 0;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    //=== VK_ANDROID_external_memory_android_hardware_buffer ===
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID = 0;
    PFN_vkGetMemoryAndroidHardwareBufferANDROID     vkGetMemoryAndroidHardwareBufferANDROID     = 0;
#else
    PFN_dummy vkGetAndroidHardwareBufferPropertiesANDROID_placeholder       = 0;
    PFN_dummy vkGetMemoryAndroidHardwareBufferANDROID_placeholder           = 0;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_AMDX_shader_enqueue ===
    PFN_vkCreateExecutionGraphPipelinesAMDX        vkCreateExecutionGraphPipelinesAMDX        = 0;
    PFN_vkGetExecutionGraphPipelineScratchSizeAMDX vkGetExecutionGraphPipelineScratchSizeAMDX = 0;
    PFN_vkGetExecutionGraphPipelineNodeIndexAMDX   vkGetExecutionGraphPipelineNodeIndexAMDX   = 0;
    PFN_vkCmdInitializeGraphScratchMemoryAMDX      vkCmdInitializeGraphScratchMemoryAMDX      = 0;
    PFN_vkCmdDispatchGraphAMDX                     vkCmdDispatchGraphAMDX                     = 0;
    PFN_vkCmdDispatchGraphIndirectAMDX             vkCmdDispatchGraphIndirectAMDX             = 0;
    PFN_vkCmdDispatchGraphIndirectCountAMDX        vkCmdDispatchGraphIndirectCountAMDX        = 0;
#else
    PFN_dummy vkCreateExecutionGraphPipelinesAMDX_placeholder               = 0;
    PFN_dummy vkGetExecutionGraphPipelineScratchSizeAMDX_placeholder        = 0;
    PFN_dummy vkGetExecutionGraphPipelineNodeIndexAMDX_placeholder          = 0;
    PFN_dummy vkCmdInitializeGraphScratchMemoryAMDX_placeholder             = 0;
    PFN_dummy vkCmdDispatchGraphAMDX_placeholder                            = 0;
    PFN_dummy vkCmdDispatchGraphIndirectAMDX_placeholder                    = 0;
    PFN_dummy vkCmdDispatchGraphIndirectCountAMDX_placeholder               = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

    //=== VK_EXT_sample_locations ===
    PFN_vkCmdSetSampleLocationsEXT                  vkCmdSetSampleLocationsEXT                  = 0;
    PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT vkGetPhysicalDeviceMultisamplePropertiesEXT = 0;

    //=== VK_KHR_get_memory_requirements2 ===
    PFN_vkGetImageMemoryRequirements2KHR       vkGetImageMemoryRequirements2KHR       = 0;
    PFN_vkGetBufferMemoryRequirements2KHR      vkGetBufferMemoryRequirements2KHR      = 0;
    PFN_vkGetImageSparseMemoryRequirements2KHR vkGetImageSparseMemoryRequirements2KHR = 0;

    //=== VK_KHR_acceleration_structure ===
    PFN_vkCreateAccelerationStructureKHR                 vkCreateAccelerationStructureKHR                 = 0;
    PFN_vkDestroyAccelerationStructureKHR                vkDestroyAccelerationStructureKHR                = 0;
    PFN_vkCmdBuildAccelerationStructuresKHR              vkCmdBuildAccelerationStructuresKHR              = 0;
    PFN_vkCmdBuildAccelerationStructuresIndirectKHR      vkCmdBuildAccelerationStructuresIndirectKHR      = 0;
    PFN_vkBuildAccelerationStructuresKHR                 vkBuildAccelerationStructuresKHR                 = 0;
    PFN_vkCopyAccelerationStructureKHR                   vkCopyAccelerationStructureKHR                   = 0;
    PFN_vkCopyAccelerationStructureToMemoryKHR           vkCopyAccelerationStructureToMemoryKHR           = 0;
    PFN_vkCopyMemoryToAccelerationStructureKHR           vkCopyMemoryToAccelerationStructureKHR           = 0;
    PFN_vkWriteAccelerationStructuresPropertiesKHR       vkWriteAccelerationStructuresPropertiesKHR       = 0;
    PFN_vkCmdCopyAccelerationStructureKHR                vkCmdCopyAccelerationStructureKHR                = 0;
    PFN_vkCmdCopyAccelerationStructureToMemoryKHR        vkCmdCopyAccelerationStructureToMemoryKHR        = 0;
    PFN_vkCmdCopyMemoryToAccelerationStructureKHR        vkCmdCopyMemoryToAccelerationStructureKHR        = 0;
    PFN_vkGetAccelerationStructureDeviceAddressKHR       vkGetAccelerationStructureDeviceAddressKHR       = 0;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR    vkCmdWriteAccelerationStructuresPropertiesKHR    = 0;
    PFN_vkGetDeviceAccelerationStructureCompatibilityKHR vkGetDeviceAccelerationStructureCompatibilityKHR = 0;
    PFN_vkGetAccelerationStructureBuildSizesKHR          vkGetAccelerationStructureBuildSizesKHR          = 0;

    //=== VK_KHR_ray_tracing_pipeline ===
    PFN_vkCmdTraceRaysKHR                                 vkCmdTraceRaysKHR                                 = 0;
    PFN_vkCreateRayTracingPipelinesKHR                    vkCreateRayTracingPipelinesKHR                    = 0;
    PFN_vkGetRayTracingShaderGroupHandlesKHR              vkGetRayTracingShaderGroupHandlesKHR              = 0;
    PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = 0;
    PFN_vkCmdTraceRaysIndirectKHR                         vkCmdTraceRaysIndirectKHR                         = 0;
    PFN_vkGetRayTracingShaderGroupStackSizeKHR            vkGetRayTracingShaderGroupStackSizeKHR            = 0;
    PFN_vkCmdSetRayTracingPipelineStackSizeKHR            vkCmdSetRayTracingPipelineStackSizeKHR            = 0;

    //=== VK_KHR_sampler_ycbcr_conversion ===
    PFN_vkCreateSamplerYcbcrConversionKHR  vkCreateSamplerYcbcrConversionKHR  = 0;
    PFN_vkDestroySamplerYcbcrConversionKHR vkDestroySamplerYcbcrConversionKHR = 0;

    //=== VK_KHR_bind_memory2 ===
    PFN_vkBindBufferMemory2KHR vkBindBufferMemory2KHR = 0;
    PFN_vkBindImageMemory2KHR  vkBindImageMemory2KHR  = 0;

    //=== VK_EXT_image_drm_format_modifier ===
    PFN_vkGetImageDrmFormatModifierPropertiesEXT vkGetImageDrmFormatModifierPropertiesEXT = 0;

    //=== VK_EXT_validation_cache ===
    PFN_vkCreateValidationCacheEXT  vkCreateValidationCacheEXT  = 0;
    PFN_vkDestroyValidationCacheEXT vkDestroyValidationCacheEXT = 0;
    PFN_vkMergeValidationCachesEXT  vkMergeValidationCachesEXT  = 0;
    PFN_vkGetValidationCacheDataEXT vkGetValidationCacheDataEXT = 0;

    //=== VK_NV_shading_rate_image ===
    PFN_vkCmdBindShadingRateImageNV          vkCmdBindShadingRateImageNV          = 0;
    PFN_vkCmdSetViewportShadingRatePaletteNV vkCmdSetViewportShadingRatePaletteNV = 0;
    PFN_vkCmdSetCoarseSampleOrderNV          vkCmdSetCoarseSampleOrderNV          = 0;

    //=== VK_NV_ray_tracing ===
    PFN_vkCreateAccelerationStructureNV                vkCreateAccelerationStructureNV                = 0;
    PFN_vkDestroyAccelerationStructureNV               vkDestroyAccelerationStructureNV               = 0;
    PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV = 0;
    PFN_vkBindAccelerationStructureMemoryNV            vkBindAccelerationStructureMemoryNV            = 0;
    PFN_vkCmdBuildAccelerationStructureNV              vkCmdBuildAccelerationStructureNV              = 0;
    PFN_vkCmdCopyAccelerationStructureNV               vkCmdCopyAccelerationStructureNV               = 0;
    PFN_vkCmdTraceRaysNV                               vkCmdTraceRaysNV                               = 0;
    PFN_vkCreateRayTracingPipelinesNV                  vkCreateRayTracingPipelinesNV                  = 0;
    PFN_vkGetRayTracingShaderGroupHandlesNV            vkGetRayTracingShaderGroupHandlesNV            = 0;
    PFN_vkGetAccelerationStructureHandleNV             vkGetAccelerationStructureHandleNV             = 0;
    PFN_vkCmdWriteAccelerationStructuresPropertiesNV   vkCmdWriteAccelerationStructuresPropertiesNV   = 0;
    PFN_vkCompileDeferredNV                            vkCompileDeferredNV                            = 0;

    //=== VK_KHR_maintenance3 ===
    PFN_vkGetDescriptorSetLayoutSupportKHR vkGetDescriptorSetLayoutSupportKHR = 0;

    //=== VK_KHR_draw_indirect_count ===
    PFN_vkCmdDrawIndirectCountKHR        vkCmdDrawIndirectCountKHR        = 0;
    PFN_vkCmdDrawIndexedIndirectCountKHR vkCmdDrawIndexedIndirectCountKHR = 0;

    //=== VK_EXT_external_memory_host ===
    PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerPropertiesEXT = 0;

    //=== VK_AMD_buffer_marker ===
    PFN_vkCmdWriteBufferMarkerAMD vkCmdWriteBufferMarkerAMD = 0;

    //=== VK_EXT_calibrated_timestamps ===
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = 0;
    PFN_vkGetCalibratedTimestampsEXT                   vkGetCalibratedTimestampsEXT                   = 0;

    //=== VK_NV_mesh_shader ===
    PFN_vkCmdDrawMeshTasksNV              vkCmdDrawMeshTasksNV              = 0;
    PFN_vkCmdDrawMeshTasksIndirectNV      vkCmdDrawMeshTasksIndirectNV      = 0;
    PFN_vkCmdDrawMeshTasksIndirectCountNV vkCmdDrawMeshTasksIndirectCountNV = 0;

    //=== VK_NV_scissor_exclusive ===
    PFN_vkCmdSetExclusiveScissorEnableNV vkCmdSetExclusiveScissorEnableNV = 0;
    PFN_vkCmdSetExclusiveScissorNV       vkCmdSetExclusiveScissorNV       = 0;

    //=== VK_NV_device_diagnostic_checkpoints ===
    PFN_vkCmdSetCheckpointNV       vkCmdSetCheckpointNV       = 0;
    PFN_vkGetQueueCheckpointDataNV vkGetQueueCheckpointDataNV = 0;

    //=== VK_KHR_timeline_semaphore ===
    PFN_vkGetSemaphoreCounterValueKHR vkGetSemaphoreCounterValueKHR = 0;
    PFN_vkWaitSemaphoresKHR           vkWaitSemaphoresKHR           = 0;
    PFN_vkSignalSemaphoreKHR          vkSignalSemaphoreKHR          = 0;

    //=== VK_INTEL_performance_query ===
    PFN_vkInitializePerformanceApiINTEL         vkInitializePerformanceApiINTEL         = 0;
    PFN_vkUninitializePerformanceApiINTEL       vkUninitializePerformanceApiINTEL       = 0;
    PFN_vkCmdSetPerformanceMarkerINTEL          vkCmdSetPerformanceMarkerINTEL          = 0;
    PFN_vkCmdSetPerformanceStreamMarkerINTEL    vkCmdSetPerformanceStreamMarkerINTEL    = 0;
    PFN_vkCmdSetPerformanceOverrideINTEL        vkCmdSetPerformanceOverrideINTEL        = 0;
    PFN_vkAcquirePerformanceConfigurationINTEL  vkAcquirePerformanceConfigurationINTEL  = 0;
    PFN_vkReleasePerformanceConfigurationINTEL  vkReleasePerformanceConfigurationINTEL  = 0;
    PFN_vkQueueSetPerformanceConfigurationINTEL vkQueueSetPerformanceConfigurationINTEL = 0;
    PFN_vkGetPerformanceParameterINTEL          vkGetPerformanceParameterINTEL          = 0;

    //=== VK_AMD_display_native_hdr ===
    PFN_vkSetLocalDimmingAMD vkSetLocalDimmingAMD = 0;

#if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_imagepipe_surface ===
    PFN_vkCreateImagePipeSurfaceFUCHSIA vkCreateImagePipeSurfaceFUCHSIA = 0;
#else
    PFN_dummy vkCreateImagePipeSurfaceFUCHSIA_placeholder                   = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
    //=== VK_EXT_metal_surface ===
    PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT = 0;
#else
    PFN_dummy vkCreateMetalSurfaceEXT_placeholder                           = 0;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

    //=== VK_KHR_fragment_shading_rate ===
    PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR vkGetPhysicalDeviceFragmentShadingRatesKHR = 0;
    PFN_vkCmdSetFragmentShadingRateKHR             vkCmdSetFragmentShadingRateKHR             = 0;

    //=== VK_EXT_buffer_device_address ===
    PFN_vkGetBufferDeviceAddressEXT vkGetBufferDeviceAddressEXT = 0;

    //=== VK_EXT_tooling_info ===
    PFN_vkGetPhysicalDeviceToolPropertiesEXT vkGetPhysicalDeviceToolPropertiesEXT = 0;

    //=== VK_KHR_present_wait ===
    PFN_vkWaitForPresentKHR vkWaitForPresentKHR = 0;

    //=== VK_NV_cooperative_matrix ===
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = 0;

    //=== VK_NV_coverage_reduction_mode ===
    PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV = 0;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_EXT_full_screen_exclusive ===
    PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT vkGetPhysicalDeviceSurfacePresentModes2EXT = 0;
    PFN_vkAcquireFullScreenExclusiveModeEXT        vkAcquireFullScreenExclusiveModeEXT        = 0;
    PFN_vkReleaseFullScreenExclusiveModeEXT        vkReleaseFullScreenExclusiveModeEXT        = 0;
    PFN_vkGetDeviceGroupSurfacePresentModes2EXT    vkGetDeviceGroupSurfacePresentModes2EXT    = 0;
#else
    PFN_dummy vkGetPhysicalDeviceSurfacePresentModes2EXT_placeholder        = 0;
    PFN_dummy vkAcquireFullScreenExclusiveModeEXT_placeholder               = 0;
    PFN_dummy vkReleaseFullScreenExclusiveModeEXT_placeholder               = 0;
    PFN_dummy vkGetDeviceGroupSurfacePresentModes2EXT_placeholder           = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_EXT_headless_surface ===
    PFN_vkCreateHeadlessSurfaceEXT vkCreateHeadlessSurfaceEXT = 0;

    //=== VK_KHR_buffer_device_address ===
    PFN_vkGetBufferDeviceAddressKHR              vkGetBufferDeviceAddressKHR              = 0;
    PFN_vkGetBufferOpaqueCaptureAddressKHR       vkGetBufferOpaqueCaptureAddressKHR       = 0;
    PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR vkGetDeviceMemoryOpaqueCaptureAddressKHR = 0;

    //=== VK_EXT_line_rasterization ===
    PFN_vkCmdSetLineStippleEXT vkCmdSetLineStippleEXT = 0;

    //=== VK_EXT_host_query_reset ===
    PFN_vkResetQueryPoolEXT vkResetQueryPoolEXT = 0;

    //=== VK_EXT_extended_dynamic_state ===
    PFN_vkCmdSetCullModeEXT              vkCmdSetCullModeEXT              = 0;
    PFN_vkCmdSetFrontFaceEXT             vkCmdSetFrontFaceEXT             = 0;
    PFN_vkCmdSetPrimitiveTopologyEXT     vkCmdSetPrimitiveTopologyEXT     = 0;
    PFN_vkCmdSetViewportWithCountEXT     vkCmdSetViewportWithCountEXT     = 0;
    PFN_vkCmdSetScissorWithCountEXT      vkCmdSetScissorWithCountEXT      = 0;
    PFN_vkCmdBindVertexBuffers2EXT       vkCmdBindVertexBuffers2EXT       = 0;
    PFN_vkCmdSetDepthTestEnableEXT       vkCmdSetDepthTestEnableEXT       = 0;
    PFN_vkCmdSetDepthWriteEnableEXT      vkCmdSetDepthWriteEnableEXT      = 0;
    PFN_vkCmdSetDepthCompareOpEXT        vkCmdSetDepthCompareOpEXT        = 0;
    PFN_vkCmdSetDepthBoundsTestEnableEXT vkCmdSetDepthBoundsTestEnableEXT = 0;
    PFN_vkCmdSetStencilTestEnableEXT     vkCmdSetStencilTestEnableEXT     = 0;
    PFN_vkCmdSetStencilOpEXT             vkCmdSetStencilOpEXT             = 0;

    //=== VK_KHR_deferred_host_operations ===
    PFN_vkCreateDeferredOperationKHR            vkCreateDeferredOperationKHR            = 0;
    PFN_vkDestroyDeferredOperationKHR           vkDestroyDeferredOperationKHR           = 0;
    PFN_vkGetDeferredOperationMaxConcurrencyKHR vkGetDeferredOperationMaxConcurrencyKHR = 0;
    PFN_vkGetDeferredOperationResultKHR         vkGetDeferredOperationResultKHR         = 0;
    PFN_vkDeferredOperationJoinKHR              vkDeferredOperationJoinKHR              = 0;

    //=== VK_KHR_pipeline_executable_properties ===
    PFN_vkGetPipelineExecutablePropertiesKHR              vkGetPipelineExecutablePropertiesKHR              = 0;
    PFN_vkGetPipelineExecutableStatisticsKHR              vkGetPipelineExecutableStatisticsKHR              = 0;
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR vkGetPipelineExecutableInternalRepresentationsKHR = 0;

    //=== VK_EXT_host_image_copy ===
    PFN_vkCopyMemoryToImageEXT          vkCopyMemoryToImageEXT          = 0;
    PFN_vkCopyImageToMemoryEXT          vkCopyImageToMemoryEXT          = 0;
    PFN_vkCopyImageToImageEXT           vkCopyImageToImageEXT           = 0;
    PFN_vkTransitionImageLayoutEXT      vkTransitionImageLayoutEXT      = 0;
    PFN_vkGetImageSubresourceLayout2EXT vkGetImageSubresourceLayout2EXT = 0;

    //=== VK_KHR_map_memory2 ===
    PFN_vkMapMemory2KHR   vkMapMemory2KHR   = 0;
    PFN_vkUnmapMemory2KHR vkUnmapMemory2KHR = 0;

    //=== VK_EXT_swapchain_maintenance1 ===
    PFN_vkReleaseSwapchainImagesEXT vkReleaseSwapchainImagesEXT = 0;

    //=== VK_NV_device_generated_commands ===
    PFN_vkGetGeneratedCommandsMemoryRequirementsNV vkGetGeneratedCommandsMemoryRequirementsNV = 0;
    PFN_vkCmdPreprocessGeneratedCommandsNV         vkCmdPreprocessGeneratedCommandsNV         = 0;
    PFN_vkCmdExecuteGeneratedCommandsNV            vkCmdExecuteGeneratedCommandsNV            = 0;
    PFN_vkCmdBindPipelineShaderGroupNV             vkCmdBindPipelineShaderGroupNV             = 0;
    PFN_vkCreateIndirectCommandsLayoutNV           vkCreateIndirectCommandsLayoutNV           = 0;
    PFN_vkDestroyIndirectCommandsLayoutNV          vkDestroyIndirectCommandsLayoutNV          = 0;

    //=== VK_EXT_depth_bias_control ===
    PFN_vkCmdSetDepthBias2EXT vkCmdSetDepthBias2EXT = 0;

    //=== VK_EXT_acquire_drm_display ===
    PFN_vkAcquireDrmDisplayEXT vkAcquireDrmDisplayEXT = 0;
    PFN_vkGetDrmDisplayEXT     vkGetDrmDisplayEXT     = 0;

    //=== VK_EXT_private_data ===
    PFN_vkCreatePrivateDataSlotEXT  vkCreatePrivateDataSlotEXT  = 0;
    PFN_vkDestroyPrivateDataSlotEXT vkDestroyPrivateDataSlotEXT = 0;
    PFN_vkSetPrivateDataEXT         vkSetPrivateDataEXT         = 0;
    PFN_vkGetPrivateDataEXT         vkGetPrivateDataEXT         = 0;

    //=== VK_KHR_video_encode_queue ===
    PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR = 0;
    PFN_vkGetEncodedVideoSessionParametersKHR                   vkGetEncodedVideoSessionParametersKHR                   = 0;
    PFN_vkCmdEncodeVideoKHR                                     vkCmdEncodeVideoKHR                                     = 0;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_NV_cuda_kernel_launch ===
    PFN_vkCreateCudaModuleNV    vkCreateCudaModuleNV    = 0;
    PFN_vkGetCudaModuleCacheNV  vkGetCudaModuleCacheNV  = 0;
    PFN_vkCreateCudaFunctionNV  vkCreateCudaFunctionNV  = 0;
    PFN_vkDestroyCudaModuleNV   vkDestroyCudaModuleNV   = 0;
    PFN_vkDestroyCudaFunctionNV vkDestroyCudaFunctionNV = 0;
    PFN_vkCmdCudaLaunchKernelNV vkCmdCudaLaunchKernelNV = 0;
#else
    PFN_dummy vkCreateCudaModuleNV_placeholder                              = 0;
    PFN_dummy vkGetCudaModuleCacheNV_placeholder                            = 0;
    PFN_dummy vkCreateCudaFunctionNV_placeholder                            = 0;
    PFN_dummy vkDestroyCudaModuleNV_placeholder                             = 0;
    PFN_dummy vkDestroyCudaFunctionNV_placeholder                           = 0;
    PFN_dummy vkCmdCudaLaunchKernelNV_placeholder                           = 0;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
    //=== VK_EXT_metal_objects ===
    PFN_vkExportMetalObjectsEXT vkExportMetalObjectsEXT = 0;
#else
    PFN_dummy vkExportMetalObjectsEXT_placeholder                           = 0;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

    //=== VK_KHR_synchronization2 ===
    PFN_vkCmdSetEvent2KHR           vkCmdSetEvent2KHR           = 0;
    PFN_vkCmdResetEvent2KHR         vkCmdResetEvent2KHR         = 0;
    PFN_vkCmdWaitEvents2KHR         vkCmdWaitEvents2KHR         = 0;
    PFN_vkCmdPipelineBarrier2KHR    vkCmdPipelineBarrier2KHR    = 0;
    PFN_vkCmdWriteTimestamp2KHR     vkCmdWriteTimestamp2KHR     = 0;
    PFN_vkQueueSubmit2KHR           vkQueueSubmit2KHR           = 0;
    PFN_vkCmdWriteBufferMarker2AMD  vkCmdWriteBufferMarker2AMD  = 0;
    PFN_vkGetQueueCheckpointData2NV vkGetQueueCheckpointData2NV = 0;

    //=== VK_EXT_descriptor_buffer ===
    PFN_vkGetDescriptorSetLayoutSizeEXT                          vkGetDescriptorSetLayoutSizeEXT                          = 0;
    PFN_vkGetDescriptorSetLayoutBindingOffsetEXT                 vkGetDescriptorSetLayoutBindingOffsetEXT                 = 0;
    PFN_vkGetDescriptorEXT                                       vkGetDescriptorEXT                                       = 0;
    PFN_vkCmdBindDescriptorBuffersEXT                            vkCmdBindDescriptorBuffersEXT                            = 0;
    PFN_vkCmdSetDescriptorBufferOffsetsEXT                       vkCmdSetDescriptorBufferOffsetsEXT                       = 0;
    PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT             vkCmdBindDescriptorBufferEmbeddedSamplersEXT             = 0;
    PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT                vkGetBufferOpaqueCaptureDescriptorDataEXT                = 0;
    PFN_vkGetImageOpaqueCaptureDescriptorDataEXT                 vkGetImageOpaqueCaptureDescriptorDataEXT                 = 0;
    PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT             vkGetImageViewOpaqueCaptureDescriptorDataEXT             = 0;
    PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT               vkGetSamplerOpaqueCaptureDescriptorDataEXT               = 0;
    PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT = 0;

    //=== VK_NV_fragment_shading_rate_enums ===
    PFN_vkCmdSetFragmentShadingRateEnumNV vkCmdSetFragmentShadingRateEnumNV = 0;

    //=== VK_EXT_mesh_shader ===
    PFN_vkCmdDrawMeshTasksEXT              vkCmdDrawMeshTasksEXT              = 0;
    PFN_vkCmdDrawMeshTasksIndirectEXT      vkCmdDrawMeshTasksIndirectEXT      = 0;
    PFN_vkCmdDrawMeshTasksIndirectCountEXT vkCmdDrawMeshTasksIndirectCountEXT = 0;

    //=== VK_KHR_copy_commands2 ===
    PFN_vkCmdCopyBuffer2KHR        vkCmdCopyBuffer2KHR        = 0;
    PFN_vkCmdCopyImage2KHR         vkCmdCopyImage2KHR         = 0;
    PFN_vkCmdCopyBufferToImage2KHR vkCmdCopyBufferToImage2KHR = 0;
    PFN_vkCmdCopyImageToBuffer2KHR vkCmdCopyImageToBuffer2KHR = 0;
    PFN_vkCmdBlitImage2KHR         vkCmdBlitImage2KHR         = 0;
    PFN_vkCmdResolveImage2KHR      vkCmdResolveImage2KHR      = 0;

    //=== VK_EXT_device_fault ===
    PFN_vkGetDeviceFaultInfoEXT vkGetDeviceFaultInfoEXT = 0;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_NV_acquire_winrt_display ===
    PFN_vkAcquireWinrtDisplayNV vkAcquireWinrtDisplayNV = 0;
    PFN_vkGetWinrtDisplayNV     vkGetWinrtDisplayNV     = 0;
#else
    PFN_dummy vkAcquireWinrtDisplayNV_placeholder                           = 0;
    PFN_dummy vkGetWinrtDisplayNV_placeholder                               = 0;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
    //=== VK_EXT_directfb_surface ===
    PFN_vkCreateDirectFBSurfaceEXT                        vkCreateDirectFBSurfaceEXT                        = 0;
    PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT vkGetPhysicalDeviceDirectFBPresentationSupportEXT = 0;
#else
    PFN_dummy vkCreateDirectFBSurfaceEXT_placeholder                        = 0;
    PFN_dummy vkGetPhysicalDeviceDirectFBPresentationSupportEXT_placeholder = 0;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

    //=== VK_EXT_vertex_input_dynamic_state ===
    PFN_vkCmdSetVertexInputEXT vkCmdSetVertexInputEXT = 0;

#if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_external_memory ===
    PFN_vkGetMemoryZirconHandleFUCHSIA           vkGetMemoryZirconHandleFUCHSIA           = 0;
    PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA vkGetMemoryZirconHandlePropertiesFUCHSIA = 0;
#else
    PFN_dummy vkGetMemoryZirconHandleFUCHSIA_placeholder                    = 0;
    PFN_dummy vkGetMemoryZirconHandlePropertiesFUCHSIA_placeholder          = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_external_semaphore ===
    PFN_vkImportSemaphoreZirconHandleFUCHSIA vkImportSemaphoreZirconHandleFUCHSIA = 0;
    PFN_vkGetSemaphoreZirconHandleFUCHSIA    vkGetSemaphoreZirconHandleFUCHSIA    = 0;
#else
    PFN_dummy vkImportSemaphoreZirconHandleFUCHSIA_placeholder              = 0;
    PFN_dummy vkGetSemaphoreZirconHandleFUCHSIA_placeholder                 = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_buffer_collection ===
    PFN_vkCreateBufferCollectionFUCHSIA               vkCreateBufferCollectionFUCHSIA               = 0;
    PFN_vkSetBufferCollectionImageConstraintsFUCHSIA  vkSetBufferCollectionImageConstraintsFUCHSIA  = 0;
    PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA vkSetBufferCollectionBufferConstraintsFUCHSIA = 0;
    PFN_vkDestroyBufferCollectionFUCHSIA              vkDestroyBufferCollectionFUCHSIA              = 0;
    PFN_vkGetBufferCollectionPropertiesFUCHSIA        vkGetBufferCollectionPropertiesFUCHSIA        = 0;
#else
    PFN_dummy vkCreateBufferCollectionFUCHSIA_placeholder                   = 0;
    PFN_dummy vkSetBufferCollectionImageConstraintsFUCHSIA_placeholder      = 0;
    PFN_dummy vkSetBufferCollectionBufferConstraintsFUCHSIA_placeholder     = 0;
    PFN_dummy vkDestroyBufferCollectionFUCHSIA_placeholder                  = 0;
    PFN_dummy vkGetBufferCollectionPropertiesFUCHSIA_placeholder            = 0;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

    //=== VK_HUAWEI_subpass_shading ===
    PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI = 0;
    PFN_vkCmdSubpassShadingHUAWEI                       vkCmdSubpassShadingHUAWEI                       = 0;

    //=== VK_HUAWEI_invocation_mask ===
    PFN_vkCmdBindInvocationMaskHUAWEI vkCmdBindInvocationMaskHUAWEI = 0;

    //=== VK_NV_external_memory_rdma ===
    PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV = 0;

    //=== VK_EXT_pipeline_properties ===
    PFN_vkGetPipelinePropertiesEXT vkGetPipelinePropertiesEXT = 0;

    //=== VK_EXT_extended_dynamic_state2 ===
    PFN_vkCmdSetPatchControlPointsEXT      vkCmdSetPatchControlPointsEXT      = 0;
    PFN_vkCmdSetRasterizerDiscardEnableEXT vkCmdSetRasterizerDiscardEnableEXT = 0;
    PFN_vkCmdSetDepthBiasEnableEXT         vkCmdSetDepthBiasEnableEXT         = 0;
    PFN_vkCmdSetLogicOpEXT                 vkCmdSetLogicOpEXT                 = 0;
    PFN_vkCmdSetPrimitiveRestartEnableEXT  vkCmdSetPrimitiveRestartEnableEXT  = 0;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    //=== VK_QNX_screen_surface ===
    PFN_vkCreateScreenSurfaceQNX                        vkCreateScreenSurfaceQNX                        = 0;
    PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX vkGetPhysicalDeviceScreenPresentationSupportQNX = 0;
#else
    PFN_dummy vkCreateScreenSurfaceQNX_placeholder                          = 0;
    PFN_dummy vkGetPhysicalDeviceScreenPresentationSupportQNX_placeholder   = 0;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

    //=== VK_EXT_color_write_enable ===
    PFN_vkCmdSetColorWriteEnableEXT vkCmdSetColorWriteEnableEXT = 0;

    //=== VK_KHR_ray_tracing_maintenance1 ===
    PFN_vkCmdTraceRaysIndirect2KHR vkCmdTraceRaysIndirect2KHR = 0;

    //=== VK_EXT_multi_draw ===
    PFN_vkCmdDrawMultiEXT        vkCmdDrawMultiEXT        = 0;
    PFN_vkCmdDrawMultiIndexedEXT vkCmdDrawMultiIndexedEXT = 0;

    //=== VK_EXT_opacity_micromap ===
    PFN_vkCreateMicromapEXT                 vkCreateMicromapEXT                 = 0;
    PFN_vkDestroyMicromapEXT                vkDestroyMicromapEXT                = 0;
    PFN_vkCmdBuildMicromapsEXT              vkCmdBuildMicromapsEXT              = 0;
    PFN_vkBuildMicromapsEXT                 vkBuildMicromapsEXT                 = 0;
    PFN_vkCopyMicromapEXT                   vkCopyMicromapEXT                   = 0;
    PFN_vkCopyMicromapToMemoryEXT           vkCopyMicromapToMemoryEXT           = 0;
    PFN_vkCopyMemoryToMicromapEXT           vkCopyMemoryToMicromapEXT           = 0;
    PFN_vkWriteMicromapsPropertiesEXT       vkWriteMicromapsPropertiesEXT       = 0;
    PFN_vkCmdCopyMicromapEXT                vkCmdCopyMicromapEXT                = 0;
    PFN_vkCmdCopyMicromapToMemoryEXT        vkCmdCopyMicromapToMemoryEXT        = 0;
    PFN_vkCmdCopyMemoryToMicromapEXT        vkCmdCopyMemoryToMicromapEXT        = 0;
    PFN_vkCmdWriteMicromapsPropertiesEXT    vkCmdWriteMicromapsPropertiesEXT    = 0;
    PFN_vkGetDeviceMicromapCompatibilityEXT vkGetDeviceMicromapCompatibilityEXT = 0;
    PFN_vkGetMicromapBuildSizesEXT          vkGetMicromapBuildSizesEXT          = 0;

    //=== VK_HUAWEI_cluster_culling_shader ===
    PFN_vkCmdDrawClusterHUAWEI         vkCmdDrawClusterHUAWEI         = 0;
    PFN_vkCmdDrawClusterIndirectHUAWEI vkCmdDrawClusterIndirectHUAWEI = 0;

    //=== VK_EXT_pageable_device_local_memory ===
    PFN_vkSetDeviceMemoryPriorityEXT vkSetDeviceMemoryPriorityEXT = 0;

    //=== VK_KHR_maintenance4 ===
    PFN_vkGetDeviceBufferMemoryRequirementsKHR      vkGetDeviceBufferMemoryRequirementsKHR      = 0;
    PFN_vkGetDeviceImageMemoryRequirementsKHR       vkGetDeviceImageMemoryRequirementsKHR       = 0;
    PFN_vkGetDeviceImageSparseMemoryRequirementsKHR vkGetDeviceImageSparseMemoryRequirementsKHR = 0;

    //=== VK_VALVE_descriptor_set_host_mapping ===
    PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE vkGetDescriptorSetLayoutHostMappingInfoVALVE = 0;
    PFN_vkGetDescriptorSetHostMappingVALVE           vkGetDescriptorSetHostMappingVALVE           = 0;

    //=== VK_NV_copy_memory_indirect ===
    PFN_vkCmdCopyMemoryIndirectNV        vkCmdCopyMemoryIndirectNV        = 0;
    PFN_vkCmdCopyMemoryToImageIndirectNV vkCmdCopyMemoryToImageIndirectNV = 0;

    //=== VK_NV_memory_decompression ===
    PFN_vkCmdDecompressMemoryNV              vkCmdDecompressMemoryNV              = 0;
    PFN_vkCmdDecompressMemoryIndirectCountNV vkCmdDecompressMemoryIndirectCountNV = 0;

    //=== VK_NV_device_generated_commands_compute ===
    PFN_vkGetPipelineIndirectMemoryRequirementsNV vkGetPipelineIndirectMemoryRequirementsNV = 0;
    PFN_vkCmdUpdatePipelineIndirectBufferNV       vkCmdUpdatePipelineIndirectBufferNV       = 0;
    PFN_vkGetPipelineIndirectDeviceAddressNV      vkGetPipelineIndirectDeviceAddressNV      = 0;

    //=== VK_EXT_extended_dynamic_state3 ===
    PFN_vkCmdSetTessellationDomainOriginEXT         vkCmdSetTessellationDomainOriginEXT         = 0;
    PFN_vkCmdSetDepthClampEnableEXT                 vkCmdSetDepthClampEnableEXT                 = 0;
    PFN_vkCmdSetPolygonModeEXT                      vkCmdSetPolygonModeEXT                      = 0;
    PFN_vkCmdSetRasterizationSamplesEXT             vkCmdSetRasterizationSamplesEXT             = 0;
    PFN_vkCmdSetSampleMaskEXT                       vkCmdSetSampleMaskEXT                       = 0;
    PFN_vkCmdSetAlphaToCoverageEnableEXT            vkCmdSetAlphaToCoverageEnableEXT            = 0;
    PFN_vkCmdSetAlphaToOneEnableEXT                 vkCmdSetAlphaToOneEnableEXT                 = 0;
    PFN_vkCmdSetLogicOpEnableEXT                    vkCmdSetLogicOpEnableEXT                    = 0;
    PFN_vkCmdSetColorBlendEnableEXT                 vkCmdSetColorBlendEnableEXT                 = 0;
    PFN_vkCmdSetColorBlendEquationEXT               vkCmdSetColorBlendEquationEXT               = 0;
    PFN_vkCmdSetColorWriteMaskEXT                   vkCmdSetColorWriteMaskEXT                   = 0;
    PFN_vkCmdSetRasterizationStreamEXT              vkCmdSetRasterizationStreamEXT              = 0;
    PFN_vkCmdSetConservativeRasterizationModeEXT    vkCmdSetConservativeRasterizationModeEXT    = 0;
    PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT vkCmdSetExtraPrimitiveOverestimationSizeEXT = 0;
    PFN_vkCmdSetDepthClipEnableEXT                  vkCmdSetDepthClipEnableEXT                  = 0;
    PFN_vkCmdSetSampleLocationsEnableEXT            vkCmdSetSampleLocationsEnableEXT            = 0;
    PFN_vkCmdSetColorBlendAdvancedEXT               vkCmdSetColorBlendAdvancedEXT               = 0;
    PFN_vkCmdSetProvokingVertexModeEXT              vkCmdSetProvokingVertexModeEXT              = 0;
    PFN_vkCmdSetLineRasterizationModeEXT            vkCmdSetLineRasterizationModeEXT            = 0;
    PFN_vkCmdSetLineStippleEnableEXT                vkCmdSetLineStippleEnableEXT                = 0;
    PFN_vkCmdSetDepthClipNegativeOneToOneEXT        vkCmdSetDepthClipNegativeOneToOneEXT        = 0;
    PFN_vkCmdSetViewportWScalingEnableNV            vkCmdSetViewportWScalingEnableNV            = 0;
    PFN_vkCmdSetViewportSwizzleNV                   vkCmdSetViewportSwizzleNV                   = 0;
    PFN_vkCmdSetCoverageToColorEnableNV             vkCmdSetCoverageToColorEnableNV             = 0;
    PFN_vkCmdSetCoverageToColorLocationNV           vkCmdSetCoverageToColorLocationNV           = 0;
    PFN_vkCmdSetCoverageModulationModeNV            vkCmdSetCoverageModulationModeNV            = 0;
    PFN_vkCmdSetCoverageModulationTableEnableNV     vkCmdSetCoverageModulationTableEnableNV     = 0;
    PFN_vkCmdSetCoverageModulationTableNV           vkCmdSetCoverageModulationTableNV           = 0;
    PFN_vkCmdSetShadingRateImageEnableNV            vkCmdSetShadingRateImageEnableNV            = 0;
    PFN_vkCmdSetRepresentativeFragmentTestEnableNV  vkCmdSetRepresentativeFragmentTestEnableNV  = 0;
    PFN_vkCmdSetCoverageReductionModeNV             vkCmdSetCoverageReductionModeNV             = 0;

    //=== VK_EXT_shader_module_identifier ===
    PFN_vkGetShaderModuleIdentifierEXT           vkGetShaderModuleIdentifierEXT           = 0;
    PFN_vkGetShaderModuleCreateInfoIdentifierEXT vkGetShaderModuleCreateInfoIdentifierEXT = 0;

    //=== VK_NV_optical_flow ===
    PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV vkGetPhysicalDeviceOpticalFlowImageFormatsNV = 0;
    PFN_vkCreateOpticalFlowSessionNV                 vkCreateOpticalFlowSessionNV                 = 0;
    PFN_vkDestroyOpticalFlowSessionNV                vkDestroyOpticalFlowSessionNV                = 0;
    PFN_vkBindOpticalFlowSessionImageNV              vkBindOpticalFlowSessionImageNV              = 0;
    PFN_vkCmdOpticalFlowExecuteNV                    vkCmdOpticalFlowExecuteNV                    = 0;

    //=== VK_KHR_maintenance5 ===
    PFN_vkCmdBindIndexBuffer2KHR             vkCmdBindIndexBuffer2KHR             = 0;
    PFN_vkGetRenderingAreaGranularityKHR     vkGetRenderingAreaGranularityKHR     = 0;
    PFN_vkGetDeviceImageSubresourceLayoutKHR vkGetDeviceImageSubresourceLayoutKHR = 0;
    PFN_vkGetImageSubresourceLayout2KHR      vkGetImageSubresourceLayout2KHR      = 0;

    //=== VK_EXT_shader_object ===
    PFN_vkCreateShadersEXT       vkCreateShadersEXT       = 0;
    PFN_vkDestroyShaderEXT       vkDestroyShaderEXT       = 0;
    PFN_vkGetShaderBinaryDataEXT vkGetShaderBinaryDataEXT = 0;
    PFN_vkCmdBindShadersEXT      vkCmdBindShadersEXT      = 0;

    //=== VK_QCOM_tile_properties ===
    PFN_vkGetFramebufferTilePropertiesQCOM      vkGetFramebufferTilePropertiesQCOM      = 0;
    PFN_vkGetDynamicRenderingTilePropertiesQCOM vkGetDynamicRenderingTilePropertiesQCOM = 0;

    //=== VK_NV_low_latency2 ===
    PFN_vkSetLatencySleepModeNV  vkSetLatencySleepModeNV  = 0;
    PFN_vkLatencySleepNV         vkLatencySleepNV         = 0;
    PFN_vkSetLatencyMarkerNV     vkSetLatencyMarkerNV     = 0;
    PFN_vkGetLatencyTimingsNV    vkGetLatencyTimingsNV    = 0;
    PFN_vkQueueNotifyOutOfBandNV vkQueueNotifyOutOfBandNV = 0;

    //=== VK_KHR_cooperative_matrix ===
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = 0;

    //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
    PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT vkCmdSetAttachmentFeedbackLoopEnableEXT = 0;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    //=== VK_QNX_external_memory_screen_buffer ===
    PFN_vkGetScreenBufferPropertiesQNX vkGetScreenBufferPropertiesQNX = 0;
#else
    PFN_dummy vkGetScreenBufferPropertiesQNX_placeholder                    = 0;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

    //=== VK_KHR_calibrated_timestamps ===
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = 0;
    PFN_vkGetCalibratedTimestampsKHR                   vkGetCalibratedTimestampsKHR                   = 0;

    //=== VK_KHR_maintenance6 ===
    PFN_vkCmdBindDescriptorSets2KHR                   vkCmdBindDescriptorSets2KHR                   = 0;
    PFN_vkCmdPushConstants2KHR                        vkCmdPushConstants2KHR                        = 0;
    PFN_vkCmdPushDescriptorSet2KHR                    vkCmdPushDescriptorSet2KHR                    = 0;
    PFN_vkCmdPushDescriptorSetWithTemplate2KHR        vkCmdPushDescriptorSetWithTemplate2KHR        = 0;
    PFN_vkCmdSetDescriptorBufferOffsets2EXT           vkCmdSetDescriptorBufferOffsets2EXT           = 0;
    PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT vkCmdBindDescriptorBufferEmbeddedSamplers2EXT = 0;

  public:
    DispatchLoaderDynamic() VULKAN_HPP_NOEXCEPT                                    = default;
    DispatchLoaderDynamic( DispatchLoaderDynamic const & rhs ) VULKAN_HPP_NOEXCEPT = default;

    DispatchLoaderDynamic( PFN_vkGetInstanceProcAddr getInstanceProcAddr ) VULKAN_HPP_NOEXCEPT
    {
      init( getInstanceProcAddr );
    }

    // This interface does not require a linked vulkan library.
    DispatchLoaderDynamic( VkInstance                instance,
                           PFN_vkGetInstanceProcAddr getInstanceProcAddr,
                           VkDevice                  device            = {},
                           PFN_vkGetDeviceProcAddr   getDeviceProcAddr = nullptr ) VULKAN_HPP_NOEXCEPT
    {
      init( instance, getInstanceProcAddr, device, getDeviceProcAddr );
    }

    template <typename DynamicLoader
#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
              = VULKAN_HPP_NAMESPACE::DynamicLoader
#endif
              >
    void init()
    {
      static DynamicLoader dl;
      init( dl );
    }

    template <typename DynamicLoader>
    void init( DynamicLoader const & dl ) VULKAN_HPP_NOEXCEPT
    {
      PFN_vkGetInstanceProcAddr getInstanceProcAddr = dl.template getProcAddress<PFN_vkGetInstanceProcAddr>( "vkGetInstanceProcAddr" );
      init( getInstanceProcAddr );
    }

    void init( PFN_vkGetInstanceProcAddr getInstanceProcAddr ) VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getInstanceProcAddr );

      vkGetInstanceProcAddr = getInstanceProcAddr;

      //=== VK_VERSION_1_0 ===
      vkCreateInstance = PFN_vkCreateInstance( vkGetInstanceProcAddr( NULL, "vkCreateInstance" ) );
      vkEnumerateInstanceExtensionProperties =
        PFN_vkEnumerateInstanceExtensionProperties( vkGetInstanceProcAddr( NULL, "vkEnumerateInstanceExtensionProperties" ) );
      vkEnumerateInstanceLayerProperties = PFN_vkEnumerateInstanceLayerProperties( vkGetInstanceProcAddr( NULL, "vkEnumerateInstanceLayerProperties" ) );

      //=== VK_VERSION_1_1 ===
      vkEnumerateInstanceVersion = PFN_vkEnumerateInstanceVersion( vkGetInstanceProcAddr( NULL, "vkEnumerateInstanceVersion" ) );
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

      //=== VK_VERSION_1_0 ===
      vkDestroyInstance                   = PFN_vkDestroyInstance( vkGetInstanceProcAddr( instance, "vkDestroyInstance" ) );
      vkEnumeratePhysicalDevices          = PFN_vkEnumeratePhysicalDevices( vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDevices" ) );
      vkGetPhysicalDeviceFeatures         = PFN_vkGetPhysicalDeviceFeatures( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures" ) );
      vkGetPhysicalDeviceFormatProperties = PFN_vkGetPhysicalDeviceFormatProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties" ) );
      vkGetPhysicalDeviceImageFormatProperties =
        PFN_vkGetPhysicalDeviceImageFormatProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties" ) );
      vkGetPhysicalDeviceProperties = PFN_vkGetPhysicalDeviceProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties" ) );
      vkGetPhysicalDeviceQueueFamilyProperties =
        PFN_vkGetPhysicalDeviceQueueFamilyProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties" ) );
      vkGetPhysicalDeviceMemoryProperties = PFN_vkGetPhysicalDeviceMemoryProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties" ) );
      vkGetDeviceProcAddr                 = PFN_vkGetDeviceProcAddr( vkGetInstanceProcAddr( instance, "vkGetDeviceProcAddr" ) );
      vkCreateDevice                      = PFN_vkCreateDevice( vkGetInstanceProcAddr( instance, "vkCreateDevice" ) );
      vkDestroyDevice                     = PFN_vkDestroyDevice( vkGetInstanceProcAddr( instance, "vkDestroyDevice" ) );
      vkEnumerateDeviceExtensionProperties =
        PFN_vkEnumerateDeviceExtensionProperties( vkGetInstanceProcAddr( instance, "vkEnumerateDeviceExtensionProperties" ) );
      vkEnumerateDeviceLayerProperties   = PFN_vkEnumerateDeviceLayerProperties( vkGetInstanceProcAddr( instance, "vkEnumerateDeviceLayerProperties" ) );
      vkGetDeviceQueue                   = PFN_vkGetDeviceQueue( vkGetInstanceProcAddr( instance, "vkGetDeviceQueue" ) );
      vkQueueSubmit                      = PFN_vkQueueSubmit( vkGetInstanceProcAddr( instance, "vkQueueSubmit" ) );
      vkQueueWaitIdle                    = PFN_vkQueueWaitIdle( vkGetInstanceProcAddr( instance, "vkQueueWaitIdle" ) );
      vkDeviceWaitIdle                   = PFN_vkDeviceWaitIdle( vkGetInstanceProcAddr( instance, "vkDeviceWaitIdle" ) );
      vkAllocateMemory                   = PFN_vkAllocateMemory( vkGetInstanceProcAddr( instance, "vkAllocateMemory" ) );
      vkFreeMemory                       = PFN_vkFreeMemory( vkGetInstanceProcAddr( instance, "vkFreeMemory" ) );
      vkMapMemory                        = PFN_vkMapMemory( vkGetInstanceProcAddr( instance, "vkMapMemory" ) );
      vkUnmapMemory                      = PFN_vkUnmapMemory( vkGetInstanceProcAddr( instance, "vkUnmapMemory" ) );
      vkFlushMappedMemoryRanges          = PFN_vkFlushMappedMemoryRanges( vkGetInstanceProcAddr( instance, "vkFlushMappedMemoryRanges" ) );
      vkInvalidateMappedMemoryRanges     = PFN_vkInvalidateMappedMemoryRanges( vkGetInstanceProcAddr( instance, "vkInvalidateMappedMemoryRanges" ) );
      vkGetDeviceMemoryCommitment        = PFN_vkGetDeviceMemoryCommitment( vkGetInstanceProcAddr( instance, "vkGetDeviceMemoryCommitment" ) );
      vkBindBufferMemory                 = PFN_vkBindBufferMemory( vkGetInstanceProcAddr( instance, "vkBindBufferMemory" ) );
      vkBindImageMemory                  = PFN_vkBindImageMemory( vkGetInstanceProcAddr( instance, "vkBindImageMemory" ) );
      vkGetBufferMemoryRequirements      = PFN_vkGetBufferMemoryRequirements( vkGetInstanceProcAddr( instance, "vkGetBufferMemoryRequirements" ) );
      vkGetImageMemoryRequirements       = PFN_vkGetImageMemoryRequirements( vkGetInstanceProcAddr( instance, "vkGetImageMemoryRequirements" ) );
      vkGetImageSparseMemoryRequirements = PFN_vkGetImageSparseMemoryRequirements( vkGetInstanceProcAddr( instance, "vkGetImageSparseMemoryRequirements" ) );
      vkGetPhysicalDeviceSparseImageFormatProperties =
        PFN_vkGetPhysicalDeviceSparseImageFormatProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties" ) );
      vkQueueBindSparse            = PFN_vkQueueBindSparse( vkGetInstanceProcAddr( instance, "vkQueueBindSparse" ) );
      vkCreateFence                = PFN_vkCreateFence( vkGetInstanceProcAddr( instance, "vkCreateFence" ) );
      vkDestroyFence               = PFN_vkDestroyFence( vkGetInstanceProcAddr( instance, "vkDestroyFence" ) );
      vkResetFences                = PFN_vkResetFences( vkGetInstanceProcAddr( instance, "vkResetFences" ) );
      vkGetFenceStatus             = PFN_vkGetFenceStatus( vkGetInstanceProcAddr( instance, "vkGetFenceStatus" ) );
      vkWaitForFences              = PFN_vkWaitForFences( vkGetInstanceProcAddr( instance, "vkWaitForFences" ) );
      vkCreateSemaphore            = PFN_vkCreateSemaphore( vkGetInstanceProcAddr( instance, "vkCreateSemaphore" ) );
      vkDestroySemaphore           = PFN_vkDestroySemaphore( vkGetInstanceProcAddr( instance, "vkDestroySemaphore" ) );
      vkCreateEvent                = PFN_vkCreateEvent( vkGetInstanceProcAddr( instance, "vkCreateEvent" ) );
      vkDestroyEvent               = PFN_vkDestroyEvent( vkGetInstanceProcAddr( instance, "vkDestroyEvent" ) );
      vkGetEventStatus             = PFN_vkGetEventStatus( vkGetInstanceProcAddr( instance, "vkGetEventStatus" ) );
      vkSetEvent                   = PFN_vkSetEvent( vkGetInstanceProcAddr( instance, "vkSetEvent" ) );
      vkResetEvent                 = PFN_vkResetEvent( vkGetInstanceProcAddr( instance, "vkResetEvent" ) );
      vkCreateQueryPool            = PFN_vkCreateQueryPool( vkGetInstanceProcAddr( instance, "vkCreateQueryPool" ) );
      vkDestroyQueryPool           = PFN_vkDestroyQueryPool( vkGetInstanceProcAddr( instance, "vkDestroyQueryPool" ) );
      vkGetQueryPoolResults        = PFN_vkGetQueryPoolResults( vkGetInstanceProcAddr( instance, "vkGetQueryPoolResults" ) );
      vkCreateBuffer               = PFN_vkCreateBuffer( vkGetInstanceProcAddr( instance, "vkCreateBuffer" ) );
      vkDestroyBuffer              = PFN_vkDestroyBuffer( vkGetInstanceProcAddr( instance, "vkDestroyBuffer" ) );
      vkCreateBufferView           = PFN_vkCreateBufferView( vkGetInstanceProcAddr( instance, "vkCreateBufferView" ) );
      vkDestroyBufferView          = PFN_vkDestroyBufferView( vkGetInstanceProcAddr( instance, "vkDestroyBufferView" ) );
      vkCreateImage                = PFN_vkCreateImage( vkGetInstanceProcAddr( instance, "vkCreateImage" ) );
      vkDestroyImage               = PFN_vkDestroyImage( vkGetInstanceProcAddr( instance, "vkDestroyImage" ) );
      vkGetImageSubresourceLayout  = PFN_vkGetImageSubresourceLayout( vkGetInstanceProcAddr( instance, "vkGetImageSubresourceLayout" ) );
      vkCreateImageView            = PFN_vkCreateImageView( vkGetInstanceProcAddr( instance, "vkCreateImageView" ) );
      vkDestroyImageView           = PFN_vkDestroyImageView( vkGetInstanceProcAddr( instance, "vkDestroyImageView" ) );
      vkCreateShaderModule         = PFN_vkCreateShaderModule( vkGetInstanceProcAddr( instance, "vkCreateShaderModule" ) );
      vkDestroyShaderModule        = PFN_vkDestroyShaderModule( vkGetInstanceProcAddr( instance, "vkDestroyShaderModule" ) );
      vkCreatePipelineCache        = PFN_vkCreatePipelineCache( vkGetInstanceProcAddr( instance, "vkCreatePipelineCache" ) );
      vkDestroyPipelineCache       = PFN_vkDestroyPipelineCache( vkGetInstanceProcAddr( instance, "vkDestroyPipelineCache" ) );
      vkGetPipelineCacheData       = PFN_vkGetPipelineCacheData( vkGetInstanceProcAddr( instance, "vkGetPipelineCacheData" ) );
      vkMergePipelineCaches        = PFN_vkMergePipelineCaches( vkGetInstanceProcAddr( instance, "vkMergePipelineCaches" ) );
      vkCreateGraphicsPipelines    = PFN_vkCreateGraphicsPipelines( vkGetInstanceProcAddr( instance, "vkCreateGraphicsPipelines" ) );
      vkCreateComputePipelines     = PFN_vkCreateComputePipelines( vkGetInstanceProcAddr( instance, "vkCreateComputePipelines" ) );
      vkDestroyPipeline            = PFN_vkDestroyPipeline( vkGetInstanceProcAddr( instance, "vkDestroyPipeline" ) );
      vkCreatePipelineLayout       = PFN_vkCreatePipelineLayout( vkGetInstanceProcAddr( instance, "vkCreatePipelineLayout" ) );
      vkDestroyPipelineLayout      = PFN_vkDestroyPipelineLayout( vkGetInstanceProcAddr( instance, "vkDestroyPipelineLayout" ) );
      vkCreateSampler              = PFN_vkCreateSampler( vkGetInstanceProcAddr( instance, "vkCreateSampler" ) );
      vkDestroySampler             = PFN_vkDestroySampler( vkGetInstanceProcAddr( instance, "vkDestroySampler" ) );
      vkCreateDescriptorSetLayout  = PFN_vkCreateDescriptorSetLayout( vkGetInstanceProcAddr( instance, "vkCreateDescriptorSetLayout" ) );
      vkDestroyDescriptorSetLayout = PFN_vkDestroyDescriptorSetLayout( vkGetInstanceProcAddr( instance, "vkDestroyDescriptorSetLayout" ) );
      vkCreateDescriptorPool       = PFN_vkCreateDescriptorPool( vkGetInstanceProcAddr( instance, "vkCreateDescriptorPool" ) );
      vkDestroyDescriptorPool      = PFN_vkDestroyDescriptorPool( vkGetInstanceProcAddr( instance, "vkDestroyDescriptorPool" ) );
      vkResetDescriptorPool        = PFN_vkResetDescriptorPool( vkGetInstanceProcAddr( instance, "vkResetDescriptorPool" ) );
      vkAllocateDescriptorSets     = PFN_vkAllocateDescriptorSets( vkGetInstanceProcAddr( instance, "vkAllocateDescriptorSets" ) );
      vkFreeDescriptorSets         = PFN_vkFreeDescriptorSets( vkGetInstanceProcAddr( instance, "vkFreeDescriptorSets" ) );
      vkUpdateDescriptorSets       = PFN_vkUpdateDescriptorSets( vkGetInstanceProcAddr( instance, "vkUpdateDescriptorSets" ) );
      vkCreateFramebuffer          = PFN_vkCreateFramebuffer( vkGetInstanceProcAddr( instance, "vkCreateFramebuffer" ) );
      vkDestroyFramebuffer         = PFN_vkDestroyFramebuffer( vkGetInstanceProcAddr( instance, "vkDestroyFramebuffer" ) );
      vkCreateRenderPass           = PFN_vkCreateRenderPass( vkGetInstanceProcAddr( instance, "vkCreateRenderPass" ) );
      vkDestroyRenderPass          = PFN_vkDestroyRenderPass( vkGetInstanceProcAddr( instance, "vkDestroyRenderPass" ) );
      vkGetRenderAreaGranularity   = PFN_vkGetRenderAreaGranularity( vkGetInstanceProcAddr( instance, "vkGetRenderAreaGranularity" ) );
      vkCreateCommandPool          = PFN_vkCreateCommandPool( vkGetInstanceProcAddr( instance, "vkCreateCommandPool" ) );
      vkDestroyCommandPool         = PFN_vkDestroyCommandPool( vkGetInstanceProcAddr( instance, "vkDestroyCommandPool" ) );
      vkResetCommandPool           = PFN_vkResetCommandPool( vkGetInstanceProcAddr( instance, "vkResetCommandPool" ) );
      vkAllocateCommandBuffers     = PFN_vkAllocateCommandBuffers( vkGetInstanceProcAddr( instance, "vkAllocateCommandBuffers" ) );
      vkFreeCommandBuffers         = PFN_vkFreeCommandBuffers( vkGetInstanceProcAddr( instance, "vkFreeCommandBuffers" ) );
      vkBeginCommandBuffer         = PFN_vkBeginCommandBuffer( vkGetInstanceProcAddr( instance, "vkBeginCommandBuffer" ) );
      vkEndCommandBuffer           = PFN_vkEndCommandBuffer( vkGetInstanceProcAddr( instance, "vkEndCommandBuffer" ) );
      vkResetCommandBuffer         = PFN_vkResetCommandBuffer( vkGetInstanceProcAddr( instance, "vkResetCommandBuffer" ) );
      vkCmdBindPipeline            = PFN_vkCmdBindPipeline( vkGetInstanceProcAddr( instance, "vkCmdBindPipeline" ) );
      vkCmdSetViewport             = PFN_vkCmdSetViewport( vkGetInstanceProcAddr( instance, "vkCmdSetViewport" ) );
      vkCmdSetScissor              = PFN_vkCmdSetScissor( vkGetInstanceProcAddr( instance, "vkCmdSetScissor" ) );
      vkCmdSetLineWidth            = PFN_vkCmdSetLineWidth( vkGetInstanceProcAddr( instance, "vkCmdSetLineWidth" ) );
      vkCmdSetDepthBias            = PFN_vkCmdSetDepthBias( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBias" ) );
      vkCmdSetBlendConstants       = PFN_vkCmdSetBlendConstants( vkGetInstanceProcAddr( instance, "vkCmdSetBlendConstants" ) );
      vkCmdSetDepthBounds          = PFN_vkCmdSetDepthBounds( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBounds" ) );
      vkCmdSetStencilCompareMask   = PFN_vkCmdSetStencilCompareMask( vkGetInstanceProcAddr( instance, "vkCmdSetStencilCompareMask" ) );
      vkCmdSetStencilWriteMask     = PFN_vkCmdSetStencilWriteMask( vkGetInstanceProcAddr( instance, "vkCmdSetStencilWriteMask" ) );
      vkCmdSetStencilReference     = PFN_vkCmdSetStencilReference( vkGetInstanceProcAddr( instance, "vkCmdSetStencilReference" ) );
      vkCmdBindDescriptorSets      = PFN_vkCmdBindDescriptorSets( vkGetInstanceProcAddr( instance, "vkCmdBindDescriptorSets" ) );
      vkCmdBindIndexBuffer         = PFN_vkCmdBindIndexBuffer( vkGetInstanceProcAddr( instance, "vkCmdBindIndexBuffer" ) );
      vkCmdBindVertexBuffers       = PFN_vkCmdBindVertexBuffers( vkGetInstanceProcAddr( instance, "vkCmdBindVertexBuffers" ) );
      vkCmdDraw                    = PFN_vkCmdDraw( vkGetInstanceProcAddr( instance, "vkCmdDraw" ) );
      vkCmdDrawIndexed             = PFN_vkCmdDrawIndexed( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexed" ) );
      vkCmdDrawIndirect            = PFN_vkCmdDrawIndirect( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirect" ) );
      vkCmdDrawIndexedIndirect     = PFN_vkCmdDrawIndexedIndirect( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexedIndirect" ) );
      vkCmdDispatch                = PFN_vkCmdDispatch( vkGetInstanceProcAddr( instance, "vkCmdDispatch" ) );
      vkCmdDispatchIndirect        = PFN_vkCmdDispatchIndirect( vkGetInstanceProcAddr( instance, "vkCmdDispatchIndirect" ) );
      vkCmdCopyBuffer              = PFN_vkCmdCopyBuffer( vkGetInstanceProcAddr( instance, "vkCmdCopyBuffer" ) );
      vkCmdCopyImage               = PFN_vkCmdCopyImage( vkGetInstanceProcAddr( instance, "vkCmdCopyImage" ) );
      vkCmdBlitImage               = PFN_vkCmdBlitImage( vkGetInstanceProcAddr( instance, "vkCmdBlitImage" ) );
      vkCmdCopyBufferToImage       = PFN_vkCmdCopyBufferToImage( vkGetInstanceProcAddr( instance, "vkCmdCopyBufferToImage" ) );
      vkCmdCopyImageToBuffer       = PFN_vkCmdCopyImageToBuffer( vkGetInstanceProcAddr( instance, "vkCmdCopyImageToBuffer" ) );
      vkCmdUpdateBuffer            = PFN_vkCmdUpdateBuffer( vkGetInstanceProcAddr( instance, "vkCmdUpdateBuffer" ) );
      vkCmdFillBuffer              = PFN_vkCmdFillBuffer( vkGetInstanceProcAddr( instance, "vkCmdFillBuffer" ) );
      vkCmdClearColorImage         = PFN_vkCmdClearColorImage( vkGetInstanceProcAddr( instance, "vkCmdClearColorImage" ) );
      vkCmdClearDepthStencilImage  = PFN_vkCmdClearDepthStencilImage( vkGetInstanceProcAddr( instance, "vkCmdClearDepthStencilImage" ) );
      vkCmdClearAttachments        = PFN_vkCmdClearAttachments( vkGetInstanceProcAddr( instance, "vkCmdClearAttachments" ) );
      vkCmdResolveImage            = PFN_vkCmdResolveImage( vkGetInstanceProcAddr( instance, "vkCmdResolveImage" ) );
      vkCmdSetEvent                = PFN_vkCmdSetEvent( vkGetInstanceProcAddr( instance, "vkCmdSetEvent" ) );
      vkCmdResetEvent              = PFN_vkCmdResetEvent( vkGetInstanceProcAddr( instance, "vkCmdResetEvent" ) );
      vkCmdWaitEvents              = PFN_vkCmdWaitEvents( vkGetInstanceProcAddr( instance, "vkCmdWaitEvents" ) );
      vkCmdPipelineBarrier         = PFN_vkCmdPipelineBarrier( vkGetInstanceProcAddr( instance, "vkCmdPipelineBarrier" ) );
      vkCmdBeginQuery              = PFN_vkCmdBeginQuery( vkGetInstanceProcAddr( instance, "vkCmdBeginQuery" ) );
      vkCmdEndQuery                = PFN_vkCmdEndQuery( vkGetInstanceProcAddr( instance, "vkCmdEndQuery" ) );
      vkCmdResetQueryPool          = PFN_vkCmdResetQueryPool( vkGetInstanceProcAddr( instance, "vkCmdResetQueryPool" ) );
      vkCmdWriteTimestamp          = PFN_vkCmdWriteTimestamp( vkGetInstanceProcAddr( instance, "vkCmdWriteTimestamp" ) );
      vkCmdCopyQueryPoolResults    = PFN_vkCmdCopyQueryPoolResults( vkGetInstanceProcAddr( instance, "vkCmdCopyQueryPoolResults" ) );
      vkCmdPushConstants           = PFN_vkCmdPushConstants( vkGetInstanceProcAddr( instance, "vkCmdPushConstants" ) );
      vkCmdBeginRenderPass         = PFN_vkCmdBeginRenderPass( vkGetInstanceProcAddr( instance, "vkCmdBeginRenderPass" ) );
      vkCmdNextSubpass             = PFN_vkCmdNextSubpass( vkGetInstanceProcAddr( instance, "vkCmdNextSubpass" ) );
      vkCmdEndRenderPass           = PFN_vkCmdEndRenderPass( vkGetInstanceProcAddr( instance, "vkCmdEndRenderPass" ) );
      vkCmdExecuteCommands         = PFN_vkCmdExecuteCommands( vkGetInstanceProcAddr( instance, "vkCmdExecuteCommands" ) );

      //=== VK_VERSION_1_1 ===
      vkBindBufferMemory2                 = PFN_vkBindBufferMemory2( vkGetInstanceProcAddr( instance, "vkBindBufferMemory2" ) );
      vkBindImageMemory2                  = PFN_vkBindImageMemory2( vkGetInstanceProcAddr( instance, "vkBindImageMemory2" ) );
      vkGetDeviceGroupPeerMemoryFeatures  = PFN_vkGetDeviceGroupPeerMemoryFeatures( vkGetInstanceProcAddr( instance, "vkGetDeviceGroupPeerMemoryFeatures" ) );
      vkCmdSetDeviceMask                  = PFN_vkCmdSetDeviceMask( vkGetInstanceProcAddr( instance, "vkCmdSetDeviceMask" ) );
      vkCmdDispatchBase                   = PFN_vkCmdDispatchBase( vkGetInstanceProcAddr( instance, "vkCmdDispatchBase" ) );
      vkEnumeratePhysicalDeviceGroups     = PFN_vkEnumeratePhysicalDeviceGroups( vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceGroups" ) );
      vkGetImageMemoryRequirements2       = PFN_vkGetImageMemoryRequirements2( vkGetInstanceProcAddr( instance, "vkGetImageMemoryRequirements2" ) );
      vkGetBufferMemoryRequirements2      = PFN_vkGetBufferMemoryRequirements2( vkGetInstanceProcAddr( instance, "vkGetBufferMemoryRequirements2" ) );
      vkGetImageSparseMemoryRequirements2 = PFN_vkGetImageSparseMemoryRequirements2( vkGetInstanceProcAddr( instance, "vkGetImageSparseMemoryRequirements2" ) );
      vkGetPhysicalDeviceFeatures2        = PFN_vkGetPhysicalDeviceFeatures2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures2" ) );
      vkGetPhysicalDeviceProperties2      = PFN_vkGetPhysicalDeviceProperties2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties2" ) );
      vkGetPhysicalDeviceFormatProperties2 =
        PFN_vkGetPhysicalDeviceFormatProperties2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties2" ) );
      vkGetPhysicalDeviceImageFormatProperties2 =
        PFN_vkGetPhysicalDeviceImageFormatProperties2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties2" ) );
      vkGetPhysicalDeviceQueueFamilyProperties2 =
        PFN_vkGetPhysicalDeviceQueueFamilyProperties2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties2" ) );
      vkGetPhysicalDeviceMemoryProperties2 =
        PFN_vkGetPhysicalDeviceMemoryProperties2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties2" ) );
      vkGetPhysicalDeviceSparseImageFormatProperties2 =
        PFN_vkGetPhysicalDeviceSparseImageFormatProperties2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties2" ) );
      vkTrimCommandPool                 = PFN_vkTrimCommandPool( vkGetInstanceProcAddr( instance, "vkTrimCommandPool" ) );
      vkGetDeviceQueue2                 = PFN_vkGetDeviceQueue2( vkGetInstanceProcAddr( instance, "vkGetDeviceQueue2" ) );
      vkCreateSamplerYcbcrConversion    = PFN_vkCreateSamplerYcbcrConversion( vkGetInstanceProcAddr( instance, "vkCreateSamplerYcbcrConversion" ) );
      vkDestroySamplerYcbcrConversion   = PFN_vkDestroySamplerYcbcrConversion( vkGetInstanceProcAddr( instance, "vkDestroySamplerYcbcrConversion" ) );
      vkCreateDescriptorUpdateTemplate  = PFN_vkCreateDescriptorUpdateTemplate( vkGetInstanceProcAddr( instance, "vkCreateDescriptorUpdateTemplate" ) );
      vkDestroyDescriptorUpdateTemplate = PFN_vkDestroyDescriptorUpdateTemplate( vkGetInstanceProcAddr( instance, "vkDestroyDescriptorUpdateTemplate" ) );
      vkUpdateDescriptorSetWithTemplate = PFN_vkUpdateDescriptorSetWithTemplate( vkGetInstanceProcAddr( instance, "vkUpdateDescriptorSetWithTemplate" ) );
      vkGetPhysicalDeviceExternalBufferProperties =
        PFN_vkGetPhysicalDeviceExternalBufferProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalBufferProperties" ) );
      vkGetPhysicalDeviceExternalFenceProperties =
        PFN_vkGetPhysicalDeviceExternalFenceProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalFenceProperties" ) );
      vkGetPhysicalDeviceExternalSemaphoreProperties =
        PFN_vkGetPhysicalDeviceExternalSemaphoreProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalSemaphoreProperties" ) );
      vkGetDescriptorSetLayoutSupport = PFN_vkGetDescriptorSetLayoutSupport( vkGetInstanceProcAddr( instance, "vkGetDescriptorSetLayoutSupport" ) );

      //=== VK_VERSION_1_2 ===
      vkCmdDrawIndirectCount          = PFN_vkCmdDrawIndirectCount( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirectCount" ) );
      vkCmdDrawIndexedIndirectCount   = PFN_vkCmdDrawIndexedIndirectCount( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexedIndirectCount" ) );
      vkCreateRenderPass2             = PFN_vkCreateRenderPass2( vkGetInstanceProcAddr( instance, "vkCreateRenderPass2" ) );
      vkCmdBeginRenderPass2           = PFN_vkCmdBeginRenderPass2( vkGetInstanceProcAddr( instance, "vkCmdBeginRenderPass2" ) );
      vkCmdNextSubpass2               = PFN_vkCmdNextSubpass2( vkGetInstanceProcAddr( instance, "vkCmdNextSubpass2" ) );
      vkCmdEndRenderPass2             = PFN_vkCmdEndRenderPass2( vkGetInstanceProcAddr( instance, "vkCmdEndRenderPass2" ) );
      vkResetQueryPool                = PFN_vkResetQueryPool( vkGetInstanceProcAddr( instance, "vkResetQueryPool" ) );
      vkGetSemaphoreCounterValue      = PFN_vkGetSemaphoreCounterValue( vkGetInstanceProcAddr( instance, "vkGetSemaphoreCounterValue" ) );
      vkWaitSemaphores                = PFN_vkWaitSemaphores( vkGetInstanceProcAddr( instance, "vkWaitSemaphores" ) );
      vkSignalSemaphore               = PFN_vkSignalSemaphore( vkGetInstanceProcAddr( instance, "vkSignalSemaphore" ) );
      vkGetBufferDeviceAddress        = PFN_vkGetBufferDeviceAddress( vkGetInstanceProcAddr( instance, "vkGetBufferDeviceAddress" ) );
      vkGetBufferOpaqueCaptureAddress = PFN_vkGetBufferOpaqueCaptureAddress( vkGetInstanceProcAddr( instance, "vkGetBufferOpaqueCaptureAddress" ) );
      vkGetDeviceMemoryOpaqueCaptureAddress =
        PFN_vkGetDeviceMemoryOpaqueCaptureAddress( vkGetInstanceProcAddr( instance, "vkGetDeviceMemoryOpaqueCaptureAddress" ) );

      //=== VK_VERSION_1_3 ===
      vkGetPhysicalDeviceToolProperties   = PFN_vkGetPhysicalDeviceToolProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceToolProperties" ) );
      vkCreatePrivateDataSlot             = PFN_vkCreatePrivateDataSlot( vkGetInstanceProcAddr( instance, "vkCreatePrivateDataSlot" ) );
      vkDestroyPrivateDataSlot            = PFN_vkDestroyPrivateDataSlot( vkGetInstanceProcAddr( instance, "vkDestroyPrivateDataSlot" ) );
      vkSetPrivateData                    = PFN_vkSetPrivateData( vkGetInstanceProcAddr( instance, "vkSetPrivateData" ) );
      vkGetPrivateData                    = PFN_vkGetPrivateData( vkGetInstanceProcAddr( instance, "vkGetPrivateData" ) );
      vkCmdSetEvent2                      = PFN_vkCmdSetEvent2( vkGetInstanceProcAddr( instance, "vkCmdSetEvent2" ) );
      vkCmdResetEvent2                    = PFN_vkCmdResetEvent2( vkGetInstanceProcAddr( instance, "vkCmdResetEvent2" ) );
      vkCmdWaitEvents2                    = PFN_vkCmdWaitEvents2( vkGetInstanceProcAddr( instance, "vkCmdWaitEvents2" ) );
      vkCmdPipelineBarrier2               = PFN_vkCmdPipelineBarrier2( vkGetInstanceProcAddr( instance, "vkCmdPipelineBarrier2" ) );
      vkCmdWriteTimestamp2                = PFN_vkCmdWriteTimestamp2( vkGetInstanceProcAddr( instance, "vkCmdWriteTimestamp2" ) );
      vkQueueSubmit2                      = PFN_vkQueueSubmit2( vkGetInstanceProcAddr( instance, "vkQueueSubmit2" ) );
      vkCmdCopyBuffer2                    = PFN_vkCmdCopyBuffer2( vkGetInstanceProcAddr( instance, "vkCmdCopyBuffer2" ) );
      vkCmdCopyImage2                     = PFN_vkCmdCopyImage2( vkGetInstanceProcAddr( instance, "vkCmdCopyImage2" ) );
      vkCmdCopyBufferToImage2             = PFN_vkCmdCopyBufferToImage2( vkGetInstanceProcAddr( instance, "vkCmdCopyBufferToImage2" ) );
      vkCmdCopyImageToBuffer2             = PFN_vkCmdCopyImageToBuffer2( vkGetInstanceProcAddr( instance, "vkCmdCopyImageToBuffer2" ) );
      vkCmdBlitImage2                     = PFN_vkCmdBlitImage2( vkGetInstanceProcAddr( instance, "vkCmdBlitImage2" ) );
      vkCmdResolveImage2                  = PFN_vkCmdResolveImage2( vkGetInstanceProcAddr( instance, "vkCmdResolveImage2" ) );
      vkCmdBeginRendering                 = PFN_vkCmdBeginRendering( vkGetInstanceProcAddr( instance, "vkCmdBeginRendering" ) );
      vkCmdEndRendering                   = PFN_vkCmdEndRendering( vkGetInstanceProcAddr( instance, "vkCmdEndRendering" ) );
      vkCmdSetCullMode                    = PFN_vkCmdSetCullMode( vkGetInstanceProcAddr( instance, "vkCmdSetCullMode" ) );
      vkCmdSetFrontFace                   = PFN_vkCmdSetFrontFace( vkGetInstanceProcAddr( instance, "vkCmdSetFrontFace" ) );
      vkCmdSetPrimitiveTopology           = PFN_vkCmdSetPrimitiveTopology( vkGetInstanceProcAddr( instance, "vkCmdSetPrimitiveTopology" ) );
      vkCmdSetViewportWithCount           = PFN_vkCmdSetViewportWithCount( vkGetInstanceProcAddr( instance, "vkCmdSetViewportWithCount" ) );
      vkCmdSetScissorWithCount            = PFN_vkCmdSetScissorWithCount( vkGetInstanceProcAddr( instance, "vkCmdSetScissorWithCount" ) );
      vkCmdBindVertexBuffers2             = PFN_vkCmdBindVertexBuffers2( vkGetInstanceProcAddr( instance, "vkCmdBindVertexBuffers2" ) );
      vkCmdSetDepthTestEnable             = PFN_vkCmdSetDepthTestEnable( vkGetInstanceProcAddr( instance, "vkCmdSetDepthTestEnable" ) );
      vkCmdSetDepthWriteEnable            = PFN_vkCmdSetDepthWriteEnable( vkGetInstanceProcAddr( instance, "vkCmdSetDepthWriteEnable" ) );
      vkCmdSetDepthCompareOp              = PFN_vkCmdSetDepthCompareOp( vkGetInstanceProcAddr( instance, "vkCmdSetDepthCompareOp" ) );
      vkCmdSetDepthBoundsTestEnable       = PFN_vkCmdSetDepthBoundsTestEnable( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBoundsTestEnable" ) );
      vkCmdSetStencilTestEnable           = PFN_vkCmdSetStencilTestEnable( vkGetInstanceProcAddr( instance, "vkCmdSetStencilTestEnable" ) );
      vkCmdSetStencilOp                   = PFN_vkCmdSetStencilOp( vkGetInstanceProcAddr( instance, "vkCmdSetStencilOp" ) );
      vkCmdSetRasterizerDiscardEnable     = PFN_vkCmdSetRasterizerDiscardEnable( vkGetInstanceProcAddr( instance, "vkCmdSetRasterizerDiscardEnable" ) );
      vkCmdSetDepthBiasEnable             = PFN_vkCmdSetDepthBiasEnable( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBiasEnable" ) );
      vkCmdSetPrimitiveRestartEnable      = PFN_vkCmdSetPrimitiveRestartEnable( vkGetInstanceProcAddr( instance, "vkCmdSetPrimitiveRestartEnable" ) );
      vkGetDeviceBufferMemoryRequirements = PFN_vkGetDeviceBufferMemoryRequirements( vkGetInstanceProcAddr( instance, "vkGetDeviceBufferMemoryRequirements" ) );
      vkGetDeviceImageMemoryRequirements  = PFN_vkGetDeviceImageMemoryRequirements( vkGetInstanceProcAddr( instance, "vkGetDeviceImageMemoryRequirements" ) );
      vkGetDeviceImageSparseMemoryRequirements =
        PFN_vkGetDeviceImageSparseMemoryRequirements( vkGetInstanceProcAddr( instance, "vkGetDeviceImageSparseMemoryRequirements" ) );

      //=== VK_KHR_surface ===
      vkDestroySurfaceKHR = PFN_vkDestroySurfaceKHR( vkGetInstanceProcAddr( instance, "vkDestroySurfaceKHR" ) );
      vkGetPhysicalDeviceSurfaceSupportKHR =
        PFN_vkGetPhysicalDeviceSurfaceSupportKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceSupportKHR" ) );
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
        PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR" ) );
      vkGetPhysicalDeviceSurfaceFormatsKHR =
        PFN_vkGetPhysicalDeviceSurfaceFormatsKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceFormatsKHR" ) );
      vkGetPhysicalDeviceSurfacePresentModesKHR =
        PFN_vkGetPhysicalDeviceSurfacePresentModesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfacePresentModesKHR" ) );

      //=== VK_KHR_swapchain ===
      vkCreateSwapchainKHR    = PFN_vkCreateSwapchainKHR( vkGetInstanceProcAddr( instance, "vkCreateSwapchainKHR" ) );
      vkDestroySwapchainKHR   = PFN_vkDestroySwapchainKHR( vkGetInstanceProcAddr( instance, "vkDestroySwapchainKHR" ) );
      vkGetSwapchainImagesKHR = PFN_vkGetSwapchainImagesKHR( vkGetInstanceProcAddr( instance, "vkGetSwapchainImagesKHR" ) );
      vkAcquireNextImageKHR   = PFN_vkAcquireNextImageKHR( vkGetInstanceProcAddr( instance, "vkAcquireNextImageKHR" ) );
      vkQueuePresentKHR       = PFN_vkQueuePresentKHR( vkGetInstanceProcAddr( instance, "vkQueuePresentKHR" ) );
      vkGetDeviceGroupPresentCapabilitiesKHR =
        PFN_vkGetDeviceGroupPresentCapabilitiesKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceGroupPresentCapabilitiesKHR" ) );
      vkGetDeviceGroupSurfacePresentModesKHR =
        PFN_vkGetDeviceGroupSurfacePresentModesKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceGroupSurfacePresentModesKHR" ) );
      vkGetPhysicalDevicePresentRectanglesKHR =
        PFN_vkGetPhysicalDevicePresentRectanglesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDevicePresentRectanglesKHR" ) );
      vkAcquireNextImage2KHR = PFN_vkAcquireNextImage2KHR( vkGetInstanceProcAddr( instance, "vkAcquireNextImage2KHR" ) );

      //=== VK_KHR_display ===
      vkGetPhysicalDeviceDisplayPropertiesKHR =
        PFN_vkGetPhysicalDeviceDisplayPropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPropertiesKHR" ) );
      vkGetPhysicalDeviceDisplayPlanePropertiesKHR =
        PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR" ) );
      vkGetDisplayPlaneSupportedDisplaysKHR =
        PFN_vkGetDisplayPlaneSupportedDisplaysKHR( vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneSupportedDisplaysKHR" ) );
      vkGetDisplayModePropertiesKHR    = PFN_vkGetDisplayModePropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetDisplayModePropertiesKHR" ) );
      vkCreateDisplayModeKHR           = PFN_vkCreateDisplayModeKHR( vkGetInstanceProcAddr( instance, "vkCreateDisplayModeKHR" ) );
      vkGetDisplayPlaneCapabilitiesKHR = PFN_vkGetDisplayPlaneCapabilitiesKHR( vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneCapabilitiesKHR" ) );
      vkCreateDisplayPlaneSurfaceKHR   = PFN_vkCreateDisplayPlaneSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateDisplayPlaneSurfaceKHR" ) );

      //=== VK_KHR_display_swapchain ===
      vkCreateSharedSwapchainsKHR = PFN_vkCreateSharedSwapchainsKHR( vkGetInstanceProcAddr( instance, "vkCreateSharedSwapchainsKHR" ) );

#if defined( VK_USE_PLATFORM_XLIB_KHR )
      //=== VK_KHR_xlib_surface ===
      vkCreateXlibSurfaceKHR = PFN_vkCreateXlibSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateXlibSurfaceKHR" ) );
      vkGetPhysicalDeviceXlibPresentationSupportKHR =
        PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceXlibPresentationSupportKHR" ) );
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
      //=== VK_KHR_xcb_surface ===
      vkCreateXcbSurfaceKHR = PFN_vkCreateXcbSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateXcbSurfaceKHR" ) );
      vkGetPhysicalDeviceXcbPresentationSupportKHR =
        PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceXcbPresentationSupportKHR" ) );
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      //=== VK_KHR_wayland_surface ===
      vkCreateWaylandSurfaceKHR = PFN_vkCreateWaylandSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateWaylandSurfaceKHR" ) );
      vkGetPhysicalDeviceWaylandPresentationSupportKHR =
        PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceWaylandPresentationSupportKHR" ) );
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      //=== VK_KHR_android_surface ===
      vkCreateAndroidSurfaceKHR = PFN_vkCreateAndroidSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateAndroidSurfaceKHR" ) );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_win32_surface ===
      vkCreateWin32SurfaceKHR = PFN_vkCreateWin32SurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateWin32SurfaceKHR" ) );
      vkGetPhysicalDeviceWin32PresentationSupportKHR =
        PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceWin32PresentationSupportKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_EXT_debug_report ===
      vkCreateDebugReportCallbackEXT  = PFN_vkCreateDebugReportCallbackEXT( vkGetInstanceProcAddr( instance, "vkCreateDebugReportCallbackEXT" ) );
      vkDestroyDebugReportCallbackEXT = PFN_vkDestroyDebugReportCallbackEXT( vkGetInstanceProcAddr( instance, "vkDestroyDebugReportCallbackEXT" ) );
      vkDebugReportMessageEXT         = PFN_vkDebugReportMessageEXT( vkGetInstanceProcAddr( instance, "vkDebugReportMessageEXT" ) );

      //=== VK_EXT_debug_marker ===
      vkDebugMarkerSetObjectTagEXT  = PFN_vkDebugMarkerSetObjectTagEXT( vkGetInstanceProcAddr( instance, "vkDebugMarkerSetObjectTagEXT" ) );
      vkDebugMarkerSetObjectNameEXT = PFN_vkDebugMarkerSetObjectNameEXT( vkGetInstanceProcAddr( instance, "vkDebugMarkerSetObjectNameEXT" ) );
      vkCmdDebugMarkerBeginEXT      = PFN_vkCmdDebugMarkerBeginEXT( vkGetInstanceProcAddr( instance, "vkCmdDebugMarkerBeginEXT" ) );
      vkCmdDebugMarkerEndEXT        = PFN_vkCmdDebugMarkerEndEXT( vkGetInstanceProcAddr( instance, "vkCmdDebugMarkerEndEXT" ) );
      vkCmdDebugMarkerInsertEXT     = PFN_vkCmdDebugMarkerInsertEXT( vkGetInstanceProcAddr( instance, "vkCmdDebugMarkerInsertEXT" ) );

      //=== VK_KHR_video_queue ===
      vkGetPhysicalDeviceVideoCapabilitiesKHR =
        PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR" ) );
      vkGetPhysicalDeviceVideoFormatPropertiesKHR =
        PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceVideoFormatPropertiesKHR" ) );
      vkCreateVideoSessionKHR  = PFN_vkCreateVideoSessionKHR( vkGetInstanceProcAddr( instance, "vkCreateVideoSessionKHR" ) );
      vkDestroyVideoSessionKHR = PFN_vkDestroyVideoSessionKHR( vkGetInstanceProcAddr( instance, "vkDestroyVideoSessionKHR" ) );
      vkGetVideoSessionMemoryRequirementsKHR =
        PFN_vkGetVideoSessionMemoryRequirementsKHR( vkGetInstanceProcAddr( instance, "vkGetVideoSessionMemoryRequirementsKHR" ) );
      vkBindVideoSessionMemoryKHR        = PFN_vkBindVideoSessionMemoryKHR( vkGetInstanceProcAddr( instance, "vkBindVideoSessionMemoryKHR" ) );
      vkCreateVideoSessionParametersKHR  = PFN_vkCreateVideoSessionParametersKHR( vkGetInstanceProcAddr( instance, "vkCreateVideoSessionParametersKHR" ) );
      vkUpdateVideoSessionParametersKHR  = PFN_vkUpdateVideoSessionParametersKHR( vkGetInstanceProcAddr( instance, "vkUpdateVideoSessionParametersKHR" ) );
      vkDestroyVideoSessionParametersKHR = PFN_vkDestroyVideoSessionParametersKHR( vkGetInstanceProcAddr( instance, "vkDestroyVideoSessionParametersKHR" ) );
      vkCmdBeginVideoCodingKHR           = PFN_vkCmdBeginVideoCodingKHR( vkGetInstanceProcAddr( instance, "vkCmdBeginVideoCodingKHR" ) );
      vkCmdEndVideoCodingKHR             = PFN_vkCmdEndVideoCodingKHR( vkGetInstanceProcAddr( instance, "vkCmdEndVideoCodingKHR" ) );
      vkCmdControlVideoCodingKHR         = PFN_vkCmdControlVideoCodingKHR( vkGetInstanceProcAddr( instance, "vkCmdControlVideoCodingKHR" ) );

      //=== VK_KHR_video_decode_queue ===
      vkCmdDecodeVideoKHR = PFN_vkCmdDecodeVideoKHR( vkGetInstanceProcAddr( instance, "vkCmdDecodeVideoKHR" ) );

      //=== VK_EXT_transform_feedback ===
      vkCmdBindTransformFeedbackBuffersEXT =
        PFN_vkCmdBindTransformFeedbackBuffersEXT( vkGetInstanceProcAddr( instance, "vkCmdBindTransformFeedbackBuffersEXT" ) );
      vkCmdBeginTransformFeedbackEXT = PFN_vkCmdBeginTransformFeedbackEXT( vkGetInstanceProcAddr( instance, "vkCmdBeginTransformFeedbackEXT" ) );
      vkCmdEndTransformFeedbackEXT   = PFN_vkCmdEndTransformFeedbackEXT( vkGetInstanceProcAddr( instance, "vkCmdEndTransformFeedbackEXT" ) );
      vkCmdBeginQueryIndexedEXT      = PFN_vkCmdBeginQueryIndexedEXT( vkGetInstanceProcAddr( instance, "vkCmdBeginQueryIndexedEXT" ) );
      vkCmdEndQueryIndexedEXT        = PFN_vkCmdEndQueryIndexedEXT( vkGetInstanceProcAddr( instance, "vkCmdEndQueryIndexedEXT" ) );
      vkCmdDrawIndirectByteCountEXT  = PFN_vkCmdDrawIndirectByteCountEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirectByteCountEXT" ) );

      //=== VK_NVX_binary_import ===
      vkCreateCuModuleNVX    = PFN_vkCreateCuModuleNVX( vkGetInstanceProcAddr( instance, "vkCreateCuModuleNVX" ) );
      vkCreateCuFunctionNVX  = PFN_vkCreateCuFunctionNVX( vkGetInstanceProcAddr( instance, "vkCreateCuFunctionNVX" ) );
      vkDestroyCuModuleNVX   = PFN_vkDestroyCuModuleNVX( vkGetInstanceProcAddr( instance, "vkDestroyCuModuleNVX" ) );
      vkDestroyCuFunctionNVX = PFN_vkDestroyCuFunctionNVX( vkGetInstanceProcAddr( instance, "vkDestroyCuFunctionNVX" ) );
      vkCmdCuLaunchKernelNVX = PFN_vkCmdCuLaunchKernelNVX( vkGetInstanceProcAddr( instance, "vkCmdCuLaunchKernelNVX" ) );

      //=== VK_NVX_image_view_handle ===
      vkGetImageViewHandleNVX  = PFN_vkGetImageViewHandleNVX( vkGetInstanceProcAddr( instance, "vkGetImageViewHandleNVX" ) );
      vkGetImageViewAddressNVX = PFN_vkGetImageViewAddressNVX( vkGetInstanceProcAddr( instance, "vkGetImageViewAddressNVX" ) );

      //=== VK_AMD_draw_indirect_count ===
      vkCmdDrawIndirectCountAMD = PFN_vkCmdDrawIndirectCountAMD( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirectCountAMD" ) );
      if ( !vkCmdDrawIndirectCount )
        vkCmdDrawIndirectCount = vkCmdDrawIndirectCountAMD;
      vkCmdDrawIndexedIndirectCountAMD = PFN_vkCmdDrawIndexedIndirectCountAMD( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexedIndirectCountAMD" ) );
      if ( !vkCmdDrawIndexedIndirectCount )
        vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountAMD;

      //=== VK_AMD_shader_info ===
      vkGetShaderInfoAMD = PFN_vkGetShaderInfoAMD( vkGetInstanceProcAddr( instance, "vkGetShaderInfoAMD" ) );

      //=== VK_KHR_dynamic_rendering ===
      vkCmdBeginRenderingKHR = PFN_vkCmdBeginRenderingKHR( vkGetInstanceProcAddr( instance, "vkCmdBeginRenderingKHR" ) );
      if ( !vkCmdBeginRendering )
        vkCmdBeginRendering = vkCmdBeginRenderingKHR;
      vkCmdEndRenderingKHR = PFN_vkCmdEndRenderingKHR( vkGetInstanceProcAddr( instance, "vkCmdEndRenderingKHR" ) );
      if ( !vkCmdEndRendering )
        vkCmdEndRendering = vkCmdEndRenderingKHR;

#if defined( VK_USE_PLATFORM_GGP )
      //=== VK_GGP_stream_descriptor_surface ===
      vkCreateStreamDescriptorSurfaceGGP = PFN_vkCreateStreamDescriptorSurfaceGGP( vkGetInstanceProcAddr( instance, "vkCreateStreamDescriptorSurfaceGGP" ) );
#endif /*VK_USE_PLATFORM_GGP*/

      //=== VK_NV_external_memory_capabilities ===
      vkGetPhysicalDeviceExternalImageFormatPropertiesNV =
        PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalImageFormatPropertiesNV" ) );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_NV_external_memory_win32 ===
      vkGetMemoryWin32HandleNV = PFN_vkGetMemoryWin32HandleNV( vkGetInstanceProcAddr( instance, "vkGetMemoryWin32HandleNV" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_get_physical_device_properties2 ===
      vkGetPhysicalDeviceFeatures2KHR = PFN_vkGetPhysicalDeviceFeatures2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures2KHR" ) );
      if ( !vkGetPhysicalDeviceFeatures2 )
        vkGetPhysicalDeviceFeatures2 = vkGetPhysicalDeviceFeatures2KHR;
      vkGetPhysicalDeviceProperties2KHR = PFN_vkGetPhysicalDeviceProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceProperties2 )
        vkGetPhysicalDeviceProperties2 = vkGetPhysicalDeviceProperties2KHR;
      vkGetPhysicalDeviceFormatProperties2KHR =
        PFN_vkGetPhysicalDeviceFormatProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceFormatProperties2 )
        vkGetPhysicalDeviceFormatProperties2 = vkGetPhysicalDeviceFormatProperties2KHR;
      vkGetPhysicalDeviceImageFormatProperties2KHR =
        PFN_vkGetPhysicalDeviceImageFormatProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceImageFormatProperties2 )
        vkGetPhysicalDeviceImageFormatProperties2 = vkGetPhysicalDeviceImageFormatProperties2KHR;
      vkGetPhysicalDeviceQueueFamilyProperties2KHR =
        PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceQueueFamilyProperties2 )
        vkGetPhysicalDeviceQueueFamilyProperties2 = vkGetPhysicalDeviceQueueFamilyProperties2KHR;
      vkGetPhysicalDeviceMemoryProperties2KHR =
        PFN_vkGetPhysicalDeviceMemoryProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceMemoryProperties2 )
        vkGetPhysicalDeviceMemoryProperties2 = vkGetPhysicalDeviceMemoryProperties2KHR;
      vkGetPhysicalDeviceSparseImageFormatProperties2KHR =
        PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR" ) );
      if ( !vkGetPhysicalDeviceSparseImageFormatProperties2 )
        vkGetPhysicalDeviceSparseImageFormatProperties2 = vkGetPhysicalDeviceSparseImageFormatProperties2KHR;

      //=== VK_KHR_device_group ===
      vkGetDeviceGroupPeerMemoryFeaturesKHR =
        PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceGroupPeerMemoryFeaturesKHR" ) );
      if ( !vkGetDeviceGroupPeerMemoryFeatures )
        vkGetDeviceGroupPeerMemoryFeatures = vkGetDeviceGroupPeerMemoryFeaturesKHR;
      vkCmdSetDeviceMaskKHR = PFN_vkCmdSetDeviceMaskKHR( vkGetInstanceProcAddr( instance, "vkCmdSetDeviceMaskKHR" ) );
      if ( !vkCmdSetDeviceMask )
        vkCmdSetDeviceMask = vkCmdSetDeviceMaskKHR;
      vkCmdDispatchBaseKHR = PFN_vkCmdDispatchBaseKHR( vkGetInstanceProcAddr( instance, "vkCmdDispatchBaseKHR" ) );
      if ( !vkCmdDispatchBase )
        vkCmdDispatchBase = vkCmdDispatchBaseKHR;

#if defined( VK_USE_PLATFORM_VI_NN )
      //=== VK_NN_vi_surface ===
      vkCreateViSurfaceNN = PFN_vkCreateViSurfaceNN( vkGetInstanceProcAddr( instance, "vkCreateViSurfaceNN" ) );
#endif /*VK_USE_PLATFORM_VI_NN*/

      //=== VK_KHR_maintenance1 ===
      vkTrimCommandPoolKHR = PFN_vkTrimCommandPoolKHR( vkGetInstanceProcAddr( instance, "vkTrimCommandPoolKHR" ) );
      if ( !vkTrimCommandPool )
        vkTrimCommandPool = vkTrimCommandPoolKHR;

      //=== VK_KHR_device_group_creation ===
      vkEnumeratePhysicalDeviceGroupsKHR = PFN_vkEnumeratePhysicalDeviceGroupsKHR( vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceGroupsKHR" ) );
      if ( !vkEnumeratePhysicalDeviceGroups )
        vkEnumeratePhysicalDeviceGroups = vkEnumeratePhysicalDeviceGroupsKHR;

      //=== VK_KHR_external_memory_capabilities ===
      vkGetPhysicalDeviceExternalBufferPropertiesKHR =
        PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalBufferPropertiesKHR" ) );
      if ( !vkGetPhysicalDeviceExternalBufferProperties )
        vkGetPhysicalDeviceExternalBufferProperties = vkGetPhysicalDeviceExternalBufferPropertiesKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_memory_win32 ===
      vkGetMemoryWin32HandleKHR           = PFN_vkGetMemoryWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkGetMemoryWin32HandleKHR" ) );
      vkGetMemoryWin32HandlePropertiesKHR = PFN_vkGetMemoryWin32HandlePropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetMemoryWin32HandlePropertiesKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_memory_fd ===
      vkGetMemoryFdKHR           = PFN_vkGetMemoryFdKHR( vkGetInstanceProcAddr( instance, "vkGetMemoryFdKHR" ) );
      vkGetMemoryFdPropertiesKHR = PFN_vkGetMemoryFdPropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetMemoryFdPropertiesKHR" ) );

      //=== VK_KHR_external_semaphore_capabilities ===
      vkGetPhysicalDeviceExternalSemaphorePropertiesKHR =
        PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR" ) );
      if ( !vkGetPhysicalDeviceExternalSemaphoreProperties )
        vkGetPhysicalDeviceExternalSemaphoreProperties = vkGetPhysicalDeviceExternalSemaphorePropertiesKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_semaphore_win32 ===
      vkImportSemaphoreWin32HandleKHR = PFN_vkImportSemaphoreWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkImportSemaphoreWin32HandleKHR" ) );
      vkGetSemaphoreWin32HandleKHR    = PFN_vkGetSemaphoreWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkGetSemaphoreWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_semaphore_fd ===
      vkImportSemaphoreFdKHR = PFN_vkImportSemaphoreFdKHR( vkGetInstanceProcAddr( instance, "vkImportSemaphoreFdKHR" ) );
      vkGetSemaphoreFdKHR    = PFN_vkGetSemaphoreFdKHR( vkGetInstanceProcAddr( instance, "vkGetSemaphoreFdKHR" ) );

      //=== VK_KHR_push_descriptor ===
      vkCmdPushDescriptorSetKHR = PFN_vkCmdPushDescriptorSetKHR( vkGetInstanceProcAddr( instance, "vkCmdPushDescriptorSetKHR" ) );
      vkCmdPushDescriptorSetWithTemplateKHR =
        PFN_vkCmdPushDescriptorSetWithTemplateKHR( vkGetInstanceProcAddr( instance, "vkCmdPushDescriptorSetWithTemplateKHR" ) );

      //=== VK_EXT_conditional_rendering ===
      vkCmdBeginConditionalRenderingEXT = PFN_vkCmdBeginConditionalRenderingEXT( vkGetInstanceProcAddr( instance, "vkCmdBeginConditionalRenderingEXT" ) );
      vkCmdEndConditionalRenderingEXT   = PFN_vkCmdEndConditionalRenderingEXT( vkGetInstanceProcAddr( instance, "vkCmdEndConditionalRenderingEXT" ) );

      //=== VK_KHR_descriptor_update_template ===
      vkCreateDescriptorUpdateTemplateKHR = PFN_vkCreateDescriptorUpdateTemplateKHR( vkGetInstanceProcAddr( instance, "vkCreateDescriptorUpdateTemplateKHR" ) );
      if ( !vkCreateDescriptorUpdateTemplate )
        vkCreateDescriptorUpdateTemplate = vkCreateDescriptorUpdateTemplateKHR;
      vkDestroyDescriptorUpdateTemplateKHR =
        PFN_vkDestroyDescriptorUpdateTemplateKHR( vkGetInstanceProcAddr( instance, "vkDestroyDescriptorUpdateTemplateKHR" ) );
      if ( !vkDestroyDescriptorUpdateTemplate )
        vkDestroyDescriptorUpdateTemplate = vkDestroyDescriptorUpdateTemplateKHR;
      vkUpdateDescriptorSetWithTemplateKHR =
        PFN_vkUpdateDescriptorSetWithTemplateKHR( vkGetInstanceProcAddr( instance, "vkUpdateDescriptorSetWithTemplateKHR" ) );
      if ( !vkUpdateDescriptorSetWithTemplate )
        vkUpdateDescriptorSetWithTemplate = vkUpdateDescriptorSetWithTemplateKHR;

      //=== VK_NV_clip_space_w_scaling ===
      vkCmdSetViewportWScalingNV = PFN_vkCmdSetViewportWScalingNV( vkGetInstanceProcAddr( instance, "vkCmdSetViewportWScalingNV" ) );

      //=== VK_EXT_direct_mode_display ===
      vkReleaseDisplayEXT = PFN_vkReleaseDisplayEXT( vkGetInstanceProcAddr( instance, "vkReleaseDisplayEXT" ) );

#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
      //=== VK_EXT_acquire_xlib_display ===
      vkAcquireXlibDisplayEXT    = PFN_vkAcquireXlibDisplayEXT( vkGetInstanceProcAddr( instance, "vkAcquireXlibDisplayEXT" ) );
      vkGetRandROutputDisplayEXT = PFN_vkGetRandROutputDisplayEXT( vkGetInstanceProcAddr( instance, "vkGetRandROutputDisplayEXT" ) );
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

      //=== VK_EXT_display_surface_counter ===
      vkGetPhysicalDeviceSurfaceCapabilities2EXT =
        PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilities2EXT" ) );

      //=== VK_EXT_display_control ===
      vkDisplayPowerControlEXT  = PFN_vkDisplayPowerControlEXT( vkGetInstanceProcAddr( instance, "vkDisplayPowerControlEXT" ) );
      vkRegisterDeviceEventEXT  = PFN_vkRegisterDeviceEventEXT( vkGetInstanceProcAddr( instance, "vkRegisterDeviceEventEXT" ) );
      vkRegisterDisplayEventEXT = PFN_vkRegisterDisplayEventEXT( vkGetInstanceProcAddr( instance, "vkRegisterDisplayEventEXT" ) );
      vkGetSwapchainCounterEXT  = PFN_vkGetSwapchainCounterEXT( vkGetInstanceProcAddr( instance, "vkGetSwapchainCounterEXT" ) );

      //=== VK_GOOGLE_display_timing ===
      vkGetRefreshCycleDurationGOOGLE   = PFN_vkGetRefreshCycleDurationGOOGLE( vkGetInstanceProcAddr( instance, "vkGetRefreshCycleDurationGOOGLE" ) );
      vkGetPastPresentationTimingGOOGLE = PFN_vkGetPastPresentationTimingGOOGLE( vkGetInstanceProcAddr( instance, "vkGetPastPresentationTimingGOOGLE" ) );

      //=== VK_EXT_discard_rectangles ===
      vkCmdSetDiscardRectangleEXT       = PFN_vkCmdSetDiscardRectangleEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDiscardRectangleEXT" ) );
      vkCmdSetDiscardRectangleEnableEXT = PFN_vkCmdSetDiscardRectangleEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDiscardRectangleEnableEXT" ) );
      vkCmdSetDiscardRectangleModeEXT   = PFN_vkCmdSetDiscardRectangleModeEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDiscardRectangleModeEXT" ) );

      //=== VK_EXT_hdr_metadata ===
      vkSetHdrMetadataEXT = PFN_vkSetHdrMetadataEXT( vkGetInstanceProcAddr( instance, "vkSetHdrMetadataEXT" ) );

      //=== VK_KHR_create_renderpass2 ===
      vkCreateRenderPass2KHR = PFN_vkCreateRenderPass2KHR( vkGetInstanceProcAddr( instance, "vkCreateRenderPass2KHR" ) );
      if ( !vkCreateRenderPass2 )
        vkCreateRenderPass2 = vkCreateRenderPass2KHR;
      vkCmdBeginRenderPass2KHR = PFN_vkCmdBeginRenderPass2KHR( vkGetInstanceProcAddr( instance, "vkCmdBeginRenderPass2KHR" ) );
      if ( !vkCmdBeginRenderPass2 )
        vkCmdBeginRenderPass2 = vkCmdBeginRenderPass2KHR;
      vkCmdNextSubpass2KHR = PFN_vkCmdNextSubpass2KHR( vkGetInstanceProcAddr( instance, "vkCmdNextSubpass2KHR" ) );
      if ( !vkCmdNextSubpass2 )
        vkCmdNextSubpass2 = vkCmdNextSubpass2KHR;
      vkCmdEndRenderPass2KHR = PFN_vkCmdEndRenderPass2KHR( vkGetInstanceProcAddr( instance, "vkCmdEndRenderPass2KHR" ) );
      if ( !vkCmdEndRenderPass2 )
        vkCmdEndRenderPass2 = vkCmdEndRenderPass2KHR;

      //=== VK_KHR_shared_presentable_image ===
      vkGetSwapchainStatusKHR = PFN_vkGetSwapchainStatusKHR( vkGetInstanceProcAddr( instance, "vkGetSwapchainStatusKHR" ) );

      //=== VK_KHR_external_fence_capabilities ===
      vkGetPhysicalDeviceExternalFencePropertiesKHR =
        PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalFencePropertiesKHR" ) );
      if ( !vkGetPhysicalDeviceExternalFenceProperties )
        vkGetPhysicalDeviceExternalFenceProperties = vkGetPhysicalDeviceExternalFencePropertiesKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_fence_win32 ===
      vkImportFenceWin32HandleKHR = PFN_vkImportFenceWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkImportFenceWin32HandleKHR" ) );
      vkGetFenceWin32HandleKHR    = PFN_vkGetFenceWin32HandleKHR( vkGetInstanceProcAddr( instance, "vkGetFenceWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_fence_fd ===
      vkImportFenceFdKHR = PFN_vkImportFenceFdKHR( vkGetInstanceProcAddr( instance, "vkImportFenceFdKHR" ) );
      vkGetFenceFdKHR    = PFN_vkGetFenceFdKHR( vkGetInstanceProcAddr( instance, "vkGetFenceFdKHR" ) );

      //=== VK_KHR_performance_query ===
      vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR = PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
        vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR" ) );
      vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR = PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR" ) );
      vkAcquireProfilingLockKHR = PFN_vkAcquireProfilingLockKHR( vkGetInstanceProcAddr( instance, "vkAcquireProfilingLockKHR" ) );
      vkReleaseProfilingLockKHR = PFN_vkReleaseProfilingLockKHR( vkGetInstanceProcAddr( instance, "vkReleaseProfilingLockKHR" ) );

      //=== VK_KHR_get_surface_capabilities2 ===
      vkGetPhysicalDeviceSurfaceCapabilities2KHR =
        PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR" ) );
      vkGetPhysicalDeviceSurfaceFormats2KHR =
        PFN_vkGetPhysicalDeviceSurfaceFormats2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceFormats2KHR" ) );

      //=== VK_KHR_get_display_properties2 ===
      vkGetPhysicalDeviceDisplayProperties2KHR =
        PFN_vkGetPhysicalDeviceDisplayProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayProperties2KHR" ) );
      vkGetPhysicalDeviceDisplayPlaneProperties2KHR =
        PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPlaneProperties2KHR" ) );
      vkGetDisplayModeProperties2KHR    = PFN_vkGetDisplayModeProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetDisplayModeProperties2KHR" ) );
      vkGetDisplayPlaneCapabilities2KHR = PFN_vkGetDisplayPlaneCapabilities2KHR( vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneCapabilities2KHR" ) );

#if defined( VK_USE_PLATFORM_IOS_MVK )
      //=== VK_MVK_ios_surface ===
      vkCreateIOSSurfaceMVK = PFN_vkCreateIOSSurfaceMVK( vkGetInstanceProcAddr( instance, "vkCreateIOSSurfaceMVK" ) );
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
      //=== VK_MVK_macos_surface ===
      vkCreateMacOSSurfaceMVK = PFN_vkCreateMacOSSurfaceMVK( vkGetInstanceProcAddr( instance, "vkCreateMacOSSurfaceMVK" ) );
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

      //=== VK_EXT_debug_utils ===
      vkSetDebugUtilsObjectNameEXT    = PFN_vkSetDebugUtilsObjectNameEXT( vkGetInstanceProcAddr( instance, "vkSetDebugUtilsObjectNameEXT" ) );
      vkSetDebugUtilsObjectTagEXT     = PFN_vkSetDebugUtilsObjectTagEXT( vkGetInstanceProcAddr( instance, "vkSetDebugUtilsObjectTagEXT" ) );
      vkQueueBeginDebugUtilsLabelEXT  = PFN_vkQueueBeginDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkQueueBeginDebugUtilsLabelEXT" ) );
      vkQueueEndDebugUtilsLabelEXT    = PFN_vkQueueEndDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkQueueEndDebugUtilsLabelEXT" ) );
      vkQueueInsertDebugUtilsLabelEXT = PFN_vkQueueInsertDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkQueueInsertDebugUtilsLabelEXT" ) );
      vkCmdBeginDebugUtilsLabelEXT    = PFN_vkCmdBeginDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkCmdBeginDebugUtilsLabelEXT" ) );
      vkCmdEndDebugUtilsLabelEXT      = PFN_vkCmdEndDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkCmdEndDebugUtilsLabelEXT" ) );
      vkCmdInsertDebugUtilsLabelEXT   = PFN_vkCmdInsertDebugUtilsLabelEXT( vkGetInstanceProcAddr( instance, "vkCmdInsertDebugUtilsLabelEXT" ) );
      vkCreateDebugUtilsMessengerEXT  = PFN_vkCreateDebugUtilsMessengerEXT( vkGetInstanceProcAddr( instance, "vkCreateDebugUtilsMessengerEXT" ) );
      vkDestroyDebugUtilsMessengerEXT = PFN_vkDestroyDebugUtilsMessengerEXT( vkGetInstanceProcAddr( instance, "vkDestroyDebugUtilsMessengerEXT" ) );
      vkSubmitDebugUtilsMessageEXT    = PFN_vkSubmitDebugUtilsMessageEXT( vkGetInstanceProcAddr( instance, "vkSubmitDebugUtilsMessageEXT" ) );

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      //=== VK_ANDROID_external_memory_android_hardware_buffer ===
      vkGetAndroidHardwareBufferPropertiesANDROID =
        PFN_vkGetAndroidHardwareBufferPropertiesANDROID( vkGetInstanceProcAddr( instance, "vkGetAndroidHardwareBufferPropertiesANDROID" ) );
      vkGetMemoryAndroidHardwareBufferANDROID =
        PFN_vkGetMemoryAndroidHardwareBufferANDROID( vkGetInstanceProcAddr( instance, "vkGetMemoryAndroidHardwareBufferANDROID" ) );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_AMDX_shader_enqueue ===
      vkCreateExecutionGraphPipelinesAMDX = PFN_vkCreateExecutionGraphPipelinesAMDX( vkGetInstanceProcAddr( instance, "vkCreateExecutionGraphPipelinesAMDX" ) );
      vkGetExecutionGraphPipelineScratchSizeAMDX =
        PFN_vkGetExecutionGraphPipelineScratchSizeAMDX( vkGetInstanceProcAddr( instance, "vkGetExecutionGraphPipelineScratchSizeAMDX" ) );
      vkGetExecutionGraphPipelineNodeIndexAMDX =
        PFN_vkGetExecutionGraphPipelineNodeIndexAMDX( vkGetInstanceProcAddr( instance, "vkGetExecutionGraphPipelineNodeIndexAMDX" ) );
      vkCmdInitializeGraphScratchMemoryAMDX =
        PFN_vkCmdInitializeGraphScratchMemoryAMDX( vkGetInstanceProcAddr( instance, "vkCmdInitializeGraphScratchMemoryAMDX" ) );
      vkCmdDispatchGraphAMDX              = PFN_vkCmdDispatchGraphAMDX( vkGetInstanceProcAddr( instance, "vkCmdDispatchGraphAMDX" ) );
      vkCmdDispatchGraphIndirectAMDX      = PFN_vkCmdDispatchGraphIndirectAMDX( vkGetInstanceProcAddr( instance, "vkCmdDispatchGraphIndirectAMDX" ) );
      vkCmdDispatchGraphIndirectCountAMDX = PFN_vkCmdDispatchGraphIndirectCountAMDX( vkGetInstanceProcAddr( instance, "vkCmdDispatchGraphIndirectCountAMDX" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

      //=== VK_EXT_sample_locations ===
      vkCmdSetSampleLocationsEXT = PFN_vkCmdSetSampleLocationsEXT( vkGetInstanceProcAddr( instance, "vkCmdSetSampleLocationsEXT" ) );
      vkGetPhysicalDeviceMultisamplePropertiesEXT =
        PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMultisamplePropertiesEXT" ) );

      //=== VK_KHR_get_memory_requirements2 ===
      vkGetImageMemoryRequirements2KHR = PFN_vkGetImageMemoryRequirements2KHR( vkGetInstanceProcAddr( instance, "vkGetImageMemoryRequirements2KHR" ) );
      if ( !vkGetImageMemoryRequirements2 )
        vkGetImageMemoryRequirements2 = vkGetImageMemoryRequirements2KHR;
      vkGetBufferMemoryRequirements2KHR = PFN_vkGetBufferMemoryRequirements2KHR( vkGetInstanceProcAddr( instance, "vkGetBufferMemoryRequirements2KHR" ) );
      if ( !vkGetBufferMemoryRequirements2 )
        vkGetBufferMemoryRequirements2 = vkGetBufferMemoryRequirements2KHR;
      vkGetImageSparseMemoryRequirements2KHR =
        PFN_vkGetImageSparseMemoryRequirements2KHR( vkGetInstanceProcAddr( instance, "vkGetImageSparseMemoryRequirements2KHR" ) );
      if ( !vkGetImageSparseMemoryRequirements2 )
        vkGetImageSparseMemoryRequirements2 = vkGetImageSparseMemoryRequirements2KHR;

      //=== VK_KHR_acceleration_structure ===
      vkCreateAccelerationStructureKHR    = PFN_vkCreateAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkCreateAccelerationStructureKHR" ) );
      vkDestroyAccelerationStructureKHR   = PFN_vkDestroyAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkDestroyAccelerationStructureKHR" ) );
      vkCmdBuildAccelerationStructuresKHR = PFN_vkCmdBuildAccelerationStructuresKHR( vkGetInstanceProcAddr( instance, "vkCmdBuildAccelerationStructuresKHR" ) );
      vkCmdBuildAccelerationStructuresIndirectKHR =
        PFN_vkCmdBuildAccelerationStructuresIndirectKHR( vkGetInstanceProcAddr( instance, "vkCmdBuildAccelerationStructuresIndirectKHR" ) );
      vkBuildAccelerationStructuresKHR = PFN_vkBuildAccelerationStructuresKHR( vkGetInstanceProcAddr( instance, "vkBuildAccelerationStructuresKHR" ) );
      vkCopyAccelerationStructureKHR   = PFN_vkCopyAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkCopyAccelerationStructureKHR" ) );
      vkCopyAccelerationStructureToMemoryKHR =
        PFN_vkCopyAccelerationStructureToMemoryKHR( vkGetInstanceProcAddr( instance, "vkCopyAccelerationStructureToMemoryKHR" ) );
      vkCopyMemoryToAccelerationStructureKHR =
        PFN_vkCopyMemoryToAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkCopyMemoryToAccelerationStructureKHR" ) );
      vkWriteAccelerationStructuresPropertiesKHR =
        PFN_vkWriteAccelerationStructuresPropertiesKHR( vkGetInstanceProcAddr( instance, "vkWriteAccelerationStructuresPropertiesKHR" ) );
      vkCmdCopyAccelerationStructureKHR = PFN_vkCmdCopyAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkCmdCopyAccelerationStructureKHR" ) );
      vkCmdCopyAccelerationStructureToMemoryKHR =
        PFN_vkCmdCopyAccelerationStructureToMemoryKHR( vkGetInstanceProcAddr( instance, "vkCmdCopyAccelerationStructureToMemoryKHR" ) );
      vkCmdCopyMemoryToAccelerationStructureKHR =
        PFN_vkCmdCopyMemoryToAccelerationStructureKHR( vkGetInstanceProcAddr( instance, "vkCmdCopyMemoryToAccelerationStructureKHR" ) );
      vkGetAccelerationStructureDeviceAddressKHR =
        PFN_vkGetAccelerationStructureDeviceAddressKHR( vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureDeviceAddressKHR" ) );
      vkCmdWriteAccelerationStructuresPropertiesKHR =
        PFN_vkCmdWriteAccelerationStructuresPropertiesKHR( vkGetInstanceProcAddr( instance, "vkCmdWriteAccelerationStructuresPropertiesKHR" ) );
      vkGetDeviceAccelerationStructureCompatibilityKHR =
        PFN_vkGetDeviceAccelerationStructureCompatibilityKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceAccelerationStructureCompatibilityKHR" ) );
      vkGetAccelerationStructureBuildSizesKHR =
        PFN_vkGetAccelerationStructureBuildSizesKHR( vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureBuildSizesKHR" ) );

      //=== VK_KHR_ray_tracing_pipeline ===
      vkCmdTraceRaysKHR              = PFN_vkCmdTraceRaysKHR( vkGetInstanceProcAddr( instance, "vkCmdTraceRaysKHR" ) );
      vkCreateRayTracingPipelinesKHR = PFN_vkCreateRayTracingPipelinesKHR( vkGetInstanceProcAddr( instance, "vkCreateRayTracingPipelinesKHR" ) );
      vkGetRayTracingShaderGroupHandlesKHR =
        PFN_vkGetRayTracingShaderGroupHandlesKHR( vkGetInstanceProcAddr( instance, "vkGetRayTracingShaderGroupHandlesKHR" ) );
      vkGetRayTracingCaptureReplayShaderGroupHandlesKHR =
        PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR( vkGetInstanceProcAddr( instance, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR" ) );
      vkCmdTraceRaysIndirectKHR = PFN_vkCmdTraceRaysIndirectKHR( vkGetInstanceProcAddr( instance, "vkCmdTraceRaysIndirectKHR" ) );
      vkGetRayTracingShaderGroupStackSizeKHR =
        PFN_vkGetRayTracingShaderGroupStackSizeKHR( vkGetInstanceProcAddr( instance, "vkGetRayTracingShaderGroupStackSizeKHR" ) );
      vkCmdSetRayTracingPipelineStackSizeKHR =
        PFN_vkCmdSetRayTracingPipelineStackSizeKHR( vkGetInstanceProcAddr( instance, "vkCmdSetRayTracingPipelineStackSizeKHR" ) );

      //=== VK_KHR_sampler_ycbcr_conversion ===
      vkCreateSamplerYcbcrConversionKHR = PFN_vkCreateSamplerYcbcrConversionKHR( vkGetInstanceProcAddr( instance, "vkCreateSamplerYcbcrConversionKHR" ) );
      if ( !vkCreateSamplerYcbcrConversion )
        vkCreateSamplerYcbcrConversion = vkCreateSamplerYcbcrConversionKHR;
      vkDestroySamplerYcbcrConversionKHR = PFN_vkDestroySamplerYcbcrConversionKHR( vkGetInstanceProcAddr( instance, "vkDestroySamplerYcbcrConversionKHR" ) );
      if ( !vkDestroySamplerYcbcrConversion )
        vkDestroySamplerYcbcrConversion = vkDestroySamplerYcbcrConversionKHR;

      //=== VK_KHR_bind_memory2 ===
      vkBindBufferMemory2KHR = PFN_vkBindBufferMemory2KHR( vkGetInstanceProcAddr( instance, "vkBindBufferMemory2KHR" ) );
      if ( !vkBindBufferMemory2 )
        vkBindBufferMemory2 = vkBindBufferMemory2KHR;
      vkBindImageMemory2KHR = PFN_vkBindImageMemory2KHR( vkGetInstanceProcAddr( instance, "vkBindImageMemory2KHR" ) );
      if ( !vkBindImageMemory2 )
        vkBindImageMemory2 = vkBindImageMemory2KHR;

      //=== VK_EXT_image_drm_format_modifier ===
      vkGetImageDrmFormatModifierPropertiesEXT =
        PFN_vkGetImageDrmFormatModifierPropertiesEXT( vkGetInstanceProcAddr( instance, "vkGetImageDrmFormatModifierPropertiesEXT" ) );

      //=== VK_EXT_validation_cache ===
      vkCreateValidationCacheEXT  = PFN_vkCreateValidationCacheEXT( vkGetInstanceProcAddr( instance, "vkCreateValidationCacheEXT" ) );
      vkDestroyValidationCacheEXT = PFN_vkDestroyValidationCacheEXT( vkGetInstanceProcAddr( instance, "vkDestroyValidationCacheEXT" ) );
      vkMergeValidationCachesEXT  = PFN_vkMergeValidationCachesEXT( vkGetInstanceProcAddr( instance, "vkMergeValidationCachesEXT" ) );
      vkGetValidationCacheDataEXT = PFN_vkGetValidationCacheDataEXT( vkGetInstanceProcAddr( instance, "vkGetValidationCacheDataEXT" ) );

      //=== VK_NV_shading_rate_image ===
      vkCmdBindShadingRateImageNV = PFN_vkCmdBindShadingRateImageNV( vkGetInstanceProcAddr( instance, "vkCmdBindShadingRateImageNV" ) );
      vkCmdSetViewportShadingRatePaletteNV =
        PFN_vkCmdSetViewportShadingRatePaletteNV( vkGetInstanceProcAddr( instance, "vkCmdSetViewportShadingRatePaletteNV" ) );
      vkCmdSetCoarseSampleOrderNV = PFN_vkCmdSetCoarseSampleOrderNV( vkGetInstanceProcAddr( instance, "vkCmdSetCoarseSampleOrderNV" ) );

      //=== VK_NV_ray_tracing ===
      vkCreateAccelerationStructureNV  = PFN_vkCreateAccelerationStructureNV( vkGetInstanceProcAddr( instance, "vkCreateAccelerationStructureNV" ) );
      vkDestroyAccelerationStructureNV = PFN_vkDestroyAccelerationStructureNV( vkGetInstanceProcAddr( instance, "vkDestroyAccelerationStructureNV" ) );
      vkGetAccelerationStructureMemoryRequirementsNV =
        PFN_vkGetAccelerationStructureMemoryRequirementsNV( vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureMemoryRequirementsNV" ) );
      vkBindAccelerationStructureMemoryNV = PFN_vkBindAccelerationStructureMemoryNV( vkGetInstanceProcAddr( instance, "vkBindAccelerationStructureMemoryNV" ) );
      vkCmdBuildAccelerationStructureNV   = PFN_vkCmdBuildAccelerationStructureNV( vkGetInstanceProcAddr( instance, "vkCmdBuildAccelerationStructureNV" ) );
      vkCmdCopyAccelerationStructureNV    = PFN_vkCmdCopyAccelerationStructureNV( vkGetInstanceProcAddr( instance, "vkCmdCopyAccelerationStructureNV" ) );
      vkCmdTraceRaysNV                    = PFN_vkCmdTraceRaysNV( vkGetInstanceProcAddr( instance, "vkCmdTraceRaysNV" ) );
      vkCreateRayTracingPipelinesNV       = PFN_vkCreateRayTracingPipelinesNV( vkGetInstanceProcAddr( instance, "vkCreateRayTracingPipelinesNV" ) );
      vkGetRayTracingShaderGroupHandlesNV = PFN_vkGetRayTracingShaderGroupHandlesNV( vkGetInstanceProcAddr( instance, "vkGetRayTracingShaderGroupHandlesNV" ) );
      if ( !vkGetRayTracingShaderGroupHandlesKHR )
        vkGetRayTracingShaderGroupHandlesKHR = vkGetRayTracingShaderGroupHandlesNV;
      vkGetAccelerationStructureHandleNV = PFN_vkGetAccelerationStructureHandleNV( vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureHandleNV" ) );
      vkCmdWriteAccelerationStructuresPropertiesNV =
        PFN_vkCmdWriteAccelerationStructuresPropertiesNV( vkGetInstanceProcAddr( instance, "vkCmdWriteAccelerationStructuresPropertiesNV" ) );
      vkCompileDeferredNV = PFN_vkCompileDeferredNV( vkGetInstanceProcAddr( instance, "vkCompileDeferredNV" ) );

      //=== VK_KHR_maintenance3 ===
      vkGetDescriptorSetLayoutSupportKHR = PFN_vkGetDescriptorSetLayoutSupportKHR( vkGetInstanceProcAddr( instance, "vkGetDescriptorSetLayoutSupportKHR" ) );
      if ( !vkGetDescriptorSetLayoutSupport )
        vkGetDescriptorSetLayoutSupport = vkGetDescriptorSetLayoutSupportKHR;

      //=== VK_KHR_draw_indirect_count ===
      vkCmdDrawIndirectCountKHR = PFN_vkCmdDrawIndirectCountKHR( vkGetInstanceProcAddr( instance, "vkCmdDrawIndirectCountKHR" ) );
      if ( !vkCmdDrawIndirectCount )
        vkCmdDrawIndirectCount = vkCmdDrawIndirectCountKHR;
      vkCmdDrawIndexedIndirectCountKHR = PFN_vkCmdDrawIndexedIndirectCountKHR( vkGetInstanceProcAddr( instance, "vkCmdDrawIndexedIndirectCountKHR" ) );
      if ( !vkCmdDrawIndexedIndirectCount )
        vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountKHR;

      //=== VK_EXT_external_memory_host ===
      vkGetMemoryHostPointerPropertiesEXT = PFN_vkGetMemoryHostPointerPropertiesEXT( vkGetInstanceProcAddr( instance, "vkGetMemoryHostPointerPropertiesEXT" ) );

      //=== VK_AMD_buffer_marker ===
      vkCmdWriteBufferMarkerAMD = PFN_vkCmdWriteBufferMarkerAMD( vkGetInstanceProcAddr( instance, "vkCmdWriteBufferMarkerAMD" ) );

      //=== VK_EXT_calibrated_timestamps ===
      vkGetPhysicalDeviceCalibrateableTimeDomainsEXT =
        PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT" ) );
      if ( !vkGetPhysicalDeviceCalibrateableTimeDomainsKHR )
        vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
      vkGetCalibratedTimestampsEXT = PFN_vkGetCalibratedTimestampsEXT( vkGetInstanceProcAddr( instance, "vkGetCalibratedTimestampsEXT" ) );
      if ( !vkGetCalibratedTimestampsKHR )
        vkGetCalibratedTimestampsKHR = vkGetCalibratedTimestampsEXT;

      //=== VK_NV_mesh_shader ===
      vkCmdDrawMeshTasksNV              = PFN_vkCmdDrawMeshTasksNV( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksNV" ) );
      vkCmdDrawMeshTasksIndirectNV      = PFN_vkCmdDrawMeshTasksIndirectNV( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksIndirectNV" ) );
      vkCmdDrawMeshTasksIndirectCountNV = PFN_vkCmdDrawMeshTasksIndirectCountNV( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksIndirectCountNV" ) );

      //=== VK_NV_scissor_exclusive ===
      vkCmdSetExclusiveScissorEnableNV = PFN_vkCmdSetExclusiveScissorEnableNV( vkGetInstanceProcAddr( instance, "vkCmdSetExclusiveScissorEnableNV" ) );
      vkCmdSetExclusiveScissorNV       = PFN_vkCmdSetExclusiveScissorNV( vkGetInstanceProcAddr( instance, "vkCmdSetExclusiveScissorNV" ) );

      //=== VK_NV_device_diagnostic_checkpoints ===
      vkCmdSetCheckpointNV       = PFN_vkCmdSetCheckpointNV( vkGetInstanceProcAddr( instance, "vkCmdSetCheckpointNV" ) );
      vkGetQueueCheckpointDataNV = PFN_vkGetQueueCheckpointDataNV( vkGetInstanceProcAddr( instance, "vkGetQueueCheckpointDataNV" ) );

      //=== VK_KHR_timeline_semaphore ===
      vkGetSemaphoreCounterValueKHR = PFN_vkGetSemaphoreCounterValueKHR( vkGetInstanceProcAddr( instance, "vkGetSemaphoreCounterValueKHR" ) );
      if ( !vkGetSemaphoreCounterValue )
        vkGetSemaphoreCounterValue = vkGetSemaphoreCounterValueKHR;
      vkWaitSemaphoresKHR = PFN_vkWaitSemaphoresKHR( vkGetInstanceProcAddr( instance, "vkWaitSemaphoresKHR" ) );
      if ( !vkWaitSemaphores )
        vkWaitSemaphores = vkWaitSemaphoresKHR;
      vkSignalSemaphoreKHR = PFN_vkSignalSemaphoreKHR( vkGetInstanceProcAddr( instance, "vkSignalSemaphoreKHR" ) );
      if ( !vkSignalSemaphore )
        vkSignalSemaphore = vkSignalSemaphoreKHR;

      //=== VK_INTEL_performance_query ===
      vkInitializePerformanceApiINTEL   = PFN_vkInitializePerformanceApiINTEL( vkGetInstanceProcAddr( instance, "vkInitializePerformanceApiINTEL" ) );
      vkUninitializePerformanceApiINTEL = PFN_vkUninitializePerformanceApiINTEL( vkGetInstanceProcAddr( instance, "vkUninitializePerformanceApiINTEL" ) );
      vkCmdSetPerformanceMarkerINTEL    = PFN_vkCmdSetPerformanceMarkerINTEL( vkGetInstanceProcAddr( instance, "vkCmdSetPerformanceMarkerINTEL" ) );
      vkCmdSetPerformanceStreamMarkerINTEL =
        PFN_vkCmdSetPerformanceStreamMarkerINTEL( vkGetInstanceProcAddr( instance, "vkCmdSetPerformanceStreamMarkerINTEL" ) );
      vkCmdSetPerformanceOverrideINTEL = PFN_vkCmdSetPerformanceOverrideINTEL( vkGetInstanceProcAddr( instance, "vkCmdSetPerformanceOverrideINTEL" ) );
      vkAcquirePerformanceConfigurationINTEL =
        PFN_vkAcquirePerformanceConfigurationINTEL( vkGetInstanceProcAddr( instance, "vkAcquirePerformanceConfigurationINTEL" ) );
      vkReleasePerformanceConfigurationINTEL =
        PFN_vkReleasePerformanceConfigurationINTEL( vkGetInstanceProcAddr( instance, "vkReleasePerformanceConfigurationINTEL" ) );
      vkQueueSetPerformanceConfigurationINTEL =
        PFN_vkQueueSetPerformanceConfigurationINTEL( vkGetInstanceProcAddr( instance, "vkQueueSetPerformanceConfigurationINTEL" ) );
      vkGetPerformanceParameterINTEL = PFN_vkGetPerformanceParameterINTEL( vkGetInstanceProcAddr( instance, "vkGetPerformanceParameterINTEL" ) );

      //=== VK_AMD_display_native_hdr ===
      vkSetLocalDimmingAMD = PFN_vkSetLocalDimmingAMD( vkGetInstanceProcAddr( instance, "vkSetLocalDimmingAMD" ) );

#if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_imagepipe_surface ===
      vkCreateImagePipeSurfaceFUCHSIA = PFN_vkCreateImagePipeSurfaceFUCHSIA( vkGetInstanceProcAddr( instance, "vkCreateImagePipeSurfaceFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
      //=== VK_EXT_metal_surface ===
      vkCreateMetalSurfaceEXT = PFN_vkCreateMetalSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateMetalSurfaceEXT" ) );
#endif /*VK_USE_PLATFORM_METAL_EXT*/

      //=== VK_KHR_fragment_shading_rate ===
      vkGetPhysicalDeviceFragmentShadingRatesKHR =
        PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFragmentShadingRatesKHR" ) );
      vkCmdSetFragmentShadingRateKHR = PFN_vkCmdSetFragmentShadingRateKHR( vkGetInstanceProcAddr( instance, "vkCmdSetFragmentShadingRateKHR" ) );

      //=== VK_EXT_buffer_device_address ===
      vkGetBufferDeviceAddressEXT = PFN_vkGetBufferDeviceAddressEXT( vkGetInstanceProcAddr( instance, "vkGetBufferDeviceAddressEXT" ) );
      if ( !vkGetBufferDeviceAddress )
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressEXT;

      //=== VK_EXT_tooling_info ===
      vkGetPhysicalDeviceToolPropertiesEXT =
        PFN_vkGetPhysicalDeviceToolPropertiesEXT( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceToolPropertiesEXT" ) );
      if ( !vkGetPhysicalDeviceToolProperties )
        vkGetPhysicalDeviceToolProperties = vkGetPhysicalDeviceToolPropertiesEXT;

      //=== VK_KHR_present_wait ===
      vkWaitForPresentKHR = PFN_vkWaitForPresentKHR( vkGetInstanceProcAddr( instance, "vkWaitForPresentKHR" ) );

      //=== VK_NV_cooperative_matrix ===
      vkGetPhysicalDeviceCooperativeMatrixPropertiesNV =
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV" ) );

      //=== VK_NV_coverage_reduction_mode ===
      vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV = PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV" ) );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_EXT_full_screen_exclusive ===
      vkGetPhysicalDeviceSurfacePresentModes2EXT =
        PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfacePresentModes2EXT" ) );
      vkAcquireFullScreenExclusiveModeEXT = PFN_vkAcquireFullScreenExclusiveModeEXT( vkGetInstanceProcAddr( instance, "vkAcquireFullScreenExclusiveModeEXT" ) );
      vkReleaseFullScreenExclusiveModeEXT = PFN_vkReleaseFullScreenExclusiveModeEXT( vkGetInstanceProcAddr( instance, "vkReleaseFullScreenExclusiveModeEXT" ) );
      vkGetDeviceGroupSurfacePresentModes2EXT =
        PFN_vkGetDeviceGroupSurfacePresentModes2EXT( vkGetInstanceProcAddr( instance, "vkGetDeviceGroupSurfacePresentModes2EXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_EXT_headless_surface ===
      vkCreateHeadlessSurfaceEXT = PFN_vkCreateHeadlessSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateHeadlessSurfaceEXT" ) );

      //=== VK_KHR_buffer_device_address ===
      vkGetBufferDeviceAddressKHR = PFN_vkGetBufferDeviceAddressKHR( vkGetInstanceProcAddr( instance, "vkGetBufferDeviceAddressKHR" ) );
      if ( !vkGetBufferDeviceAddress )
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressKHR;
      vkGetBufferOpaqueCaptureAddressKHR = PFN_vkGetBufferOpaqueCaptureAddressKHR( vkGetInstanceProcAddr( instance, "vkGetBufferOpaqueCaptureAddressKHR" ) );
      if ( !vkGetBufferOpaqueCaptureAddress )
        vkGetBufferOpaqueCaptureAddress = vkGetBufferOpaqueCaptureAddressKHR;
      vkGetDeviceMemoryOpaqueCaptureAddressKHR =
        PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceMemoryOpaqueCaptureAddressKHR" ) );
      if ( !vkGetDeviceMemoryOpaqueCaptureAddress )
        vkGetDeviceMemoryOpaqueCaptureAddress = vkGetDeviceMemoryOpaqueCaptureAddressKHR;

      //=== VK_EXT_line_rasterization ===
      vkCmdSetLineStippleEXT = PFN_vkCmdSetLineStippleEXT( vkGetInstanceProcAddr( instance, "vkCmdSetLineStippleEXT" ) );

      //=== VK_EXT_host_query_reset ===
      vkResetQueryPoolEXT = PFN_vkResetQueryPoolEXT( vkGetInstanceProcAddr( instance, "vkResetQueryPoolEXT" ) );
      if ( !vkResetQueryPool )
        vkResetQueryPool = vkResetQueryPoolEXT;

      //=== VK_EXT_extended_dynamic_state ===
      vkCmdSetCullModeEXT = PFN_vkCmdSetCullModeEXT( vkGetInstanceProcAddr( instance, "vkCmdSetCullModeEXT" ) );
      if ( !vkCmdSetCullMode )
        vkCmdSetCullMode = vkCmdSetCullModeEXT;
      vkCmdSetFrontFaceEXT = PFN_vkCmdSetFrontFaceEXT( vkGetInstanceProcAddr( instance, "vkCmdSetFrontFaceEXT" ) );
      if ( !vkCmdSetFrontFace )
        vkCmdSetFrontFace = vkCmdSetFrontFaceEXT;
      vkCmdSetPrimitiveTopologyEXT = PFN_vkCmdSetPrimitiveTopologyEXT( vkGetInstanceProcAddr( instance, "vkCmdSetPrimitiveTopologyEXT" ) );
      if ( !vkCmdSetPrimitiveTopology )
        vkCmdSetPrimitiveTopology = vkCmdSetPrimitiveTopologyEXT;
      vkCmdSetViewportWithCountEXT = PFN_vkCmdSetViewportWithCountEXT( vkGetInstanceProcAddr( instance, "vkCmdSetViewportWithCountEXT" ) );
      if ( !vkCmdSetViewportWithCount )
        vkCmdSetViewportWithCount = vkCmdSetViewportWithCountEXT;
      vkCmdSetScissorWithCountEXT = PFN_vkCmdSetScissorWithCountEXT( vkGetInstanceProcAddr( instance, "vkCmdSetScissorWithCountEXT" ) );
      if ( !vkCmdSetScissorWithCount )
        vkCmdSetScissorWithCount = vkCmdSetScissorWithCountEXT;
      vkCmdBindVertexBuffers2EXT = PFN_vkCmdBindVertexBuffers2EXT( vkGetInstanceProcAddr( instance, "vkCmdBindVertexBuffers2EXT" ) );
      if ( !vkCmdBindVertexBuffers2 )
        vkCmdBindVertexBuffers2 = vkCmdBindVertexBuffers2EXT;
      vkCmdSetDepthTestEnableEXT = PFN_vkCmdSetDepthTestEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthTestEnableEXT" ) );
      if ( !vkCmdSetDepthTestEnable )
        vkCmdSetDepthTestEnable = vkCmdSetDepthTestEnableEXT;
      vkCmdSetDepthWriteEnableEXT = PFN_vkCmdSetDepthWriteEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthWriteEnableEXT" ) );
      if ( !vkCmdSetDepthWriteEnable )
        vkCmdSetDepthWriteEnable = vkCmdSetDepthWriteEnableEXT;
      vkCmdSetDepthCompareOpEXT = PFN_vkCmdSetDepthCompareOpEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthCompareOpEXT" ) );
      if ( !vkCmdSetDepthCompareOp )
        vkCmdSetDepthCompareOp = vkCmdSetDepthCompareOpEXT;
      vkCmdSetDepthBoundsTestEnableEXT = PFN_vkCmdSetDepthBoundsTestEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBoundsTestEnableEXT" ) );
      if ( !vkCmdSetDepthBoundsTestEnable )
        vkCmdSetDepthBoundsTestEnable = vkCmdSetDepthBoundsTestEnableEXT;
      vkCmdSetStencilTestEnableEXT = PFN_vkCmdSetStencilTestEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetStencilTestEnableEXT" ) );
      if ( !vkCmdSetStencilTestEnable )
        vkCmdSetStencilTestEnable = vkCmdSetStencilTestEnableEXT;
      vkCmdSetStencilOpEXT = PFN_vkCmdSetStencilOpEXT( vkGetInstanceProcAddr( instance, "vkCmdSetStencilOpEXT" ) );
      if ( !vkCmdSetStencilOp )
        vkCmdSetStencilOp = vkCmdSetStencilOpEXT;

      //=== VK_KHR_deferred_host_operations ===
      vkCreateDeferredOperationKHR  = PFN_vkCreateDeferredOperationKHR( vkGetInstanceProcAddr( instance, "vkCreateDeferredOperationKHR" ) );
      vkDestroyDeferredOperationKHR = PFN_vkDestroyDeferredOperationKHR( vkGetInstanceProcAddr( instance, "vkDestroyDeferredOperationKHR" ) );
      vkGetDeferredOperationMaxConcurrencyKHR =
        PFN_vkGetDeferredOperationMaxConcurrencyKHR( vkGetInstanceProcAddr( instance, "vkGetDeferredOperationMaxConcurrencyKHR" ) );
      vkGetDeferredOperationResultKHR = PFN_vkGetDeferredOperationResultKHR( vkGetInstanceProcAddr( instance, "vkGetDeferredOperationResultKHR" ) );
      vkDeferredOperationJoinKHR      = PFN_vkDeferredOperationJoinKHR( vkGetInstanceProcAddr( instance, "vkDeferredOperationJoinKHR" ) );

      //=== VK_KHR_pipeline_executable_properties ===
      vkGetPipelineExecutablePropertiesKHR =
        PFN_vkGetPipelineExecutablePropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetPipelineExecutablePropertiesKHR" ) );
      vkGetPipelineExecutableStatisticsKHR =
        PFN_vkGetPipelineExecutableStatisticsKHR( vkGetInstanceProcAddr( instance, "vkGetPipelineExecutableStatisticsKHR" ) );
      vkGetPipelineExecutableInternalRepresentationsKHR =
        PFN_vkGetPipelineExecutableInternalRepresentationsKHR( vkGetInstanceProcAddr( instance, "vkGetPipelineExecutableInternalRepresentationsKHR" ) );

      //=== VK_EXT_host_image_copy ===
      vkCopyMemoryToImageEXT          = PFN_vkCopyMemoryToImageEXT( vkGetInstanceProcAddr( instance, "vkCopyMemoryToImageEXT" ) );
      vkCopyImageToMemoryEXT          = PFN_vkCopyImageToMemoryEXT( vkGetInstanceProcAddr( instance, "vkCopyImageToMemoryEXT" ) );
      vkCopyImageToImageEXT           = PFN_vkCopyImageToImageEXT( vkGetInstanceProcAddr( instance, "vkCopyImageToImageEXT" ) );
      vkTransitionImageLayoutEXT      = PFN_vkTransitionImageLayoutEXT( vkGetInstanceProcAddr( instance, "vkTransitionImageLayoutEXT" ) );
      vkGetImageSubresourceLayout2EXT = PFN_vkGetImageSubresourceLayout2EXT( vkGetInstanceProcAddr( instance, "vkGetImageSubresourceLayout2EXT" ) );
      if ( !vkGetImageSubresourceLayout2KHR )
        vkGetImageSubresourceLayout2KHR = vkGetImageSubresourceLayout2EXT;

      //=== VK_KHR_map_memory2 ===
      vkMapMemory2KHR   = PFN_vkMapMemory2KHR( vkGetInstanceProcAddr( instance, "vkMapMemory2KHR" ) );
      vkUnmapMemory2KHR = PFN_vkUnmapMemory2KHR( vkGetInstanceProcAddr( instance, "vkUnmapMemory2KHR" ) );

      //=== VK_EXT_swapchain_maintenance1 ===
      vkReleaseSwapchainImagesEXT = PFN_vkReleaseSwapchainImagesEXT( vkGetInstanceProcAddr( instance, "vkReleaseSwapchainImagesEXT" ) );

      //=== VK_NV_device_generated_commands ===
      vkGetGeneratedCommandsMemoryRequirementsNV =
        PFN_vkGetGeneratedCommandsMemoryRequirementsNV( vkGetInstanceProcAddr( instance, "vkGetGeneratedCommandsMemoryRequirementsNV" ) );
      vkCmdPreprocessGeneratedCommandsNV = PFN_vkCmdPreprocessGeneratedCommandsNV( vkGetInstanceProcAddr( instance, "vkCmdPreprocessGeneratedCommandsNV" ) );
      vkCmdExecuteGeneratedCommandsNV    = PFN_vkCmdExecuteGeneratedCommandsNV( vkGetInstanceProcAddr( instance, "vkCmdExecuteGeneratedCommandsNV" ) );
      vkCmdBindPipelineShaderGroupNV     = PFN_vkCmdBindPipelineShaderGroupNV( vkGetInstanceProcAddr( instance, "vkCmdBindPipelineShaderGroupNV" ) );
      vkCreateIndirectCommandsLayoutNV   = PFN_vkCreateIndirectCommandsLayoutNV( vkGetInstanceProcAddr( instance, "vkCreateIndirectCommandsLayoutNV" ) );
      vkDestroyIndirectCommandsLayoutNV  = PFN_vkDestroyIndirectCommandsLayoutNV( vkGetInstanceProcAddr( instance, "vkDestroyIndirectCommandsLayoutNV" ) );

      //=== VK_EXT_depth_bias_control ===
      vkCmdSetDepthBias2EXT = PFN_vkCmdSetDepthBias2EXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBias2EXT" ) );

      //=== VK_EXT_acquire_drm_display ===
      vkAcquireDrmDisplayEXT = PFN_vkAcquireDrmDisplayEXT( vkGetInstanceProcAddr( instance, "vkAcquireDrmDisplayEXT" ) );
      vkGetDrmDisplayEXT     = PFN_vkGetDrmDisplayEXT( vkGetInstanceProcAddr( instance, "vkGetDrmDisplayEXT" ) );

      //=== VK_EXT_private_data ===
      vkCreatePrivateDataSlotEXT = PFN_vkCreatePrivateDataSlotEXT( vkGetInstanceProcAddr( instance, "vkCreatePrivateDataSlotEXT" ) );
      if ( !vkCreatePrivateDataSlot )
        vkCreatePrivateDataSlot = vkCreatePrivateDataSlotEXT;
      vkDestroyPrivateDataSlotEXT = PFN_vkDestroyPrivateDataSlotEXT( vkGetInstanceProcAddr( instance, "vkDestroyPrivateDataSlotEXT" ) );
      if ( !vkDestroyPrivateDataSlot )
        vkDestroyPrivateDataSlot = vkDestroyPrivateDataSlotEXT;
      vkSetPrivateDataEXT = PFN_vkSetPrivateDataEXT( vkGetInstanceProcAddr( instance, "vkSetPrivateDataEXT" ) );
      if ( !vkSetPrivateData )
        vkSetPrivateData = vkSetPrivateDataEXT;
      vkGetPrivateDataEXT = PFN_vkGetPrivateDataEXT( vkGetInstanceProcAddr( instance, "vkGetPrivateDataEXT" ) );
      if ( !vkGetPrivateData )
        vkGetPrivateData = vkGetPrivateDataEXT;

      //=== VK_KHR_video_encode_queue ===
      vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR = PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR(
        vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR" ) );
      vkGetEncodedVideoSessionParametersKHR =
        PFN_vkGetEncodedVideoSessionParametersKHR( vkGetInstanceProcAddr( instance, "vkGetEncodedVideoSessionParametersKHR" ) );
      vkCmdEncodeVideoKHR = PFN_vkCmdEncodeVideoKHR( vkGetInstanceProcAddr( instance, "vkCmdEncodeVideoKHR" ) );

#if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_NV_cuda_kernel_launch ===
      vkCreateCudaModuleNV    = PFN_vkCreateCudaModuleNV( vkGetInstanceProcAddr( instance, "vkCreateCudaModuleNV" ) );
      vkGetCudaModuleCacheNV  = PFN_vkGetCudaModuleCacheNV( vkGetInstanceProcAddr( instance, "vkGetCudaModuleCacheNV" ) );
      vkCreateCudaFunctionNV  = PFN_vkCreateCudaFunctionNV( vkGetInstanceProcAddr( instance, "vkCreateCudaFunctionNV" ) );
      vkDestroyCudaModuleNV   = PFN_vkDestroyCudaModuleNV( vkGetInstanceProcAddr( instance, "vkDestroyCudaModuleNV" ) );
      vkDestroyCudaFunctionNV = PFN_vkDestroyCudaFunctionNV( vkGetInstanceProcAddr( instance, "vkDestroyCudaFunctionNV" ) );
      vkCmdCudaLaunchKernelNV = PFN_vkCmdCudaLaunchKernelNV( vkGetInstanceProcAddr( instance, "vkCmdCudaLaunchKernelNV" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
      //=== VK_EXT_metal_objects ===
      vkExportMetalObjectsEXT = PFN_vkExportMetalObjectsEXT( vkGetInstanceProcAddr( instance, "vkExportMetalObjectsEXT" ) );
#endif /*VK_USE_PLATFORM_METAL_EXT*/

      //=== VK_KHR_synchronization2 ===
      vkCmdSetEvent2KHR = PFN_vkCmdSetEvent2KHR( vkGetInstanceProcAddr( instance, "vkCmdSetEvent2KHR" ) );
      if ( !vkCmdSetEvent2 )
        vkCmdSetEvent2 = vkCmdSetEvent2KHR;
      vkCmdResetEvent2KHR = PFN_vkCmdResetEvent2KHR( vkGetInstanceProcAddr( instance, "vkCmdResetEvent2KHR" ) );
      if ( !vkCmdResetEvent2 )
        vkCmdResetEvent2 = vkCmdResetEvent2KHR;
      vkCmdWaitEvents2KHR = PFN_vkCmdWaitEvents2KHR( vkGetInstanceProcAddr( instance, "vkCmdWaitEvents2KHR" ) );
      if ( !vkCmdWaitEvents2 )
        vkCmdWaitEvents2 = vkCmdWaitEvents2KHR;
      vkCmdPipelineBarrier2KHR = PFN_vkCmdPipelineBarrier2KHR( vkGetInstanceProcAddr( instance, "vkCmdPipelineBarrier2KHR" ) );
      if ( !vkCmdPipelineBarrier2 )
        vkCmdPipelineBarrier2 = vkCmdPipelineBarrier2KHR;
      vkCmdWriteTimestamp2KHR = PFN_vkCmdWriteTimestamp2KHR( vkGetInstanceProcAddr( instance, "vkCmdWriteTimestamp2KHR" ) );
      if ( !vkCmdWriteTimestamp2 )
        vkCmdWriteTimestamp2 = vkCmdWriteTimestamp2KHR;
      vkQueueSubmit2KHR = PFN_vkQueueSubmit2KHR( vkGetInstanceProcAddr( instance, "vkQueueSubmit2KHR" ) );
      if ( !vkQueueSubmit2 )
        vkQueueSubmit2 = vkQueueSubmit2KHR;
      vkCmdWriteBufferMarker2AMD  = PFN_vkCmdWriteBufferMarker2AMD( vkGetInstanceProcAddr( instance, "vkCmdWriteBufferMarker2AMD" ) );
      vkGetQueueCheckpointData2NV = PFN_vkGetQueueCheckpointData2NV( vkGetInstanceProcAddr( instance, "vkGetQueueCheckpointData2NV" ) );

      //=== VK_EXT_descriptor_buffer ===
      vkGetDescriptorSetLayoutSizeEXT = PFN_vkGetDescriptorSetLayoutSizeEXT( vkGetInstanceProcAddr( instance, "vkGetDescriptorSetLayoutSizeEXT" ) );
      vkGetDescriptorSetLayoutBindingOffsetEXT =
        PFN_vkGetDescriptorSetLayoutBindingOffsetEXT( vkGetInstanceProcAddr( instance, "vkGetDescriptorSetLayoutBindingOffsetEXT" ) );
      vkGetDescriptorEXT                 = PFN_vkGetDescriptorEXT( vkGetInstanceProcAddr( instance, "vkGetDescriptorEXT" ) );
      vkCmdBindDescriptorBuffersEXT      = PFN_vkCmdBindDescriptorBuffersEXT( vkGetInstanceProcAddr( instance, "vkCmdBindDescriptorBuffersEXT" ) );
      vkCmdSetDescriptorBufferOffsetsEXT = PFN_vkCmdSetDescriptorBufferOffsetsEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDescriptorBufferOffsetsEXT" ) );
      vkCmdBindDescriptorBufferEmbeddedSamplersEXT =
        PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT( vkGetInstanceProcAddr( instance, "vkCmdBindDescriptorBufferEmbeddedSamplersEXT" ) );
      vkGetBufferOpaqueCaptureDescriptorDataEXT =
        PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT( vkGetInstanceProcAddr( instance, "vkGetBufferOpaqueCaptureDescriptorDataEXT" ) );
      vkGetImageOpaqueCaptureDescriptorDataEXT =
        PFN_vkGetImageOpaqueCaptureDescriptorDataEXT( vkGetInstanceProcAddr( instance, "vkGetImageOpaqueCaptureDescriptorDataEXT" ) );
      vkGetImageViewOpaqueCaptureDescriptorDataEXT =
        PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT( vkGetInstanceProcAddr( instance, "vkGetImageViewOpaqueCaptureDescriptorDataEXT" ) );
      vkGetSamplerOpaqueCaptureDescriptorDataEXT =
        PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT( vkGetInstanceProcAddr( instance, "vkGetSamplerOpaqueCaptureDescriptorDataEXT" ) );
      vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT = PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT(
        vkGetInstanceProcAddr( instance, "vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT" ) );

      //=== VK_NV_fragment_shading_rate_enums ===
      vkCmdSetFragmentShadingRateEnumNV = PFN_vkCmdSetFragmentShadingRateEnumNV( vkGetInstanceProcAddr( instance, "vkCmdSetFragmentShadingRateEnumNV" ) );

      //=== VK_EXT_mesh_shader ===
      vkCmdDrawMeshTasksEXT              = PFN_vkCmdDrawMeshTasksEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksEXT" ) );
      vkCmdDrawMeshTasksIndirectEXT      = PFN_vkCmdDrawMeshTasksIndirectEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksIndirectEXT" ) );
      vkCmdDrawMeshTasksIndirectCountEXT = PFN_vkCmdDrawMeshTasksIndirectCountEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawMeshTasksIndirectCountEXT" ) );

      //=== VK_KHR_copy_commands2 ===
      vkCmdCopyBuffer2KHR = PFN_vkCmdCopyBuffer2KHR( vkGetInstanceProcAddr( instance, "vkCmdCopyBuffer2KHR" ) );
      if ( !vkCmdCopyBuffer2 )
        vkCmdCopyBuffer2 = vkCmdCopyBuffer2KHR;
      vkCmdCopyImage2KHR = PFN_vkCmdCopyImage2KHR( vkGetInstanceProcAddr( instance, "vkCmdCopyImage2KHR" ) );
      if ( !vkCmdCopyImage2 )
        vkCmdCopyImage2 = vkCmdCopyImage2KHR;
      vkCmdCopyBufferToImage2KHR = PFN_vkCmdCopyBufferToImage2KHR( vkGetInstanceProcAddr( instance, "vkCmdCopyBufferToImage2KHR" ) );
      if ( !vkCmdCopyBufferToImage2 )
        vkCmdCopyBufferToImage2 = vkCmdCopyBufferToImage2KHR;
      vkCmdCopyImageToBuffer2KHR = PFN_vkCmdCopyImageToBuffer2KHR( vkGetInstanceProcAddr( instance, "vkCmdCopyImageToBuffer2KHR" ) );
      if ( !vkCmdCopyImageToBuffer2 )
        vkCmdCopyImageToBuffer2 = vkCmdCopyImageToBuffer2KHR;
      vkCmdBlitImage2KHR = PFN_vkCmdBlitImage2KHR( vkGetInstanceProcAddr( instance, "vkCmdBlitImage2KHR" ) );
      if ( !vkCmdBlitImage2 )
        vkCmdBlitImage2 = vkCmdBlitImage2KHR;
      vkCmdResolveImage2KHR = PFN_vkCmdResolveImage2KHR( vkGetInstanceProcAddr( instance, "vkCmdResolveImage2KHR" ) );
      if ( !vkCmdResolveImage2 )
        vkCmdResolveImage2 = vkCmdResolveImage2KHR;

      //=== VK_EXT_device_fault ===
      vkGetDeviceFaultInfoEXT = PFN_vkGetDeviceFaultInfoEXT( vkGetInstanceProcAddr( instance, "vkGetDeviceFaultInfoEXT" ) );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_NV_acquire_winrt_display ===
      vkAcquireWinrtDisplayNV = PFN_vkAcquireWinrtDisplayNV( vkGetInstanceProcAddr( instance, "vkAcquireWinrtDisplayNV" ) );
      vkGetWinrtDisplayNV     = PFN_vkGetWinrtDisplayNV( vkGetInstanceProcAddr( instance, "vkGetWinrtDisplayNV" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      //=== VK_EXT_directfb_surface ===
      vkCreateDirectFBSurfaceEXT = PFN_vkCreateDirectFBSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateDirectFBSurfaceEXT" ) );
      vkGetPhysicalDeviceDirectFBPresentationSupportEXT =
        PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDirectFBPresentationSupportEXT" ) );
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

      //=== VK_EXT_vertex_input_dynamic_state ===
      vkCmdSetVertexInputEXT = PFN_vkCmdSetVertexInputEXT( vkGetInstanceProcAddr( instance, "vkCmdSetVertexInputEXT" ) );

#if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_external_memory ===
      vkGetMemoryZirconHandleFUCHSIA = PFN_vkGetMemoryZirconHandleFUCHSIA( vkGetInstanceProcAddr( instance, "vkGetMemoryZirconHandleFUCHSIA" ) );
      vkGetMemoryZirconHandlePropertiesFUCHSIA =
        PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA( vkGetInstanceProcAddr( instance, "vkGetMemoryZirconHandlePropertiesFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_external_semaphore ===
      vkImportSemaphoreZirconHandleFUCHSIA =
        PFN_vkImportSemaphoreZirconHandleFUCHSIA( vkGetInstanceProcAddr( instance, "vkImportSemaphoreZirconHandleFUCHSIA" ) );
      vkGetSemaphoreZirconHandleFUCHSIA = PFN_vkGetSemaphoreZirconHandleFUCHSIA( vkGetInstanceProcAddr( instance, "vkGetSemaphoreZirconHandleFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_buffer_collection ===
      vkCreateBufferCollectionFUCHSIA = PFN_vkCreateBufferCollectionFUCHSIA( vkGetInstanceProcAddr( instance, "vkCreateBufferCollectionFUCHSIA" ) );
      vkSetBufferCollectionImageConstraintsFUCHSIA =
        PFN_vkSetBufferCollectionImageConstraintsFUCHSIA( vkGetInstanceProcAddr( instance, "vkSetBufferCollectionImageConstraintsFUCHSIA" ) );
      vkSetBufferCollectionBufferConstraintsFUCHSIA =
        PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA( vkGetInstanceProcAddr( instance, "vkSetBufferCollectionBufferConstraintsFUCHSIA" ) );
      vkDestroyBufferCollectionFUCHSIA = PFN_vkDestroyBufferCollectionFUCHSIA( vkGetInstanceProcAddr( instance, "vkDestroyBufferCollectionFUCHSIA" ) );
      vkGetBufferCollectionPropertiesFUCHSIA =
        PFN_vkGetBufferCollectionPropertiesFUCHSIA( vkGetInstanceProcAddr( instance, "vkGetBufferCollectionPropertiesFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

      //=== VK_HUAWEI_subpass_shading ===
      vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI =
        PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI( vkGetInstanceProcAddr( instance, "vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI" ) );
      vkCmdSubpassShadingHUAWEI = PFN_vkCmdSubpassShadingHUAWEI( vkGetInstanceProcAddr( instance, "vkCmdSubpassShadingHUAWEI" ) );

      //=== VK_HUAWEI_invocation_mask ===
      vkCmdBindInvocationMaskHUAWEI = PFN_vkCmdBindInvocationMaskHUAWEI( vkGetInstanceProcAddr( instance, "vkCmdBindInvocationMaskHUAWEI" ) );

      //=== VK_NV_external_memory_rdma ===
      vkGetMemoryRemoteAddressNV = PFN_vkGetMemoryRemoteAddressNV( vkGetInstanceProcAddr( instance, "vkGetMemoryRemoteAddressNV" ) );

      //=== VK_EXT_pipeline_properties ===
      vkGetPipelinePropertiesEXT = PFN_vkGetPipelinePropertiesEXT( vkGetInstanceProcAddr( instance, "vkGetPipelinePropertiesEXT" ) );

      //=== VK_EXT_extended_dynamic_state2 ===
      vkCmdSetPatchControlPointsEXT      = PFN_vkCmdSetPatchControlPointsEXT( vkGetInstanceProcAddr( instance, "vkCmdSetPatchControlPointsEXT" ) );
      vkCmdSetRasterizerDiscardEnableEXT = PFN_vkCmdSetRasterizerDiscardEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetRasterizerDiscardEnableEXT" ) );
      if ( !vkCmdSetRasterizerDiscardEnable )
        vkCmdSetRasterizerDiscardEnable = vkCmdSetRasterizerDiscardEnableEXT;
      vkCmdSetDepthBiasEnableEXT = PFN_vkCmdSetDepthBiasEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthBiasEnableEXT" ) );
      if ( !vkCmdSetDepthBiasEnable )
        vkCmdSetDepthBiasEnable = vkCmdSetDepthBiasEnableEXT;
      vkCmdSetLogicOpEXT                = PFN_vkCmdSetLogicOpEXT( vkGetInstanceProcAddr( instance, "vkCmdSetLogicOpEXT" ) );
      vkCmdSetPrimitiveRestartEnableEXT = PFN_vkCmdSetPrimitiveRestartEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetPrimitiveRestartEnableEXT" ) );
      if ( !vkCmdSetPrimitiveRestartEnable )
        vkCmdSetPrimitiveRestartEnable = vkCmdSetPrimitiveRestartEnableEXT;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      //=== VK_QNX_screen_surface ===
      vkCreateScreenSurfaceQNX = PFN_vkCreateScreenSurfaceQNX( vkGetInstanceProcAddr( instance, "vkCreateScreenSurfaceQNX" ) );
      vkGetPhysicalDeviceScreenPresentationSupportQNX =
        PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceScreenPresentationSupportQNX" ) );
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

      //=== VK_EXT_color_write_enable ===
      vkCmdSetColorWriteEnableEXT = PFN_vkCmdSetColorWriteEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetColorWriteEnableEXT" ) );

      //=== VK_KHR_ray_tracing_maintenance1 ===
      vkCmdTraceRaysIndirect2KHR = PFN_vkCmdTraceRaysIndirect2KHR( vkGetInstanceProcAddr( instance, "vkCmdTraceRaysIndirect2KHR" ) );

      //=== VK_EXT_multi_draw ===
      vkCmdDrawMultiEXT        = PFN_vkCmdDrawMultiEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawMultiEXT" ) );
      vkCmdDrawMultiIndexedEXT = PFN_vkCmdDrawMultiIndexedEXT( vkGetInstanceProcAddr( instance, "vkCmdDrawMultiIndexedEXT" ) );

      //=== VK_EXT_opacity_micromap ===
      vkCreateMicromapEXT                 = PFN_vkCreateMicromapEXT( vkGetInstanceProcAddr( instance, "vkCreateMicromapEXT" ) );
      vkDestroyMicromapEXT                = PFN_vkDestroyMicromapEXT( vkGetInstanceProcAddr( instance, "vkDestroyMicromapEXT" ) );
      vkCmdBuildMicromapsEXT              = PFN_vkCmdBuildMicromapsEXT( vkGetInstanceProcAddr( instance, "vkCmdBuildMicromapsEXT" ) );
      vkBuildMicromapsEXT                 = PFN_vkBuildMicromapsEXT( vkGetInstanceProcAddr( instance, "vkBuildMicromapsEXT" ) );
      vkCopyMicromapEXT                   = PFN_vkCopyMicromapEXT( vkGetInstanceProcAddr( instance, "vkCopyMicromapEXT" ) );
      vkCopyMicromapToMemoryEXT           = PFN_vkCopyMicromapToMemoryEXT( vkGetInstanceProcAddr( instance, "vkCopyMicromapToMemoryEXT" ) );
      vkCopyMemoryToMicromapEXT           = PFN_vkCopyMemoryToMicromapEXT( vkGetInstanceProcAddr( instance, "vkCopyMemoryToMicromapEXT" ) );
      vkWriteMicromapsPropertiesEXT       = PFN_vkWriteMicromapsPropertiesEXT( vkGetInstanceProcAddr( instance, "vkWriteMicromapsPropertiesEXT" ) );
      vkCmdCopyMicromapEXT                = PFN_vkCmdCopyMicromapEXT( vkGetInstanceProcAddr( instance, "vkCmdCopyMicromapEXT" ) );
      vkCmdCopyMicromapToMemoryEXT        = PFN_vkCmdCopyMicromapToMemoryEXT( vkGetInstanceProcAddr( instance, "vkCmdCopyMicromapToMemoryEXT" ) );
      vkCmdCopyMemoryToMicromapEXT        = PFN_vkCmdCopyMemoryToMicromapEXT( vkGetInstanceProcAddr( instance, "vkCmdCopyMemoryToMicromapEXT" ) );
      vkCmdWriteMicromapsPropertiesEXT    = PFN_vkCmdWriteMicromapsPropertiesEXT( vkGetInstanceProcAddr( instance, "vkCmdWriteMicromapsPropertiesEXT" ) );
      vkGetDeviceMicromapCompatibilityEXT = PFN_vkGetDeviceMicromapCompatibilityEXT( vkGetInstanceProcAddr( instance, "vkGetDeviceMicromapCompatibilityEXT" ) );
      vkGetMicromapBuildSizesEXT          = PFN_vkGetMicromapBuildSizesEXT( vkGetInstanceProcAddr( instance, "vkGetMicromapBuildSizesEXT" ) );

      //=== VK_HUAWEI_cluster_culling_shader ===
      vkCmdDrawClusterHUAWEI         = PFN_vkCmdDrawClusterHUAWEI( vkGetInstanceProcAddr( instance, "vkCmdDrawClusterHUAWEI" ) );
      vkCmdDrawClusterIndirectHUAWEI = PFN_vkCmdDrawClusterIndirectHUAWEI( vkGetInstanceProcAddr( instance, "vkCmdDrawClusterIndirectHUAWEI" ) );

      //=== VK_EXT_pageable_device_local_memory ===
      vkSetDeviceMemoryPriorityEXT = PFN_vkSetDeviceMemoryPriorityEXT( vkGetInstanceProcAddr( instance, "vkSetDeviceMemoryPriorityEXT" ) );

      //=== VK_KHR_maintenance4 ===
      vkGetDeviceBufferMemoryRequirementsKHR =
        PFN_vkGetDeviceBufferMemoryRequirementsKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceBufferMemoryRequirementsKHR" ) );
      if ( !vkGetDeviceBufferMemoryRequirements )
        vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirementsKHR;
      vkGetDeviceImageMemoryRequirementsKHR =
        PFN_vkGetDeviceImageMemoryRequirementsKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceImageMemoryRequirementsKHR" ) );
      if ( !vkGetDeviceImageMemoryRequirements )
        vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirementsKHR;
      vkGetDeviceImageSparseMemoryRequirementsKHR =
        PFN_vkGetDeviceImageSparseMemoryRequirementsKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceImageSparseMemoryRequirementsKHR" ) );
      if ( !vkGetDeviceImageSparseMemoryRequirements )
        vkGetDeviceImageSparseMemoryRequirements = vkGetDeviceImageSparseMemoryRequirementsKHR;

      //=== VK_VALVE_descriptor_set_host_mapping ===
      vkGetDescriptorSetLayoutHostMappingInfoVALVE =
        PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE( vkGetInstanceProcAddr( instance, "vkGetDescriptorSetLayoutHostMappingInfoVALVE" ) );
      vkGetDescriptorSetHostMappingVALVE = PFN_vkGetDescriptorSetHostMappingVALVE( vkGetInstanceProcAddr( instance, "vkGetDescriptorSetHostMappingVALVE" ) );

      //=== VK_NV_copy_memory_indirect ===
      vkCmdCopyMemoryIndirectNV        = PFN_vkCmdCopyMemoryIndirectNV( vkGetInstanceProcAddr( instance, "vkCmdCopyMemoryIndirectNV" ) );
      vkCmdCopyMemoryToImageIndirectNV = PFN_vkCmdCopyMemoryToImageIndirectNV( vkGetInstanceProcAddr( instance, "vkCmdCopyMemoryToImageIndirectNV" ) );

      //=== VK_NV_memory_decompression ===
      vkCmdDecompressMemoryNV = PFN_vkCmdDecompressMemoryNV( vkGetInstanceProcAddr( instance, "vkCmdDecompressMemoryNV" ) );
      vkCmdDecompressMemoryIndirectCountNV =
        PFN_vkCmdDecompressMemoryIndirectCountNV( vkGetInstanceProcAddr( instance, "vkCmdDecompressMemoryIndirectCountNV" ) );

      //=== VK_NV_device_generated_commands_compute ===
      vkGetPipelineIndirectMemoryRequirementsNV =
        PFN_vkGetPipelineIndirectMemoryRequirementsNV( vkGetInstanceProcAddr( instance, "vkGetPipelineIndirectMemoryRequirementsNV" ) );
      vkCmdUpdatePipelineIndirectBufferNV = PFN_vkCmdUpdatePipelineIndirectBufferNV( vkGetInstanceProcAddr( instance, "vkCmdUpdatePipelineIndirectBufferNV" ) );
      vkGetPipelineIndirectDeviceAddressNV =
        PFN_vkGetPipelineIndirectDeviceAddressNV( vkGetInstanceProcAddr( instance, "vkGetPipelineIndirectDeviceAddressNV" ) );

      //=== VK_EXT_extended_dynamic_state3 ===
      vkCmdSetTessellationDomainOriginEXT = PFN_vkCmdSetTessellationDomainOriginEXT( vkGetInstanceProcAddr( instance, "vkCmdSetTessellationDomainOriginEXT" ) );
      vkCmdSetDepthClampEnableEXT         = PFN_vkCmdSetDepthClampEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthClampEnableEXT" ) );
      vkCmdSetPolygonModeEXT              = PFN_vkCmdSetPolygonModeEXT( vkGetInstanceProcAddr( instance, "vkCmdSetPolygonModeEXT" ) );
      vkCmdSetRasterizationSamplesEXT     = PFN_vkCmdSetRasterizationSamplesEXT( vkGetInstanceProcAddr( instance, "vkCmdSetRasterizationSamplesEXT" ) );
      vkCmdSetSampleMaskEXT               = PFN_vkCmdSetSampleMaskEXT( vkGetInstanceProcAddr( instance, "vkCmdSetSampleMaskEXT" ) );
      vkCmdSetAlphaToCoverageEnableEXT    = PFN_vkCmdSetAlphaToCoverageEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetAlphaToCoverageEnableEXT" ) );
      vkCmdSetAlphaToOneEnableEXT         = PFN_vkCmdSetAlphaToOneEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetAlphaToOneEnableEXT" ) );
      vkCmdSetLogicOpEnableEXT            = PFN_vkCmdSetLogicOpEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetLogicOpEnableEXT" ) );
      vkCmdSetColorBlendEnableEXT         = PFN_vkCmdSetColorBlendEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetColorBlendEnableEXT" ) );
      vkCmdSetColorBlendEquationEXT       = PFN_vkCmdSetColorBlendEquationEXT( vkGetInstanceProcAddr( instance, "vkCmdSetColorBlendEquationEXT" ) );
      vkCmdSetColorWriteMaskEXT           = PFN_vkCmdSetColorWriteMaskEXT( vkGetInstanceProcAddr( instance, "vkCmdSetColorWriteMaskEXT" ) );
      vkCmdSetRasterizationStreamEXT      = PFN_vkCmdSetRasterizationStreamEXT( vkGetInstanceProcAddr( instance, "vkCmdSetRasterizationStreamEXT" ) );
      vkCmdSetConservativeRasterizationModeEXT =
        PFN_vkCmdSetConservativeRasterizationModeEXT( vkGetInstanceProcAddr( instance, "vkCmdSetConservativeRasterizationModeEXT" ) );
      vkCmdSetExtraPrimitiveOverestimationSizeEXT =
        PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT( vkGetInstanceProcAddr( instance, "vkCmdSetExtraPrimitiveOverestimationSizeEXT" ) );
      vkCmdSetDepthClipEnableEXT       = PFN_vkCmdSetDepthClipEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthClipEnableEXT" ) );
      vkCmdSetSampleLocationsEnableEXT = PFN_vkCmdSetSampleLocationsEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetSampleLocationsEnableEXT" ) );
      vkCmdSetColorBlendAdvancedEXT    = PFN_vkCmdSetColorBlendAdvancedEXT( vkGetInstanceProcAddr( instance, "vkCmdSetColorBlendAdvancedEXT" ) );
      vkCmdSetProvokingVertexModeEXT   = PFN_vkCmdSetProvokingVertexModeEXT( vkGetInstanceProcAddr( instance, "vkCmdSetProvokingVertexModeEXT" ) );
      vkCmdSetLineRasterizationModeEXT = PFN_vkCmdSetLineRasterizationModeEXT( vkGetInstanceProcAddr( instance, "vkCmdSetLineRasterizationModeEXT" ) );
      vkCmdSetLineStippleEnableEXT     = PFN_vkCmdSetLineStippleEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetLineStippleEnableEXT" ) );
      vkCmdSetDepthClipNegativeOneToOneEXT =
        PFN_vkCmdSetDepthClipNegativeOneToOneEXT( vkGetInstanceProcAddr( instance, "vkCmdSetDepthClipNegativeOneToOneEXT" ) );
      vkCmdSetViewportWScalingEnableNV  = PFN_vkCmdSetViewportWScalingEnableNV( vkGetInstanceProcAddr( instance, "vkCmdSetViewportWScalingEnableNV" ) );
      vkCmdSetViewportSwizzleNV         = PFN_vkCmdSetViewportSwizzleNV( vkGetInstanceProcAddr( instance, "vkCmdSetViewportSwizzleNV" ) );
      vkCmdSetCoverageToColorEnableNV   = PFN_vkCmdSetCoverageToColorEnableNV( vkGetInstanceProcAddr( instance, "vkCmdSetCoverageToColorEnableNV" ) );
      vkCmdSetCoverageToColorLocationNV = PFN_vkCmdSetCoverageToColorLocationNV( vkGetInstanceProcAddr( instance, "vkCmdSetCoverageToColorLocationNV" ) );
      vkCmdSetCoverageModulationModeNV  = PFN_vkCmdSetCoverageModulationModeNV( vkGetInstanceProcAddr( instance, "vkCmdSetCoverageModulationModeNV" ) );
      vkCmdSetCoverageModulationTableEnableNV =
        PFN_vkCmdSetCoverageModulationTableEnableNV( vkGetInstanceProcAddr( instance, "vkCmdSetCoverageModulationTableEnableNV" ) );
      vkCmdSetCoverageModulationTableNV = PFN_vkCmdSetCoverageModulationTableNV( vkGetInstanceProcAddr( instance, "vkCmdSetCoverageModulationTableNV" ) );
      vkCmdSetShadingRateImageEnableNV  = PFN_vkCmdSetShadingRateImageEnableNV( vkGetInstanceProcAddr( instance, "vkCmdSetShadingRateImageEnableNV" ) );
      vkCmdSetRepresentativeFragmentTestEnableNV =
        PFN_vkCmdSetRepresentativeFragmentTestEnableNV( vkGetInstanceProcAddr( instance, "vkCmdSetRepresentativeFragmentTestEnableNV" ) );
      vkCmdSetCoverageReductionModeNV = PFN_vkCmdSetCoverageReductionModeNV( vkGetInstanceProcAddr( instance, "vkCmdSetCoverageReductionModeNV" ) );

      //=== VK_EXT_shader_module_identifier ===
      vkGetShaderModuleIdentifierEXT = PFN_vkGetShaderModuleIdentifierEXT( vkGetInstanceProcAddr( instance, "vkGetShaderModuleIdentifierEXT" ) );
      vkGetShaderModuleCreateInfoIdentifierEXT =
        PFN_vkGetShaderModuleCreateInfoIdentifierEXT( vkGetInstanceProcAddr( instance, "vkGetShaderModuleCreateInfoIdentifierEXT" ) );

      //=== VK_NV_optical_flow ===
      vkGetPhysicalDeviceOpticalFlowImageFormatsNV =
        PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceOpticalFlowImageFormatsNV" ) );
      vkCreateOpticalFlowSessionNV    = PFN_vkCreateOpticalFlowSessionNV( vkGetInstanceProcAddr( instance, "vkCreateOpticalFlowSessionNV" ) );
      vkDestroyOpticalFlowSessionNV   = PFN_vkDestroyOpticalFlowSessionNV( vkGetInstanceProcAddr( instance, "vkDestroyOpticalFlowSessionNV" ) );
      vkBindOpticalFlowSessionImageNV = PFN_vkBindOpticalFlowSessionImageNV( vkGetInstanceProcAddr( instance, "vkBindOpticalFlowSessionImageNV" ) );
      vkCmdOpticalFlowExecuteNV       = PFN_vkCmdOpticalFlowExecuteNV( vkGetInstanceProcAddr( instance, "vkCmdOpticalFlowExecuteNV" ) );

      //=== VK_KHR_maintenance5 ===
      vkCmdBindIndexBuffer2KHR         = PFN_vkCmdBindIndexBuffer2KHR( vkGetInstanceProcAddr( instance, "vkCmdBindIndexBuffer2KHR" ) );
      vkGetRenderingAreaGranularityKHR = PFN_vkGetRenderingAreaGranularityKHR( vkGetInstanceProcAddr( instance, "vkGetRenderingAreaGranularityKHR" ) );
      vkGetDeviceImageSubresourceLayoutKHR =
        PFN_vkGetDeviceImageSubresourceLayoutKHR( vkGetInstanceProcAddr( instance, "vkGetDeviceImageSubresourceLayoutKHR" ) );
      vkGetImageSubresourceLayout2KHR = PFN_vkGetImageSubresourceLayout2KHR( vkGetInstanceProcAddr( instance, "vkGetImageSubresourceLayout2KHR" ) );

      //=== VK_EXT_shader_object ===
      vkCreateShadersEXT       = PFN_vkCreateShadersEXT( vkGetInstanceProcAddr( instance, "vkCreateShadersEXT" ) );
      vkDestroyShaderEXT       = PFN_vkDestroyShaderEXT( vkGetInstanceProcAddr( instance, "vkDestroyShaderEXT" ) );
      vkGetShaderBinaryDataEXT = PFN_vkGetShaderBinaryDataEXT( vkGetInstanceProcAddr( instance, "vkGetShaderBinaryDataEXT" ) );
      vkCmdBindShadersEXT      = PFN_vkCmdBindShadersEXT( vkGetInstanceProcAddr( instance, "vkCmdBindShadersEXT" ) );

      //=== VK_QCOM_tile_properties ===
      vkGetFramebufferTilePropertiesQCOM = PFN_vkGetFramebufferTilePropertiesQCOM( vkGetInstanceProcAddr( instance, "vkGetFramebufferTilePropertiesQCOM" ) );
      vkGetDynamicRenderingTilePropertiesQCOM =
        PFN_vkGetDynamicRenderingTilePropertiesQCOM( vkGetInstanceProcAddr( instance, "vkGetDynamicRenderingTilePropertiesQCOM" ) );

      //=== VK_NV_low_latency2 ===
      vkSetLatencySleepModeNV  = PFN_vkSetLatencySleepModeNV( vkGetInstanceProcAddr( instance, "vkSetLatencySleepModeNV" ) );
      vkLatencySleepNV         = PFN_vkLatencySleepNV( vkGetInstanceProcAddr( instance, "vkLatencySleepNV" ) );
      vkSetLatencyMarkerNV     = PFN_vkSetLatencyMarkerNV( vkGetInstanceProcAddr( instance, "vkSetLatencyMarkerNV" ) );
      vkGetLatencyTimingsNV    = PFN_vkGetLatencyTimingsNV( vkGetInstanceProcAddr( instance, "vkGetLatencyTimingsNV" ) );
      vkQueueNotifyOutOfBandNV = PFN_vkQueueNotifyOutOfBandNV( vkGetInstanceProcAddr( instance, "vkQueueNotifyOutOfBandNV" ) );

      //=== VK_KHR_cooperative_matrix ===
      vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR" ) );

      //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
      vkCmdSetAttachmentFeedbackLoopEnableEXT =
        PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT( vkGetInstanceProcAddr( instance, "vkCmdSetAttachmentFeedbackLoopEnableEXT" ) );

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      //=== VK_QNX_external_memory_screen_buffer ===
      vkGetScreenBufferPropertiesQNX = PFN_vkGetScreenBufferPropertiesQNX( vkGetInstanceProcAddr( instance, "vkGetScreenBufferPropertiesQNX" ) );
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

      //=== VK_KHR_calibrated_timestamps ===
      vkGetPhysicalDeviceCalibrateableTimeDomainsKHR =
        PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR" ) );
      vkGetCalibratedTimestampsKHR = PFN_vkGetCalibratedTimestampsKHR( vkGetInstanceProcAddr( instance, "vkGetCalibratedTimestampsKHR" ) );

      //=== VK_KHR_maintenance6 ===
      vkCmdBindDescriptorSets2KHR = PFN_vkCmdBindDescriptorSets2KHR( vkGetInstanceProcAddr( instance, "vkCmdBindDescriptorSets2KHR" ) );
      vkCmdPushConstants2KHR      = PFN_vkCmdPushConstants2KHR( vkGetInstanceProcAddr( instance, "vkCmdPushConstants2KHR" ) );
      vkCmdPushDescriptorSet2KHR  = PFN_vkCmdPushDescriptorSet2KHR( vkGetInstanceProcAddr( instance, "vkCmdPushDescriptorSet2KHR" ) );
      vkCmdPushDescriptorSetWithTemplate2KHR =
        PFN_vkCmdPushDescriptorSetWithTemplate2KHR( vkGetInstanceProcAddr( instance, "vkCmdPushDescriptorSetWithTemplate2KHR" ) );
      vkCmdSetDescriptorBufferOffsets2EXT = PFN_vkCmdSetDescriptorBufferOffsets2EXT( vkGetInstanceProcAddr( instance, "vkCmdSetDescriptorBufferOffsets2EXT" ) );
      vkCmdBindDescriptorBufferEmbeddedSamplers2EXT =
        PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT( vkGetInstanceProcAddr( instance, "vkCmdBindDescriptorBufferEmbeddedSamplers2EXT" ) );
    }

    void init( VULKAN_HPP_NAMESPACE::Device deviceCpp ) VULKAN_HPP_NOEXCEPT
    {
      VkDevice device = static_cast<VkDevice>( deviceCpp );

      //=== VK_VERSION_1_0 ===
      vkGetDeviceProcAddr                = PFN_vkGetDeviceProcAddr( vkGetDeviceProcAddr( device, "vkGetDeviceProcAddr" ) );
      vkDestroyDevice                    = PFN_vkDestroyDevice( vkGetDeviceProcAddr( device, "vkDestroyDevice" ) );
      vkGetDeviceQueue                   = PFN_vkGetDeviceQueue( vkGetDeviceProcAddr( device, "vkGetDeviceQueue" ) );
      vkQueueSubmit                      = PFN_vkQueueSubmit( vkGetDeviceProcAddr( device, "vkQueueSubmit" ) );
      vkQueueWaitIdle                    = PFN_vkQueueWaitIdle( vkGetDeviceProcAddr( device, "vkQueueWaitIdle" ) );
      vkDeviceWaitIdle                   = PFN_vkDeviceWaitIdle( vkGetDeviceProcAddr( device, "vkDeviceWaitIdle" ) );
      vkAllocateMemory                   = PFN_vkAllocateMemory( vkGetDeviceProcAddr( device, "vkAllocateMemory" ) );
      vkFreeMemory                       = PFN_vkFreeMemory( vkGetDeviceProcAddr( device, "vkFreeMemory" ) );
      vkMapMemory                        = PFN_vkMapMemory( vkGetDeviceProcAddr( device, "vkMapMemory" ) );
      vkUnmapMemory                      = PFN_vkUnmapMemory( vkGetDeviceProcAddr( device, "vkUnmapMemory" ) );
      vkFlushMappedMemoryRanges          = PFN_vkFlushMappedMemoryRanges( vkGetDeviceProcAddr( device, "vkFlushMappedMemoryRanges" ) );
      vkInvalidateMappedMemoryRanges     = PFN_vkInvalidateMappedMemoryRanges( vkGetDeviceProcAddr( device, "vkInvalidateMappedMemoryRanges" ) );
      vkGetDeviceMemoryCommitment        = PFN_vkGetDeviceMemoryCommitment( vkGetDeviceProcAddr( device, "vkGetDeviceMemoryCommitment" ) );
      vkBindBufferMemory                 = PFN_vkBindBufferMemory( vkGetDeviceProcAddr( device, "vkBindBufferMemory" ) );
      vkBindImageMemory                  = PFN_vkBindImageMemory( vkGetDeviceProcAddr( device, "vkBindImageMemory" ) );
      vkGetBufferMemoryRequirements      = PFN_vkGetBufferMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements" ) );
      vkGetImageMemoryRequirements       = PFN_vkGetImageMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements" ) );
      vkGetImageSparseMemoryRequirements = PFN_vkGetImageSparseMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements" ) );
      vkQueueBindSparse                  = PFN_vkQueueBindSparse( vkGetDeviceProcAddr( device, "vkQueueBindSparse" ) );
      vkCreateFence                      = PFN_vkCreateFence( vkGetDeviceProcAddr( device, "vkCreateFence" ) );
      vkDestroyFence                     = PFN_vkDestroyFence( vkGetDeviceProcAddr( device, "vkDestroyFence" ) );
      vkResetFences                      = PFN_vkResetFences( vkGetDeviceProcAddr( device, "vkResetFences" ) );
      vkGetFenceStatus                   = PFN_vkGetFenceStatus( vkGetDeviceProcAddr( device, "vkGetFenceStatus" ) );
      vkWaitForFences                    = PFN_vkWaitForFences( vkGetDeviceProcAddr( device, "vkWaitForFences" ) );
      vkCreateSemaphore                  = PFN_vkCreateSemaphore( vkGetDeviceProcAddr( device, "vkCreateSemaphore" ) );
      vkDestroySemaphore                 = PFN_vkDestroySemaphore( vkGetDeviceProcAddr( device, "vkDestroySemaphore" ) );
      vkCreateEvent                      = PFN_vkCreateEvent( vkGetDeviceProcAddr( device, "vkCreateEvent" ) );
      vkDestroyEvent                     = PFN_vkDestroyEvent( vkGetDeviceProcAddr( device, "vkDestroyEvent" ) );
      vkGetEventStatus                   = PFN_vkGetEventStatus( vkGetDeviceProcAddr( device, "vkGetEventStatus" ) );
      vkSetEvent                         = PFN_vkSetEvent( vkGetDeviceProcAddr( device, "vkSetEvent" ) );
      vkResetEvent                       = PFN_vkResetEvent( vkGetDeviceProcAddr( device, "vkResetEvent" ) );
      vkCreateQueryPool                  = PFN_vkCreateQueryPool( vkGetDeviceProcAddr( device, "vkCreateQueryPool" ) );
      vkDestroyQueryPool                 = PFN_vkDestroyQueryPool( vkGetDeviceProcAddr( device, "vkDestroyQueryPool" ) );
      vkGetQueryPoolResults              = PFN_vkGetQueryPoolResults( vkGetDeviceProcAddr( device, "vkGetQueryPoolResults" ) );
      vkCreateBuffer                     = PFN_vkCreateBuffer( vkGetDeviceProcAddr( device, "vkCreateBuffer" ) );
      vkDestroyBuffer                    = PFN_vkDestroyBuffer( vkGetDeviceProcAddr( device, "vkDestroyBuffer" ) );
      vkCreateBufferView                 = PFN_vkCreateBufferView( vkGetDeviceProcAddr( device, "vkCreateBufferView" ) );
      vkDestroyBufferView                = PFN_vkDestroyBufferView( vkGetDeviceProcAddr( device, "vkDestroyBufferView" ) );
      vkCreateImage                      = PFN_vkCreateImage( vkGetDeviceProcAddr( device, "vkCreateImage" ) );
      vkDestroyImage                     = PFN_vkDestroyImage( vkGetDeviceProcAddr( device, "vkDestroyImage" ) );
      vkGetImageSubresourceLayout        = PFN_vkGetImageSubresourceLayout( vkGetDeviceProcAddr( device, "vkGetImageSubresourceLayout" ) );
      vkCreateImageView                  = PFN_vkCreateImageView( vkGetDeviceProcAddr( device, "vkCreateImageView" ) );
      vkDestroyImageView                 = PFN_vkDestroyImageView( vkGetDeviceProcAddr( device, "vkDestroyImageView" ) );
      vkCreateShaderModule               = PFN_vkCreateShaderModule( vkGetDeviceProcAddr( device, "vkCreateShaderModule" ) );
      vkDestroyShaderModule              = PFN_vkDestroyShaderModule( vkGetDeviceProcAddr( device, "vkDestroyShaderModule" ) );
      vkCreatePipelineCache              = PFN_vkCreatePipelineCache( vkGetDeviceProcAddr( device, "vkCreatePipelineCache" ) );
      vkDestroyPipelineCache             = PFN_vkDestroyPipelineCache( vkGetDeviceProcAddr( device, "vkDestroyPipelineCache" ) );
      vkGetPipelineCacheData             = PFN_vkGetPipelineCacheData( vkGetDeviceProcAddr( device, "vkGetPipelineCacheData" ) );
      vkMergePipelineCaches              = PFN_vkMergePipelineCaches( vkGetDeviceProcAddr( device, "vkMergePipelineCaches" ) );
      vkCreateGraphicsPipelines          = PFN_vkCreateGraphicsPipelines( vkGetDeviceProcAddr( device, "vkCreateGraphicsPipelines" ) );
      vkCreateComputePipelines           = PFN_vkCreateComputePipelines( vkGetDeviceProcAddr( device, "vkCreateComputePipelines" ) );
      vkDestroyPipeline                  = PFN_vkDestroyPipeline( vkGetDeviceProcAddr( device, "vkDestroyPipeline" ) );
      vkCreatePipelineLayout             = PFN_vkCreatePipelineLayout( vkGetDeviceProcAddr( device, "vkCreatePipelineLayout" ) );
      vkDestroyPipelineLayout            = PFN_vkDestroyPipelineLayout( vkGetDeviceProcAddr( device, "vkDestroyPipelineLayout" ) );
      vkCreateSampler                    = PFN_vkCreateSampler( vkGetDeviceProcAddr( device, "vkCreateSampler" ) );
      vkDestroySampler                   = PFN_vkDestroySampler( vkGetDeviceProcAddr( device, "vkDestroySampler" ) );
      vkCreateDescriptorSetLayout        = PFN_vkCreateDescriptorSetLayout( vkGetDeviceProcAddr( device, "vkCreateDescriptorSetLayout" ) );
      vkDestroyDescriptorSetLayout       = PFN_vkDestroyDescriptorSetLayout( vkGetDeviceProcAddr( device, "vkDestroyDescriptorSetLayout" ) );
      vkCreateDescriptorPool             = PFN_vkCreateDescriptorPool( vkGetDeviceProcAddr( device, "vkCreateDescriptorPool" ) );
      vkDestroyDescriptorPool            = PFN_vkDestroyDescriptorPool( vkGetDeviceProcAddr( device, "vkDestroyDescriptorPool" ) );
      vkResetDescriptorPool              = PFN_vkResetDescriptorPool( vkGetDeviceProcAddr( device, "vkResetDescriptorPool" ) );
      vkAllocateDescriptorSets           = PFN_vkAllocateDescriptorSets( vkGetDeviceProcAddr( device, "vkAllocateDescriptorSets" ) );
      vkFreeDescriptorSets               = PFN_vkFreeDescriptorSets( vkGetDeviceProcAddr( device, "vkFreeDescriptorSets" ) );
      vkUpdateDescriptorSets             = PFN_vkUpdateDescriptorSets( vkGetDeviceProcAddr( device, "vkUpdateDescriptorSets" ) );
      vkCreateFramebuffer                = PFN_vkCreateFramebuffer( vkGetDeviceProcAddr( device, "vkCreateFramebuffer" ) );
      vkDestroyFramebuffer               = PFN_vkDestroyFramebuffer( vkGetDeviceProcAddr( device, "vkDestroyFramebuffer" ) );
      vkCreateRenderPass                 = PFN_vkCreateRenderPass( vkGetDeviceProcAddr( device, "vkCreateRenderPass" ) );
      vkDestroyRenderPass                = PFN_vkDestroyRenderPass( vkGetDeviceProcAddr( device, "vkDestroyRenderPass" ) );
      vkGetRenderAreaGranularity         = PFN_vkGetRenderAreaGranularity( vkGetDeviceProcAddr( device, "vkGetRenderAreaGranularity" ) );
      vkCreateCommandPool                = PFN_vkCreateCommandPool( vkGetDeviceProcAddr( device, "vkCreateCommandPool" ) );
      vkDestroyCommandPool               = PFN_vkDestroyCommandPool( vkGetDeviceProcAddr( device, "vkDestroyCommandPool" ) );
      vkResetCommandPool                 = PFN_vkResetCommandPool( vkGetDeviceProcAddr( device, "vkResetCommandPool" ) );
      vkAllocateCommandBuffers           = PFN_vkAllocateCommandBuffers( vkGetDeviceProcAddr( device, "vkAllocateCommandBuffers" ) );
      vkFreeCommandBuffers               = PFN_vkFreeCommandBuffers( vkGetDeviceProcAddr( device, "vkFreeCommandBuffers" ) );
      vkBeginCommandBuffer               = PFN_vkBeginCommandBuffer( vkGetDeviceProcAddr( device, "vkBeginCommandBuffer" ) );
      vkEndCommandBuffer                 = PFN_vkEndCommandBuffer( vkGetDeviceProcAddr( device, "vkEndCommandBuffer" ) );
      vkResetCommandBuffer               = PFN_vkResetCommandBuffer( vkGetDeviceProcAddr( device, "vkResetCommandBuffer" ) );
      vkCmdBindPipeline                  = PFN_vkCmdBindPipeline( vkGetDeviceProcAddr( device, "vkCmdBindPipeline" ) );
      vkCmdSetViewport                   = PFN_vkCmdSetViewport( vkGetDeviceProcAddr( device, "vkCmdSetViewport" ) );
      vkCmdSetScissor                    = PFN_vkCmdSetScissor( vkGetDeviceProcAddr( device, "vkCmdSetScissor" ) );
      vkCmdSetLineWidth                  = PFN_vkCmdSetLineWidth( vkGetDeviceProcAddr( device, "vkCmdSetLineWidth" ) );
      vkCmdSetDepthBias                  = PFN_vkCmdSetDepthBias( vkGetDeviceProcAddr( device, "vkCmdSetDepthBias" ) );
      vkCmdSetBlendConstants             = PFN_vkCmdSetBlendConstants( vkGetDeviceProcAddr( device, "vkCmdSetBlendConstants" ) );
      vkCmdSetDepthBounds                = PFN_vkCmdSetDepthBounds( vkGetDeviceProcAddr( device, "vkCmdSetDepthBounds" ) );
      vkCmdSetStencilCompareMask         = PFN_vkCmdSetStencilCompareMask( vkGetDeviceProcAddr( device, "vkCmdSetStencilCompareMask" ) );
      vkCmdSetStencilWriteMask           = PFN_vkCmdSetStencilWriteMask( vkGetDeviceProcAddr( device, "vkCmdSetStencilWriteMask" ) );
      vkCmdSetStencilReference           = PFN_vkCmdSetStencilReference( vkGetDeviceProcAddr( device, "vkCmdSetStencilReference" ) );
      vkCmdBindDescriptorSets            = PFN_vkCmdBindDescriptorSets( vkGetDeviceProcAddr( device, "vkCmdBindDescriptorSets" ) );
      vkCmdBindIndexBuffer               = PFN_vkCmdBindIndexBuffer( vkGetDeviceProcAddr( device, "vkCmdBindIndexBuffer" ) );
      vkCmdBindVertexBuffers             = PFN_vkCmdBindVertexBuffers( vkGetDeviceProcAddr( device, "vkCmdBindVertexBuffers" ) );
      vkCmdDraw                          = PFN_vkCmdDraw( vkGetDeviceProcAddr( device, "vkCmdDraw" ) );
      vkCmdDrawIndexed                   = PFN_vkCmdDrawIndexed( vkGetDeviceProcAddr( device, "vkCmdDrawIndexed" ) );
      vkCmdDrawIndirect                  = PFN_vkCmdDrawIndirect( vkGetDeviceProcAddr( device, "vkCmdDrawIndirect" ) );
      vkCmdDrawIndexedIndirect           = PFN_vkCmdDrawIndexedIndirect( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirect" ) );
      vkCmdDispatch                      = PFN_vkCmdDispatch( vkGetDeviceProcAddr( device, "vkCmdDispatch" ) );
      vkCmdDispatchIndirect              = PFN_vkCmdDispatchIndirect( vkGetDeviceProcAddr( device, "vkCmdDispatchIndirect" ) );
      vkCmdCopyBuffer                    = PFN_vkCmdCopyBuffer( vkGetDeviceProcAddr( device, "vkCmdCopyBuffer" ) );
      vkCmdCopyImage                     = PFN_vkCmdCopyImage( vkGetDeviceProcAddr( device, "vkCmdCopyImage" ) );
      vkCmdBlitImage                     = PFN_vkCmdBlitImage( vkGetDeviceProcAddr( device, "vkCmdBlitImage" ) );
      vkCmdCopyBufferToImage             = PFN_vkCmdCopyBufferToImage( vkGetDeviceProcAddr( device, "vkCmdCopyBufferToImage" ) );
      vkCmdCopyImageToBuffer             = PFN_vkCmdCopyImageToBuffer( vkGetDeviceProcAddr( device, "vkCmdCopyImageToBuffer" ) );
      vkCmdUpdateBuffer                  = PFN_vkCmdUpdateBuffer( vkGetDeviceProcAddr( device, "vkCmdUpdateBuffer" ) );
      vkCmdFillBuffer                    = PFN_vkCmdFillBuffer( vkGetDeviceProcAddr( device, "vkCmdFillBuffer" ) );
      vkCmdClearColorImage               = PFN_vkCmdClearColorImage( vkGetDeviceProcAddr( device, "vkCmdClearColorImage" ) );
      vkCmdClearDepthStencilImage        = PFN_vkCmdClearDepthStencilImage( vkGetDeviceProcAddr( device, "vkCmdClearDepthStencilImage" ) );
      vkCmdClearAttachments              = PFN_vkCmdClearAttachments( vkGetDeviceProcAddr( device, "vkCmdClearAttachments" ) );
      vkCmdResolveImage                  = PFN_vkCmdResolveImage( vkGetDeviceProcAddr( device, "vkCmdResolveImage" ) );
      vkCmdSetEvent                      = PFN_vkCmdSetEvent( vkGetDeviceProcAddr( device, "vkCmdSetEvent" ) );
      vkCmdResetEvent                    = PFN_vkCmdResetEvent( vkGetDeviceProcAddr( device, "vkCmdResetEvent" ) );
      vkCmdWaitEvents                    = PFN_vkCmdWaitEvents( vkGetDeviceProcAddr( device, "vkCmdWaitEvents" ) );
      vkCmdPipelineBarrier               = PFN_vkCmdPipelineBarrier( vkGetDeviceProcAddr( device, "vkCmdPipelineBarrier" ) );
      vkCmdBeginQuery                    = PFN_vkCmdBeginQuery( vkGetDeviceProcAddr( device, "vkCmdBeginQuery" ) );
      vkCmdEndQuery                      = PFN_vkCmdEndQuery( vkGetDeviceProcAddr( device, "vkCmdEndQuery" ) );
      vkCmdResetQueryPool                = PFN_vkCmdResetQueryPool( vkGetDeviceProcAddr( device, "vkCmdResetQueryPool" ) );
      vkCmdWriteTimestamp                = PFN_vkCmdWriteTimestamp( vkGetDeviceProcAddr( device, "vkCmdWriteTimestamp" ) );
      vkCmdCopyQueryPoolResults          = PFN_vkCmdCopyQueryPoolResults( vkGetDeviceProcAddr( device, "vkCmdCopyQueryPoolResults" ) );
      vkCmdPushConstants                 = PFN_vkCmdPushConstants( vkGetDeviceProcAddr( device, "vkCmdPushConstants" ) );
      vkCmdBeginRenderPass               = PFN_vkCmdBeginRenderPass( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass" ) );
      vkCmdNextSubpass                   = PFN_vkCmdNextSubpass( vkGetDeviceProcAddr( device, "vkCmdNextSubpass" ) );
      vkCmdEndRenderPass                 = PFN_vkCmdEndRenderPass( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass" ) );
      vkCmdExecuteCommands               = PFN_vkCmdExecuteCommands( vkGetDeviceProcAddr( device, "vkCmdExecuteCommands" ) );

      //=== VK_VERSION_1_1 ===
      vkBindBufferMemory2                 = PFN_vkBindBufferMemory2( vkGetDeviceProcAddr( device, "vkBindBufferMemory2" ) );
      vkBindImageMemory2                  = PFN_vkBindImageMemory2( vkGetDeviceProcAddr( device, "vkBindImageMemory2" ) );
      vkGetDeviceGroupPeerMemoryFeatures  = PFN_vkGetDeviceGroupPeerMemoryFeatures( vkGetDeviceProcAddr( device, "vkGetDeviceGroupPeerMemoryFeatures" ) );
      vkCmdSetDeviceMask                  = PFN_vkCmdSetDeviceMask( vkGetDeviceProcAddr( device, "vkCmdSetDeviceMask" ) );
      vkCmdDispatchBase                   = PFN_vkCmdDispatchBase( vkGetDeviceProcAddr( device, "vkCmdDispatchBase" ) );
      vkGetImageMemoryRequirements2       = PFN_vkGetImageMemoryRequirements2( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements2" ) );
      vkGetBufferMemoryRequirements2      = PFN_vkGetBufferMemoryRequirements2( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements2" ) );
      vkGetImageSparseMemoryRequirements2 = PFN_vkGetImageSparseMemoryRequirements2( vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements2" ) );
      vkTrimCommandPool                   = PFN_vkTrimCommandPool( vkGetDeviceProcAddr( device, "vkTrimCommandPool" ) );
      vkGetDeviceQueue2                   = PFN_vkGetDeviceQueue2( vkGetDeviceProcAddr( device, "vkGetDeviceQueue2" ) );
      vkCreateSamplerYcbcrConversion      = PFN_vkCreateSamplerYcbcrConversion( vkGetDeviceProcAddr( device, "vkCreateSamplerYcbcrConversion" ) );
      vkDestroySamplerYcbcrConversion     = PFN_vkDestroySamplerYcbcrConversion( vkGetDeviceProcAddr( device, "vkDestroySamplerYcbcrConversion" ) );
      vkCreateDescriptorUpdateTemplate    = PFN_vkCreateDescriptorUpdateTemplate( vkGetDeviceProcAddr( device, "vkCreateDescriptorUpdateTemplate" ) );
      vkDestroyDescriptorUpdateTemplate   = PFN_vkDestroyDescriptorUpdateTemplate( vkGetDeviceProcAddr( device, "vkDestroyDescriptorUpdateTemplate" ) );
      vkUpdateDescriptorSetWithTemplate   = PFN_vkUpdateDescriptorSetWithTemplate( vkGetDeviceProcAddr( device, "vkUpdateDescriptorSetWithTemplate" ) );
      vkGetDescriptorSetLayoutSupport     = PFN_vkGetDescriptorSetLayoutSupport( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutSupport" ) );

      //=== VK_VERSION_1_2 ===
      vkCmdDrawIndirectCount          = PFN_vkCmdDrawIndirectCount( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCount" ) );
      vkCmdDrawIndexedIndirectCount   = PFN_vkCmdDrawIndexedIndirectCount( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCount" ) );
      vkCreateRenderPass2             = PFN_vkCreateRenderPass2( vkGetDeviceProcAddr( device, "vkCreateRenderPass2" ) );
      vkCmdBeginRenderPass2           = PFN_vkCmdBeginRenderPass2( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass2" ) );
      vkCmdNextSubpass2               = PFN_vkCmdNextSubpass2( vkGetDeviceProcAddr( device, "vkCmdNextSubpass2" ) );
      vkCmdEndRenderPass2             = PFN_vkCmdEndRenderPass2( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass2" ) );
      vkResetQueryPool                = PFN_vkResetQueryPool( vkGetDeviceProcAddr( device, "vkResetQueryPool" ) );
      vkGetSemaphoreCounterValue      = PFN_vkGetSemaphoreCounterValue( vkGetDeviceProcAddr( device, "vkGetSemaphoreCounterValue" ) );
      vkWaitSemaphores                = PFN_vkWaitSemaphores( vkGetDeviceProcAddr( device, "vkWaitSemaphores" ) );
      vkSignalSemaphore               = PFN_vkSignalSemaphore( vkGetDeviceProcAddr( device, "vkSignalSemaphore" ) );
      vkGetBufferDeviceAddress        = PFN_vkGetBufferDeviceAddress( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddress" ) );
      vkGetBufferOpaqueCaptureAddress = PFN_vkGetBufferOpaqueCaptureAddress( vkGetDeviceProcAddr( device, "vkGetBufferOpaqueCaptureAddress" ) );
      vkGetDeviceMemoryOpaqueCaptureAddress =
        PFN_vkGetDeviceMemoryOpaqueCaptureAddress( vkGetDeviceProcAddr( device, "vkGetDeviceMemoryOpaqueCaptureAddress" ) );

      //=== VK_VERSION_1_3 ===
      vkCreatePrivateDataSlot             = PFN_vkCreatePrivateDataSlot( vkGetDeviceProcAddr( device, "vkCreatePrivateDataSlot" ) );
      vkDestroyPrivateDataSlot            = PFN_vkDestroyPrivateDataSlot( vkGetDeviceProcAddr( device, "vkDestroyPrivateDataSlot" ) );
      vkSetPrivateData                    = PFN_vkSetPrivateData( vkGetDeviceProcAddr( device, "vkSetPrivateData" ) );
      vkGetPrivateData                    = PFN_vkGetPrivateData( vkGetDeviceProcAddr( device, "vkGetPrivateData" ) );
      vkCmdSetEvent2                      = PFN_vkCmdSetEvent2( vkGetDeviceProcAddr( device, "vkCmdSetEvent2" ) );
      vkCmdResetEvent2                    = PFN_vkCmdResetEvent2( vkGetDeviceProcAddr( device, "vkCmdResetEvent2" ) );
      vkCmdWaitEvents2                    = PFN_vkCmdWaitEvents2( vkGetDeviceProcAddr( device, "vkCmdWaitEvents2" ) );
      vkCmdPipelineBarrier2               = PFN_vkCmdPipelineBarrier2( vkGetDeviceProcAddr( device, "vkCmdPipelineBarrier2" ) );
      vkCmdWriteTimestamp2                = PFN_vkCmdWriteTimestamp2( vkGetDeviceProcAddr( device, "vkCmdWriteTimestamp2" ) );
      vkQueueSubmit2                      = PFN_vkQueueSubmit2( vkGetDeviceProcAddr( device, "vkQueueSubmit2" ) );
      vkCmdCopyBuffer2                    = PFN_vkCmdCopyBuffer2( vkGetDeviceProcAddr( device, "vkCmdCopyBuffer2" ) );
      vkCmdCopyImage2                     = PFN_vkCmdCopyImage2( vkGetDeviceProcAddr( device, "vkCmdCopyImage2" ) );
      vkCmdCopyBufferToImage2             = PFN_vkCmdCopyBufferToImage2( vkGetDeviceProcAddr( device, "vkCmdCopyBufferToImage2" ) );
      vkCmdCopyImageToBuffer2             = PFN_vkCmdCopyImageToBuffer2( vkGetDeviceProcAddr( device, "vkCmdCopyImageToBuffer2" ) );
      vkCmdBlitImage2                     = PFN_vkCmdBlitImage2( vkGetDeviceProcAddr( device, "vkCmdBlitImage2" ) );
      vkCmdResolveImage2                  = PFN_vkCmdResolveImage2( vkGetDeviceProcAddr( device, "vkCmdResolveImage2" ) );
      vkCmdBeginRendering                 = PFN_vkCmdBeginRendering( vkGetDeviceProcAddr( device, "vkCmdBeginRendering" ) );
      vkCmdEndRendering                   = PFN_vkCmdEndRendering( vkGetDeviceProcAddr( device, "vkCmdEndRendering" ) );
      vkCmdSetCullMode                    = PFN_vkCmdSetCullMode( vkGetDeviceProcAddr( device, "vkCmdSetCullMode" ) );
      vkCmdSetFrontFace                   = PFN_vkCmdSetFrontFace( vkGetDeviceProcAddr( device, "vkCmdSetFrontFace" ) );
      vkCmdSetPrimitiveTopology           = PFN_vkCmdSetPrimitiveTopology( vkGetDeviceProcAddr( device, "vkCmdSetPrimitiveTopology" ) );
      vkCmdSetViewportWithCount           = PFN_vkCmdSetViewportWithCount( vkGetDeviceProcAddr( device, "vkCmdSetViewportWithCount" ) );
      vkCmdSetScissorWithCount            = PFN_vkCmdSetScissorWithCount( vkGetDeviceProcAddr( device, "vkCmdSetScissorWithCount" ) );
      vkCmdBindVertexBuffers2             = PFN_vkCmdBindVertexBuffers2( vkGetDeviceProcAddr( device, "vkCmdBindVertexBuffers2" ) );
      vkCmdSetDepthTestEnable             = PFN_vkCmdSetDepthTestEnable( vkGetDeviceProcAddr( device, "vkCmdSetDepthTestEnable" ) );
      vkCmdSetDepthWriteEnable            = PFN_vkCmdSetDepthWriteEnable( vkGetDeviceProcAddr( device, "vkCmdSetDepthWriteEnable" ) );
      vkCmdSetDepthCompareOp              = PFN_vkCmdSetDepthCompareOp( vkGetDeviceProcAddr( device, "vkCmdSetDepthCompareOp" ) );
      vkCmdSetDepthBoundsTestEnable       = PFN_vkCmdSetDepthBoundsTestEnable( vkGetDeviceProcAddr( device, "vkCmdSetDepthBoundsTestEnable" ) );
      vkCmdSetStencilTestEnable           = PFN_vkCmdSetStencilTestEnable( vkGetDeviceProcAddr( device, "vkCmdSetStencilTestEnable" ) );
      vkCmdSetStencilOp                   = PFN_vkCmdSetStencilOp( vkGetDeviceProcAddr( device, "vkCmdSetStencilOp" ) );
      vkCmdSetRasterizerDiscardEnable     = PFN_vkCmdSetRasterizerDiscardEnable( vkGetDeviceProcAddr( device, "vkCmdSetRasterizerDiscardEnable" ) );
      vkCmdSetDepthBiasEnable             = PFN_vkCmdSetDepthBiasEnable( vkGetDeviceProcAddr( device, "vkCmdSetDepthBiasEnable" ) );
      vkCmdSetPrimitiveRestartEnable      = PFN_vkCmdSetPrimitiveRestartEnable( vkGetDeviceProcAddr( device, "vkCmdSetPrimitiveRestartEnable" ) );
      vkGetDeviceBufferMemoryRequirements = PFN_vkGetDeviceBufferMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetDeviceBufferMemoryRequirements" ) );
      vkGetDeviceImageMemoryRequirements  = PFN_vkGetDeviceImageMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetDeviceImageMemoryRequirements" ) );
      vkGetDeviceImageSparseMemoryRequirements =
        PFN_vkGetDeviceImageSparseMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetDeviceImageSparseMemoryRequirements" ) );

      //=== VK_KHR_swapchain ===
      vkCreateSwapchainKHR    = PFN_vkCreateSwapchainKHR( vkGetDeviceProcAddr( device, "vkCreateSwapchainKHR" ) );
      vkDestroySwapchainKHR   = PFN_vkDestroySwapchainKHR( vkGetDeviceProcAddr( device, "vkDestroySwapchainKHR" ) );
      vkGetSwapchainImagesKHR = PFN_vkGetSwapchainImagesKHR( vkGetDeviceProcAddr( device, "vkGetSwapchainImagesKHR" ) );
      vkAcquireNextImageKHR   = PFN_vkAcquireNextImageKHR( vkGetDeviceProcAddr( device, "vkAcquireNextImageKHR" ) );
      vkQueuePresentKHR       = PFN_vkQueuePresentKHR( vkGetDeviceProcAddr( device, "vkQueuePresentKHR" ) );
      vkGetDeviceGroupPresentCapabilitiesKHR =
        PFN_vkGetDeviceGroupPresentCapabilitiesKHR( vkGetDeviceProcAddr( device, "vkGetDeviceGroupPresentCapabilitiesKHR" ) );
      vkGetDeviceGroupSurfacePresentModesKHR =
        PFN_vkGetDeviceGroupSurfacePresentModesKHR( vkGetDeviceProcAddr( device, "vkGetDeviceGroupSurfacePresentModesKHR" ) );
      vkAcquireNextImage2KHR = PFN_vkAcquireNextImage2KHR( vkGetDeviceProcAddr( device, "vkAcquireNextImage2KHR" ) );

      //=== VK_KHR_display_swapchain ===
      vkCreateSharedSwapchainsKHR = PFN_vkCreateSharedSwapchainsKHR( vkGetDeviceProcAddr( device, "vkCreateSharedSwapchainsKHR" ) );

      //=== VK_EXT_debug_marker ===
      vkDebugMarkerSetObjectTagEXT  = PFN_vkDebugMarkerSetObjectTagEXT( vkGetDeviceProcAddr( device, "vkDebugMarkerSetObjectTagEXT" ) );
      vkDebugMarkerSetObjectNameEXT = PFN_vkDebugMarkerSetObjectNameEXT( vkGetDeviceProcAddr( device, "vkDebugMarkerSetObjectNameEXT" ) );
      vkCmdDebugMarkerBeginEXT      = PFN_vkCmdDebugMarkerBeginEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerBeginEXT" ) );
      vkCmdDebugMarkerEndEXT        = PFN_vkCmdDebugMarkerEndEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerEndEXT" ) );
      vkCmdDebugMarkerInsertEXT     = PFN_vkCmdDebugMarkerInsertEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerInsertEXT" ) );

      //=== VK_KHR_video_queue ===
      vkCreateVideoSessionKHR  = PFN_vkCreateVideoSessionKHR( vkGetDeviceProcAddr( device, "vkCreateVideoSessionKHR" ) );
      vkDestroyVideoSessionKHR = PFN_vkDestroyVideoSessionKHR( vkGetDeviceProcAddr( device, "vkDestroyVideoSessionKHR" ) );
      vkGetVideoSessionMemoryRequirementsKHR =
        PFN_vkGetVideoSessionMemoryRequirementsKHR( vkGetDeviceProcAddr( device, "vkGetVideoSessionMemoryRequirementsKHR" ) );
      vkBindVideoSessionMemoryKHR        = PFN_vkBindVideoSessionMemoryKHR( vkGetDeviceProcAddr( device, "vkBindVideoSessionMemoryKHR" ) );
      vkCreateVideoSessionParametersKHR  = PFN_vkCreateVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkCreateVideoSessionParametersKHR" ) );
      vkUpdateVideoSessionParametersKHR  = PFN_vkUpdateVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkUpdateVideoSessionParametersKHR" ) );
      vkDestroyVideoSessionParametersKHR = PFN_vkDestroyVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkDestroyVideoSessionParametersKHR" ) );
      vkCmdBeginVideoCodingKHR           = PFN_vkCmdBeginVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdBeginVideoCodingKHR" ) );
      vkCmdEndVideoCodingKHR             = PFN_vkCmdEndVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdEndVideoCodingKHR" ) );
      vkCmdControlVideoCodingKHR         = PFN_vkCmdControlVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdControlVideoCodingKHR" ) );

      //=== VK_KHR_video_decode_queue ===
      vkCmdDecodeVideoKHR = PFN_vkCmdDecodeVideoKHR( vkGetDeviceProcAddr( device, "vkCmdDecodeVideoKHR" ) );

      //=== VK_EXT_transform_feedback ===
      vkCmdBindTransformFeedbackBuffersEXT = PFN_vkCmdBindTransformFeedbackBuffersEXT( vkGetDeviceProcAddr( device, "vkCmdBindTransformFeedbackBuffersEXT" ) );
      vkCmdBeginTransformFeedbackEXT       = PFN_vkCmdBeginTransformFeedbackEXT( vkGetDeviceProcAddr( device, "vkCmdBeginTransformFeedbackEXT" ) );
      vkCmdEndTransformFeedbackEXT         = PFN_vkCmdEndTransformFeedbackEXT( vkGetDeviceProcAddr( device, "vkCmdEndTransformFeedbackEXT" ) );
      vkCmdBeginQueryIndexedEXT            = PFN_vkCmdBeginQueryIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdBeginQueryIndexedEXT" ) );
      vkCmdEndQueryIndexedEXT              = PFN_vkCmdEndQueryIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdEndQueryIndexedEXT" ) );
      vkCmdDrawIndirectByteCountEXT        = PFN_vkCmdDrawIndirectByteCountEXT( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectByteCountEXT" ) );

      //=== VK_NVX_binary_import ===
      vkCreateCuModuleNVX    = PFN_vkCreateCuModuleNVX( vkGetDeviceProcAddr( device, "vkCreateCuModuleNVX" ) );
      vkCreateCuFunctionNVX  = PFN_vkCreateCuFunctionNVX( vkGetDeviceProcAddr( device, "vkCreateCuFunctionNVX" ) );
      vkDestroyCuModuleNVX   = PFN_vkDestroyCuModuleNVX( vkGetDeviceProcAddr( device, "vkDestroyCuModuleNVX" ) );
      vkDestroyCuFunctionNVX = PFN_vkDestroyCuFunctionNVX( vkGetDeviceProcAddr( device, "vkDestroyCuFunctionNVX" ) );
      vkCmdCuLaunchKernelNVX = PFN_vkCmdCuLaunchKernelNVX( vkGetDeviceProcAddr( device, "vkCmdCuLaunchKernelNVX" ) );

      //=== VK_NVX_image_view_handle ===
      vkGetImageViewHandleNVX  = PFN_vkGetImageViewHandleNVX( vkGetDeviceProcAddr( device, "vkGetImageViewHandleNVX" ) );
      vkGetImageViewAddressNVX = PFN_vkGetImageViewAddressNVX( vkGetDeviceProcAddr( device, "vkGetImageViewAddressNVX" ) );

      //=== VK_AMD_draw_indirect_count ===
      vkCmdDrawIndirectCountAMD = PFN_vkCmdDrawIndirectCountAMD( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCountAMD" ) );
      if ( !vkCmdDrawIndirectCount )
        vkCmdDrawIndirectCount = vkCmdDrawIndirectCountAMD;
      vkCmdDrawIndexedIndirectCountAMD = PFN_vkCmdDrawIndexedIndirectCountAMD( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCountAMD" ) );
      if ( !vkCmdDrawIndexedIndirectCount )
        vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountAMD;

      //=== VK_AMD_shader_info ===
      vkGetShaderInfoAMD = PFN_vkGetShaderInfoAMD( vkGetDeviceProcAddr( device, "vkGetShaderInfoAMD" ) );

      //=== VK_KHR_dynamic_rendering ===
      vkCmdBeginRenderingKHR = PFN_vkCmdBeginRenderingKHR( vkGetDeviceProcAddr( device, "vkCmdBeginRenderingKHR" ) );
      if ( !vkCmdBeginRendering )
        vkCmdBeginRendering = vkCmdBeginRenderingKHR;
      vkCmdEndRenderingKHR = PFN_vkCmdEndRenderingKHR( vkGetDeviceProcAddr( device, "vkCmdEndRenderingKHR" ) );
      if ( !vkCmdEndRendering )
        vkCmdEndRendering = vkCmdEndRenderingKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_NV_external_memory_win32 ===
      vkGetMemoryWin32HandleNV = PFN_vkGetMemoryWin32HandleNV( vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandleNV" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_device_group ===
      vkGetDeviceGroupPeerMemoryFeaturesKHR =
        PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR( vkGetDeviceProcAddr( device, "vkGetDeviceGroupPeerMemoryFeaturesKHR" ) );
      if ( !vkGetDeviceGroupPeerMemoryFeatures )
        vkGetDeviceGroupPeerMemoryFeatures = vkGetDeviceGroupPeerMemoryFeaturesKHR;
      vkCmdSetDeviceMaskKHR = PFN_vkCmdSetDeviceMaskKHR( vkGetDeviceProcAddr( device, "vkCmdSetDeviceMaskKHR" ) );
      if ( !vkCmdSetDeviceMask )
        vkCmdSetDeviceMask = vkCmdSetDeviceMaskKHR;
      vkCmdDispatchBaseKHR = PFN_vkCmdDispatchBaseKHR( vkGetDeviceProcAddr( device, "vkCmdDispatchBaseKHR" ) );
      if ( !vkCmdDispatchBase )
        vkCmdDispatchBase = vkCmdDispatchBaseKHR;

      //=== VK_KHR_maintenance1 ===
      vkTrimCommandPoolKHR = PFN_vkTrimCommandPoolKHR( vkGetDeviceProcAddr( device, "vkTrimCommandPoolKHR" ) );
      if ( !vkTrimCommandPool )
        vkTrimCommandPool = vkTrimCommandPoolKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_memory_win32 ===
      vkGetMemoryWin32HandleKHR           = PFN_vkGetMemoryWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandleKHR" ) );
      vkGetMemoryWin32HandlePropertiesKHR = PFN_vkGetMemoryWin32HandlePropertiesKHR( vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandlePropertiesKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_memory_fd ===
      vkGetMemoryFdKHR           = PFN_vkGetMemoryFdKHR( vkGetDeviceProcAddr( device, "vkGetMemoryFdKHR" ) );
      vkGetMemoryFdPropertiesKHR = PFN_vkGetMemoryFdPropertiesKHR( vkGetDeviceProcAddr( device, "vkGetMemoryFdPropertiesKHR" ) );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_semaphore_win32 ===
      vkImportSemaphoreWin32HandleKHR = PFN_vkImportSemaphoreWin32HandleKHR( vkGetDeviceProcAddr( device, "vkImportSemaphoreWin32HandleKHR" ) );
      vkGetSemaphoreWin32HandleKHR    = PFN_vkGetSemaphoreWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_semaphore_fd ===
      vkImportSemaphoreFdKHR = PFN_vkImportSemaphoreFdKHR( vkGetDeviceProcAddr( device, "vkImportSemaphoreFdKHR" ) );
      vkGetSemaphoreFdKHR    = PFN_vkGetSemaphoreFdKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreFdKHR" ) );

      //=== VK_KHR_push_descriptor ===
      vkCmdPushDescriptorSetKHR = PFN_vkCmdPushDescriptorSetKHR( vkGetDeviceProcAddr( device, "vkCmdPushDescriptorSetKHR" ) );
      vkCmdPushDescriptorSetWithTemplateKHR =
        PFN_vkCmdPushDescriptorSetWithTemplateKHR( vkGetDeviceProcAddr( device, "vkCmdPushDescriptorSetWithTemplateKHR" ) );

      //=== VK_EXT_conditional_rendering ===
      vkCmdBeginConditionalRenderingEXT = PFN_vkCmdBeginConditionalRenderingEXT( vkGetDeviceProcAddr( device, "vkCmdBeginConditionalRenderingEXT" ) );
      vkCmdEndConditionalRenderingEXT   = PFN_vkCmdEndConditionalRenderingEXT( vkGetDeviceProcAddr( device, "vkCmdEndConditionalRenderingEXT" ) );

      //=== VK_KHR_descriptor_update_template ===
      vkCreateDescriptorUpdateTemplateKHR = PFN_vkCreateDescriptorUpdateTemplateKHR( vkGetDeviceProcAddr( device, "vkCreateDescriptorUpdateTemplateKHR" ) );
      if ( !vkCreateDescriptorUpdateTemplate )
        vkCreateDescriptorUpdateTemplate = vkCreateDescriptorUpdateTemplateKHR;
      vkDestroyDescriptorUpdateTemplateKHR = PFN_vkDestroyDescriptorUpdateTemplateKHR( vkGetDeviceProcAddr( device, "vkDestroyDescriptorUpdateTemplateKHR" ) );
      if ( !vkDestroyDescriptorUpdateTemplate )
        vkDestroyDescriptorUpdateTemplate = vkDestroyDescriptorUpdateTemplateKHR;
      vkUpdateDescriptorSetWithTemplateKHR = PFN_vkUpdateDescriptorSetWithTemplateKHR( vkGetDeviceProcAddr( device, "vkUpdateDescriptorSetWithTemplateKHR" ) );
      if ( !vkUpdateDescriptorSetWithTemplate )
        vkUpdateDescriptorSetWithTemplate = vkUpdateDescriptorSetWithTemplateKHR;

      //=== VK_NV_clip_space_w_scaling ===
      vkCmdSetViewportWScalingNV = PFN_vkCmdSetViewportWScalingNV( vkGetDeviceProcAddr( device, "vkCmdSetViewportWScalingNV" ) );

      //=== VK_EXT_display_control ===
      vkDisplayPowerControlEXT  = PFN_vkDisplayPowerControlEXT( vkGetDeviceProcAddr( device, "vkDisplayPowerControlEXT" ) );
      vkRegisterDeviceEventEXT  = PFN_vkRegisterDeviceEventEXT( vkGetDeviceProcAddr( device, "vkRegisterDeviceEventEXT" ) );
      vkRegisterDisplayEventEXT = PFN_vkRegisterDisplayEventEXT( vkGetDeviceProcAddr( device, "vkRegisterDisplayEventEXT" ) );
      vkGetSwapchainCounterEXT  = PFN_vkGetSwapchainCounterEXT( vkGetDeviceProcAddr( device, "vkGetSwapchainCounterEXT" ) );

      //=== VK_GOOGLE_display_timing ===
      vkGetRefreshCycleDurationGOOGLE   = PFN_vkGetRefreshCycleDurationGOOGLE( vkGetDeviceProcAddr( device, "vkGetRefreshCycleDurationGOOGLE" ) );
      vkGetPastPresentationTimingGOOGLE = PFN_vkGetPastPresentationTimingGOOGLE( vkGetDeviceProcAddr( device, "vkGetPastPresentationTimingGOOGLE" ) );

      //=== VK_EXT_discard_rectangles ===
      vkCmdSetDiscardRectangleEXT       = PFN_vkCmdSetDiscardRectangleEXT( vkGetDeviceProcAddr( device, "vkCmdSetDiscardRectangleEXT" ) );
      vkCmdSetDiscardRectangleEnableEXT = PFN_vkCmdSetDiscardRectangleEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDiscardRectangleEnableEXT" ) );
      vkCmdSetDiscardRectangleModeEXT   = PFN_vkCmdSetDiscardRectangleModeEXT( vkGetDeviceProcAddr( device, "vkCmdSetDiscardRectangleModeEXT" ) );

      //=== VK_EXT_hdr_metadata ===
      vkSetHdrMetadataEXT = PFN_vkSetHdrMetadataEXT( vkGetDeviceProcAddr( device, "vkSetHdrMetadataEXT" ) );

      //=== VK_KHR_create_renderpass2 ===
      vkCreateRenderPass2KHR = PFN_vkCreateRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCreateRenderPass2KHR" ) );
      if ( !vkCreateRenderPass2 )
        vkCreateRenderPass2 = vkCreateRenderPass2KHR;
      vkCmdBeginRenderPass2KHR = PFN_vkCmdBeginRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass2KHR" ) );
      if ( !vkCmdBeginRenderPass2 )
        vkCmdBeginRenderPass2 = vkCmdBeginRenderPass2KHR;
      vkCmdNextSubpass2KHR = PFN_vkCmdNextSubpass2KHR( vkGetDeviceProcAddr( device, "vkCmdNextSubpass2KHR" ) );
      if ( !vkCmdNextSubpass2 )
        vkCmdNextSubpass2 = vkCmdNextSubpass2KHR;
      vkCmdEndRenderPass2KHR = PFN_vkCmdEndRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass2KHR" ) );
      if ( !vkCmdEndRenderPass2 )
        vkCmdEndRenderPass2 = vkCmdEndRenderPass2KHR;

      //=== VK_KHR_shared_presentable_image ===
      vkGetSwapchainStatusKHR = PFN_vkGetSwapchainStatusKHR( vkGetDeviceProcAddr( device, "vkGetSwapchainStatusKHR" ) );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_fence_win32 ===
      vkImportFenceWin32HandleKHR = PFN_vkImportFenceWin32HandleKHR( vkGetDeviceProcAddr( device, "vkImportFenceWin32HandleKHR" ) );
      vkGetFenceWin32HandleKHR    = PFN_vkGetFenceWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetFenceWin32HandleKHR" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_fence_fd ===
      vkImportFenceFdKHR = PFN_vkImportFenceFdKHR( vkGetDeviceProcAddr( device, "vkImportFenceFdKHR" ) );
      vkGetFenceFdKHR    = PFN_vkGetFenceFdKHR( vkGetDeviceProcAddr( device, "vkGetFenceFdKHR" ) );

      //=== VK_KHR_performance_query ===
      vkAcquireProfilingLockKHR = PFN_vkAcquireProfilingLockKHR( vkGetDeviceProcAddr( device, "vkAcquireProfilingLockKHR" ) );
      vkReleaseProfilingLockKHR = PFN_vkReleaseProfilingLockKHR( vkGetDeviceProcAddr( device, "vkReleaseProfilingLockKHR" ) );

      //=== VK_EXT_debug_utils ===
      vkSetDebugUtilsObjectNameEXT    = PFN_vkSetDebugUtilsObjectNameEXT( vkGetDeviceProcAddr( device, "vkSetDebugUtilsObjectNameEXT" ) );
      vkSetDebugUtilsObjectTagEXT     = PFN_vkSetDebugUtilsObjectTagEXT( vkGetDeviceProcAddr( device, "vkSetDebugUtilsObjectTagEXT" ) );
      vkQueueBeginDebugUtilsLabelEXT  = PFN_vkQueueBeginDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueBeginDebugUtilsLabelEXT" ) );
      vkQueueEndDebugUtilsLabelEXT    = PFN_vkQueueEndDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueEndDebugUtilsLabelEXT" ) );
      vkQueueInsertDebugUtilsLabelEXT = PFN_vkQueueInsertDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueInsertDebugUtilsLabelEXT" ) );
      vkCmdBeginDebugUtilsLabelEXT    = PFN_vkCmdBeginDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdBeginDebugUtilsLabelEXT" ) );
      vkCmdEndDebugUtilsLabelEXT      = PFN_vkCmdEndDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdEndDebugUtilsLabelEXT" ) );
      vkCmdInsertDebugUtilsLabelEXT   = PFN_vkCmdInsertDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdInsertDebugUtilsLabelEXT" ) );

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      //=== VK_ANDROID_external_memory_android_hardware_buffer ===
      vkGetAndroidHardwareBufferPropertiesANDROID =
        PFN_vkGetAndroidHardwareBufferPropertiesANDROID( vkGetDeviceProcAddr( device, "vkGetAndroidHardwareBufferPropertiesANDROID" ) );
      vkGetMemoryAndroidHardwareBufferANDROID =
        PFN_vkGetMemoryAndroidHardwareBufferANDROID( vkGetDeviceProcAddr( device, "vkGetMemoryAndroidHardwareBufferANDROID" ) );
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_AMDX_shader_enqueue ===
      vkCreateExecutionGraphPipelinesAMDX = PFN_vkCreateExecutionGraphPipelinesAMDX( vkGetDeviceProcAddr( device, "vkCreateExecutionGraphPipelinesAMDX" ) );
      vkGetExecutionGraphPipelineScratchSizeAMDX =
        PFN_vkGetExecutionGraphPipelineScratchSizeAMDX( vkGetDeviceProcAddr( device, "vkGetExecutionGraphPipelineScratchSizeAMDX" ) );
      vkGetExecutionGraphPipelineNodeIndexAMDX =
        PFN_vkGetExecutionGraphPipelineNodeIndexAMDX( vkGetDeviceProcAddr( device, "vkGetExecutionGraphPipelineNodeIndexAMDX" ) );
      vkCmdInitializeGraphScratchMemoryAMDX =
        PFN_vkCmdInitializeGraphScratchMemoryAMDX( vkGetDeviceProcAddr( device, "vkCmdInitializeGraphScratchMemoryAMDX" ) );
      vkCmdDispatchGraphAMDX              = PFN_vkCmdDispatchGraphAMDX( vkGetDeviceProcAddr( device, "vkCmdDispatchGraphAMDX" ) );
      vkCmdDispatchGraphIndirectAMDX      = PFN_vkCmdDispatchGraphIndirectAMDX( vkGetDeviceProcAddr( device, "vkCmdDispatchGraphIndirectAMDX" ) );
      vkCmdDispatchGraphIndirectCountAMDX = PFN_vkCmdDispatchGraphIndirectCountAMDX( vkGetDeviceProcAddr( device, "vkCmdDispatchGraphIndirectCountAMDX" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

      //=== VK_EXT_sample_locations ===
      vkCmdSetSampleLocationsEXT = PFN_vkCmdSetSampleLocationsEXT( vkGetDeviceProcAddr( device, "vkCmdSetSampleLocationsEXT" ) );

      //=== VK_KHR_get_memory_requirements2 ===
      vkGetImageMemoryRequirements2KHR = PFN_vkGetImageMemoryRequirements2KHR( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements2KHR" ) );
      if ( !vkGetImageMemoryRequirements2 )
        vkGetImageMemoryRequirements2 = vkGetImageMemoryRequirements2KHR;
      vkGetBufferMemoryRequirements2KHR = PFN_vkGetBufferMemoryRequirements2KHR( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements2KHR" ) );
      if ( !vkGetBufferMemoryRequirements2 )
        vkGetBufferMemoryRequirements2 = vkGetBufferMemoryRequirements2KHR;
      vkGetImageSparseMemoryRequirements2KHR =
        PFN_vkGetImageSparseMemoryRequirements2KHR( vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements2KHR" ) );
      if ( !vkGetImageSparseMemoryRequirements2 )
        vkGetImageSparseMemoryRequirements2 = vkGetImageSparseMemoryRequirements2KHR;

      //=== VK_KHR_acceleration_structure ===
      vkCreateAccelerationStructureKHR    = PFN_vkCreateAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCreateAccelerationStructureKHR" ) );
      vkDestroyAccelerationStructureKHR   = PFN_vkDestroyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkDestroyAccelerationStructureKHR" ) );
      vkCmdBuildAccelerationStructuresKHR = PFN_vkCmdBuildAccelerationStructuresKHR( vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructuresKHR" ) );
      vkCmdBuildAccelerationStructuresIndirectKHR =
        PFN_vkCmdBuildAccelerationStructuresIndirectKHR( vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructuresIndirectKHR" ) );
      vkBuildAccelerationStructuresKHR = PFN_vkBuildAccelerationStructuresKHR( vkGetDeviceProcAddr( device, "vkBuildAccelerationStructuresKHR" ) );
      vkCopyAccelerationStructureKHR   = PFN_vkCopyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCopyAccelerationStructureKHR" ) );
      vkCopyAccelerationStructureToMemoryKHR =
        PFN_vkCopyAccelerationStructureToMemoryKHR( vkGetDeviceProcAddr( device, "vkCopyAccelerationStructureToMemoryKHR" ) );
      vkCopyMemoryToAccelerationStructureKHR =
        PFN_vkCopyMemoryToAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCopyMemoryToAccelerationStructureKHR" ) );
      vkWriteAccelerationStructuresPropertiesKHR =
        PFN_vkWriteAccelerationStructuresPropertiesKHR( vkGetDeviceProcAddr( device, "vkWriteAccelerationStructuresPropertiesKHR" ) );
      vkCmdCopyAccelerationStructureKHR = PFN_vkCmdCopyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureKHR" ) );
      vkCmdCopyAccelerationStructureToMemoryKHR =
        PFN_vkCmdCopyAccelerationStructureToMemoryKHR( vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureToMemoryKHR" ) );
      vkCmdCopyMemoryToAccelerationStructureKHR =
        PFN_vkCmdCopyMemoryToAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCmdCopyMemoryToAccelerationStructureKHR" ) );
      vkGetAccelerationStructureDeviceAddressKHR =
        PFN_vkGetAccelerationStructureDeviceAddressKHR( vkGetDeviceProcAddr( device, "vkGetAccelerationStructureDeviceAddressKHR" ) );
      vkCmdWriteAccelerationStructuresPropertiesKHR =
        PFN_vkCmdWriteAccelerationStructuresPropertiesKHR( vkGetDeviceProcAddr( device, "vkCmdWriteAccelerationStructuresPropertiesKHR" ) );
      vkGetDeviceAccelerationStructureCompatibilityKHR =
        PFN_vkGetDeviceAccelerationStructureCompatibilityKHR( vkGetDeviceProcAddr( device, "vkGetDeviceAccelerationStructureCompatibilityKHR" ) );
      vkGetAccelerationStructureBuildSizesKHR =
        PFN_vkGetAccelerationStructureBuildSizesKHR( vkGetDeviceProcAddr( device, "vkGetAccelerationStructureBuildSizesKHR" ) );

      //=== VK_KHR_ray_tracing_pipeline ===
      vkCmdTraceRaysKHR                    = PFN_vkCmdTraceRaysKHR( vkGetDeviceProcAddr( device, "vkCmdTraceRaysKHR" ) );
      vkCreateRayTracingPipelinesKHR       = PFN_vkCreateRayTracingPipelinesKHR( vkGetDeviceProcAddr( device, "vkCreateRayTracingPipelinesKHR" ) );
      vkGetRayTracingShaderGroupHandlesKHR = PFN_vkGetRayTracingShaderGroupHandlesKHR( vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupHandlesKHR" ) );
      vkGetRayTracingCaptureReplayShaderGroupHandlesKHR =
        PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR( vkGetDeviceProcAddr( device, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR" ) );
      vkCmdTraceRaysIndirectKHR = PFN_vkCmdTraceRaysIndirectKHR( vkGetDeviceProcAddr( device, "vkCmdTraceRaysIndirectKHR" ) );
      vkGetRayTracingShaderGroupStackSizeKHR =
        PFN_vkGetRayTracingShaderGroupStackSizeKHR( vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupStackSizeKHR" ) );
      vkCmdSetRayTracingPipelineStackSizeKHR =
        PFN_vkCmdSetRayTracingPipelineStackSizeKHR( vkGetDeviceProcAddr( device, "vkCmdSetRayTracingPipelineStackSizeKHR" ) );

      //=== VK_KHR_sampler_ycbcr_conversion ===
      vkCreateSamplerYcbcrConversionKHR = PFN_vkCreateSamplerYcbcrConversionKHR( vkGetDeviceProcAddr( device, "vkCreateSamplerYcbcrConversionKHR" ) );
      if ( !vkCreateSamplerYcbcrConversion )
        vkCreateSamplerYcbcrConversion = vkCreateSamplerYcbcrConversionKHR;
      vkDestroySamplerYcbcrConversionKHR = PFN_vkDestroySamplerYcbcrConversionKHR( vkGetDeviceProcAddr( device, "vkDestroySamplerYcbcrConversionKHR" ) );
      if ( !vkDestroySamplerYcbcrConversion )
        vkDestroySamplerYcbcrConversion = vkDestroySamplerYcbcrConversionKHR;

      //=== VK_KHR_bind_memory2 ===
      vkBindBufferMemory2KHR = PFN_vkBindBufferMemory2KHR( vkGetDeviceProcAddr( device, "vkBindBufferMemory2KHR" ) );
      if ( !vkBindBufferMemory2 )
        vkBindBufferMemory2 = vkBindBufferMemory2KHR;
      vkBindImageMemory2KHR = PFN_vkBindImageMemory2KHR( vkGetDeviceProcAddr( device, "vkBindImageMemory2KHR" ) );
      if ( !vkBindImageMemory2 )
        vkBindImageMemory2 = vkBindImageMemory2KHR;

      //=== VK_EXT_image_drm_format_modifier ===
      vkGetImageDrmFormatModifierPropertiesEXT =
        PFN_vkGetImageDrmFormatModifierPropertiesEXT( vkGetDeviceProcAddr( device, "vkGetImageDrmFormatModifierPropertiesEXT" ) );

      //=== VK_EXT_validation_cache ===
      vkCreateValidationCacheEXT  = PFN_vkCreateValidationCacheEXT( vkGetDeviceProcAddr( device, "vkCreateValidationCacheEXT" ) );
      vkDestroyValidationCacheEXT = PFN_vkDestroyValidationCacheEXT( vkGetDeviceProcAddr( device, "vkDestroyValidationCacheEXT" ) );
      vkMergeValidationCachesEXT  = PFN_vkMergeValidationCachesEXT( vkGetDeviceProcAddr( device, "vkMergeValidationCachesEXT" ) );
      vkGetValidationCacheDataEXT = PFN_vkGetValidationCacheDataEXT( vkGetDeviceProcAddr( device, "vkGetValidationCacheDataEXT" ) );

      //=== VK_NV_shading_rate_image ===
      vkCmdBindShadingRateImageNV          = PFN_vkCmdBindShadingRateImageNV( vkGetDeviceProcAddr( device, "vkCmdBindShadingRateImageNV" ) );
      vkCmdSetViewportShadingRatePaletteNV = PFN_vkCmdSetViewportShadingRatePaletteNV( vkGetDeviceProcAddr( device, "vkCmdSetViewportShadingRatePaletteNV" ) );
      vkCmdSetCoarseSampleOrderNV          = PFN_vkCmdSetCoarseSampleOrderNV( vkGetDeviceProcAddr( device, "vkCmdSetCoarseSampleOrderNV" ) );

      //=== VK_NV_ray_tracing ===
      vkCreateAccelerationStructureNV  = PFN_vkCreateAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCreateAccelerationStructureNV" ) );
      vkDestroyAccelerationStructureNV = PFN_vkDestroyAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkDestroyAccelerationStructureNV" ) );
      vkGetAccelerationStructureMemoryRequirementsNV =
        PFN_vkGetAccelerationStructureMemoryRequirementsNV( vkGetDeviceProcAddr( device, "vkGetAccelerationStructureMemoryRequirementsNV" ) );
      vkBindAccelerationStructureMemoryNV = PFN_vkBindAccelerationStructureMemoryNV( vkGetDeviceProcAddr( device, "vkBindAccelerationStructureMemoryNV" ) );
      vkCmdBuildAccelerationStructureNV   = PFN_vkCmdBuildAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructureNV" ) );
      vkCmdCopyAccelerationStructureNV    = PFN_vkCmdCopyAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureNV" ) );
      vkCmdTraceRaysNV                    = PFN_vkCmdTraceRaysNV( vkGetDeviceProcAddr( device, "vkCmdTraceRaysNV" ) );
      vkCreateRayTracingPipelinesNV       = PFN_vkCreateRayTracingPipelinesNV( vkGetDeviceProcAddr( device, "vkCreateRayTracingPipelinesNV" ) );
      vkGetRayTracingShaderGroupHandlesNV = PFN_vkGetRayTracingShaderGroupHandlesNV( vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupHandlesNV" ) );
      if ( !vkGetRayTracingShaderGroupHandlesKHR )
        vkGetRayTracingShaderGroupHandlesKHR = vkGetRayTracingShaderGroupHandlesNV;
      vkGetAccelerationStructureHandleNV = PFN_vkGetAccelerationStructureHandleNV( vkGetDeviceProcAddr( device, "vkGetAccelerationStructureHandleNV" ) );
      vkCmdWriteAccelerationStructuresPropertiesNV =
        PFN_vkCmdWriteAccelerationStructuresPropertiesNV( vkGetDeviceProcAddr( device, "vkCmdWriteAccelerationStructuresPropertiesNV" ) );
      vkCompileDeferredNV = PFN_vkCompileDeferredNV( vkGetDeviceProcAddr( device, "vkCompileDeferredNV" ) );

      //=== VK_KHR_maintenance3 ===
      vkGetDescriptorSetLayoutSupportKHR = PFN_vkGetDescriptorSetLayoutSupportKHR( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutSupportKHR" ) );
      if ( !vkGetDescriptorSetLayoutSupport )
        vkGetDescriptorSetLayoutSupport = vkGetDescriptorSetLayoutSupportKHR;

      //=== VK_KHR_draw_indirect_count ===
      vkCmdDrawIndirectCountKHR = PFN_vkCmdDrawIndirectCountKHR( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCountKHR" ) );
      if ( !vkCmdDrawIndirectCount )
        vkCmdDrawIndirectCount = vkCmdDrawIndirectCountKHR;
      vkCmdDrawIndexedIndirectCountKHR = PFN_vkCmdDrawIndexedIndirectCountKHR( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCountKHR" ) );
      if ( !vkCmdDrawIndexedIndirectCount )
        vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountKHR;

      //=== VK_EXT_external_memory_host ===
      vkGetMemoryHostPointerPropertiesEXT = PFN_vkGetMemoryHostPointerPropertiesEXT( vkGetDeviceProcAddr( device, "vkGetMemoryHostPointerPropertiesEXT" ) );

      //=== VK_AMD_buffer_marker ===
      vkCmdWriteBufferMarkerAMD = PFN_vkCmdWriteBufferMarkerAMD( vkGetDeviceProcAddr( device, "vkCmdWriteBufferMarkerAMD" ) );

      //=== VK_EXT_calibrated_timestamps ===
      vkGetCalibratedTimestampsEXT = PFN_vkGetCalibratedTimestampsEXT( vkGetDeviceProcAddr( device, "vkGetCalibratedTimestampsEXT" ) );
      if ( !vkGetCalibratedTimestampsKHR )
        vkGetCalibratedTimestampsKHR = vkGetCalibratedTimestampsEXT;

      //=== VK_NV_mesh_shader ===
      vkCmdDrawMeshTasksNV              = PFN_vkCmdDrawMeshTasksNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksNV" ) );
      vkCmdDrawMeshTasksIndirectNV      = PFN_vkCmdDrawMeshTasksIndirectNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksIndirectNV" ) );
      vkCmdDrawMeshTasksIndirectCountNV = PFN_vkCmdDrawMeshTasksIndirectCountNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksIndirectCountNV" ) );

      //=== VK_NV_scissor_exclusive ===
      vkCmdSetExclusiveScissorEnableNV = PFN_vkCmdSetExclusiveScissorEnableNV( vkGetDeviceProcAddr( device, "vkCmdSetExclusiveScissorEnableNV" ) );
      vkCmdSetExclusiveScissorNV       = PFN_vkCmdSetExclusiveScissorNV( vkGetDeviceProcAddr( device, "vkCmdSetExclusiveScissorNV" ) );

      //=== VK_NV_device_diagnostic_checkpoints ===
      vkCmdSetCheckpointNV       = PFN_vkCmdSetCheckpointNV( vkGetDeviceProcAddr( device, "vkCmdSetCheckpointNV" ) );
      vkGetQueueCheckpointDataNV = PFN_vkGetQueueCheckpointDataNV( vkGetDeviceProcAddr( device, "vkGetQueueCheckpointDataNV" ) );

      //=== VK_KHR_timeline_semaphore ===
      vkGetSemaphoreCounterValueKHR = PFN_vkGetSemaphoreCounterValueKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreCounterValueKHR" ) );
      if ( !vkGetSemaphoreCounterValue )
        vkGetSemaphoreCounterValue = vkGetSemaphoreCounterValueKHR;
      vkWaitSemaphoresKHR = PFN_vkWaitSemaphoresKHR( vkGetDeviceProcAddr( device, "vkWaitSemaphoresKHR" ) );
      if ( !vkWaitSemaphores )
        vkWaitSemaphores = vkWaitSemaphoresKHR;
      vkSignalSemaphoreKHR = PFN_vkSignalSemaphoreKHR( vkGetDeviceProcAddr( device, "vkSignalSemaphoreKHR" ) );
      if ( !vkSignalSemaphore )
        vkSignalSemaphore = vkSignalSemaphoreKHR;

      //=== VK_INTEL_performance_query ===
      vkInitializePerformanceApiINTEL      = PFN_vkInitializePerformanceApiINTEL( vkGetDeviceProcAddr( device, "vkInitializePerformanceApiINTEL" ) );
      vkUninitializePerformanceApiINTEL    = PFN_vkUninitializePerformanceApiINTEL( vkGetDeviceProcAddr( device, "vkUninitializePerformanceApiINTEL" ) );
      vkCmdSetPerformanceMarkerINTEL       = PFN_vkCmdSetPerformanceMarkerINTEL( vkGetDeviceProcAddr( device, "vkCmdSetPerformanceMarkerINTEL" ) );
      vkCmdSetPerformanceStreamMarkerINTEL = PFN_vkCmdSetPerformanceStreamMarkerINTEL( vkGetDeviceProcAddr( device, "vkCmdSetPerformanceStreamMarkerINTEL" ) );
      vkCmdSetPerformanceOverrideINTEL     = PFN_vkCmdSetPerformanceOverrideINTEL( vkGetDeviceProcAddr( device, "vkCmdSetPerformanceOverrideINTEL" ) );
      vkAcquirePerformanceConfigurationINTEL =
        PFN_vkAcquirePerformanceConfigurationINTEL( vkGetDeviceProcAddr( device, "vkAcquirePerformanceConfigurationINTEL" ) );
      vkReleasePerformanceConfigurationINTEL =
        PFN_vkReleasePerformanceConfigurationINTEL( vkGetDeviceProcAddr( device, "vkReleasePerformanceConfigurationINTEL" ) );
      vkQueueSetPerformanceConfigurationINTEL =
        PFN_vkQueueSetPerformanceConfigurationINTEL( vkGetDeviceProcAddr( device, "vkQueueSetPerformanceConfigurationINTEL" ) );
      vkGetPerformanceParameterINTEL = PFN_vkGetPerformanceParameterINTEL( vkGetDeviceProcAddr( device, "vkGetPerformanceParameterINTEL" ) );

      //=== VK_AMD_display_native_hdr ===
      vkSetLocalDimmingAMD = PFN_vkSetLocalDimmingAMD( vkGetDeviceProcAddr( device, "vkSetLocalDimmingAMD" ) );

      //=== VK_KHR_fragment_shading_rate ===
      vkCmdSetFragmentShadingRateKHR = PFN_vkCmdSetFragmentShadingRateKHR( vkGetDeviceProcAddr( device, "vkCmdSetFragmentShadingRateKHR" ) );

      //=== VK_EXT_buffer_device_address ===
      vkGetBufferDeviceAddressEXT = PFN_vkGetBufferDeviceAddressEXT( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddressEXT" ) );
      if ( !vkGetBufferDeviceAddress )
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressEXT;

      //=== VK_KHR_present_wait ===
      vkWaitForPresentKHR = PFN_vkWaitForPresentKHR( vkGetDeviceProcAddr( device, "vkWaitForPresentKHR" ) );

#if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_EXT_full_screen_exclusive ===
      vkAcquireFullScreenExclusiveModeEXT = PFN_vkAcquireFullScreenExclusiveModeEXT( vkGetDeviceProcAddr( device, "vkAcquireFullScreenExclusiveModeEXT" ) );
      vkReleaseFullScreenExclusiveModeEXT = PFN_vkReleaseFullScreenExclusiveModeEXT( vkGetDeviceProcAddr( device, "vkReleaseFullScreenExclusiveModeEXT" ) );
      vkGetDeviceGroupSurfacePresentModes2EXT =
        PFN_vkGetDeviceGroupSurfacePresentModes2EXT( vkGetDeviceProcAddr( device, "vkGetDeviceGroupSurfacePresentModes2EXT" ) );
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_buffer_device_address ===
      vkGetBufferDeviceAddressKHR = PFN_vkGetBufferDeviceAddressKHR( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddressKHR" ) );
      if ( !vkGetBufferDeviceAddress )
        vkGetBufferDeviceAddress = vkGetBufferDeviceAddressKHR;
      vkGetBufferOpaqueCaptureAddressKHR = PFN_vkGetBufferOpaqueCaptureAddressKHR( vkGetDeviceProcAddr( device, "vkGetBufferOpaqueCaptureAddressKHR" ) );
      if ( !vkGetBufferOpaqueCaptureAddress )
        vkGetBufferOpaqueCaptureAddress = vkGetBufferOpaqueCaptureAddressKHR;
      vkGetDeviceMemoryOpaqueCaptureAddressKHR =
        PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR( vkGetDeviceProcAddr( device, "vkGetDeviceMemoryOpaqueCaptureAddressKHR" ) );
      if ( !vkGetDeviceMemoryOpaqueCaptureAddress )
        vkGetDeviceMemoryOpaqueCaptureAddress = vkGetDeviceMemoryOpaqueCaptureAddressKHR;

      //=== VK_EXT_line_rasterization ===
      vkCmdSetLineStippleEXT = PFN_vkCmdSetLineStippleEXT( vkGetDeviceProcAddr( device, "vkCmdSetLineStippleEXT" ) );

      //=== VK_EXT_host_query_reset ===
      vkResetQueryPoolEXT = PFN_vkResetQueryPoolEXT( vkGetDeviceProcAddr( device, "vkResetQueryPoolEXT" ) );
      if ( !vkResetQueryPool )
        vkResetQueryPool = vkResetQueryPoolEXT;

      //=== VK_EXT_extended_dynamic_state ===
      vkCmdSetCullModeEXT = PFN_vkCmdSetCullModeEXT( vkGetDeviceProcAddr( device, "vkCmdSetCullModeEXT" ) );
      if ( !vkCmdSetCullMode )
        vkCmdSetCullMode = vkCmdSetCullModeEXT;
      vkCmdSetFrontFaceEXT = PFN_vkCmdSetFrontFaceEXT( vkGetDeviceProcAddr( device, "vkCmdSetFrontFaceEXT" ) );
      if ( !vkCmdSetFrontFace )
        vkCmdSetFrontFace = vkCmdSetFrontFaceEXT;
      vkCmdSetPrimitiveTopologyEXT = PFN_vkCmdSetPrimitiveTopologyEXT( vkGetDeviceProcAddr( device, "vkCmdSetPrimitiveTopologyEXT" ) );
      if ( !vkCmdSetPrimitiveTopology )
        vkCmdSetPrimitiveTopology = vkCmdSetPrimitiveTopologyEXT;
      vkCmdSetViewportWithCountEXT = PFN_vkCmdSetViewportWithCountEXT( vkGetDeviceProcAddr( device, "vkCmdSetViewportWithCountEXT" ) );
      if ( !vkCmdSetViewportWithCount )
        vkCmdSetViewportWithCount = vkCmdSetViewportWithCountEXT;
      vkCmdSetScissorWithCountEXT = PFN_vkCmdSetScissorWithCountEXT( vkGetDeviceProcAddr( device, "vkCmdSetScissorWithCountEXT" ) );
      if ( !vkCmdSetScissorWithCount )
        vkCmdSetScissorWithCount = vkCmdSetScissorWithCountEXT;
      vkCmdBindVertexBuffers2EXT = PFN_vkCmdBindVertexBuffers2EXT( vkGetDeviceProcAddr( device, "vkCmdBindVertexBuffers2EXT" ) );
      if ( !vkCmdBindVertexBuffers2 )
        vkCmdBindVertexBuffers2 = vkCmdBindVertexBuffers2EXT;
      vkCmdSetDepthTestEnableEXT = PFN_vkCmdSetDepthTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthTestEnableEXT" ) );
      if ( !vkCmdSetDepthTestEnable )
        vkCmdSetDepthTestEnable = vkCmdSetDepthTestEnableEXT;
      vkCmdSetDepthWriteEnableEXT = PFN_vkCmdSetDepthWriteEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthWriteEnableEXT" ) );
      if ( !vkCmdSetDepthWriteEnable )
        vkCmdSetDepthWriteEnable = vkCmdSetDepthWriteEnableEXT;
      vkCmdSetDepthCompareOpEXT = PFN_vkCmdSetDepthCompareOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthCompareOpEXT" ) );
      if ( !vkCmdSetDepthCompareOp )
        vkCmdSetDepthCompareOp = vkCmdSetDepthCompareOpEXT;
      vkCmdSetDepthBoundsTestEnableEXT = PFN_vkCmdSetDepthBoundsTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthBoundsTestEnableEXT" ) );
      if ( !vkCmdSetDepthBoundsTestEnable )
        vkCmdSetDepthBoundsTestEnable = vkCmdSetDepthBoundsTestEnableEXT;
      vkCmdSetStencilTestEnableEXT = PFN_vkCmdSetStencilTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetStencilTestEnableEXT" ) );
      if ( !vkCmdSetStencilTestEnable )
        vkCmdSetStencilTestEnable = vkCmdSetStencilTestEnableEXT;
      vkCmdSetStencilOpEXT = PFN_vkCmdSetStencilOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetStencilOpEXT" ) );
      if ( !vkCmdSetStencilOp )
        vkCmdSetStencilOp = vkCmdSetStencilOpEXT;

      //=== VK_KHR_deferred_host_operations ===
      vkCreateDeferredOperationKHR  = PFN_vkCreateDeferredOperationKHR( vkGetDeviceProcAddr( device, "vkCreateDeferredOperationKHR" ) );
      vkDestroyDeferredOperationKHR = PFN_vkDestroyDeferredOperationKHR( vkGetDeviceProcAddr( device, "vkDestroyDeferredOperationKHR" ) );
      vkGetDeferredOperationMaxConcurrencyKHR =
        PFN_vkGetDeferredOperationMaxConcurrencyKHR( vkGetDeviceProcAddr( device, "vkGetDeferredOperationMaxConcurrencyKHR" ) );
      vkGetDeferredOperationResultKHR = PFN_vkGetDeferredOperationResultKHR( vkGetDeviceProcAddr( device, "vkGetDeferredOperationResultKHR" ) );
      vkDeferredOperationJoinKHR      = PFN_vkDeferredOperationJoinKHR( vkGetDeviceProcAddr( device, "vkDeferredOperationJoinKHR" ) );

      //=== VK_KHR_pipeline_executable_properties ===
      vkGetPipelineExecutablePropertiesKHR = PFN_vkGetPipelineExecutablePropertiesKHR( vkGetDeviceProcAddr( device, "vkGetPipelineExecutablePropertiesKHR" ) );
      vkGetPipelineExecutableStatisticsKHR = PFN_vkGetPipelineExecutableStatisticsKHR( vkGetDeviceProcAddr( device, "vkGetPipelineExecutableStatisticsKHR" ) );
      vkGetPipelineExecutableInternalRepresentationsKHR =
        PFN_vkGetPipelineExecutableInternalRepresentationsKHR( vkGetDeviceProcAddr( device, "vkGetPipelineExecutableInternalRepresentationsKHR" ) );

      //=== VK_EXT_host_image_copy ===
      vkCopyMemoryToImageEXT          = PFN_vkCopyMemoryToImageEXT( vkGetDeviceProcAddr( device, "vkCopyMemoryToImageEXT" ) );
      vkCopyImageToMemoryEXT          = PFN_vkCopyImageToMemoryEXT( vkGetDeviceProcAddr( device, "vkCopyImageToMemoryEXT" ) );
      vkCopyImageToImageEXT           = PFN_vkCopyImageToImageEXT( vkGetDeviceProcAddr( device, "vkCopyImageToImageEXT" ) );
      vkTransitionImageLayoutEXT      = PFN_vkTransitionImageLayoutEXT( vkGetDeviceProcAddr( device, "vkTransitionImageLayoutEXT" ) );
      vkGetImageSubresourceLayout2EXT = PFN_vkGetImageSubresourceLayout2EXT( vkGetDeviceProcAddr( device, "vkGetImageSubresourceLayout2EXT" ) );
      if ( !vkGetImageSubresourceLayout2KHR )
        vkGetImageSubresourceLayout2KHR = vkGetImageSubresourceLayout2EXT;

      //=== VK_KHR_map_memory2 ===
      vkMapMemory2KHR   = PFN_vkMapMemory2KHR( vkGetDeviceProcAddr( device, "vkMapMemory2KHR" ) );
      vkUnmapMemory2KHR = PFN_vkUnmapMemory2KHR( vkGetDeviceProcAddr( device, "vkUnmapMemory2KHR" ) );

      //=== VK_EXT_swapchain_maintenance1 ===
      vkReleaseSwapchainImagesEXT = PFN_vkReleaseSwapchainImagesEXT( vkGetDeviceProcAddr( device, "vkReleaseSwapchainImagesEXT" ) );

      //=== VK_NV_device_generated_commands ===
      vkGetGeneratedCommandsMemoryRequirementsNV =
        PFN_vkGetGeneratedCommandsMemoryRequirementsNV( vkGetDeviceProcAddr( device, "vkGetGeneratedCommandsMemoryRequirementsNV" ) );
      vkCmdPreprocessGeneratedCommandsNV = PFN_vkCmdPreprocessGeneratedCommandsNV( vkGetDeviceProcAddr( device, "vkCmdPreprocessGeneratedCommandsNV" ) );
      vkCmdExecuteGeneratedCommandsNV    = PFN_vkCmdExecuteGeneratedCommandsNV( vkGetDeviceProcAddr( device, "vkCmdExecuteGeneratedCommandsNV" ) );
      vkCmdBindPipelineShaderGroupNV     = PFN_vkCmdBindPipelineShaderGroupNV( vkGetDeviceProcAddr( device, "vkCmdBindPipelineShaderGroupNV" ) );
      vkCreateIndirectCommandsLayoutNV   = PFN_vkCreateIndirectCommandsLayoutNV( vkGetDeviceProcAddr( device, "vkCreateIndirectCommandsLayoutNV" ) );
      vkDestroyIndirectCommandsLayoutNV  = PFN_vkDestroyIndirectCommandsLayoutNV( vkGetDeviceProcAddr( device, "vkDestroyIndirectCommandsLayoutNV" ) );

      //=== VK_EXT_depth_bias_control ===
      vkCmdSetDepthBias2EXT = PFN_vkCmdSetDepthBias2EXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthBias2EXT" ) );

      //=== VK_EXT_private_data ===
      vkCreatePrivateDataSlotEXT = PFN_vkCreatePrivateDataSlotEXT( vkGetDeviceProcAddr( device, "vkCreatePrivateDataSlotEXT" ) );
      if ( !vkCreatePrivateDataSlot )
        vkCreatePrivateDataSlot = vkCreatePrivateDataSlotEXT;
      vkDestroyPrivateDataSlotEXT = PFN_vkDestroyPrivateDataSlotEXT( vkGetDeviceProcAddr( device, "vkDestroyPrivateDataSlotEXT" ) );
      if ( !vkDestroyPrivateDataSlot )
        vkDestroyPrivateDataSlot = vkDestroyPrivateDataSlotEXT;
      vkSetPrivateDataEXT = PFN_vkSetPrivateDataEXT( vkGetDeviceProcAddr( device, "vkSetPrivateDataEXT" ) );
      if ( !vkSetPrivateData )
        vkSetPrivateData = vkSetPrivateDataEXT;
      vkGetPrivateDataEXT = PFN_vkGetPrivateDataEXT( vkGetDeviceProcAddr( device, "vkGetPrivateDataEXT" ) );
      if ( !vkGetPrivateData )
        vkGetPrivateData = vkGetPrivateDataEXT;

      //=== VK_KHR_video_encode_queue ===
      vkGetEncodedVideoSessionParametersKHR =
        PFN_vkGetEncodedVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkGetEncodedVideoSessionParametersKHR" ) );
      vkCmdEncodeVideoKHR = PFN_vkCmdEncodeVideoKHR( vkGetDeviceProcAddr( device, "vkCmdEncodeVideoKHR" ) );

#if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_NV_cuda_kernel_launch ===
      vkCreateCudaModuleNV    = PFN_vkCreateCudaModuleNV( vkGetDeviceProcAddr( device, "vkCreateCudaModuleNV" ) );
      vkGetCudaModuleCacheNV  = PFN_vkGetCudaModuleCacheNV( vkGetDeviceProcAddr( device, "vkGetCudaModuleCacheNV" ) );
      vkCreateCudaFunctionNV  = PFN_vkCreateCudaFunctionNV( vkGetDeviceProcAddr( device, "vkCreateCudaFunctionNV" ) );
      vkDestroyCudaModuleNV   = PFN_vkDestroyCudaModuleNV( vkGetDeviceProcAddr( device, "vkDestroyCudaModuleNV" ) );
      vkDestroyCudaFunctionNV = PFN_vkDestroyCudaFunctionNV( vkGetDeviceProcAddr( device, "vkDestroyCudaFunctionNV" ) );
      vkCmdCudaLaunchKernelNV = PFN_vkCmdCudaLaunchKernelNV( vkGetDeviceProcAddr( device, "vkCmdCudaLaunchKernelNV" ) );
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
      //=== VK_EXT_metal_objects ===
      vkExportMetalObjectsEXT = PFN_vkExportMetalObjectsEXT( vkGetDeviceProcAddr( device, "vkExportMetalObjectsEXT" ) );
#endif /*VK_USE_PLATFORM_METAL_EXT*/

      //=== VK_KHR_synchronization2 ===
      vkCmdSetEvent2KHR = PFN_vkCmdSetEvent2KHR( vkGetDeviceProcAddr( device, "vkCmdSetEvent2KHR" ) );
      if ( !vkCmdSetEvent2 )
        vkCmdSetEvent2 = vkCmdSetEvent2KHR;
      vkCmdResetEvent2KHR = PFN_vkCmdResetEvent2KHR( vkGetDeviceProcAddr( device, "vkCmdResetEvent2KHR" ) );
      if ( !vkCmdResetEvent2 )
        vkCmdResetEvent2 = vkCmdResetEvent2KHR;
      vkCmdWaitEvents2KHR = PFN_vkCmdWaitEvents2KHR( vkGetDeviceProcAddr( device, "vkCmdWaitEvents2KHR" ) );
      if ( !vkCmdWaitEvents2 )
        vkCmdWaitEvents2 = vkCmdWaitEvents2KHR;
      vkCmdPipelineBarrier2KHR = PFN_vkCmdPipelineBarrier2KHR( vkGetDeviceProcAddr( device, "vkCmdPipelineBarrier2KHR" ) );
      if ( !vkCmdPipelineBarrier2 )
        vkCmdPipelineBarrier2 = vkCmdPipelineBarrier2KHR;
      vkCmdWriteTimestamp2KHR = PFN_vkCmdWriteTimestamp2KHR( vkGetDeviceProcAddr( device, "vkCmdWriteTimestamp2KHR" ) );
      if ( !vkCmdWriteTimestamp2 )
        vkCmdWriteTimestamp2 = vkCmdWriteTimestamp2KHR;
      vkQueueSubmit2KHR = PFN_vkQueueSubmit2KHR( vkGetDeviceProcAddr( device, "vkQueueSubmit2KHR" ) );
      if ( !vkQueueSubmit2 )
        vkQueueSubmit2 = vkQueueSubmit2KHR;
      vkCmdWriteBufferMarker2AMD  = PFN_vkCmdWriteBufferMarker2AMD( vkGetDeviceProcAddr( device, "vkCmdWriteBufferMarker2AMD" ) );
      vkGetQueueCheckpointData2NV = PFN_vkGetQueueCheckpointData2NV( vkGetDeviceProcAddr( device, "vkGetQueueCheckpointData2NV" ) );

      //=== VK_EXT_descriptor_buffer ===
      vkGetDescriptorSetLayoutSizeEXT = PFN_vkGetDescriptorSetLayoutSizeEXT( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutSizeEXT" ) );
      vkGetDescriptorSetLayoutBindingOffsetEXT =
        PFN_vkGetDescriptorSetLayoutBindingOffsetEXT( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutBindingOffsetEXT" ) );
      vkGetDescriptorEXT                 = PFN_vkGetDescriptorEXT( vkGetDeviceProcAddr( device, "vkGetDescriptorEXT" ) );
      vkCmdBindDescriptorBuffersEXT      = PFN_vkCmdBindDescriptorBuffersEXT( vkGetDeviceProcAddr( device, "vkCmdBindDescriptorBuffersEXT" ) );
      vkCmdSetDescriptorBufferOffsetsEXT = PFN_vkCmdSetDescriptorBufferOffsetsEXT( vkGetDeviceProcAddr( device, "vkCmdSetDescriptorBufferOffsetsEXT" ) );
      vkCmdBindDescriptorBufferEmbeddedSamplersEXT =
        PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT( vkGetDeviceProcAddr( device, "vkCmdBindDescriptorBufferEmbeddedSamplersEXT" ) );
      vkGetBufferOpaqueCaptureDescriptorDataEXT =
        PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT( vkGetDeviceProcAddr( device, "vkGetBufferOpaqueCaptureDescriptorDataEXT" ) );
      vkGetImageOpaqueCaptureDescriptorDataEXT =
        PFN_vkGetImageOpaqueCaptureDescriptorDataEXT( vkGetDeviceProcAddr( device, "vkGetImageOpaqueCaptureDescriptorDataEXT" ) );
      vkGetImageViewOpaqueCaptureDescriptorDataEXT =
        PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT( vkGetDeviceProcAddr( device, "vkGetImageViewOpaqueCaptureDescriptorDataEXT" ) );
      vkGetSamplerOpaqueCaptureDescriptorDataEXT =
        PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT( vkGetDeviceProcAddr( device, "vkGetSamplerOpaqueCaptureDescriptorDataEXT" ) );
      vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT = PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT(
        vkGetDeviceProcAddr( device, "vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT" ) );

      //=== VK_NV_fragment_shading_rate_enums ===
      vkCmdSetFragmentShadingRateEnumNV = PFN_vkCmdSetFragmentShadingRateEnumNV( vkGetDeviceProcAddr( device, "vkCmdSetFragmentShadingRateEnumNV" ) );

      //=== VK_EXT_mesh_shader ===
      vkCmdDrawMeshTasksEXT              = PFN_vkCmdDrawMeshTasksEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksEXT" ) );
      vkCmdDrawMeshTasksIndirectEXT      = PFN_vkCmdDrawMeshTasksIndirectEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksIndirectEXT" ) );
      vkCmdDrawMeshTasksIndirectCountEXT = PFN_vkCmdDrawMeshTasksIndirectCountEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksIndirectCountEXT" ) );

      //=== VK_KHR_copy_commands2 ===
      vkCmdCopyBuffer2KHR = PFN_vkCmdCopyBuffer2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyBuffer2KHR" ) );
      if ( !vkCmdCopyBuffer2 )
        vkCmdCopyBuffer2 = vkCmdCopyBuffer2KHR;
      vkCmdCopyImage2KHR = PFN_vkCmdCopyImage2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyImage2KHR" ) );
      if ( !vkCmdCopyImage2 )
        vkCmdCopyImage2 = vkCmdCopyImage2KHR;
      vkCmdCopyBufferToImage2KHR = PFN_vkCmdCopyBufferToImage2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyBufferToImage2KHR" ) );
      if ( !vkCmdCopyBufferToImage2 )
        vkCmdCopyBufferToImage2 = vkCmdCopyBufferToImage2KHR;
      vkCmdCopyImageToBuffer2KHR = PFN_vkCmdCopyImageToBuffer2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyImageToBuffer2KHR" ) );
      if ( !vkCmdCopyImageToBuffer2 )
        vkCmdCopyImageToBuffer2 = vkCmdCopyImageToBuffer2KHR;
      vkCmdBlitImage2KHR = PFN_vkCmdBlitImage2KHR( vkGetDeviceProcAddr( device, "vkCmdBlitImage2KHR" ) );
      if ( !vkCmdBlitImage2 )
        vkCmdBlitImage2 = vkCmdBlitImage2KHR;
      vkCmdResolveImage2KHR = PFN_vkCmdResolveImage2KHR( vkGetDeviceProcAddr( device, "vkCmdResolveImage2KHR" ) );
      if ( !vkCmdResolveImage2 )
        vkCmdResolveImage2 = vkCmdResolveImage2KHR;

      //=== VK_EXT_device_fault ===
      vkGetDeviceFaultInfoEXT = PFN_vkGetDeviceFaultInfoEXT( vkGetDeviceProcAddr( device, "vkGetDeviceFaultInfoEXT" ) );

      //=== VK_EXT_vertex_input_dynamic_state ===
      vkCmdSetVertexInputEXT = PFN_vkCmdSetVertexInputEXT( vkGetDeviceProcAddr( device, "vkCmdSetVertexInputEXT" ) );

#if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_external_memory ===
      vkGetMemoryZirconHandleFUCHSIA = PFN_vkGetMemoryZirconHandleFUCHSIA( vkGetDeviceProcAddr( device, "vkGetMemoryZirconHandleFUCHSIA" ) );
      vkGetMemoryZirconHandlePropertiesFUCHSIA =
        PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA( vkGetDeviceProcAddr( device, "vkGetMemoryZirconHandlePropertiesFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_external_semaphore ===
      vkImportSemaphoreZirconHandleFUCHSIA = PFN_vkImportSemaphoreZirconHandleFUCHSIA( vkGetDeviceProcAddr( device, "vkImportSemaphoreZirconHandleFUCHSIA" ) );
      vkGetSemaphoreZirconHandleFUCHSIA    = PFN_vkGetSemaphoreZirconHandleFUCHSIA( vkGetDeviceProcAddr( device, "vkGetSemaphoreZirconHandleFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_buffer_collection ===
      vkCreateBufferCollectionFUCHSIA = PFN_vkCreateBufferCollectionFUCHSIA( vkGetDeviceProcAddr( device, "vkCreateBufferCollectionFUCHSIA" ) );
      vkSetBufferCollectionImageConstraintsFUCHSIA =
        PFN_vkSetBufferCollectionImageConstraintsFUCHSIA( vkGetDeviceProcAddr( device, "vkSetBufferCollectionImageConstraintsFUCHSIA" ) );
      vkSetBufferCollectionBufferConstraintsFUCHSIA =
        PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA( vkGetDeviceProcAddr( device, "vkSetBufferCollectionBufferConstraintsFUCHSIA" ) );
      vkDestroyBufferCollectionFUCHSIA = PFN_vkDestroyBufferCollectionFUCHSIA( vkGetDeviceProcAddr( device, "vkDestroyBufferCollectionFUCHSIA" ) );
      vkGetBufferCollectionPropertiesFUCHSIA =
        PFN_vkGetBufferCollectionPropertiesFUCHSIA( vkGetDeviceProcAddr( device, "vkGetBufferCollectionPropertiesFUCHSIA" ) );
#endif /*VK_USE_PLATFORM_FUCHSIA*/

      //=== VK_HUAWEI_subpass_shading ===
      vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI =
        PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI( vkGetDeviceProcAddr( device, "vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI" ) );
      vkCmdSubpassShadingHUAWEI = PFN_vkCmdSubpassShadingHUAWEI( vkGetDeviceProcAddr( device, "vkCmdSubpassShadingHUAWEI" ) );

      //=== VK_HUAWEI_invocation_mask ===
      vkCmdBindInvocationMaskHUAWEI = PFN_vkCmdBindInvocationMaskHUAWEI( vkGetDeviceProcAddr( device, "vkCmdBindInvocationMaskHUAWEI" ) );

      //=== VK_NV_external_memory_rdma ===
      vkGetMemoryRemoteAddressNV = PFN_vkGetMemoryRemoteAddressNV( vkGetDeviceProcAddr( device, "vkGetMemoryRemoteAddressNV" ) );

      //=== VK_EXT_pipeline_properties ===
      vkGetPipelinePropertiesEXT = PFN_vkGetPipelinePropertiesEXT( vkGetDeviceProcAddr( device, "vkGetPipelinePropertiesEXT" ) );

      //=== VK_EXT_extended_dynamic_state2 ===
      vkCmdSetPatchControlPointsEXT      = PFN_vkCmdSetPatchControlPointsEXT( vkGetDeviceProcAddr( device, "vkCmdSetPatchControlPointsEXT" ) );
      vkCmdSetRasterizerDiscardEnableEXT = PFN_vkCmdSetRasterizerDiscardEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetRasterizerDiscardEnableEXT" ) );
      if ( !vkCmdSetRasterizerDiscardEnable )
        vkCmdSetRasterizerDiscardEnable = vkCmdSetRasterizerDiscardEnableEXT;
      vkCmdSetDepthBiasEnableEXT = PFN_vkCmdSetDepthBiasEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthBiasEnableEXT" ) );
      if ( !vkCmdSetDepthBiasEnable )
        vkCmdSetDepthBiasEnable = vkCmdSetDepthBiasEnableEXT;
      vkCmdSetLogicOpEXT                = PFN_vkCmdSetLogicOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetLogicOpEXT" ) );
      vkCmdSetPrimitiveRestartEnableEXT = PFN_vkCmdSetPrimitiveRestartEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetPrimitiveRestartEnableEXT" ) );
      if ( !vkCmdSetPrimitiveRestartEnable )
        vkCmdSetPrimitiveRestartEnable = vkCmdSetPrimitiveRestartEnableEXT;

      //=== VK_EXT_color_write_enable ===
      vkCmdSetColorWriteEnableEXT = PFN_vkCmdSetColorWriteEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetColorWriteEnableEXT" ) );

      //=== VK_KHR_ray_tracing_maintenance1 ===
      vkCmdTraceRaysIndirect2KHR = PFN_vkCmdTraceRaysIndirect2KHR( vkGetDeviceProcAddr( device, "vkCmdTraceRaysIndirect2KHR" ) );

      //=== VK_EXT_multi_draw ===
      vkCmdDrawMultiEXT        = PFN_vkCmdDrawMultiEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMultiEXT" ) );
      vkCmdDrawMultiIndexedEXT = PFN_vkCmdDrawMultiIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMultiIndexedEXT" ) );

      //=== VK_EXT_opacity_micromap ===
      vkCreateMicromapEXT                 = PFN_vkCreateMicromapEXT( vkGetDeviceProcAddr( device, "vkCreateMicromapEXT" ) );
      vkDestroyMicromapEXT                = PFN_vkDestroyMicromapEXT( vkGetDeviceProcAddr( device, "vkDestroyMicromapEXT" ) );
      vkCmdBuildMicromapsEXT              = PFN_vkCmdBuildMicromapsEXT( vkGetDeviceProcAddr( device, "vkCmdBuildMicromapsEXT" ) );
      vkBuildMicromapsEXT                 = PFN_vkBuildMicromapsEXT( vkGetDeviceProcAddr( device, "vkBuildMicromapsEXT" ) );
      vkCopyMicromapEXT                   = PFN_vkCopyMicromapEXT( vkGetDeviceProcAddr( device, "vkCopyMicromapEXT" ) );
      vkCopyMicromapToMemoryEXT           = PFN_vkCopyMicromapToMemoryEXT( vkGetDeviceProcAddr( device, "vkCopyMicromapToMemoryEXT" ) );
      vkCopyMemoryToMicromapEXT           = PFN_vkCopyMemoryToMicromapEXT( vkGetDeviceProcAddr( device, "vkCopyMemoryToMicromapEXT" ) );
      vkWriteMicromapsPropertiesEXT       = PFN_vkWriteMicromapsPropertiesEXT( vkGetDeviceProcAddr( device, "vkWriteMicromapsPropertiesEXT" ) );
      vkCmdCopyMicromapEXT                = PFN_vkCmdCopyMicromapEXT( vkGetDeviceProcAddr( device, "vkCmdCopyMicromapEXT" ) );
      vkCmdCopyMicromapToMemoryEXT        = PFN_vkCmdCopyMicromapToMemoryEXT( vkGetDeviceProcAddr( device, "vkCmdCopyMicromapToMemoryEXT" ) );
      vkCmdCopyMemoryToMicromapEXT        = PFN_vkCmdCopyMemoryToMicromapEXT( vkGetDeviceProcAddr( device, "vkCmdCopyMemoryToMicromapEXT" ) );
      vkCmdWriteMicromapsPropertiesEXT    = PFN_vkCmdWriteMicromapsPropertiesEXT( vkGetDeviceProcAddr( device, "vkCmdWriteMicromapsPropertiesEXT" ) );
      vkGetDeviceMicromapCompatibilityEXT = PFN_vkGetDeviceMicromapCompatibilityEXT( vkGetDeviceProcAddr( device, "vkGetDeviceMicromapCompatibilityEXT" ) );
      vkGetMicromapBuildSizesEXT          = PFN_vkGetMicromapBuildSizesEXT( vkGetDeviceProcAddr( device, "vkGetMicromapBuildSizesEXT" ) );

      //=== VK_HUAWEI_cluster_culling_shader ===
      vkCmdDrawClusterHUAWEI         = PFN_vkCmdDrawClusterHUAWEI( vkGetDeviceProcAddr( device, "vkCmdDrawClusterHUAWEI" ) );
      vkCmdDrawClusterIndirectHUAWEI = PFN_vkCmdDrawClusterIndirectHUAWEI( vkGetDeviceProcAddr( device, "vkCmdDrawClusterIndirectHUAWEI" ) );

      //=== VK_EXT_pageable_device_local_memory ===
      vkSetDeviceMemoryPriorityEXT = PFN_vkSetDeviceMemoryPriorityEXT( vkGetDeviceProcAddr( device, "vkSetDeviceMemoryPriorityEXT" ) );

      //=== VK_KHR_maintenance4 ===
      vkGetDeviceBufferMemoryRequirementsKHR =
        PFN_vkGetDeviceBufferMemoryRequirementsKHR( vkGetDeviceProcAddr( device, "vkGetDeviceBufferMemoryRequirementsKHR" ) );
      if ( !vkGetDeviceBufferMemoryRequirements )
        vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirementsKHR;
      vkGetDeviceImageMemoryRequirementsKHR =
        PFN_vkGetDeviceImageMemoryRequirementsKHR( vkGetDeviceProcAddr( device, "vkGetDeviceImageMemoryRequirementsKHR" ) );
      if ( !vkGetDeviceImageMemoryRequirements )
        vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirementsKHR;
      vkGetDeviceImageSparseMemoryRequirementsKHR =
        PFN_vkGetDeviceImageSparseMemoryRequirementsKHR( vkGetDeviceProcAddr( device, "vkGetDeviceImageSparseMemoryRequirementsKHR" ) );
      if ( !vkGetDeviceImageSparseMemoryRequirements )
        vkGetDeviceImageSparseMemoryRequirements = vkGetDeviceImageSparseMemoryRequirementsKHR;

      //=== VK_VALVE_descriptor_set_host_mapping ===
      vkGetDescriptorSetLayoutHostMappingInfoVALVE =
        PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutHostMappingInfoVALVE" ) );
      vkGetDescriptorSetHostMappingVALVE = PFN_vkGetDescriptorSetHostMappingVALVE( vkGetDeviceProcAddr( device, "vkGetDescriptorSetHostMappingVALVE" ) );

      //=== VK_NV_copy_memory_indirect ===
      vkCmdCopyMemoryIndirectNV        = PFN_vkCmdCopyMemoryIndirectNV( vkGetDeviceProcAddr( device, "vkCmdCopyMemoryIndirectNV" ) );
      vkCmdCopyMemoryToImageIndirectNV = PFN_vkCmdCopyMemoryToImageIndirectNV( vkGetDeviceProcAddr( device, "vkCmdCopyMemoryToImageIndirectNV" ) );

      //=== VK_NV_memory_decompression ===
      vkCmdDecompressMemoryNV              = PFN_vkCmdDecompressMemoryNV( vkGetDeviceProcAddr( device, "vkCmdDecompressMemoryNV" ) );
      vkCmdDecompressMemoryIndirectCountNV = PFN_vkCmdDecompressMemoryIndirectCountNV( vkGetDeviceProcAddr( device, "vkCmdDecompressMemoryIndirectCountNV" ) );

      //=== VK_NV_device_generated_commands_compute ===
      vkGetPipelineIndirectMemoryRequirementsNV =
        PFN_vkGetPipelineIndirectMemoryRequirementsNV( vkGetDeviceProcAddr( device, "vkGetPipelineIndirectMemoryRequirementsNV" ) );
      vkCmdUpdatePipelineIndirectBufferNV  = PFN_vkCmdUpdatePipelineIndirectBufferNV( vkGetDeviceProcAddr( device, "vkCmdUpdatePipelineIndirectBufferNV" ) );
      vkGetPipelineIndirectDeviceAddressNV = PFN_vkGetPipelineIndirectDeviceAddressNV( vkGetDeviceProcAddr( device, "vkGetPipelineIndirectDeviceAddressNV" ) );

      //=== VK_EXT_extended_dynamic_state3 ===
      vkCmdSetTessellationDomainOriginEXT = PFN_vkCmdSetTessellationDomainOriginEXT( vkGetDeviceProcAddr( device, "vkCmdSetTessellationDomainOriginEXT" ) );
      vkCmdSetDepthClampEnableEXT         = PFN_vkCmdSetDepthClampEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthClampEnableEXT" ) );
      vkCmdSetPolygonModeEXT              = PFN_vkCmdSetPolygonModeEXT( vkGetDeviceProcAddr( device, "vkCmdSetPolygonModeEXT" ) );
      vkCmdSetRasterizationSamplesEXT     = PFN_vkCmdSetRasterizationSamplesEXT( vkGetDeviceProcAddr( device, "vkCmdSetRasterizationSamplesEXT" ) );
      vkCmdSetSampleMaskEXT               = PFN_vkCmdSetSampleMaskEXT( vkGetDeviceProcAddr( device, "vkCmdSetSampleMaskEXT" ) );
      vkCmdSetAlphaToCoverageEnableEXT    = PFN_vkCmdSetAlphaToCoverageEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetAlphaToCoverageEnableEXT" ) );
      vkCmdSetAlphaToOneEnableEXT         = PFN_vkCmdSetAlphaToOneEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetAlphaToOneEnableEXT" ) );
      vkCmdSetLogicOpEnableEXT            = PFN_vkCmdSetLogicOpEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetLogicOpEnableEXT" ) );
      vkCmdSetColorBlendEnableEXT         = PFN_vkCmdSetColorBlendEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetColorBlendEnableEXT" ) );
      vkCmdSetColorBlendEquationEXT       = PFN_vkCmdSetColorBlendEquationEXT( vkGetDeviceProcAddr( device, "vkCmdSetColorBlendEquationEXT" ) );
      vkCmdSetColorWriteMaskEXT           = PFN_vkCmdSetColorWriteMaskEXT( vkGetDeviceProcAddr( device, "vkCmdSetColorWriteMaskEXT" ) );
      vkCmdSetRasterizationStreamEXT      = PFN_vkCmdSetRasterizationStreamEXT( vkGetDeviceProcAddr( device, "vkCmdSetRasterizationStreamEXT" ) );
      vkCmdSetConservativeRasterizationModeEXT =
        PFN_vkCmdSetConservativeRasterizationModeEXT( vkGetDeviceProcAddr( device, "vkCmdSetConservativeRasterizationModeEXT" ) );
      vkCmdSetExtraPrimitiveOverestimationSizeEXT =
        PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT( vkGetDeviceProcAddr( device, "vkCmdSetExtraPrimitiveOverestimationSizeEXT" ) );
      vkCmdSetDepthClipEnableEXT           = PFN_vkCmdSetDepthClipEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthClipEnableEXT" ) );
      vkCmdSetSampleLocationsEnableEXT     = PFN_vkCmdSetSampleLocationsEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetSampleLocationsEnableEXT" ) );
      vkCmdSetColorBlendAdvancedEXT        = PFN_vkCmdSetColorBlendAdvancedEXT( vkGetDeviceProcAddr( device, "vkCmdSetColorBlendAdvancedEXT" ) );
      vkCmdSetProvokingVertexModeEXT       = PFN_vkCmdSetProvokingVertexModeEXT( vkGetDeviceProcAddr( device, "vkCmdSetProvokingVertexModeEXT" ) );
      vkCmdSetLineRasterizationModeEXT     = PFN_vkCmdSetLineRasterizationModeEXT( vkGetDeviceProcAddr( device, "vkCmdSetLineRasterizationModeEXT" ) );
      vkCmdSetLineStippleEnableEXT         = PFN_vkCmdSetLineStippleEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetLineStippleEnableEXT" ) );
      vkCmdSetDepthClipNegativeOneToOneEXT = PFN_vkCmdSetDepthClipNegativeOneToOneEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthClipNegativeOneToOneEXT" ) );
      vkCmdSetViewportWScalingEnableNV     = PFN_vkCmdSetViewportWScalingEnableNV( vkGetDeviceProcAddr( device, "vkCmdSetViewportWScalingEnableNV" ) );
      vkCmdSetViewportSwizzleNV            = PFN_vkCmdSetViewportSwizzleNV( vkGetDeviceProcAddr( device, "vkCmdSetViewportSwizzleNV" ) );
      vkCmdSetCoverageToColorEnableNV      = PFN_vkCmdSetCoverageToColorEnableNV( vkGetDeviceProcAddr( device, "vkCmdSetCoverageToColorEnableNV" ) );
      vkCmdSetCoverageToColorLocationNV    = PFN_vkCmdSetCoverageToColorLocationNV( vkGetDeviceProcAddr( device, "vkCmdSetCoverageToColorLocationNV" ) );
      vkCmdSetCoverageModulationModeNV     = PFN_vkCmdSetCoverageModulationModeNV( vkGetDeviceProcAddr( device, "vkCmdSetCoverageModulationModeNV" ) );
      vkCmdSetCoverageModulationTableEnableNV =
        PFN_vkCmdSetCoverageModulationTableEnableNV( vkGetDeviceProcAddr( device, "vkCmdSetCoverageModulationTableEnableNV" ) );
      vkCmdSetCoverageModulationTableNV = PFN_vkCmdSetCoverageModulationTableNV( vkGetDeviceProcAddr( device, "vkCmdSetCoverageModulationTableNV" ) );
      vkCmdSetShadingRateImageEnableNV  = PFN_vkCmdSetShadingRateImageEnableNV( vkGetDeviceProcAddr( device, "vkCmdSetShadingRateImageEnableNV" ) );
      vkCmdSetRepresentativeFragmentTestEnableNV =
        PFN_vkCmdSetRepresentativeFragmentTestEnableNV( vkGetDeviceProcAddr( device, "vkCmdSetRepresentativeFragmentTestEnableNV" ) );
      vkCmdSetCoverageReductionModeNV = PFN_vkCmdSetCoverageReductionModeNV( vkGetDeviceProcAddr( device, "vkCmdSetCoverageReductionModeNV" ) );

      //=== VK_EXT_shader_module_identifier ===
      vkGetShaderModuleIdentifierEXT = PFN_vkGetShaderModuleIdentifierEXT( vkGetDeviceProcAddr( device, "vkGetShaderModuleIdentifierEXT" ) );
      vkGetShaderModuleCreateInfoIdentifierEXT =
        PFN_vkGetShaderModuleCreateInfoIdentifierEXT( vkGetDeviceProcAddr( device, "vkGetShaderModuleCreateInfoIdentifierEXT" ) );

      //=== VK_NV_optical_flow ===
      vkCreateOpticalFlowSessionNV    = PFN_vkCreateOpticalFlowSessionNV( vkGetDeviceProcAddr( device, "vkCreateOpticalFlowSessionNV" ) );
      vkDestroyOpticalFlowSessionNV   = PFN_vkDestroyOpticalFlowSessionNV( vkGetDeviceProcAddr( device, "vkDestroyOpticalFlowSessionNV" ) );
      vkBindOpticalFlowSessionImageNV = PFN_vkBindOpticalFlowSessionImageNV( vkGetDeviceProcAddr( device, "vkBindOpticalFlowSessionImageNV" ) );
      vkCmdOpticalFlowExecuteNV       = PFN_vkCmdOpticalFlowExecuteNV( vkGetDeviceProcAddr( device, "vkCmdOpticalFlowExecuteNV" ) );

      //=== VK_KHR_maintenance5 ===
      vkCmdBindIndexBuffer2KHR             = PFN_vkCmdBindIndexBuffer2KHR( vkGetDeviceProcAddr( device, "vkCmdBindIndexBuffer2KHR" ) );
      vkGetRenderingAreaGranularityKHR     = PFN_vkGetRenderingAreaGranularityKHR( vkGetDeviceProcAddr( device, "vkGetRenderingAreaGranularityKHR" ) );
      vkGetDeviceImageSubresourceLayoutKHR = PFN_vkGetDeviceImageSubresourceLayoutKHR( vkGetDeviceProcAddr( device, "vkGetDeviceImageSubresourceLayoutKHR" ) );
      vkGetImageSubresourceLayout2KHR      = PFN_vkGetImageSubresourceLayout2KHR( vkGetDeviceProcAddr( device, "vkGetImageSubresourceLayout2KHR" ) );

      //=== VK_EXT_shader_object ===
      vkCreateShadersEXT       = PFN_vkCreateShadersEXT( vkGetDeviceProcAddr( device, "vkCreateShadersEXT" ) );
      vkDestroyShaderEXT       = PFN_vkDestroyShaderEXT( vkGetDeviceProcAddr( device, "vkDestroyShaderEXT" ) );
      vkGetShaderBinaryDataEXT = PFN_vkGetShaderBinaryDataEXT( vkGetDeviceProcAddr( device, "vkGetShaderBinaryDataEXT" ) );
      vkCmdBindShadersEXT      = PFN_vkCmdBindShadersEXT( vkGetDeviceProcAddr( device, "vkCmdBindShadersEXT" ) );

      //=== VK_QCOM_tile_properties ===
      vkGetFramebufferTilePropertiesQCOM = PFN_vkGetFramebufferTilePropertiesQCOM( vkGetDeviceProcAddr( device, "vkGetFramebufferTilePropertiesQCOM" ) );
      vkGetDynamicRenderingTilePropertiesQCOM =
        PFN_vkGetDynamicRenderingTilePropertiesQCOM( vkGetDeviceProcAddr( device, "vkGetDynamicRenderingTilePropertiesQCOM" ) );

      //=== VK_NV_low_latency2 ===
      vkSetLatencySleepModeNV  = PFN_vkSetLatencySleepModeNV( vkGetDeviceProcAddr( device, "vkSetLatencySleepModeNV" ) );
      vkLatencySleepNV         = PFN_vkLatencySleepNV( vkGetDeviceProcAddr( device, "vkLatencySleepNV" ) );
      vkSetLatencyMarkerNV     = PFN_vkSetLatencyMarkerNV( vkGetDeviceProcAddr( device, "vkSetLatencyMarkerNV" ) );
      vkGetLatencyTimingsNV    = PFN_vkGetLatencyTimingsNV( vkGetDeviceProcAddr( device, "vkGetLatencyTimingsNV" ) );
      vkQueueNotifyOutOfBandNV = PFN_vkQueueNotifyOutOfBandNV( vkGetDeviceProcAddr( device, "vkQueueNotifyOutOfBandNV" ) );

      //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
      vkCmdSetAttachmentFeedbackLoopEnableEXT =
        PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetAttachmentFeedbackLoopEnableEXT" ) );

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      //=== VK_QNX_external_memory_screen_buffer ===
      vkGetScreenBufferPropertiesQNX = PFN_vkGetScreenBufferPropertiesQNX( vkGetDeviceProcAddr( device, "vkGetScreenBufferPropertiesQNX" ) );
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

      //=== VK_KHR_calibrated_timestamps ===
      vkGetCalibratedTimestampsKHR = PFN_vkGetCalibratedTimestampsKHR( vkGetDeviceProcAddr( device, "vkGetCalibratedTimestampsKHR" ) );

      //=== VK_KHR_maintenance6 ===
      vkCmdBindDescriptorSets2KHR = PFN_vkCmdBindDescriptorSets2KHR( vkGetDeviceProcAddr( device, "vkCmdBindDescriptorSets2KHR" ) );
      vkCmdPushConstants2KHR      = PFN_vkCmdPushConstants2KHR( vkGetDeviceProcAddr( device, "vkCmdPushConstants2KHR" ) );
      vkCmdPushDescriptorSet2KHR  = PFN_vkCmdPushDescriptorSet2KHR( vkGetDeviceProcAddr( device, "vkCmdPushDescriptorSet2KHR" ) );
      vkCmdPushDescriptorSetWithTemplate2KHR =
        PFN_vkCmdPushDescriptorSetWithTemplate2KHR( vkGetDeviceProcAddr( device, "vkCmdPushDescriptorSetWithTemplate2KHR" ) );
      vkCmdSetDescriptorBufferOffsets2EXT = PFN_vkCmdSetDescriptorBufferOffsets2EXT( vkGetDeviceProcAddr( device, "vkCmdSetDescriptorBufferOffsets2EXT" ) );
      vkCmdBindDescriptorBufferEmbeddedSamplers2EXT =
        PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT( vkGetDeviceProcAddr( device, "vkCmdBindDescriptorBufferEmbeddedSamplers2EXT" ) );
    }

    template <typename DynamicLoader>
    void init( VULKAN_HPP_NAMESPACE::Instance const & instance, VULKAN_HPP_NAMESPACE::Device const & device, DynamicLoader const & dl ) VULKAN_HPP_NOEXCEPT
    {
      PFN_vkGetInstanceProcAddr getInstanceProcAddr = dl.template getProcAddress<PFN_vkGetInstanceProcAddr>( "vkGetInstanceProcAddr" );
      PFN_vkGetDeviceProcAddr   getDeviceProcAddr   = dl.template getProcAddress<PFN_vkGetDeviceProcAddr>( "vkGetDeviceProcAddr" );
      init( static_cast<VkInstance>( instance ), getInstanceProcAddr, static_cast<VkDevice>( device ), device ? getDeviceProcAddr : nullptr );
    }

    template <typename DynamicLoader
#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
              = VULKAN_HPP_NAMESPACE::DynamicLoader
#endif
              >
    void init( VULKAN_HPP_NAMESPACE::Instance const & instance, VULKAN_HPP_NAMESPACE::Device const & device ) VULKAN_HPP_NOEXCEPT
    {
      static DynamicLoader dl;
      init( instance, device, dl );
    }
  };
}  // namespace VULKAN_HPP_NAMESPACE
#endif
