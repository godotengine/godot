/*
 * mptTypeTraits.h
 * ---------------
 * Purpose: C++11 similar type_traits header plus some OpenMPT specific traits.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include <type_traits>



OPENMPT_NAMESPACE_BEGIN



namespace mpt {



template <std::size_t size> struct int_of_size { };
template <> struct int_of_size<1> { typedef int8  type; };
template <> struct int_of_size<2> { typedef int16 type; };
template <> struct int_of_size<3> { typedef int32 type; };
template <> struct int_of_size<4> { typedef int32 type; };
template <> struct int_of_size<5> { typedef int64 type; };
template <> struct int_of_size<6> { typedef int64 type; };
template <> struct int_of_size<7> { typedef int64 type; };
template <> struct int_of_size<8> { typedef int64 type; };

template <std::size_t size> struct uint_of_size { };
template <> struct uint_of_size<1> { typedef uint8  type; };
template <> struct uint_of_size<2> { typedef uint16 type; };
template <> struct uint_of_size<3> { typedef uint32 type; };
template <> struct uint_of_size<4> { typedef uint32 type; };
template <> struct uint_of_size<5> { typedef uint64 type; };
template <> struct uint_of_size<6> { typedef uint64 type; };
template <> struct uint_of_size<7> { typedef uint64 type; };
template <> struct uint_of_size<8> { typedef uint64 type; };


// Tell which types are safe for mpt::byte_cast.
// signed char is actually not allowed to alias into an object representation,
// which means that, if the actual type is not itself signed char but char or
// unsigned char instead, dereferencing the signed char pointer is undefined
// behaviour.
template <typename T> struct is_byte_castable : public std::false_type { };
template <> struct is_byte_castable<char>                : public std::true_type { };
template <> struct is_byte_castable<unsigned char>       : public std::true_type { };
template <> struct is_byte_castable<const char>          : public std::true_type { };
template <> struct is_byte_castable<const unsigned char> : public std::true_type { };


// Tell which types are safe to binary write into files.
// By default, no types are safe.
// When a safe type gets defined,
// also specialize this template so that IO functions will work.
template <typename T> struct is_binary_safe : public std::false_type { }; 

// Specialization for byte types.
template <> struct is_binary_safe<char>  : public std::true_type { };
template <> struct is_binary_safe<uint8> : public std::true_type { };
template <> struct is_binary_safe<int8>  : public std::true_type { };

// Generic Specialization for arrays.
template <typename T, std::size_t N> struct is_binary_safe<T[N]> : public is_binary_safe<T> { };
template <typename T, std::size_t N> struct is_binary_safe<const T[N]> : public is_binary_safe<T> { };

template <typename T>
struct GetRawBytesFunctor
{
	inline const mpt::byte * operator () (const T & v) const
	{
		STATIC_ASSERT(mpt::is_binary_safe<typename std::remove_const<T>::type>::value);
		return reinterpret_cast<const mpt::byte *>(&v);
	}
	inline mpt::byte * operator () (T & v) const
	{
		STATIC_ASSERT(mpt::is_binary_safe<typename std::remove_const<T>::type>::value);
		return reinterpret_cast<mpt::byte *>(&v);
	}
};

template <typename T, std::size_t N>
struct GetRawBytesFunctor<T[N]>
{
	inline const mpt::byte * operator () (const T (&v)[N]) const
	{
		STATIC_ASSERT(mpt::is_binary_safe<typename std::remove_const<T>::type>::value);
		return reinterpret_cast<const mpt::byte *>(v);
	}
	inline mpt::byte * operator () (T (&v)[N]) const
	{
		STATIC_ASSERT(mpt::is_binary_safe<typename std::remove_const<T>::type>::value);
		return reinterpret_cast<mpt::byte *>(v);
	}
};

template <typename T, std::size_t N>
struct GetRawBytesFunctor<const T[N]>
{
	inline const mpt::byte * operator () (const T (&v)[N]) const
	{
		STATIC_ASSERT(mpt::is_binary_safe<typename std::remove_const<T>::type>::value);
		return reinterpret_cast<const mpt::byte *>(v);
	}
};

// In order to be able to partially specialize it,
// as_raw_memory is implemented via a class template.
// Do not overload or specialize as_raw_memory directly.
// Using a wrapper (by default just around a cast to const mpt::byte *),
// allows for implementing raw memory access
// via on-demand generating a cached serialized representation.
template <typename T> inline const mpt::byte * as_raw_memory(const T & v)
{
	STATIC_ASSERT(mpt::is_binary_safe<typename std::remove_const<T>::type>::value);
	return mpt::GetRawBytesFunctor<T>()(v);
}
template <typename T> inline mpt::byte * as_raw_memory(T & v)
{
	STATIC_ASSERT(mpt::is_binary_safe<typename std::remove_const<T>::type>::value);
	return mpt::GetRawBytesFunctor<T>()(v);
}

} // namespace mpt

#define MPT_BINARY_STRUCT(type, size) \
	MPT_STATIC_ASSERT(sizeof( type ) == (size) ); \
	MPT_STATIC_ASSERT(alignof( type ) == 1); \
	MPT_STATIC_ASSERT(std::is_standard_layout< type >::value); \
	namespace mpt { \
		template <> struct is_binary_safe< type > : public std::true_type { }; \
	} \
/**/


OPENMPT_NAMESPACE_END
