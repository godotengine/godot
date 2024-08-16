// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

JPH_NAMESPACE_BEGIN

/// Implements the FNV-1a hash algorithm
/// @see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
/// @param inData Data block of bytes
/// @param inSize Number of bytes
/// @param inSeed Seed of the hash (can be used to pass in the hash of a previous operation, otherwise leave default)
/// @return Hash
inline uint64 HashBytes(const void *inData, uint inSize, uint64 inSeed = 0xcbf29ce484222325UL)
{
	uint64 hash = inSeed;
	for (const uint8 *data = reinterpret_cast<const uint8 *>(inData); data < reinterpret_cast<const uint8 *>(inData) + inSize; ++data)
	{
		hash = hash ^ uint64(*data);
		hash = hash * 0x100000001b3UL;
	}
	return hash;
}

/// A 64 bit hash function by Thomas Wang, Jan 1997
/// See: http://web.archive.org/web/20071223173210/http://www.concentric.net/~Ttwang/tech/inthash.htm
/// @param inValue Value to hash
/// @return Hash
inline uint64 Hash64(uint64 inValue)
{
	uint64 hash = inValue;
	hash = (~hash) + (hash << 21); // hash = (hash << 21) - hash - 1;
	hash = hash ^ (hash >> 24);
	hash = (hash + (hash << 3)) + (hash << 8); // hash * 265
	hash = hash ^ (hash >> 14);
	hash = (hash + (hash << 2)) + (hash << 4); // hash * 21
	hash = hash ^ (hash >> 28);
	hash = hash + (hash << 31);
	return hash;
}

/// @brief Helper function that hashes a single value into ioSeed
/// Taken from: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
template <typename T>
inline void HashCombineHelper(size_t &ioSeed, const T &inValue)
{
	std::hash<T> hasher;
	ioSeed ^= hasher(inValue) + 0x9e3779b9 + (ioSeed << 6) + (ioSeed >> 2);
}

/// Hash combiner to use a custom struct in an unordered map or set
///
/// Usage:
///
///		struct SomeHashKey
///		{
///			std::string key1;
///			std::string key2;
///			bool key3;
///		};
///
///		JPH_MAKE_HASHABLE(SomeHashKey, t.key1, t.key2, t.key3)
template <typename... Values>
inline void HashCombine(std::size_t &ioSeed, Values... inValues)
{
	// Hash all values together using a fold expression
	(HashCombineHelper(ioSeed, inValues), ...);
}

JPH_NAMESPACE_END

JPH_SUPPRESS_WARNING_PUSH
JPH_CLANG_SUPPRESS_WARNING("-Wc++98-compat-pedantic")

#define JPH_MAKE_HASH_STRUCT(type, name, ...)				\
	struct [[nodiscard]] name								\
	{														\
		std::size_t operator()(const type &t) const			\
		{													\
			std::size_t ret = 0;							\
			::JPH::HashCombine(ret, __VA_ARGS__);			\
			return ret;										\
		}													\
	};

#define JPH_MAKE_HASHABLE(type, ...)						\
	JPH_SUPPRESS_WARNING_PUSH								\
	JPH_SUPPRESS_WARNINGS									\
	namespace std											\
	{														\
		template<>											\
		JPH_MAKE_HASH_STRUCT(type, hash<type>, __VA_ARGS__)	\
	}														\
	JPH_SUPPRESS_WARNING_POP

JPH_SUPPRESS_WARNING_POP
