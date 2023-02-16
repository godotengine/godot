// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include <cstdlib>


PSD_NAMESPACE_BEGIN

/// \ingroup Util
/// \namespace endianUtil
/// \brief Provides endian conversion routines.
namespace endianUtil
{
	/// Converts from big-endian to native-endian, and returns the converted value.
	template <typename T>
	PSD_INLINE T BigEndianToNative(T value);

	/// Converts from little-endian to native-endian, and returns the converted value.
	template <typename T>
	PSD_INLINE T LittleEndianToNative(T value);

	/// Converts from native-endian to big-endian, and returns the converted value.
	template <typename T>
	PSD_INLINE T NativeToBigEndian(T value);

	/// Converts from native-endian to little-endian, and returns the converted value.
	template <typename T>
	PSD_INLINE T NativeToLittleEndian(T value);
}

#include "PsdEndianConversion.inl"

PSD_NAMESPACE_END
