// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

namespace util
{
	/// \ingroup Util
	/// \brief Casts from one type into another via a union.
	/// \details This type of cast is similar to a reinterpret_cast, but never violates the strict aliasing rule, and should
	/// therefore be used e.g. when trying to treat one type of data as a different, non-compatible one.
	/// \code
	///   float f = 1.0f;
	///   uint32_t bits = union_cast<uint32_t>(f);
	/// \endcode
	/// \remark Both types must have the same size.
	template <typename TO, typename FROM>
	PSD_INLINE TO union_cast(FROM from);
}

#include "PsdUnionCast.inl"

PSD_NAMESPACE_END
