// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include <type_traits>


PSD_NAMESPACE_BEGIN

namespace util
{
	/// \ingroup Util
	/// \brief Wrapper around std::is_unsigned, because it is not supported by all compilers.
	template <typename T>
	struct IsUnsigned
	{
#if PSD_USE_MSVC && PSD_USE_MSVC_VER <= 2008
		static const bool value = std::tr1::is_unsigned<T>::value;
#else
		static const bool value = std::is_unsigned<T>::value;
#endif
	};

	template <typename T> const bool IsUnsigned<T>::value;
}

PSD_NAMESPACE_END
