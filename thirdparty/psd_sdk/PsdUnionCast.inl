// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

namespace util
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename TO, typename FROM>
	PSD_INLINE TO union_cast(FROM from)
	{
		static_assert(sizeof(TO) == sizeof(FROM), "Size mismatch. Cannot use a union_cast for types of different sizes.");

		union
		{
			FROM castFrom;
			TO castTo;
		};

		castFrom = from;
		return castTo;
	}
}
