// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

namespace bitUtil
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	inline bool IsPowerOfTwo(T x)
	{
		return (x & (x - 1)) == 0;
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	inline T RoundUpToMultiple(T numToRound, T multipleOf)
	{
		// make sure that this function is only called for unsigned types, and ensure that we want to round to a power-of-two
		static_assert(util::IsUnsigned<T>::value == true, "T must be an unsigned type.");
		PSD_ASSERT(IsPowerOfTwo(multipleOf), "Expected a power-of-two.");

		return (numToRound + (multipleOf - 1u)) & ~(multipleOf - 1u);
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	template <typename T>
	inline T RoundDownToMultiple(T numToRound, T multipleOf)
	{
		// make sure that this function is only called for unsigned types, and ensure that we want to round to a power-of-two
		static_assert(util::IsUnsigned<T>::value == true, "T must be an unsigned type.");
		PSD_ASSERT(IsPowerOfTwo(multipleOf), "Expected a power-of-two.");

		return numToRound & ~(multipleOf - 1u);
	}
}
