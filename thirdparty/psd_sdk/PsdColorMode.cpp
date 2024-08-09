// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdColorMode.h"


PSD_NAMESPACE_BEGIN

namespace colorMode
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	const char* ToString(unsigned int value)
	{
		#define	IMPLEMENT_CASE(value)		case colorMode::value: return #value

		switch (value)
		{
			IMPLEMENT_CASE(BITMAP);
			IMPLEMENT_CASE(GRAYSCALE);
			IMPLEMENT_CASE(INDEXED);
			IMPLEMENT_CASE(RGB);
			IMPLEMENT_CASE(CMYK);
			IMPLEMENT_CASE(MULTICHANNEL);
			IMPLEMENT_CASE(DUOTONE);
			IMPLEMENT_CASE(LAB);
			default:
				return "Unknown";
		}

		#undef IMPLEMENT_CASE
	}
}

PSD_NAMESPACE_END
