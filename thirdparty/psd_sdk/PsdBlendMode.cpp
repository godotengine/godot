// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdBlendMode.h"

#include "PsdKey.h"


PSD_NAMESPACE_BEGIN

namespace blendMode
{
	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	Enum KeyToEnum(uint32_t key)
	{
		#define IMPLEMENT_CASE(a, b, c, d, value)		case util::Key<a, b, c, d>::VALUE:	return value

		switch (key)
		{
			IMPLEMENT_CASE('p', 'a', 's', 's', PASS_THROUGH);
			IMPLEMENT_CASE('n', 'o', 'r', 'm', NORMAL);
			IMPLEMENT_CASE('d', 'i', 's', 's', DISSOLVE);
			IMPLEMENT_CASE('d', 'a', 'r', 'k', DARKEN);
			IMPLEMENT_CASE('m', 'u', 'l', ' ', MULTIPLY);
			IMPLEMENT_CASE('i', 'd', 'i', 'v', COLOR_BURN);
			IMPLEMENT_CASE('l', 'b', 'r', 'n', LINEAR_BURN);
			IMPLEMENT_CASE('d', 'k', 'C', 'l', DARKER_COLOR);
			IMPLEMENT_CASE('l', 'i', 't', 'e', LIGHTEN);
			IMPLEMENT_CASE('s', 'c', 'r', 'n', SCREEN);
			IMPLEMENT_CASE('d', 'i', 'v', ' ', COLOR_DODGE);
			IMPLEMENT_CASE('l', 'd', 'd', 'g', LINEAR_DODGE);
			IMPLEMENT_CASE('l', 'g', 'C', 'l', LIGHTER_COLOR);
			IMPLEMENT_CASE('o', 'v', 'e', 'r', OVERLAY);
			IMPLEMENT_CASE('s', 'L', 'i', 't', SOFT_LIGHT);
			IMPLEMENT_CASE('h', 'L', 'i', 't', HARD_LIGHT);
			IMPLEMENT_CASE('v', 'L', 'i', 't', VIVID_LIGHT);
			IMPLEMENT_CASE('l', 'L', 'i', 't', LINEAR_LIGHT);
			IMPLEMENT_CASE('p', 'L', 'i', 't', PIN_LIGHT);
			IMPLEMENT_CASE('h', 'M', 'i', 'x', HARD_MIX);
			IMPLEMENT_CASE('d', 'i', 'f', 'f', DIFFERENCE);
			IMPLEMENT_CASE('s', 'm', 'u', 'd', EXCLUSION);
			IMPLEMENT_CASE('f', 's', 'u', 'b', SUBTRACT);
			IMPLEMENT_CASE('f', 'd', 'i', 'v', DIVIDE);
			IMPLEMENT_CASE('h', 'u', 'e', ' ', HUE);
			IMPLEMENT_CASE('s', 'a', 't', ' ', SATURATION);
			IMPLEMENT_CASE('c', 'o', 'l', 'r', COLOR);
			IMPLEMENT_CASE('l', 'u', 'm', ' ', LUMINOSITY);
			default:
				return UNKNOWN;
		};

		#undef IMPLEMENT_CASE
	}


	// ---------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------------------
	const char* ToString(Enum value)
	{
		#define IMPLEMENT_CASE(value)		case value:	return #value

		switch (value)
		{
			IMPLEMENT_CASE(PASS_THROUGH);
			IMPLEMENT_CASE(NORMAL);
			IMPLEMENT_CASE(DISSOLVE);
			IMPLEMENT_CASE(DARKEN);
			IMPLEMENT_CASE(MULTIPLY);
			IMPLEMENT_CASE(COLOR_BURN);
			IMPLEMENT_CASE(LINEAR_BURN);
			IMPLEMENT_CASE(DARKER_COLOR);
			IMPLEMENT_CASE(LIGHTEN);
			IMPLEMENT_CASE(SCREEN);
			IMPLEMENT_CASE(COLOR_DODGE);
			IMPLEMENT_CASE(LINEAR_DODGE);
			IMPLEMENT_CASE(LIGHTER_COLOR);
			IMPLEMENT_CASE(OVERLAY);
			IMPLEMENT_CASE(SOFT_LIGHT);
			IMPLEMENT_CASE(HARD_LIGHT);
			IMPLEMENT_CASE(VIVID_LIGHT);
			IMPLEMENT_CASE(LINEAR_LIGHT);
			IMPLEMENT_CASE(PIN_LIGHT);
			IMPLEMENT_CASE(HARD_MIX);
			IMPLEMENT_CASE(DIFFERENCE);
			IMPLEMENT_CASE(EXCLUSION);
			IMPLEMENT_CASE(SUBTRACT);
			IMPLEMENT_CASE(DIVIDE);
			IMPLEMENT_CASE(HUE);
			IMPLEMENT_CASE(SATURATION);
			IMPLEMENT_CASE(COLOR);
			IMPLEMENT_CASE(LUMINOSITY);
			IMPLEMENT_CASE(UNKNOWN);
		};

		return "Unhandled blend mode";

		#undef IMPLEMENT_CASE
	}
}

PSD_NAMESPACE_END
