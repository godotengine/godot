// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \namespace blendMode
/// \brief A namespace holding all blend modes known by Photoshop.
namespace blendMode
{
	enum Enum
	{
		PASS_THROUGH,					///< Key = "pass"
		NORMAL,							///< Key = "norm"
		DISSOLVE,						///< Key = "diss"
		DARKEN,							///< Key = "dark"
		MULTIPLY,						///< Key = "mul "
		COLOR_BURN,						///< Key = "idiv"
		LINEAR_BURN,					///< Key = "lbrn"
		DARKER_COLOR,					///< Key = "dkCl"
		LIGHTEN,						///< Key = "lite"
		SCREEN,							///< Key = "scrn"
		COLOR_DODGE,					///< Key = "div "
		LINEAR_DODGE,					///< Key = "lddg"
		LIGHTER_COLOR,					///< Key = "lgCl"
		OVERLAY,						///< Key = "over"
		SOFT_LIGHT,						///< Key = "sLit"
		HARD_LIGHT,						///< Key = "hLit"
		VIVID_LIGHT,					///< Key = "vLit"
		LINEAR_LIGHT,					///< Key = "lLit"
		PIN_LIGHT,						///< Key = "pLit"
		HARD_MIX,						///< Key = "hMix"
		DIFFERENCE,						///< Key = "diff"
		EXCLUSION,						///< Key = "smud"
		SUBTRACT,						///< Key = "fsub"
		DIVIDE,							///< Key = "fdiv"
		HUE,							///< Key = "hue "
		SATURATION,						///< Key = "sat "
		COLOR,							///< Key = "colr"
		LUMINOSITY,						///< Key = "lum "

		UNKNOWN
	};

	/// Converts a given \a key to the corresponding \ref blendMode::Enum.
	Enum KeyToEnum(uint32_t key);

	/// Converts any of the \ref blendMode::Enum values into a string literal.
	const char* ToString(Enum value);
}

PSD_NAMESPACE_END
