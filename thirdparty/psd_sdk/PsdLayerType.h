// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \namespace layerType
/// \brief A namespace holding layer types known by Photoshop.
namespace layerType
{
	enum Enum
	{
		ANY = 0,								///< Any other type of layer.
		OPEN_FOLDER = 1,						///< Open folder.
		CLOSED_FOLDER = 2,						///< Closed folder.
		SECTION_DIVIDER = 3						///< Bounding section divider, hidden in the UI.
	};
}

PSD_NAMESPACE_END
