// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \namespace exportColorMode
/// \brief A namespace denoting a color mode used for exporting a PSD document. Enumerator values denote the number of channels in the document as well as the PSD mode identifier.
/// \sa colorMode
namespace exportColorMode
{
	enum Enum
	{
		GRAYSCALE = 1,
		RGB = 3
	};
}

PSD_NAMESPACE_END
