// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_PUSH_WARNING_LEVEL(0)
	#define MINIZ_HEADER_FILE_ONLY
	#include "Psdminiz.c"
	#undef MINIZ_HEADER_FILE_ONLY
PSD_POP_WARNING_LEVEL
