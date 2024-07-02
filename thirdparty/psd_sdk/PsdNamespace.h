// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


/// \def PSD_NAMESPACE_NAME
/// \ingroup Platform
/// \brief Macro used to configure the name of the PSD library namespace.
#define PSD_NAMESPACE_NAME						psd


/// \def PSD_NAMESPACE
/// \ingroup Platform
/// \brief Macro used to refer to a symbol in the PSD library namespace.
/// \sa PSD_USING_NAMESPACE
#define PSD_NAMESPACE							PSD_NAMESPACE_NAME


/// \def PSD_NAMESPACE_BEGIN
/// \ingroup Platform
/// \brief Macro used to open a namespace in the PSD library.
/// \sa PSD_NAMESPACE_END
#define PSD_NAMESPACE_BEGIN						namespace PSD_NAMESPACE_NAME {


/// \def PSD_NAMESPACE_END
/// \ingroup Platform
/// \brief Macro used to close a namespace previously opened with \ref PSD_NAMESPACE_BEGIN.
/// \sa PSD_NAMESPACE_BEGIN
#define PSD_NAMESPACE_END						}


/// \def PSD_USING_NAMESPACE
/// \ingroup Platfrom
/// \brief Macro used to make the PSD library namespace available in a translation unit.
/// \sa PSD_NAMESPACE
#define PSD_USING_NAMESPACE						using namespace PSD_NAMESPACE_NAME
