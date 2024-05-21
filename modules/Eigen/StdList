// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_STDLIST_MODULE_H
#define EIGEN_STDLIST_MODULE_H

#include "Core"
#include <list>

#if EIGEN_COMP_MSVC && EIGEN_OS_WIN64 && (EIGEN_MAX_STATIC_ALIGN_BYTES<=16) /* MSVC auto aligns up to 16 bytes in 64 bit builds */

#define EIGEN_DEFINE_STL_LIST_SPECIALIZATION(...)

#else

#include "src/StlSupport/StdList.h"

#endif

#endif // EIGEN_STDLIST_MODULE_H
