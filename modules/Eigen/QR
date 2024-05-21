// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_QR_MODULE_H
#define EIGEN_QR_MODULE_H

#include "Core"

#include "src/Core/util/DisableStupidWarnings.h"

#include "Cholesky"
#include "Jacobi"
#include "Householder"

/** \defgroup QR_Module QR module
  *
  *
  *
  * This module provides various QR decompositions
  * This module also provides some MatrixBase methods, including:
  *  - MatrixBase::householderQr()
  *  - MatrixBase::colPivHouseholderQr()
  *  - MatrixBase::fullPivHouseholderQr()
  *
  * \code
  * #include <Eigen/QR>
  * \endcode
  */

#include "src/QR/HouseholderQR.h"
#include "src/QR/FullPivHouseholderQR.h"
#include "src/QR/ColPivHouseholderQR.h"
#include "src/QR/CompleteOrthogonalDecomposition.h"
#ifdef EIGEN_USE_LAPACKE
#ifdef EIGEN_USE_MKL
#include "mkl_lapacke.h"
#else
#include "src/misc/lapacke.h"
#endif
#include "src/QR/HouseholderQR_LAPACKE.h"
#include "src/QR/ColPivHouseholderQR_LAPACKE.h"
#endif

#include "src/Core/util/ReenableStupidWarnings.h"

#endif // EIGEN_QR_MODULE_H
/* vim: set filetype=cpp et sw=2 ts=2 ai: */
