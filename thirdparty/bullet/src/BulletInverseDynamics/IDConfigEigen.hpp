///@file Configuration for Inverse Dynamics Library with Eigen
#ifndef INVDYNCONFIG_EIGEN_HPP_
#define INVDYNCONFIG_EIGEN_HPP_
#define btInverseDynamics btInverseDynamicsEigen
#ifdef BT_USE_DOUBLE_PRECISION
// choose double/single precision version
typedef double idScalar;
#else
typedef float idScalar;
#endif

// use std::vector for arrays
#include <vector>
// this is to make it work with C++2003, otherwise we could do this
// template <typename T>
// using idArray = std::vector<T>;
template <typename T>
struct idArray {
	typedef std::vector<T> type;
};
typedef std::vector<int>::size_type idArrayIdx;
// default to standard malloc/free
#include <cstdlib>
#define ID_DECLARE_ALIGNED_ALLOCATOR() EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// Note on interfaces:
// Eigen::Matrix has data(), to get c-array storage
// HOWEVER: default storage is column-major!
#define ID_LINEAR_MATH_USE_EIGEN
#include "Eigen/Eigen"
#include "details/IDEigenInterface.hpp"
#endif
