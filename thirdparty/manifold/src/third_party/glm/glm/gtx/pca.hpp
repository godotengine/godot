/// @ref gtx_pca
/// @file glm/gtx/pca.hpp
///
/// @see core (dependence)
/// @see ext_scalar_relational (dependence)
///
/// @defgroup gtx_pca GLM_GTX_pca
/// @ingroup gtx
///
/// Include <glm/gtx/pca.hpp> to use the features of this extension.
///
/// Implements functions required for fundamental 'princple component analysis' in 2D, 3D, and 4D:
///   1) Computing a covariance matrics from a list of _relative_ position vectors
///   2) Compute the eigenvalues and eigenvectors of the covariance matrics
/// This is useful, e.g., to compute an object-aligned bounding box from vertices of an object.
/// https://en.wikipedia.org/wiki/Principal_component_analysis
///
/// Example:
/// ```
/// std::vector<glm::dvec3> ptData;
/// // ... fill ptData with some point data, e.g. vertices
/// 
/// glm::dvec3 center = computeCenter(ptData);
/// 
/// glm::dmat3 covarMat = glm::computeCovarianceMatrix(ptData.data(), ptData.size(), center);
/// 
/// glm::dvec3 evals;
/// glm::dmat3 evecs;
/// int evcnt = glm::findEigenvaluesSymReal(covarMat, evals, evecs);
/// 
/// if(evcnt != 3)
///     // ... error handling
/// 
/// glm::sortEigenvalues(evals, evecs);
/// 
/// // ... now evecs[0] points in the direction (symmetric) of the largest spatial distribuion within ptData
/// ```

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../ext/scalar_relational.hpp"


#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_pca is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_pca extension included")
#	endif
#endif

namespace glm {
	/// @addtogroup gtx_pca
	/// @{

	/// Compute a covariance matrix form an array of relative coordinates `v` (e.g., relative to the center of gravity of the object)
	/// @param v Points to a memory holding `n` times vectors
	template<length_t D, typename T, qualifier Q>
	GLM_INLINE mat<D, D, T, Q> computeCovarianceMatrix(vec<D, T, Q> const* v, size_t n);

	/// Compute a covariance matrix form an array of absolute coordinates `v` and a precomputed center of gravity `c`
	/// @param v Points to a memory holding `n` times vectors
	template<length_t D, typename T, qualifier Q>
	GLM_INLINE mat<D, D, T, Q> computeCovarianceMatrix(vec<D, T, Q> const* v, size_t n, vec<D, T, Q> const& c);

	/// Compute a covariance matrix form a pair of iterators `b` (begin) and `e` (end) of a container with relative coordinates (e.g., relative to the center of gravity of the object)
	/// Dereferencing an iterator of type I must yield a `vec&lt;D, T, Q%gt;`
	template<length_t D, typename T, qualifier Q, typename I>
	GLM_FUNC_DECL mat<D, D, T, Q> computeCovarianceMatrix(I const& b, I const& e);

	/// Compute a covariance matrix form a pair of iterators `b` (begin) and `e` (end) of a container with absolute coordinates and a precomputed center of gravity `c`
	/// Dereferencing an iterator of type I must yield a `vec&lt;D, T, Q%gt;`
	template<length_t D, typename T, qualifier Q, typename I>
	GLM_FUNC_DECL mat<D, D, T, Q> computeCovarianceMatrix(I const& b, I const& e, vec<D, T, Q> const& c);

	/// Assuming the provided covariance matrix `covarMat` is symmetric and real-valued, this function find the `D` Eigenvalues of the matrix, and also provides the corresponding Eigenvectors.
	/// Note: the data in `outEigenvalues` and `outEigenvectors` are in matching order, i.e. `outEigenvector[i]` is the Eigenvector of the Eigenvalue `outEigenvalue[i]`.
	/// This is a numeric implementation to find the Eigenvalues, using 'QL decomposition` (variant of QR decomposition: https://en.wikipedia.org/wiki/QR_decomposition).
	/// @param covarMat A symmetric, real-valued covariance matrix, e.g. computed from `computeCovarianceMatrix`.
	/// @param outEigenvalues Vector to receive the found eigenvalues
	/// @param outEigenvectors Matrix to receive the found eigenvectors corresponding to the found eigenvalues, as column vectors
	/// @return The number of eigenvalues found, usually D if the precondition of the covariance matrix is met.
	template<length_t D, typename T, qualifier Q>
	GLM_FUNC_DECL unsigned int findEigenvaluesSymReal
	(
		mat<D, D, T, Q> const& covarMat,
		vec<D, T, Q>& outEigenvalues,
		mat<D, D, T, Q>& outEigenvectors
	);

	/// Sorts a group of Eigenvalues&Eigenvectors, for largest Eigenvalue to smallest Eigenvalue.
	/// The data in `outEigenvalues` and `outEigenvectors` are assumed to be matching order, i.e. `outEigenvector[i]` is the Eigenvector of the Eigenvalue `outEigenvalue[i]`.
	template<typename T, qualifier Q>
	GLM_INLINE void sortEigenvalues(vec<2, T, Q>& eigenvalues, mat<2, 2, T, Q>& eigenvectors);

	/// Sorts a group of Eigenvalues&Eigenvectors, for largest Eigenvalue to smallest Eigenvalue.
	/// The data in `outEigenvalues` and `outEigenvectors` are assumed to be matching order, i.e. `outEigenvector[i]` is the Eigenvector of the Eigenvalue `outEigenvalue[i]`.
	template<typename T, qualifier Q>
	GLM_INLINE void sortEigenvalues(vec<3, T, Q>& eigenvalues, mat<3, 3, T, Q>& eigenvectors);

	/// Sorts a group of Eigenvalues&Eigenvectors, for largest Eigenvalue to smallest Eigenvalue.
	/// The data in `outEigenvalues` and `outEigenvectors` are assumed to be matching order, i.e. `outEigenvector[i]` is the Eigenvector of the Eigenvalue `outEigenvalue[i]`.
	template<typename T, qualifier Q>
	GLM_INLINE void sortEigenvalues(vec<4, T, Q>& eigenvalues, mat<4, 4, T, Q>& eigenvectors);

	/// @}
}//namespace glm

#include "pca.inl"
