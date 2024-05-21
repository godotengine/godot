// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SOLVERBASE_H
#define EIGEN_SOLVERBASE_H

namespace Eigen {

namespace internal {



} // end namespace internal

/** \class SolverBase
  * \brief A base class for matrix decomposition and solvers
  *
  * \tparam Derived the actual type of the decomposition/solver.
  *
  * Any matrix decomposition inheriting this base class provide the following API:
  *
  * \code
  * MatrixType A, b, x;
  * DecompositionType dec(A);
  * x = dec.solve(b);             // solve A   * x = b
  * x = dec.transpose().solve(b); // solve A^T * x = b
  * x = dec.adjoint().solve(b);   // solve A'  * x = b
  * \endcode
  *
  * \warning Currently, any other usage of transpose() and adjoint() are not supported and will produce compilation errors.
  *
  * \sa class PartialPivLU, class FullPivLU
  */
template<typename Derived>
class SolverBase : public EigenBase<Derived>
{
  public:

    typedef EigenBase<Derived> Base;
    typedef typename internal::traits<Derived>::Scalar Scalar;
    typedef Scalar CoeffReturnType;

    enum {
      RowsAtCompileTime = internal::traits<Derived>::RowsAtCompileTime,
      ColsAtCompileTime = internal::traits<Derived>::ColsAtCompileTime,
      SizeAtCompileTime = (internal::size_at_compile_time<internal::traits<Derived>::RowsAtCompileTime,
                                                          internal::traits<Derived>::ColsAtCompileTime>::ret),
      MaxRowsAtCompileTime = internal::traits<Derived>::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = internal::traits<Derived>::MaxColsAtCompileTime,
      MaxSizeAtCompileTime = (internal::size_at_compile_time<internal::traits<Derived>::MaxRowsAtCompileTime,
                                                             internal::traits<Derived>::MaxColsAtCompileTime>::ret),
      IsVectorAtCompileTime = internal::traits<Derived>::MaxRowsAtCompileTime == 1
                           || internal::traits<Derived>::MaxColsAtCompileTime == 1
    };

    /** Default constructor */
    SolverBase()
    {}

    ~SolverBase()
    {}

    using Base::derived;

    /** \returns an expression of the solution x of \f$ A x = b \f$ using the current decomposition of A.
      */
    template<typename Rhs>
    inline const Solve<Derived, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(derived().rows()==b.rows() && "solve(): invalid number of rows of the right hand side matrix b");
      return Solve<Derived, Rhs>(derived(), b.derived());
    }

    /** \internal the return type of transpose() */
    typedef typename internal::add_const<Transpose<const Derived> >::type ConstTransposeReturnType;
    /** \returns an expression of the transposed of the factored matrix.
      *
      * A typical usage is to solve for the transposed problem A^T x = b:
      * \code x = dec.transpose().solve(b); \endcode
      *
      * \sa adjoint(), solve()
      */
    inline ConstTransposeReturnType transpose() const
    {
      return ConstTransposeReturnType(derived());
    }

    /** \internal the return type of adjoint() */
    typedef typename internal::conditional<NumTraits<Scalar>::IsComplex,
                        CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, ConstTransposeReturnType>,
                        ConstTransposeReturnType
                     >::type AdjointReturnType;
    /** \returns an expression of the adjoint of the factored matrix
      *
      * A typical usage is to solve for the adjoint problem A' x = b:
      * \code x = dec.adjoint().solve(b); \endcode
      *
      * For real scalar types, this function is equivalent to transpose().
      *
      * \sa transpose(), solve()
      */
    inline AdjointReturnType adjoint() const
    {
      return AdjointReturnType(derived().transpose());
    }

  protected:
};

namespace internal {

template<typename Derived>
struct generic_xpr_base<Derived, MatrixXpr, SolverStorage>
{
  typedef SolverBase<Derived> type;

};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_SOLVERBASE_H
