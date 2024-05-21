// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_MATRIX_VECTOR_H
#define EIGEN_GENERAL_MATRIX_VECTOR_H

namespace Eigen {

namespace internal {

/* Optimized col-major matrix * vector product:
 * This algorithm processes the matrix per vertical panels,
 * which are then processed horizontaly per chunck of 8*PacketSize x 1 vertical segments.
 *
 * Mixing type logic: C += alpha * A * B
 *  |  A  |  B  |alpha| comments
 *  |real |cplx |cplx | no vectorization
 *  |real |cplx |real | alpha is converted to a cplx when calling the run function, no vectorization
 *  |cplx |real |cplx | invalid, the caller has to do tmp: = A * B; C += alpha*tmp
 *  |cplx |real |real | optimal case, vectorization possible via real-cplx mul
 *
 * The same reasoning apply for the transposed case.
 */
template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index,LhsScalar,LhsMapper,ColMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>
{
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

enum {
  Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable
              && int(packet_traits<LhsScalar>::size)==int(packet_traits<RhsScalar>::size),
  LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
  RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
  ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1
};

typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
typedef typename packet_traits<ResScalar>::type  _ResPacket;

typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

EIGEN_DONT_INLINE static void run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  RhsScalar alpha);
};

template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<Index,LhsScalar,LhsMapper,ColMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>::run(
  Index rows, Index cols,
  const LhsMapper& alhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  RhsScalar alpha)
{
  EIGEN_UNUSED_VARIABLE(resIncr);
  eigen_internal_assert(resIncr==1);

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate propoer code.
  LhsMapper lhs(alhs);

  conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
  conj_helper<LhsPacket,RhsPacket,ConjugateLhs,ConjugateRhs> pcj;
  const Index lhsStride = lhs.stride();
  // TODO: for padded aligned inputs, we could enable aligned reads
  enum { LhsAlignment = Unaligned };

  const Index n8 = rows-8*ResPacketSize+1;
  const Index n4 = rows-4*ResPacketSize+1;
  const Index n3 = rows-3*ResPacketSize+1;
  const Index n2 = rows-2*ResPacketSize+1;
  const Index n1 = rows-1*ResPacketSize+1;

  // TODO: improve the following heuristic:
  const Index block_cols = cols<128 ? cols : (lhsStride*sizeof(LhsScalar)<32000?16:4);
  ResPacket palpha = pset1<ResPacket>(alpha);

  for(Index j2=0; j2<cols; j2+=block_cols)
  {
    Index jend = numext::mini(j2+block_cols,cols);
    Index i=0;
    for(; i<n8; i+=ResPacketSize*8)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
                c1 = pset1<ResPacket>(ResScalar(0)),
                c2 = pset1<ResPacket>(ResScalar(0)),
                c3 = pset1<ResPacket>(ResScalar(0)),
                c4 = pset1<ResPacket>(ResScalar(0)),
                c5 = pset1<ResPacket>(ResScalar(0)),
                c6 = pset1<ResPacket>(ResScalar(0)),
                c7 = pset1<ResPacket>(ResScalar(0));

      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*0,j),b0,c0);
        c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*1,j),b0,c1);
        c2 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*2,j),b0,c2);
        c3 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*3,j),b0,c3);
        c4 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*4,j),b0,c4);
        c5 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*5,j),b0,c5);
        c6 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*6,j),b0,c6);
        c7 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*7,j),b0,c7);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      pstoreu(res+i+ResPacketSize*1, pmadd(c1,palpha,ploadu<ResPacket>(res+i+ResPacketSize*1)));
      pstoreu(res+i+ResPacketSize*2, pmadd(c2,palpha,ploadu<ResPacket>(res+i+ResPacketSize*2)));
      pstoreu(res+i+ResPacketSize*3, pmadd(c3,palpha,ploadu<ResPacket>(res+i+ResPacketSize*3)));
      pstoreu(res+i+ResPacketSize*4, pmadd(c4,palpha,ploadu<ResPacket>(res+i+ResPacketSize*4)));
      pstoreu(res+i+ResPacketSize*5, pmadd(c5,palpha,ploadu<ResPacket>(res+i+ResPacketSize*5)));
      pstoreu(res+i+ResPacketSize*6, pmadd(c6,palpha,ploadu<ResPacket>(res+i+ResPacketSize*6)));
      pstoreu(res+i+ResPacketSize*7, pmadd(c7,palpha,ploadu<ResPacket>(res+i+ResPacketSize*7)));
    }
    if(i<n4)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
                c1 = pset1<ResPacket>(ResScalar(0)),
                c2 = pset1<ResPacket>(ResScalar(0)),
                c3 = pset1<ResPacket>(ResScalar(0));

      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*0,j),b0,c0);
        c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*1,j),b0,c1);
        c2 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*2,j),b0,c2);
        c3 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*3,j),b0,c3);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      pstoreu(res+i+ResPacketSize*1, pmadd(c1,palpha,ploadu<ResPacket>(res+i+ResPacketSize*1)));
      pstoreu(res+i+ResPacketSize*2, pmadd(c2,palpha,ploadu<ResPacket>(res+i+ResPacketSize*2)));
      pstoreu(res+i+ResPacketSize*3, pmadd(c3,palpha,ploadu<ResPacket>(res+i+ResPacketSize*3)));

      i+=ResPacketSize*4;
    }
    if(i<n3)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
                c1 = pset1<ResPacket>(ResScalar(0)),
                c2 = pset1<ResPacket>(ResScalar(0));

      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*0,j),b0,c0);
        c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*1,j),b0,c1);
        c2 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*2,j),b0,c2);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      pstoreu(res+i+ResPacketSize*1, pmadd(c1,palpha,ploadu<ResPacket>(res+i+ResPacketSize*1)));
      pstoreu(res+i+ResPacketSize*2, pmadd(c2,palpha,ploadu<ResPacket>(res+i+ResPacketSize*2)));

      i+=ResPacketSize*3;
    }
    if(i<n2)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
                c1 = pset1<ResPacket>(ResScalar(0));

      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*0,j),b0,c0);
        c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*1,j),b0,c1);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      pstoreu(res+i+ResPacketSize*1, pmadd(c1,palpha,ploadu<ResPacket>(res+i+ResPacketSize*1)));
      i+=ResPacketSize*2;
    }
    if(i<n1)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0));
      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+0,j),b0,c0);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      i+=ResPacketSize;
    }
    for(;i<rows;++i)
    {
      ResScalar c0(0);
      for(Index j=j2; j<jend; j+=1)
        c0 += cj.pmul(lhs(i,j), rhs(j,0));
      res[i] += alpha*c0;
    }
  }
}

/* Optimized row-major matrix * vector product:
 * This algorithm processes 4 rows at onces that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 *
 * Mixing type logic:
 *  - alpha is always a complex (or converted to a complex)
 *  - no vectorization
 */
template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index,LhsScalar,LhsMapper,RowMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>
{
typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

enum {
  Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable
              && int(packet_traits<LhsScalar>::size)==int(packet_traits<RhsScalar>::size),
  LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
  RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
  ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1
};

typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
typedef typename packet_traits<ResScalar>::type  _ResPacket;

typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

EIGEN_DONT_INLINE static void run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  ResScalar alpha);
};

template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<Index,LhsScalar,LhsMapper,RowMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>::run(
  Index rows, Index cols,
  const LhsMapper& alhs,
  const RhsMapper& rhs,
  ResScalar* res, Index resIncr,
  ResScalar alpha)
{
  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate propoer code.
  LhsMapper lhs(alhs);

  eigen_internal_assert(rhs.stride()==1);
  conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
  conj_helper<LhsPacket,RhsPacket,ConjugateLhs,ConjugateRhs> pcj;

  // TODO: fine tune the following heuristic. The rationale is that if the matrix is very large,
  //       processing 8 rows at once might be counter productive wrt cache.
  const Index n8 = lhs.stride()*sizeof(LhsScalar)>32000 ? 0 : rows-7;
  const Index n4 = rows-3;
  const Index n2 = rows-1;

  // TODO: for padded aligned inputs, we could enable aligned reads
  enum { LhsAlignment = Unaligned };

  Index i=0;
  for(; i<n8; i+=8)
  {
    ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
              c1 = pset1<ResPacket>(ResScalar(0)),
              c2 = pset1<ResPacket>(ResScalar(0)),
              c3 = pset1<ResPacket>(ResScalar(0)),
              c4 = pset1<ResPacket>(ResScalar(0)),
              c5 = pset1<ResPacket>(ResScalar(0)),
              c6 = pset1<ResPacket>(ResScalar(0)),
              c7 = pset1<ResPacket>(ResScalar(0));

    Index j=0;
    for(; j+LhsPacketSize<=cols; j+=LhsPacketSize)
    {
      RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j,0);

      c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+0,j),b0,c0);
      c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+1,j),b0,c1);
      c2 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+2,j),b0,c2);
      c3 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+3,j),b0,c3);
      c4 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+4,j),b0,c4);
      c5 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+5,j),b0,c5);
      c6 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+6,j),b0,c6);
      c7 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+7,j),b0,c7);
    }
    ResScalar cc0 = predux(c0);
    ResScalar cc1 = predux(c1);
    ResScalar cc2 = predux(c2);
    ResScalar cc3 = predux(c3);
    ResScalar cc4 = predux(c4);
    ResScalar cc5 = predux(c5);
    ResScalar cc6 = predux(c6);
    ResScalar cc7 = predux(c7);
    for(; j<cols; ++j)
    {
      RhsScalar b0 = rhs(j,0);

      cc0 += cj.pmul(lhs(i+0,j), b0);
      cc1 += cj.pmul(lhs(i+1,j), b0);
      cc2 += cj.pmul(lhs(i+2,j), b0);
      cc3 += cj.pmul(lhs(i+3,j), b0);
      cc4 += cj.pmul(lhs(i+4,j), b0);
      cc5 += cj.pmul(lhs(i+5,j), b0);
      cc6 += cj.pmul(lhs(i+6,j), b0);
      cc7 += cj.pmul(lhs(i+7,j), b0);
    }
    res[(i+0)*resIncr] += alpha*cc0;
    res[(i+1)*resIncr] += alpha*cc1;
    res[(i+2)*resIncr] += alpha*cc2;
    res[(i+3)*resIncr] += alpha*cc3;
    res[(i+4)*resIncr] += alpha*cc4;
    res[(i+5)*resIncr] += alpha*cc5;
    res[(i+6)*resIncr] += alpha*cc6;
    res[(i+7)*resIncr] += alpha*cc7;
  }
  for(; i<n4; i+=4)
  {
    ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
              c1 = pset1<ResPacket>(ResScalar(0)),
              c2 = pset1<ResPacket>(ResScalar(0)),
              c3 = pset1<ResPacket>(ResScalar(0));

    Index j=0;
    for(; j+LhsPacketSize<=cols; j+=LhsPacketSize)
    {
      RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j,0);

      c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+0,j),b0,c0);
      c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+1,j),b0,c1);
      c2 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+2,j),b0,c2);
      c3 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+3,j),b0,c3);
    }
    ResScalar cc0 = predux(c0);
    ResScalar cc1 = predux(c1);
    ResScalar cc2 = predux(c2);
    ResScalar cc3 = predux(c3);
    for(; j<cols; ++j)
    {
      RhsScalar b0 = rhs(j,0);

      cc0 += cj.pmul(lhs(i+0,j), b0);
      cc1 += cj.pmul(lhs(i+1,j), b0);
      cc2 += cj.pmul(lhs(i+2,j), b0);
      cc3 += cj.pmul(lhs(i+3,j), b0);
    }
    res[(i+0)*resIncr] += alpha*cc0;
    res[(i+1)*resIncr] += alpha*cc1;
    res[(i+2)*resIncr] += alpha*cc2;
    res[(i+3)*resIncr] += alpha*cc3;
  }
  for(; i<n2; i+=2)
  {
    ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
              c1 = pset1<ResPacket>(ResScalar(0));

    Index j=0;
    for(; j+LhsPacketSize<=cols; j+=LhsPacketSize)
    {
      RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j,0);

      c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+0,j),b0,c0);
      c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+1,j),b0,c1);
    }
    ResScalar cc0 = predux(c0);
    ResScalar cc1 = predux(c1);
    for(; j<cols; ++j)
    {
      RhsScalar b0 = rhs(j,0);

      cc0 += cj.pmul(lhs(i+0,j), b0);
      cc1 += cj.pmul(lhs(i+1,j), b0);
    }
    res[(i+0)*resIncr] += alpha*cc0;
    res[(i+1)*resIncr] += alpha*cc1;
  }
  for(; i<rows; ++i)
  {
    ResPacket c0 = pset1<ResPacket>(ResScalar(0));
    Index j=0;
    for(; j+LhsPacketSize<=cols; j+=LhsPacketSize)
    {
      RhsPacket b0 = rhs.template load<RhsPacket,Unaligned>(j,0);
      c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i,j),b0,c0);
    }
    ResScalar cc0 = predux(c0);
    for(; j<cols; ++j)
    {
      cc0 += cj.pmul(lhs(i,j), rhs(j,0));
    }
    res[i*resIncr] += alpha*cc0;
  }
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_GENERAL_MATRIX_VECTOR_H
