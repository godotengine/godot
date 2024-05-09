/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// Portions of this code are derived from
//
// Manjunath Kudlur's Carbon library
//
// and
//
// Based on Boost.Phoenix v1.2
// Copyright (c) 2001-2002 Joel de Guzman

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/functional/actor.h>
#include <thrust/tuple.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace functional
{

// XXX we should just take a single EvalTuple
template<typename Eval0,
         typename Eval1  = thrust::null_type,
         typename Eval2  = thrust::null_type,
         typename Eval3  = thrust::null_type,
         typename Eval4  = thrust::null_type,
         typename Eval5  = thrust::null_type,
         typename Eval6  = thrust::null_type,
         typename Eval7  = thrust::null_type,
         typename Eval8  = thrust::null_type,
         typename Eval9  = thrust::null_type,
         typename Eval10 = thrust::null_type>
  class composite;

template<typename Eval0, typename Eval1>
  class composite<
    Eval0,
    Eval1,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type
  >
{
  public:
    template<typename Env>
      struct result
    {
      typedef typename Eval0::template result<
        thrust::tuple<
          typename Eval1::template result<Env>::type
        >
      >::type type;
    };

    __host__ __device__
    composite(const Eval0 &e0, const Eval1 &e1)
      : m_eval0(e0),
        m_eval1(e1)
    {}

    template<typename Env>
    __host__ __device__
    typename result<Env>::type
    eval(const Env &x) const
    {
      typename Eval1::template result<Env>::type result1 = m_eval1.eval(x);
      return m_eval0.eval(thrust::tie(result1));
    }

  private:
    Eval0 m_eval0;
    Eval1 m_eval1;
}; // end composite<Eval0,Eval1>

template<typename Eval0, typename Eval1, typename Eval2>
  class composite<
    Eval0,
    Eval1,
    Eval2,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type,
    thrust::null_type
  >
{
  public:
    template<typename Env>
      struct result
    {
      typedef typename Eval0::template result<
        thrust::tuple<
          typename Eval1::template result<Env>::type,
          typename Eval2::template result<Env>::type
        >
      >::type type;
    };

    __host__ __device__
    composite(const Eval0 &e0, const Eval1 &e1, const Eval2 &e2)
      : m_eval0(e0),
        m_eval1(e1),
        m_eval2(e2)
    {}

    template<typename Env>
    __host__ __device__
    typename result<Env>::type
    eval(const Env &x) const
    {
      typename Eval1::template result<Env>::type result1 = m_eval1.eval(x);
      typename Eval2::template result<Env>::type result2 = m_eval2.eval(x);
      return m_eval0.eval(thrust::tie(result1,result2));
    }

  private:
    Eval0 m_eval0;
    Eval1 m_eval1;
    Eval2 m_eval2;
}; // end composite<Eval0,Eval1,Eval2>

template<typename Eval0, typename Eval1>
__host__ __device__
  actor<composite<Eval0,Eval1> > compose(const Eval0 &e0, const Eval1 &e1)
{
  return actor<composite<Eval0,Eval1> >(composite<Eval0,Eval1>(e0,e1));
}

template<typename Eval0, typename Eval1, typename Eval2>
__host__ __device__
  actor<composite<Eval0,Eval1,Eval2> > compose(const Eval0 &e0, const Eval1 &e1, const Eval2 &e2)
{
  return actor<composite<Eval0,Eval1,Eval2> >(composite<Eval0,Eval1,Eval2>(e0,e1,e2));
}

} // end functional
} // end detail
THRUST_NAMESPACE_END

