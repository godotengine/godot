/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MKLDNN_THREAD_PARALLEL_ND_HPP
#define MKLDNN_THREAD_PARALLEL_ND_HPP

/* This header must be included by mkldnn_thread.hpp only */

/* Functions:
 *  - parallel(nthr, f)              - executes f in parallel using at most
 *                                     nthr threads. If nthr equals 0
 *                                     mkldnn_get_max_threads() threads is
 *                                     used
 *  - for_nd(ithr, nthr, dims..., f) - multidimensional for loop for already
 *                                     created threads
 *  - parallel_nd(dims..., f)        - creates a parallel section and then
 *                                     calls for_nd
 *  - parallel_nd_in_omp(dims..., f) - queries current nthr and ithr and then
 *                                     calls for_nd (mostly for convenience)
 */

namespace mkldnn {
namespace impl {

/* general parallelization */
template <typename F>
void parallel(int nthr, F f) {
    if (nthr == 0) nthr = mkldnn_get_max_threads();
#if MKLDNN_THR == MKLDNN_THR_SEQ
    assert(nthr == 1);
    f(0, 1);
#elif MKLDNN_THR == MKLDNN_THR_OMP
    if (nthr == 1) { f(0, 1); return; }
#   pragma omp parallel num_threads(nthr)
    f(mkldnn_get_thread_num(), mkldnn_get_num_threads());
#elif MKLDNN_THR == MKLDNN_THR_TBB
    if (nthr == 1) { f(0, 1); return; }
    tbb::parallel_for(0, nthr, [&](int ithr) { f(ithr, nthr); }, tbb::static_partitioner());
#endif
}

/* for_nd section */

template <typename T0, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, F f) {
    T0 start{0}, end{0};
    balance211(D0, nthr, ithr, start, end);
    for (T0 d0 = start; d0 < end; ++d0) f(d0);
}

template <typename T0, typename T1, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1);
        utils::nd_iterator_step(d0, D0, d1, D1);
    }
}

template <typename T0, typename T1, typename T2, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0}; T2 d2{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
         typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
         typename T5, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, const T5 &D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0}; T5 d5{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4,
            d5, D5);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4, d5);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

// Skip a lambda function in the parameter pack.
template <typename T>
constexpr size_t get_work_amount(const T &v) { return 1; }
template <typename T, typename ...Args>
constexpr size_t get_work_amount(const T &v, Args &&...args)
{ return (size_t)v * get_work_amount(utils::forward<Args>(args)...); }

/* parallel_nd and parallel_nd_in_omp section */

#if MKLDNN_THR != MKLDNN_THR_TBB
template <typename ...Args>
void parallel_nd(Args &&...args) {
#if MKLDNN_THR == MKLDNN_THR_SEQ
    for_nd(0, 1, utils::forward<Args>(args)...);
#elif MKLDNN_THR == MKLDNN_THR_OMP
    const bool do_parallel = get_work_amount(utils::forward<Args>(args)...) > 1;
#   pragma omp parallel if (do_parallel)
    {
        const int nthr = !do_parallel ? 1 : mkldnn_get_num_threads();
        const int ithr = !do_parallel ? 0 : mkldnn_get_thread_num();
        for_nd(ithr, nthr, utils::forward<Args>(args)...);
    }
#endif
}
#else // MKLDNN_THR != MKLDNN_THR_TBB

// gcc 4.8 has a bug with passing parameter pack to lambdas.
// So have to explicitly instantiate all the cases.

template <typename T0, typename F>
void parallel_nd(const T0 &D0, F f) {
    const size_t work_amount = (size_t)D0;
    if (work_amount == 0) return;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, work_amount), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t iwork = r.begin(); iwork != r.end(); ++iwork) {
            f(T0(iwork));
        }
    }, tbb::static_partitioner());
}

template <typename T0, typename T1, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, work_amount), [&](const tbb::blocked_range<size_t>& r) {
        T0 d0{0}; T1 d1{0};
        utils::nd_iterator_init(r.begin(), d0, D0, d1, D1);
        for (size_t iwork = r.begin(); iwork != r.end(); ++iwork) {
            f(d0, d1);
            utils::nd_iterator_step(d0, D0, d1, D1);
        }
    }, tbb::static_partitioner());
}

template <typename T0, typename T1, typename T2, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, work_amount), [&](const tbb::blocked_range<size_t>& r) {
        T0 d0{0}; T1 d1{0}; T2 d2{0};
        utils::nd_iterator_init(r.begin(), d0, D0, d1, D1, d2, D2);
        for (size_t iwork = r.begin(); iwork != r.end(); ++iwork) {
            f(d0, d1, d2);
            utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
        }
    }, tbb::static_partitioner());
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, work_amount), [&](const tbb::blocked_range<size_t>& r) {
        T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0};
        utils::nd_iterator_init(r.begin(), d0, D0, d1, D1, d2, D2, d3, D3);
        for (size_t iwork = r.begin(); iwork != r.end(); ++iwork) {
            f(d0, d1, d2, d3);
            utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
        }
    }, tbb::static_partitioner());
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
         typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
        const T4 &D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, work_amount), [&](const tbb::blocked_range<size_t>& r) {
        T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0};
        utils::nd_iterator_init(r.begin(), d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
        for (size_t iwork = r.begin(); iwork != r.end(); ++iwork) {
            f(d0, d1, d2, d3, d4);
            utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
        }
    }, tbb::static_partitioner());
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
         typename T5, typename F>
void parallel_nd(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
        const T4 &D4, const T5 &D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, work_amount), [&](const tbb::blocked_range<size_t>& r) {
        T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0}; T5 d5{0};
        utils::nd_iterator_init(r.begin(), d0, D0, d1, D1, d2, D2, d3, D3, d4, D4,
                d5, D5);
        for (size_t iwork = r.begin(); iwork != r.end(); ++iwork) {
            f(d0, d1, d2, d3, d4, d5);
            utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
        }
    }, tbb::static_partitioner());
}
#endif

template <typename ...Args>
void parallel_nd_in_omp(Args &&...args) {
#if MKLDNN_THR == MKLDNN_THR_SEQ
    for_nd(0, 1, utils::forward<Args>(args)...);
#elif MKLDNN_THR == MKLDNN_THR_OMP
    for_nd(mkldnn_get_thread_num(), mkldnn_get_num_threads(),
            utils::forward<Args>(args)...);
#elif MKLDNN_THR == MKLDNN_THR_TBB
    assert(!"unsupported parallel_nd_in_omp()");
#endif
}

} // namespace impl
} // namespace mkldnn

#endif
