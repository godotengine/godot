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

#ifndef _JIT_UNI_REORDER_HPP
#define _JIT_UNI_REORDER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "cpu_primitive.hpp"
#include "cpu_reorder_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace tr {

constexpr int max_ndims = MKLDNN_MAX_NDIMS;

struct node_t {
    size_t n;
    ptrdiff_t is; // input stride
    ptrdiff_t os; // output stride
    ptrdiff_t ss; // scale stride
};

enum class scale_type_t { NONE, COMMON, MANY };

struct prb_t {
    data_type_t itype;
    data_type_t otype;
    int ndims;
    node_t nodes[max_ndims];
    ptrdiff_t ioff;
    ptrdiff_t ooff;
    scale_type_t scale_type;
    float beta;
};

status_t prb_init(prb_t &prb, const memory_desc_t &imd,
        const memory_desc_t &omd, const primitive_attr_t *attr);

/** sorts the problem nodes so that output strides come in ascending order */
void prb_normalize(prb_t &p);

/** folds nodes together if possible */
void prb_simplify(prb_t &p);

/** splits the node dim into two of sizes n1 and n / n1
 * @warning n must be multiple of n1 */
void prb_node_split(prb_t &p, int dim, size_t n1);

/** swaps d0 and d1 nodes */
void prb_node_swap(prb_t &p, int d0, int d1);

/** moves node d0 to the d1 position.
 * nodes (d0, d1] are shifted to the left if d0 < d1 or
 * to the right if d0 > d1 */
void prb_node_move(prb_t &p, int d0, int d1);

/** dumps the problem to stdout */
void prb_dump(const prb_t &p);

struct call_param_t {
    const void *in;
    void *out;
    const float *scale;
};

struct kernel_t {
    struct desc_t {
        int id;
        prb_t prb;
    };

    kernel_t(const desc_t &desc): desc_(desc), ker_(nullptr) {}
    void operator()(const call_param_t *c) const { assert(ker_); ker_(c); }
    virtual ~kernel_t() {}

    /** inits kernel descriptor:
     *      desc            -- kernel descriptor (output)
     *      prb             -- transposition problem (input)
     *      ndims_ker_max   -- limit the maximum number of dimensions kernel
     *                         will process (optional, 0 -- no limitation) */
    static status_t desc_init(desc_t &desc, const prb_t &prb,
            int ndims_ker_max = 0);

    /** creates kernel for the problem described in desc */
    static kernel_t *create(const desc_t &desc);

protected:
    const desc_t desc_;
    const prb_t &prb_ = desc_.prb;
    void (*ker_)(const call_param_t *);
};

/* TODO: add trans_t class */

}

/* for cpu reorder list */
status_t jit_uni_reorder_create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr,
        engine_t *src_engine, const memory_desc_t *src_md,
        engine_t *dst_engine, const memory_desc_t *dst_md);

}
}
}

#endif
