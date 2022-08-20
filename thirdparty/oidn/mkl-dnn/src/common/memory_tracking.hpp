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

#ifndef MEMORY_TRACKING_HPP
#define MEMORY_TRACKING_HPP

#include <assert.h>
#include <unordered_map>

#include "nstl.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace memory_tracking {

/* Memory tracking capabilities
 *
 * The main purpose of this header file is to provide uniform way to register
 * required memory for a scratchpad at a primitive descriptor creation time
 * and then easily access it having only the base address of the scratchpad.
 *
 * Primitives might contain multiple disjoint parts that require temporary
 * buffers (known as scratchpad) during their execution. A primitive descriptor
 * should summarize all the needs into one single number -- the buffer size
 * that would be requested from a user. At execution time, the corresponding
 * primitive will receive a base pointer to a scratchpad. It then needs to
 * provide each part of algorithm the corresponding piece of memory. Three main
 * challenges here are:
 * 1. Track correct offset (from the base scratchpad address) for each piece
 * 2. Algorithm might require that different memory pieces to be aligned, so
 *    the scratchpad size is no more just a sum of size of the corresponding
 *    subparts.
 * 3. While a primitive is responsible for its scratchpad, the implementation
 *    might use some other basic blocks (e.g. cpu_reducer) that also require
 *    scratchpad memory. So there should be a simple way of passing the
 *    information back and force between the main algorithm (a primitive) and
 *    auxiliary stuff that lives completely separately from it (e.g. reducer).
 *
 * To address these challenges this header file provides 3 structures:
 * 1. registry_t  -- the class the stores the information about requested
 *                   memory. The information includes required size and desired
 *                   alignment for each piece. This class is also responsible
 *                   for computing the right offset to a given piece using the
 *                   base pointer.
 *                   This class is basically a ledger with all entries.
 *                   Lives in primitive descriptors.
 *
 * 2. registrar_t -- the interface to a registry_t to book memory. Used at
 *                   primitive descriptor creation time only. Contains a
 *                   reference to the corresponding *mutable* registry.
 *                   Always modifiable.
 *                   Allows chaining (using prefixes).
 *
 * 3. grantor_t   -- the interface to a registry_t to access memory. Used at
 *                   primitive execution time only. Contains a reference to
 *                   the corresponding *constant* registry and base pointer.
 *                   Always constant.
 *                   Allows chaining (using prefixes).
 *
 * Both registrar_t and grantor_t allow chaining with extra prefix provided.
 * The feature is useful when a primitive offload a part of computations to
 * some other primitives which require their own scratchpad space
 * (e.g. reducer). Prefixes are used to avoid key collision in cases when
 * multiple sub-primitive (e.g. multiple reducers) are used.
 *
 * A short example below demonstrates how to use aforementioned classes. In it
 * the main primitive is convolution that uses scratchpad for keeping padded
 * bias. It also needs a reducer, that needs its own space as well.
 *
 *  ``` c++
 *  struct reducer_t {
 *      static void init(registrar_t &scratchpad) {
 *          // preserve space for the reduction (one page aligned)
 *          scratchpad.book(key_space, sizeof(float) * 980 * 1024, 4096);
 *      }
 *
 *      void exec(const grantor_t &scratchpad) {
 *          // get the pointer to preserved space. scratchpad came from
 *          // upper primitive (convolution in this example)
 *          auto space = scratchpad.get<float>(key_reducer_space);
 *
 *          space[:] += ...;
 *      }
 *  };
 *
 *  struct conv_t {
 *      struct pd_t {
 *          void init() {
 *              registrar_t scratchpad(scratchpad_registry_);
 *
 *              // preserve a space for padded bias (using default alignment)
 *              scratchpad.book(key_conv_padded_bias, 128);
 *
 *              // create a proxy registrar for the reducer All entries made
 *              // by reducer would live in convolution's registry, but would
 *              // have their own `prefix`, so no interference with conv's
 *              // buffers.
 *              registrar_t reducer_scratchpad(scratchpad, prefix_reducer);
 *
 *              reducer_t::init(reducer_scratchpad);
 *          }
 *
 *          registry_t scratchpad_registry_;
 *      }
 *
 *      void exec() {
 *          // get the base pointer to a scratchpad memory from a user
 *          void *scratchpad_ptr = this->input(MKLDNN_MEM_SCRATCHPAD);
 *
 *          // create a grantor to the scratchpad (and provide the base
 *          // pointer).
 *          grantor_t scratchpad(pd()->scratchpad_registry_, scratchpad_ptr);
 *
 *          // access the padded_bias (need only key name and the grantor)
 *          auto padded_bias = scratchpad.get<float>(key_conv_padded_bias);
 *
 *          // to give the `right` grantor to reducer we need to add the
 *          // corresponding prefix, so that reducer would be able to access
 *          // its keys. The call is very similar to the one in pd_t::init
 *          // with only difference in types: grantor_t vs registrar_t.
 *          grantor_t reducer_scratchpad(scratchpad, prefix_reducer);
 *          reducer->exec(reducer_scratchpad);
 *      }
 *  };
 *  ```
 */


/* namespace with common keys and prefixes */
namespace names {
enum {
    key_none = 0,
    key_bnorm_tmp_mean,
    key_bnorm_tmp_var,
    key_bnorm_tmp_diff_ss,
    key_bnorm_tmp_stats,
    key_bnorm_reduction,
    key_concat_iptrs,
    key_concat_istrides,
    key_concat_nelems,
    key_concat_optrs,
    key_conv_adjusted_scales,
    key_conv_bia_reduction,
    key_conv_gemm_col,
    key_conv_gemm_imtr,
    key_conv_int_dat_in_acc_dt,
    key_conv_padded_bias,
    key_conv_rtus_space,
    key_conv_tr_diff_dst,
    key_conv_tr_diff_dst_bctx,
    key_conv_tr_src,
    key_conv_tr_src_bctx,
    key_conv_wei_reduction,
    key_conv_wei_bia_reduction,
    key_conv_wei_bia_reduction_bctx,
    key_iprod_int_dat_in_acc_dt,
    key_reducer_space,
    key_reducer_space_bctx,
    key_reorder_wino_plain,
    key_reorder_wino_transform_space,
    key_reorder_rnn_weights_quantization,
    key_reorder_rnn_weights_reduction,
    key_rnn_space,
    key_rnn_ptrs_bia,
    key_rnn_ptrs_wei_layer,
    key_rnn_ptrs_wei_iter,
    key_softmax_reduction,
    key_wino_U,
    key_wino_V,
    key_wino_M,
    key_barrier,
};

enum {
    prefix_none = 0,
    prefix_reducer_bia,
    prefix_reducer_wei,
};
}

// level 0: 00 00 00 xxx
// level 1: 00 00 aa xxx
// level 2: 00 aa bb xxx
// level 3: aa bb cc xxx
// max # of levels: 3 + 1 (base_level)
// here:
//      xxx        : [1 ..    MAX_KEY) : key
//      aa, bb, cc : [1 .. MAX_PREFIX) : prefixes for levels 1, 2, and 3

using key_t = uint32_t;
enum { MAX_KEY = (1u << 10), MAX_PREFIX = (1u << 7), };

/// generates global key based on a prefix and a local key
inline key_t make_key(key_t prefix, key_t key) { return prefix + key; }

/// generates global prefix based on the global parent and the local ones
inline key_t make_prefix(key_t parent_prefix, key_t prefix)
{ return MAX_PREFIX * parent_prefix + MAX_KEY * prefix; }

struct registrar_t;
struct grantor_t;

struct registry_t {
    void book(const key_t &key, size_t size, size_t alignment) {
        if (size == 0) return;
        assert(offset_map_.count(key) == 0);

        size = utils::rnd_up(size, minimal_alignment);
        alignment = nstl::max<size_t>(alignment, minimal_alignment);
        offset_map_[key] = entry_t{size_, size, alignment};

        size_ += size + alignment - minimal_alignment;
    }

    void *get(const key_t &key, void *base_ptr) const {
        if (base_ptr == nullptr) { assert(size() == 0); return nullptr; }
        if (offset_map_.count(key) != 1) return nullptr;

        const auto &e = offset_map_.at(key);
        base_ptr = utils::align_ptr<void>(base_ptr, minimal_alignment);
        char *ptr = (char *)base_ptr + e.offset;
        return utils::align_ptr<void>(ptr, e.alignment);
    }

    size_t size() const
    { return size_ > 0 ? size_ + minimal_alignment - 1 : 0; }

    registrar_t registrar();
    grantor_t grantor(void *base_ptr) const;

protected:
    enum { minimal_alignment = 64 };
    struct entry_t { size_t offset, size, alignment; };

    std::unordered_map<key_t, entry_t> offset_map_;
    size_t size_ = 0;
};

struct registrar_t {
    enum { default_alignment = 64 };

    registrar_t(registry_t &registry): registry_(registry), prefix_(0) {}
    registrar_t(registrar_t &parent, const key_t &prefix)
        : registry_(parent.registry_)
        , prefix_(make_prefix(parent.prefix_, prefix)) {}

    void book(const key_t &key, size_t size,
            size_t alignment = default_alignment)
    { registry_.book(make_key(prefix_, key), size, alignment); }

protected:
    registry_t &registry_;
    const key_t prefix_;
};

struct grantor_t {
    grantor_t(const registry_t &registry, void *base_ptr)
        : registry_(registry), prefix_(0), base_ptr_(base_ptr) {}
    grantor_t(const grantor_t &parent, const key_t &prefix)
        : registry_(parent.registry_)
        , prefix_(make_prefix(parent.prefix_, prefix))
        , base_ptr_(parent.base_ptr_) {}

    template <typename T = void> T *get(const key_t &key) const
    { return (T *)registry_.get(make_key(prefix_, key), base_ptr_); }

protected:
    const registry_t &registry_;
    const key_t prefix_;
    void *base_ptr_;
};

inline registrar_t registry_t::registrar() { return registrar_t(*this); }
inline grantor_t registry_t::grantor(void *base_ptr) const
{ return grantor_t(*this, base_ptr); }

}
}
}

#endif
