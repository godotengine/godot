/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef MEMORY_DESC_WRAPPER_HPP
#define MEMORY_DESC_WRAPPER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {

/** thin wrapper class over \struct memory_desc_t which allows easy
 * manipulations with underlying C structure, which is taken by reference */
struct memory_desc_wrapper: public c_compatible {
    const memory_desc_t *md_;

    /** constructor which takes a reference to a constant underlying C memory
     * descriptor \param md */
    memory_desc_wrapper(const memory_desc_t *md): md_(md) {}
    memory_desc_wrapper(const memory_desc_t &md): memory_desc_wrapper(&md) {}

    /* implementing attributes */
    int ndims() const { return md_->ndims; }
    const dims_t &dims() const { return md_->dims; }
    data_type_t data_type() const { return md_->data_type; }

    const dims_t &padded_dims() const { return md_->padded_dims; }
    const dims_t &padded_offsets() const { return md_->padded_offsets; }
    dim_t offset0() const { return md_->offset0; }

    format_kind_t format_kind() const { return md_->format_kind; }

    bool is_blocking_desc() const
    { return format_kind() == format_kind::blocked; }
    bool is_wino_desc() const
    { return format_kind() == format_kind::wino; }
    bool is_rnn_packed_desc() const
    { return format_kind() == format_kind::rnn_packed; }

    const blocking_desc_t &blocking_desc() const {
        assert(is_blocking_desc());
        return md_->format_desc.blocking;
    }
    const wino_desc_t &wino_desc() const {
        assert(is_wino_desc());
        return md_->format_desc.wino_desc;
    }
    const rnn_packed_desc_t &rnn_packed_desc() const {
        assert(is_rnn_packed_desc());
        return md_->format_desc.rnn_packed_desc;
    }

    const memory_extra_desc_t &extra() const { return md_->extra; }

    /* some useful function */

    /** returns the number of elements including padding if \param with_padding
     * is true, and the number of data elements otherwise */
    dim_t nelems(bool with_padding = false) const {
        if (is_zero()) return 0;
        return utils::array_product(
                with_padding ? padded_dims() : dims(), ndims());
    }

    /** returns true if memory descriptor is zero */
    bool is_zero() const { return ndims() == 0; }

    /** returns true if memory descriptor contains zero as one of its dim */
    bool has_zero_dim() const { return nelems() == 0; }

    /** return the size of data type (a shortcut) */
    size_t data_type_size() const
    { return types::data_type_size(data_type()); }

    /** return the size of data type of additional buffer */
    size_t additional_buffer_data_size() const {
        if (extra().flags & memory_extra_flags::compensation_conv_s8s8)
            return sizeof(int32_t);
        return 0;
    }

    /** return true if memory format has additional buffer */
    bool is_additional_buffer() const {
        return (extra().flags & memory_extra_flags::compensation_conv_s8s8);
    }

    /** returns the size of additional buffer */
    size_t additional_buffer_size() const {
        if (extra().flags & memory_extra_flags::compensation_conv_s8s8) {
            int cmask = extra().compensation_mask;
            assert(cmask == 1 || cmask == 3);
            dim_t prod = 1;
            for (int d = 0; d < ndims(); ++d)
                if (cmask & (1<<d)) prod *= padded_dims()[d];
            return prod * additional_buffer_data_size();
        }

        return 0;
    }

    /** returns the size required to store described memory
     * note: if offset0 != 0 returns 0 (need to specify the behavior) */
    size_t size() const {
        if (is_zero() || has_zero_dim() || format_kind() == format_kind::any)
            return 0;

        if (format_kind() == format_kind::wino) {
            return wino_desc().size;
        } else if (format_kind() == format_kind::rnn_packed) {
            return rnn_packed_desc().size;
        } else {
            if (offset0() != 0) return 0;

            dims_t blocks = {0};
            compute_blocks(blocks);

            const auto &bd = blocking_desc();

            size_t max_size = 0;
            for (int d = 0; d < ndims(); ++d)
                max_size = nstl::max<size_t>(max_size,
                        padded_dims()[d] / blocks[d] * bd.strides[d]);

            if (max_size == 1 && bd.inner_nblks != 0) {
                max_size = utils::array_product(bd.inner_blks, bd.inner_nblks);
            }

            return max_size * data_type_size() + additional_buffer_size();
        }
    }

    /** returns true if data is dense in memory */
    bool is_dense(bool with_padding = false) const {
        if (utils::one_of(format_kind(), format_kind::undef, format_kind::any))
            return false;
        return nelems(with_padding) * data_type_size() == size();
    }

    /** returns true if memory desc is fully defined */
    bool is_defined() const { return format_kind() != format_kind::any; }

    /** returns true if the only (potentially) padded dim is \param dim */
    bool only_padded_dim(int dim) const {
        for (int d = 0; d < ndims(); ++d)
            if (d != dim && dims()[d] != padded_dims()[d])
                return false;
        return true;
    }

    /** returns true if memory desc has blocked layout and block dims are 1s */
    bool is_plain() const {
        if (!is_blocking_desc()) return false;
        return blocking_desc().inner_nblks == 0;
    }

    /** returns overall block sizes */
    void compute_blocks(dims_t blocks) const {
        if (!is_blocking_desc()) {
            utils::array_set(blocks, 0, ndims());
            return;
        }

        utils::array_set(blocks, 1, ndims());

        const auto &bd = blocking_desc();
        for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
            blocks[bd.inner_idxs[iblk]] *= bd.inner_blks[iblk];
    }

    /* comparison section */

    bool operator==(const memory_desc_wrapper &rhs) const
    { return *this->md_ == *rhs.md_; }
    bool operator!=(const memory_desc_wrapper &rhs) const
    { return !operator==(rhs); }
    bool operator==(const memory_desc_t &rhs) const
    { return operator==(memory_desc_wrapper(rhs)); }
    bool operator!=(const memory_desc_t &rhs) const
    { return !operator==(rhs); }

    /** returns true if data (w/o padding if with_padding == false and w/
     * padding otherwise) have the same physical structure, i.e. dimensions,
     * strides, and blocked structure. Depending on with_data_type flag
     * data_type is taken or not taken into account. dim_start allows to check
     * similarity for the logical part of data [dim_start .. ndims()].
     * CAUTION: format kind any and undef are not similar to whatever, hence the
     * following statement might be true: lhs == rhs && !lhs.similar_to(rhs) */
    /* TODO: revise */
    bool similar_to(const memory_desc_wrapper &rhs,
            bool with_padding = true, bool with_data_type = true,
            int dim_start = 0) const;

    /** returns true if one memory can be reordered to another */
    bool consistent_with(const memory_desc_wrapper &rhs) const;

    /** returns true if the memory desc corresponds to the given format tag and
     * strides.
     * @sa memory_desc_matches_tag */
    bool matches_tag(format_tag_t tag, const dims_t strides = nullptr) const {
        return memory_desc_matches_tag(*md_, tag, strides);
    }

    /** returns matching tag (or undef if match is not found)
     * XXX: This is a workaround that eventually should go away! */
    template <typename... Tags>
    format_tag_t matches_one_of_tag(Tags ...tags) const {
        for (const auto tag: {tags...}) {
            if (memory_desc_matches_tag(*md_, tag))
                return tag;
        }
        return format_tag::undef;
    }

    /* offset section */

    /** returns physical offset by logical one. logical offset is represented by
     * an array \param pos. if \param is_pos_padded is true \param pos
     * represents the position in already padded area */
    dim_t off_v(const dims_t pos, bool is_pos_padded = false) const {
        assert(is_blocking_desc());
        const blocking_desc_t &blk = blocking_desc();

        dims_t pos_copy = {0};
        for (int d = 0; d < ndims(); ++d)
            pos_copy[d] = pos[d] + (is_pos_padded ? 0 : padded_offsets()[d]);

        dim_t phys_offset = offset0();

        if (blk.inner_nblks > 0) {
            dim_t blk_stride = 1;
            for (int iblk = blk.inner_nblks - 1; iblk >= 0; --iblk) {
                const int d = blk.inner_idxs[iblk];
                const dim_t p = pos_copy[d] % blk.inner_blks[iblk];

                phys_offset += p * blk_stride;

                pos_copy[d] /= blk.inner_blks[iblk];

                blk_stride *= blk.inner_blks[iblk];
            }
        }

        for (int d = 0; d < ndims(); ++d) {
            const dim_t p = pos_copy[d];
            phys_offset += p * blk.strides[d];
        }

        return phys_offset;
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a scalar \param l_offset. if \param is_pos_padded is true, \param
     * l_offset represents logical offset in already padded area */
    dim_t off_l(dim_t l_offset, bool is_pos_padded = false) const {
        assert(is_blocking_desc());
        dims_t pos;
        for (int rd = 0; rd < ndims(); ++rd) {
            const int d = ndims() - 1 - rd;
            const dim_t cur_dim = is_pos_padded ? padded_dims()[d] : dims()[d];
            pos[d] = l_offset % cur_dim;
            l_offset /= cur_dim;
        }
        return off_v(pos, is_pos_padded);
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a tuple of indices (\param xn, ..., \param x1, \param x0) */
    template<typename... Args>
    dim_t off(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = { args... };
        return off_v(pos, false);
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a tuple of indices (\param xn, ..., \param x1, \param x0) in already
     * padded area */
    template<typename... Args>
    dim_t off_padding(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = { args... };
        return off_v(pos, true);
    }

    /** returns physical offset by logical one. Logical offset is represented by
     * a tuple of block indices (\param bn, ..., \param b1, \param b0). It is a
     * user responsibility to adjust the result to get offset within blocks */
    template<typename ...Args>
    dim_t blk_off(Args... args) const {
        return _blk_off<sizeof...(args), Args...>(args...);
    }

    template<bool skip_first, typename T, typename ...Args>
    dim_t blk_off(T xn, Args... args) const {
        return skip_first
            ? blk_off<Args...>(args...)
            : blk_off<T, Args...>(xn, args...);
    }

    /* static functions section */
    /* TODO: replace with non-static, once md_ becomes non-const ref */

    static status_t compute_blocking(memory_desc_t &memory_desc,
            format_tag_t tag);

private:
    /* TODO: put logical_offset in utils */
    template<typename T>
    dim_t logical_offset(T x0) const { return x0; }

    template<typename T, typename... Args>
    dim_t logical_offset(T xn, Args... args) const {
        const size_t n_args = sizeof...(args);
        return xn * utils::array_product<n_args>(
                &dims()[ndims() - n_args]) + logical_offset(args...);
    }

    template<int ORIG_LEN, typename ...Void>
    dim_t _blk_off() const { return offset0(); }

    template<int ORIG_LEN, typename T, typename ...Args>
    dim_t _blk_off(T xc, Args ...args) const {
        assert(is_blocking_desc());
        constexpr int dc = ORIG_LEN - sizeof...(args) - 1;
        return xc * blocking_desc().strides[dc]
            + _blk_off<ORIG_LEN, Args...>(args...);
    }
};

inline bool memory_desc_wrapper::similar_to(const memory_desc_wrapper &rhs,
        bool with_padding, bool with_data_type, int dim_start) const {
    using namespace utils;

    if (one_of(format_kind(), format_kind::undef, format_kind::any))
        return false;
    if (is_wino_desc() || is_rnn_packed_desc())
        return false;

    const int ds = dim_start;
    const auto &blk = blocking_desc();
    const auto &r_blk = rhs.blocking_desc();

    return ndims() == rhs.ndims()
        && dim_start <= ndims() /* guard */
        && format_kind() == rhs.format_kind()
        && IMPLICATION(with_data_type, data_type() == rhs.data_type())
        && array_cmp(dims() + ds, rhs.dims() + ds, ndims() - ds)
        && array_cmp(blk.strides + ds, r_blk.strides + ds, ndims() - ds)
        && blk.inner_nblks == r_blk.inner_nblks
        && array_cmp(blk.inner_blks, r_blk.inner_blks, blk.inner_nblks)
        && array_cmp(blk.inner_idxs, r_blk.inner_idxs, blk.inner_nblks)
        && IMPLICATION(with_padding, true
                && array_cmp(padded_dims() + ds, rhs.padded_dims() + ds,
                    ndims() - ds)
                && array_cmp(padded_offsets() + ds, rhs.padded_offsets() + ds,
                    ndims() - ds));
}

inline bool memory_desc_wrapper::consistent_with(
        const memory_desc_wrapper &rhs) const {
    if (ndims() == rhs.ndims()) {
        for (int d = 0; d < ndims(); ++d) {
            if (dims()[d] != rhs.dims()[d]) return false;
        }
        return true;
    } else {
        /* TODO: revise.
         * is the following possible?
         * [1, a, b] <--reorder--> [a, b]
         * [a, 1, b] <--reorder--> [a, b]
         * not, at least for now */
        return false;
    }
}

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
