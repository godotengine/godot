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
#ifndef PRIMITIVE_ITERATOR_HPP
#define PRIMITIVE_ITERATOR_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"

struct mkldnn_primitive_desc_iterator: public mkldnn::impl::c_compatible {
    using pd_create_f = mkldnn::impl::engine_t::primitive_desc_create_f;

    mkldnn_primitive_desc_iterator(mkldnn::impl::engine_t *engine, const mkldnn::impl::op_desc_t *op_desc,
            const mkldnn::impl::primitive_attr_t *attr, const mkldnn::impl::primitive_desc_t *hint_fwd_pd)
        : idx_(-1), engine_(engine), pd_(nullptr), op_desc_(op_desc)
        , attr_(attr ? *attr : mkldnn::impl::primitive_attr_t()), hint_fwd_pd_(hint_fwd_pd)
        , impl_list_(engine_->get_implementation_list()), last_idx_(0)
    {
        while (impl_list_[last_idx_] != nullptr) ++last_idx_;
    }
    ~mkldnn_primitive_desc_iterator() { if (pd_) delete pd_; }

    bool operator==(const mkldnn::impl::primitive_desc_iterator_t& rhs) const
    { return idx_ == rhs.idx_ && engine_ == rhs.engine_; }
    bool operator!=(const mkldnn::impl::primitive_desc_iterator_t& rhs) const
    { return !operator==(rhs); }

    mkldnn::impl::primitive_desc_iterator_t end() const
    { return mkldnn_primitive_desc_iterator(engine_, last_idx_); }

    mkldnn::impl::primitive_desc_iterator_t &operator++() {
        if (pd_) { delete pd_; pd_ = nullptr; }
        while (++idx_ != last_idx_) {
            auto s = impl_list_[idx_](&pd_, op_desc_, &attr_, engine_,
                    hint_fwd_pd_);
            if (s ==  mkldnn::impl::status::success) break;
        }
        return *this;
    }

    mkldnn::impl::primitive_desc_t *operator*() const {
        if (*this == end() || pd_ == nullptr) return nullptr;
        return pd_->clone();
    }

protected:
    int idx_;
    mkldnn::impl::engine_t *engine_;
    mkldnn::impl::primitive_desc_t *pd_;
    const mkldnn::impl::op_desc_t *op_desc_;
    const mkldnn::impl::primitive_attr_t attr_;
    const mkldnn::impl::primitive_desc_t *hint_fwd_pd_;
    const pd_create_f *impl_list_;
    int last_idx_;

private:
    mkldnn_primitive_desc_iterator(mkldnn::impl::engine_t *engine, int last_idx)
        : idx_(last_idx), engine_(engine), pd_(nullptr)
        , op_desc_(nullptr), hint_fwd_pd_(nullptr)
        , impl_list_(nullptr), last_idx_(last_idx) {}
};

#endif
