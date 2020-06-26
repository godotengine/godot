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

#ifndef CPU_MEMORY_HPP
#define CPU_MEMORY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory.hpp"
#include "memory_desc_wrapper.hpp"

#include "cpu_engine.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_memory_t: public memory_t {
    cpu_memory_t(cpu_engine_t *engine, const memory_desc_t *md, void *handle)
        : memory_t(engine, md)
        , own_data_(handle == MKLDNN_NATIVE_HANDLE_ALLOCATE)
        , data_((char *)handle) {}

    cpu_memory_t(cpu_engine_t *engine, const memory_desc_t *md)
        : cpu_memory_t(engine, md, nullptr) {}

    ~cpu_memory_t() { if (own_data_) free(data_); }

    virtual status_t init() override {
        if (own_data_) {
            data_ = nullptr;
            const size_t size = memory_desc_wrapper(this->md()).size();
            if (size) {
                data_ = (char *)malloc(size, 64);
                if (data_ == nullptr)
                    return status::out_of_memory;
            }
        }
        return zero_pad();
    }

    cpu_engine_t *engine() const { return (cpu_engine_t *)memory_t::engine(); }

    virtual status_t get_data_handle(void **handle) const override {
        *handle = static_cast<void *>(data_);
        return status::success;
    }

    virtual mkldnn::impl::status_t set_data_handle(void *handle) override {
        if (own_data_) { free(data_); own_data_ = false; }
        data_ = static_cast<char *>(handle);
        return zero_pad();
    }

    virtual mkldnn::impl::status_t zero_pad() const override;

private:
    bool own_data_;
    char *data_;

    template <mkldnn::impl::data_type_t>
    mkldnn::impl::status_t typed_zero_pad() const;

    cpu_memory_t(const cpu_memory_t &) = delete;
    cpu_memory_t &operator=(const cpu_memory_t &) = delete;
    cpu_memory_t &operator=(cpu_memory_t &&) = delete;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
