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

#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <assert.h>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"

struct mkldnn_memory: public mkldnn::impl::c_compatible {
    mkldnn_memory(mkldnn::impl::engine_t *engine,
            const mkldnn::impl::memory_desc_t *md)
        : engine_(engine), md_(*md) {}
    virtual ~mkldnn_memory() {}

    /** allocates/initializes memory */
    virtual mkldnn::impl::status_t init() = 0;

    /** returns memory's engine */
    mkldnn::impl::engine_t *engine() const { return engine_; }
    /** returns memory's description */
    const mkldnn::impl::memory_desc_t *md() const { return &md_; }

    /** returns data handle */
    virtual mkldnn::impl::status_t get_data_handle(void **handle) const = 0;

    /** sets data handle */
    virtual mkldnn::impl::status_t set_data_handle(void *handle) = 0;

    /** zeros padding */
    virtual mkldnn::impl::status_t zero_pad() const
    { return mkldnn::impl::status::success; }

protected:
    mkldnn::impl::engine_t *engine_;
    const mkldnn::impl::memory_desc_t md_;

private:
    mkldnn_memory() = delete;
    mkldnn_memory(const mkldnn_memory &) = delete;
    mkldnn_memory(mkldnn_memory &&) = delete;
    mkldnn_memory &operator=(const mkldnn_memory &) = delete;
    mkldnn_memory &operator=(mkldnn_memory &&) = delete;
};

#endif
