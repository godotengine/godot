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

#ifndef STREAM_HPP
#define STREAM_HPP

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"

struct mkldnn_stream: public mkldnn::impl::c_compatible {
    mkldnn_stream(mkldnn::impl::engine_t *engine, unsigned flags)
        : engine_(engine), flags_(flags) {}
    virtual ~mkldnn_stream() {}

    /** returns stream's engine */
    mkldnn::impl::engine_t *engine() const { return engine_; }

    /** returns stream's kind */
    unsigned flags() const { return flags_; }

protected:
    mkldnn::impl::engine_t *engine_;
    unsigned flags_;
};

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
