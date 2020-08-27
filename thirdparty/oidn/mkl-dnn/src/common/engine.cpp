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

#include "mkldnn.h"
#include "engine.hpp"
#include "nstl.hpp"

#include "c_types_map.hpp"
#include "../cpu/cpu_engine.hpp"

namespace mkldnn {
namespace impl {

engine_factory_t *engine_factories[] = {
    &cpu::engine_factory,
    nullptr,
};

static inline engine_factory_t *get_engine_factory(engine_kind_t kind) {
    for (engine_factory_t **ef = engine_factories; *ef; ef++)
        if ((*ef)->kind() == kind)
            return *ef;
    return nullptr;
}

}
}

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

size_t mkldnn_engine_get_count(engine_kind_t kind) {
    engine_factory_t *ef = get_engine_factory(kind);
    return ef != nullptr ? ef->count() : 0;
}

status_t mkldnn_engine_create(engine_t **engine,
        engine_kind_t kind, size_t index) {
    if (engine == nullptr)
        return invalid_arguments;

    engine_factory_t *ef = get_engine_factory(kind);
    if (ef == nullptr || index >= ef->count())
        return invalid_arguments;

    return ef->engine_create(engine, index);
}

status_t mkldnn_engine_get_kind(engine_t *engine, engine_kind_t *kind) {
    if (engine == nullptr)
        return invalid_arguments;
    *kind = engine->kind();
    return success;
}

status_t mkldnn_engine_destroy(engine_t *engine) {
    /* TODO: engine->dec_ref_count(); */
    delete engine;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
