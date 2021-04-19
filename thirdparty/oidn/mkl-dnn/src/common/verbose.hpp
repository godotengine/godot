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

#ifndef VERBOSE_HPP
#define VERBOSE_HPP

#include <stdio.h>
#include <cinttypes>

#include "mkldnn_debug.h"
#include "c_types_map.hpp"
#include "utils.hpp"
#include "z_magic.hpp"

namespace mkldnn {
namespace impl {

struct verbose_t {
    int level;
};

const verbose_t *mkldnn_verbose();
double get_msec();
const char *get_isa_info();

#if !defined(DISABLE_VERBOSE)
#define MKLDNN_VERBOSE_BUF_LEN 1024
#else
#define MKLDNN_VERBOSE_BUF_LEN 1
#endif

void init_info(batch_normalization_pd_t *s, char *buffer);
void init_info(concat_pd_t *s, char *buffer);
void init_info(convolution_pd_t *s, char *buffer);
void init_info(deconvolution_pd_t *s, char *buffer);
void init_info(eltwise_pd_t *s, char *buffer);
void init_info(inner_product_pd_t *s, char *buffer);
void init_info(lrn_pd_t *s, char *buffer);
void init_info(pooling_pd_t *s, char *buffer);
void init_info(reorder_pd_t *s, char *buffer);
void init_info(rnn_pd_t *s, char *buffer);
void init_info(shuffle_pd_t *s, char *buffer);
void init_info(softmax_pd_t *s, char *buffer);
void init_info(sum_pd_t *s, char *buffer);

}
}

#endif
