/*
 * Copyright 2015-2017 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
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
 */

#ifndef SPIRV_CROSS_EXTERNAL_INTERFACE_H
#define SPIRV_CROSS_EXTERNAL_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct spirv_cross_shader spirv_cross_shader_t;

struct spirv_cross_interface
{
	spirv_cross_shader_t *(*construct)(void);
	void (*destruct)(spirv_cross_shader_t *thiz);
	void (*invoke)(spirv_cross_shader_t *thiz);
};

void spirv_cross_set_stage_input(spirv_cross_shader_t *thiz, unsigned location, void *data, size_t size);

void spirv_cross_set_stage_output(spirv_cross_shader_t *thiz, unsigned location, void *data, size_t size);

void spirv_cross_set_push_constant(spirv_cross_shader_t *thiz, void *data, size_t size);

void spirv_cross_set_uniform_constant(spirv_cross_shader_t *thiz, unsigned location, void *data, size_t size);

void spirv_cross_set_resource(spirv_cross_shader_t *thiz, unsigned set, unsigned binding, void **data, size_t size);

const struct spirv_cross_interface *spirv_cross_get_interface(void);

typedef enum spirv_cross_builtin {
	SPIRV_CROSS_BUILTIN_POSITION = 0,
	SPIRV_CROSS_BUILTIN_FRAG_COORD = 1,
	SPIRV_CROSS_BUILTIN_WORK_GROUP_ID = 2,
	SPIRV_CROSS_BUILTIN_NUM_WORK_GROUPS = 3,
	SPIRV_CROSS_NUM_BUILTINS
} spirv_cross_builtin;

void spirv_cross_set_builtin(spirv_cross_shader_t *thiz, spirv_cross_builtin builtin, void *data, size_t size);

#define SPIRV_CROSS_NUM_DESCRIPTOR_SETS 4
#define SPIRV_CROSS_NUM_DESCRIPTOR_BINDINGS 16
#define SPIRV_CROSS_NUM_STAGE_INPUTS 16
#define SPIRV_CROSS_NUM_STAGE_OUTPUTS 16
#define SPIRV_CROSS_NUM_UNIFORM_CONSTANTS 32

enum spirv_cross_format
{
	SPIRV_CROSS_FORMAT_R8_UNORM = 0,
	SPIRV_CROSS_FORMAT_R8G8_UNORM = 1,
	SPIRV_CROSS_FORMAT_R8G8B8_UNORM = 2,
	SPIRV_CROSS_FORMAT_R8G8B8A8_UNORM = 3,

	SPIRV_CROSS_NUM_FORMATS
};

enum spirv_cross_wrap
{
	SPIRV_CROSS_WRAP_CLAMP_TO_EDGE = 0,
	SPIRV_CROSS_WRAP_REPEAT = 1,

	SPIRV_CROSS_NUM_WRAP
};

enum spirv_cross_filter
{
	SPIRV_CROSS_FILTER_NEAREST = 0,
	SPIRV_CROSS_FILTER_LINEAR = 1,

	SPIRV_CROSS_NUM_FILTER
};

enum spirv_cross_mipfilter
{
	SPIRV_CROSS_MIPFILTER_BASE = 0,
	SPIRV_CROSS_MIPFILTER_NEAREST = 1,
	SPIRV_CROSS_MIPFILTER_LINEAR = 2,

	SPIRV_CROSS_NUM_MIPFILTER
};

struct spirv_cross_miplevel
{
	const void *data;
	unsigned width, height;
	size_t stride;
};

struct spirv_cross_sampler_info
{
	const struct spirv_cross_miplevel *mipmaps;
	unsigned num_mipmaps;

	enum spirv_cross_format format;
	enum spirv_cross_wrap wrap_s;
	enum spirv_cross_wrap wrap_t;
	enum spirv_cross_filter min_filter;
	enum spirv_cross_filter mag_filter;
	enum spirv_cross_mipfilter mip_filter;
};

typedef struct spirv_cross_sampler_2d spirv_cross_sampler_2d_t;
spirv_cross_sampler_2d_t *spirv_cross_create_sampler_2d(const struct spirv_cross_sampler_info *info);
void spirv_cross_destroy_sampler_2d(spirv_cross_sampler_2d_t *samp);

#ifdef __cplusplus
}
#endif

#endif
