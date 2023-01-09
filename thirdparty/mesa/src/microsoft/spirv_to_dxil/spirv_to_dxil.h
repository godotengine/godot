/*
 * Copyright © Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef SPIRV_TO_DXIL_H
#define SPIRV_TO_DXIL_H

#include "dxil_versions.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct GodotNirCallbacks GodotNirCallbacks;

#ifdef __cplusplus
extern "C" {
#endif

// NB: I've copy and pasted some types into this header so we don't have to
// include other headers. This will surely break if any of these types change.

// Copy of gl_shader_stage
typedef enum {
   DXIL_SPIRV_SHADER_NONE = -1,
   DXIL_SPIRV_SHADER_VERTEX = 0,
   DXIL_SPIRV_SHADER_TESS_CTRL = 1,
   DXIL_SPIRV_SHADER_TESS_EVAL = 2,
   DXIL_SPIRV_SHADER_GEOMETRY = 3,
   DXIL_SPIRV_SHADER_FRAGMENT = 4,
   DXIL_SPIRV_SHADER_COMPUTE = 5,
   DXIL_SPIRV_SHADER_KERNEL = 14,
} dxil_spirv_shader_stage;

// Copy of nir_spirv_const_value
typedef union {
   bool b;
   float f32;
   double f64;
   int8_t i8;
   uint8_t u8;
   int16_t i16;
   uint16_t u16;
   int32_t i32;
   uint32_t u32;
   int64_t i64;
   uint64_t u64;
} dxil_spirv_const_value;

// Copy of nir_spirv_specialization
struct dxil_spirv_specialization {
   uint32_t id;
   dxil_spirv_const_value value;
   bool defined_on_module;
};

struct dxil_spirv_metadata {
   bool requires_runtime_data;
};

struct dxil_spirv_object {
   struct dxil_spirv_metadata metadata;
   struct {
      void *buffer;
      size_t size;
   } binary;
};

/* This struct describes the layout of data expected in the CB bound to
 * runtime_data_cbv during compute shader execution */
struct dxil_spirv_compute_runtime_data {
   /* Total number of groups dispatched (i.e. value passed to Dispatch()) */
   uint32_t group_count_x;
   uint32_t group_count_y;
   uint32_t group_count_z;
};

#define DXIL_SPIRV_Y_FLIP_MASK BITFIELD_MASK(DXIL_SPIRV_MAX_VIEWPORT)
#define DXIL_SPIRV_Z_FLIP_SHIFT DXIL_SPIRV_MAX_VIEWPORT
#define DXIL_SPIRV_Z_FLIP_MASK BITFIELD_RANGE(DXIL_SPIRV_Z_FLIP_SHIFT, DXIL_SPIRV_MAX_VIEWPORT)

/* This struct describes the layout of data expected in the CB bound to
 * runtime_data_cbv during vertex stages */
struct dxil_spirv_vertex_runtime_data {
   uint32_t first_vertex;
   uint32_t base_instance;
   bool is_indexed_draw;
   // The lower 16bits of this mask encode Y-flips (one bit per viewport)
   // The higher 16bits of this maks encode Z-flips (one bit per viewport)
   union {
      uint32_t yz_flip_mask;
      struct {
         uint16_t y_flip_mask;
         uint16_t z_flip_mask;
      };
   };
   uint32_t draw_id;
   float viewport_width;
   float viewport_height;
   uint32_t view_index;
};

enum dxil_spirv_yz_flip_mode {
   DXIL_SPIRV_YZ_FLIP_NONE = 0,
   // Y-flip is unconditional: pos.y = -pos.y
   // Z-flip is unconditional: pos.z = -pos.z + 1.0f
   DXIL_SPIRV_Y_FLIP_UNCONDITIONAL = 1 << 0,
   DXIL_SPIRV_Z_FLIP_UNCONDITIONAL = 1 << 1,
   DXIL_SPIRV_YZ_FLIP_UNCONDITIONAL = DXIL_SPIRV_Y_FLIP_UNCONDITIONAL | DXIL_SPIRV_Z_FLIP_UNCONDITIONAL,
   // Y-flip/Z-flip info are passed through a sysval
   DXIL_SPIRV_Y_FLIP_CONDITIONAL = 1 << 2,
   DXIL_SPIRV_Z_FLIP_CONDITIONAL = 1 << 3,
   DXIL_SPIRV_YZ_FLIP_CONDITIONAL = DXIL_SPIRV_Y_FLIP_CONDITIONAL | DXIL_SPIRV_Z_FLIP_CONDITIONAL,
};

#define DXIL_SPIRV_MAX_VIEWPORT 16

struct dxil_spirv_runtime_conf {
   struct {
      uint32_t register_space;
      uint32_t base_shader_register;
   } runtime_data_cbv;

   struct {
      uint32_t register_space;
      uint32_t base_shader_register;
   } push_constant_cbv;

   // Set true if vertex and instance ids have already been converted to
   // zero-based. Otherwise, runtime_data will be required to lower them.
   bool zero_based_vertex_instance_id;

   struct {
      // mode != DXIL_SPIRV_YZ_FLIP_NONE only valid on vertex/geometry stages.
      enum dxil_spirv_yz_flip_mode mode;

      // Y/Z flip masks (one bit per viewport)
      uint16_t y_mask;
      uint16_t z_mask;
   } yz_flip;

   // The caller supports read-only images to be turned into SRV accesses,
   // which allows us to run the nir_opt_access() pass
   bool read_only_images_as_srvs;

   // Force sample rate shading on a fragment shader
   bool force_sample_rate_shading;

   // View index needs to be lowered to a UBO lookup
   bool lower_view_index;
   // View index also needs to be forwarded to RT layer output
   bool lower_view_index_to_rt_layer;
};

struct dxil_spirv_debug_options {
   bool dump_nir;
};

typedef void (*dxil_spirv_msg_callback)(void *priv, const char *msg);

struct dxil_spirv_logger {
   void *priv;
   dxil_spirv_msg_callback log;
};

/**
 * Compile a SPIR-V module into DXIL.
 * \param  words  SPIR-V module to compile
 * \param  word_count  number of words in the SPIR-V module
 * \param  specializations  specialization constants to compile with the shader
 * \param  num_specializations  number of specialization constants
 * \param  stage  shader stage
 * \param  entry_point_name  name of shader entrypoint
 * \param  conf  configuration for spriv_to_dxil
 * \param  out_dxil  will contain the DXIL bytes on success (call spirv_to_dxil_free after use)
 * \return  true if compilation succeeded
 */
bool
spirv_to_dxil(const uint32_t *words, size_t word_count,
              struct dxil_spirv_specialization *specializations,
              unsigned int num_specializations, dxil_spirv_shader_stage stage,
              const char *entry_point_name,
              enum dxil_shader_model shader_model_max,
              enum dxil_validator_version validator_version_max,
              const struct dxil_spirv_debug_options *debug_options,
              const struct dxil_spirv_runtime_conf *conf,
              const struct dxil_spirv_logger *logger,
              const GodotNirCallbacks *godot_nir_callbacks,
              struct dxil_spirv_object *out_dxil);

/**
 * Free the buffer allocated by spirv_to_dxil.
 */
void
spirv_to_dxil_free(struct dxil_spirv_object *dxil);

uint64_t
spirv_to_dxil_get_version(void);

#ifdef __cplusplus
}
#endif

#endif
