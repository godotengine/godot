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

#include "dxil_spirv_nir.h"
#include "spirv_to_dxil.h"
#include "dxil_nir.h"
#include "nir_to_dxil.h"
#include "shader_enums.h"
#include "spirv/nir_spirv.h"
#include "util/blob.h"

#include "git_sha1.h"
#include "vulkan/vulkan.h"

#include "drivers/d3d12/d3d12_godot_nir_bridge.h"

static_assert(DXIL_SPIRV_SHADER_NONE == (int)MESA_SHADER_NONE, "must match");
static_assert(DXIL_SPIRV_SHADER_VERTEX == (int)MESA_SHADER_VERTEX, "must match");
static_assert(DXIL_SPIRV_SHADER_TESS_CTRL == (int)MESA_SHADER_TESS_CTRL, "must match");
static_assert(DXIL_SPIRV_SHADER_TESS_EVAL == (int)MESA_SHADER_TESS_EVAL, "must match");
static_assert(DXIL_SPIRV_SHADER_GEOMETRY == (int)MESA_SHADER_GEOMETRY, "must match");
static_assert(DXIL_SPIRV_SHADER_FRAGMENT == (int)MESA_SHADER_FRAGMENT, "must match");
static_assert(DXIL_SPIRV_SHADER_COMPUTE == (int)MESA_SHADER_COMPUTE, "must match");
static_assert(DXIL_SPIRV_SHADER_KERNEL == (int)MESA_SHADER_KERNEL, "must match");

bool
spirv_to_dxil(const uint32_t *words, size_t word_count,
              struct dxil_spirv_specialization *specializations,
              unsigned int num_specializations, dxil_spirv_shader_stage stage,
              const char *entry_point_name,
              enum dxil_shader_model shader_model_max,
              enum dxil_validator_version validator_version_max,
              const struct dxil_spirv_debug_options *dgb_opts,
              const struct dxil_spirv_runtime_conf *conf,
              const struct dxil_spirv_logger *logger,
              const GodotNirCallbacks *godot_nir_callbacks,
              struct dxil_spirv_object *out_dxil)
{
   if (stage == DXIL_SPIRV_SHADER_NONE || stage == DXIL_SPIRV_SHADER_KERNEL)
      return false;


   glsl_type_singleton_init_or_ref();

   struct nir_to_dxil_options opts = {
      .environment = DXIL_ENVIRONMENT_VULKAN,
      .shader_model_max = shader_model_max,
      .validator_version_max = validator_version_max,
      .godot_nir_callbacks = godot_nir_callbacks,
   };

   const struct spirv_to_nir_options *spirv_opts = dxil_spirv_nir_get_spirv_options();
   struct nir_shader_compiler_options nir_options = *dxil_get_nir_compiler_options();
   // We will manually handle base_vertex when vertex_id and instance_id have
   // have been already converted to zero-base.
   nir_options.lower_base_vertex = !conf->zero_based_vertex_instance_id;
   nir_options.lower_helper_invocation = opts.shader_model_max < SHADER_MODEL_6_6;

   nir_shader *nir = spirv_to_nir(
      words, word_count, (struct nir_spirv_specialization *)specializations,
      num_specializations, (gl_shader_stage)stage, entry_point_name,
      spirv_opts, &nir_options);
   if (!nir) {
      glsl_type_singleton_decref();
      return false;
   }

   nir_validate_shader(nir,
                       "Validate before feeding NIR to the DXIL compiler");

   dxil_spirv_nir_prep(nir);

   bool requires_runtime_data;
   dxil_spirv_nir_passes(nir, conf, &requires_runtime_data);

   if (dgb_opts->dump_nir)
      nir_print_shader(nir, stderr);

   struct dxil_logger logger_inner = {.priv = logger->priv,
                                      .log = logger->log};

   struct blob dxil_blob;
   if (!nir_to_dxil(nir, &opts, &logger_inner, &dxil_blob)) {
      if (dxil_blob.allocated)
         blob_finish(&dxil_blob);
      ralloc_free(nir);
      glsl_type_singleton_decref();
      return false;
   }

   ralloc_free(nir);
   out_dxil->metadata.requires_runtime_data = requires_runtime_data;
   blob_finish_get_buffer(&dxil_blob, &out_dxil->binary.buffer,
                          &out_dxil->binary.size);

   glsl_type_singleton_decref();
   return true;
}

void
spirv_to_dxil_free(struct dxil_spirv_object *dxil)
{
   free(dxil->binary.buffer);
}

uint64_t
spirv_to_dxil_get_version()
{
   const char sha1[] = MESA_GIT_SHA1;
   const char* dash = strchr(sha1, '-');
   if (dash) {
      return strtoull(dash + 1, NULL, 16);
   }
   return 0;
}
