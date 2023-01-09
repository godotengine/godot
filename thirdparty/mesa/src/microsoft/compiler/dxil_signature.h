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

#ifndef DXIL_SIGNATURE_H
#define DXIL_SIGNATURE_H

#include "dxil_enums.h"
#include "nir.h"
#include "util/string_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* struct taken from DXILContainer
 * Enums values were replaced by uint32_t since they must occupy 32 bit
 */

struct dxil_signature_element {
   uint32_t stream;                   // Stream index (parameters must appear in non-decreasing stream order)
   uint32_t semantic_name_offset;     // Offset to char * stream from start of DxilProgramSignature.
   uint32_t semantic_index;           // Semantic Index
   uint32_t system_value;             // Semantic type. Similar to DxilSemantic::Kind, but a serialized rather than processing rep.
   uint32_t comp_type;                // Type of bits.
   uint32_t reg;                      // Register Index (row index)
   uint8_t  mask;                     // Mask (column allocation)
   union {                            // Unconditional cases useful for validation of shader linkage.
      uint8_t never_writes_mask;      // For an output signature, the shader the signature belongs to never
                                      // writes the masked components of the output register.
      uint8_t always_reads_mask;      // For an input signature, the shader the signature belongs to always
                                      // reads the masked components of the input register.
   };
   uint16_t pad;
   uint32_t min_precision;             // Minimum precision of input/output data
};

struct dxil_signature_record {
   struct dxil_signature_element elements[32];
   unsigned num_elements;
   const char *sysvalue;
   char *name;
   uint8_t sig_comp_type;
};

struct dxil_psv_sem_index_table {
   uint32_t data[80];
   uint32_t size;
};

struct dxil_psv_signature_element {
   uint32_t semantic_name_offset;          // Offset into StringTable
   uint32_t semantic_indexes_offset;       // Offset into PSVSemanticIndexTable, count == Rows
   uint8_t rows;                   // Number of rows this element occupies
   uint8_t start_row;               // Starting row of packing location if allocated
   uint8_t cols_and_start;           // 0:4 = Cols, 4:6 = StartCol, 6:7 == Allocated
   uint8_t semantic_kind;           // PSVSemanticKind
   uint8_t component_type;          // DxilProgramSigCompType
   uint8_t interpolation_mode;      // DXIL::InterpolationMode or D3D10_SB_INTERPOLATION_MODE
   uint8_t dynamic_mask_and_stream;   // 0:4 = DynamicIndexMask, 4:6 = OutputStream (0-3)
   uint8_t reserved;
};

struct dxil_psv_runtime_info_0 {
   union {
      struct {
         char output_position_present;
      } vs;

      struct {
         uint32_t input_control_point_count;
         uint32_t output_control_point_count;
         uint32_t tessellator_domain;
         uint32_t tessellator_output_primitive;
      } hs;

      struct {
         uint32_t input_control_point_count;
         char output_position_present;
         uint32_t tessellator_domain;
      } ds;

      struct {
         uint32_t input_primitive;
         uint32_t output_toplology;
         uint32_t output_stream_mask;
         char output_position_present;
      } gs;

      struct {
         char depth_output;
         char sample_frequency;
      } ps;

      /* Maximum sized defining the union size (MSInfo)*/
      struct {
         uint32_t dummy1[3];
         uint16_t dummy2[2];
      } dummy;
   };
   uint32_t min_expected_wave_lane_count;  // minimum lane count required, 0 if unused
   uint32_t max_expected_wave_lane_count;  // maximum lane count required, 0xffffffff if unused
};

struct dxil_psv_runtime_info_1 {
   struct dxil_psv_runtime_info_0 psv0;
   uint8_t shader_stage;              // PSVShaderKind
   uint8_t uses_view_id;
   union {
     uint16_t max_vertex_count;          // MaxVertexCount for GS only (max 1024)
     uint8_t sig_patch_const_or_prim_vectors;  // Output for HS; Input for DS; Primitive output for MS (overlaps MS1::SigPrimVectors)
     // struct { uint8_t dummy[2]; } fill;
   };

   // PSVSignatureElement counts
   uint8_t sig_input_elements;
   uint8_t sig_output_elements;
   uint8_t sig_patch_const_or_prim_elements;

   // Number of packed vectors per signature
   uint8_t sig_input_vectors;
   uint8_t sig_output_vectors[4];
};

struct dxil_psv_runtime_info_2 {
   struct dxil_psv_runtime_info_1 psv1;
   uint32_t num_threads_x;
   uint32_t num_threads_y;
   uint32_t num_threads_z;
};

struct dxil_mdnode;
struct dxil_module;

void
preprocess_signatures(struct dxil_module *mod, nir_shader *s, unsigned input_clip_size);

const struct dxil_mdnode *
get_signatures(struct dxil_module *mod);

#ifdef __cplusplus
}
#endif

#endif
