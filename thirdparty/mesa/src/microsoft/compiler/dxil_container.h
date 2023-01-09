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

#ifndef DXIL_CONTAINER_H
#define DXIL_CONTAINER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "util/blob.h"

#include "dxil_signature.h"

#define DXIL_MAX_PARTS 8
struct dxil_container {
   struct blob parts;
   unsigned part_offsets[DXIL_MAX_PARTS];
   unsigned num_parts;
};

enum dxil_resource_type {
  DXIL_RES_INVALID = 0,
  DXIL_RES_SAMPLER = 1,
  DXIL_RES_CBV = 2,
  DXIL_RES_SRV_TYPED = 3,
  DXIL_RES_SRV_RAW = 4,
  DXIL_RES_SRV_STRUCTURED = 5,
  DXIL_RES_UAV_TYPED = 6,
  DXIL_RES_UAV_RAW = 7,
  DXIL_RES_UAV_STRUCTURED,
  DXIL_RES_UAV_STRUCTURED_WITH_COUNTER,
  DXIL_RES_NUM_ENTRIES /* should always be last */
};

#define DXIL_FOURCC(ch0, ch1, ch2, ch3) ( \
  (uint32_t)(ch0)        | (uint32_t)(ch1) << 8 | \
  (uint32_t)(ch2) << 16  | (uint32_t)(ch3) << 24)

enum dxil_part_fourcc {
   DXIL_RDEF = DXIL_FOURCC('R', 'D', 'E', 'F'),
   DXIL_ISG1 = DXIL_FOURCC('I', 'S', 'G', '1'),
   DXIL_OSG1 = DXIL_FOURCC('O', 'S', 'G', '1'),
   DXIL_PSG1 = DXIL_FOURCC('P', 'S', 'G', '1'),
   DXIL_STAT = DXIL_FOURCC('S', 'T', 'A', 'T'),
   DXIL_ILDB = DXIL_FOURCC('I', 'L', 'D', 'B'),
   DXIL_ILDN = DXIL_FOURCC('I', 'L', 'D', 'N'),
   DXIL_SFI0 = DXIL_FOURCC('S', 'F', 'I', '0'),
   DXIL_PRIV = DXIL_FOURCC('P', 'R', 'I', 'V'),
   DXIL_RTS0 = DXIL_FOURCC('R', 'T', 'S', '0'),
   DXIL_DXIL = DXIL_FOURCC('D', 'X', 'I', 'L'),
   DXIL_PSV0 = DXIL_FOURCC('P', 'S', 'V', '0'),
   DXIL_RDAT = DXIL_FOURCC('R', 'D', 'A', 'T'),
   DXIL_HASH = DXIL_FOURCC('H', 'A', 'S', 'H'),
};

struct dxil_resource_v0 {
   uint32_t resource_type;
   uint32_t space;
   uint32_t lower_bound;
   uint32_t upper_bound;
};

struct dxil_resource_v1 {
   struct dxil_resource_v0 v0;
   uint32_t resource_kind;
   uint32_t resource_flags;
};

struct dxil_validation_state {
   struct dxil_psv_runtime_info_2 state;
   union {
      const struct dxil_resource_v0 *v0;
      const struct dxil_resource_v1 *v1;
   } resources;
   uint32_t num_resources;
};

void
dxil_container_init(struct dxil_container *c);

void
dxil_container_finish(struct dxil_container *c);

struct dxil_features;

bool
dxil_container_add_features(struct dxil_container *c,
                            const struct dxil_features *features);


bool
dxil_container_add_io_signature(struct dxil_container *c,
                                enum dxil_part_fourcc part,
                                unsigned num_records,
                                struct dxil_signature_record *io,
                                bool validator_7);

bool
dxil_container_add_state_validation(struct dxil_container *c,
                                    const struct dxil_module *m,
                                    struct dxil_validation_state *state);

bool
dxil_container_add_module(struct dxil_container *c,
                          const struct dxil_module *m,
                          uint64_t *extra_bit_offset);

bool
dxil_container_write(struct dxil_container *c, struct blob *blob, uint64_t *extra_bit_offset);

#ifdef __cplusplus
}
#endif

#endif
