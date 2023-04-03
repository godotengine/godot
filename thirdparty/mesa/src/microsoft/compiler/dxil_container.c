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

#include "dxil_container.h"
#include "dxil_module.h"

#include "util/u_debug.h"

#include <assert.h>

const uint32_t DXIL_DXBC = DXIL_FOURCC('D', 'X', 'B', 'C');

void
dxil_container_init(struct dxil_container *c)
{
   blob_init(&c->parts);
   c->num_parts = 0;
}

void
dxil_container_finish(struct dxil_container *c)
{
   blob_finish(&c->parts);
}

static bool
add_part_header(struct dxil_container *c,
                enum dxil_part_fourcc fourcc,
                uint32_t part_size)
{
   assert(c->parts.size < UINT_MAX);
   unsigned offset = (unsigned)c->parts.size;
   if (!blob_write_bytes(&c->parts, &fourcc, sizeof(fourcc)) ||
       !blob_write_bytes(&c->parts, &part_size, sizeof(part_size)))
      return false;

   assert(c->num_parts < DXIL_MAX_PARTS);
   c->part_offsets[c->num_parts++] = offset;
   return true;
}

static bool
add_part(struct dxil_container *c,
         enum dxil_part_fourcc fourcc,
         const void *part_data, uint32_t part_size)
{
   return add_part_header(c, fourcc, part_size) &&
          blob_write_bytes(&c->parts, part_data, part_size);
}

bool
dxil_container_add_features(struct dxil_container *c,
                            const struct dxil_features *features)
{
   /* DXIL feature info is a bitfield packed into a uint64_t. */
   static_assert(sizeof(struct dxil_features) <= sizeof(uint64_t),
                 "Expected dxil_features to fit into a uint64_t");
   uint64_t bits = 0;
   memcpy(&bits, features, sizeof(struct dxil_features));
   return add_part(c, DXIL_SFI0, &bits, sizeof(uint64_t));
}

typedef struct {
   struct {
      const char *name;
      uint32_t offset;
   } entries[DXIL_SHADER_MAX_IO_ROWS];
   uint32_t num_entries;
} name_offset_cache_t;

static uint32_t
get_semantic_name_offset(name_offset_cache_t *cache, const char *name,
                         struct _mesa_string_buffer *buf, uint32_t buf_offset,
                         bool validator_7)
{
   uint32_t offset = buf->length + buf_offset;

   /* DXC doesn't de-duplicate arbitrary semantic names until validator 1.7, only SVs. */
   if (validator_7 || strncmp(name, "SV_", 3) == 0) {
      /* consider replacing this with a binary search using rb_tree */
      for (unsigned i = 0; i < cache->num_entries; ++i) {
         if (!strcmp(name, cache->entries[i].name))
            return cache->entries[i].offset;
      }

      cache->entries[cache->num_entries].name = name;
      cache->entries[cache->num_entries].offset = offset;
      ++cache->num_entries;
   }
   _mesa_string_buffer_append_len(buf, name, strlen(name) + 1);

   return offset;
}

static uint32_t
collect_semantic_names(unsigned num_records,
                       struct dxil_signature_record *io_data,
                       struct _mesa_string_buffer *buf,
                       uint32_t buf_offset,
                       bool validator_7)
{
   name_offset_cache_t cache;
   cache.num_entries = 0;

   for (unsigned i = 0; i < num_records; ++i) {
      struct dxil_signature_record *io = &io_data[i];
      uint32_t offset = get_semantic_name_offset(&cache, io->name, buf, buf_offset, validator_7);
      for (unsigned j = 0; j < io->num_elements; ++j)
         io->elements[j].semantic_name_offset = offset;
   }
   if (validator_7 && buf->length % sizeof(uint32_t) != 0) {
      unsigned padding_to_add = sizeof(uint32_t) - (buf->length % sizeof(uint32_t));
      char padding[sizeof(uint32_t)] = { 0 };
      _mesa_string_buffer_append_len(buf, padding, padding_to_add);
   }
   return buf_offset + buf->length;
}

bool
dxil_container_add_io_signature(struct dxil_container *c,
                                enum dxil_part_fourcc part,
                                unsigned num_records,
                                struct dxil_signature_record *io_data,
                                bool validator_7)
{
   struct {
      uint32_t param_count;
      uint32_t param_offset;
   } header;
   header.param_count = 0;
   uint32_t fixed_size = sizeof(header);
   header.param_offset = fixed_size;

   bool retval = true;

   for (unsigned i = 0; i < num_records; ++i) {
      /* TODO:
       * - Here we need to check whether the value is actually part of the
       * signature */
      fixed_size += sizeof(struct dxil_signature_element) * io_data[i].num_elements;
      header.param_count += io_data[i].num_elements;
   }

   struct _mesa_string_buffer *names =
         _mesa_string_buffer_create(NULL, 1024);

   uint32_t last_offset = collect_semantic_names(num_records, io_data,
                                                 names, fixed_size,
                                                 validator_7);


   if (!add_part_header(c, part, last_offset) ||
       !blob_write_bytes(&c->parts, &header, sizeof(header))) {
      retval = false;
      goto cleanup;
   }

   /* write all parts */
   for (unsigned i = 0; i < num_records; ++i)
      for (unsigned j = 0; j < io_data[i].num_elements; ++j) {
         if (!blob_write_bytes(&c->parts, &io_data[i].elements[j],
                              sizeof(io_data[i].elements[j]))) {
            retval = false;
            goto cleanup;
         }
      }

   /* write all names */

   if (!blob_write_bytes(&c->parts, names->buf, names->length))
      retval = false;

cleanup:
   _mesa_string_buffer_destroy(names);
   return retval;
}

static uint32_t
compute_mask_dwords_from_vectors(uint32_t vectors)
{
   return ((vectors + 7) >> 3);
}
static uint32_t
compute_input_output_table_dwords(unsigned input_vectors, unsigned output_vectors)
{
   return compute_mask_dwords_from_vectors(output_vectors) * input_vectors * 4;
}

bool
dxil_container_add_state_validation(struct dxil_container *c,
                                    const struct dxil_module *m,
                                    struct dxil_validation_state *state)
{
   uint32_t psv_size = m->minor_validator >= 6 ?
      sizeof(struct dxil_psv_runtime_info_2) :
      sizeof(struct dxil_psv_runtime_info_1);
   uint32_t resource_bind_info_size = m->minor_validator >= 6 ?
      sizeof(struct dxil_resource_v1) : sizeof(struct dxil_resource_v0);
   uint32_t dxil_pvs_sig_size = sizeof(struct dxil_psv_signature_element);
   uint32_t resource_count = state->num_resources;

   uint32_t size = psv_size + 2 * sizeof(uint32_t);
   if (resource_count > 0) {
      size += sizeof (uint32_t) +
              resource_bind_info_size * resource_count;
   }
   uint32_t string_table_size = (m->sem_string_table->length + 3) & ~3u;
   size  += sizeof(uint32_t) + string_table_size;

   size  += sizeof(uint32_t) + m->sem_index_table.size * sizeof(uint32_t);

   if (m->num_sig_inputs || m->num_sig_outputs || m->num_sig_patch_consts) {
      size  += sizeof(uint32_t);
   }

   size += dxil_pvs_sig_size * m->num_sig_inputs;
   size += dxil_pvs_sig_size * m->num_sig_outputs;
   size += dxil_pvs_sig_size * m->num_sig_patch_consts;

   state->state.psv1.sig_input_vectors = (uint8_t)m->num_psv_inputs;

   for (unsigned i = 0; i < 4; ++i)
      state->state.psv1.sig_output_vectors[i] = (uint8_t)m->num_psv_outputs[i];

   uint32_t view_id_table_size = 0;
   if (state->state.psv1.uses_view_id) {
      for (unsigned i = 0; i < 4; ++i) {
         if (state->state.psv1.sig_output_vectors[i] > 0)
            view_id_table_size += sizeof(uint32_t) *
            compute_mask_dwords_from_vectors(state->state.psv1.sig_output_vectors[i]);
      }
      if (state->state.psv1.shader_stage == DXIL_HULL_SHADER && state->state.psv1.sig_patch_const_or_prim_vectors) {
         view_id_table_size += sizeof(uint32_t) *
            compute_mask_dwords_from_vectors(state->state.psv1.sig_patch_const_or_prim_vectors);
      }
   }
   size += view_id_table_size;

   uint32_t dependency_table_size = 0;
   if (state->state.psv1.sig_input_vectors > 0) {
      for (unsigned i = 0; i < 4; ++i) {
         if (state->state.psv1.sig_output_vectors[i] > 0)
            dependency_table_size += sizeof(uint32_t) *
            compute_input_output_table_dwords(state->state.psv1.sig_input_vectors,
               state->state.psv1.sig_output_vectors[i]);
      }
      if (state->state.psv1.shader_stage == DXIL_HULL_SHADER && state->state.psv1.sig_patch_const_or_prim_vectors) {
         dependency_table_size += sizeof(uint32_t) * compute_input_output_table_dwords(state->state.psv1.sig_input_vectors,
            state->state.psv1.sig_patch_const_or_prim_vectors);
      }
   }
   if (state->state.psv1.shader_stage == DXIL_DOMAIN_SHADER &&
       state->state.psv1.sig_patch_const_or_prim_vectors &&
       state->state.psv1.sig_output_vectors[0]) {
      dependency_table_size += sizeof(uint32_t) * compute_input_output_table_dwords(
         state->state.psv1.sig_patch_const_or_prim_vectors, state->state.psv1.sig_output_vectors[0]);
   }
   size += dependency_table_size;

   if (!add_part_header(c, DXIL_PSV0, size))
      return false;

   if (!blob_write_bytes(&c->parts, &psv_size, sizeof(psv_size)))
       return false;

   if (!blob_write_bytes(&c->parts, &state->state, psv_size))
      return false;

   if (!blob_write_bytes(&c->parts, &resource_count, sizeof(resource_count)))
      return false;

   if (resource_count > 0) {
      if (!blob_write_bytes(&c->parts, &resource_bind_info_size, sizeof(resource_bind_info_size)) ||
          !blob_write_bytes(&c->parts, state->resources.v0, resource_bind_info_size * state->num_resources))
         return false;
   }


   uint32_t fill = 0;
   if (!blob_write_bytes(&c->parts, &string_table_size, sizeof(string_table_size)) ||
       !blob_write_bytes(&c->parts, m->sem_string_table->buf, m->sem_string_table->length) ||
       !blob_write_bytes(&c->parts, &fill, string_table_size - m->sem_string_table->length))
      return false;

   if (!blob_write_bytes(&c->parts, &m->sem_index_table.size, sizeof(uint32_t)))
      return false;

   if (m->sem_index_table.size > 0) {
      if (!blob_write_bytes(&c->parts, m->sem_index_table.data,
                            m->sem_index_table.size * sizeof(uint32_t)))
         return false;
   }

   if (m->num_sig_inputs || m->num_sig_outputs || m->num_sig_patch_consts) {
      if (!blob_write_bytes(&c->parts, &dxil_pvs_sig_size, sizeof(dxil_pvs_sig_size)))
         return false;

      if (!blob_write_bytes(&c->parts, &m->psv_inputs, dxil_pvs_sig_size * m->num_sig_inputs))
         return false;

      if (!blob_write_bytes(&c->parts, &m->psv_outputs, dxil_pvs_sig_size * m->num_sig_outputs))
         return false;

      if (!blob_write_bytes(&c->parts, &m->psv_patch_consts, dxil_pvs_sig_size * m->num_sig_patch_consts))
         return false;
   }

   // TODO: Handle case when ViewID is used
   for (uint32_t i = 0; i < view_id_table_size; ++i)
      blob_write_uint8(&c->parts, 0);

   // TODO: Handle sig input output dependency table
   for (uint32_t i = 0; i < dependency_table_size; ++i)
      blob_write_uint8(&c->parts, 0);

   return true;
}

bool
dxil_container_add_module(struct dxil_container *c,
                          const struct dxil_module *m,
                          uint64_t *extra_bit_offset)
{
   assert(m->buf.buf_bits == 0); // make sure the module is fully flushed
   uint32_t version = (m->shader_kind << 16) |
                      (m->major_version << 4) |
                      m->minor_version;
   uint32_t size = 6 * sizeof(uint32_t) + m->buf.blob.size;
   assert(size % sizeof(uint32_t) == 0);
   uint32_t uint32_size = size / sizeof(uint32_t);
   uint32_t magic = 0x4C495844;
   uint32_t dxil_version = 1 << 8; // I have no idea...
   uint32_t bitcode_offset = 16;
   uint32_t bitcode_size = m->buf.blob.size;

   if (!(add_part_header(c, DXIL_DXIL, size) &&
          blob_write_bytes(&c->parts, &version, sizeof(version)) &&
          blob_write_bytes(&c->parts, &uint32_size, sizeof(uint32_size)) &&
          blob_write_bytes(&c->parts, &magic, sizeof(magic)) &&
          blob_write_bytes(&c->parts, &dxil_version, sizeof(dxil_version)) &&
          blob_write_bytes(&c->parts, &bitcode_offset, sizeof(bitcode_offset)) &&
          blob_write_bytes(&c->parts, &bitcode_size, sizeof(bitcode_size))))
      return false;

   *extra_bit_offset += c->parts.size * 8;

   return blob_write_bytes(&c->parts, m->buf.blob.data, m->buf.blob.size);
}

bool
dxil_container_write(struct dxil_container *c, struct blob *blob, uint64_t *extra_bit_offset)
{
   assert(blob->size == 0);
   if (!blob_write_bytes(blob, &DXIL_DXBC, sizeof(DXIL_DXBC)))
      return false;

   const uint8_t unsigned_digest[16] = { 0 }; // null-digest means unsigned
   if (!blob_write_bytes(blob, unsigned_digest, sizeof(unsigned_digest)))
      return false;

   uint16_t major_version = 1;
   uint16_t minor_version = 0;
   if (!blob_write_bytes(blob, &major_version, sizeof(major_version)) ||
       !blob_write_bytes(blob, &minor_version, sizeof(minor_version)))
      return false;

   size_t header_size = 32 + 4 * c->num_parts;
   size_t size = header_size + c->parts.size;
   assert(size <= UINT32_MAX);
   uint32_t container_size = (uint32_t)size;
   if (!blob_write_bytes(blob, &container_size, sizeof(container_size)))
      return false;

   uint32_t part_offsets[DXIL_MAX_PARTS];
   for (int i = 0; i < c->num_parts; ++i) {
      size_t offset = header_size + c->part_offsets[i];
      assert(offset <= UINT32_MAX);
      part_offsets[i] = (uint32_t)offset;
   }

   if (!blob_write_bytes(blob, &c->num_parts, sizeof(c->num_parts)) ||
      !blob_write_bytes(blob, part_offsets, sizeof(uint32_t) * c->num_parts))
      return false;

   *extra_bit_offset += blob->size * 8;

   if (!blob_write_bytes(blob, c->parts.data, c->parts.size))
      return false;

   return true;
}
