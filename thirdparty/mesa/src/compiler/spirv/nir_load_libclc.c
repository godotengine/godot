/*
 * Copyright © 2020 Intel Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_serialize.h"
#include "nir_spirv.h"
#include "util/mesa-sha1.h"

#ifdef DYNAMIC_LIBCLC_PATH
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef HAVE_STATIC_LIBCLC_ZSTD
#include <zstd.h>
#endif

#ifdef HAVE_STATIC_LIBCLC_SPIRV
#include "spirv-mesa3d-.spv.h"
#endif

#ifdef HAVE_STATIC_LIBCLC_SPIRV64
#include "spirv64-mesa3d-.spv.h"
#endif

struct clc_file {
   unsigned bit_size;
   const char *static_data;
   size_t static_data_size;
   const char *sys_path;
};

static const struct clc_file libclc_files[] = {
   {
      .bit_size = 32,
#ifdef HAVE_STATIC_LIBCLC_SPIRV
      .static_data = libclc_spirv_mesa3d_spv,
      .static_data_size = sizeof(libclc_spirv_mesa3d_spv),
#endif
#ifdef DYNAMIC_LIBCLC_PATH
      .sys_path = DYNAMIC_LIBCLC_PATH "spirv-mesa3d-.spv",
#endif
   },
   {
      .bit_size = 64,
#ifdef HAVE_STATIC_LIBCLC_SPIRV64
      .static_data = libclc_spirv64_mesa3d_spv,
      .static_data_size = sizeof(libclc_spirv64_mesa3d_spv),
#endif
#ifdef DYNAMIC_LIBCLC_PATH
      .sys_path = DYNAMIC_LIBCLC_PATH "spirv64-mesa3d-.spv",
#endif
   },
};

static const struct clc_file *
get_libclc_file(unsigned ptr_bit_size)
{
   assert(ptr_bit_size == 32 || ptr_bit_size == 64);
   return &libclc_files[ptr_bit_size / 64];
}

struct clc_data {
   const struct clc_file *file;

   unsigned char cache_key[20];

   int fd;
   const void *data;
   size_t size;
};

static bool
open_clc_data(struct clc_data *clc, unsigned ptr_bit_size)
{
   memset(clc, 0, sizeof(*clc));
   clc->file = get_libclc_file(ptr_bit_size);
   clc->fd = -1;

   if (clc->file->static_data) {
      snprintf((char *)clc->cache_key, sizeof(clc->cache_key),
               "libclc-spirv%d", ptr_bit_size);
      return true;
   }

#ifdef DYNAMIC_LIBCLC_PATH
   if (clc->file->sys_path != NULL) {
      int fd = open(clc->file->sys_path, O_RDONLY);
      if (fd < 0)
         return false;

      struct stat stat;
      int ret = fstat(fd, &stat);
      if (ret < 0) {
         fprintf(stderr, "fstat failed on %s: %m\n", clc->file->sys_path);
         close(fd);
         return false;
      }

      struct mesa_sha1 ctx;
      _mesa_sha1_init(&ctx);
      _mesa_sha1_update(&ctx, clc->file->sys_path, strlen(clc->file->sys_path));
      _mesa_sha1_update(&ctx, &stat.st_mtim, sizeof(stat.st_mtim));
      _mesa_sha1_final(&ctx, clc->cache_key);

      clc->fd = fd;

      return true;
   }
#endif

   return false;
}

#define SPIRV_WORD_SIZE 4

static bool
map_clc_data(struct clc_data *clc)
{
   if (clc->file->static_data) {
#ifdef HAVE_STATIC_LIBCLC_ZSTD
      unsigned long long cmp_size =
         ZSTD_getFrameContentSize(clc->file->static_data,
                                  clc->file->static_data_size);
      if (cmp_size == ZSTD_CONTENTSIZE_UNKNOWN ||
          cmp_size == ZSTD_CONTENTSIZE_ERROR) {
         fprintf(stderr, "Could not determine the decompressed size of the "
                         "libclc SPIR-V\n");
         return false;
      }

      size_t frame_size =
         ZSTD_findFrameCompressedSize(clc->file->static_data,
                                      clc->file->static_data_size);
      if (ZSTD_isError(frame_size)) {
         fprintf(stderr, "Could not determine the size of the first ZSTD frame "
                         "when decompressing libclc SPIR-V: %s\n",
                 ZSTD_getErrorName(frame_size));
         return false;
      }

      void *dest = malloc(cmp_size + 1);
      size_t size = ZSTD_decompress(dest, cmp_size, clc->file->static_data,
                                    frame_size);
      if (ZSTD_isError(size)) {
         free(dest);
         fprintf(stderr, "Error decompressing libclc SPIR-V: %s\n",
                 ZSTD_getErrorName(size));
         return false;
      }

      clc->data = dest;
      clc->size = size;
#else
      clc->data = clc->file->static_data;
      clc->size = clc->file->static_data_size;
#endif
      return true;
   }

#ifdef DYNAMIC_LIBCLC_PATH
   if (clc->file->sys_path != NULL) {
      off_t len = lseek(clc->fd, 0, SEEK_END);
      if (len % SPIRV_WORD_SIZE != 0) {
         fprintf(stderr, "File length isn't a multiple of the word size\n");
         return false;
      }
      clc->size = len;

      clc->data = mmap(NULL, len, PROT_READ, MAP_PRIVATE, clc->fd, 0);
      if (clc->data == MAP_FAILED) {
         fprintf(stderr, "Failed to mmap libclc SPIR-V: %m\n");
         return false;
      }

      return true;
   }
#endif

   return true;
}

static void
close_clc_data(struct clc_data *clc)
{
   if (clc->file->static_data) {
#ifdef HAVE_STATIC_LIBCLC_ZSTD
      free((void *)clc->data);
#endif
      return;
   }

#ifdef DYNAMIC_LIBCLC_PATH
   if (clc->file->sys_path != NULL) {
      if (clc->data)
         munmap((void *)clc->data, clc->size);
      close(clc->fd);
   }
#endif
}

/** Returns true if libclc is found
 *
 * If libclc is compiled in statically, this always returns true.  If we
 * depend on a dynamic libclc, this opens and tries to stat the file.
 */
bool
nir_can_find_libclc(unsigned ptr_bit_size)
{
   struct clc_data clc;
   if (open_clc_data(&clc, ptr_bit_size)) {
      close_clc_data(&clc);
      return true;
   } else {
      return false;
   }
}

/** Adds generic pointer variants of libclc functions
 *
 * Libclc currently doesn't contain generic variants for a bunch of functions
 * like `frexp` but the OpenCL spec with generic pointers requires them.  We
 * really should fix libclc but, in the mean time, we can easily duplicate
 * every function that works on global memory and make it also work on generic
 * memory.
 */
static void
libclc_add_generic_variants(nir_shader *shader)
{
   nir_foreach_function(func, shader) {
      /* These don't need generic variants */
      if (strstr(func->name, "async_work_group_strided_copy"))
         continue;

      char *U3AS1 = strstr(func->name, "U3AS1");
      if (U3AS1 == NULL)
         continue;

      ptrdiff_t offset_1 = U3AS1 - func->name + 4;
      assert(offset_1 < strlen(func->name) && func->name[offset_1] == '1');

      char *generic_name = ralloc_strdup(shader, func->name);
      assert(generic_name[offset_1] == '1');
      generic_name[offset_1] = '4';

      if (nir_shader_get_function_for_name(shader, generic_name))
         continue;

      nir_function *gfunc = nir_function_create(shader, generic_name);
      gfunc->num_params = func->num_params;
      gfunc->params = ralloc_array(shader, nir_parameter, gfunc->num_params);
      for (unsigned i = 0; i < gfunc->num_params; i++)
         gfunc->params[i] = func->params[i];

      gfunc->impl = nir_function_impl_clone(shader, func->impl);
      gfunc->impl->function = gfunc;

      /* Rewrite any global pointers to generic */
      nir_foreach_block(block, gfunc->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (!nir_deref_mode_may_be(deref, nir_var_mem_global))
               continue;

            assert(deref->type != nir_deref_type_var);
            assert(nir_deref_mode_is(deref, nir_var_mem_global));

            deref->modes = nir_var_mem_generic;
         }
      }

      nir_metadata_preserve(gfunc->impl, nir_metadata_none);
   }
}

nir_shader *
nir_load_libclc_shader(unsigned ptr_bit_size,
                       struct disk_cache *disk_cache,
                       const struct spirv_to_nir_options *spirv_options,
                       const nir_shader_compiler_options *nir_options)
{
   assert(ptr_bit_size ==
          nir_address_format_bit_size(spirv_options->global_addr_format));

   struct clc_data clc;
   if (!open_clc_data(&clc, ptr_bit_size))
      return NULL;

#ifdef ENABLE_SHADER_CACHE
   cache_key cache_key;
   if (disk_cache) {
      disk_cache_compute_key(disk_cache, clc.cache_key,
                             sizeof(clc.cache_key), cache_key);

      size_t buffer_size;
      uint8_t *buffer = disk_cache_get(disk_cache, cache_key, &buffer_size);
      if (buffer) {
         struct blob_reader blob;
         blob_reader_init(&blob, buffer, buffer_size);
         nir_shader *nir = nir_deserialize(NULL, nir_options, &blob);
         free(buffer);
         close_clc_data(&clc);
         return nir;
      }
   }
#endif

   if (!map_clc_data(&clc)) {
      close_clc_data(&clc);
      return NULL;
   }

   struct spirv_to_nir_options spirv_lib_options = *spirv_options;
   spirv_lib_options.create_library = true;

   assert(clc.size % SPIRV_WORD_SIZE == 0);
   nir_shader *nir = spirv_to_nir(clc.data, clc.size / SPIRV_WORD_SIZE,
                                  NULL, 0, MESA_SHADER_KERNEL, NULL,
                                  &spirv_lib_options, nir_options);
   nir_validate_shader(nir, "after nir_load_clc_shader");

   /* nir_inline_libclc will assume that the functions in this shader are
    * already ready to lower.  This means we need to inline any function_temp
    * initializers and lower any early returns.
    */
   nir->info.internal = true;
   NIR_PASS_V(nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS_V(nir, nir_lower_returns);

   NIR_PASS_V(nir, libclc_add_generic_variants);

   /* TODO: One day, we may want to run some optimizations on the libclc
    * shader once and cache them to save time in each shader call.
    */

#ifdef ENABLE_SHADER_CACHE
   if (disk_cache) {
      struct blob blob;
      blob_init(&blob);
      nir_serialize(&blob, nir, false);
      disk_cache_put(disk_cache, cache_key, blob.data, blob.size, NULL);
   }
#endif

   close_clc_data(&clc);
   return nir;
}
