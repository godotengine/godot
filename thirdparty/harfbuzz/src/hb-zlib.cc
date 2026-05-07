#ifndef HB_ZLIB_CC
#define HB_ZLIB_CC
#ifdef HB_ZLIB_CC /* Pacify -Wunused-macros. */

#include "hb.hh"

#include "hb-zlib.hh"

#ifdef HAVE_ZLIB
#include <zlib.h>
#endif

hb_blob_t *
hb_blob_decompress_gzip (hb_blob_t *blob,
                         unsigned   max_output_len)
{
#ifndef HAVE_ZLIB
  return nullptr;
#else
  unsigned compressed_len = 0;
  const uint8_t *compressed = (const uint8_t *) hb_blob_get_data (blob, &compressed_len);
  if (!compressed || !compressed_len)
    return nullptr;

  z_stream stream = {};
  stream.next_in = (Bytef *) compressed;
  stream.avail_in = compressed_len;

  if (inflateInit2 (&stream, 16 + MAX_WBITS) != Z_OK)
    return nullptr;
  HB_SCOPE_GUARD (inflateEnd (&stream));

  uint32_t expected_size = 0;
  hb_gzip_get_uncompressed_size ((const char *) compressed,
                                 compressed_len,
                                 &expected_size);

  size_t allocated = hb_min ((size_t) hb_max (expected_size, 4096u),
                             (size_t) max_output_len);
  char *output = (char *) hb_malloc (allocated);
  if (!output)
    return nullptr;
  auto output_guard = hb_make_scope_guard ([&]() { hb_free (output); });

  int status = Z_OK;
  while (true)
  {
    size_t produced = (size_t) stream.total_out;
    if (unlikely (produced >= (size_t) max_output_len))
      return nullptr;

    if (produced == allocated)
    {
      size_t new_allocated = hb_min (allocated * 2, (size_t) max_output_len);
      if (unlikely (new_allocated <= allocated))
        return nullptr;

      char *new_output = (char *) hb_realloc (output, new_allocated);
      if (unlikely (!new_output))
        return nullptr;

      output = new_output;
      allocated = new_allocated;
    }

    stream.next_out = (Bytef *) output + stream.total_out;
    stream.avail_out = (uInt) (allocated - (size_t) stream.total_out);

    status = inflate (&stream, Z_FINISH);
    if (status == Z_STREAM_END)
      break;

    if ((status == Z_OK || status == Z_BUF_ERROR) && stream.avail_out == 0)
      continue;

    return nullptr;
  }

  if (unlikely ((size_t) stream.total_out > (size_t) max_output_len))
    return nullptr;

  output_guard.release ();
  return hb_blob_create_or_fail (output,
                                 (unsigned) stream.total_out,
                                 HB_MEMORY_MODE_WRITABLE,
                                 output,
                                 hb_free);
#endif
}

#endif /* HB_ZLIB_CC pacify */
#endif /* HB_ZLIB_CC guard */
