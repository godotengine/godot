#ifndef HB_ZLIB_HH
#define HB_ZLIB_HH

#include "hb-blob.hh"

static inline bool
hb_blob_is_gzip (const char *data,
                 unsigned    data_len)
{
  return data_len >= 3 &&
         (unsigned char) data[0] == 0x1Fu &&
         (unsigned char) data[1] == 0x8Bu &&
         (unsigned char) data[2] == 0x08u;
}

static inline bool
hb_gzip_get_uncompressed_size (const char *data,
                               unsigned    data_len,
                               uint32_t   *size)
{
  if (data_len < 4)
    return false;

  const unsigned char *trailer = (const unsigned char *) data + data_len - 4;
  if (size)
    *size = (uint32_t) trailer[0] |
            ((uint32_t) trailer[1] << 8) |
            ((uint32_t) trailer[2] << 16) |
            ((uint32_t) trailer[3] << 24);
  return true;
}

HB_INTERNAL hb_blob_t *
hb_blob_decompress_gzip (hb_blob_t *blob,
                         unsigned   max_output_len);

#endif /* HB_ZLIB_HH */
