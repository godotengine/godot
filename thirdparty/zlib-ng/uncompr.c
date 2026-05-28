/* uncompr.c -- decompress a memory buffer
 * Copyright (C) 1995-2003, 2010, 2014, 2016 Jean-loup Gailly, Mark Adler.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zutil.h"

/* ===========================================================================
     Decompresses the source buffer into the destination buffer.  *sourceLen is
   the byte length of the source buffer. Upon entry, *destLen is the total size
   of the destination buffer, which must be large enough to hold the entire
   uncompressed data. (The size of the uncompressed data must have been saved
   previously by the compressor and transmitted to the decompressor by some
   mechanism outside the scope of this compression library.) Upon exit,
   *destLen is the size of the decompressed data and *sourceLen is the number
   of source bytes consumed. Upon return, source + *sourceLen points to the
   first unused input byte.

     uncompress returns Z_OK if success, Z_MEM_ERROR if there was not enough
   memory, Z_BUF_ERROR if there was not enough room in the output buffer, or
   Z_DATA_ERROR if the input data was corrupted, including if the input data is
   an incomplete zlib stream.
*/
z_int32_t Z_EXPORT PREFIX(uncompress2)(unsigned char *dest, z_uintmax_t *destLen, const unsigned char *source, z_uintmax_t *sourceLen) {
    PREFIX3(stream) stream;
    int err;
    const unsigned int max = (unsigned int)-1;
    z_uintmax_t len, left;
    unsigned char buf[1];    /* for detection of incomplete stream when *destLen == 0 */

    len = *sourceLen;
    if (*destLen) {
        left = *destLen;
        *destLen = 0;
    } else {
        left = 1;
        dest = buf;
    }

    stream.next_in = (z_const unsigned char *)source;
    stream.avail_in = 0;
    stream.zalloc = NULL;
    stream.zfree = NULL;
    stream.opaque = NULL;

    err = PREFIX(inflateInit)(&stream);
    if (err != Z_OK) return err;

    stream.next_out = dest;
    stream.avail_out = 0;

    do {
        if (stream.avail_out == 0) {
            stream.avail_out = left > (unsigned long)max ? max : (unsigned int)left;
            left -= stream.avail_out;
        }
        if (stream.avail_in == 0) {
            stream.avail_in = len > (unsigned long)max ? max : (unsigned int)len;
            len -= stream.avail_in;
        }
        err = PREFIX(inflate)(&stream, Z_NO_FLUSH);
    } while (err == Z_OK);

    *sourceLen -= len + stream.avail_in;
    if (dest != buf)
        *destLen = (z_size_t)stream.total_out;
    else if (stream.total_out && err == Z_BUF_ERROR)
        left = 1;

    PREFIX(inflateEnd)(&stream);
    return err == Z_STREAM_END ? Z_OK :
           err == Z_NEED_DICT ? Z_DATA_ERROR  :
           err == Z_BUF_ERROR && left + stream.avail_out ? Z_DATA_ERROR :
           err;
}

z_int32_t Z_EXPORT PREFIX(uncompress)(unsigned char *dest, z_uintmax_t *destLen, const unsigned char *source, z_uintmax_t sourceLen) {
    return PREFIX(uncompress2)(dest, destLen, source, &sourceLen);
}
