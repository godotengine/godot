/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2010-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @~English
 *
 * @brief Implementation of ktxStream for FILE.
 *
 * @author Maksim Kolesin, Under Development
 * @author Georg Kolling, Imagination Technology
 * @author Mark Callow, HI Corporation
 */

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <string.h>
/* I need these on Linux. Why? */
#define __USE_LARGEFILE 1  // For declaration of ftello, etc.
#define __USE_POSIX 1      // For declaration of fileno.
#define _POSIX_SOURCE 1    // For both the above in Emscripten.
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>     // For stat.h on Windows
#define __USE_MISC 1       // For declaration of S_IF...
#include <sys/stat.h>

#include "ktx.h"
#include "ktxint.h"
#include "filestream.h"

// Gotta love Windows :-(
#if defined(_MSC_VER)
  #if defined(_WIN64)
    #define ftello _ftelli64
    #define fseeko _fseeki64
  #else
    #define ftello ftell
    #define fseeko fseek
  #endif
  #define fileno _fileno
  #define fstat _fstat
  #define stat _stat
  #define S_IFIFO _S_IFIFO
  #define S_IFSOCK 0xC000
  typedef unsigned short mode_t;
#endif

#if defined(__MINGW32__)
    #define S_IFSOCK 0xC000
#endif

#define KTX_FILE_STREAM_MAX (1 << (sizeof(ktx_off_t) - 1) - 1)

/**
 * @~English
 * @brief Read bytes from a ktxFileStream.
 *
 * @param [in]  str     pointer to the ktxStream from which to read.
 * @param [out] dst     pointer to a block of memory with a size
 *                      of at least @p size bytes, converted to a void*.
 * @param [in,out] count   pointer to total count of bytes to be read.
 *                         On completion set to number of bytes read.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p dst is @c NULL or @p src is @c NULL.
 * @exception KTX_FILE_READ_ERROR  an error occurred while reading the file.
 * @exception KTX_FILE_UNEXPECTED_EOF not enough data to satisfy the request.
 */
static
KTX_error_code ktxFileStream_read(ktxStream* str, void* dst, const ktx_size_t count)
{
    ktx_size_t nread;

    if (!str || !dst)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeFile);

    if ((nread = fread(dst, 1, count, str->data.file)) != count) {
        if (feof(str->data.file)) {
            return KTX_FILE_UNEXPECTED_EOF;
        } else {
            return KTX_FILE_READ_ERROR;
        }
    }
    str->readpos += count;

    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Skip bytes in a ktxFileStream.
 *
 * @param [in] str           pointer to a ktxStream object.
 * @param [in] count         number of bytes to be skipped.
 *
 * In order to support applications reading from stdin, read characters
 * rather than using seek functions.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p str is @c NULL or @p count is less than zero.
 * @exception KTX_INVALID_OPERATION skipping @p count bytes would go beyond EOF.
 * @exception KTX_FILE_READ_ERROR  an error occurred while reading the file.
 * @exception KTX_FILE_UNEXPECTED_EOF not enough data to satisfy the request.
 *                                    @p count is set to the number of bytes
 *                                    skipped.
 */
static
KTX_error_code ktxFileStream_skip(ktxStream* str, const ktx_size_t count)
{
    if (!str)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeFile);

    for (ktx_uint32_t i = 0; i < count; i++) {
        int ret = getc(str->data.file);
        if (ret == EOF) {
            if (feof(str->data.file)) {
                return KTX_FILE_UNEXPECTED_EOF;
            } else {
                return KTX_FILE_READ_ERROR;
            }
        }
    }
    str->readpos += count;

    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Write bytes to a ktxFileStream.
 *
 * @param [in] str      pointer to the ktxStream that is the destination of the
 *                      write.
 * @param [in] src      pointer to the array of elements to be written,
 *                      converted to a const void*.
 * @param [in] size     size in bytes of each element to be written.
 * @param [in] count    number of elements, each one with a @p size of size
 *                      bytes.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p str is @c NULL or @p src is @c NULL.
 * @exception KTX_FILE_OVERFLOW the requested write would caused the file to
 *                              exceed the maximum supported file size.
 * @exception KTX_FILE_WRITE_ERROR a system error occurred while writing the
 *                                 file.
 */
static
KTX_error_code ktxFileStream_write(ktxStream* str, const void *src,
                                   const ktx_size_t size,
                                   const ktx_size_t count)
{
    if (!str || !src)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeFile);

    if (fwrite(src, size, count, str->data.file) != count) {
        if (errno == EFBIG || errno == EOVERFLOW)
            return KTX_FILE_OVERFLOW;
        else
            return KTX_FILE_WRITE_ERROR;
    }

    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Get the current read/write position in a ktxFileStream.
 *
 * @param [in] str      pointer to the ktxStream to query.
 * @param [in,out] off  pointer to variable to receive the offset value.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_ISPIPE file descriptor underlying stream is associated
 *                            with a pipe or FIFO so does not have a
 *                            file-position indicator.
 * @exception KTX_INVALID_VALUE @p str or @p pos is @c NULL.
 */
static
KTX_error_code ktxFileStream_getpos(ktxStream* str, ktx_off_t* pos)
{
    ktx_off_t ftellval;

    if (!str || !pos)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeFile);

    if (str->data.file == stdin) {
        *pos = str->readpos;
    } else {
        /* The cast quiets an Xcode warning when building for "Generic iOS Device".
         * I'm not sure why.
         */
        ftellval = (ktx_off_t)ftello(str->data.file);
        if (ftellval < 0) {
            switch (errno) {
              case ESPIPE: return KTX_FILE_ISPIPE;
              case EOVERFLOW: return KTX_FILE_OVERFLOW;
            }
        }

        *pos = ftellval;
    }

    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Set the current read/write position in a ktxFileStream.
 *
 * Offset of 0 is the start of the file. This function operates
 * like Linux > 3.1's @c lseek() when it is passed a @c whence
 * of @c SEEK_DATA as it returns an error if the seek would
 * go beyond the end of the file.
 *
 * @param [in] str    pointer to the ktxStream whose r/w position is to be set.
 * @param [in] off    pointer to the offset value to set.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * Throws the same exceptions as ktxFileStream_getsize() for the reasons given
 * there plus the following:
 *
 * @exception KTX_INVALID_VALUE @p str is @c NULL.
 * @exception KTX_INVALID_OPERATION @p pos is > the size of the file or an
 *                                  fseek error occurred.
 */
static
KTX_error_code ktxFileStream_setpos(ktxStream* str, ktx_off_t pos)
{
    ktx_size_t fileSize;
    KTX_error_code result;

    if (!str)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeFile);

    if (str->data.file == stdin) {
        if (pos > str->readpos)
            return str->skip(str, pos - str->readpos);
        else
            return KTX_FILE_ISPIPE;
    }

    result = str->getsize(str, &fileSize);

    if (result != KTX_SUCCESS) {
        // Device is likely not seekable.
        return result;
    }

    if (pos > (ktx_off_t)fileSize)
        return KTX_INVALID_OPERATION;

    if (fseeko(str->data.file, pos, SEEK_SET) < 0)
            return KTX_FILE_SEEK_ERROR;
    else
            return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Get the size of a ktxFileStream in bytes.
 *
 * @param [in] str       pointer to the ktxStream whose size is to be queried.
 * @param [in,out] size  pointer to a variable in which size will be written.
 *
 * @return    KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_OVERFLOW size is too large to be returned in a
 *                              @c ktx_size_t.
 * @exception KTX_FILE_ISPIPE file descriptor underlying stream is associated
 *                            with a pipe or FIFO so does not have a
 *                            file-position indicator.
 * @exception KTX_FILE_READ_ERROR a system error occurred while getting the
 *                                size.
 * @exception KTX_INVALID_VALUE @p str or @p size is @c NULL.
 * @exception KTX_INVALID_OPERATION stream is a tty.
 */
static
KTX_error_code ktxFileStream_getsize(ktxStream* str, ktx_size_t* size)
{
    struct stat statbuf;
    int statret;

    if (!str || !size)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeFile);

    // Need to flush so that fstat will return the current size.
    // Can ignore return value. The only error that can happen is to tell you
    // it was a NOP because the file is read only.
#if (defined(_MSC_VER) && _MSC_VER < 1900) || defined(__MINGW64__) && !defined(_UCRT)
    // Bug in VS2013 msvcrt. fflush on FILE open for READ changes file offset
    // to 4096.
    if (str->data.file->_flag & _IOWRT)
        (void)fflush(str->data.file);
#else
    (void)fflush(str->data.file);
#endif
    statret = fstat(fileno(str->data.file), &statbuf);
    if (statret < 0) {
        switch (errno) {
          case EOVERFLOW: return KTX_FILE_OVERFLOW;
          case EIO:
          default:
            return KTX_FILE_READ_ERROR;
        }
    }

    mode_t ftype = statbuf.st_mode & S_IFMT;
    if (ftype == S_IFIFO || ftype == S_IFSOCK)
        return KTX_FILE_ISPIPE;

    if (statbuf.st_mode & S_IFCHR)
        return KTX_INVALID_OPERATION;

    *size = (ktx_size_t)statbuf.st_size; /* See _getpos for why this cast. */

    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Initialize a ktxFileStream.
 *
 * @param [in] str      pointer to the ktxStream to initialize.
 * @param [in] file     pointer to the underlying FILE object.
 * @param [in] closeFileOnDestruct if not false, stdio file pointer will be closed when ktxStream
 *             is destructed.
 *
 * @return      KTX_SUCCESS on success, KTX_INVALID_VALUE on error.
 *
 * @exception KTX_INVALID_VALUE @p stream is @c NULL or @p file is @c NULL.
 */
KTX_error_code ktxFileStream_construct(ktxStream* str, FILE* file,
                                       ktx_bool_t closeFileOnDestruct)
{
    if (!str || !file)
        return KTX_INVALID_VALUE;

    str->data.file = file;
    str->readpos = 0;
    str->type = eStreamTypeFile;
    str->read = ktxFileStream_read;
    str->skip = ktxFileStream_skip;
    str->write = ktxFileStream_write;
    str->getpos = ktxFileStream_getpos;
    str->setpos = ktxFileStream_setpos;
    str->getsize = ktxFileStream_getsize;
    str->destruct = ktxFileStream_destruct;
    str->closeOnDestruct = closeFileOnDestruct;

    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Destruct the stream, potentially closing the underlying FILE.
 *
 * This only closes the underyling FILE if the @c closeOnDestruct parameter to
 * ktxFileStream_construct() was not @c KTX_FALSE.
 *
 * @param [in] str pointer to the ktxStream whose FILE is to potentially
 *             be closed.
 */
void
ktxFileStream_destruct(ktxStream* str)
{
    assert(str && str->type == eStreamTypeFile);

    if (str->closeOnDestruct)
        fclose(str->data.file);
    str->data.file = 0;
}
