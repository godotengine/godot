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
 * @brief Implementation of ktxStream for memory.
 *
 * @author Maksim Kolesin, Under Development
 * @author Georg Kolling, Imagination Technology
 * @author Mark Callow, HI Corporation
 */

#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "ktx.h"
#include "ktxint.h"
#include "memstream.h"

/**
* @brief Default allocation size for a ktxMemStream.
*/
#define KTX_MEM_DEFAULT_ALLOCATED_SIZE 256

/**
 * @brief Structure to store information about data allocated for ktxMemStream.
 */
struct ktxMem
{
    const ktx_uint8_t* robytes;/*!< pointer to read-only data */
    ktx_uint8_t* bytes;        /*!< pointer to rw data. */
    ktx_size_t alloc_size;       /*!< allocated size of the memory block. */
    ktx_size_t used_size;        /*!< bytes used. Effectively the write position. */
    ktx_off_t pos;               /*!< read/write position. */
};

static KTX_error_code ktxMem_expand(ktxMem* pMem, const ktx_size_t size);

/**
 * @brief Initialize a ktxMem struct for read-write.
 *
 * Memory for the stream data is allocated internally but the
 * caller is responsible for freeing the memory. A pointer to
 * the memory can be obtained with ktxMem_getdata().
 *
 * @sa ktxMem_getdata.
 *
 * @param [in] pMem pointer to the @c ktxMem to initialize.
 */
static KTX_error_code
ktxMem_construct(ktxMem* pMem)
{
    pMem->pos = 0;
    pMem->alloc_size = 0;
    pMem->robytes = 0;
    pMem->bytes = 0;
    pMem->used_size = 0;
    return ktxMem_expand(pMem, KTX_MEM_DEFAULT_ALLOCATED_SIZE);
}

/**
 * @brief Create & initialize a ktxMem struct for read-write.
 *
 * @sa ktxMem_construct.
 *
 * @param [in,out] ppMem pointer to the location in which to return
 *                       a pointer to the newly created @c ktxMem.
 *
 * @return     KTX_SUCCESS on success, KTX_OUT_OF_MEMORY on error.
 *
 * @exception  KTX_OUT_OF_MEMORY    System failed to allocate sufficient pMemory.
 */
static KTX_error_code
ktxMem_create(ktxMem** ppMem)
{
    ktxMem* pNewMem = (ktxMem*)malloc(sizeof(ktxMem));
    if (pNewMem) {
        KTX_error_code result = ktxMem_construct(pNewMem);
        if (result == KTX_SUCCESS)
            *ppMem = pNewMem;
        return result;
    }
    else {
        return KTX_OUT_OF_MEMORY;
    }
}

/**
 * @brief Initialize a ktxMem struct for read-only.
 *
 * @param [in] pMem     pointer to the @c ktxMem to initialize.
 * @param [in] bytes    pointer to the data to be read.
 * @param [in] numBytes number of bytes of data.
 */
static void
ktxMem_construct_ro(ktxMem* pMem, const void* bytes, ktx_size_t numBytes)
{
    pMem->pos = 0;
    pMem->robytes = bytes;
    pMem->bytes = 0;
    pMem->used_size = numBytes;
    pMem->alloc_size = numBytes;
}

/**
 * @brief Create & initialize a ktxMem struct for read-only.
 *
 * @sa ktxMem_construct.
 *
 * @param [in,out] ppMem    pointer to the location in which to return
 *                          a pointer to the newly created @c ktxMem.
 * @param [in]     bytes    pointer to the data to be read.
 * @param [in]     numBytes number of bytes of data.
 *
 * @return     KTX_SUCCESS on success, KTX_OUT_OF_MEMORY on error.
 *
 * @exception  KTX_OUT_OF_MEMORY    System failed to allocate sufficient pMemory.
 */
static KTX_error_code
ktxMem_create_ro(ktxMem** ppMem, const void* bytes, ktx_size_t numBytes)
{
    ktxMem* pNewMem = (ktxMem*)malloc(sizeof(ktxMem));
    if (pNewMem) {
        ktxMem_construct_ro(pNewMem, bytes, numBytes);
        *ppMem = pNewMem;
        return KTX_SUCCESS;
    }
    else {
        return KTX_OUT_OF_MEMORY;
    }
}

/*
 * ktxMem_destruct not needed as ktxMem_construct caller is reponsible
 * for freeing the data written.
 */

/**
 * @brief Free the memory of a struct ktxMem.
 *
 * @param pMem pointer to ktxMem to free.
 */
static void
ktxMem_destroy(ktxMem* pMem, ktx_bool_t freeData)
{
    assert(pMem != NULL);
    if (freeData) {
        free(pMem->bytes);
    }
    free(pMem);
}

#ifdef KTXMEM_CLEAR_USED
/**
 * @brief Clear the data of a memory stream.
 *
 * @param pMem pointer to ktxMem to clear.
 */
static void
ktxMem_clear(ktxMem* pMem)
{
    assert(pMem != NULL);
    memset(pMem, 0, sizeof(ktxMem));
}
#endif

/**
 * @~English
 * @brief Expand a ktxMem to fit to a new size.
 *
 * @param [in] pMem          pointer to ktxMem struct to expand.
 * @param [in] newsize       minimum new size required.
 *
 * @return     KTX_SUCCESS on success, KTX_OUT_OF_MEMORY on error.
 *
 * @exception  KTX_OUT_OF_MEMORY    System failed to allocate sufficient pMemory.
 */
static KTX_error_code
ktxMem_expand(ktxMem *pMem, const ktx_size_t newsize)
{
    ktx_size_t new_alloc_size;

    assert(pMem != NULL && newsize != 0);

    new_alloc_size = pMem->alloc_size == 0 ?
                     KTX_MEM_DEFAULT_ALLOCATED_SIZE : pMem->alloc_size;
    while (new_alloc_size < newsize) {
        ktx_size_t alloc_size = new_alloc_size;
        new_alloc_size <<= 1;
        if (new_alloc_size < alloc_size) {
            /* Overflow. Set to maximum size. newsize can't be larger. */
            new_alloc_size = (ktx_size_t)-1L;
        }
    }

    if (new_alloc_size == pMem->alloc_size)
        return KTX_SUCCESS;

    if (!pMem->bytes)
        pMem->bytes = (ktx_uint8_t*)malloc(new_alloc_size);
    else
        pMem->bytes = (ktx_uint8_t*)realloc(pMem->bytes, new_alloc_size);

    if (!pMem->bytes)
    {
        pMem->alloc_size = 0;
        pMem->used_size = 0;
        return KTX_OUT_OF_MEMORY;
    }

    pMem->alloc_size = new_alloc_size;
    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Read bytes from a ktxMemStream.
 *
 * @param [in]     str      pointer to ktxMem struct, converted to a void*, that
 *                          specifies an input stream.
 * @param [in,out] dst      pointer to memory where to copy read bytes.
 * @param [in,out] count    pointer to number of bytes to read.
 *
 * @return      KTX_SUCCESS on success, KTX_INVALID_VALUE on error.
 *
 * @exception KTX_INVALID_VALUE     @p str or @p dst is @c NULL or @p str->data is
 *                                  @c NULL.
 * @exception KTX_FILE_UNEXPECTED_EOF not enough data to satisfy the request.
 */
static
KTX_error_code ktxMemStream_read(ktxStream* str, void* dst, const ktx_size_t count)
{
    ktxMem* mem;
    ktx_off_t newpos;
    const ktx_uint8_t* bytes;


    if (!str || (mem = str->data.mem)== 0)
        return KTX_INVALID_VALUE;

    newpos = mem->pos + count;
    /* The first clause checks for overflow. */
    if (newpos < mem->pos || (ktx_uint32_t)newpos > mem->used_size)
        return KTX_FILE_UNEXPECTED_EOF;

    bytes = mem->robytes ? mem->robytes : mem->bytes;
    memcpy(dst, bytes + mem->pos, count);
    mem->pos = newpos;

    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Skip bytes in a ktxMemStream.
 *
 * @param [in] str      pointer to the ktxStream on which to operate.
 * @param [in] count    number of bytes to skip.
 *
 * @return      KTX_SUCCESS on success, KTX_INVALID_VALUE on error.
 *
 * @exception KTX_INVALID_VALUE     @p str or @p mem is @c NULL or sufficient
 *                                  data is not available in ktxMem.
 * @exception KTX_FILE_UNEXPECTED_EOF not enough data to satisfy the request.
 */
static
KTX_error_code ktxMemStream_skip(ktxStream* str, const ktx_size_t count)
{
    ktxMem* mem;
    ktx_off_t newpos;

    if (!str || (mem = str->data.mem) == 0)
        return KTX_INVALID_VALUE;

    newpos = mem->pos + count;
    /* The first clause checks for overflow. */
    if (newpos < mem->pos || (ktx_uint32_t)newpos > mem->used_size)
        return KTX_FILE_UNEXPECTED_EOF;

    mem->pos = newpos;

    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Write bytes to a ktxMemStream.
 *
 * @param [out] str    pointer to the ktxStream that specifies the destination.
 * @param [in] src     pointer to the array of elements to be written,
 *                     converted to a const void*.
 * @param [in] size    size in bytes of each element to be written.
 * @param [in] count   number of elements, each one with a @p size of size
 *                     bytes.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_OVERFLOW        write would result in file exceeding the
 *                                     maximum permissible size.
 * @exception KTX_INVALID_OPERATION    @p str is a read-only stream.
 * @exception KTX_INVALID_VALUE        @p dst is @c NULL or @p mem is @c NULL.
 * @exception KTX_OUT_OF_MEMORY        See ktxMem_expand() for causes.
 */
static
KTX_error_code ktxMemStream_write(ktxStream* str, const void* src,
                                  const ktx_size_t size, const ktx_size_t count)
{
    ktxMem* mem;
    KTX_error_code rc = KTX_SUCCESS;
    ktx_size_t new_size;

    if (!str || (mem = str->data.mem) == 0)
        return KTX_INVALID_VALUE;

    if (mem->robytes)
        return KTX_INVALID_OPERATION; /* read-only */

    new_size = mem->pos + (size*count);
    //if (new_size < mem->used_size)
    if ((ktx_off_t)new_size < mem->pos)
        return KTX_FILE_OVERFLOW;

    if (mem->alloc_size < new_size) {
        rc = ktxMem_expand(mem, new_size);
        if (rc != KTX_SUCCESS)
            return rc;
    }

    memcpy(mem->bytes + mem->pos, src, size*count);
    mem->pos += size*count;
    if (mem->pos > (ktx_off_t)mem->used_size)
        mem->used_size = mem->pos;


    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Get the current read/write position in a ktxMemStream.
 *
 * @param [in] str      pointer to the ktxStream to query.
 * @param [in,out] off  pointer to variable to receive the offset value.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p str or @p pos is @c NULL.
 */
static
KTX_error_code ktxMemStream_getpos(ktxStream* str, ktx_off_t* const pos)
{
    if (!str || !pos)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeMemory);

    *pos = str->data.mem->pos;
    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Set the current read/write position in a ktxMemStream.
 *
 * Offset of 0 is the start of the file.
 *
 * @param [in] str    pointer to the ktxStream whose r/w position is to be set.
 * @param [in] off    pointer to the offset value to set.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p str is @c NULL.
 * @exception KTX_INVALID_OPERATION @p pos > size of the allocated memory.
 */
static
KTX_error_code ktxMemStream_setpos(ktxStream* str, ktx_off_t pos)
{
    if (!str)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeMemory);

    if (pos > (ktx_off_t)str->data.mem->alloc_size)
        return KTX_INVALID_OPERATION;

    str->data.mem->pos = pos;
    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Get a pointer to a ktxMemStream's data.
 *
 * Gets a pointer to data that has been written to the stream. Returned
 * pointer will be 0 if stream is read-only.
 *
 * @param [in] str       pointer to the ktxStream whose data pointer is to
 *                       be queried.
 * @param [in,out] ppBytes  pointer to a variable in which the data pointer
 *                       will be written.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p str or @p ppBytes is @c NULL.
 */
KTX_error_code ktxMemStream_getdata(ktxStream* str, ktx_uint8_t** ppBytes)
{
    if (!str || !ppBytes)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeMemory);

    *ppBytes = str->data.mem->bytes;
    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Get the size of a ktxMemStream in bytes.
 *
 * @param [in] str       pointer to the ktxStream whose size is to be queried.
 * @param [in,out] size  pointer to a variable in which size will be written.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p str or @p pSize is @c NULL.
 */
static
KTX_error_code ktxMemStream_getsize(ktxStream* str, ktx_size_t* pSize)
{
    if (!str || !pSize)
        return KTX_INVALID_VALUE;

    assert(str->type == eStreamTypeMemory);

    *pSize = str->data.mem->used_size;
    return KTX_SUCCESS;
}

/**
 * @~English
 * @brief Setup ktxMemStream function pointers.
 */
void
ktxMemStream_setup(ktxStream* str)
{
    str->type = eStreamTypeMemory;
    str->read = ktxMemStream_read;
    str->skip = ktxMemStream_skip;
    str->write = ktxMemStream_write;
    str->getpos = ktxMemStream_getpos;
    str->setpos = ktxMemStream_setpos;
    str->getsize = ktxMemStream_getsize;
    str->destruct = ktxMemStream_destruct;
}

/**
 * @~English
 * @brief Initialize a read-write ktxMemStream.
 *
 * Memory is allocated as data is written. The caller of this is
 * responsible for freeing this memory unless @a freeOnDestruct
 * is not KTX_FALSE.
 *
 * @param [in] str             pointer to a ktxStream struct to initialize.
 * @param [in] freeOnDestruct  If not KTX_FALSE memory holding the data will
 *                             be freed by the destructor.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE     @p str is @c NULL.
 * @exception KTX_OUT_OF_MEMORY     system failed to allocate sufficient memory.
 */
KTX_error_code ktxMemStream_construct(ktxStream* str,
                                      ktx_bool_t freeOnDestruct)
{
    ktxMem* mem;
    KTX_error_code result = KTX_SUCCESS;

    if (!str)
        return KTX_INVALID_VALUE;

    result = ktxMem_create(&mem);

    if (KTX_SUCCESS == result) {
        str->data.mem = mem;
        ktxMemStream_setup(str);
        str->closeOnDestruct = freeOnDestruct;
    }

    return result;
}

/**
 * @~English
 * @brief Initialize a read-only ktxMemStream.
 *
 * @param [in] str      pointer to a ktxStream struct to initialize.
 * @param [in] bytes    pointer to an array of bytes containing the data.
 * @param [in] numBytes     size of array of data for ktxMemStream.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE     @p str or @p mem is @c NULL or @p numBytes
 *                                  is 0.
 *                                  or @p size is less than 0.
 * @exception KTX_OUT_OF_MEMORY     system failed to allocate sufficient memory.
 */
KTX_error_code ktxMemStream_construct_ro(ktxStream* str,
                                         const ktx_uint8_t* bytes,
                                         const ktx_size_t numBytes)
{
    ktxMem* mem;
    KTX_error_code result = KTX_SUCCESS;

    if (!str || !bytes || numBytes == 0)
        return KTX_INVALID_VALUE;

    result = ktxMem_create_ro(&mem, bytes, numBytes);

    if (KTX_SUCCESS == result) {
        str->data.mem = mem;
        ktxMemStream_setup(str);
        str->closeOnDestruct = KTX_FALSE;
    }

    return result;
}

/**
 * @~English
 * @brief Free the memory used by a ktxMemStream.
 *
 * This only frees the memory used to store the data written to the stream,
 * if the @c freeOnDestruct parameter to ktxMemStream_construct() was not
 * @c KTX_FALSE. Otherwise it is the responsibility of the caller of
 * ktxMemStream_construct() and a pointer to this memory should be retrieved
 * using ktxMemStream_getdata() before calling this function.
 *
 * @sa ktxMemStream_construct, ktxMemStream_getdata.
 *
 * @param [in] str pointer to the ktxStream whose memory is
 *                 to be freed.
 */
void
ktxMemStream_destruct(ktxStream* str)
{
    assert(str && str->type == eStreamTypeMemory);

    ktxMem_destroy(str->data.mem, str->closeOnDestruct);
    str->data.mem = NULL;
}

