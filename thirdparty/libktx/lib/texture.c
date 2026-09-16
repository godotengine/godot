/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2018-2020 Mark Callow.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file
 * @~English
 *
 * @brief ktxTexture implementation.
 *
 * @author Mark Callow, github.com/MarkCallow
 */

#if defined(_WIN32)
  #define _CRT_SECURE_NO_WARNINGS
  #ifndef __cplusplus
    #undef inline
    #define inline __inline
  #endif // __cplusplus
#endif

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ktx.h"
#include "ktxint.h"
#include "formatsize.h"
#include "filestream.h"
#include "memstream.h"
#include "texture1.h"
#include "texture2.h"
#include "unused.h"

ktx_size_t ktxTexture_GetDataSize(ktxTexture* This);

static ktx_uint32_t padRow(ktx_uint32_t* rowBytes);

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Construct (initialize) a ktxTexture base class instance.
 *
 * @param[in] This pointer to a ktxTexture-sized block of memory to
 *                 initialize.
 * @param[in] createInfo pointer to a ktxTextureCreateInfo struct with
 *                       information describing the texture.
 * @param[in] formatSize pointer to a ktxFormatSize giving size information
 *                       about the texture's elements.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @c glInternalFormat in @p createInfo is not a
 *                              valid OpenGL internal format value.
 * @exception KTX_INVALID_VALUE @c numDimensions in @p createInfo is not 1, 2
 *                              or 3.
 * @exception KTX_INVALID_VALUE One of <tt>base{Width,Height,Depth}</tt> in
 *                              @p createInfo is 0.
 * @exception KTX_INVALID_VALUE @c numFaces in @p createInfo is not 1 or 6.
 * @exception KTX_INVALID_VALUE @c numLevels in @p createInfo is 0.
 * @exception KTX_INVALID_OPERATION
 *                              The <tt>base{Width,Height,Depth}</tt> specified
 *                              in @p createInfo are inconsistent with
 *                              @c numDimensions.
 * @exception KTX_INVALID_OPERATION
 *                              @p createInfo is requesting a 3D array or
 *                              3D cubemap texture.
 * @exception KTX_INVALID_OPERATION
 *                              @p createInfo is requesting a cubemap with
 *                              non-square or non-2D images.
 * @exception KTX_INVALID_OPERATION
 *                              @p createInfo is requesting more mip levels
 *                              than needed for the specified
 *                              <tt>base{Width,Height,Depth}</tt>.
 * @exception KTX_OUT_OF_MEMORY Not enough memory for the texture.
 */
KTX_error_code
ktxTexture_construct(ktxTexture* This,
                     const ktxTextureCreateInfo* const createInfo,
                     ktxFormatSize* formatSize)
{
    DECLARE_PROTECTED(ktxTexture);

    memset(This, 0, sizeof(*This));
    This->_protected = (struct ktxTexture_protected*)malloc(sizeof(*prtctd));
    if (!This->_protected)
        return KTX_OUT_OF_MEMORY;
    prtctd = This->_protected;
    memset(prtctd, 0, sizeof(*prtctd));
    memcpy(&prtctd->_formatSize, formatSize, sizeof(prtctd->_formatSize));

    This->isCompressed = (formatSize->flags & KTX_FORMAT_SIZE_COMPRESSED_BIT);

    This->orientation.x = KTX_ORIENT_X_RIGHT;
    This->orientation.y = KTX_ORIENT_Y_DOWN;
    This->orientation.z = KTX_ORIENT_Z_OUT;

    /* Check texture dimensions. KTX files can store 8 types of textures:
     * 1D, 2D, 3D, cube, and array variants of these.
     */
    if (createInfo->numDimensions < 1 || createInfo->numDimensions > 3)
        return KTX_INVALID_VALUE;

    if (createInfo->baseWidth == 0 || createInfo->baseHeight == 0
        || createInfo->baseDepth == 0)
        return KTX_INVALID_VALUE;

    switch (createInfo->numDimensions) {
      case 1:
        if (createInfo->baseHeight > 1 || createInfo->baseDepth > 1)
            return KTX_INVALID_OPERATION;
        break;

      case 2:
        if (createInfo->baseDepth > 1)
            return KTX_INVALID_OPERATION;
        break;

      case 3:
        /* 3D array textures and 3D cubemaps are not supported by either
         * OpenGL or Vulkan.
         */
        if (createInfo->isArray || createInfo->numFaces != 1
            || createInfo->numLayers != 1)
            return KTX_INVALID_OPERATION;
        break;
    }
    This->numDimensions = createInfo->numDimensions;
    This->baseWidth = createInfo->baseWidth;
    This->baseDepth = createInfo->baseDepth;
    This->baseHeight = createInfo->baseHeight;

    if (createInfo->numLayers == 0)
        return KTX_INVALID_VALUE;
    This->numLayers = createInfo->numLayers;
    This->isArray = createInfo->isArray;

    if (createInfo->numFaces == 6) {
        if (This->numDimensions != 2) {
            /* cube map needs 2D faces */
            return KTX_INVALID_OPERATION;
        }
        if (createInfo->baseWidth != createInfo->baseHeight) {
            /* cube maps require square images */
            return KTX_INVALID_OPERATION;
        }
        This->isCubemap = KTX_TRUE;
    } else if (createInfo->numFaces != 1) {
        /* numFaces must be either 1 or 6 */
        return KTX_INVALID_VALUE;
    }
    This->numFaces = createInfo->numFaces;

    /* Check number of mipmap levels */
    if (createInfo->numLevels == 0)
        return KTX_INVALID_VALUE;
    This->numLevels = createInfo->numLevels;
    This->generateMipmaps = createInfo->generateMipmaps;

    if (createInfo->numLevels > 1) {
        GLuint max_dim = MAX(MAX(createInfo->baseWidth, createInfo->baseHeight),
                             createInfo->baseDepth);
        if (max_dim < ((GLuint)1 << (This->numLevels - 1)))
        {
            /* Can't have more mip levels than 1 + log2(max(width, height, depth)) */
            return KTX_INVALID_OPERATION;
        }
    }

    ktxHashList_Construct(&This->kvDataHead);
    return KTX_SUCCESS;
}

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Construct (initialize) the part of a ktxTexture base class that is
 *        not related to the stream contents.
 *
 * @param[in] This pointer to a ktxTexture-sized block of memory to
 *                 initialize.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 */
KTX_error_code
ktxTexture_constructFromStream(ktxTexture* This, ktxStream* pStream,
                               ktxTextureCreateFlags createFlags)
{
    ktxStream* stream;
    UNUSED(createFlags); // Reference to keep compiler happy.

    assert(This != NULL);
    assert(pStream->data.mem != NULL);
    assert(pStream->type == eStreamTypeFile
           || pStream->type == eStreamTypeMemory
           || pStream->type == eStreamTypeCustom);

    This->_protected = (struct ktxTexture_protected *)
                                malloc(sizeof(struct ktxTexture_protected));
    stream = ktxTexture_getStream(This);
    // Copy stream info into struct for later use.
    *stream = *pStream;

    This->orientation.x = KTX_ORIENT_X_RIGHT;
    This->orientation.y = KTX_ORIENT_Y_DOWN;
    This->orientation.z = KTX_ORIENT_Z_OUT;

    return KTX_SUCCESS;
}


/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Free the memory associated with the texture contents
 *
 * @param[in] This pointer to the ktxTextureInt whose texture contents are
 *                 to be freed.
 */
void
ktxTexture_destruct(ktxTexture* This)
{
    ktxStream stream = *(ktxTexture_getStream(This));

    if (stream.data.file != NULL)
        stream.destruct(&stream);
    if (This->kvDataHead != NULL)
        ktxHashList_Destruct(&This->kvDataHead);
    if (This->kvData != NULL)
        free(This->kvData);
    if (This->pData != NULL)
        free(This->pData);
    free(This->_protected);
}


/**
 * @defgroup reader Reader
 * @brief Read KTX-formatted data.
 * @{
 */

typedef enum { KTX1, KTX2 } ktxFileType_;
typedef union {
    KTX_header ktx;
    KTX_header2 ktx2;
} ktxHeaderUnion_;

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Determine if stream data is KTX1 or KTX2.
 *
 * @param pStream   pointer to the ktxStream to examine.
 * @param pFileType pointer to a ktxFileType enum where the type of the data
 *                  will be written.
 * @param pHeader   pointer to a ktxHeaderUnion where the header info. will be
 *                  written.
 */
static KTX_error_code
ktxDetermineFileType_(ktxStream* pStream, ktxFileType_* pFileType,
                      ktxHeaderUnion_* pHeader)
{
    ktx_uint8_t ktx_ident_ref[12] = KTX_IDENTIFIER_REF;
    ktx_uint8_t ktx2_ident_ref[12] = KTX2_IDENTIFIER_REF;
    KTX_error_code result;

    assert(pStream != NULL && pFileType != NULL);
    assert(pStream->data.mem != NULL);
    assert(pStream->type == eStreamTypeFile
           || pStream->type == eStreamTypeMemory
           || pStream->type == eStreamTypeCustom);

    result = pStream->read(pStream, pHeader, sizeof(ktx2_ident_ref));
    if (result == KTX_SUCCESS) {
#if BIG_ENDIAN
        // byte swap the heaader fields
#endif
        // Compare identifier, is this a KTX  or KTX2 file?
        if (!memcmp(pHeader->ktx.identifier, ktx_ident_ref, 12)) {
                *pFileType = KTX1;
        } else if (!memcmp(pHeader->ktx2.identifier, ktx2_ident_ref, 12)) {
                *pFileType = KTX2;
        } else {
                return KTX_UNKNOWN_FILE_FORMAT;
        }
        // Read rest of header.
        if (*pFileType == KTX1) {
            // Read rest of header.
            result = pStream->read(pStream, &pHeader->ktx.endianness,
                                  KTX_HEADER_SIZE - sizeof(ktx_ident_ref));
        } else {
           result = pStream->read(pStream, &pHeader->ktx2.vkFormat,
                                 KTX2_HEADER_SIZE - sizeof(ktx2_ident_ref));
        }
    }
    return result;
}

/**
 * @memberof ktxTexture
 * @~English
 * @brief Create a ktx1 or ktx2 texture according to the stream
 *        data.
 *
 * See @ref ktxTexture1::ktxTexture1_CreateFromStream
 * "ktxTexture1_CreateFromStream" or
 * @ref ktxTexture2::ktxTexture2_CreateFromStream
 * "ktxTexture2_CreateFromStream" for details.
 */
KTX_error_code
ktxTexture_CreateFromStream(ktxStream* pStream,
                            ktxTextureCreateFlags createFlags,
                            ktxTexture** newTex)
{
    ktxHeaderUnion_ header;
    ktxFileType_ fileType;
    KTX_error_code result;
    ktxTexture* tex;

    result = ktxDetermineFileType_(pStream, &fileType, &header);
    if (result != KTX_SUCCESS)
        return result;

    if (fileType == KTX1) {
        ktxTexture1* tex1 = (ktxTexture1*)malloc(sizeof(ktxTexture1));
        if (tex1 == NULL)
            return KTX_OUT_OF_MEMORY;
        memset(tex1, 0, sizeof(ktxTexture1));
        result = ktxTexture1_constructFromStreamAndHeader(tex1, pStream,
                                                          &header.ktx,
                                                          createFlags);
        tex = ktxTexture(tex1);
    } else {
        ktxTexture2* tex2 = (ktxTexture2*)malloc(sizeof(ktxTexture2));
        if (tex2 == NULL)
            return KTX_OUT_OF_MEMORY;
        memset(tex2, 0, sizeof(ktxTexture2));
        result = ktxTexture2_constructFromStreamAndHeader(tex2, pStream,
                                                          &header.ktx2,
                                                          createFlags);
        tex = ktxTexture(tex2);
    }

    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture
 * @~English
 * @brief Create a ktxTexture1 or ktxTexture2 from a stdio stream according
 *        to the stream data.
 *
 * See @ref ktxTexture1::ktxTexture1_CreateFromStdioStream
 * "ktxTexture1_CreateFromStdioStream" or
 * @ref ktxTexture2::ktxTexture2_CreateFromStdioStream
 * "ktxTexture2_CreateFromStdioStream" for details.
 */
KTX_error_code
ktxTexture_CreateFromStdioStream(FILE* stdioStream,
                                 ktxTextureCreateFlags createFlags,
                                 ktxTexture** newTex)
{
    ktxStream stream;
    KTX_error_code result;

    if (stdioStream == NULL || newTex == NULL)
        return KTX_INVALID_VALUE;

    result = ktxFileStream_construct(&stream, stdioStream, KTX_FALSE);
    if (result == KTX_SUCCESS) {
        result = ktxTexture_CreateFromStream(&stream, createFlags, newTex);
    }
    return result;
}

/**
 * @memberof ktxTexture
 * @~English
 * @brief Create a ktxTexture1 or ktxTexture2 from a named KTX file according
 *        to the file contents.
 *
 * See @ref ktxTexture1::ktxTexture1_CreateFromNamedFile
 * "ktxTexture1_CreateFromNamedFile" or
 * @ref ktxTexture2::ktxTexture2_CreateFromNamedFile
 * "ktxTexture2_CreateFromNamedFile" for details.
 */
KTX_error_code
ktxTexture_CreateFromNamedFile(const char* const filename,
                               ktxTextureCreateFlags createFlags,
                               ktxTexture** newTex)
{
    KTX_error_code result;
    ktxStream stream;
    FILE* file;

    if (filename == NULL || newTex == NULL)
        return KTX_INVALID_VALUE;

    file = ktxFOpenUTF8(filename, "rb");
    if (!file)
       return KTX_FILE_OPEN_FAILED;

    result = ktxFileStream_construct(&stream, file, KTX_TRUE);
    if (result == KTX_SUCCESS) {
        result = ktxTexture_CreateFromStream(&stream, createFlags, newTex);
    }
    return result;
}

/**
 * @memberof ktxTexture
 * @~English
 * @brief Create a ktxTexture1 or ktxTexture2 from KTX-formatted data in memory
 *        according to the data contents.
 *
 * See @ref ktxTexture1::ktxTexture1_CreateFromMemory
 * "ktxTexture1_CreateFromMemory" or
 * @ref ktxTexture2::ktxTexture2_CreateFromMemory
 * "ktxTexture2_CreateFromMemory" for details.
 */
KTX_error_code
ktxTexture_CreateFromMemory(const ktx_uint8_t* bytes, ktx_size_t size,
                            ktxTextureCreateFlags createFlags,
                            ktxTexture** newTex)
{
    KTX_error_code result;
    ktxStream stream;

    if (bytes == NULL || newTex == NULL || size == 0)
        return KTX_INVALID_VALUE;

    result = ktxMemStream_construct_ro(&stream, bytes, size);
    if (result == KTX_SUCCESS) {
        result = ktxTexture_CreateFromStream(&stream, createFlags, newTex);
    }
    return result;}


/**
 * @memberof ktxTexture
 * @~English
 * @brief Return a pointer to the texture image data.
 *
 * @param[in] This pointer to the ktxTexture object of interest.
 */
ktx_uint8_t*
ktxTexture_GetData(ktxTexture* This)
{
    return This->pData;
}

/**
 * @memberof ktxTexture
 * @~English
 * @brief Return the total size of the texture image data in bytes.
 *
 * For a ktxTexture2 with supercompressionScheme != KTX_SS_NONE this will
 * return the deflated size of the data.
 *
 * @param[in] This pointer to the ktxTexture object of interest.
 */
ktx_size_t
ktxTexture_GetDataSize(ktxTexture* This)
{
    assert(This != NULL);
    return This->dataSize;
}

/**
 * @memberof ktxTexture
 * @~English
 * @brief Return the size in bytes of an elements of a texture's
 *        images.
 *
 * For uncompressed textures an element is one texel. For compressed
 * textures it is one block.
 *
 * @param[in]     This     pointer to the ktxTexture object of interest.
 */
ktx_uint32_t
ktxTexture_GetElementSize(ktxTexture* This)
{
    assert (This != NULL);

    return (This->_protected->_formatSize.blockSizeInBits / 8);
}

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Calculate & return the size in bytes of an image at the specified
 *        mip level.
 *
 * For arrays, this is the size of layer, for cubemaps, the size of a face
 * and for 3D textures, the size of a depth slice.
 *
 * The size reflects the padding of each row to KTX_GL_UNPACK_ALIGNMENT.
 *
 * @param[in]     This     pointer to the ktxTexture object of interest.
 * @param[in]     level    level of interest.
 * @param[in]     fv       enum specifying format version for which to calculate
 *                         image size.
 */
ktx_size_t
ktxTexture_calcImageSize(ktxTexture* This, ktx_uint32_t level,
                         ktxFormatVersionEnum fv)
{
    DECLARE_PROTECTED(ktxTexture);
    struct blockCount {
        ktx_uint32_t x, y;
    } blockCount;
    ktx_uint32_t blockSizeInBytes;
    ktx_uint32_t rowBytes;

    assert (This != NULL);

    float levelWidth  = (float)(This->baseWidth >> level);
    float levelHeight = (float)(This->baseHeight >> level);
    // Round up to next whole block. We can't use KTX_PADN because some of
    // the block sizes are not powers of 2.
    blockCount.x
        = (ktx_uint32_t)ceilf(levelWidth / prtctd->_formatSize.blockWidth);
    blockCount.y
        = (ktx_uint32_t)ceilf(levelHeight / prtctd->_formatSize.blockHeight);
    blockCount.x = MAX(prtctd->_formatSize.minBlocksX, blockCount.x);
    blockCount.y = MAX(prtctd->_formatSize.minBlocksY, blockCount.y);

    blockSizeInBytes = prtctd->_formatSize.blockSizeInBits / 8;

    if (prtctd->_formatSize.flags & KTX_FORMAT_SIZE_COMPRESSED_BIT) {
        assert(This->isCompressed);
        return blockCount.x * blockCount.y * blockSizeInBytes;
    } else {
        assert(prtctd->_formatSize.blockWidth == 1U
               && prtctd->_formatSize.blockHeight == 1U
               && prtctd->_formatSize.blockDepth == 1U);
        rowBytes = blockCount.x * blockSizeInBytes;
        if (fv == KTX_FORMAT_VERSION_ONE)
            (void)padRow(&rowBytes);
        return rowBytes * blockCount.y;
    }
}

/**
 * @memberof ktxTexture
 * @~English
 * @brief Iterate over the levels or faces in a ktxTexture object.
 *
 * Blocks of image data are passed to an application-supplied callback
 * function. This is not a strict per-image iteration. Rather it reflects how
 * OpenGL needs the images. For most textures the block of data includes all
 * images of a mip level which implies all layers of an array. However, for
 * non-array cube map textures the block is a single face of the mip level,
 * i.e the callback is called once for each face.
 *
 * This function works even if @p This->pData == 0 so it can be used to
 * obtain offsets and sizes for each level by callers who have loaded the data
 * externally.
 *
 * @param[in]     This      pointer to the ktxTexture object of interest.
 * @param[in,out] iterCb    the address of a callback function which is called
 *                          with the data for each image block.
 * @param[in,out] userdata  the address of application-specific data which is
 *                          passed to the callback along with the image data.
 *
 * @return  KTX_SUCCESS on success, other KTX_* enum values on error. The
 *          following are returned directly by this function. @p iterCb may
 *          return these for other causes or may return additional errors.
 *
 * @exception KTX_FILE_DATA_ERROR   Mip level sizes are increasing not
 *                                  decreasing
 * @exception KTX_INVALID_VALUE     @p This is @c NULL or @p iterCb is @c NULL.
 *
 */
KTX_error_code
ktxTexture_IterateLevelFaces(ktxTexture* This, PFNKTXITERCB iterCb,
                             void* userdata)
{
    ktx_uint32_t    miplevel;
    KTX_error_code  result = KTX_SUCCESS;

    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (iterCb == NULL)
        return KTX_INVALID_VALUE;

    for (miplevel = 0; miplevel < This->numLevels; ++miplevel)
    {
        ktx_uint32_t faceLodSize;
        ktx_uint32_t face;
        ktx_uint32_t innerIterations;
        GLsizei      width, height, depth;

        /* Array textures have the same number of layers at each mip level. */
        width = MAX(1, This->baseWidth  >> miplevel);
        height = MAX(1, This->baseHeight >> miplevel);
        depth = MAX(1, This->baseDepth  >> miplevel);

        faceLodSize = (ktx_uint32_t)ktxTexture_calcFaceLodSize(
                                                    This, miplevel);

        /* All array layers are passed in a group because that is how
         * GL & Vulkan need them. Hence no
         *    for (layer = 0; layer < This->numLayers)
         */
        if (This->isCubemap && !This->isArray)
            innerIterations = This->numFaces;
        else
            innerIterations = 1;
        for (face = 0; face < innerIterations; ++face)
        {
            /* And all z_slices are also passed as a group hence no
             *    for (slice = 0; slice < This->depth)
             */
            ktx_size_t offset;

            ktxTexture_GetImageOffset(This, miplevel, 0, face, &offset);
            result = iterCb(miplevel, face,
                            width, height, depth,
                            faceLodSize, This->pData + offset, userdata);

            if (result != KTX_SUCCESS)
                break;
        }
    }

    return result;
}

/**
 * @internal
 * @brief  Calculate and apply the padding needed to comply with
 *         KTX_GL_UNPACK_ALIGNMENT.
 *
 * For uncompressed textures, KTX format specifies KTX_GL_UNPACK_ALIGNMENT = 4.
 *
 * @param[in,out] rowBytes    pointer to variable containing the packed no. of
 *                            bytes in a row. The no. of bytes after padding
 *                            is written into this location.
 * @return the no. of bytes of padding.
 */
static ktx_uint32_t
padRow(ktx_uint32_t* rowBytes)
{
    ktx_uint32_t rowPadding;

    assert (rowBytes != NULL);

    rowPadding = _KTX_PAD_UNPACK_ALIGN_LEN(*rowBytes);
    *rowBytes += rowPadding;
    return rowPadding;
}

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Calculate the size of an array layer at the specified mip level.
 *
 * The size of a layer is the size of an image * either the number of faces
 * or the number of depth slices. This is the size of a layer as needed to
 * find the offset within the array of images of a level and layer so the size
 * reflects any @c cubePadding.
 *
 * @param[in]  This     pointer to the ktxTexture object of interest.
 * @param[in] level     level whose layer size to return.
 *
 * @return the layer size in bytes.
 */
ktx_size_t
ktxTexture_layerSize(ktxTexture* This, ktx_uint32_t level,
                    ktxFormatVersionEnum fv)
{
    /*
     * As there are no 3D cubemaps, the image's z block count will always be
     * 1 for cubemaps and numFaces will always be 1 for 3D textures so the
     * multiply is safe. 3D cubemaps, if they existed, would require
     * imageSize * (blockCount.z + This->numFaces);
     */
    DECLARE_PROTECTED(ktxTexture);
    ktx_uint32_t blockCountZ;
    ktx_size_t imageSize, layerSize;

    assert (This != NULL);
    assert (prtctd->_formatSize.blockDepth != 0);

    blockCountZ = ((This->baseDepth >> level) + prtctd->_formatSize.blockDepth - 1) / prtctd->_formatSize.blockDepth;
    blockCountZ = MAX(1, blockCountZ);
    imageSize = ktxTexture_calcImageSize(This, level, fv);
    layerSize = imageSize * blockCountZ;
    if (fv == KTX_FORMAT_VERSION_ONE && KTX_GL_UNPACK_ALIGNMENT != 4) {
        if (This->isCubemap && !This->isArray) {
            /* cubePadding. NOTE: this adds padding after the last face too. */
            layerSize += _KTX_PAD4(layerSize);
        }
    }
    return layerSize * This->numFaces;
}

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Calculate the size of the specified mip level.
 *
 * The size of a level is the size of a layer * the number of layers.
 *
 * @param[in]  This     pointer to the ktxTexture object of interest.
 * @param[in] level     level whose layer size to return.
 *
 * @return the level size in bytes.
 */
ktx_size_t
ktxTexture_calcLevelSize(ktxTexture* This, ktx_uint32_t level,
                         ktxFormatVersionEnum fv)
{
    assert (This != NULL);
    assert (level < This->numLevels);
    return ktxTexture_layerSize(This, level, fv) * This->numLayers;
}

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Calculate the faceLodSize of the specified mip level.
 *
 * The faceLodSize of a level for most textures is the size of a level. For
 * non-array cube map textures is the size of a face. This is the size that
 * must be provided to OpenGL when uploading textures. Faces get uploaded 1
 * at a time while all layers of an array or all slices of a 3D texture are
 * uploaded together.
 *
 * @param[in]  This     pointer to the ktxTexture object of interest.
 * @param[in] level     level whose layer size to return.
 *
 * @return the faceLodSize size in bytes.
 */
ktx_size_t
ktxTexture_doCalcFaceLodSize(ktxTexture* This, ktx_uint32_t level,
                             ktxFormatVersionEnum fv)
{
    /*
     * For non-array cubemaps this is the size of a face. For everything
     * else it is the size of the level.
     */
    if (This->isCubemap && !This->isArray)
        return ktxTexture_calcImageSize(This, level, fv);
    else
        return ktxTexture_calcLevelSize(This, level, fv);
}


/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Return the number of bytes needed to store all the image data for
 *        a ktxTexture.
 *
 * The caclulated size does not include space for storing the @c imageSize
 * fields of each mip level.
 *
 * @param[in]     This  pointer to the ktxTexture object of interest.
 * @param[in]     fv    enum specifying format version for which to calculate
 *                      image size.
 *
 * @return the data size in bytes.
 */
ktx_size_t
ktxTexture_calcDataSizeTexture(ktxTexture* This)
{
    assert (This != NULL);
    return ktxTexture_calcDataSizeLevels(This, This->numLevels);
}

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Get information about rows of an uncompresssed texture image at a
 *        specified level.
 *
 * For an image at @p level of a ktxTexture provide the number of rows, the
 * packed (unpadded) number of bytes in a row and the padding necessary to
 * comply with KTX_GL_UNPACK_ALIGNMENT.
 *
 * @param[in]     This     pointer to the ktxTexture object of interest.
 * @param[in]     level    level of interest.
 * @param[in,out] numRows  pointer to location to store the number of rows.
 * @param[in,out] pRowLengthBytes pointer to location to store number of bytes
 *                                in a row.
 * @param[in.out] pRowPadding pointer to location to store the number of bytes
 *                            of padding.
 */
void
ktxTexture_rowInfo(ktxTexture* This, ktx_uint32_t level,
                   ktx_uint32_t* numRows, ktx_uint32_t* pRowLengthBytes,
                   ktx_uint32_t* pRowPadding)
{
    DECLARE_PROTECTED(ktxTexture);
    struct blockCount {
        ktx_uint32_t x;
    } blockCount;

    assert (This != NULL);

    assert(!This->isCompressed);
    assert(prtctd->_formatSize.blockWidth == 1U
           && prtctd->_formatSize.blockHeight == 1U
           && prtctd->_formatSize.blockDepth == 1U);

    blockCount.x = MAX(1, (This->baseWidth / prtctd->_formatSize.blockWidth)  >> level);
    *numRows = MAX(1, (This->baseHeight / prtctd->_formatSize.blockHeight)  >> level);

    *pRowLengthBytes = blockCount.x * prtctd->_formatSize.blockSizeInBits / 8;
    *pRowPadding = padRow(pRowLengthBytes);
}

/**
 * @memberof ktxTexture
 * @~English
 * @brief Return pitch between rows of a texture image level in bytes.
 *
 * For uncompressed textures the pitch is the number of bytes between
 * rows of texels. For compressed textures it is the number of bytes
 * between rows of blocks. The value is padded to GL_UNPACK_ALIGNMENT,
 * if necessary. For all currently known compressed formats padding
 * will not be necessary.
 *
 * @param[in]     This     pointer to the ktxTexture object of interest.
 * @param[in]     level    level of interest.
 *
 * @return  the row pitch in bytes.
 */
 ktx_uint32_t
 ktxTexture_GetRowPitch(ktxTexture* This, ktx_uint32_t level)
 {
    DECLARE_PROTECTED(ktxTexture)
    struct blockCount {
        ktx_uint32_t x;
    } blockCount;
    ktx_uint32_t pitch;

    blockCount.x = MAX(1, (This->baseWidth / prtctd->_formatSize.blockWidth)  >> level);
    pitch = blockCount.x * prtctd->_formatSize.blockSizeInBits / 8;
    (void)padRow(&pitch);

    return pitch;
 }

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Query if a ktxTexture has an active stream.
 *
 * Tests if a ktxTexture has unread image data. The internal stream is closed
 * once all the images have been read.
 *
 * @param[in]     This     pointer to the ktxTexture object of interest.
 *
 * @return KTX_TRUE if there is an active stream, KTX_FALSE otherwise.
 */
ktx_bool_t
ktxTexture_isActiveStream(ktxTexture* This)
{
    assert(This != NULL);
    ktxStream* stream = ktxTexture_getStream(This);
    return stream->data.file != NULL;
}

/** @} */

