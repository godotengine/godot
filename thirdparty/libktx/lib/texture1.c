/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file
 * @~English
 *
 * @brief ktxTexture1 implementation. Support for KTX format.
 *
 * @author Mark Callow, github.com/MarkCallow
 */

#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdlib.h>
#include <string.h>

#include "dfdutils/dfd.h"
#include "ktx.h"
#include "ktxint.h"
#include "filestream.h"
#include "memstream.h"
#include "texture1.h"
#include "unused.h"
#include "gl_format.h"

typedef struct ktxTexture1_private {
   ktx_bool_t   _needSwap;
} ktxTexture1_private;

struct ktxTexture_vtbl ktxTexture1_vtbl;
struct ktxTexture_vtblInt ktxTexture1_vtblInt;

static KTX_error_code
ktxTexture1_constructCommon(ktxTexture1* This)
{
    assert(This != NULL);

    This->classId = ktxTexture1_c;
    This->vtbl = &ktxTexture1_vtbl;
    This->_protected->_vtbl = ktxTexture1_vtblInt;
    This->_private = (ktxTexture1_private*)malloc(sizeof(ktxTexture1_private));
    if (This->_private == NULL) {
        return KTX_OUT_OF_MEMORY;
    }
	memset(This->_private, 0, sizeof(*This->_private));

    return KTX_SUCCESS;
}

/**
 * @memberof ktxTexture1 @private
 * @copydoc ktxTexture2_construct
 */
static KTX_error_code
ktxTexture1_construct(ktxTexture1* This,
                      const ktxTextureCreateInfo* const createInfo,
                      ktxTextureCreateStorageEnum storageAllocation)
{
    ktxTexture_protected* prtctd;
    ktxFormatSize formatSize;
    GLuint typeSize;
    GLenum glFormat;
    KTX_error_code result;

	memset(This, 0, sizeof(*This));

    This->glInternalformat = createInfo->glInternalformat;
    glGetFormatSize(This->glInternalformat, &formatSize);
    if (formatSize.blockSizeInBits == 0) {
        // Most likely a deprecated legacy format.
        return KTX_UNSUPPORTED_TEXTURE_TYPE;
    }
    glFormat= glGetFormatFromInternalFormat(createInfo->glInternalformat);
    if (glFormat == GL_INVALID_VALUE) {
            return KTX_INVALID_VALUE;
    }
    result =  ktxTexture_construct(ktxTexture(This), createInfo, &formatSize);
    if (result != KTX_SUCCESS)
        return result;

    result = ktxTexture1_constructCommon(This);
    if (result != KTX_SUCCESS)
        return result;
    prtctd = This->_protected;

    This->isCompressed
                    = (formatSize.flags & KTX_FORMAT_SIZE_COMPRESSED_BIT);
    if (This->isCompressed) {
        This->glFormat = 0;
        This->glBaseInternalformat = glFormat;
        This->glType = 0;
        prtctd->_typeSize = 1;
    } else {
        This->glBaseInternalformat = This->glFormat = glFormat;
        This->glType
                = glGetTypeFromInternalFormat(createInfo->glInternalformat);
        if (This->glType == GL_INVALID_VALUE) {
            result = KTX_INVALID_VALUE;
            goto cleanup;
        }
        typeSize = glGetTypeSizeFromType(This->glType);
        assert(typeSize != GL_INVALID_VALUE);

        /* Do some sanity checking */
        if (typeSize != 1 &&
            typeSize != 2 &&
            typeSize != 4)
        {
            /* Only 8, 16, and 32-bit types are supported for byte-swapping.
             * See UNPACK_SWAP_BYTES & table 8.4 in the OpenGL 4.4 spec.
             */
            result = KTX_INVALID_VALUE;
            goto cleanup;
        }
        prtctd->_typeSize = typeSize;
    }

    if (storageAllocation == KTX_TEXTURE_CREATE_ALLOC_STORAGE) {
        This->dataSize
                    = ktxTexture_calcDataSizeTexture(ktxTexture(This));
        This->pData = malloc(This->dataSize);
        if (This->pData == NULL) {
            result = KTX_OUT_OF_MEMORY;
            goto cleanup;
        }
    }
    return result;

cleanup:
    ktxTexture1_destruct(This);
    ktxTexture_destruct(ktxTexture(This));
    return result;
}

/**
 * @memberof ktxTexture1 @private
 * @brief Construct a ktxTexture1 from a ktxStream reading from a KTX source.
 *
 * The KTX header, that must have been read prior to calling this, is passed
 * to the function.
 *
 * The stream object is copied into the constructed ktxTexture1.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture1 is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture1.
 *
 * @param[in] This pointer to a ktxTexture1-sized block of memory to
 *                 initialize.
 * @param[in] pStream pointer to the stream to read.
 * @param[in] pHeader pointer to a KTX header that has already been read from
 *            the stream.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_DATA_ERROR
 *                              Source data is inconsistent with the KTX
 *                              specification.
 * @exception KTX_FILE_READ_ERROR
 *                              An error occurred while reading the source.
 * @exception KTX_FILE_UNEXPECTED_EOF
 *                              Not enough data in the source.
 * @exception KTX_OUT_OF_MEMORY Not enough memory to load either the images or
 *                              the key-value data.
 * @exception KTX_UNKNOWN_FILE_FORMAT
 *                              The source is not in KTX format.
 * @exception KTX_UNSUPPORTED_TEXTURE_TYPE
 *                              The source describes a texture type not
 *                              supported by OpenGL or Vulkan, e.g, a 3D array.
 */
KTX_error_code
ktxTexture1_constructFromStreamAndHeader(ktxTexture1* This, ktxStream* pStream,
                                          KTX_header* pHeader,
                                          ktxTextureCreateFlags createFlags)
{
    ktxTexture1_private* private;
    KTX_error_code result;
    KTX_supplemental_info suppInfo;
    ktxStream* stream;
    ktx_off_t pos;
    ktx_size_t size;
    ktxFormatSize formatSize;

    assert(pHeader != NULL && pStream != NULL);

	memset(This, 0, sizeof(*This));
    result = ktxTexture_constructFromStream(ktxTexture(This), pStream, createFlags);
    if (result != KTX_SUCCESS)
        return result;
    result = ktxTexture1_constructCommon(This);
    if (result != KTX_SUCCESS) {
        ktxTexture_destruct(ktxTexture(This));
        return result;
    }

    private = This->_private;
    stream = ktxTexture1_getStream(This);

    result = ktxCheckHeader1_(pHeader, &suppInfo);
    if (result != KTX_SUCCESS)
        goto cleanup;

    /*
     * Initialize from pHeader info.
     */
    This->glFormat = pHeader->glFormat;
    This->glInternalformat = pHeader->glInternalformat;
    This->glType = pHeader->glType;
    glGetFormatSize(This->glInternalformat, &formatSize);
    if (formatSize.blockSizeInBits == 0) {
        // Most likely a deprecated legacy format.
        result = KTX_UNSUPPORTED_TEXTURE_TYPE;
        goto cleanup;
    }
    This->_protected->_formatSize = formatSize;
    This->glBaseInternalformat = pHeader->glBaseInternalformat;
    // Can these be done by a ktxTexture_constructFromStream?
    This->numDimensions = suppInfo.textureDimension;
    This->baseWidth = pHeader->pixelWidth;
    assert(suppInfo.textureDimension > 0 && suppInfo.textureDimension < 4);
    switch (suppInfo.textureDimension) {
      case 1:
        This->baseHeight = This->baseDepth = 1;
        break;
      case 2:
        This->baseHeight = pHeader->pixelHeight;
        This->baseDepth = 1;
        break;
      case 3:
        This->baseHeight = pHeader->pixelHeight;
        This->baseDepth = pHeader->pixelDepth;
        break;
    }
    if (pHeader->numberOfArrayElements > 0) {
        This->numLayers = pHeader->numberOfArrayElements;
        This->isArray = KTX_TRUE;
    } else {
        This->numLayers = 1;
        This->isArray = KTX_FALSE;
    }
    This->numFaces = pHeader->numberOfFaces;
    if (pHeader->numberOfFaces == 6)
        This->isCubemap = KTX_TRUE;
    else
        This->isCubemap = KTX_FALSE;
    This->numLevels = pHeader->numberOfMipLevels;
    This->isCompressed = suppInfo.compressed;
    This->generateMipmaps = suppInfo.generateMipmaps;
    if (pHeader->endianness == KTX_ENDIAN_REF_REV)
        private->_needSwap = KTX_TRUE;
    This->_protected->_typeSize = pHeader->glTypeSize;

    /*
     * Make an empty hash list.
     */
    ktxHashList_Construct(&This->kvDataHead);
    /*
     * Load KVData.
     */
    if (pHeader->bytesOfKeyValueData > 0) {
        if (!(createFlags & KTX_TEXTURE_CREATE_SKIP_KVDATA_BIT)) {
            ktx_uint32_t kvdLen = pHeader->bytesOfKeyValueData;
            ktx_uint8_t* pKvd;

            pKvd = malloc(kvdLen);
            if (pKvd == NULL) {
                result = KTX_OUT_OF_MEMORY;
                goto cleanup;
            }

            result = stream->read(stream, pKvd, kvdLen);
            if (result != KTX_SUCCESS) {
                free(pKvd);
                goto cleanup;
            }

            if (private->_needSwap) {
                /* Swap the counts inside the key & value data. */
                ktx_uint8_t* src = pKvd;
                ktx_uint8_t* end = pKvd + kvdLen;
                while (src < end) {
                    ktx_uint32_t* pKeyAndValueByteSize = (ktx_uint32_t*)src;
                    _ktxSwapEndian32(pKeyAndValueByteSize, 1);
                    src += _KTX_PAD4(*pKeyAndValueByteSize);
                }
            }

            if (!(createFlags & KTX_TEXTURE_CREATE_RAW_KVDATA_BIT)) {
                char* orientation;
                ktx_uint32_t orientationLen;

                result = ktxHashList_Deserialize(&This->kvDataHead,
                                                 kvdLen, pKvd);
                free(pKvd);
                if (result != KTX_SUCCESS) {
                    goto cleanup;
                }

                result = ktxHashList_FindValue(&This->kvDataHead,
                                               KTX_ORIENTATION_KEY,
                                               &orientationLen,
                                               (void**)&orientation);
                assert(result != KTX_INVALID_VALUE);
                if (result == KTX_SUCCESS) {
                    ktx_uint32_t count;
                    char orient[4] = {0, 0, 0, 0};

                    count = sscanf(orientation, KTX_ORIENTATION3_FMT,
                                   &orient[0],
                                   &orient[1],
                                   &orient[2]);

                    if (count > This->numDimensions) {
                        // KTX 1 is less strict than KTX2 so there is a chance
                        // of having more dimensions than needed.
                        count = This->numDimensions;
                    }
                    switch (This->numDimensions) {
                      case 3:
                        This->orientation.z = orient[2];
                        FALLTHROUGH;
                      case 2:
                        This->orientation.y = orient[1];
                        FALLTHROUGH;
                      case 1:
                        This->orientation.x = orient[0];
                    }
                }
            } else {
                This->kvDataLen = kvdLen;
                This->kvData = pKvd;
            }
        } else {
            stream->skip(stream, pHeader->bytesOfKeyValueData);
        }
    }

    /*
     * Get the size of the image data.
     */
    result = stream->getsize(stream, &size);
    if (result != KTX_SUCCESS)
        goto cleanup;

    result = stream->getpos(stream, &pos);
    if (result != KTX_SUCCESS)
        goto cleanup;

                                /* Remove space for faceLodSize fields */
    This->dataSize = size - pos - This->numLevels * sizeof(ktx_uint32_t);

    /*
     * Load the images, if requested.
     */
    if (createFlags & KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT) {
        result = ktxTexture1_LoadImageData(This, NULL, 0);
    }
    if (result == KTX_SUCCESS)
        return result;

cleanup:
    ktxTexture1_destruct(This);
    return result;
}

/**
 * @memberof ktxTexture1 @private
 * @brief Construct a ktxTexture1 from a ktxStream reading from a KTX source.
 *
 * The stream object is copied into the constructed ktxTexture1.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture1 is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture1.
 *
 * @param[in] This pointer to a ktxTexture1-sized block of memory to
 *            initialize.
 * @param[in] pStream pointer to the stream to read.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return    KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_READ_ERROR
 *                              An error occurred while reading the source.
 *
 * For other exceptions see ktxTexture1_constructFromStreamAndHeader().
 */
static KTX_error_code
ktxTexture1_constructFromStream(ktxTexture1* This, ktxStream* pStream,
                                ktxTextureCreateFlags createFlags)
{
    KTX_header header;
    KTX_error_code result;

    // Read header.
    result = pStream->read(pStream, &header, KTX_HEADER_SIZE);
    if (result != KTX_SUCCESS)
        return result;

    return ktxTexture1_constructFromStreamAndHeader(This, pStream,
                                                    &header, createFlags);
}

/**
 * @memberof ktxTexture1 @private
 * @brief Construct a ktxTexture1 from a stdio stream reading from a KTX source.
 *
 * See ktxTextureInt_constructFromStream for details.
 *
 * @note Do not close the stdio stream until you are finished with the texture
 *       object.
 *
 * @param[in] This pointer to a ktxTextureInt-sized block of memory to
 *                 initialize.
 * @param[in] stdioStream a stdio FILE pointer opened on the source.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE Either @p stdiostream or @p This is null.
 *
 * For other exceptions, see ktxTexture_constructFromStream().
 */
static KTX_error_code
ktxTexture1_constructFromStdioStream(ktxTexture1* This, FILE* stdioStream,
                                     ktxTextureCreateFlags createFlags)
{
    ktxStream stream;
    KTX_error_code result;

    if (stdioStream == NULL || This == NULL)
        return KTX_INVALID_VALUE;

    result = ktxFileStream_construct(&stream, stdioStream, KTX_FALSE);
    if (result == KTX_SUCCESS)
        result = ktxTexture1_constructFromStream(This, &stream, createFlags);
    return result;
}

/**
 * @memberof ktxTexture1 @private
 * @brief Construct a ktxTexture1 from a named KTX file.
 *
 * The file name must be encoded in utf-8. On Windows convert unicode names
 * to utf-8 with @c WideCharToMultiByte(CP_UTF8, ...) before calling.
 *
 * See ktxTextureInt_constructFromStream for details.
 *
 * @param[in] This pointer to a ktxTextureInt-sized block of memory to
 *                 initialize.
 * @param[in] filename    pointer to a char array containing the file name.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_OPEN_FAILED The file could not be opened.
 * @exception KTX_INVALID_VALUE @p filename is @c NULL.
 *
 * For other exceptions, see ktxTexture_constructFromStream().
 */
static KTX_error_code
ktxTexture1_constructFromNamedFile(ktxTexture1* This,
                                   const char* const filename,
                                   ktxTextureCreateFlags createFlags)
{
    FILE* file;
    ktxStream stream;
    KTX_error_code result;

    if (This == NULL || filename == NULL)
        return KTX_INVALID_VALUE;

    file = ktxFOpenUTF8(filename, "rb");
    if (!file)
       return KTX_FILE_OPEN_FAILED;

    result = ktxFileStream_construct(&stream, file, KTX_TRUE);
    if (result == KTX_SUCCESS)
        result = ktxTexture1_constructFromStream(This, &stream, createFlags);

    return result;
}

/**
 * @memberof ktxTexture1 @private
 * @brief Construct a ktxTexture1 from KTX-formatted data in memory.
 *
 * See ktxTextureInt_constructFromStream for details.
 *
 * @param[in] This  pointer to a ktxTextureInt-sized block of memory to
 *                  initialize.
 * @param[in] bytes pointer to the memory containing the serialized KTX data.
 * @param[in] size  length of the KTX data in bytes.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE Either @p bytes is NULL or @p size is 0.
 *
 * For other exceptions, see ktxTexture_constructFromStream().
 */
static KTX_error_code
ktxTexture1_constructFromMemory(ktxTexture1* This,
                                  const ktx_uint8_t* bytes, ktx_size_t size,
                                  ktxTextureCreateFlags createFlags)
{
    ktxStream stream;
    KTX_error_code result;

    if (bytes == NULL || size == 0)
        return KTX_INVALID_VALUE;

    result = ktxMemStream_construct_ro(&stream, bytes, size);
    if (result == KTX_SUCCESS)
        result = ktxTexture1_constructFromStream(This, &stream, createFlags);

    return result;
}

void
ktxTexture1_destruct(ktxTexture1* This)
{
    if (This->_private) free(This->_private);
    ktxTexture_destruct(ktxTexture(This));
}

/**
 * @defgroup reader Reader
 * @brief Read KTX-formatted data.
 * @{
 */

/**
 * @memberof ktxTexture1
 * @ingroup writer
 * @brief Create a new empty ktxTexture1.
 *
 * The address of the newly created ktxTexture1 is written to the location
 * pointed at by @p newTex.
 *
 * @param[in] createInfo pointer to a ktxTextureCreateInfo struct with
 *                       information describing the texture.
 * @param[in] storageAllocation
 *                       enum indicating whether or not to allocate storage
 *                       for the texture images.
 * @param[in,out] newTex pointer to a location in which store the address of
 *                       the newly created texture.
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
 * @exception KTX_OUT_OF_MEMORY Not enough memory for the texture's images.
 */
KTX_error_code
ktxTexture1_Create(const ktxTextureCreateInfo* const createInfo,
                   ktxTextureCreateStorageEnum storageAllocation,
                   ktxTexture1** newTex)
{
    KTX_error_code result;

    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture1* tex = (ktxTexture1*)malloc(sizeof(ktxTexture1));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture1_construct(tex, createInfo, storageAllocation);
    if (result != KTX_SUCCESS) {
        free(tex);
    } else {
        *newTex = tex;
    }
    return result;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Create a ktxTexture1 from a stdio stream reading from a KTX source.
 *
 * The address of a newly created texture reflecting the contents of the
 * stdio stream is written to the location pointed at by @p newTex.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture1 is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture1.
 *
 * @param[in] stdioStream stdio FILE pointer created from the desired file.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 * @param[in,out] newTex  pointer to a location in which store the address of
 *                        the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p newTex is @c NULL.
 * @exception KTX_FILE_DATA_ERROR
 *                              Source data is inconsistent with the KTX
 *                              specification.
 * @exception KTX_FILE_READ_ERROR
 *                              An error occurred while reading the source.
 * @exception KTX_FILE_UNEXPECTED_EOF
 *                              Not enough data in the source.
 * @exception KTX_OUT_OF_MEMORY Not enough memory to create the texture object,
 *                              load the images or load the key-value data.
 * @exception KTX_UNKNOWN_FILE_FORMAT
 *                              The source is not in KTX format.
 * @exception KTX_UNSUPPORTED_TEXTURE_TYPE
 *                              The source describes a texture type not
 *                              supported by OpenGL or Vulkan, e.g, a 3D array.
 */
KTX_error_code
ktxTexture1_CreateFromStdioStream(FILE* stdioStream,
                                  ktxTextureCreateFlags createFlags,
                                  ktxTexture1** newTex)
{
    KTX_error_code result;
    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture1* tex = (ktxTexture1*)malloc(sizeof(ktxTexture1));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture1_constructFromStdioStream(tex, stdioStream,
                                                  createFlags);
    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture1*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Create a ktxTexture1 from a named KTX file.
 *
 * The address of a newly created texture reflecting the contents of the
 * file is written to the location pointed at by @p newTex.
 *
 * The file name must be encoded in utf-8. On Windows convert unicode names
 * to utf-8 with @c WideCharToMultiByte(CP_UTF8, ...) before calling.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture1 is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture1.
 *
 * @param[in] filename    pointer to a char array containing the file name.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 * @param[in,out] newTex  pointer to a location in which store the address of
 *                        the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_OPEN_FAILED The file could not be opened.
 * @exception KTX_INVALID_VALUE @p filename is @c NULL.
 *
 * For other exceptions, see ktxTexture1_CreateFromStdioStream().
 */
KTX_error_code
ktxTexture1_CreateFromNamedFile(const char* const filename,
                                ktxTextureCreateFlags createFlags,
                                ktxTexture1** newTex)
{
    KTX_error_code result;

    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture1* tex = (ktxTexture1*)malloc(sizeof(ktxTexture1));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture1_constructFromNamedFile(tex, filename, createFlags);
    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture1*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Create a ktxTexture1 from KTX-formatted data in memory.
 *
 * The address of a newly created texture reflecting the contents of the
 * serialized KTX data is written to the location pointed at by @p newTex.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture1 is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture1.
 *
 * @param[in] bytes pointer to the memory containing the serialized KTX data.
 * @param[in] size  length of the KTX data in bytes.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 * @param[in,out] newTex  pointer to a location in which store the address of
 *                        the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE Either @p bytes is NULL or @p size is 0.
 *
 * For other exceptions, see ktxTexture1_CreateFromStdioStream().
 */
KTX_error_code
ktxTexture1_CreateFromMemory(const ktx_uint8_t* bytes, ktx_size_t size,
                             ktxTextureCreateFlags createFlags,
                             ktxTexture1** newTex)
{
    KTX_error_code result;
    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture1* tex = (ktxTexture1*)malloc(sizeof(ktxTexture1));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture1_constructFromMemory(tex, bytes, size,
                                             createFlags);
    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture1*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Create a ktxTexture1 from KTX-formatted data from a `ktxStream`.
 *
 * The address of a newly created texture reflecting the contents of the
 * serialized KTX data is written to the location pointed at by @p newTex.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture1 is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture1.
 *
 * @param[in] pStream pointer to the stream to read KTX data from.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 * @param[in,out] newTex  pointer to a location in which store the address of
 *                        the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * For exceptions, see ktxTexture1_CreateFromStdioStream().
 */
KTX_error_code
ktxTexture1_CreateFromStream(ktxStream* pStream,
                             ktxTextureCreateFlags createFlags,
                             ktxTexture1** newTex)
{
    KTX_error_code result;
    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture1* tex = (ktxTexture1*)malloc(sizeof(ktxTexture1));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture1_constructFromStream(tex, pStream, createFlags);
    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture1*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Destroy a ktxTexture1 object.
 *
 * This frees the memory associated with the texture contents and the memory
 * of the ktxTexture1 object. This does @e not delete any OpenGL or Vulkan
 * texture objects created by ktxTexture1_GLUpload or ktxTexture1_VkUpload.
 *
 * @param[in] This pointer to the ktxTexture1 object to destroy
 */
void
ktxTexture1_Destroy(ktxTexture1* This)
{
    ktxTexture1_destruct(This);
    free(This);
}

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Calculate the size of the image data for the specified number
 *        of levels.
 *
 * The data size is the sum of the sizes of each level up to the number
 * specified and includes any @c mipPadding.
 *
 * @param[in] This     pointer to the ktxTexture object of interest.
 * @param[in] levels   number of levels whose data size to return.
 *
 * @return the data size in bytes.
 */
ktx_size_t
ktxTexture1_calcDataSizeLevels(ktxTexture1* This, ktx_uint32_t levels)
{
    ktx_uint32_t i;
    ktx_size_t dataSize = 0;

    assert(This != NULL);
    assert(levels <= This->numLevels);
    for (i = 0; i < levels; i++) {
        ktx_size_t levelSize = ktxTexture_calcLevelSize(ktxTexture(This), i,
                                                        KTX_FORMAT_VERSION_ONE);
        /* mipPadding. NOTE: this adds padding after the last level too. */
        #if KTX_GL_UNPACK_ALIGNMENT != 4
            dataSize += _KTX_PAD4(levelSize);
        #else
            dataSize += levelSize;
        #endif
    }
    return dataSize;
}

/**
 * @memberof ktxTexture1 @private
 * @~English
 *
 * @copydoc ktxTexture::ktxTexture_doCalcFaceLodSize
 */
ktx_size_t
ktxTexture1_calcFaceLodSize(ktxTexture1* This, ktx_uint32_t level)
{
    return ktxTexture_doCalcFaceLodSize(ktxTexture(This), level,
                                        KTX_FORMAT_VERSION_ONE);
}

/**
 * @memberof ktxTexture @private
 * @~English
 * @brief Return the offset of a level in bytes from the start of the image
 *        data in a ktxTexture.
 *
 * The caclulated size does not include space for storing the @c imageSize
 * fields of each mip level.
 *
 * @param[in]     This  pointer to the ktxTexture object of interest.
 * @param[in]     level level whose offset to return.
 * @param[in]     fv    enum specifying format version for which to calculate
 *                      image size.
 *
 * @return the data size in bytes.
 */
ktx_size_t
ktxTexture1_calcLevelOffset(ktxTexture1* This, ktx_uint32_t level)
{
    assert (This != NULL);
    assert (level < This->numLevels);
    return ktxTexture1_calcDataSizeLevels(This, level);
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Find the offset of an image within a ktxTexture's image data.
 *
 * As there is no such thing as a 3D cubemap we make the 3rd location parameter
 * do double duty.
 *
 * @param[in]     This      pointer to the ktxTexture object of interest.
 * @param[in]     level     mip level of the image.
 * @param[in]     layer     array layer of the image.
 * @param[in]     faceSlice cube map face or depth slice of the image.
 * @param[in,out] pOffset   pointer to location to store the offset.
 *
 * @return  KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_OPERATION
 *                         @p level, @p layer or @p faceSlice exceed the
 *                         dimensions of the texture.
 * @exception KTX_INVALID_VALID @p This is NULL.
 */
KTX_error_code
ktxTexture1_GetImageOffset(ktxTexture1* This, ktx_uint32_t level,
                          ktx_uint32_t layer, ktx_uint32_t faceSlice,
                          ktx_size_t* pOffset)
{

    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (level >= This->numLevels || layer >= This->numLayers)
        return KTX_INVALID_OPERATION;

    if (This->isCubemap) {
        if (faceSlice >= This->numFaces)
            return KTX_INVALID_OPERATION;
    } else {
        ktx_uint32_t maxSlice = MAX(1, This->baseDepth >> level);
        if (faceSlice >= maxSlice)
            return KTX_INVALID_OPERATION;
    }

    // Get the size of the data up to the start of the indexed level.
    *pOffset = ktxTexture_calcDataSizeLevels(ktxTexture(This), level);

    // All layers, faces & slices within a level are the same size.
    if (layer != 0) {
        ktx_size_t layerSize;
        layerSize = ktxTexture_layerSize(ktxTexture(This), level,
                                                    KTX_FORMAT_VERSION_ONE);
        *pOffset += layer * layerSize;
    }
    if (faceSlice != 0) {
        ktx_size_t imageSize;
        imageSize = ktxTexture_GetImageSize(ktxTexture(This), level);
#if (KTX_GL_UNPACK_ALIGNMENT != 4)
        if (This->isCubemap)
            _KTX_PAD4(imageSize); // Account for cubePadding.
#endif
        *pOffset += faceSlice * imageSize;
    }

    return KTX_SUCCESS;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Return the total size in bytes of the uncompressed data of a ktxTexture1.
 *
 * This always returns the value of @c This->dataSize. The function is provided for
 * symmetry with ktxTexture2.
 *
 * @param[in]     This     pointer to the ktxTexture1 object of interest.
 * @return    The size of the data in the texture.
 */
ktx_size_t
ktxTexture1_GetDataSizeUncompressed(ktxTexture1* This)
{
    return This->dataSize;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Calculate & return the size in bytes of an image at the specified
 *        mip level.
 *
 * For arrays, this is the size of a layer, for cubemaps, the size of a face
 * and for 3D textures, the size of a depth slice.
 *
 * The size reflects the padding of each row to KTX_GL_UNPACK_ALIGNMENT.
 *
 * @param[in]     This     pointer to the ktxTexture1 object of interest.
 * @param[in]     level    level of interest.
 */
ktx_size_t
ktxTexture1_GetImageSize(ktxTexture1* This, ktx_uint32_t level)
{
    return ktxTexture_calcImageSize(ktxTexture(This), level,
                                    KTX_FORMAT_VERSION_ONE);
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Calculate & return the size in bytes of all the  images in the specified
 *        mip level.
 *
 * For arrays, this is the size of all layers in the level, for cubemaps, the size of all
 * faces in the level and for 3D textures, the size of all depth slices in the level.
 *
 * The size reflects the padding of each row to KTX_GL_UNPACK_ALIGNMENT.
 *
 * @param[in]     This     pointer to the ktxTexture1 object of interest.
 * @param[in]     level    level of interest.
 */
ktx_size_t
ktxTexture1_GetLevelSize(ktxTexture1* This, ktx_uint32_t level)
{
    return ktxTexture_calcLevelSize(ktxTexture(This), level,
                                    KTX_FORMAT_VERSION_ONE);
}

/**
 * @memberof ktxTexture1 @private
 * @~English
 * @brief Return the size of the primitive type of a single color component
 *
 * @param[in]     This       pointer to the ktxTexture1 object of interest.
 *
 * @return the type size in bytes.
 */
ktx_uint32_t
ktxTexture1_glTypeSize(ktxTexture1* This)
{
    assert(This != NULL);
    return This->_protected->_typeSize;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Iterate over the mip levels in a ktxTexture1 object.
 *
 * This is almost identical to @ref ktxTexture::ktxTexture_IterateLevelFaces
 * "ktxTexture_IterateLevelFaces". The difference is that the blocks of image
 * data for non-array cube maps include all faces of a mip level.
 *
 * This function works even if @p This->pData == 0 so it can be used to
 * obtain offsets and sizes for each level by callers who have loaded the data
 * externally.
 *
 * @param[in]     This     handle of the 1 opened on the data.
 * @param[in,out] iterCb   the address of a callback function which is called
 *                         with the data for each image block.
 * @param[in,out] userdata the address of application-specific data which is
 *                         passed to the callback along with the image data.
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
ktxTexture1_IterateLevels(ktxTexture1* This, PFNKTXITERCB iterCb, void* userdata)
{
    ktx_uint32_t    miplevel;
    KTX_error_code  result = KTX_SUCCESS;

    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (iterCb == NULL)
        return KTX_INVALID_VALUE;

    for (miplevel = 0; miplevel < This->numLevels; ++miplevel)
    {
        GLsizei width, height, depth;
        ktx_uint32_t levelSize;
        ktx_size_t offset;

        /* Array textures have the same number of layers at each mip level. */
        width = MAX(1, This->baseWidth  >> miplevel);
        height = MAX(1, This->baseHeight >> miplevel);
        depth = MAX(1, This->baseDepth  >> miplevel);

        levelSize = (ktx_uint32_t)ktxTexture_calcLevelSize(ktxTexture(This),
                                                       miplevel,
                                                       KTX_FORMAT_VERSION_ONE);

        /* All array layers are passed in a group because that is how
         * GL & Vulkan need them. Hence no
         *    for (layer = 0; layer < This->numLayers)
         */
        ktxTexture_GetImageOffset(ktxTexture(This), miplevel, 0, 0, &offset);
        result = iterCb(miplevel, 0, width, height, depth,
                         levelSize, This->pData + offset, userdata);
        if (result != KTX_SUCCESS)
            break;
    }

    return result;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Iterate over the images in a ktxTexture1 object while loading the
 *        image data.
 *
 * This operates similarly to @ref ktxTexture::ktxTexture_IterateLevelFaces
 * "ktxTexture_IterateLevelFaces" except that it loads the images from the
 * ktxTexture1's source to a temporary buffer while iterating. The callback
 * function must copy the image data if it wishes to preserve it as the
 * temporary buffer is reused for each level and is freed when this function
 * exits.
 *
 * This function is helpful for reducing memory usage when uploading the data
 * to a graphics API.
 *
 * @param[in]     This     pointer to the ktxTexture1 object of interest.
 * @param[in,out] iterCb   the address of a callback function which is called
 *                         with the data for each image.
 * @param[in,out] userdata the address of application-specific data which is
 *                         passed to the callback along with the image data.
 *
 * @return  KTX_SUCCESS on success, other KTX_* enum values on error. The
 *          following are returned directly by this function. @p iterCb may
 *          return these for other causes or may return additional errors.
 *
 * @exception KTX_FILE_DATA_ERROR   mip level sizes are increasing not
 *                                  decreasing
 * @exception KTX_INVALID_OPERATION the ktxTexture1 was not created from a
 *                                  stream, i.e there is no data to load, or
 *                                  this ktxTexture1's images have already
 *                                  been loaded.
 * @exception KTX_INVALID_VALUE     @p This is @c NULL or @p iterCb is @c NULL.
 * @exception KTX_OUT_OF_MEMORY     not enough memory to allocate a block to
 *                                  hold the base level image.
 */
KTX_error_code
ktxTexture1_IterateLoadLevelFaces(ktxTexture1* This, PFNKTXITERCB iterCb,
                                  void* userdata)
{
    DECLARE_PRIVATE(ktxTexture1);
    struct ktxTexture_protected* prtctd = This->_protected;
    ktxStream* stream = (ktxStream *)&prtctd->_stream;
    ktx_uint32_t    dataSize = 0;
    ktx_uint32_t    miplevel;
    KTX_error_code  result = KTX_SUCCESS;
    void*           data = NULL;

    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (This->classId != ktxTexture1_c)
        return KTX_INVALID_OPERATION;

    if (iterCb == NULL)
        return KTX_INVALID_VALUE;

    if (prtctd->_stream.data.file == NULL)
        // This Texture not created from a stream or images are already loaded.
        return KTX_INVALID_OPERATION;

    for (miplevel = 0; miplevel < This->numLevels; ++miplevel)
    {
        ktx_uint32_t faceLodSize;
        ktx_uint32_t faceLodSizePadded;
        ktx_uint32_t face;
        ktx_uint32_t innerIterations;
        GLsizei      width, height, depth;

        /* Array textures have the same number of layers at each mip level. */
        width = MAX(1, This->baseWidth  >> miplevel);
        height = MAX(1, This->baseHeight >> miplevel);
        depth = MAX(1, This->baseDepth  >> miplevel);

        result = stream->read(stream, &faceLodSize, sizeof(ktx_uint32_t));
        if (result != KTX_SUCCESS) {
            goto cleanup;
        }
        if (private->_needSwap) {
            _ktxSwapEndian32(&faceLodSize, 1);
        }
#if (KTX_GL_UNPACK_ALIGNMENT != 4)
        faceLodSizePadded = _KTX_PAD4(faceLodSize);
#else
        faceLodSizePadded = faceLodSize;
#endif
        if (!data) {
            /* allocate memory sufficient for the base miplevel */
            data = malloc(faceLodSizePadded);
            if (!data) {
                result = KTX_OUT_OF_MEMORY;
                goto cleanup;
            }
            dataSize = faceLodSizePadded;
        }
        else if (dataSize < faceLodSizePadded) {
            /* subsequent miplevels cannot be larger than the base miplevel */
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }

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
             *    for (z_slice = 0; z_slice < This->depth)
             */
            result = stream->read(stream, data, faceLodSizePadded);
            if (result != KTX_SUCCESS) {
                goto cleanup;
            }

            /* Perform endianness conversion on texture data */
            if (private->_needSwap) {
                if (prtctd->_typeSize == 2)
                    _ktxSwapEndian16((ktx_uint16_t*)data, faceLodSize / 2);
                else if (prtctd->_typeSize == 4)
                    _ktxSwapEndian32((ktx_uint32_t*)data, faceLodSize / 4);
            }

            result = iterCb(miplevel, face,
                             width, height, depth,
                             faceLodSize, data, userdata);
        }
    }

cleanup:
    free(data);
    // No further need for this.
    stream->destruct(stream);

    return result;
}

/**
 * @memberof ktxTexture1
 * @~English
 * @brief Load all the image data from the ktxTexture1's source.
 *
 * The data is loaded into the provided buffer or to an internally allocated
 * buffer, if @p pBuffer is @c NULL.
 *
 * @param[in] This pointer to the ktxTexture object of interest.
 * @param[in] pBuffer pointer to the buffer in which to load the image data.
 * @param[in] bufSize size of the buffer pointed at by @p pBuffer.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p This is NULL.
 * @exception KTX_INVALID_VALUE @p bufSize is less than the the image data size.
 * @exception KTX_INVALID_OPERATION
 *                              The data has already been loaded or the
 *                              ktxTexture was not created from a KTX source.
 * @exception KTX_OUT_OF_MEMORY Insufficient memory for the image data.
 */
KTX_error_code
ktxTexture1_LoadImageData(ktxTexture1* This,
                          ktx_uint8_t* pBuffer, ktx_size_t bufSize)
{
    DECLARE_PROTECTED(ktxTexture);
    DECLARE_PRIVATE(ktxTexture1);
    ktx_uint32_t    miplevel;
    ktx_uint8_t*    pDest;
    ktx_uint8_t*    pDestEnd;
    KTX_error_code  result = KTX_SUCCESS;

    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (prtctd->_stream.data.file == NULL)
        // This Texture not created from a stream or images already loaded;
        return KTX_INVALID_OPERATION;

    if (pBuffer == NULL) {
        This->pData = malloc(This->dataSize);
        if (This->pData == NULL)
            return KTX_OUT_OF_MEMORY;
        pDest = This->pData;
        pDestEnd = pDest + This->dataSize;
    } else if (bufSize < This->dataSize) {
        return KTX_INVALID_VALUE;
    } else {
        pDest = pBuffer;
        pDestEnd = pBuffer + bufSize;
    }

    // Need to loop through for correct byte swapping
    for (miplevel = 0; miplevel < This->numLevels; ++miplevel)
    {
        ktx_uint32_t faceLodSize;
        ktx_uint32_t faceLodSizePadded;
        ktx_uint32_t face;
        ktx_uint32_t innerIterations;

        result = prtctd->_stream.read(&prtctd->_stream, &faceLodSize,
                                      sizeof(ktx_uint32_t));
        if (result != KTX_SUCCESS) {
            goto cleanup;
        }
        if (private->_needSwap) {
            _ktxSwapEndian32(&faceLodSize, 1);
        }
#if (KTX_GL_UNPACK_ALIGNMENT != 4)
        faceLodSizePadded = _KTX_PAD4(faceLodSize);
#else
        faceLodSizePadded = faceLodSize;
#endif

        if (This->isCubemap && !This->isArray)
            innerIterations = This->numFaces;
        else
            innerIterations = 1;
        for (face = 0; face < innerIterations; ++face)
        {
            if (pDest + faceLodSizePadded > pDestEnd) {
                result = KTX_INVALID_VALUE;
                goto cleanup;
            }
            result = prtctd->_stream.read(&prtctd->_stream, pDest,
                                          faceLodSizePadded);
            if (result != KTX_SUCCESS) {
                goto cleanup;
            }

            /* Perform endianness conversion on texture data */
            if (private->_needSwap) {
                if (prtctd->_typeSize == 2)
                    _ktxSwapEndian16((ktx_uint16_t*)pDest, faceLodSize / 2);
                else if (prtctd->_typeSize == 4)
                    _ktxSwapEndian32((ktx_uint32_t*)pDest, faceLodSize / 4);
            }

            pDest += faceLodSizePadded;
        }
    }

cleanup:
    // No further need for This->
    prtctd->_stream.destruct(&prtctd->_stream);
    return result;
}

ktx_bool_t
ktxTexture1_NeedsTranscoding(ktxTexture1* This)
{
    UNUSED(This);
    return KTX_FALSE;
}

#if !KTX_FEATURE_WRITE

/*
 * Stubs for writer functions that return a proper error code
 */

KTX_error_code
ktxTexture1_SetImageFromMemory(ktxTexture1* This, ktx_uint32_t level,
                               ktx_uint32_t layer, ktx_uint32_t faceSlice,
                               const ktx_uint8_t* src, ktx_size_t srcSize)
{
    UNUSED(This);
    UNUSED(level);
    UNUSED(layer);
    UNUSED(faceSlice);
    UNUSED(src);
    UNUSED(srcSize);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture1_SetImageFromStdioStream(ktxTexture1* This, ktx_uint32_t level,
                                    ktx_uint32_t layer, ktx_uint32_t faceSlice,
                                    FILE* src, ktx_size_t srcSize)
{
    UNUSED(This);
    UNUSED(level);
    UNUSED(layer);
    UNUSED(faceSlice);
    UNUSED(src);
    UNUSED(srcSize);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture1_WriteToStdioStream(ktxTexture1* This, FILE* dstsstr)
{
    UNUSED(This);
    UNUSED(dstsstr);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture1_WriteToNamedFile(ktxTexture1* This, const char* const dstname)
{
    UNUSED(This);
    UNUSED(dstname);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture1_WriteToMemory(ktxTexture1* This,
                          ktx_uint8_t** ppDstBytes, ktx_size_t* pSize)
{
    UNUSED(This);
    UNUSED(ppDstBytes);
    UNUSED(pSize);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture1_WriteToStream(ktxTexture1* This,
                          ktxStream* dststr)
{
    UNUSED(This);
    UNUSED(dststr);
    return KTX_INVALID_OPERATION;
}

#endif

/*
 * Initialized here at the end to avoid the need for multiple declarations of
 * these functions.
 */

struct ktxTexture_vtblInt ktxTexture1_vtblInt = {
    (PFNCALCDATASIZELEVELS)ktxTexture1_calcDataSizeLevels,
    (PFNCALCFACELODSIZE)ktxTexture1_calcFaceLodSize,
    (PFNCALCLEVELOFFSET)ktxTexture1_calcLevelOffset
};

struct ktxTexture_vtbl ktxTexture1_vtbl = {
    (PFNKTEXDESTROY)ktxTexture1_Destroy,
    (PFNKTEXGETIMAGEOFFSET)ktxTexture1_GetImageOffset,
    (PFNKTEXGETDATASIZEUNCOMPRESSED)ktxTexture1_GetDataSizeUncompressed,
    (PFNKTEXGETIMAGESIZE)ktxTexture1_GetImageSize,
    (PFNKTEXGETLEVELSIZE)ktxTexture1_GetLevelSize,
    (PFNKTEXITERATELEVELS)ktxTexture1_IterateLevels,
    (PFNKTEXITERATELOADLEVELFACES)ktxTexture1_IterateLoadLevelFaces,
    (PFNKTEXNEEDSTRANSCODING)ktxTexture1_NeedsTranscoding,
    (PFNKTEXLOADIMAGEDATA)ktxTexture1_LoadImageData,
    (PFNKTEXSETIMAGEFROMMEMORY)ktxTexture1_SetImageFromMemory,
    (PFNKTEXSETIMAGEFROMSTDIOSTREAM)ktxTexture1_SetImageFromStdioStream,
    (PFNKTEXWRITETOSTDIOSTREAM)ktxTexture1_WriteToStdioStream,
    (PFNKTEXWRITETONAMEDFILE)ktxTexture1_WriteToNamedFile,
    (PFNKTEXWRITETOMEMORY)ktxTexture1_WriteToMemory,
    (PFNKTEXWRITETOSTREAM)ktxTexture1_WriteToStream,
};

/** @} */

