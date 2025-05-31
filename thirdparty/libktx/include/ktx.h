/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

#ifndef KTX_H_A55A6F00956F42F3A137C11929827FE1
#define KTX_H_A55A6F00956F42F3A137C11929827FE1

/*
 * Copyright 2010-2018 The Khronos Group, Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See the accompanying LICENSE.md for licensing details for all files in
 * the KTX library and KTX loader tests.
 */

/**
 * @file
 * @~English
 *
 * @brief Declares the public functions and structures of the
 *        KTX API.
 *
 * @author Mark Callow, Edgewise Consulting and while at HI Corporation
 * @author Based on original work by Georg Kolling, Imagination Technology
 *
 * @snippet{doc} version.h API version
 */

#include <limits.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/types.h>

#include <KHR/khr_df.h>

/*
 * Don't use khrplatform.h in order not to break apps existing
 * before these definitions were needed.
 */
#if defined(KHRONOS_STATIC)
  #define KTX_API
#elif defined(_WIN32) || defined(__CYGWIN__)
  #if !defined(KTX_API)
    #if __GNUC__
      #define KTX_API __attribute__ ((dllimport))
    #elif _MSC_VER
      #define KTX_API __declspec(dllimport)
    #else
      #error "Your compiler's equivalent of dllimport is unknown"
    #endif
  #endif
#elif defined(__ANDROID__)
  #define KTX_API __attribute__((visibility("default")))
#else
  #define KTX_API
#endif

#if defined(_WIN32) && !defined(KHRONOS_STATIC)
  #if !defined(KTX_APIENTRY)
    #define KTX_APIENTRY __stdcall
  #endif
#else
  #define KTX_APIENTRY
#endif

/* To avoid including <KHR/khrplatform.h> define our own types. */
typedef unsigned char ktx_uint8_t;
typedef bool ktx_bool_t;
#ifdef _MSC_VER
typedef unsigned __int16 ktx_uint16_t;
typedef   signed __int16 ktx_int16_t;
typedef unsigned __int32 ktx_uint32_t;
typedef   signed __int32 ktx_int32_t;
typedef          size_t  ktx_size_t;
typedef unsigned __int64 ktx_uint64_t;
typedef   signed __int64 ktx_int64_t;
#else
#include <stdint.h>
typedef uint16_t ktx_uint16_t;
typedef  int16_t ktx_int16_t;
typedef uint32_t ktx_uint32_t;
typedef  int32_t ktx_int32_t;
typedef   size_t ktx_size_t;
typedef uint64_t ktx_uint64_t;
typedef  int64_t ktx_int64_t;
#endif

/* This will cause compilation to fail if size of uint32 != 4. */
typedef unsigned char ktx_uint32_t_SIZE_ASSERT[sizeof(ktx_uint32_t) == 4];

/*
 * This #if allows libktx to be compiled with strict c99. It avoids
 * compiler warnings or even errors when a gl.h is already included.
 * "Redefinition of (type) is a c11 feature". Obviously this doesn't help if
 * gl.h comes after. However nobody has complained about the unguarded typedefs
 * since they were introduced so this is unlikely to be a problem in practice.
 * Presumably everybody is using platform default compilers not c99 or else
 * they are using C++.
 */
#if !defined(GL_NO_ERROR)
  /*
   * To avoid having to including gl.h ...
   */
  typedef unsigned char GLboolean;
  typedef unsigned int GLenum;
  typedef int GLint;
  typedef int GLsizei;
  typedef unsigned int GLuint;
  typedef unsigned char GLubyte;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @~English
 * @brief Key string for standard writer metadata.
 */
#define KTX_ANIMDATA_KEY "KTXanimData"
/**
 * @~English
 * @brief Key string for standard orientation metadata.
 */
#define KTX_ORIENTATION_KEY "KTXorientation"
/**
 * @~English
 * @brief Key string for standard swizzle metadata.
 */
#define KTX_SWIZZLE_KEY "KTXswizzle"
/**
 * @~English
 * @brief Key string for standard writer metadata.
 */
#define KTX_WRITER_KEY "KTXwriter"
/**
 * @~English
 * @brief Key string for standard writer supercompression parameter metadata.
 */
#define KTX_WRITER_SCPARAMS_KEY "KTXwriterScParams"
/**
 * @~English
 * @brief Standard KTX 1 format for 1D orientation value.
 */
#define KTX_ORIENTATION1_FMT "S=%c"
/**
 * @~English
 * @brief Standard KTX 1 format for 2D orientation value.
 */
#define KTX_ORIENTATION2_FMT "S=%c,T=%c"
/**
 * @~English
 * @brief Standard KTX 1 format for 3D orientation value.
 */
#define KTX_ORIENTATION3_FMT "S=%c,T=%c,R=%c"
/**
 * @~English
 * @brief Required unpack alignment
 */
#define KTX_GL_UNPACK_ALIGNMENT 4
#define KTX_FACESLICE_WHOLE_LEVEL UINT_MAX

#define KTX_TRUE  true
#define KTX_FALSE false

/**
 * @~English
 * @brief Error codes returned by library functions.
 */
typedef enum ktx_error_code_e {
    KTX_SUCCESS = 0,         /*!< Operation was successful. */
    KTX_FILE_DATA_ERROR,     /*!< The data in the file is inconsistent with the spec. */
    KTX_FILE_ISPIPE,         /*!< The file is a pipe or named pipe. */
    KTX_FILE_OPEN_FAILED,    /*!< The target file could not be opened. */
    KTX_FILE_OVERFLOW,       /*!< The operation would exceed the max file size. */
    KTX_FILE_READ_ERROR,     /*!< An error occurred while reading from the file. */
    KTX_FILE_SEEK_ERROR,     /*!< An error occurred while seeking in the file. */
    KTX_FILE_UNEXPECTED_EOF, /*!< File does not have enough data to satisfy request. */
    KTX_FILE_WRITE_ERROR,    /*!< An error occurred while writing to the file. */
    KTX_GL_ERROR,            /*!< GL operations resulted in an error. */
    KTX_INVALID_OPERATION,   /*!< The operation is not allowed in the current state. */
    KTX_INVALID_VALUE,       /*!< A parameter value was not valid. */
    KTX_NOT_FOUND,           /*!< Requested metadata key or required dynamically loaded GPU function was not found. */
    KTX_OUT_OF_MEMORY,       /*!< Not enough memory to complete the operation. */
    KTX_TRANSCODE_FAILED,    /*!< Transcoding of block compressed texture failed. */
    KTX_UNKNOWN_FILE_FORMAT, /*!< The file not a KTX file */
    KTX_UNSUPPORTED_TEXTURE_TYPE, /*!< The KTX file specifies an unsupported texture type. */
    KTX_UNSUPPORTED_FEATURE,  /*!< Feature not included in in-use library or not yet implemented. */
    KTX_LIBRARY_NOT_LINKED,  /*!< Library dependency (OpenGL or Vulkan) not linked into application. */
    KTX_DECOMPRESS_LENGTH_ERROR, /*!< Decompressed byte count does not match expected byte size */
    KTX_DECOMPRESS_CHECKSUM_ERROR, /*!< Checksum mismatch when decompressing */
    KTX_ERROR_MAX_ENUM = KTX_DECOMPRESS_CHECKSUM_ERROR /*!< For safety checks. */
} ktx_error_code_e;
/**
 * @deprecated
 * @~English
 * @brief For backward compatibility
 */
#define KTX_error_code ktx_error_code_e

#define KTX_IDENTIFIER_REF  { 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A }
#define KTX_ENDIAN_REF      (0x04030201)
#define KTX_ENDIAN_REF_REV  (0x01020304)
#define KTX_HEADER_SIZE     (64)

/**
 * @~English
 * @brief Result codes returned by library functions.
 */
 typedef enum ktx_error_code_e ktxResult;

/**
 * @class ktxHashList
 * @~English
 * @brief Opaque handle to a ktxHashList.
 */
typedef struct ktxKVListEntry* ktxHashList;

typedef struct ktxStream ktxStream;

#define KTX_APIENTRYP KTX_APIENTRY *
/**
 * @class ktxHashListEntry
 * @~English
 * @brief Opaque handle to an entry in a @ref ktxHashList.
 */
typedef struct ktxKVListEntry ktxHashListEntry;

typedef enum ktxOrientationX {
    KTX_ORIENT_X_LEFT = 'l', KTX_ORIENT_X_RIGHT = 'r'
} ktxOrientationX;

typedef enum ktxOrientationY {
    KTX_ORIENT_Y_UP = 'u', KTX_ORIENT_Y_DOWN = 'd'
} ktxOrientationY;

typedef enum ktxOrientationZ {
    KTX_ORIENT_Z_IN = 'i', KTX_ORIENT_Z_OUT = 'o'
} ktxOrientationZ;

typedef enum class_id {
    ktxTexture1_c = 1,
    ktxTexture2_c = 2
} class_id;

/**
 * @~English
 * @brief Struct describing the logical orientation of an image.
 */
struct ktxOrientation {
    ktxOrientationX x;  /*!< Orientation in X */
    ktxOrientationY y;  /*!< Orientation in Y */
    ktxOrientationZ z;  /*!< Orientation in Z */
};

#define KTXTEXTURECLASSDEFN                   \
    class_id classId;                         \
    struct ktxTexture_vtbl* vtbl;             \
    struct ktxTexture_vvtbl* vvtbl;           \
    struct ktxTexture_protected* _protected;  \
    ktx_bool_t   isArray;                     \
    ktx_bool_t   isCubemap;                   \
    ktx_bool_t   isCompressed;                \
    ktx_bool_t   generateMipmaps;             \
    ktx_uint32_t baseWidth;                   \
    ktx_uint32_t baseHeight;                  \
    ktx_uint32_t baseDepth;                   \
    ktx_uint32_t numDimensions;               \
    ktx_uint32_t numLevels;                   \
    ktx_uint32_t numLayers;                   \
    ktx_uint32_t numFaces;                    \
    struct ktxOrientation orientation;        \
    ktxHashList  kvDataHead;                  \
    ktx_uint32_t kvDataLen;                   \
    ktx_uint8_t* kvData;                      \
    ktx_size_t dataSize;                      \
    ktx_uint8_t* pData;


/**
 * @class ktxTexture
 * @~English
 * @brief Base class representing a texture.
 *
 * ktxTextures should be created only by one of the provided
 * functions and these fields should be considered read-only.
 */
typedef struct ktxTexture {
    KTXTEXTURECLASSDEFN
} ktxTexture;
/**
 * @typedef ktxTexture::classId
 * @~English
 * @brief Identify the class type.
 *
 * Since there are no public ktxTexture constructors, this can only have
 * values of ktxTexture1_c or ktxTexture2_c.
 */
/**
 * @typedef ktxTexture::vtbl
 * @~English
 * @brief Pointer to the class's vtble.
 */
/**
 * @typedef ktxTexture::vvtbl
 * @~English
 * @brief Pointer to the class's vtble for Vulkan functions.
 *
 * A separate vtble is used so this header does not need to include vulkan.h.
 */
/**
 * @typedef ktxTexture::_protected
 * @~English
 * @brief Opaque pointer to the class's protected variables.
 */
/**
 * @typedef ktxTexture::isArray
 * @~English
 *
 * KTX_TRUE if the texture is an array texture, i.e,
 * a GL_TEXTURE_*_ARRAY target is to be used.
 */
/**
 * @typedef ktxTexture::isCubemap
 * @~English
 *
 * KTX_TRUE if the texture is a cubemap or cubemap array.
 */
/**
 * @typedef ktxTexture::isCubemap
 * @~English
 *
 * KTX_TRUE if the texture's format is a block compressed format.
 */
/**
 * @typedef ktxTexture::generateMipmaps
 * @~English
 *
 * KTX_TRUE if mipmaps should be generated for the texture by
 * ktxTexture_GLUpload() or ktxTexture_VkUpload().
 */
/**n
 * @typedef ktxTexture::baseWidth
 * @~English
 * @brief Width of the texture's base level.
 */
/**
 * @typedef ktxTexture::baseHeight
 * @~English
 * @brief Height of the texture's base level.
 */
/**
 * @typedef ktxTexture::baseDepth
 * @~English
 * @brief Depth of the texture's base level.
 */
/**
 * @typedef ktxTexture::numDimensions
 * @~English
 * @brief Number of dimensions in the texture: 1, 2 or 3.
 */
/**
 * @typedef ktxTexture::numLevels
 * @~English
 * @brief Number of mip levels in the texture.
 *
 * Must be 1, if @c generateMipmaps is KTX_TRUE. Can be less than a
 * full pyramid but always starts at the base level.
 */
/**
 * @typedef ktxTexture::numLevels
 * @~English
 * @brief Number of array layers in the texture.
 */
/**
 * @typedef ktxTexture::numFaces
 * @~English
 * @brief Number of faces: 6 for cube maps, 1 otherwise.
 */
/**
 * @typedef ktxTexture::orientation
 * @~English
 * @brief Describes the logical orientation of the images in each dimension.
 *
 * ktxOrientationX for X, ktxOrientationY for Y and ktxOrientationZ for Z.
 */
/**
 * @typedef ktxTexture::kvDataHead
 * @~English
 * @brief Head of the hash list of metadata.
 */
/**
 * @typedef ktxTexture::kvDataLen
 * @~English
 * @brief Length of the metadata, if it has been extracted in its raw form,
 *       otherwise 0.
 */
/**
 * @typedef ktxTexture::kvData
 * @~English
 * @brief Pointer to the metadata, if it has been extracted in its raw form,
 *       otherwise NULL.
 */
/**
 * @typedef ktxTexture::dataSize
 * @~English
 * @brief Byte length of the texture's uncompressed image data.
 */
/**
 * @typedef ktxTexture::pData
 * @~English
 * @brief Pointer to the start of the image data.
 */

/**
 * @memberof ktxTexture
 * @~English
 * @brief Signature of function called by the <tt>ktxTexture_Iterate*</tt>
 *        functions to receive image data.
 *
 * The function parameters are used to pass values which change for each image.
 * Obtain values which are uniform across all images from the @c ktxTexture
 * object.
 *
 * @param [in] miplevel        MIP level from 0 to the max level which is
 *                             dependent on the texture size.
 * @param [in] face            usually 0; for cube maps, one of the 6 cube
 *                             faces in the order +X, -X, +Y, -Y, +Z, -Z,
 *                             0 to 5.
 * @param [in] width           width of the image.
 * @param [in] height          height of the image or, for 1D textures
 *                             textures, 1.
 * @param [in] depth           depth of the image or, for 1D & 2D
 *                             textures, 1.
 * @param [in] faceLodSize     number of bytes of data pointed at by
 *                             @p pixels.
 * @param [in] pixels          pointer to the image data.
 * @param [in,out] userdata    pointer for the application to pass data to and
 *                             from the callback function.
 */

typedef KTX_error_code
    (* PFNKTXITERCB)(int miplevel, int face,
                     int width, int height, int depth,
                     ktx_uint64_t faceLodSize,
                     void* pixels, void* userdata);

/* Don't use KTX_APIENTRYP to avoid a Doxygen bug. */
typedef void (KTX_APIENTRY* PFNKTEXDESTROY)(ktxTexture* This);
typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXGETIMAGEOFFSET)(ktxTexture* This, ktx_uint32_t level,
                                          ktx_uint32_t layer,
                                          ktx_uint32_t faceSlice,
                                          ktx_size_t* pOffset);
typedef ktx_size_t
    (KTX_APIENTRY* PFNKTEXGETDATASIZEUNCOMPRESSED)(ktxTexture* This);
typedef ktx_size_t
    (KTX_APIENTRY* PFNKTEXGETIMAGESIZE)(ktxTexture* This, ktx_uint32_t level);
typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXITERATELEVELS)(ktxTexture* This, PFNKTXITERCB iterCb,
                                         void* userdata);

typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXITERATELOADLEVELFACES)(ktxTexture* This,
                                                 PFNKTXITERCB iterCb,
                                                 void* userdata);
typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXLOADIMAGEDATA)(ktxTexture* This,
                                         ktx_uint8_t* pBuffer,
                                         ktx_size_t bufSize);
typedef ktx_bool_t
    (KTX_APIENTRY* PFNKTEXNEEDSTRANSCODING)(ktxTexture* This);

typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXSETIMAGEFROMMEMORY)(ktxTexture* This,
                                              ktx_uint32_t level,
                                              ktx_uint32_t layer,
                                              ktx_uint32_t faceSlice,
                                              const ktx_uint8_t* src,
                                              ktx_size_t srcSize);

typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXSETIMAGEFROMSTDIOSTREAM)(ktxTexture* This,
                                                   ktx_uint32_t level,
                                                   ktx_uint32_t layer,
                                                   ktx_uint32_t faceSlice,
                                                   FILE* src, ktx_size_t srcSize);
typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXWRITETOSTDIOSTREAM)(ktxTexture* This, FILE* dstsstr);
typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXWRITETONAMEDFILE)(ktxTexture* This,
                                            const char* const dstname);
typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXWRITETOMEMORY)(ktxTexture* This,
                                         ktx_uint8_t** bytes, ktx_size_t* size);
typedef KTX_error_code
    (KTX_APIENTRY* PFNKTEXWRITETOSTREAM)(ktxTexture* This,
                                         ktxStream* dststr);

/**
 * @memberof ktxTexture
 * @~English
 * @brief Table of virtual ktxTexture methods.
 */
 struct ktxTexture_vtbl {
    PFNKTEXDESTROY Destroy;
    PFNKTEXGETIMAGEOFFSET GetImageOffset;
    PFNKTEXGETDATASIZEUNCOMPRESSED GetDataSizeUncompressed;
    PFNKTEXGETIMAGESIZE GetImageSize;
    PFNKTEXITERATELEVELS IterateLevels;
    PFNKTEXITERATELOADLEVELFACES IterateLoadLevelFaces;
    PFNKTEXNEEDSTRANSCODING NeedsTranscoding;
    PFNKTEXLOADIMAGEDATA LoadImageData;
    PFNKTEXSETIMAGEFROMMEMORY SetImageFromMemory;
    PFNKTEXSETIMAGEFROMSTDIOSTREAM SetImageFromStdioStream;
    PFNKTEXWRITETOSTDIOSTREAM WriteToStdioStream;
    PFNKTEXWRITETONAMEDFILE WriteToNamedFile;
    PFNKTEXWRITETOMEMORY WriteToMemory;
    PFNKTEXWRITETOSTREAM WriteToStream;
};

/****************************************************************
 * Macros to give some backward compatibility to the previous API
 ****************************************************************/

/**
 * @~English
 * @brief Helper for calling the Destroy virtual method of a ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_Destroy
 */
#define ktxTexture_Destroy(This) (This)->vtbl->Destroy(This)

/**
 * @~English
 * @brief Helper for calling the GetImageOffset virtual method of a
 *        ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_GetImageOffset
 */
#define ktxTexture_GetImageOffset(This, level, layer, faceSlice, pOffset) \
            (This)->vtbl->GetImageOffset(This, level, layer, faceSlice, pOffset)

/**
 * @~English
 * @brief Helper for calling the GetDataSizeUncompressed virtual method of a ktxTexture.
 *
 * For a ktxTexture1 this will always return the value of This->dataSize.
 *
 * @copydetails ktxTexture2.ktxTexture2_GetDataSizeUncompressed
 */
#define ktxTexture_GetDataSizeUncompressed(This) \
                                (This)->vtbl->GetDataSizeUncompressed(This)

/**
 * @~English
 * @brief Helper for calling the GetImageSize virtual method of a ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_GetImageSize
 */
#define ktxTexture_GetImageSize(This, level) \
            (This)->vtbl->GetImageSize(This, level)

/**
 * @~English
 * @brief Helper for calling the IterateLevels virtual method of a ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_IterateLevels
 */
#define ktxTexture_IterateLevels(This, iterCb, userdata) \
                            (This)->vtbl->IterateLevels(This, iterCb, userdata)

/**
 * @~English
 * @brief Helper for calling the IterateLoadLevelFaces virtual method of a
 * ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_IterateLoadLevelFaces
 */
 #define ktxTexture_IterateLoadLevelFaces(This, iterCb, userdata) \
                    (This)->vtbl->IterateLoadLevelFaces(This, iterCb, userdata)

/**
 * @~English
 * @brief Helper for calling the LoadImageData virtual method of a ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_LoadImageData
 */
#define ktxTexture_LoadImageData(This, pBuffer, bufSize) \
                    (This)->vtbl->LoadImageData(This, pBuffer, bufSize)

/**
 * @~English
 * @brief Helper for calling the NeedsTranscoding virtual method of a ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_NeedsTranscoding
 */
#define ktxTexture_NeedsTranscoding(This) (This)->vtbl->NeedsTranscoding(This)

/**
 * @~English
 * @brief Helper for calling the SetImageFromMemory virtual method of a
 *        ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_SetImageFromMemory
 */
#define ktxTexture_SetImageFromMemory(This, level, layer, faceSlice, \
                                      src, srcSize)                  \
    (This)->vtbl->SetImageFromMemory(This, level, layer, faceSlice, src, srcSize)

/**
 * @~English
 * @brief Helper for calling the SetImageFromStdioStream virtual method of a
 *        ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_SetImageFromStdioStream
 */
#define ktxTexture_SetImageFromStdioStream(This, level, layer, faceSlice, \
                                           src, srcSize)                  \
    (This)->vtbl->SetImageFromStdioStream(This, level, layer, faceSlice,  \
                                        src, srcSize)

/**
 * @~English
 * @brief Helper for calling the WriteToStdioStream virtual method of a
 *        ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_WriteToStdioStream
 */
#define ktxTexture_WriteToStdioStream(This, dstsstr) \
                                (This)->vtbl->WriteToStdioStream(This, dstsstr)

/**
 * @~English
 * @brief Helper for calling the WriteToNamedfile virtual method of a
 *        ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_WriteToNamedFile
 */
#define ktxTexture_WriteToNamedFile(This, dstname) \
                                (This)->vtbl->WriteToNamedFile(This, dstname)

/**
 * @~English
 * @brief Helper for calling the WriteToMemory virtual method of a ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_WriteToMemory
 */
#define ktxTexture_WriteToMemory(This, ppDstBytes, pSize) \
                  (This)->vtbl->WriteToMemory(This, ppDstBytes, pSize)

/**
 * @~English
 * @brief Helper for calling the WriteToStream virtual method of a ktxTexture.
 * @copydoc ktxTexture2.ktxTexture2_WriteToStream
 */
#define ktxTexture_WriteToStream(This, dststr) \
                  (This)->vtbl->WriteToStream(This, dststr)


/**
 * @class ktxTexture1
 * @~English
 * @brief Class representing a KTX version 1 format texture.
 *
 * ktxTextures should be created only by one of the ktxTexture_Create*
 * functions and these fields should be considered read-only.
 */
typedef struct ktxTexture1 {
    KTXTEXTURECLASSDEFN
    ktx_uint32_t glFormat; /*!< Format of the texture data, e.g., GL_RGB. */
    ktx_uint32_t glInternalformat; /*!< Internal format of the texture data,
                                        e.g., GL_RGB8. */
    ktx_uint32_t glBaseInternalformat; /*!< Base format of the texture data,
                                            e.g., GL_RGB. */
    ktx_uint32_t glType; /*!< Type of the texture data, e.g, GL_UNSIGNED_BYTE.*/
    struct ktxTexture1_private* _private; /*!< Private data. */
} ktxTexture1;

/*===========================================================*
* KTX format version 2                                      *
*===========================================================*/

/**
 * @~English
 * @brief Enumerators identifying the supercompression scheme.
 */
typedef enum ktxSupercmpScheme {
    KTX_SS_NONE = 0,            /*!< No supercompression. */
    KTX_SS_BASIS_LZ = 1,        /*!< Basis LZ supercompression. */
    KTX_SS_ZSTD = 2,            /*!< ZStd supercompression. */
    KTX_SS_ZLIB = 3,            /*!< ZLIB supercompression. */
    KTX_SS_BEGIN_RANGE = KTX_SS_NONE,
    KTX_SS_END_RANGE = KTX_SS_ZLIB,
    KTX_SS_BEGIN_VENDOR_RANGE = 0x10000,
    KTX_SS_END_VENDOR_RANGE = 0x1ffff,
    KTX_SS_BEGIN_RESERVED = 0x20000,
    KTX_SUPERCOMPRESSION_BASIS = KTX_SS_BASIS_LZ,
        /*!< @deprecated Will be removed before v4 release. Use  KTX_SS_BASIS_LZ instead. */
    KTX_SUPERCOMPRESSION_ZSTD = KTX_SS_ZSTD
        /*!< @deprecated Will be removed before v4 release. Use  KTX_SS_ZSTD instead. */
} ktxSupercmpScheme;

/**
 * @class ktxTexture2
 * @~English
 * @brief Class representing a KTX version 2 format texture.
 *
 * ktxTextures should be created only by one of the ktxTexture_Create*
 * functions and these fields should be considered read-only.
 */
typedef struct ktxTexture2 {
    KTXTEXTURECLASSDEFN
    ktx_uint32_t  vkFormat;
    ktx_uint32_t* pDfd;
    ktxSupercmpScheme supercompressionScheme;
    ktx_bool_t isVideo;
    ktx_uint32_t duration;
    ktx_uint32_t timescale;
    ktx_uint32_t loopcount;
    struct ktxTexture2_private* _private;  /*!< Private data. */
} ktxTexture2;

/**
 * @brief Helper for casting ktxTexture1 and ktxTexture2 to ktxTexture.
 *
 * Use with caution.
 */
#define ktxTexture(t) ((ktxTexture*)t)

/**
 * @memberof ktxTexture
 * @~English
 * @brief Structure for passing texture information to ktxTexture1\_Create() and
 *        ktxTexture2\_Create().
 *
 * @sa @ref ktxTexture1::ktxTexture1\_Create() "ktxTexture1_Create()"
 * @sa @ref ktxTexture2::ktxTexture2\_Create() "ktxTexture2_Create()"
 */
typedef struct
{
    ktx_uint32_t glInternalformat; /*!< Internal format for the texture, e.g.,
                                        GL_RGB8. Ignored when creating a
                                        ktxTexture2. */
    ktx_uint32_t vkFormat;   /*!< VkFormat for texture. Ignored when creating a
                                  ktxTexture1. */
    ktx_uint32_t* pDfd;      /*!< Pointer to DFD. Used only when creating a
                                  ktxTexture2 and only if vkFormat is
                                  VK_FORMAT_UNDEFINED. */
    ktx_uint32_t baseWidth;  /*!< Width of the base level of the texture. */
    ktx_uint32_t baseHeight; /*!< Height of the base level of the texture. */
    ktx_uint32_t baseDepth;  /*!< Depth of the base level of the texture. */
    ktx_uint32_t numDimensions; /*!< Number of dimensions in the texture, 1, 2
                                     or 3. */
    ktx_uint32_t numLevels; /*!< Number of mip levels in the texture. Should be
                                 1 if @c generateMipmaps is KTX_TRUE; */
    ktx_uint32_t numLayers; /*!< Number of array layers in the texture. */
    ktx_uint32_t numFaces;  /*!< Number of faces: 6 for cube maps, 1 otherwise. */
    ktx_bool_t   isArray;  /*!< Set to KTX_TRUE if the texture is to be an
                                array texture. Means OpenGL will use a
                                GL_TEXTURE_*_ARRAY target. */
    ktx_bool_t   generateMipmaps; /*!< Set to KTX_TRUE if mipmaps should be
                                       generated for the texture when loading
                                       into a 3D API. */
} ktxTextureCreateInfo;

/**
 * @memberof ktxTexture
 * @~English
 * @brief Enum for requesting, or not, allocation of storage for images.
 *
 * @sa ktxTexture1_Create() and ktxTexture2_Create().
 */
typedef enum {
    KTX_TEXTURE_CREATE_NO_STORAGE = 0,  /*!< Don't allocate any image storage. */
    KTX_TEXTURE_CREATE_ALLOC_STORAGE = 1 /*!< Allocate image storage. */
} ktxTextureCreateStorageEnum;

/**
 * @memberof ktxTexture
 * @~English
 * @brief Flags for requesting services during creation.
 *
 * @sa ktxTexture_CreateFrom*
 */
enum ktxTextureCreateFlagBits {
    KTX_TEXTURE_CREATE_NO_FLAGS = 0x00,
    KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT = 0x01,
                                   /*!< Load the images from the KTX source. */
    KTX_TEXTURE_CREATE_RAW_KVDATA_BIT = 0x02,
                                   /*!< Load the raw key-value data instead of
                                        creating a @c ktxHashList from it. */
    KTX_TEXTURE_CREATE_SKIP_KVDATA_BIT = 0x04,
                                   /*!< Skip any key-value data. This overrides
                                        the RAW_KVDATA_BIT. */
    KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT = 0x08
                                   /*!< Load texture compatible with the rules
                                        of KHR_texture_basisu glTF extension */
};
/**
 * @memberof ktxTexture
 * @~English
 * @brief Type for TextureCreateFlags parameters.
 *
 * @sa ktxTexture_CreateFrom*()
 */
typedef ktx_uint32_t ktxTextureCreateFlags;

/*===========================================================*
* ktxStream
*===========================================================*/

/*
 * This is unsigned to allow ktxmemstreams to use the
 * full amount of memory available. Platforms will
 * limit the size of ktxfilestreams to, e.g, MAX_LONG
 * on 32-bit and ktxfilestreams raises errors if
 * offset values exceed the limits. This choice may
 * need to be revisited if we ever start needing -ve
 * offsets.
 *
 * Should the 2GB file size handling limit on 32-bit
 * platforms become a problem, ktxfilestream will have
 * to be changed to explicitly handle large files by
 * using the 64-bit stream functions.
 */
#if defined(_MSC_VER) && defined(_WIN64)
  typedef unsigned __int64 ktx_off_t;
#else
  typedef   off_t ktx_off_t;
#endif
typedef struct ktxMem ktxMem;
typedef struct ktxStream ktxStream;

enum streamType { eStreamTypeFile = 1, eStreamTypeMemory = 2, eStreamTypeCustom = 3 };

/**
 * @~English
 * @brief type for a pointer to a stream reading function
 */
typedef KTX_error_code (*ktxStream_read)(ktxStream* str, void* dst,
                                         const ktx_size_t count);
/**
 * @~English
 * @brief type for a pointer to a stream skipping function
 */
typedef KTX_error_code (*ktxStream_skip)(ktxStream* str,
                                         const ktx_size_t count);

/**
 * @~English
 * @brief type for a pointer to a stream writing function
 */
typedef KTX_error_code (*ktxStream_write)(ktxStream* str, const void *src,
                                          const ktx_size_t size,
                                          const ktx_size_t count);

/**
 * @~English
 * @brief type for a pointer to a stream position query function
 */
typedef KTX_error_code (*ktxStream_getpos)(ktxStream* str, ktx_off_t* const offset);

/**
 * @~English
 * @brief type for a pointer to a stream position query function
 */
typedef KTX_error_code (*ktxStream_setpos)(ktxStream* str, const ktx_off_t offset);

/**
 * @~English
 * @brief type for a pointer to a stream size query function
 */
typedef KTX_error_code (*ktxStream_getsize)(ktxStream* str, ktx_size_t* const size);

/**
 * @~English
 * @brief Destruct a stream
 */
typedef void (*ktxStream_destruct)(ktxStream* str);

/**
 * @~English
 *
 * @brief Interface of ktxStream.
 *
 * @author Maksim Kolesin
 * @author Georg Kolling, Imagination Technology
 * @author Mark Callow, HI Corporation
 */
struct ktxStream
{
    ktxStream_read read;   /*!< pointer to function for reading bytes. */
    ktxStream_skip skip;   /*!< pointer to function for skipping bytes. */
    ktxStream_write write; /*!< pointer to function for writing bytes. */
    ktxStream_getpos getpos; /*!< pointer to function for getting current position in stream. */
    ktxStream_setpos setpos; /*!< pointer to function for setting current position in stream. */
    ktxStream_getsize getsize; /*!< pointer to function for querying size. */
    ktxStream_destruct destruct; /*!< destruct the stream. */

    enum streamType type;
    union {
        FILE* file;        /**< a stdio FILE pointer for a ktxFileStream. */
        ktxMem* mem;       /**< a pointer to a ktxMem struct for a ktxMemStream. */
        struct
        {
            void* address;           /**< pointer to the data. */
            void* allocatorAddress;  /**< pointer to a memory allocator. */
            ktx_size_t size;         /**< size of the data. */
        } custom_ptr;      /**< pointer to a struct for custom streams. */
    } data;                /**< pointer to the stream data. */
    ktx_off_t readpos;     /**< used by FileStream for stdin. */
    ktx_bool_t closeOnDestruct; /**< Close FILE* or dispose of memory on destruct. */
};

/*
 * See the implementation files for the full documentation of the following
 * functions.
 */

/*
 * These four create a ktxTexture1 or ktxTexture2 according to the data
 * header, and return a pointer to the base ktxTexture class.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture_CreateFromStdioStream(FILE* stdioStream,
                                 ktxTextureCreateFlags createFlags,
                                 ktxTexture** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture_CreateFromNamedFile(const char* const filename,
                               ktxTextureCreateFlags createFlags,
                               ktxTexture** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture_CreateFromMemory(const ktx_uint8_t* bytes, ktx_size_t size,
                            ktxTextureCreateFlags createFlags,
                            ktxTexture** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture_CreateFromStream(ktxStream* stream,
                            ktxTextureCreateFlags createFlags,
                            ktxTexture** newTex);

/*
 * Returns a pointer to the image data of a ktxTexture object.
 */
KTX_API ktx_uint8_t* KTX_APIENTRY
ktxTexture_GetData(ktxTexture* This);

/*
 * Returns the pitch of a row of an image at the specified level.
 * Similar to the rowPitch in a VkSubResourceLayout.
 */
KTX_API ktx_uint32_t KTX_APIENTRY
ktxTexture_GetRowPitch(ktxTexture* This, ktx_uint32_t level);

 /*
  * Return the element size of the texture's images.
  */
KTX_API ktx_uint32_t KTX_APIENTRY
ktxTexture_GetElementSize(ktxTexture* This);

/*
 * Returns the size of all the image data of a ktxTexture object in bytes.
 */
KTX_API ktx_size_t KTX_APIENTRY
ktxTexture_GetDataSize(ktxTexture* This);

/* Uploads a texture to OpenGL {,ES}. */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture_GLUpload(ktxTexture* This, GLuint* pTexture, GLenum* pTarget,
                    GLenum* pGlerror);

/*
 * Iterate over the levels or faces in a ktxTexture object.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture_IterateLevelFaces(ktxTexture* This, PFNKTXITERCB iterCb,
                             void* userdata);
/*
 * Create a new ktxTexture1.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_Create(ktxTextureCreateInfo* createInfo,
                   ktxTextureCreateStorageEnum storageAllocation,
                   ktxTexture1** newTex);

/*
 * These four create a ktxTexture1 provided the data is in KTX format.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_CreateFromStdioStream(FILE* stdioStream,
                                 ktxTextureCreateFlags createFlags,
                                 ktxTexture1** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_CreateFromNamedFile(const char* const filename,
                               ktxTextureCreateFlags createFlags,
                               ktxTexture1** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_CreateFromMemory(const ktx_uint8_t* bytes, ktx_size_t size,
                            ktxTextureCreateFlags createFlags,
                            ktxTexture1** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_CreateFromStream(ktxStream* stream,
                             ktxTextureCreateFlags createFlags,
                             ktxTexture1** newTex);

KTX_API ktx_bool_t KTX_APIENTRY
ktxTexture1_NeedsTranscoding(ktxTexture1* This);

/*
 * Write a ktxTexture object to a stdio stream in KTX format.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_WriteKTX2ToStdioStream(ktxTexture1* This, FILE* dstsstr);

/*
 * Write a ktxTexture object to a named file in KTX format.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_WriteKTX2ToNamedFile(ktxTexture1* This, const char* const dstname);

/*
 * Write a ktxTexture object to a block of memory in KTX format.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_WriteKTX2ToMemory(ktxTexture1* This,
                             ktx_uint8_t** bytes, ktx_size_t* size);

/*
 * Write a ktxTexture object to a ktxStream in KTX format.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_WriteKTX2ToStream(ktxTexture1* This, ktxStream *dststr);

/*
 * Create a new ktxTexture2.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_Create(ktxTextureCreateInfo* createInfo,
                   ktxTextureCreateStorageEnum storageAllocation,
                   ktxTexture2** newTex);

/*
 * Create a new ktxTexture2 as a copy of an existing texture.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CreateCopy(ktxTexture2* orig, ktxTexture2** newTex);

 /*
  * These four create a ktxTexture2 provided the data is in KTX2 format.
  */
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CreateFromStdioStream(FILE* stdioStream,
                                 ktxTextureCreateFlags createFlags,
                                 ktxTexture2** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CreateFromNamedFile(const char* const filename,
                               ktxTextureCreateFlags createFlags,
                               ktxTexture2** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CreateFromMemory(const ktx_uint8_t* bytes, ktx_size_t size,
                            ktxTextureCreateFlags createFlags,
                            ktxTexture2** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CreateFromStream(ktxStream* stream,
                             ktxTextureCreateFlags createFlags,
                             ktxTexture2** newTex);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CompressBasis(ktxTexture2* This, ktx_uint32_t quality);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_DeflateZstd(ktxTexture2* This, ktx_uint32_t level);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_DeflateZLIB(ktxTexture2* This, ktx_uint32_t level);

KTX_API void KTX_APIENTRY
ktxTexture2_GetComponentInfo(ktxTexture2* This, ktx_uint32_t* numComponents,
                             ktx_uint32_t* componentByteLength);

KTX_API ktx_uint32_t KTX_APIENTRY
ktxTexture2_GetNumComponents(ktxTexture2* This);

KTX_API khr_df_transfer_e KTX_APIENTRY
ktxTexture2_GetOETF_e(ktxTexture2* This);

// For backward compatibility
KTX_API ktx_uint32_t KTX_APIENTRY
ktxTexture2_GetOETF(ktxTexture2* This);

KTX_API khr_df_model_e KTX_APIENTRY
ktxTexture2_GetColorModel_e(ktxTexture2* This);

KTX_API ktx_bool_t KTX_APIENTRY
ktxTexture2_GetPremultipliedAlpha(ktxTexture2* This);

KTX_API ktx_bool_t KTX_APIENTRY
ktxTexture2_NeedsTranscoding(ktxTexture2* This);

/**
 * @~English
 * @brief Flags specifiying UASTC encoding options.
 */
typedef enum ktx_pack_uastc_flag_bits_e {
    KTX_PACK_UASTC_LEVEL_FASTEST  = 0,
        /*!< Fastest compression. 43.45dB. */
    KTX_PACK_UASTC_LEVEL_FASTER   = 1,
        /*!< Faster compression. 46.49dB. */
    KTX_PACK_UASTC_LEVEL_DEFAULT  = 2,
        /*!< Default compression. 47.47dB. */
    KTX_PACK_UASTC_LEVEL_SLOWER   = 3,
        /*!< Slower compression. 48.01dB. */
    KTX_PACK_UASTC_LEVEL_VERYSLOW = 4,
        /*!< Very slow compression. 48.24dB. */
    KTX_PACK_UASTC_MAX_LEVEL = KTX_PACK_UASTC_LEVEL_VERYSLOW,
        /*!< Maximum supported quality level. */
    KTX_PACK_UASTC_LEVEL_MASK     = 0xF,
        /*!< Mask to extract the level from the other bits. */
    KTX_PACK_UASTC_FAVOR_UASTC_ERROR = 8,
        /*!< Optimize for lowest UASTC error. */
    KTX_PACK_UASTC_FAVOR_BC7_ERROR = 16,
        /*!< Optimize for lowest BC7 error. */
    KTX_PACK_UASTC_ETC1_FASTER_HINTS = 64,
        /*!< Optimize for faster transcoding to ETC1. */
    KTX_PACK_UASTC_ETC1_FASTEST_HINTS = 128,
        /*!< Optimize for fastest transcoding to ETC1. */
    KTX_PACK_UASTC__ETC1_DISABLE_FLIP_AND_INDIVIDUAL = 256
        /*!< Not documented in BasisU code. */
} ktx_pack_uastc_flag_bits_e;
typedef ktx_uint32_t ktx_pack_uastc_flags;

/**
 * @~English
 * @brief Options specifiying ASTC encoding quality levels.
 */
typedef enum ktx_pack_astc_quality_levels_e {
    KTX_PACK_ASTC_QUALITY_LEVEL_FASTEST  = 0,
        /*!< Fastest compression. */
    KTX_PACK_ASTC_QUALITY_LEVEL_FAST   = 10,
        /*!< Fast compression. */
    KTX_PACK_ASTC_QUALITY_LEVEL_MEDIUM   = 60,
        /*!< Medium compression. */
    KTX_PACK_ASTC_QUALITY_LEVEL_THOROUGH   = 98,
        /*!< Slower compression. */
    KTX_PACK_ASTC_QUALITY_LEVEL_EXHAUSTIVE = 100,
        /*!< Very slow compression. */
    KTX_PACK_ASTC_QUALITY_LEVEL_MAX = KTX_PACK_ASTC_QUALITY_LEVEL_EXHAUSTIVE,
        /*!< Maximum supported quality level. */
} ktx_pack_astc_quality_levels_e;

/**
 * @~English
 * @brief Options specifiying ASTC encoding block dimensions
 */
typedef enum ktx_pack_astc_block_dimension_e {
    // 2D formats
    KTX_PACK_ASTC_BLOCK_DIMENSION_4x4,                    //: 8.00 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_5x4,                    //: 6.40 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_5x5,                    //: 5.12 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_6x5,                    //: 4.27 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_6x6,                    //: 3.56 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_8x5,                    //: 3.20 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_8x6,                    //: 2.67 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_10x5,                   //: 2.56 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_10x6,                   //: 2.13 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_8x8,                    //: 2.00 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_10x8,                   //: 1.60 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_10x10,                  //: 1.28 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_12x10,                  //: 1.07 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_12x12,                  //: 0.89 bpp
    // 3D formats
    KTX_PACK_ASTC_BLOCK_DIMENSION_3x3x3,                  //: 4.74 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_4x3x3,                  //: 3.56 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_4x4x3,                  //: 2.67 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_4x4x4,                  //: 2.00 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_5x4x4,                  //: 1.60 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_5x5x4,                  //: 1.28 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_5x5x5,                  //: 1.02 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_6x5x5,                  //: 0.85 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_6x6x5,                  //: 0.71 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_6x6x6,                  //: 0.59 bpp
    KTX_PACK_ASTC_BLOCK_DIMENSION_MAX = KTX_PACK_ASTC_BLOCK_DIMENSION_6x6x6
        /*!< Maximum supported blocks. */
} ktx_pack_astc_block_dimension_e;

/**
 * @~English
 * @brief Options specifying ASTC encoder profile mode
 *        This and function is used later to derive the profile.
 */
typedef enum ktx_pack_astc_encoder_mode_e {
    KTX_PACK_ASTC_ENCODER_MODE_DEFAULT,
    KTX_PACK_ASTC_ENCODER_MODE_LDR,
    KTX_PACK_ASTC_ENCODER_MODE_HDR,
    KTX_PACK_ASTC_ENCODER_MODE_MAX = KTX_PACK_ASTC_ENCODER_MODE_HDR
} ktx_pack_astc_encoder_mode_e;

extern KTX_API const ktx_uint32_t KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL;

/**
 * @memberof ktxTexture
 * @~English
 * @brief Structure for passing extended parameters to
 *        ktxTexture_CompressAstc.
 *
 * Passing a struct initialized to 0 (e.g. " = {0};") will use blockDimension
 * 4x4, mode LDR and qualityLevel FASTEST. Setting qualityLevel to
 * KTX_PACK_ASTC_QUALITY_LEVEL_MEDIUM is recommended.
 */
typedef struct ktxAstcParams {
    ktx_uint32_t structSize;
        /*!< Size of this struct. Used so library can tell which version
             of struct is being passed.
         */

    ktx_bool_t verbose;
        /*!< If true, prints Astc encoder operation details to
             @c stdout. Not recommended for GUI apps.
         */

    ktx_uint32_t threadCount;
        /*!< Number of threads used for compression. Default is 1.
         */

    /* astcenc params */
    ktx_uint32_t blockDimension;
        /*!< Combinations of block dimensions that astcenc supports
          i.e. 6x6, 8x8, 6x5 etc
         */

    ktx_uint32_t mode;
        /*!< Can be {ldr/hdr} from astcenc
         */

    ktx_uint32_t qualityLevel;
        /*!< astcenc supports -fastest, -fast, -medium, -thorough, -exhaustive
         */

    ktx_bool_t normalMap;
        /*!< Tunes codec parameters for better quality on normal maps
          In this mode normals are compressed to X,Y components
          Discarding Z component, reader will need to generate Z
          component in shaders.
         */

    ktx_bool_t perceptual;
        /*!< The codec should optimize for perceptual error, instead of direct
           RMS error. This aims to improves perceived image quality, but
           typically lowers the measured PSNR score. Perceptual methods are
           currently only available for normal maps and RGB color data.
         */

    char inputSwizzle[4];
         /*!< A swizzle to provide as input to astcenc. It must match the regular
             expression /^[rgba01]{4}$/.
          */
} ktxAstcParams;

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CompressAstcEx(ktxTexture2* This, ktxAstcParams* params);

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CompressAstc(ktxTexture2* This, ktx_uint32_t quality);

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Structure for passing extended parameters to
 *        ktxTexture2_CompressBasisEx().
 *
 * If you only want default values, use ktxTexture2_CompressBasis(). Here, at a minimum you
 * must initialize the structure as follows:
 * @code
 *  ktxBasisParams params = {0};
 *  params.structSize = sizeof(params);
 *  params.compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL;
 * @endcode
 *
 * @e compressionLevel has to be explicitly set because 0 is a valid @e compressionLevel
 * but is not the default used by the BasisU encoder when no value is set. Only the other
 * settings that are to be non-default must be non-zero.
 */
typedef struct ktxBasisParams {
    ktx_uint32_t structSize;
        /*!< Size of this struct. Used so library can tell which version
             of struct is being passed.
         */
    ktx_bool_t uastc;
        /*!<  True to use UASTC base, false to use ETC1S base. */
    ktx_bool_t verbose;
        /*!< If true, prints Basis Universal encoder operation details to
             @c stdout. Not recommended for GUI apps.
         */
    ktx_bool_t noSSE;
        /*!< True to forbid use of the SSE instruction set. Ignored if CPU
             does not support SSE. */
    ktx_uint32_t threadCount;
        /*!< Number of threads used for compression. Default is 1. */

    /* ETC1S params */

    ktx_uint32_t compressionLevel;
        /*!< Encoding speed vs. quality tradeoff. Range is [0,5]. Higher values
             are slower, but give higher quality. There is no default. Callers
             must explicitly set this value. Callers can use
             KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL as a default value.
             Currently this is 2.
        */
    ktx_uint32_t qualityLevel;
        /*!< Compression quality. Range is [1,255].  Lower gives better
             compression/lower quality/faster. Higher gives less compression
             /higher quality/slower.  This automatically determines values for
             @c maxEndpoints, @c maxSelectors,
             @c endpointRDOThreshold and @c selectorRDOThreshold
             for the target quality level. Setting these parameters overrides
             the values determined by @c qualityLevel which defaults to
             128 if neither it nor both of @c maxEndpoints and
             @c maxSelectors have been set.
             @note @e Both of @c maxEndpoints and @c maxSelectors
             must be set for them to have any effect.
             @note qualityLevel will only determine values for
             @c endpointRDOThreshold and @c selectorRDOThreshold
             when its value exceeds 128, otherwise their defaults will be used.
        */
    ktx_uint32_t maxEndpoints;
        /*!< Manually set the max number of color endpoint clusters.
             Range is [1,16128]. Default is 0, unset. If this is set, maxSelectors
             must also be set, otherwise the value will be ignored.
         */
    float endpointRDOThreshold;
        /*!< Set endpoint RDO quality threshold. The default is 1.25. Lower is
             higher quality but less quality per output bit (try [1.0,3.0].
             This will override the value chosen by @c qualityLevel.
         */
    ktx_uint32_t maxSelectors;
        /*!< Manually set the max number of color selector clusters. Range
             is [1,16128]. Default is 0, unset. If this is set, maxEndpoints
             must also be set, otherwise the value will be ignored.
         */
    float selectorRDOThreshold;
        /*!< Set selector RDO quality threshold. The default is 1.5. Lower is
             higher quality but less quality per output bit (try [1.0,3.0]).
             This will override the value chosen by @c qualityLevel.
         */
    char inputSwizzle[4];
        /*!< A swizzle to apply before encoding. It must match the regular
             expression /^[rgba01]{4}$/. If both this and preSwizzle
             are specified ktxTexture_CompressBasisEx will raise
             KTX_INVALID_OPERATION.
         */
    ktx_bool_t normalMap;
        /*!< Tunes codec parameters for better quality on normal maps (no
             selector RDO, no endpoint RDO) and sets the texture's DFD appropriately.
             Only valid for linear textures.
         */
    ktx_bool_t separateRGToRGB_A;
        /*!< @deprecated. This was and is a no-op. 2-component inputs have always been
             automatically separated using an "rrrg" inputSwizzle. @sa inputSwizzle and normalMode.
         */
    ktx_bool_t preSwizzle;
        /*!< If the texture has @c KTXswizzle metadata, apply it before
             compressing. Swizzling, like @c rabb may yield drastically
             different error metrics if done after supercompression.
         */
    ktx_bool_t noEndpointRDO;
        /*!< Disable endpoint rate distortion optimizations. Slightly faster,
             less noisy output, but lower quality per output bit. Default is
             KTX_FALSE.
         */
    ktx_bool_t noSelectorRDO;
        /*!< Disable selector rate distortion optimizations. Slightly faster,
             less noisy output, but lower quality per output bit. Default is
             KTX_FALSE.
         */

    /* UASTC params */

    ktx_pack_uastc_flags uastcFlags;
        /*!<  A set of ::ktx_pack_uastc_flag_bits_e controlling UASTC
             encoding. The most important value is the level given in the
             least-significant 4 bits which selects a speed vs quality tradeoff
             as shown in the following table:

                Level/Speed | Quality
                :-----: | :-------:
                KTX_PACK_UASTC_LEVEL_FASTEST | 43.45dB
                KTX_PACK_UASTC_LEVEL_FASTER | 46.49dB
                KTX_PACK_UASTC_LEVEL_DEFAULT | 47.47dB
                KTX_PACK_UASTC_LEVEL_SLOWER  | 48.01dB
                KTX_PACK_UASTC_LEVEL_VERYSLOW | 48.24dB
         */
    ktx_bool_t uastcRDO;
        /*!< Enable Rate Distortion Optimization (RDO) post-processing.
         */
    float uastcRDOQualityScalar;
        /*!< UASTC RDO quality scalar (lambda). Lower values yield higher
             quality/larger LZ compressed files, higher values yield lower
             quality/smaller LZ compressed files. A good range to try is [.2,4].
             Full range is [.001,50.0]. Default is 1.0.
         */
    ktx_uint32_t uastcRDODictSize;
        /*!< UASTC RDO dictionary size in bytes. Default is 4096. Lower
             values=faster, but give less compression. Range is [64,65536].
         */
    float uastcRDOMaxSmoothBlockErrorScale;
        /*!< UASTC RDO max smooth block error scale. Range is [1,300].
             Default is 10.0, 1.0 is disabled. Larger values suppress more
             artifacts (and allocate more bits) on smooth blocks.
         */
    float uastcRDOMaxSmoothBlockStdDev;
        /*!< UASTC RDO max smooth block standard deviation. Range is
             [.01,65536.0]. Default is 18.0. Larger values expand the range of
             blocks considered smooth.
         */
    ktx_bool_t uastcRDODontFavorSimplerModes;
        /*!< Do not favor simpler UASTC modes in RDO mode.
         */
    ktx_bool_t uastcRDONoMultithreading;
        /*!< Disable RDO multithreading (slightly higher compression,
             deterministic).
         */

} ktxBasisParams;

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_CompressBasisEx(ktxTexture2* This, ktxBasisParams* params);

/**
 * @~English
 * @brief Enumerators for specifying the transcode target format.
 *
 * For BasisU/ETC1S format, @e Opaque and @e alpha here refer to 2 separate
 * RGB images, a.k.a slices within the BasisU compressed data. For UASTC
 * format they refer to the RGB and the alpha components of the UASTC data. If
 * the original image had only 2 components, R will be in the opaque portion
 * and G in the alpha portion. The R value will be replicated in the RGB
 * components. In the case of BasisU the G value will be replicated in all 3
 * components of the alpha slice. If the original image had only 1 component
 * it's value is replicated in all 3 components of the opaque portion and
 * there is no alpha.
 *
 * @note You should not transcode sRGB encoded data to @c KTX_TTF_BC4_R,
 * @c KTX_TTF_BC5_RG, @c KTX_TTF_ETC2_EAC_R{,G}11, @c KTX_TTF_RGB565,
 * @c KTX_TTF_BGR565 or @c KTX_TTF_RGBA4444 formats as neither OpenGL nor
 * Vulkan support sRGB variants of these. Doing sRGB decoding in the shader
 * will not produce correct results if any texture filtering is being used.
 */
typedef enum ktx_transcode_fmt_e {
        // Compressed formats

        // ETC1-2
        KTX_TTF_ETC1_RGB = 0,
            /*!< Opaque only. Returns RGB or alpha data, if
                 KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS flag is
                 specified. */
        KTX_TTF_ETC2_RGBA = 1,
            /*!< Opaque+alpha. EAC_A8 block followed by an ETC1 block. The
                 alpha channel will be opaque for textures without an alpha
                 channel. */

        // BC1-5, BC7 (desktop, some mobile devices)
        KTX_TTF_BC1_RGB = 2,
            /*!< Opaque only, no punchthrough alpha support yet.  Returns RGB
                 or alpha data, if KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS
                 flag is specified. */
        KTX_TTF_BC3_RGBA = 3,
            /*!< Opaque+alpha. BC4 block with alpha followed by a BC1 block. The
                 alpha channel will be opaque for textures without an alpha
                 channel. */
        KTX_TTF_BC4_R = 4,
            /*!< One BC4 block. R = opaque.g or alpha.g, if
                 KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS flag is
                 specified. */
        KTX_TTF_BC5_RG = 5,
            /*!< Two BC4 blocks, R=opaque.g and G=alpha.g The texture should
                 have an alpha channel (if not G will be all 255's. For tangent
                 space normal maps. */
        KTX_TTF_BC7_RGBA = 6,
            /*!< RGB or RGBA mode 5 for ETC1S, modes 1, 2, 3, 4, 5, 6, 7 for
                 UASTC. */

        // PVRTC1 4bpp (mobile, PowerVR devices)
        KTX_TTF_PVRTC1_4_RGB = 8,
            /*!< Opaque only. Returns RGB or alpha data, if
                 KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS flag is
                 specified. */
        KTX_TTF_PVRTC1_4_RGBA = 9,
            /*!< Opaque+alpha. Most useful for simple opacity maps. If the
                 texture doesn't have an alpha channel KTX_TTF_PVRTC1_4_RGB
                 will be used instead. Lowest quality of any supported
                 texture format. */

        // ASTC (mobile, Intel devices, hopefully all desktop GPU's one day)
        KTX_TTF_ASTC_4x4_RGBA = 10,
            /*!< Opaque+alpha, ASTC 4x4. The alpha channel will be opaque for
                 textures without an alpha channel.  The transcoder uses
                 RGB/RGBA/L/LA modes, void extent, and up to two ([0,47] and
                 [0,255]) endpoint precisions. */

        // ATC and FXT1 formats are not supported by KTX2 as there
        // are no equivalent VkFormats.

        KTX_TTF_PVRTC2_4_RGB = 18,
            /*!< Opaque-only. Almost BC1 quality, much faster to transcode
                 and supports arbitrary texture dimensions (unlike
                 PVRTC1 RGB). */
        KTX_TTF_PVRTC2_4_RGBA = 19,
            /*!< Opaque+alpha. Slower to transcode than cTFPVRTC2_4_RGB.
                 Premultiplied alpha is highly recommended, otherwise the
                 color channel can leak into the alpha channel on transparent
                 blocks. */

        KTX_TTF_ETC2_EAC_R11 = 20,
            /*!< R only (ETC2 EAC R11 unsigned). R = opaque.g or alpha.g, if
                 KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS flag is
                 specified. */
        KTX_TTF_ETC2_EAC_RG11 = 21,
            /*!< RG only (ETC2 EAC RG11 unsigned), R=opaque.g, G=alpha.g. The
                 texture should have an alpha channel (if not G will be all
                 255's. For tangent space normal maps. */

        // Uncompressed (raw pixel) formats
        KTX_TTF_RGBA32 = 13,
            /*!< 32bpp RGBA image stored in raster (not block) order in
                 memory, R is first byte, A is last byte. */
        KTX_TTF_RGB565 = 14,
            /*!< 16bpp RGB image stored in raster (not block) order in memory,
                 R at bit position 11. */
        KTX_TTF_BGR565 = 15,
            /*!< 16bpp RGB image stored in raster (not block) order in memory,
                 R at bit position 0. */
        KTX_TTF_RGBA4444 = 16,
            /*!< 16bpp RGBA image stored in raster (not block) order in memory,
                 R at bit position 12, A at bit position 0. */

        // Values for automatic selection of RGB or RGBA depending if alpha
        // present.
        KTX_TTF_ETC = 22,
            /*!< Automatically selects @c KTX_TTF_ETC1_RGB or
                 @c KTX_TTF_ETC2_RGBA according to presence of alpha. */
        KTX_TTF_BC1_OR_3 = 23,
            /*!< Automatically selects @c KTX_TTF_BC1_RGB or
                 @c KTX_TTF_BC3_RGBA according to presence of alpha. */

        KTX_TTF_NOSELECTION = 0x7fffffff,

        // Old enums for compatibility with code compiled against previous
        // versions of libktx.
        KTX_TF_ETC1 = KTX_TTF_ETC1_RGB,
            //!< @deprecated. Use #KTX_TTF_ETC1_RGB.
        KTX_TF_ETC2 = KTX_TTF_ETC,
            //!< @deprecated. Use #KTX_TTF_ETC.
        KTX_TF_BC1 = KTX_TTF_BC1_RGB,
            //!< @deprecated. Use #KTX_TTF_BC1_RGB.
        KTX_TF_BC3 = KTX_TTF_BC3_RGBA,
            //!< @deprecated. Use #KTX_TTF_BC3_RGBA.
        KTX_TF_BC4 = KTX_TTF_BC4_R,
            //!< @deprecated. Use #KTX_TTF_BC4_R.
        KTX_TF_BC5 = KTX_TTF_BC5_RG,
            //!< @deprecated. Use #KTX_TTF_BC5_RG.
        KTX_TTF_BC7_M6_RGB = KTX_TTF_BC7_RGBA,
            //!< @deprecated. Use #KTX_TTF_BC7_RGBA.
        KTX_TTF_BC7_M5_RGBA = KTX_TTF_BC7_RGBA,
            //!< @deprecated. Use #KTX_TTF_BC7_RGBA.
        KTX_TF_BC7_M6_OPAQUE_ONLY = KTX_TTF_BC7_RGBA,
            //!< @deprecated. Use #KTX_TTF_BC7_RGBA
        KTX_TF_PVRTC1_4_OPAQUE_ONLY = KTX_TTF_PVRTC1_4_RGB
            //!< @deprecated. Use #KTX_TTF_PVRTC1_4_RGB.
} ktx_transcode_fmt_e;

/**
 * @~English
 * @brief Flags guiding transcoding of Basis Universal compressed textures.
 */
typedef enum ktx_transcode_flag_bits_e {
    KTX_TF_PVRTC_DECODE_TO_NEXT_POW2 = 2,
        /*!< PVRTC1: decode non-pow2 ETC1S texture level to the next larger
             power of 2 (not implemented yet, but we're going to support it).
             Ignored if the slice's dimensions are already a power of 2.
         */
    KTX_TF_TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS = 4,
        /*!< When decoding to an opaque texture format, if the Basis data has
             alpha, decode the alpha slice instead of the color slice to the
             output texture format. Has no effect if there is no alpha data.
         */
    KTX_TF_HIGH_QUALITY = 32,
        /*!< Request higher quality transcode of UASTC to BC1, BC3, ETC2_EAC_R11 and
             ETC2_EAC_RG11. The flag is unused by other UASTC transcoders.
         */
} ktx_transcode_flag_bits_e;
typedef ktx_uint32_t ktx_transcode_flags;

KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_TranscodeBasis(ktxTexture2* This, ktx_transcode_fmt_e fmt,
                           ktx_transcode_flags transcodeFlags);

/*
 * Returns a string corresponding to a KTX error code.
 */
KTX_API const char* KTX_APIENTRY
ktxErrorString(KTX_error_code error);

/*
 * Returns a string corresponding to a supercompression scheme.
 */
KTX_API const char* KTX_APIENTRY
ktxSupercompressionSchemeString(ktxSupercmpScheme scheme);

/*
 * Returns a string corresponding to a transcode target format.
 */
KTX_API const char* KTX_APIENTRY
ktxTranscodeFormatString(ktx_transcode_fmt_e format);

KTX_API KTX_error_code KTX_APIENTRY ktxHashList_Create(ktxHashList** ppHl);
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_CreateCopy(ktxHashList** ppHl, ktxHashList orig);
KTX_API void KTX_APIENTRY ktxHashList_Construct(ktxHashList* pHl);
KTX_API void KTX_APIENTRY
ktxHashList_ConstructCopy(ktxHashList* pHl, ktxHashList orig);
KTX_API void KTX_APIENTRY ktxHashList_Destroy(ktxHashList* head);
KTX_API void KTX_APIENTRY ktxHashList_Destruct(ktxHashList* head);

/*
 * Adds a key-value pair to a hash list.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_AddKVPair(ktxHashList* pHead, const char* key,
                      unsigned int valueLen, const void* value);

/*
 * Deletes a ktxHashListEntry from a ktxHashList.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_DeleteEntry(ktxHashList* pHead,  ktxHashListEntry* pEntry);

/*
 * Finds the entry for a key in a ktxHashList and deletes it.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_DeleteKVPair(ktxHashList* pHead, const char* key);

/*
 * Looks up a key and returns the ktxHashListEntry.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_FindEntry(ktxHashList* pHead, const char* key,
                      ktxHashListEntry** ppEntry);

/*
 * Looks up a key and returns the value.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_FindValue(ktxHashList* pHead, const char* key,
                      unsigned int* pValueLen, void** pValue);

/*
 * Return the next entry in a ktxHashList.
 */
KTX_API ktxHashListEntry* KTX_APIENTRY
ktxHashList_Next(ktxHashListEntry* entry);

/*
 * Sorts a ktxHashList into order of the key codepoints.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_Sort(ktxHashList* pHead);

/*
 * Serializes a ktxHashList to a block of memory suitable for
 * writing to a KTX file.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_Serialize(ktxHashList* pHead,
                      unsigned int* kvdLen, unsigned char** kvd);

/*
 * Creates a hash table from the serialized data read from a
 * a KTX file.
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashList_Deserialize(ktxHashList* pHead, unsigned int kvdLen, void* kvd);

/*
 * Get the key from a ktxHashListEntry
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashListEntry_GetKey(ktxHashListEntry* This,
                        unsigned int* pKeyLen, char** ppKey);

/*
 * Get the value from a ktxHashListEntry
 */
KTX_API KTX_error_code KTX_APIENTRY
ktxHashListEntry_GetValue(ktxHashListEntry* This,
                          unsigned int* pValueLen, void** ppValue);

/*===========================================================*
 * Utilities for printing info about a KTX file.             *
 *===========================================================*/

KTX_API KTX_error_code KTX_APIENTRY ktxPrintInfoForStdioStream(FILE* stdioStream);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintInfoForNamedFile(const char* const filename);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintInfoForMemory(const ktx_uint8_t* bytes, ktx_size_t size);

/*===========================================================*
 * Utilities for printing info about a KTX2 file.            *
 *===========================================================*/

KTX_API KTX_error_code KTX_APIENTRY ktxPrintKTX2InfoTextForMemory(const ktx_uint8_t* bytes, ktx_size_t size);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintKTX2InfoTextForNamedFile(const char* const filename);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintKTX2InfoTextForStdioStream(FILE* stdioStream);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintKTX2InfoTextForStream(ktxStream* stream);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintKTX2InfoJSONForMemory(const ktx_uint8_t* bytes, ktx_size_t size, ktx_uint32_t base_indent, ktx_uint32_t indent_width, bool minified);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintKTX2InfoJSONForNamedFile(const char* const filename, ktx_uint32_t base_indent, ktx_uint32_t indent_width, bool minified);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintKTX2InfoJSONForStdioStream(FILE* stdioStream, ktx_uint32_t base_indent, ktx_uint32_t indent_width, bool minified);
KTX_API KTX_error_code KTX_APIENTRY ktxPrintKTX2InfoJSONForStream(ktxStream* stream, ktx_uint32_t base_indent, ktx_uint32_t indent_width, bool minified);

#ifdef __cplusplus
}
#endif

/*========================================================================*
 * For backward compatibilty with the V3 & early versions of the V4 APIs. *
 *========================================================================*/

/**
 * @deprecated Will be dropped before V4 release.
 */
#define ktx_texture_transcode_fmt_e ktx_transcode_fmt_e

/**
 * @deprecated Will be dropped before V4 release.
 */
#define ktx_texture_decode_flags ktx_transcode_flag_bits

/**
 * @deprecated Will be dropped before V4 release.
 */
#define ktxTexture_GetSize ktxTexture_GetDatasize

/**
@~English
@page libktx_history Revision History

No longer updated. Kept to preserve ancient history. For more recent history see the repo log at
https://github.com/KhronosGroup/KTX-Software. See also the Release Notes in the repo.

@section v8 Version 4.0
Added:
@li Support for KTX Version 2.
@li Support for encoding and transcoding Basis Universal images in KTX Version 2 files.
@li Function to print info about a KTX file.

@section v7 Version 3.0.1
Fixed:
@li GitHub issue #159: compile failure with recent Vulkan SDKs.
@li Incorrect mapping of GL DXT3 and DXT5 formats to Vulkan equivalents.
@li Incorrect BC4 blocksize.
@li Missing mapping of PVRTC formats from GL to Vulkan.
@li Incorrect block width and height calculations for sizes that are not
    a multiple of the block size.
@li Incorrect KTXorientation key in test images.

@section v6 Version 3.0
Added:
@li new ktxTexture object based API for reading KTX files without an OpenGL context.
@li Vulkan loader. @#include <ktxvulkan.h> to use it.

Changed:
@li ktx.h to not depend on KHR/khrplatform.h and GL{,ES*}/gl{corearb,}.h.
    Applications using OpenGL must now include these files themselves.
@li ktxLoadTexture[FMN], removing the hack of loading 1D textures as 2D textures
    when the OpenGL context does not support 1D textures.
    KTX_UNSUPPORTED_TEXTURE_TYPE is now returned.

@section v5 Version 2.0.2
Added:
@li Support for cubemap arrays.

Changed:
@li New build system

Fixed:
@li GitHub issue #40: failure to byte-swap key-value lengths.
@li GitHub issue #33: returning incorrect target when loading cubemaps.
@li GitHub PR #42: loading of texture arrays.
@li GitHub PR #41: compilation error when KTX_OPENGL_ES2=1 defined.
@li GitHub issue #39: stack-buffer-overflow in toktx
@li Don't use GL_EXTENSIONS on recent OpenGL versions.

@section v4 Version 2.0.1
Added:
@li CMake build files. Thanks to Pavel Rotjberg for the initial version.

Changed:
@li ktxWriteKTXF to check the validity of the type & format combinations
    passed to it.

Fixed:
@li Public Bugzilla <a href="http://www.khronos.org/bugzilla/show_bug.cgi?id=999">999</a>: 16-bit luminance texture cannot be written.
@li compile warnings from compilers stricter than MS Visual C++. Thanks to
    Pavel Rotjberg.

@section v3 Version 2.0
Added:
@li support for decoding ETC2 and EAC formats in the absence of a hardware
    decoder.
@li support for converting textures with legacy LUMINANCE, LUMINANCE_ALPHA,
    etc. formats to the equivalent R, RG, etc. format with an
    appropriate swizzle, when loading in OpenGL Core Profile contexts.
@li ktxErrorString function to return a string corresponding to an error code.
@li tests for ktxLoadTexture[FN] that run under OpenGL ES 3.0 and OpenGL 3.3.
    The latter includes an EGL on WGL wrapper that makes porting apps between
    OpenGL ES and OpenGL easier on Windows.
@li more texture formats to ktxLoadTexture[FN] and toktx tests.

Changed:
@li ktxLoadTexture[FMN] to discover the capabilities of the GL context at
    run time and load textures, or not, according to those capabilities.

Fixed:
@li failure of ktxWriteKTXF to pad image rows to 4 bytes as required by the KTX
    format.
@li ktxWriteKTXF exiting with KTX_FILE_WRITE_ERROR when attempting to write
    more than 1 byte of face-LOD padding.

Although there is only a very minor API change, the addition of ktxErrorString,
the functional changes are large enough to justify bumping the major revision
number.

@section v2 Version 1.0.1
Implemented ktxLoadTextureM.
Fixed the following:
@li Public Bugzilla <a href="http://www.khronos.org/bugzilla/show_bug.cgi?id=571">571</a>: crash when null passed for pIsMipmapped.
@li Public Bugzilla <a href="http://www.khronos.org/bugzilla/show_bug.cgi?id=572">572</a>: memory leak when unpacking ETC textures.
@li Public Bugzilla <a href="http://www.khronos.org/bugzilla/show_bug.cgi?id=573">573</a>: potential crash when unpacking ETC textures with unused padding pixels.
@li Public Bugzilla <a href="http://www.khronos.org/bugzilla/show_bug.cgi?id=576">576</a>: various small fixes.

Thanks to Krystian Bigaj for the ktxLoadTextureM implementation and these fixes.

@section v1 Version 1.0
Initial release.

*/

#endif /* KTX_H_A55A6F00956F42F3A137C11929827FE1 */
