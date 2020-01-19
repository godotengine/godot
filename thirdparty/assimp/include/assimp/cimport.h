/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

/** @file  cimport.h
 *  @brief Defines the C-API to the Open Asset Import Library.
 */
#pragma once
#ifndef AI_ASSIMP_H_INC
#define AI_ASSIMP_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/types.h>
#include <assimp/importerdesc.h>

#ifdef __cplusplus
extern "C" {
#endif

struct aiScene;  // aiScene.h
struct aiFileIO; // aiFileIO.h
typedef void (*aiLogStreamCallback)(const char* /* message */, char* /* user */);

// --------------------------------------------------------------------------------
/** C-API: Represents a log stream. A log stream receives all log messages and
 *  streams them _somewhere_.
 *  @see aiGetPredefinedLogStream
 *  @see aiAttachLogStream
 *  @see aiDetachLogStream */
// --------------------------------------------------------------------------------
struct aiLogStream
{
    /** callback to be called */
    aiLogStreamCallback callback;

    /** user data to be passed to the callback */
    char* user;
};


// --------------------------------------------------------------------------------
/** C-API: Represents an opaque set of settings to be used during importing.
 *  @see aiCreatePropertyStore
 *  @see aiReleasePropertyStore
 *  @see aiImportFileExWithProperties
 *  @see aiSetPropertyInteger
 *  @see aiSetPropertyFloat
 *  @see aiSetPropertyString
 *  @see aiSetPropertyMatrix
 */
// --------------------------------------------------------------------------------
struct aiPropertyStore { char sentinel; };

/** Our own C boolean type */
typedef int aiBool;

#define AI_FALSE 0
#define AI_TRUE 1

// --------------------------------------------------------------------------------
/** Reads the given file and returns its content.
 *
 * If the call succeeds, the imported data is returned in an aiScene structure.
 * The data is intended to be read-only, it stays property of the ASSIMP
 * library and will be stable until aiReleaseImport() is called. After you're
 * done with it, call aiReleaseImport() to free the resources associated with
 * this file. If the import fails, NULL is returned instead. Call
 * aiGetErrorString() to retrieve a human-readable error text.
 * @param pFile Path and filename of the file to be imported,
 *   expected to be a null-terminated c-string. NULL is not a valid value.
 * @param pFlags Optional post processing steps to be executed after
 *   a successful import. Provide a bitwise combination of the
 *   #aiPostProcessSteps flags.
 * @return Pointer to the imported data or NULL if the import failed.
 */
ASSIMP_API const C_STRUCT aiScene* aiImportFile(
    const char* pFile,
    unsigned int pFlags);

// --------------------------------------------------------------------------------
/** Reads the given file using user-defined I/O functions and returns
 *   its content.
 *
 * If the call succeeds, the imported data is returned in an aiScene structure.
 * The data is intended to be read-only, it stays property of the ASSIMP
 * library and will be stable until aiReleaseImport() is called. After you're
 * done with it, call aiReleaseImport() to free the resources associated with
 * this file. If the import fails, NULL is returned instead. Call
 * aiGetErrorString() to retrieve a human-readable error text.
 * @param pFile Path and filename of the file to be imported,
 *   expected to be a null-terminated c-string. NULL is not a valid value.
 * @param pFlags Optional post processing steps to be executed after
 *   a successful import. Provide a bitwise combination of the
 *   #aiPostProcessSteps flags.
 * @param pFS aiFileIO structure. Will be used to open the model file itself
 *   and any other files the loader needs to open.  Pass NULL to use the default
 *   implementation.
 * @return Pointer to the imported data or NULL if the import failed.
 * @note Include <aiFileIO.h> for the definition of #aiFileIO.
 */
ASSIMP_API const C_STRUCT aiScene* aiImportFileEx(
    const char* pFile,
    unsigned int pFlags,
    C_STRUCT aiFileIO* pFS);

// --------------------------------------------------------------------------------
/** Same as #aiImportFileEx, but adds an extra parameter containing importer settings.
 *
 * @param pFile Path and filename of the file to be imported,
 *   expected to be a null-terminated c-string. NULL is not a valid value.
 * @param pFlags Optional post processing steps to be executed after
 *   a successful import. Provide a bitwise combination of the
 *   #aiPostProcessSteps flags.
 * @param pFS aiFileIO structure. Will be used to open the model file itself
 *   and any other files the loader needs to open.  Pass NULL to use the default
 *   implementation.
 * @param pProps #aiPropertyStore instance containing import settings.
 * @return Pointer to the imported data or NULL if the import failed.
 * @note Include <aiFileIO.h> for the definition of #aiFileIO.
 * @see aiImportFileEx
 */
ASSIMP_API const C_STRUCT aiScene* aiImportFileExWithProperties(
    const char* pFile,
    unsigned int pFlags,
    C_STRUCT aiFileIO* pFS,
    const C_STRUCT aiPropertyStore* pProps);

// --------------------------------------------------------------------------------
/** Reads the given file from a given memory buffer,
 *
 * If the call succeeds, the contents of the file are returned as a pointer to an
 * aiScene object. The returned data is intended to be read-only, the importer keeps
 * ownership of the data and will destroy it upon destruction. If the import fails,
 * NULL is returned.
 * A human-readable error description can be retrieved by calling aiGetErrorString().
 * @param pBuffer Pointer to the file data
 * @param pLength Length of pBuffer, in bytes
 * @param pFlags Optional post processing steps to be executed after
 *   a successful import. Provide a bitwise combination of the
 *   #aiPostProcessSteps flags. If you wish to inspect the imported
 *   scene first in order to fine-tune your post-processing setup,
 *   consider to use #aiApplyPostProcessing().
 * @param pHint An additional hint to the library. If this is a non empty string,
 *   the library looks for a loader to support the file extension specified by pHint
 *   and passes the file to the first matching loader. If this loader is unable to
 *   completely the request, the library continues and tries to determine the file
 *   format on its own, a task that may or may not be successful.
 *   Check the return value, and you'll know ...
 * @return A pointer to the imported data, NULL if the import failed.
 *
 * @note This is a straightforward way to decode models from memory
 * buffers, but it doesn't handle model formats that spread their
 * data across multiple files or even directories. Examples include
 * OBJ or MD3, which outsource parts of their material info into
 * external scripts. If you need full functionality, provide
 * a custom IOSystem to make Assimp find these files and use
 * the regular aiImportFileEx()/aiImportFileExWithProperties() API.
 */
ASSIMP_API const C_STRUCT aiScene* aiImportFileFromMemory(
    const char* pBuffer,
    unsigned int pLength,
    unsigned int pFlags,
    const char* pHint);

// --------------------------------------------------------------------------------
/** Same as #aiImportFileFromMemory, but adds an extra parameter containing importer settings.
 *
 * @param pBuffer Pointer to the file data
 * @param pLength Length of pBuffer, in bytes
 * @param pFlags Optional post processing steps to be executed after
 *   a successful import. Provide a bitwise combination of the
 *   #aiPostProcessSteps flags. If you wish to inspect the imported
 *   scene first in order to fine-tune your post-processing setup,
 *   consider to use #aiApplyPostProcessing().
 * @param pHint An additional hint to the library. If this is a non empty string,
 *   the library looks for a loader to support the file extension specified by pHint
 *   and passes the file to the first matching loader. If this loader is unable to
 *   completely the request, the library continues and tries to determine the file
 *   format on its own, a task that may or may not be successful.
 *   Check the return value, and you'll know ...
 * @param pProps #aiPropertyStore instance containing import settings.
 * @return A pointer to the imported data, NULL if the import failed.
 *
 * @note This is a straightforward way to decode models from memory
 * buffers, but it doesn't handle model formats that spread their
 * data across multiple files or even directories. Examples include
 * OBJ or MD3, which outsource parts of their material info into
 * external scripts. If you need full functionality, provide
 * a custom IOSystem to make Assimp find these files and use
 * the regular aiImportFileEx()/aiImportFileExWithProperties() API.
 * @see aiImportFileFromMemory
 */
ASSIMP_API const C_STRUCT aiScene* aiImportFileFromMemoryWithProperties(
    const char* pBuffer,
    unsigned int pLength,
    unsigned int pFlags,
    const char* pHint,
    const C_STRUCT aiPropertyStore* pProps);

// --------------------------------------------------------------------------------
/** Apply post-processing to an already-imported scene.
 *
 * This is strictly equivalent to calling #aiImportFile()/#aiImportFileEx with the
 * same flags. However, you can use this separate function to inspect the imported
 * scene first to fine-tune your post-processing setup.
 * @param pScene Scene to work on.
 * @param pFlags Provide a bitwise combination of the #aiPostProcessSteps flags.
 * @return A pointer to the post-processed data. Post processing is done in-place,
 *   meaning this is still the same #aiScene which you passed for pScene. However,
 *   _if_ post-processing failed, the scene could now be NULL. That's quite a rare
 *   case, post processing steps are not really designed to 'fail'. To be exact,
 *   the #aiProcess_ValidateDataStructure flag is currently the only post processing step
 *   which can actually cause the scene to be reset to NULL.
 */
ASSIMP_API const C_STRUCT aiScene* aiApplyPostProcessing(
    const C_STRUCT aiScene* pScene,
    unsigned int pFlags);

// --------------------------------------------------------------------------------
/** Get one of the predefine log streams. This is the quick'n'easy solution to
 *  access Assimp's log system. Attaching a log stream can slightly reduce Assimp's
 *  overall import performance.
 *
 *  Usage is rather simple (this will stream the log to a file, named log.txt, and
 *  the stdout stream of the process:
 *  @code
 *    struct aiLogStream c;
 *    c = aiGetPredefinedLogStream(aiDefaultLogStream_FILE,"log.txt");
 *    aiAttachLogStream(&c);
 *    c = aiGetPredefinedLogStream(aiDefaultLogStream_STDOUT,NULL);
 *    aiAttachLogStream(&c);
 *  @endcode
 *
 *  @param pStreams One of the #aiDefaultLogStream enumerated values.
 *  @param file Solely for the #aiDefaultLogStream_FILE flag: specifies the file to write to.
 *    Pass NULL for all other flags.
 *  @return The log stream. callback is set to NULL if something went wrong.
 */
ASSIMP_API C_STRUCT aiLogStream aiGetPredefinedLogStream(
    C_ENUM aiDefaultLogStream pStreams,
    const char* file);

// --------------------------------------------------------------------------------
/** Attach a custom log stream to the libraries' logging system.
 *
 *  Attaching a log stream can slightly reduce Assimp's overall import
 *  performance. Multiple log-streams can be attached.
 *  @param stream Describes the new log stream.
 *  @note To ensure proper destruction of the logging system, you need to manually
 *    call aiDetachLogStream() on every single log stream you attach.
 *    Alternatively (for the lazy folks) #aiDetachAllLogStreams is provided.
 */
ASSIMP_API void aiAttachLogStream(
    const C_STRUCT aiLogStream* stream);

// --------------------------------------------------------------------------------
/** Enable verbose logging. Verbose logging includes debug-related stuff and
 *  detailed import statistics. This can have severe impact on import performance
 *  and memory consumption. However, it might be useful to find out why a file
 *  didn't read correctly.
 *  @param d AI_TRUE or AI_FALSE, your decision.
 */
ASSIMP_API void aiEnableVerboseLogging(aiBool d);

// --------------------------------------------------------------------------------
/** Detach a custom log stream from the libraries' logging system.
 *
 *  This is the counterpart of #aiAttachLogStream. If you attached a stream,
 *  don't forget to detach it again.
 *  @param stream The log stream to be detached.
 *  @return AI_SUCCESS if the log stream has been detached successfully.
 *  @see aiDetachAllLogStreams
 */
ASSIMP_API C_ENUM aiReturn aiDetachLogStream(
    const C_STRUCT aiLogStream* stream);

// --------------------------------------------------------------------------------
/** Detach all active log streams from the libraries' logging system.
 *  This ensures that the logging system is terminated properly and all
 *  resources allocated by it are actually freed. If you attached a stream,
 *  don't forget to detach it again.
 *  @see aiAttachLogStream
 *  @see aiDetachLogStream
 */
ASSIMP_API void aiDetachAllLogStreams(void);

// --------------------------------------------------------------------------------
/** Releases all resources associated with the given import process.
 *
 * Call this function after you're done with the imported data.
 * @param pScene The imported data to release. NULL is a valid value.
 */
ASSIMP_API void aiReleaseImport(
    const C_STRUCT aiScene* pScene);

// --------------------------------------------------------------------------------
/** Returns the error text of the last failed import process.
 *
 * @return A textual description of the error that occurred at the last
 * import process. NULL if there was no error. There can't be an error if you
 * got a non-NULL #aiScene from #aiImportFile/#aiImportFileEx/#aiApplyPostProcessing.
 */
ASSIMP_API const char* aiGetErrorString(void);

// --------------------------------------------------------------------------------
/** Returns whether a given file extension is supported by ASSIMP
 *
 * @param szExtension Extension for which the function queries support for.
 * Must include a leading dot '.'. Example: ".3ds", ".md3"
 * @return AI_TRUE if the file extension is supported.
 */
ASSIMP_API aiBool aiIsExtensionSupported(
    const char* szExtension);

// --------------------------------------------------------------------------------
/** Get a list of all file extensions supported by ASSIMP.
 *
 * If a file extension is contained in the list this does, of course, not
 * mean that ASSIMP is able to load all files with this extension.
 * @param szOut String to receive the extension list.
 * Format of the list: "*.3ds;*.obj;*.dae". NULL is not a valid parameter.
 */
ASSIMP_API void aiGetExtensionList(
    C_STRUCT aiString* szOut);

// --------------------------------------------------------------------------------
/** Get the approximated storage required by an imported asset
 * @param pIn Input asset.
 * @param in Data structure to be filled.
 */
ASSIMP_API void aiGetMemoryRequirements(
    const C_STRUCT aiScene* pIn,
    C_STRUCT aiMemoryInfo* in);



// --------------------------------------------------------------------------------
/** Create an empty property store. Property stores are used to collect import
 *  settings.
 * @return New property store. Property stores need to be manually destroyed using
 *   the #aiReleasePropertyStore API function.
 */
ASSIMP_API C_STRUCT aiPropertyStore* aiCreatePropertyStore(void);

// --------------------------------------------------------------------------------
/** Delete a property store.
 * @param p Property store to be deleted.
 */
ASSIMP_API void aiReleasePropertyStore(C_STRUCT aiPropertyStore* p);

// --------------------------------------------------------------------------------
/** Set an integer property.
 *
 *  This is the C-version of #Assimp::Importer::SetPropertyInteger(). In the C
 *  interface, properties are always shared by all imports. It is not possible to
 *  specify them per import.
 *
 * @param store Store to modify. Use #aiCreatePropertyStore to obtain a store.
 * @param szName Name of the configuration property to be set. All supported
 *   public properties are defined in the config.h header file (AI_CONFIG_XXX).
 * @param value New value for the property
 */
ASSIMP_API void aiSetImportPropertyInteger(
    C_STRUCT aiPropertyStore* store,
    const char* szName,
    int value);

// --------------------------------------------------------------------------------
/** Set a floating-point property.
 *
 *  This is the C-version of #Assimp::Importer::SetPropertyFloat(). In the C
 *  interface, properties are always shared by all imports. It is not possible to
 *  specify them per import.
 *
 * @param store Store to modify. Use #aiCreatePropertyStore to obtain a store.
 * @param szName Name of the configuration property to be set. All supported
 *   public properties are defined in the config.h header file (AI_CONFIG_XXX).
 * @param value New value for the property
 */
ASSIMP_API void aiSetImportPropertyFloat(
    C_STRUCT aiPropertyStore* store,
    const char* szName,
    ai_real value);

// --------------------------------------------------------------------------------
/** Set a string property.
 *
 *  This is the C-version of #Assimp::Importer::SetPropertyString(). In the C
 *  interface, properties are always shared by all imports. It is not possible to
 *  specify them per import.
 *
 * @param store Store to modify. Use #aiCreatePropertyStore to obtain a store.
 * @param szName Name of the configuration property to be set. All supported
 *   public properties are defined in the config.h header file (AI_CONFIG_XXX).
 * @param st New value for the property
 */
ASSIMP_API void aiSetImportPropertyString(
    C_STRUCT aiPropertyStore* store,
    const char* szName,
    const C_STRUCT aiString* st);

// --------------------------------------------------------------------------------
/** Set a matrix property.
 *
 *  This is the C-version of #Assimp::Importer::SetPropertyMatrix(). In the C
 *  interface, properties are always shared by all imports. It is not possible to
 *  specify them per import.
 *
 * @param store Store to modify. Use #aiCreatePropertyStore to obtain a store.
 * @param szName Name of the configuration property to be set. All supported
 *   public properties are defined in the config.h header file (AI_CONFIG_XXX).
 * @param mat New value for the property
 */
ASSIMP_API void aiSetImportPropertyMatrix(
    C_STRUCT aiPropertyStore* store,
    const char* szName,
    const C_STRUCT aiMatrix4x4* mat);

// --------------------------------------------------------------------------------
/** Construct a quaternion from a 3x3 rotation matrix.
 *  @param quat Receives the output quaternion.
 *  @param mat Matrix to 'quaternionize'.
 *  @see aiQuaternion(const aiMatrix3x3& pRotMatrix)
 */
ASSIMP_API void aiCreateQuaternionFromMatrix(
    C_STRUCT aiQuaternion* quat,
    const C_STRUCT aiMatrix3x3* mat);

// --------------------------------------------------------------------------------
/** Decompose a transformation matrix into its rotational, translational and
 *  scaling components.
 *
 * @param mat Matrix to decompose
 * @param scaling Receives the scaling component
 * @param rotation Receives the rotational component
 * @param position Receives the translational component.
 * @see aiMatrix4x4::Decompose (aiVector3D&, aiQuaternion&, aiVector3D&) const;
 */
ASSIMP_API void aiDecomposeMatrix(
    const C_STRUCT aiMatrix4x4* mat,
    C_STRUCT aiVector3D* scaling,
    C_STRUCT aiQuaternion* rotation,
    C_STRUCT aiVector3D* position);

// --------------------------------------------------------------------------------
/** Transpose a 4x4 matrix.
 *  @param mat Pointer to the matrix to be transposed
 */
ASSIMP_API void aiTransposeMatrix4(
    C_STRUCT aiMatrix4x4* mat);

// --------------------------------------------------------------------------------
/** Transpose a 3x3 matrix.
 *  @param mat Pointer to the matrix to be transposed
 */
ASSIMP_API void aiTransposeMatrix3(
    C_STRUCT aiMatrix3x3* mat);

// --------------------------------------------------------------------------------
/** Transform a vector by a 3x3 matrix
 *  @param vec Vector to be transformed.
 *  @param mat Matrix to transform the vector with.
 */
ASSIMP_API void aiTransformVecByMatrix3(
    C_STRUCT aiVector3D* vec,
    const C_STRUCT aiMatrix3x3* mat);

// --------------------------------------------------------------------------------
/** Transform a vector by a 4x4 matrix
 *  @param vec Vector to be transformed.
 *  @param mat Matrix to transform the vector with.
 */
ASSIMP_API void aiTransformVecByMatrix4(
    C_STRUCT aiVector3D* vec,
    const C_STRUCT aiMatrix4x4* mat);

// --------------------------------------------------------------------------------
/** Multiply two 4x4 matrices.
 *  @param dst First factor, receives result.
 *  @param src Matrix to be multiplied with 'dst'.
 */
ASSIMP_API void aiMultiplyMatrix4(
    C_STRUCT aiMatrix4x4* dst,
    const C_STRUCT aiMatrix4x4* src);

// --------------------------------------------------------------------------------
/** Multiply two 3x3 matrices.
 *  @param dst First factor, receives result.
 *  @param src Matrix to be multiplied with 'dst'.
 */
ASSIMP_API void aiMultiplyMatrix3(
    C_STRUCT aiMatrix3x3* dst,
    const C_STRUCT aiMatrix3x3* src);

// --------------------------------------------------------------------------------
/** Get a 3x3 identity matrix.
 *  @param mat Matrix to receive its personal identity
 */
ASSIMP_API void aiIdentityMatrix3(
    C_STRUCT aiMatrix3x3* mat);

// --------------------------------------------------------------------------------
/** Get a 4x4 identity matrix.
 *  @param mat Matrix to receive its personal identity
 */
ASSIMP_API void aiIdentityMatrix4(
    C_STRUCT aiMatrix4x4* mat);

// --------------------------------------------------------------------------------
/** Returns the number of import file formats available in the current Assimp build.
 * Use aiGetImportFormatDescription() to retrieve infos of a specific import format.
 */
ASSIMP_API size_t aiGetImportFormatCount(void);

// --------------------------------------------------------------------------------
/** Returns a description of the nth import file format. Use #aiGetImportFormatCount()
 * to learn how many import formats are supported.
 * @param pIndex Index of the import format to retrieve information for. Valid range is
 *    0 to #aiGetImportFormatCount()
 * @return A description of that specific import format. NULL if pIndex is out of range.
 */
ASSIMP_API const C_STRUCT aiImporterDesc* aiGetImportFormatDescription( size_t pIndex);
#ifdef __cplusplus
}
#endif

#endif // AI_ASSIMP_H_INC
