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

/** @file  Importer.hpp
 *  @brief Defines the C++-API to the Open Asset Import Library.
 */
#pragma once
#ifndef AI_ASSIMP_HPP_INC
#define AI_ASSIMP_HPP_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#ifndef __cplusplus
#   error This header requires C++ to be used. Use assimp.h for plain C.
#endif // __cplusplus

// Public ASSIMP data structures
#include <assimp/types.h>

namespace Assimp    {
    // =======================================================================
    // Public interface to Assimp
    class Importer;
    class IOStream;
    class IOSystem;
    class ProgressHandler;

    // =======================================================================
    // Plugin development
    //
    // Include the following headers for the declarations:
    // BaseImporter.h
    // BaseProcess.h
    class BaseImporter;
    class BaseProcess;
    class SharedPostProcessInfo;
    class BatchLoader;

    // =======================================================================
    // Holy stuff, only for members of the high council of the Jedi.
    class ImporterPimpl;
} //! namespace Assimp

#define AI_PROPERTY_WAS_NOT_EXISTING 0xffffffff

struct aiScene;

// importerdesc.h
struct aiImporterDesc;

/** @namespace Assimp Assimp's CPP-API and all internal APIs */
namespace Assimp    {

// ----------------------------------------------------------------------------------
/** CPP-API: The Importer class forms an C++ interface to the functionality of the
*   Open Asset Import Library.
*
* Create an object of this class and call ReadFile() to import a file.
* If the import succeeds, the function returns a pointer to the imported data.
* The data remains property of the object, it is intended to be accessed
* read-only. The imported data will be destroyed along with the Importer
* object. If the import fails, ReadFile() returns a NULL pointer. In this
* case you can retrieve a human-readable error description be calling
* GetErrorString(). You can call ReadFile() multiple times with a single Importer
* instance. Actually, constructing Importer objects involves quite many
* allocations and may take some time, so it's better to reuse them as often as
* possible.
*
* If you need the Importer to do custom file handling to access the files,
* implement IOSystem and IOStream and supply an instance of your custom
* IOSystem implementation by calling SetIOHandler() before calling ReadFile().
* If you do not assign a custion IO handler, a default handler using the
* standard C++ IO logic will be used.
*
* @note One Importer instance is not thread-safe. If you use multiple
* threads for loading, each thread should maintain its own Importer instance.
*/
class ASSIMP_API Importer   {
public:
    /**
     *  @brief The upper limit for hints.
     */
    static const unsigned int MaxLenHint = 200;

public:

    // -------------------------------------------------------------------
    /** Constructor. Creates an empty importer object.
     *
     * Call ReadFile() to start the import process. The configuration
     * property table is initially empty.
     */
    Importer();

    // -------------------------------------------------------------------
    /** Copy constructor.
     *
     * This copies the configuration properties of another Importer.
     * If this Importer owns a scene it won't be copied.
     * Call ReadFile() to start the import process.
     */
    Importer(const Importer& other)=delete;

    // -------------------------------------------------------------------
    /** Assignment operator has been deleted
     */
    Importer &operator=(const Importer &) = delete;

    // -------------------------------------------------------------------
    /** Destructor. The object kept ownership of the imported data,
     * which now will be destroyed along with the object.
     */
    ~Importer();


    // -------------------------------------------------------------------
    /** Registers a new loader.
     *
     * @param pImp Importer to be added. The Importer instance takes
     *   ownership of the pointer, so it will be automatically deleted
     *   with the Importer instance.
     * @return AI_SUCCESS if the loader has been added. The registration
     *   fails if there is already a loader for a specific file extension.
     */
    aiReturn RegisterLoader(BaseImporter* pImp);

    // -------------------------------------------------------------------
    /** Unregisters a loader.
     *
     * @param pImp Importer to be unregistered.
     * @return AI_SUCCESS if the loader has been removed. The function
     *   fails if the loader is currently in use (this could happen
     *   if the #Importer instance is used by more than one thread) or
     *   if it has not yet been registered.
     */
    aiReturn UnregisterLoader(BaseImporter* pImp);

    // -------------------------------------------------------------------
    /** Registers a new post-process step.
     *
     * At the moment, there's a small limitation: new post processing
     * steps are added to end of the list, or in other words, executed
     * last, after all built-in steps.
     * @param pImp Post-process step to be added. The Importer instance
     *   takes ownership of the pointer, so it will be automatically
     *   deleted with the Importer instance.
     * @return AI_SUCCESS if the step has been added correctly.
     */
    aiReturn RegisterPPStep(BaseProcess* pImp);

    // -------------------------------------------------------------------
    /** Unregisters a post-process step.
     *
     * @param pImp Step to be unregistered.
     * @return AI_SUCCESS if the step has been removed. The function
     *   fails if the step is currently in use (this could happen
     *   if the #Importer instance is used by more than one thread) or
     *   if it has not yet been registered.
     */
    aiReturn UnregisterPPStep(BaseProcess* pImp);

    // -------------------------------------------------------------------
    /** Set an integer configuration property.
     * @param szName Name of the property. All supported properties
     *   are defined in the aiConfig.g header (all constants share the
     *   prefix AI_CONFIG_XXX and are simple strings).
     * @param iValue New value of the property
     * @return true if the property was set before. The new value replaces
     *   the previous value in this case.
     * @note Property of different types (float, int, string ..) are kept
     *   on different stacks, so calling SetPropertyInteger() for a
     *   floating-point property has no effect - the loader will call
     *   GetPropertyFloat() to read the property, but it won't be there.
     */
    bool SetPropertyInteger(const char* szName, int iValue);

    // -------------------------------------------------------------------
    /** Set a boolean configuration property. Boolean properties
     *  are stored on the integer stack internally so it's possible
     *  to set them via #SetPropertyBool and query them with
     *  #GetPropertyBool and vice versa.
     * @see SetPropertyInteger()
     */
    bool SetPropertyBool(const char* szName, bool value)    {
        return SetPropertyInteger(szName,value);
    }

    // -------------------------------------------------------------------
    /** Set a floating-point configuration property.
     * @see SetPropertyInteger()
     */
    bool SetPropertyFloat(const char* szName, ai_real fValue);

    // -------------------------------------------------------------------
    /** Set a string configuration property.
     * @see SetPropertyInteger()
     */
    bool SetPropertyString(const char* szName, const std::string& sValue);

    // -------------------------------------------------------------------
    /** Set a matrix configuration property.
     * @see SetPropertyInteger()
     */
    bool SetPropertyMatrix(const char* szName, const aiMatrix4x4& sValue);

    // -------------------------------------------------------------------
    /** Get a configuration property.
     * @param szName Name of the property. All supported properties
     *   are defined in the aiConfig.g header (all constants share the
     *   prefix AI_CONFIG_XXX).
     * @param iErrorReturn Value that is returned if the property
     *   is not found.
     * @return Current value of the property
     * @note Property of different types (float, int, string ..) are kept
     *   on different lists, so calling SetPropertyInteger() for a
     *   floating-point property has no effect - the loader will call
     *   GetPropertyFloat() to read the property, but it won't be there.
     */
    int GetPropertyInteger(const char* szName,
        int iErrorReturn = 0xffffffff) const;

    // -------------------------------------------------------------------
    /** Get a boolean configuration property. Boolean properties
     *  are stored on the integer stack internally so it's possible
     *  to set them via #SetPropertyBool and query them with
     *  #GetPropertyBool and vice versa.
     * @see GetPropertyInteger()
     */
    bool GetPropertyBool(const char* szName, bool bErrorReturn = false) const {
        return GetPropertyInteger(szName,bErrorReturn)!=0;
    }

    // -------------------------------------------------------------------
    /** Get a floating-point configuration property
     * @see GetPropertyInteger()
     */
    ai_real GetPropertyFloat(const char* szName,
        ai_real fErrorReturn = 10e10) const;

    // -------------------------------------------------------------------
    /** Get a string configuration property
     *
     *  The return value remains valid until the property is modified.
     * @see GetPropertyInteger()
     */
    const std::string GetPropertyString(const char* szName,
        const std::string& sErrorReturn = "") const;

    // -------------------------------------------------------------------
    /** Get a matrix configuration property
     *
     *  The return value remains valid until the property is modified.
     * @see GetPropertyInteger()
     */
    const aiMatrix4x4 GetPropertyMatrix(const char* szName,
        const aiMatrix4x4& sErrorReturn = aiMatrix4x4()) const;

    // -------------------------------------------------------------------
    /** Supplies a custom IO handler to the importer to use to open and
     * access files. If you need the importer to use custom IO logic to
     * access the files, you need to provide a custom implementation of
     * IOSystem and IOFile to the importer. Then create an instance of
     * your custom IOSystem implementation and supply it by this function.
     *
     * The Importer takes ownership of the object and will destroy it
     * afterwards. The previously assigned handler will be deleted.
     * Pass NULL to take again ownership of your IOSystem and reset Assimp
     * to use its default implementation.
     *
     * @param pIOHandler The IO handler to be used in all file accesses
     *   of the Importer.
     */
    void SetIOHandler( IOSystem* pIOHandler);

    // -------------------------------------------------------------------
    /** Retrieves the IO handler that is currently set.
     * You can use #IsDefaultIOHandler() to check whether the returned
     * interface is the default IO handler provided by ASSIMP. The default
     * handler is active as long the application doesn't supply its own
     * custom IO handler via #SetIOHandler().
     * @return A valid IOSystem interface, never NULL.
     */
    IOSystem* GetIOHandler() const;

    // -------------------------------------------------------------------
    /** Checks whether a default IO handler is active
     * A default handler is active as long the application doesn't
     * supply its own custom IO handler via #SetIOHandler().
     * @return true by default
     */
    bool IsDefaultIOHandler() const;

    // -------------------------------------------------------------------
    /** Supplies a custom progress handler to the importer. This
     *  interface exposes an #Update() callback, which is called
     *  more or less periodically (please don't sue us if it
     *  isn't as periodically as you'd like it to have ...).
     *  This can be used to implement progress bars and loading
     *  timeouts.
     *  @param pHandler Progress callback interface. Pass NULL to
     *    disable progress reporting.
     *  @note Progress handlers can be used to abort the loading
     *    at almost any time.*/
    void SetProgressHandler ( ProgressHandler* pHandler );

    // -------------------------------------------------------------------
    /** Retrieves the progress handler that is currently set.
     * You can use #IsDefaultProgressHandler() to check whether the returned
     * interface is the default handler provided by ASSIMP. The default
     * handler is active as long the application doesn't supply its own
     * custom handler via #SetProgressHandler().
     * @return A valid ProgressHandler interface, never NULL.
     */
    ProgressHandler* GetProgressHandler() const;

    // -------------------------------------------------------------------
    /** Checks whether a default progress handler is active
     * A default handler is active as long the application doesn't
     * supply its own custom progress handler via #SetProgressHandler().
     * @return true by default
     */
    bool IsDefaultProgressHandler() const;

    // -------------------------------------------------------------------
    /** @brief Check whether a given set of post-processing flags
     *  is supported.
     *
     *  Some flags are mutually exclusive, others are probably
     *  not available because your excluded them from your
     *  Assimp builds. Calling this function is recommended if
     *  you're unsure.
     *
     *  @param pFlags Bitwise combination of the aiPostProcess flags.
     *  @return true if this flag combination is fine.
     */
    bool ValidateFlags(unsigned int pFlags) const;

    // -------------------------------------------------------------------
    /** Reads the given file and returns its contents if successful.
     *
     * If the call succeeds, the contents of the file are returned as a
     * pointer to an aiScene object. The returned data is intended to be
     * read-only, the importer object keeps ownership of the data and will
     * destroy it upon destruction. If the import fails, NULL is returned.
     * A human-readable error description can be retrieved by calling
     * GetErrorString(). The previous scene will be deleted during this call.
     * @param pFile Path and filename to the file to be imported.
     * @param pFlags Optional post processing steps to be executed after
     *   a successful import. Provide a bitwise combination of the
     *   #aiPostProcessSteps flags. If you wish to inspect the imported
     *   scene first in order to fine-tune your post-processing setup,
     *   consider to use #ApplyPostProcessing().
     * @return A pointer to the imported data, NULL if the import failed.
     *   The pointer to the scene remains in possession of the Importer
     *   instance. Use GetOrphanedScene() to take ownership of it.
     *
     * @note Assimp is able to determine the file format of a file
     * automatically.
     */
    const aiScene* ReadFile(
        const char* pFile,
        unsigned int pFlags);

    // -------------------------------------------------------------------
    /** Reads the given file from a memory buffer and returns its
     *  contents if successful.
     *
     * If the call succeeds, the contents of the file are returned as a
     * pointer to an aiScene object. The returned data is intended to be
     * read-only, the importer object keeps ownership of the data and will
     * destroy it upon destruction. If the import fails, NULL is returned.
     * A human-readable error description can be retrieved by calling
     * GetErrorString(). The previous scene will be deleted during this call.
     * Calling this method doesn't affect the active IOSystem.
     * @param pBuffer Pointer to the file data
     * @param pLength Length of pBuffer, in bytes
     * @param pFlags Optional post processing steps to be executed after
     *   a successful import. Provide a bitwise combination of the
     *   #aiPostProcessSteps flags. If you wish to inspect the imported
     *   scene first in order to fine-tune your post-processing setup,
     *   consider to use #ApplyPostProcessing().
     * @param pHint An additional hint to the library. If this is a non
     *   empty string, the library looks for a loader to support
     *   the file extension specified by pHint and passes the file to
     *   the first matching loader. If this loader is unable to completely
     *   the request, the library continues and tries to determine the
     *   file format on its own, a task that may or may not be successful.
     *   Check the return value, and you'll know ...
     * @return A pointer to the imported data, NULL if the import failed.
     *   The pointer to the scene remains in possession of the Importer
     *   instance. Use GetOrphanedScene() to take ownership of it.
     *
     * @note This is a straightforward way to decode models from memory
     * buffers, but it doesn't handle model formats that spread their
     * data across multiple files or even directories. Examples include
     * OBJ or MD3, which outsource parts of their material info into
     * external scripts. If you need full functionality, provide
     * a custom IOSystem to make Assimp find these files and use
     * the regular ReadFile() API.
     */
    const aiScene* ReadFileFromMemory(
        const void* pBuffer,
        size_t pLength,
        unsigned int pFlags,
        const char* pHint = "");

    // -------------------------------------------------------------------
    /** Apply post-processing to an already-imported scene.
     *
     *  This is strictly equivalent to calling #ReadFile() with the same
     *  flags. However, you can use this separate function to inspect
     *  the imported scene first to fine-tune your post-processing setup.
     *  @param pFlags Provide a bitwise combination of the
     *   #aiPostProcessSteps flags.
     *  @return A pointer to the post-processed data. This is still the
     *   same as the pointer returned by #ReadFile(). However, if
     *   post-processing fails, the scene could now be NULL.
     *   That's quite a rare case, post processing steps are not really
     *   designed to 'fail'. To be exact, the #aiProcess_ValidateDS
     *   flag is currently the only post processing step which can actually
     *   cause the scene to be reset to NULL.
     *
     *  @note The method does nothing if no scene is currently bound
     *    to the #Importer instance.  */
    const aiScene* ApplyPostProcessing(unsigned int pFlags);

    const aiScene* ApplyCustomizedPostProcessing( BaseProcess *rootProcess, bool requestValidation );

    // -------------------------------------------------------------------
    /** @brief Reads the given file and returns its contents if successful.
     *
     * This function is provided for backward compatibility.
     * See the const char* version for detailed docs.
     * @see ReadFile(const char*, pFlags)  */
    const aiScene* ReadFile(
        const std::string& pFile,
        unsigned int pFlags);

    // -------------------------------------------------------------------
    /** Frees the current scene.
     *
     *  The function does nothing if no scene has previously been
     *  read via ReadFile(). FreeScene() is called automatically by the
     *  destructor and ReadFile() itself.  */
    void FreeScene( );

    // -------------------------------------------------------------------
    /** Returns an error description of an error that occurred in ReadFile().
     *
     * Returns an empty string if no error occurred.
     * @return A description of the last error, an empty string if no
     *   error occurred. The string is never NULL.
     *
     * @note The returned function remains valid until one of the
     * following methods is called: #ReadFile(), #FreeScene(). */
    const char* GetErrorString() const;

    // -------------------------------------------------------------------
    /** Returns the scene loaded by the last successful call to ReadFile()
     *
     * @return Current scene or NULL if there is currently no scene loaded */
    const aiScene* GetScene() const;

    // -------------------------------------------------------------------
    /** Returns the scene loaded by the last successful call to ReadFile()
     *  and releases the scene from the ownership of the Importer
     *  instance. The application is now responsible for deleting the
     *  scene. Any further calls to GetScene() or GetOrphanedScene()
     *  will return NULL - until a new scene has been loaded via ReadFile().
     *
     * @return Current scene or NULL if there is currently no scene loaded
     * @note Use this method with maximal caution, and only if you have to.
     *   By design, aiScene's are exclusively maintained, allocated and
     *   deallocated by Assimp and no one else. The reasoning behind this
     *   is the golden rule that deallocations should always be done
     *   by the module that did the original allocation because heaps
     *   are not necessarily shared. GetOrphanedScene() enforces you
     *   to delete the returned scene by yourself, but this will only
     *   be fine if and only if you're using the same heap as assimp.
     *   On Windows, it's typically fine provided everything is linked
     *   against the multithreaded-dll version of the runtime library.
     *   It will work as well for static linkage with Assimp.*/
    aiScene* GetOrphanedScene();

    // -------------------------------------------------------------------
    /** Returns whether a given file extension is supported by ASSIMP.
     *
     * @param szExtension Extension to be checked.
     *   Must include a trailing dot '.'. Example: ".3ds", ".md3".
     *   Cases-insensitive.
     * @return true if the extension is supported, false otherwise */
    bool IsExtensionSupported(const char* szExtension) const;

    // -------------------------------------------------------------------
    /** @brief Returns whether a given file extension is supported by ASSIMP.
     *
     * This function is provided for backward compatibility.
     * See the const char* version for detailed and up-to-date docs.
     * @see IsExtensionSupported(const char*) */
    inline bool IsExtensionSupported(const std::string& szExtension) const;

    // -------------------------------------------------------------------
    /** Get a full list of all file extensions supported by ASSIMP.
     *
     * If a file extension is contained in the list this does of course not
     * mean that ASSIMP is able to load all files with this extension ---
     * it simply means there is an importer loaded which claims to handle
     * files with this file extension.
     * @param szOut String to receive the extension list.
     *   Format of the list: "*.3ds;*.obj;*.dae". This is useful for
     *   use with the WinAPI call GetOpenFileName(Ex). */
    void GetExtensionList(aiString& szOut) const;

    // -------------------------------------------------------------------
    /** @brief Get a full list of all file extensions supported by ASSIMP.
     *
     * This function is provided for backward compatibility.
     * See the aiString version for detailed and up-to-date docs.
     * @see GetExtensionList(aiString&)*/
    inline void GetExtensionList(std::string& szOut) const;

    // -------------------------------------------------------------------
    /** Get the number of importers currently registered with Assimp. */
    size_t GetImporterCount() const;

    // -------------------------------------------------------------------
    /** Get meta data for the importer corresponding to a specific index..
    *
    *  For the declaration of #aiImporterDesc, include <assimp/importerdesc.h>.
    *  @param index Index to query, must be within [0,GetImporterCount())
    *  @return Importer meta data structure, NULL if the index does not
    *     exist or if the importer doesn't offer meta information (
    *     importers may do this at the cost of being hated by their peers).*/
    const aiImporterDesc* GetImporterInfo(size_t index) const;

    // -------------------------------------------------------------------
    /** Find the importer corresponding to a specific index.
    *
    *  @param index Index to query, must be within [0,GetImporterCount())
    *  @return Importer instance. NULL if the index does not
    *     exist. */
    BaseImporter* GetImporter(size_t index) const;

    // -------------------------------------------------------------------
    /** Find the importer corresponding to a specific file extension.
    *
    *  This is quite similar to #IsExtensionSupported except a
    *  BaseImporter instance is returned.
    *  @param szExtension Extension to check for. The following formats
    *    are recognized (BAH being the file extension): "BAH" (comparison
    *    is case-insensitive), ".bah", "*.bah" (wild card and dot
    *    characters at the beginning of the extension are skipped).
    *  @return NULL if no importer is found*/
    BaseImporter* GetImporter (const char* szExtension) const;

    // -------------------------------------------------------------------
    /** Find the importer index corresponding to a specific file extension.
    *
    *  @param szExtension Extension to check for. The following formats
    *    are recognized (BAH being the file extension): "BAH" (comparison
    *    is case-insensitive), ".bah", "*.bah" (wild card and dot
    *    characters at the beginning of the extension are skipped).
    *  @return (size_t)-1 if no importer is found */
    size_t GetImporterIndex (const char* szExtension) const;

    // -------------------------------------------------------------------
    /** Returns the storage allocated by ASSIMP to hold the scene data
     * in memory.
     *
     * This refers to the currently loaded file, see #ReadFile().
     * @param in Data structure to be filled.
     * @note The returned memory statistics refer to the actual
     *   size of the use data of the aiScene. Heap-related overhead
     *   is (naturally) not included.*/
    void GetMemoryRequirements(aiMemoryInfo& in) const;

    // -------------------------------------------------------------------
    /** Enables "extra verbose" mode.
     *
     * 'Extra verbose' means the data structure is validated after *every*
     * single post processing step to make sure everyone modifies the data
     * structure in a well-defined manner. This is a debug feature and not
     * intended for use in production environments. */
    void SetExtraVerbose(bool bDo);

    // -------------------------------------------------------------------
    /** Private, do not use. */
    ImporterPimpl* Pimpl() { return pimpl; }
    const ImporterPimpl* Pimpl() const { return pimpl; }

protected:

    // Just because we don't want you to know how we're hacking around.
    ImporterPimpl* pimpl;
}; //! class Importer


// ----------------------------------------------------------------------------
// For compatibility, the interface of some functions taking a std::string was
// changed to const char* to avoid crashes between binary incompatible STL
// versions. This code her is inlined,  so it shouldn't cause any problems.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
AI_FORCE_INLINE const aiScene* Importer::ReadFile( const std::string& pFile,unsigned int pFlags){
    return ReadFile(pFile.c_str(),pFlags);
}
// ----------------------------------------------------------------------------
AI_FORCE_INLINE void Importer::GetExtensionList(std::string& szOut) const   {
    aiString s;
    GetExtensionList(s);
    szOut = s.data;
}
// ----------------------------------------------------------------------------
AI_FORCE_INLINE bool Importer::IsExtensionSupported(const std::string& szExtension) const   {
    return IsExtensionSupported(szExtension.c_str());
}

} // !namespace Assimp

#endif // AI_ASSIMP_HPP_INC
