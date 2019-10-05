/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

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

----------------------------------------------------------------------
*/

/** @file Definition of the base class for all importer worker classes. */
#ifndef INCLUDED_AI_BASEIMPORTER_H
#define INCLUDED_AI_BASEIMPORTER_H

#include "Exceptional.h"

#include <vector>
#include <set>
#include <map>
#include <assimp/types.h>
#include <assimp/ProgressHandler.hpp>
#include <assimp/ai_assert.h>

struct aiScene;
struct aiImporterDesc;

namespace Assimp    {

class Importer;
class IOSystem;
class BaseProcess;
class SharedPostProcessInfo;
class IOStream;

// utility to do char4 to uint32 in a portable manner
#define AI_MAKE_MAGIC(string) ((uint32_t)((string[0] << 24) + \
    (string[1] << 16) + (string[2] << 8) + string[3]))


// ---------------------------------------------------------------------------
/** FOR IMPORTER PLUGINS ONLY: The BaseImporter defines a common interface
 *  for all importer worker classes.
 *
 * The interface defines two functions: CanRead() is used to check if the
 * importer can handle the format of the given file. If an implementation of
 * this function returns true, the importer then calls ReadFile() which
 * imports the given file. ReadFile is not overridable, it just calls
 * InternReadFile() and catches any ImportErrorException that might occur.
 */
class ASSIMP_API BaseImporter {
    friend class Importer;

private:
    /* Pushes state into importer for the importer scale */
    virtual void UpdateImporterScale( Importer* pImp );

public:

    /** Constructor to be privately used by #Importer */
    BaseImporter() AI_NO_EXCEPT;

    /** Destructor, private as well */
    virtual ~BaseImporter();

    // -------------------------------------------------------------------
    /** Returns whether the class can handle the format of the given file.
     *
     * The implementation should be as quick as possible. A check for
     * the file extension is enough. If no suitable loader is found with
     * this strategy, CanRead() is called again, the 'checkSig' parameter
     * set to true this time. Now the implementation is expected to
     * perform a full check of the file structure, possibly searching the
     * first bytes of the file for magic identifiers or keywords.
     *
     * @param pFile Path and file name of the file to be examined.
     * @param pIOHandler The IO handler to use for accessing any file.
     * @param checkSig Set to true if this method is called a second time.
     *   This time, the implementation may take more time to examine the
     *   contents of the file to be loaded for magic bytes, keywords, etc
     *   to be able to load files with unknown/not existent file extensions.
     * @return true if the class can read this file, false if not.
     */
    virtual bool CanRead(
        const std::string& pFile,
        IOSystem* pIOHandler,
        bool checkSig
        ) const = 0;

    // -------------------------------------------------------------------
    /** Imports the given file and returns the imported data.
     * If the import succeeds, ownership of the data is transferred to
     * the caller. If the import fails, NULL is returned. The function
     * takes care that any partially constructed data is destroyed
     * beforehand.
     *
     * @param pImp #Importer object hosting this loader.
     * @param pFile Path of the file to be imported.
     * @param pIOHandler IO-Handler used to open this and possible other files.
     * @return The imported data or NULL if failed. If it failed a
     * human-readable error description can be retrieved by calling
     * GetErrorText()
     *
     * @note This function is not intended to be overridden. Implement
     * InternReadFile() to do the import. If an exception is thrown somewhere
     * in InternReadFile(), this function will catch it and transform it into
     *  a suitable response to the caller.
     */
    aiScene* ReadFile(
        Importer* pImp,
        const std::string& pFile,
        IOSystem* pIOHandler
        );

    // -------------------------------------------------------------------
    /** Returns the error description of the last error that occurred.
     * @return A description of the last error that occurred. An empty
     * string if there was no error.
     */
    const std::string& GetErrorText() const {
        return m_ErrorText;
    }

    // -------------------------------------------------------------------
    /** Called prior to ReadFile().
     * The function is a request to the importer to update its configuration
     * basing on the Importer's configuration property list.
     * @param pImp Importer instance
     */
    virtual void SetupProperties(
        const Importer* pImp
        );

    // -------------------------------------------------------------------
    /** Called by #Importer::GetImporterInfo to get a description of
     *  some loader features. Importers must provide this information. */
    virtual const aiImporterDesc* GetInfo() const = 0;

    /**
     * Will be called only by scale process when scaling is requested.
     */
    virtual void SetFileScale(double scale)
    {
        fileScale = scale;
    }

    virtual double GetFileScale() const
    {
        return fileScale;
    }

    enum ImporterUnits {
        M,
        MM,
        CM,
        INCHES,
        FEET
    };

    /**
     * Assimp Importer
     * unit conversions available 
     * if you need another measurment unit add it below.
     * it's currently defined in assimp that we prefer meters.
     * */
    std::map<ImporterUnits, double> importerUnits = {
        {ImporterUnits::M, 1},
        {ImporterUnits::CM, 0.01},
        {ImporterUnits::MM, 0.001},
        {ImporterUnits::INCHES, 0.0254},
        {ImporterUnits::FEET, 0.3048}
    };

    virtual void SetApplicationUnits( const ImporterUnits& unit )
    {
        importerScale = importerUnits[unit];
        applicationUnits = unit;
    }

    virtual const ImporterUnits& GetApplicationUnits()
    {
        return applicationUnits;
    }

    // -------------------------------------------------------------------
    /** Called by #Importer::GetExtensionList for each loaded importer.
     *  Take the extension list contained in the structure returned by
     *  #GetInfo and insert all file extensions into the given set.
     *  @param extension set to collect file extensions in*/
    void GetExtensionList(std::set<std::string>& extensions);
    
protected:    
    ImporterUnits applicationUnits = ImporterUnits::M;
    double importerScale = 1.0;
    double fileScale = 1.0;



    // -------------------------------------------------------------------
    /** Imports the given file into the given scene structure. The
     * function is expected to throw an ImportErrorException if there is
     * an error. If it terminates normally, the data in aiScene is
     * expected to be correct. Override this function to implement the
     * actual importing.
     * <br>
     *  The output scene must meet the following requirements:<br>
     * <ul>
     * <li>At least a root node must be there, even if its only purpose
     *     is to reference one mesh.</li>
     * <li>aiMesh::mPrimitiveTypes may be 0. The types of primitives
     *   in the mesh are determined automatically in this case.</li>
     * <li>the vertex data is stored in a pseudo-indexed "verbose" format.
     *   In fact this means that every vertex that is referenced by
     *   a face is unique. Or the other way round: a vertex index may
     *   not occur twice in a single aiMesh.</li>
     * <li>aiAnimation::mDuration may be -1. Assimp determines the length
     *   of the animation automatically in this case as the length of
     *   the longest animation channel.</li>
     * <li>aiMesh::mBitangents may be NULL if tangents and normals are
     *   given. In this case bitangents are computed as the cross product
     *   between normal and tangent.</li>
     * <li>There needn't be a material. If none is there a default material
     *   is generated. However, it is recommended practice for loaders
     *   to generate a default material for yourself that matches the
     *   default material setting for the file format better than Assimp's
     *   generic default material. Note that default materials *should*
     *   be named AI_DEFAULT_MATERIAL_NAME if they're just color-shaded
     *   or AI_DEFAULT_TEXTURED_MATERIAL_NAME if they define a (dummy)
     *   texture. </li>
     * </ul>
     * If the AI_SCENE_FLAGS_INCOMPLETE-Flag is <b>not</b> set:<ul>
     * <li> at least one mesh must be there</li>
     * <li> there may be no meshes with 0 vertices or faces</li>
     * </ul>
     * This won't be checked (except by the validation step): Assimp will
     * crash if one of the conditions is not met!
     *
     * @param pFile Path of the file to be imported.
     * @param pScene The scene object to hold the imported data.
     * NULL is not a valid parameter.
     * @param pIOHandler The IO handler to use for any file access.
     * NULL is not a valid parameter. */
    virtual void InternReadFile(
        const std::string& pFile,
        aiScene* pScene,
        IOSystem* pIOHandler
        ) = 0;

public: // static utilities

    // -------------------------------------------------------------------
    /** A utility for CanRead().
     *
     *  The function searches the header of a file for a specific token
     *  and returns true if this token is found. This works for text
     *  files only. There is a rudimentary handling of UNICODE files.
     *  The comparison is case independent.
     *
     *  @param pIOSystem IO System to work with
     *  @param file File name of the file
     *  @param tokens List of tokens to search for
     *  @param numTokens Size of the token array
     *  @param searchBytes Number of bytes to be searched for the tokens.
     */
    static bool SearchFileHeaderForToken(
        IOSystem* pIOSystem,
        const std::string&  file,
        const char** tokens,
        unsigned int numTokens,
        unsigned int searchBytes = 200,
        bool tokensSol = false,
        bool noAlphaBeforeTokens = false);

    // -------------------------------------------------------------------
    /** @brief Check whether a file has a specific file extension
     *  @param pFile Input file
     *  @param ext0 Extension to check for. Lowercase characters only, no dot!
     *  @param ext1 Optional second extension
     *  @param ext2 Optional third extension
     *  @note Case-insensitive
     */
    static bool SimpleExtensionCheck (
        const std::string& pFile,
        const char* ext0,
        const char* ext1 = NULL,
        const char* ext2 = NULL);

    // -------------------------------------------------------------------
    /** @brief Extract file extension from a string
     *  @param pFile Input file
     *  @return Extension without trailing dot, all lowercase
     */
    static std::string GetExtension (
        const std::string& pFile);

    // -------------------------------------------------------------------
    /** @brief Check whether a file starts with one or more magic tokens
     *  @param pFile Input file
     *  @param pIOHandler IO system to be used
     *  @param magic n magic tokens
     *  @params num Size of magic
     *  @param offset Offset from file start where tokens are located
     *  @param Size of one token, in bytes. Maximally 16 bytes.
     *  @return true if one of the given tokens was found
     *
     *  @note For convenience, the check is also performed for the
     *  byte-swapped variant of all tokens (big endian). Only for
     *  tokens of size 2,4.
     */
    static bool CheckMagicToken(
        IOSystem* pIOHandler,
        const std::string& pFile,
        const void* magic,
        unsigned int num,
        unsigned int offset = 0,
        unsigned int size   = 4);

    // -------------------------------------------------------------------
    /** An utility for all text file loaders. It converts a file to our
     *   UTF8 character set. Errors are reported, but ignored.
     *
     *  @param data File buffer to be converted to UTF8 data. The buffer
     *  is resized as appropriate. */
    static void ConvertToUTF8(
        std::vector<char>& data);

    // -------------------------------------------------------------------
    /** An utility for all text file loaders. It converts a file from our
     *   UTF8 character set back to ISO-8859-1. Errors are reported, but ignored.
     *
     *  @param data File buffer to be converted from UTF8 to ISO-8859-1. The buffer
     *  is resized as appropriate. */
    static void ConvertUTF8toISO8859_1(
        std::string& data);

    // -------------------------------------------------------------------
    /// @brief  Enum to define, if empty files are ok or not.
    enum TextFileMode { 
        ALLOW_EMPTY,
        FORBID_EMPTY 
    };

    // -------------------------------------------------------------------
    /** Utility for text file loaders which copies the contents of the
     *  file into a memory buffer and converts it to our UTF8
     *  representation.
     *  @param stream Stream to read from.
     *  @param data Output buffer to be resized and filled with the
     *   converted text file data. The buffer is terminated with
     *   a binary 0.
     *  @param mode Whether it is OK to load empty text files. */
    static void TextFileToBuffer(
        IOStream* stream,
        std::vector<char>& data,
        TextFileMode mode = FORBID_EMPTY);

    // -------------------------------------------------------------------
    /** Utility function to move a std::vector into a aiScene array
    *  @param vec The vector to be moved
    *  @param out The output pointer to the allocated array.
    *  @param numOut The output count of elements copied. */
    template<typename T>
    AI_FORCE_INLINE
    static void CopyVector(
        std::vector<T>& vec,
        T*& out,
        unsigned int& outLength)
    {
        outLength = unsigned(vec.size());
        if (outLength) {
            out = new T[outLength];
            std::swap_ranges(vec.begin(), vec.end(), out);
        }
    }

protected:
    /// Error description in case there was one.
    std::string m_ErrorText;
    /// Currently set progress handler.
    ProgressHandler* m_progress;
};



} // end of namespace Assimp

#endif // AI_BASEIMPORTER_H_INC
