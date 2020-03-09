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

/** @file Importer.h mostly internal stuff for use by #Assimp::Importer */
#pragma once
#ifndef INCLUDED_AI_IMPORTER_H
#define INCLUDED_AI_IMPORTER_H

#include <map>
#include <vector>
#include <string>
#include <assimp/matrix4x4.h>

struct aiScene;

namespace Assimp    {
    class ProgressHandler;
    class IOSystem;
    class BaseImporter;
    class BaseProcess;
    class SharedPostProcessInfo;


//! @cond never
// ---------------------------------------------------------------------------
/** @brief Internal PIMPL implementation for Assimp::Importer
 *
 *  Using this idiom here allows us to drop the dependency from
 *  std::vector and std::map in the public headers. Furthermore we are dropping
 *  any STL interface problems caused by mismatching STL settings. All
 *  size calculation are now done by us, not the app heap. */
class ImporterPimpl {
public:
    // Data type to store the key hash
    typedef unsigned int KeyType;

    // typedefs for our four configuration maps.
    // We don't need more, so there is no need for a generic solution
    typedef std::map<KeyType, int> IntPropertyMap;
    typedef std::map<KeyType, ai_real> FloatPropertyMap;
    typedef std::map<KeyType, std::string> StringPropertyMap;
    typedef std::map<KeyType, aiMatrix4x4> MatrixPropertyMap;

    /** IO handler to use for all file accesses. */
    IOSystem* mIOHandler;
    bool mIsDefaultHandler;

    /** Progress handler for feedback. */
    ProgressHandler* mProgressHandler;
    bool mIsDefaultProgressHandler;

    /** Format-specific importer worker objects - one for each format we can read.*/
    std::vector< BaseImporter* > mImporter;

    /** Post processing steps we can apply at the imported data. */
    std::vector< BaseProcess* > mPostProcessingSteps;

    /** The imported data, if ReadFile() was successful, NULL otherwise. */
    aiScene* mScene;

    /** The error description, if there was one. */
    std::string mErrorString;

    /** List of integer properties */
    IntPropertyMap mIntProperties;

    /** List of floating-point properties */
    FloatPropertyMap mFloatProperties;

    /** List of string properties */
    StringPropertyMap mStringProperties;

    /** List of Matrix properties */
    MatrixPropertyMap mMatrixProperties;

    /** Used for testing - extra verbose mode causes the ValidateDataStructure-Step
     *  to be executed before and after every single post-process step */
    bool bExtraVerbose;

    /** Used by post-process steps to share data */
    SharedPostProcessInfo* mPPShared;

    /// The default class constructor.
    ImporterPimpl() AI_NO_EXCEPT;
};

inline
ImporterPimpl::ImporterPimpl() AI_NO_EXCEPT
: mIOHandler( nullptr )
, mIsDefaultHandler( false )
, mProgressHandler( nullptr )
, mIsDefaultProgressHandler( false )
, mImporter()
, mPostProcessingSteps()
, mScene( nullptr )
, mErrorString()
, mIntProperties()
, mFloatProperties()
, mStringProperties()
, mMatrixProperties()
, bExtraVerbose( false )
, mPPShared( nullptr ) {
    // empty
}
//! @endcond


struct BatchData;

// ---------------------------------------------------------------------------
/** FOR IMPORTER PLUGINS ONLY: A helper class to the pleasure of importers
 *  that need to load many external meshes recursively.
 *
 *  The class uses several threads to load these meshes (or at least it
 *  could, this has not yet been implemented at the moment).
 *
 *  @note The class may not be used by more than one thread*/
class ASSIMP_API BatchLoader
{
    // friend of Importer

public:
    //! @cond never
    // -------------------------------------------------------------------
    /** Wraps a full list of configuration properties for an importer.
     *  Properties can be set using SetGenericProperty */
    struct PropertyMap
    {
        ImporterPimpl::IntPropertyMap     ints;
        ImporterPimpl::FloatPropertyMap   floats;
        ImporterPimpl::StringPropertyMap  strings;
        ImporterPimpl::MatrixPropertyMap  matrices;

        bool operator == (const PropertyMap& prop) const {
            // fixme: really isocpp? gcc complains
            return ints == prop.ints && floats == prop.floats && strings == prop.strings && matrices == prop.matrices;
        }

        bool empty () const {
            return ints.empty() && floats.empty() && strings.empty() && matrices.empty();
        }
    };
    //! @endcond

public:
    // -------------------------------------------------------------------
    /** Construct a batch loader from a given IO system to be used
     *  to access external files 
     */
    explicit BatchLoader(IOSystem* pIO, bool validate = false );

    // -------------------------------------------------------------------
    /** The class destructor.
     */
    ~BatchLoader();

    // -------------------------------------------------------------------
    /** Sets the validation step. True for enable validation during postprocess.
     *  @param  enable  True for validation.
     */
    void setValidation( bool enabled );
    
    // -------------------------------------------------------------------
    /** Returns the current validation step.
     *  @return The current validation step.
     */
    bool getValidation() const;
    
    // -------------------------------------------------------------------
    /** Add a new file to the list of files to be loaded.
     *  @param file File to be loaded
     *  @param steps Post-processing steps to be executed on the file
     *  @param map Optional configuration properties
     *  @return 'Load request channel' - an unique ID that can later
     *    be used to access the imported file data.
     *  @see GetImport */
    unsigned int AddLoadRequest (
        const std::string& file,
        unsigned int steps = 0,
        const PropertyMap* map = NULL
        );

    // -------------------------------------------------------------------
    /** Get an imported scene.
     *  This polls the import from the internal request list.
     *  If an import is requested several times, this function
     *  can be called several times, too.
     *
     *  @param which LRWC returned by AddLoadRequest().
     *  @return NULL if there is no scene with this file name
     *  in the queue of the scene hasn't been loaded yet. */
    aiScene* GetImport(
        unsigned int which
        );

    // -------------------------------------------------------------------
    /** Waits until all scenes have been loaded. This returns
     *  immediately if no scenes are queued.*/
    void LoadAll();

private:
    // No need to have that in the public API ...
    BatchData *m_data;
};

} // Namespace Assimp

#endif // INCLUDED_AI_IMPORTER_H
