/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team

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

/** @file  Importer.cpp
 *  @brief Implementation of the CPP-API class #Importer
 */

#include <assimp/version.h>
#include <assimp/config.h>
#include <assimp/importerdesc.h>

// ------------------------------------------------------------------------------------------------
/* Uncomment this line to prevent Assimp from catching unknown exceptions.
 *
 * Note that any Exception except DeadlyImportError may lead to
 * undefined behaviour -> loaders could remain in an unusable state and
 * further imports with the same Importer instance could fail/crash/burn ...
 */
// ------------------------------------------------------------------------------------------------
#ifndef ASSIMP_BUILD_DEBUG
#   define ASSIMP_CATCH_GLOBAL_EXCEPTIONS
#endif

// ------------------------------------------------------------------------------------------------
// Internal headers
// ------------------------------------------------------------------------------------------------
#include "Common/Importer.h"
#include "Common/BaseProcess.h"
#include "Common/DefaultProgressHandler.h"
#include "PostProcessing/ProcessHelper.h"
#include "Common/ScenePreprocessor.h"
#include "Common/ScenePrivate.h"

#include <assimp/BaseImporter.h>
#include <assimp/GenericProperty.h>
#include <assimp/MemoryIOWrapper.h>
#include <assimp/Profiler.h>
#include <assimp/TinyFormatter.h>
#include <assimp/Exceptional.h>
#include <assimp/Profiler.h>
#include <assimp/commonMetaData.h>

#include <set>
#include <memory>
#include <cctype>

#include <assimp/DefaultIOStream.h>
#include <assimp/DefaultIOSystem.h>

#ifndef ASSIMP_BUILD_NO_VALIDATEDS_PROCESS
#   include "PostProcessing/ValidateDataStructure.h"
#endif

using namespace Assimp::Profiling;
using namespace Assimp::Formatter;

namespace Assimp {
    // ImporterRegistry.cpp
    void GetImporterInstanceList(std::vector< BaseImporter* >& out);
	void DeleteImporterInstanceList(std::vector< BaseImporter* >& out);

    // PostStepRegistry.cpp
    void GetPostProcessingStepInstanceList(std::vector< BaseProcess* >& out);
}

using namespace Assimp;
using namespace Assimp::Intern;

// ------------------------------------------------------------------------------------------------
// Intern::AllocateFromAssimpHeap serves as abstract base class. It overrides
// new and delete (and their array counterparts) of public API classes (e.g. Logger) to
// utilize our DLL heap.
// See http://www.gotw.ca/publications/mill15.htm
// ------------------------------------------------------------------------------------------------
void* AllocateFromAssimpHeap::operator new ( size_t num_bytes)  {
    return ::operator new(num_bytes);
}

void* AllocateFromAssimpHeap::operator new ( size_t num_bytes, const std::nothrow_t& ) throw()  {
    try {
        return AllocateFromAssimpHeap::operator new( num_bytes );
    }
    catch( ... )    {
        return nullptr;
    }
}

void AllocateFromAssimpHeap::operator delete ( void* data)  {
    return ::operator delete(data);
}

void* AllocateFromAssimpHeap::operator new[] ( size_t num_bytes)    {
    return ::operator new[](num_bytes);
}

void* AllocateFromAssimpHeap::operator new[] ( size_t num_bytes, const std::nothrow_t& ) throw() {
    try {
        return AllocateFromAssimpHeap::operator new[]( num_bytes );
    } catch( ... )    {
        return nullptr;
    }
}

void AllocateFromAssimpHeap::operator delete[] ( void* data)    {
    return ::operator delete[](data);
}

// ------------------------------------------------------------------------------------------------
// Importer constructor.
Importer::Importer()
 : pimpl( new ImporterPimpl ) {
    pimpl->mScene = nullptr;
    pimpl->mErrorString = "";

    // Allocate a default IO handler
    pimpl->mIOHandler = new DefaultIOSystem;
    pimpl->mIsDefaultHandler = true;
    pimpl->bExtraVerbose     = false; // disable extra verbose mode by default

    pimpl->mProgressHandler = new DefaultProgressHandler();
    pimpl->mIsDefaultProgressHandler = true;

    GetImporterInstanceList(pimpl->mImporter);
    GetPostProcessingStepInstanceList(pimpl->mPostProcessingSteps);

    // Allocate a SharedPostProcessInfo object and store pointers to it in all post-process steps in the list.
    pimpl->mPPShared = new SharedPostProcessInfo();
    for (std::vector<BaseProcess*>::iterator it =  pimpl->mPostProcessingSteps.begin();
        it != pimpl->mPostProcessingSteps.end();
        ++it)   {

        (*it)->SetSharedData(pimpl->mPPShared);
    }
}

// ------------------------------------------------------------------------------------------------
// Destructor of Importer
Importer::~Importer() {
    // Delete all import plugins
	DeleteImporterInstanceList(pimpl->mImporter);

    // Delete all post-processing plug-ins
    for( unsigned int a = 0; a < pimpl->mPostProcessingSteps.size(); ++a ) {
        delete pimpl->mPostProcessingSteps[a];
    }

    // Delete the assigned IO and progress handler
    delete pimpl->mIOHandler;
    delete pimpl->mProgressHandler;

    // Kill imported scene. Destructor's should do that recursively
    delete pimpl->mScene;

    // Delete shared post-processing data
    delete pimpl->mPPShared;

    // and finally the pimpl itself
    delete pimpl;
}

// ------------------------------------------------------------------------------------------------
// Register a custom post-processing step
aiReturn Importer::RegisterPPStep(BaseProcess* pImp) {
    ai_assert( nullptr != pImp );
    
    ASSIMP_BEGIN_EXCEPTION_REGION();

        pimpl->mPostProcessingSteps.push_back(pImp);
        ASSIMP_LOG_INFO("Registering custom post-processing step");

    ASSIMP_END_EXCEPTION_REGION(aiReturn);
    return AI_SUCCESS;
}

// ------------------------------------------------------------------------------------------------
// Register a custom loader plugin
aiReturn Importer::RegisterLoader(BaseImporter* pImp) {
    ai_assert(nullptr != pImp);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();

    // --------------------------------------------------------------------
    // Check whether we would have two loaders for the same file extension
    // This is absolutely OK, but we should warn the developer of the new
    // loader that his code will probably never be called if the first
    // loader is a bit too lazy in his file checking.
    // --------------------------------------------------------------------
    std::set<std::string> st;
    std::string baked;
    pImp->GetExtensionList(st);

    for(std::set<std::string>::const_iterator it = st.begin(); it != st.end(); ++it) {

#ifdef ASSIMP_BUILD_DEBUG
        if (IsExtensionSupported(*it)) {
            ASSIMP_LOG_WARN_F("The file extension ", *it, " is already in use");
        }
#endif
        baked += *it;
    }

    // add the loader
    pimpl->mImporter.push_back(pImp);
    ASSIMP_LOG_INFO_F("Registering custom importer for these file extensions: ", baked);
    ASSIMP_END_EXCEPTION_REGION(aiReturn);
    
    return AI_SUCCESS;
}

// ------------------------------------------------------------------------------------------------
// Unregister a custom loader plugin
aiReturn Importer::UnregisterLoader(BaseImporter* pImp) {
    if(!pImp) {
        // unregistering a NULL importer is no problem for us ... really!
        return AI_SUCCESS;
    }

    ASSIMP_BEGIN_EXCEPTION_REGION();
    std::vector<BaseImporter*>::iterator it = std::find(pimpl->mImporter.begin(),
        pimpl->mImporter.end(),pImp);

    if (it != pimpl->mImporter.end())   {
        pimpl->mImporter.erase(it);
        ASSIMP_LOG_INFO("Unregistering custom importer: ");
        return AI_SUCCESS;
    }
    ASSIMP_LOG_WARN("Unable to remove custom importer: I can't find you ...");
    ASSIMP_END_EXCEPTION_REGION(aiReturn);

    return AI_FAILURE;
}

// ------------------------------------------------------------------------------------------------
// Unregister a custom loader plugin
aiReturn Importer::UnregisterPPStep(BaseProcess* pImp) {
    if(!pImp) {
        // unregistering a NULL ppstep is no problem for us ... really!
        return AI_SUCCESS;
    }

    ASSIMP_BEGIN_EXCEPTION_REGION();
    std::vector<BaseProcess*>::iterator it = std::find(pimpl->mPostProcessingSteps.begin(),
        pimpl->mPostProcessingSteps.end(),pImp);

    if (it != pimpl->mPostProcessingSteps.end())    {
        pimpl->mPostProcessingSteps.erase(it);
        ASSIMP_LOG_INFO("Unregistering custom post-processing step");
        return AI_SUCCESS;
    }
    ASSIMP_LOG_WARN("Unable to remove custom post-processing step: I can't find you ..");
    ASSIMP_END_EXCEPTION_REGION(aiReturn);

    return AI_FAILURE;
}

// ------------------------------------------------------------------------------------------------
// Supplies a custom IO handler to the importer to open and access files.
void Importer::SetIOHandler( IOSystem* pIOHandler) {
    ai_assert(nullptr != pimpl);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();
    // If the new handler is zero, allocate a default IO implementation.
    if (!pIOHandler) {
        // Release pointer in the possession of the caller
        pimpl->mIOHandler = new DefaultIOSystem();
        pimpl->mIsDefaultHandler = true;
    } else if (pimpl->mIOHandler != pIOHandler) { // Otherwise register the custom handler
        delete pimpl->mIOHandler;
        pimpl->mIOHandler = pIOHandler;
        pimpl->mIsDefaultHandler = false;
    }
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Get the currently set IO handler
IOSystem* Importer::GetIOHandler() const {
    ai_assert(nullptr != pimpl);
    
    return pimpl->mIOHandler;
}

// ------------------------------------------------------------------------------------------------
// Check whether a custom IO handler is currently set
bool Importer::IsDefaultIOHandler() const {
    ai_assert(nullptr != pimpl);
    
    return pimpl->mIsDefaultHandler;
}

// ------------------------------------------------------------------------------------------------
// Supplies a custom progress handler to get regular callbacks during importing
void Importer::SetProgressHandler ( ProgressHandler* pHandler ) {
    ai_assert(nullptr != pimpl);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();
    
    // If the new handler is zero, allocate a default implementation.
    if (!pHandler) {
        // Release pointer in the possession of the caller
        pimpl->mProgressHandler = new DefaultProgressHandler();
        pimpl->mIsDefaultProgressHandler = true;
    } else if (pimpl->mProgressHandler != pHandler) { // Otherwise register the custom handler
        delete pimpl->mProgressHandler;
        pimpl->mProgressHandler = pHandler;
        pimpl->mIsDefaultProgressHandler = false;
    }
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Get the currently set progress handler
ProgressHandler* Importer::GetProgressHandler() const {
    ai_assert(nullptr != pimpl);
    
    return pimpl->mProgressHandler;
}

// ------------------------------------------------------------------------------------------------
// Check whether a custom progress handler is currently set
bool Importer::IsDefaultProgressHandler() const {
    ai_assert(nullptr != pimpl);
    
    return pimpl->mIsDefaultProgressHandler;
}

// ------------------------------------------------------------------------------------------------
// Validate post process step flags
bool _ValidateFlags(unsigned int pFlags) {
    if (pFlags & aiProcess_GenSmoothNormals && pFlags & aiProcess_GenNormals)   {
        ASSIMP_LOG_ERROR("#aiProcess_GenSmoothNormals and #aiProcess_GenNormals are incompatible");
        return false;
    }
    if (pFlags & aiProcess_OptimizeGraph && pFlags & aiProcess_PreTransformVertices)    {
        ASSIMP_LOG_ERROR("#aiProcess_OptimizeGraph and #aiProcess_PreTransformVertices are incompatible");
        return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// Free the current scene
void Importer::FreeScene( ) {
    ai_assert(nullptr != pimpl);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();

    delete pimpl->mScene;
    pimpl->mScene = nullptr;

    pimpl->mErrorString = "";
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Get the current error string, if any
const char* Importer::GetErrorString() const {
    ai_assert(nullptr != pimpl);
    
    // Must remain valid as long as ReadFile() or FreeFile() are not called
    return pimpl->mErrorString.c_str();
}

// ------------------------------------------------------------------------------------------------
// Enable extra-verbose mode
void Importer::SetExtraVerbose(bool bDo) {
    ai_assert(nullptr != pimpl);
    
    pimpl->bExtraVerbose = bDo;
}

// ------------------------------------------------------------------------------------------------
// Get the current scene
const aiScene* Importer::GetScene() const {
    ai_assert(nullptr != pimpl);
    
    return pimpl->mScene;
}

// ------------------------------------------------------------------------------------------------
// Orphan the current scene and return it.
aiScene* Importer::GetOrphanedScene() {
    ai_assert(nullptr != pimpl);
    
    aiScene* s = pimpl->mScene;

    ASSIMP_BEGIN_EXCEPTION_REGION();
    pimpl->mScene = nullptr;

    pimpl->mErrorString = ""; // reset error string
    ASSIMP_END_EXCEPTION_REGION(aiScene*);
    
    return s;
}

// ------------------------------------------------------------------------------------------------
// Validate post-processing flags
bool Importer::ValidateFlags(unsigned int pFlags) const {
    ASSIMP_BEGIN_EXCEPTION_REGION();
    // run basic checks for mutually exclusive flags
    if(!_ValidateFlags(pFlags)) {
        return false;
    }

    // ValidateDS does not anymore occur in the pp list, it plays an awesome extra role ...
#ifdef ASSIMP_BUILD_NO_VALIDATEDS_PROCESS
    if (pFlags & aiProcess_ValidateDataStructure) {
        return false;
    }
#endif
    pFlags &= ~aiProcess_ValidateDataStructure;

    // Now iterate through all bits which are set in the flags and check whether we find at least
    // one pp plugin which handles it.
    for (unsigned int mask = 1; mask < (1u << (sizeof(unsigned int)*8-1));mask <<= 1) {

        if (pFlags & mask) {

            bool have = false;
            for( unsigned int a = 0; a < pimpl->mPostProcessingSteps.size(); a++)   {
                if (pimpl->mPostProcessingSteps[a]-> IsActive(mask) ) {

                    have = true;
                    break;
                }
            }
            if (!have) {
                return false;
            }
        }
    }
    ASSIMP_END_EXCEPTION_REGION(bool);
    return true;
}

// ------------------------------------------------------------------------------------------------
const aiScene* Importer::ReadFileFromMemory( const void* pBuffer,
    size_t pLength,
    unsigned int pFlags,
    const char* pHint /*= ""*/) {
    ai_assert(nullptr != pimpl);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();
    if (!pHint) {
        pHint = "";
    }

    if (!pBuffer || !pLength || strlen(pHint) > MaxLenHint ) {
        pimpl->mErrorString = "Invalid parameters passed to ReadFileFromMemory()";
        return nullptr;
    }

    // prevent deletion of the previous IOHandler
    IOSystem* io = pimpl->mIOHandler;
    pimpl->mIOHandler = nullptr;

    SetIOHandler(new MemoryIOSystem((const uint8_t*)pBuffer,pLength,io));

    // read the file and recover the previous IOSystem
    static const size_t BufSize(Importer::MaxLenHint + 28);
    char fbuff[BufSize];
    ai_snprintf(fbuff, BufSize, "%s.%s",AI_MEMORYIO_MAGIC_FILENAME,pHint);

    ReadFile(fbuff,pFlags);
    SetIOHandler(io);

    ASSIMP_END_EXCEPTION_REGION_WITH_ERROR_STRING(const aiScene*, pimpl->mErrorString);
    return pimpl->mScene;
}

// ------------------------------------------------------------------------------------------------
void WriteLogOpening(const std::string& file) {
    
    ASSIMP_LOG_INFO_F("Load ", file);

    // print a full version dump. This is nice because we don't
    // need to ask the authors of incoming bug reports for
    // the library version they're using - a log dump is
    // sufficient.
    const unsigned int flags( aiGetCompileFlags() );
    std::stringstream stream;
    stream << "Assimp " << aiGetVersionMajor() << "." << aiGetVersionMinor() << "." << aiGetVersionRevision() << " "
#if defined(ASSIMP_BUILD_ARCHITECTURE)
        << ASSIMP_BUILD_ARCHITECTURE
#elif defined(_M_IX86) || defined(__x86_32__) || defined(__i386__)
        << "x86"
#elif defined(_M_X64) || defined(__x86_64__)
        << "amd64"
#elif defined(_M_IA64) || defined(__ia64__)
        << "itanium"
#elif defined(__ppc__) || defined(__powerpc__)
        << "ppc32"
#elif defined(__powerpc64__)
        << "ppc64"
#elif defined(__arm__)
        << "arm"
#else
        << "<unknown architecture>"
#endif
        << " "
#if defined(ASSIMP_BUILD_COMPILER)
        << ( ASSIMP_BUILD_COMPILER )
#elif defined(_MSC_VER)
        << "msvc"
#elif defined(__GNUC__)
        << "gcc"
#else
        << "<unknown compiler>"
#endif

#ifdef ASSIMP_BUILD_DEBUG
        << " debug"
#endif

        << (flags & ASSIMP_CFLAGS_NOBOOST ? " noboost" : "")
        << (flags & ASSIMP_CFLAGS_SHARED  ? " shared" : "")
        << (flags & ASSIMP_CFLAGS_SINGLETHREADED  ? " singlethreaded" : "");

        ASSIMP_LOG_DEBUG(stream.str());
}

// ------------------------------------------------------------------------------------------------
// Reads the given file and returns its contents if successful.
const aiScene* Importer::ReadFile( const char* _pFile, unsigned int pFlags) {
    ai_assert(nullptr != pimpl);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();
    const std::string pFile(_pFile);

    // ----------------------------------------------------------------------
    // Put a large try block around everything to catch all std::exception's
    // that might be thrown by STL containers or by new().
    // ImportErrorException's are throw by ourselves and caught elsewhere.
    //-----------------------------------------------------------------------

    WriteLogOpening(pFile);

#ifdef ASSIMP_CATCH_GLOBAL_EXCEPTIONS
    try
#endif // ! ASSIMP_CATCH_GLOBAL_EXCEPTIONS
    {
        // Check whether this Importer instance has already loaded
        // a scene. In this case we need to delete the old one
        if (pimpl->mScene)  {

            ASSIMP_LOG_DEBUG("(Deleting previous scene)");
            FreeScene();
        }

        // First check if the file is accessible at all
        if( !pimpl->mIOHandler->Exists( pFile)) {

            pimpl->mErrorString = "Unable to open file \"" + pFile + "\".";
            ASSIMP_LOG_ERROR(pimpl->mErrorString);
            return nullptr;
        }

        std::unique_ptr<Profiler> profiler(GetPropertyInteger(AI_CONFIG_GLOB_MEASURE_TIME,0)?new Profiler():NULL);
        if (profiler) {
            profiler->BeginRegion("total");
        }

        // Find an worker class which can handle the file
        BaseImporter* imp = nullptr;
        SetPropertyInteger("importerIndex", -1);
        for( unsigned int a = 0; a < pimpl->mImporter.size(); a++)  {

            if( pimpl->mImporter[a]->CanRead( pFile, pimpl->mIOHandler, false)) {
                imp = pimpl->mImporter[a];
                SetPropertyInteger("importerIndex", a);
                break;
            }
        }

        if (!imp)   {
            // not so bad yet ... try format auto detection.
            const std::string::size_type s = pFile.find_last_of('.');
            if (s != std::string::npos) {
                ASSIMP_LOG_INFO("File extension not known, trying signature-based detection");
                for( unsigned int a = 0; a < pimpl->mImporter.size(); a++)  {
                    if( pimpl->mImporter[a]->CanRead( pFile, pimpl->mIOHandler, true)) {
                        imp = pimpl->mImporter[a];
                        SetPropertyInteger("importerIndex", a);
                        break;
                    }
                }
            }
            // Put a proper error message if no suitable importer was found
            if( !imp)   {
                pimpl->mErrorString = "No suitable reader found for the file format of file \"" + pFile + "\".";
                ASSIMP_LOG_ERROR(pimpl->mErrorString);
                return nullptr;
            }
        }

        // Get file size for progress handler
        IOStream * fileIO = pimpl->mIOHandler->Open( pFile );
        uint32_t fileSize = 0;
        if (fileIO)
        {
            fileSize = static_cast<uint32_t>(fileIO->FileSize());
            pimpl->mIOHandler->Close( fileIO );
        }

        // Dispatch the reading to the worker class for this format
        const aiImporterDesc *desc( imp->GetInfo() );
        std::string ext( "unknown" );
        if ( nullptr != desc ) {
            ext = desc->mName;
        }
        ASSIMP_LOG_INFO("Found a matching importer for this file format: " + ext + "." );
        pimpl->mProgressHandler->UpdateFileRead( 0, fileSize );

        if (profiler) {
            profiler->BeginRegion("import");
        }

        pimpl->mScene = imp->ReadFile( this, pFile, pimpl->mIOHandler);
        pimpl->mProgressHandler->UpdateFileRead( fileSize, fileSize );

        if (profiler) {
            profiler->EndRegion("import");
        }

        SetPropertyString("sourceFilePath", pFile);

        // If successful, apply all active post processing steps to the imported data
        if( pimpl->mScene)  {
            if (!pimpl->mScene->mMetaData || !pimpl->mScene->mMetaData->HasKey(AI_METADATA_SOURCE_FORMAT)) {
                if (!pimpl->mScene->mMetaData) {
                    pimpl->mScene->mMetaData = new aiMetadata;
                }
                pimpl->mScene->mMetaData->Add(AI_METADATA_SOURCE_FORMAT, aiString(ext));
            }

#ifndef ASSIMP_BUILD_NO_VALIDATEDS_PROCESS
            // The ValidateDS process is an exception. It is executed first, even before ScenePreprocessor is called.
            if (pFlags & aiProcess_ValidateDataStructure) {
                ValidateDSProcess ds;
                ds.ExecuteOnScene (this);
                if (!pimpl->mScene) {
                    return nullptr;
                }
            }
#endif // no validation

            // Preprocess the scene and prepare it for post-processing
            if (profiler) {
                profiler->BeginRegion("preprocess");
            }

            ScenePreprocessor pre(pimpl->mScene);
            pre.ProcessScene();

            if (profiler) {
                profiler->EndRegion("preprocess");
            }

            // Ensure that the validation process won't be called twice
            ApplyPostProcessing(pFlags & (~aiProcess_ValidateDataStructure));
        }
        // if failed, extract the error string
        else if( !pimpl->mScene) {
            pimpl->mErrorString = imp->GetErrorText();
        }

        // clear any data allocated by post-process steps
        pimpl->mPPShared->Clean();

        if (profiler) {
            profiler->EndRegion("total");
        }
    }
#ifdef ASSIMP_CATCH_GLOBAL_EXCEPTIONS
    catch (std::exception &e) {
#if (defined _MSC_VER) &&   (defined _CPPRTTI)
        // if we have RTTI get the full name of the exception that occurred
        pimpl->mErrorString = std::string(typeid( e ).name()) + ": " + e.what();
#else
        pimpl->mErrorString = std::string("std::exception: ") + e.what();
#endif

        ASSIMP_LOG_ERROR(pimpl->mErrorString);
        delete pimpl->mScene; pimpl->mScene = nullptr;
    }
#endif // ! ASSIMP_CATCH_GLOBAL_EXCEPTIONS

    // either successful or failure - the pointer expresses it anyways
    ASSIMP_END_EXCEPTION_REGION_WITH_ERROR_STRING(const aiScene*, pimpl->mErrorString);
    
    return pimpl->mScene;
}


// ------------------------------------------------------------------------------------------------
// Apply post-processing to the currently bound scene
const aiScene* Importer::ApplyPostProcessing(unsigned int pFlags) {
    ai_assert(nullptr != pimpl);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();
    // Return immediately if no scene is active
    if (!pimpl->mScene) {
        return nullptr;
    }

    // If no flags are given, return the current scene with no further action
    if (!pFlags) {
        return pimpl->mScene;
    }

    // In debug builds: run basic flag validation
    ai_assert(_ValidateFlags(pFlags));
    ASSIMP_LOG_INFO("Entering post processing pipeline");

#ifndef ASSIMP_BUILD_NO_VALIDATEDS_PROCESS
    // The ValidateDS process plays an exceptional role. It isn't contained in the global
    // list of post-processing steps, so we need to call it manually.
    if (pFlags & aiProcess_ValidateDataStructure) {
        ValidateDSProcess ds;
        ds.ExecuteOnScene (this);
        if (!pimpl->mScene) {
            return nullptr;
        }
    }
#endif // no validation
#ifdef ASSIMP_BUILD_DEBUG
    if (pimpl->bExtraVerbose)
    {
#ifdef ASSIMP_BUILD_NO_VALIDATEDS_PROCESS
        ASSIMP_LOG_ERROR("Verbose Import is not available due to build settings");
#endif  // no validation
        pFlags |= aiProcess_ValidateDataStructure;
    }
#else
    if (pimpl->bExtraVerbose) {
        ASSIMP_LOG_WARN("Not a debug build, ignoring extra verbose setting");
    }
#endif // ! DEBUG

    std::unique_ptr<Profiler> profiler(GetPropertyInteger(AI_CONFIG_GLOB_MEASURE_TIME,0)?new Profiler():NULL);
    for( unsigned int a = 0; a < pimpl->mPostProcessingSteps.size(); a++)   {
        BaseProcess* process = pimpl->mPostProcessingSteps[a];
        pimpl->mProgressHandler->UpdatePostProcess(static_cast<int>(a), static_cast<int>(pimpl->mPostProcessingSteps.size()) );
        if( process->IsActive( pFlags)) {
            if (profiler) {
                profiler->BeginRegion("postprocess");
            }

            process->ExecuteOnScene ( this );

            if (profiler) {
                profiler->EndRegion("postprocess");
            }
        }
        if( !pimpl->mScene) {
            break;
        }
#ifdef ASSIMP_BUILD_DEBUG

#ifdef ASSIMP_BUILD_NO_VALIDATEDS_PROCESS
        continue;
#endif  // no validation

        // If the extra verbose mode is active, execute the ValidateDataStructureStep again - after each step
        if (pimpl->bExtraVerbose)   {
            ASSIMP_LOG_DEBUG("Verbose Import: re-validating data structures");

            ValidateDSProcess ds;
            ds.ExecuteOnScene (this);
            if( !pimpl->mScene) {
                ASSIMP_LOG_ERROR("Verbose Import: failed to re-validate data structures");
                break;
            }
        }
#endif // ! DEBUG
    }
    pimpl->mProgressHandler->UpdatePostProcess( static_cast<int>(pimpl->mPostProcessingSteps.size()), 
        static_cast<int>(pimpl->mPostProcessingSteps.size()) );

    // update private scene flags
    if( pimpl->mScene ) {
      ScenePriv(pimpl->mScene)->mPPStepsApplied |= pFlags;
    }

    // clear any data allocated by post-process steps
    pimpl->mPPShared->Clean();
    ASSIMP_LOG_INFO("Leaving post processing pipeline");

    ASSIMP_END_EXCEPTION_REGION(const aiScene*);
    
    return pimpl->mScene;
}

// ------------------------------------------------------------------------------------------------
const aiScene* Importer::ApplyCustomizedPostProcessing( BaseProcess *rootProcess, bool requestValidation ) {
    ai_assert(nullptr != pimpl);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();

    // Return immediately if no scene is active
    if ( nullptr == pimpl->mScene ) {
        return nullptr;
    }

    // If no flags are given, return the current scene with no further action
    if ( NULL == rootProcess ) {
        return pimpl->mScene;
    }

    // In debug builds: run basic flag validation
    ASSIMP_LOG_INFO( "Entering customized post processing pipeline" );

#ifndef ASSIMP_BUILD_NO_VALIDATEDS_PROCESS
    // The ValidateDS process plays an exceptional role. It isn't contained in the global
    // list of post-processing steps, so we need to call it manually.
    if ( requestValidation )
    {
        ValidateDSProcess ds;
        ds.ExecuteOnScene( this );
        if ( !pimpl->mScene ) {
            return nullptr;
        }
    }
#endif // no validation
#ifdef ASSIMP_BUILD_DEBUG
    if ( pimpl->bExtraVerbose )
    {
#ifdef ASSIMP_BUILD_NO_VALIDATEDS_PROCESS
        ASSIMP_LOG_ERROR( "Verbose Import is not available due to build settings" );
#endif  // no validation
    }
#else
    if ( pimpl->bExtraVerbose ) {
        ASSIMP_LOG_WARN( "Not a debug build, ignoring extra verbose setting" );
    }
#endif // ! DEBUG

    std::unique_ptr<Profiler> profiler( GetPropertyInteger( AI_CONFIG_GLOB_MEASURE_TIME, 0 ) ? new Profiler() : NULL );

    if ( profiler ) {
        profiler->BeginRegion( "postprocess" );
    }

    rootProcess->ExecuteOnScene( this );

    if ( profiler ) {
        profiler->EndRegion( "postprocess" );
    }

    // If the extra verbose mode is active, execute the ValidateDataStructureStep again - after each step
    if ( pimpl->bExtraVerbose || requestValidation  ) {
        ASSIMP_LOG_DEBUG( "Verbose Import: revalidating data structures" );

        ValidateDSProcess ds;
        ds.ExecuteOnScene( this );
        if ( !pimpl->mScene ) {
            ASSIMP_LOG_ERROR( "Verbose Import: failed to revalidate data structures" );
        }
    }

    // clear any data allocated by post-process steps
    pimpl->mPPShared->Clean();
    ASSIMP_LOG_INFO( "Leaving customized post processing pipeline" );

    ASSIMP_END_EXCEPTION_REGION( const aiScene* );

    return pimpl->mScene;
}

// ------------------------------------------------------------------------------------------------
// Helper function to check whether an extension is supported by ASSIMP
bool Importer::IsExtensionSupported(const char* szExtension) const {
    return nullptr != GetImporter(szExtension);
}

// ------------------------------------------------------------------------------------------------
size_t Importer::GetImporterCount() const {
    ai_assert(nullptr != pimpl);
    
    return pimpl->mImporter.size();
}

// ------------------------------------------------------------------------------------------------
const aiImporterDesc* Importer::GetImporterInfo(size_t index) const {
    ai_assert(nullptr != pimpl);
    
    if (index >= pimpl->mImporter.size()) {
        return nullptr;
    }
    return pimpl->mImporter[index]->GetInfo();
}


// ------------------------------------------------------------------------------------------------
BaseImporter* Importer::GetImporter (size_t index) const {
    ai_assert(nullptr != pimpl);
    
    if (index >= pimpl->mImporter.size()) {
        return nullptr;
    }
    return pimpl->mImporter[index];
}

// ------------------------------------------------------------------------------------------------
// Find a loader plugin for a given file extension
BaseImporter* Importer::GetImporter (const char* szExtension) const {
    ai_assert(nullptr != pimpl);
    
    return GetImporter(GetImporterIndex(szExtension));
}

// ------------------------------------------------------------------------------------------------
// Find a loader plugin for a given file extension
size_t Importer::GetImporterIndex (const char* szExtension) const {
    ai_assert(nullptr != pimpl);
    ai_assert(nullptr != szExtension);

    ASSIMP_BEGIN_EXCEPTION_REGION();

    // skip over wildcard and dot characters at string head --
    for ( ; *szExtension == '*' || *szExtension == '.'; ++szExtension );

    std::string ext(szExtension);
    if (ext.empty()) {
        return static_cast<size_t>(-1);
    }
    std::transform( ext.begin(), ext.end(), ext.begin(), ToLower<char> );

    std::set<std::string> str;
    for (std::vector<BaseImporter*>::const_iterator i =  pimpl->mImporter.begin();i != pimpl->mImporter.end();++i)  {
        str.clear();

        (*i)->GetExtensionList(str);
        for (std::set<std::string>::const_iterator it = str.begin(); it != str.end(); ++it) {
            if (ext == *it) {
                return std::distance(static_cast< std::vector<BaseImporter*>::const_iterator >(pimpl->mImporter.begin()), i);
            }
        }
    }
    ASSIMP_END_EXCEPTION_REGION(size_t);
    return static_cast<size_t>(-1);
}

// ------------------------------------------------------------------------------------------------
// Helper function to build a list of all file extensions supported by ASSIMP
void Importer::GetExtensionList(aiString& szOut) const {
    ai_assert(nullptr != pimpl);
    
    ASSIMP_BEGIN_EXCEPTION_REGION();
    std::set<std::string> str;
    for (std::vector<BaseImporter*>::const_iterator i =  pimpl->mImporter.begin();i != pimpl->mImporter.end();++i)  {
        (*i)->GetExtensionList(str);
    }

	// List can be empty
	if( !str.empty() ) {
		for (std::set<std::string>::const_iterator it = str.begin();; ) {
			szOut.Append("*.");
			szOut.Append((*it).c_str());

			if (++it == str.end()) {
				break;
			}
			szOut.Append(";");
		}
	}
    ASSIMP_END_EXCEPTION_REGION(void);
}

// ------------------------------------------------------------------------------------------------
// Set a configuration property
bool Importer::SetPropertyInteger(const char* szName, int iValue) {
    ai_assert(nullptr != pimpl);
    
    bool existing;
    ASSIMP_BEGIN_EXCEPTION_REGION();
        existing = SetGenericProperty<int>(pimpl->mIntProperties, szName,iValue);
    ASSIMP_END_EXCEPTION_REGION(bool);
    return existing;
}

// ------------------------------------------------------------------------------------------------
// Set a configuration property
bool Importer::SetPropertyFloat(const char* szName, ai_real iValue) {
    ai_assert(nullptr != pimpl);
    
    bool existing;
    ASSIMP_BEGIN_EXCEPTION_REGION();
        existing = SetGenericProperty<ai_real>(pimpl->mFloatProperties, szName,iValue);
    ASSIMP_END_EXCEPTION_REGION(bool);
    return existing;
}

// ------------------------------------------------------------------------------------------------
// Set a configuration property
bool Importer::SetPropertyString(const char* szName, const std::string& value) {
    ai_assert(nullptr != pimpl);
    
    bool existing;
    ASSIMP_BEGIN_EXCEPTION_REGION();
        existing = SetGenericProperty<std::string>(pimpl->mStringProperties, szName,value);
    ASSIMP_END_EXCEPTION_REGION(bool);
    return existing;
}

// ------------------------------------------------------------------------------------------------
// Set a configuration property
bool Importer::SetPropertyMatrix(const char* szName, const aiMatrix4x4& value) {
    ai_assert(nullptr != pimpl);
    
    bool existing;
    ASSIMP_BEGIN_EXCEPTION_REGION();
        existing = SetGenericProperty<aiMatrix4x4>(pimpl->mMatrixProperties, szName,value);
    ASSIMP_END_EXCEPTION_REGION(bool);
    return existing;
}

// ------------------------------------------------------------------------------------------------
// Get a configuration property
int Importer::GetPropertyInteger(const char* szName, int iErrorReturn /*= 0xffffffff*/) const {
    ai_assert(nullptr != pimpl);
    
    return GetGenericProperty<int>(pimpl->mIntProperties,szName,iErrorReturn);
}

// ------------------------------------------------------------------------------------------------
// Get a configuration property
ai_real Importer::GetPropertyFloat(const char* szName, ai_real iErrorReturn /*= 10e10*/) const {
    ai_assert(nullptr != pimpl);
    
    return GetGenericProperty<ai_real>(pimpl->mFloatProperties,szName,iErrorReturn);
}

// ------------------------------------------------------------------------------------------------
// Get a configuration property
std::string Importer::GetPropertyString(const char* szName, const std::string& iErrorReturn /*= ""*/) const {
    ai_assert(nullptr != pimpl);
    
    return GetGenericProperty<std::string>(pimpl->mStringProperties,szName,iErrorReturn);
}

// ------------------------------------------------------------------------------------------------
// Get a configuration property
aiMatrix4x4 Importer::GetPropertyMatrix(const char* szName, const aiMatrix4x4& iErrorReturn /*= aiMatrix4x4()*/) const {
    ai_assert(nullptr != pimpl);
    
    return GetGenericProperty<aiMatrix4x4>(pimpl->mMatrixProperties,szName,iErrorReturn);
}

// ------------------------------------------------------------------------------------------------
// Get the memory requirements of a single node
inline 
void AddNodeWeight(unsigned int& iScene,const aiNode* pcNode) {
    if ( nullptr == pcNode ) {
        return;
    }
    iScene += sizeof(aiNode);
    iScene += sizeof(unsigned int) * pcNode->mNumMeshes;
    iScene += sizeof(void*) * pcNode->mNumChildren;

    for (unsigned int i = 0; i < pcNode->mNumChildren;++i) {
        AddNodeWeight(iScene,pcNode->mChildren[i]);
    }
}

// ------------------------------------------------------------------------------------------------
// Get the memory requirements of the scene
void Importer::GetMemoryRequirements(aiMemoryInfo& in) const {
    ai_assert(nullptr != pimpl);
    
    in = aiMemoryInfo();
    aiScene* mScene = pimpl->mScene;

    // return if we have no scene loaded
    if (!mScene)
        return;

    in.total = sizeof(aiScene);

    // add all meshes
    for (unsigned int i = 0; i < mScene->mNumMeshes;++i) {
        in.meshes += sizeof(aiMesh);
        if (mScene->mMeshes[i]->HasPositions()) {
            in.meshes += sizeof(aiVector3D) * mScene->mMeshes[i]->mNumVertices;
        }

        if (mScene->mMeshes[i]->HasNormals()) {
            in.meshes += sizeof(aiVector3D) * mScene->mMeshes[i]->mNumVertices;
        }

        if (mScene->mMeshes[i]->HasTangentsAndBitangents()) {
            in.meshes += sizeof(aiVector3D) * mScene->mMeshes[i]->mNumVertices * 2;
        }

        for (unsigned int a = 0; a < AI_MAX_NUMBER_OF_COLOR_SETS;++a) {
            if (mScene->mMeshes[i]->HasVertexColors(a)) {
                in.meshes += sizeof(aiColor4D) * mScene->mMeshes[i]->mNumVertices;
            } else {
                break;
            }
        }
        for (unsigned int a = 0; a < AI_MAX_NUMBER_OF_TEXTURECOORDS;++a) {
            if (mScene->mMeshes[i]->HasTextureCoords(a)) {
                in.meshes += sizeof(aiVector3D) * mScene->mMeshes[i]->mNumVertices;
            } else {
                break;
            }
        }
        if (mScene->mMeshes[i]->HasBones()) {
            in.meshes += sizeof(void*) * mScene->mMeshes[i]->mNumBones;
            for (unsigned int p = 0; p < mScene->mMeshes[i]->mNumBones;++p) {
                in.meshes += sizeof(aiBone);
                in.meshes += mScene->mMeshes[i]->mBones[p]->mNumWeights * sizeof(aiVertexWeight);
            }
        }
        in.meshes += (sizeof(aiFace) + 3 * sizeof(unsigned int))*mScene->mMeshes[i]->mNumFaces;
    }
    in.total += in.meshes;

    // add all embedded textures
    for (unsigned int i = 0; i < mScene->mNumTextures;++i) {
        const aiTexture* pc = mScene->mTextures[i];
        in.textures += sizeof(aiTexture);
        if (pc->mHeight) {
            in.textures += 4 * pc->mHeight * pc->mWidth;
        } else {
            in.textures += pc->mWidth;
        }
    }
    in.total += in.textures;

    // add all animations
    for (unsigned int i = 0; i < mScene->mNumAnimations;++i) {
        const aiAnimation* pc = mScene->mAnimations[i];
        in.animations += sizeof(aiAnimation);

        // add all bone anims
        for (unsigned int a = 0; a < pc->mNumChannels; ++a) {
            const aiNodeAnim* pc2 = pc->mChannels[i];
            in.animations += sizeof(aiNodeAnim);
            in.animations += pc2->mNumPositionKeys * sizeof(aiVectorKey);
            in.animations += pc2->mNumScalingKeys * sizeof(aiVectorKey);
            in.animations += pc2->mNumRotationKeys * sizeof(aiQuatKey);
        }
    }
    in.total += in.animations;

    // add all cameras and all lights
    in.total += in.cameras = sizeof(aiCamera) *  mScene->mNumCameras;
    in.total += in.lights  = sizeof(aiLight)  *  mScene->mNumLights;

    // add all nodes
    AddNodeWeight(in.nodes,mScene->mRootNode);
    in.total += in.nodes;

    // add all materials
    for (unsigned int i = 0; i < mScene->mNumMaterials;++i) {
        const aiMaterial* pc = mScene->mMaterials[i];
        in.materials += sizeof(aiMaterial);
        in.materials += pc->mNumAllocated * sizeof(void*);

        for (unsigned int a = 0; a < pc->mNumProperties;++a) {
            in.materials += pc->mProperties[a]->mDataLength;
        }
    }

    in.total += in.materials;
}
