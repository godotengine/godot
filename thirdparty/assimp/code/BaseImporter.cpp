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

/** @file  BaseImporter.cpp
 *  @brief Implementation of BaseImporter
 */

#include <assimp/BaseImporter.h>
#include <assimp/ParsingUtils.h>
#include "FileSystemFilter.h"
#include "Importer.h"
#include <assimp/ByteSwapper.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/importerdesc.h>

#include <ios>
#include <list>
#include <memory>
#include <sstream>
#include <cctype>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
BaseImporter::BaseImporter() AI_NO_EXCEPT
: m_progress() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
BaseImporter::~BaseImporter() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Imports the given file and returns the imported data.
aiScene* BaseImporter::ReadFile(const Importer* pImp, const std::string& pFile, IOSystem* pIOHandler) {
    m_progress = pImp->GetProgressHandler();
    if (nullptr == m_progress) {
        return nullptr;
    }

    ai_assert(m_progress);

    // Gather configuration properties for this run
    SetupProperties( pImp );

    // Construct a file system filter to improve our success ratio at reading external files
    FileSystemFilter filter(pFile,pIOHandler);

    // create a scene object to hold the data
    std::unique_ptr<aiScene> sc(new aiScene());

    // dispatch importing
    try
    {
        InternReadFile( pFile, sc.get(), &filter);

    } catch( const std::exception& err )    {
        // extract error description
        m_ErrorText = err.what();
        ASSIMP_LOG_ERROR(m_ErrorText);
        return nullptr;
    }

    // return what we gathered from the import.
    return sc.release();
}

// ------------------------------------------------------------------------------------------------
void BaseImporter::SetupProperties(const Importer* /*pImp*/)
{
    // the default implementation does nothing
}

// ------------------------------------------------------------------------------------------------
void BaseImporter::GetExtensionList(std::set<std::string>& extensions) {
    const aiImporterDesc* desc = GetInfo();
    ai_assert(desc != nullptr);

    const char* ext = desc->mFileExtensions;
    ai_assert(ext != nullptr );

    const char* last = ext;
    do {
        if (!*ext || *ext == ' ') {
            extensions.insert(std::string(last,ext-last));
            ai_assert(ext-last > 0);
            last = ext;
            while(*last == ' ') {
                ++last;
            }
        }
    }
    while(*ext++);
}

// ------------------------------------------------------------------------------------------------
/*static*/ bool BaseImporter::SearchFileHeaderForToken( IOSystem* pIOHandler,
    const std::string&  pFile,
    const char**        tokens,
    unsigned int        numTokens,
    unsigned int        searchBytes /* = 200 */,
    bool                tokensSol /* false */,
    bool                noAlphaBeforeTokens /* false */)
{
    ai_assert( nullptr != tokens );
    ai_assert( 0 != numTokens );
    ai_assert( 0 != searchBytes);

    if ( nullptr == pIOHandler ) {
        return false;
    }

    std::unique_ptr<IOStream> pStream (pIOHandler->Open(pFile));
    if (pStream.get() ) {
        // read 200 characters from the file
        std::unique_ptr<char[]> _buffer (new char[searchBytes+1 /* for the '\0' */]);
        char *buffer( _buffer.get() );
        const size_t read( pStream->Read(buffer,1,searchBytes) );
        if( 0 == read ) {
            return false;
        }

        for( size_t i = 0; i < read; ++i ) {
            buffer[ i ] = static_cast<char>( ::tolower( buffer[ i ] ) );
        }

        // It is not a proper handling of unicode files here ...
        // ehm ... but it works in most cases.
        char* cur = buffer,*cur2 = buffer,*end = &buffer[read];
        while (cur != end)  {
            if( *cur ) {
                *cur2++ = *cur;
            }
            ++cur;
        }
        *cur2 = '\0';

        std::string token;
        for (unsigned int i = 0; i < numTokens; ++i ) {
            ai_assert( nullptr != tokens[i] );
            const size_t len( strlen( tokens[ i ] ) );
            token.clear();
            const char *ptr( tokens[ i ] );
            for ( size_t tokIdx = 0; tokIdx < len; ++tokIdx ) {
                token.push_back( static_cast<char>( tolower( *ptr ) ) );
                ++ptr;
            }
            const char* r = strstr( buffer, token.c_str() );
            if( !r ) {
                continue;
            }
            // We need to make sure that we didn't accidentially identify the end of another token as our token,
            // e.g. in a previous version the "gltf " present in some gltf files was detected as "f "
            if (noAlphaBeforeTokens && (r != buffer && isalpha(r[-1]))) {
                continue;
            }
            // We got a match, either we don't care where it is, or it happens to
            // be in the beginning of the file / line
            if (!tokensSol || r == buffer || r[-1] == '\r' || r[-1] == '\n') {
                ASSIMP_LOG_DEBUG_F( "Found positive match for header keyword: ", tokens[i] );
                return true;
            }
        }
    }

    return false;
}

// ------------------------------------------------------------------------------------------------
// Simple check for file extension
/*static*/ bool BaseImporter::SimpleExtensionCheck (const std::string& pFile,
    const char* ext0,
    const char* ext1,
    const char* ext2)
{
    std::string::size_type pos = pFile.find_last_of('.');

    // no file extension - can't read
    if( pos == std::string::npos)
        return false;

    const char* ext_real = & pFile[ pos+1 ];
    if( !ASSIMP_stricmp(ext_real,ext0) )
        return true;

    // check for other, optional, file extensions
    if (ext1 && !ASSIMP_stricmp(ext_real,ext1))
        return true;

    if (ext2 && !ASSIMP_stricmp(ext_real,ext2))
        return true;

    return false;
}

// ------------------------------------------------------------------------------------------------
// Get file extension from path
std::string BaseImporter::GetExtension( const std::string& file ) {
    std::string::size_type pos = file.find_last_of('.');

    // no file extension at all
    if (pos == std::string::npos) {
        return "";
    }


    // thanks to Andy Maloney for the hint
    std::string ret = file.substr( pos + 1 );
    std::transform( ret.begin(), ret.end(), ret.begin(), ToLower<char>);

    return ret;
}

// ------------------------------------------------------------------------------------------------
// Check for magic bytes at the beginning of the file.
/* static */ bool BaseImporter::CheckMagicToken(IOSystem* pIOHandler, const std::string& pFile,
    const void* _magic, unsigned int num, unsigned int offset, unsigned int size)
{
    ai_assert( size <= 16 );
    ai_assert( _magic );

    if (!pIOHandler) {
        return false;
    }
    union {
        const char* magic;
        const uint16_t* magic_u16;
        const uint32_t* magic_u32;
    };
    magic = reinterpret_cast<const char*>(_magic);
    std::unique_ptr<IOStream> pStream (pIOHandler->Open(pFile));
    if (pStream.get() ) {

        // skip to offset
        pStream->Seek(offset,aiOrigin_SET);

        // read 'size' characters from the file
        union {
            char data[16];
            uint16_t data_u16[8];
            uint32_t data_u32[4];
        };
        if(size != pStream->Read(data,1,size)) {
            return false;
        }

        for (unsigned int i = 0; i < num; ++i) {
            // also check against big endian versions of tokens with size 2,4
            // that's just for convenience, the chance that we cause conflicts
            // is quite low and it can save some lines and prevent nasty bugs
            if (2 == size) {
                uint16_t rev = *magic_u16;
                ByteSwap::Swap(&rev);
                if (data_u16[0] == *magic_u16 || data_u16[0] == rev) {
                    return true;
                }
            }
            else if (4 == size) {
                uint32_t rev = *magic_u32;
                ByteSwap::Swap(&rev);
                if (data_u32[0] == *magic_u32 || data_u32[0] == rev) {
                    return true;
                }
            }
            else {
                // any length ... just compare
                if(!memcmp(magic,data,size)) {
                    return true;
                }
            }
            magic += size;
        }
    }
    return false;
}

#include "../contrib/utf8cpp/source/utf8.h"

// ------------------------------------------------------------------------------------------------
// Convert to UTF8 data
void BaseImporter::ConvertToUTF8(std::vector<char>& data)
{
    //ConversionResult result;
    if(data.size() < 8) {
        throw DeadlyImportError("File is too small");
    }

    // UTF 8 with BOM
    if((uint8_t)data[0] == 0xEF && (uint8_t)data[1] == 0xBB && (uint8_t)data[2] == 0xBF) {
        ASSIMP_LOG_DEBUG("Found UTF-8 BOM ...");

        std::copy(data.begin()+3,data.end(),data.begin());
        data.resize(data.size()-3);
        return;
    }
    
    
    // UTF 32 BE with BOM
    if(*((uint32_t*)&data.front()) == 0xFFFE0000) {

        // swap the endianness ..
        for(uint32_t* p = (uint32_t*)&data.front(), *end = (uint32_t*)&data.back(); p <= end; ++p) {
            AI_SWAP4P(p);
        }
    }

    // UTF 32 LE with BOM
    if(*((uint32_t*)&data.front()) == 0x0000FFFE) {
        ASSIMP_LOG_DEBUG("Found UTF-32 BOM ...");

        std::vector<char> output;
        int *ptr = (int*)&data[ 0 ];
        int *end = ptr + ( data.size() / sizeof(int) ) +1;
        utf8::utf32to8( ptr, end, back_inserter(output));
        return;
    }

    // UTF 16 BE with BOM
    if(*((uint16_t*)&data.front()) == 0xFFFE) {

        // swap the endianness ..
        for(uint16_t* p = (uint16_t*)&data.front(), *end = (uint16_t*)&data.back(); p <= end; ++p) {
            ByteSwap::Swap2(p);
        }
    }

    // UTF 16 LE with BOM
    if(*((uint16_t*)&data.front()) == 0xFEFF) {
        ASSIMP_LOG_DEBUG("Found UTF-16 BOM ...");

        std::vector<unsigned char> output;
        utf8::utf16to8(data.begin(), data.end(), back_inserter(output));
        return;
    }
}

// ------------------------------------------------------------------------------------------------
// Convert to UTF8 data to ISO-8859-1
void BaseImporter::ConvertUTF8toISO8859_1(std::string& data)
{
    size_t size = data.size();
    size_t i = 0, j = 0;

    while(i < size) {
        if ((unsigned char) data[i] < (size_t) 0x80) {
            data[j] = data[i];
        } else if(i < size - 1) {
            if((unsigned char) data[i] == 0xC2) {
                data[j] = data[++i];
            } else if((unsigned char) data[i] == 0xC3) {
                data[j] = ((unsigned char) data[++i] + 0x40);
            } else {
                std::stringstream stream;
                stream << "UTF8 code " << std::hex << data[i] << data[i + 1] << " can not be converted into ISA-8859-1.";
                ASSIMP_LOG_ERROR( stream.str() );

                data[j++] = data[i++];
                data[j] = data[i];
            }
        } else {
            ASSIMP_LOG_ERROR("UTF8 code but only one character remaining");

            data[j] = data[i];
        }

        i++; j++;
    }

    data.resize(j);
}

// ------------------------------------------------------------------------------------------------
void BaseImporter::TextFileToBuffer(IOStream* stream,
    std::vector<char>& data,
    TextFileMode mode)
{
    ai_assert(nullptr != stream);

    const size_t fileSize = stream->FileSize();
    if (mode == FORBID_EMPTY) {
        if(!fileSize) {
            throw DeadlyImportError("File is empty");
        }
    }

    data.reserve(fileSize+1);
    data.resize(fileSize);
    if(fileSize > 0) {
        if(fileSize != stream->Read( &data[0], 1, fileSize)) {
            throw DeadlyImportError("File read error");
        }

        ConvertToUTF8(data);
    }

    // append a binary zero to simplify string parsing
    data.push_back(0);
}

// ------------------------------------------------------------------------------------------------
namespace Assimp {
    // Represents an import request
    struct LoadRequest {
        LoadRequest(const std::string& _file, unsigned int _flags,const BatchLoader::PropertyMap* _map, unsigned int _id)
        : file(_file)
        , flags(_flags)
        , refCnt(1)
        , scene(NULL)
        , loaded(false)
        , id(_id) {
            if ( _map ) {
                map = *_map;
            }
        }

        bool operator== ( const std::string& f ) const {
            return file == f;
        }

        const std::string        file;
        unsigned int             flags;
        unsigned int             refCnt;
        aiScene                 *scene;
        bool                     loaded;
        BatchLoader::PropertyMap map;
        unsigned int             id;
    };
}

// ------------------------------------------------------------------------------------------------
// BatchLoader::pimpl data structure
struct Assimp::BatchData {
    BatchData( IOSystem* pIO, bool validate )
    : pIOSystem( pIO )
    , pImporter( nullptr )
    , next_id(0xffff)
    , validate( validate ) {
        ai_assert( nullptr != pIO );
        
        pImporter = new Importer();
        pImporter->SetIOHandler( pIO );
    }

    ~BatchData() {
        pImporter->SetIOHandler( nullptr ); /* get pointer back into our possession */
        delete pImporter;
    }

    // IO system to be used for all imports
    IOSystem* pIOSystem;

    // Importer used to load all meshes
    Importer* pImporter;

    // List of all imports
    std::list<LoadRequest> requests;

    // Base path
    std::string pathBase;

    // Id for next item
    unsigned int next_id;

    // Validation enabled state
    bool validate;
};

typedef std::list<LoadRequest>::iterator LoadReqIt;

// ------------------------------------------------------------------------------------------------
BatchLoader::BatchLoader(IOSystem* pIO, bool validate ) {
    ai_assert(nullptr != pIO);

    m_data = new BatchData( pIO, validate );
}

// ------------------------------------------------------------------------------------------------
BatchLoader::~BatchLoader()
{
    // delete all scenes what have not been polled by the user
    for ( LoadReqIt it = m_data->requests.begin();it != m_data->requests.end(); ++it) {
        delete (*it).scene;
    }
    delete m_data;
}

// ------------------------------------------------------------------------------------------------
void BatchLoader::setValidation( bool enabled ) {
    m_data->validate = enabled;
}

// ------------------------------------------------------------------------------------------------
bool BatchLoader::getValidation() const {
    return m_data->validate;
}

// ------------------------------------------------------------------------------------------------
unsigned int BatchLoader::AddLoadRequest(const std::string& file,
    unsigned int steps /*= 0*/, const PropertyMap* map /*= NULL*/)
{
    ai_assert(!file.empty());

    // check whether we have this loading request already
    for ( LoadReqIt it = m_data->requests.begin();it != m_data->requests.end(); ++it)  {
        // Call IOSystem's path comparison function here
        if ( m_data->pIOSystem->ComparePaths((*it).file,file)) {
            if (map) {
                if ( !( ( *it ).map == *map ) ) {
                    continue;
                }
            }
            else if ( !( *it ).map.empty() ) {
                continue;
            }

            (*it).refCnt++;
            return (*it).id;
        }
    }

    // no, we don't have it. So add it to the queue ...
    m_data->requests.push_back(LoadRequest(file,steps,map, m_data->next_id));
    return m_data->next_id++;
}

// ------------------------------------------------------------------------------------------------
aiScene* BatchLoader::GetImport( unsigned int which )
{
    for ( LoadReqIt it = m_data->requests.begin();it != m_data->requests.end(); ++it) {
        if ((*it).id == which && (*it).loaded)  {
            aiScene* sc = (*it).scene;
            if (!(--(*it).refCnt))  {
                m_data->requests.erase(it);
            }
            return sc;
        }
    }
    return nullptr;
}

// ------------------------------------------------------------------------------------------------
void BatchLoader::LoadAll()
{
    // no threaded implementation for the moment
    for ( LoadReqIt it = m_data->requests.begin();it != m_data->requests.end(); ++it) {
        // force validation in debug builds
        unsigned int pp = (*it).flags;
        if ( m_data->validate ) {
            pp |= aiProcess_ValidateDataStructure;
        }

        // setup config properties if necessary
        ImporterPimpl* pimpl = m_data->pImporter->Pimpl();
        pimpl->mFloatProperties  = (*it).map.floats;
        pimpl->mIntProperties    = (*it).map.ints;
        pimpl->mStringProperties = (*it).map.strings;
        pimpl->mMatrixProperties = (*it).map.matrices;

        if (!DefaultLogger::isNullLogger())
        {
            ASSIMP_LOG_INFO("%%% BEGIN EXTERNAL FILE %%%");
            ASSIMP_LOG_INFO_F("File: ", (*it).file);
        }
        m_data->pImporter->ReadFile((*it).file,pp);
        (*it).scene = m_data->pImporter->GetOrphanedScene();
        (*it).loaded = true;

        ASSIMP_LOG_INFO("%%% END EXTERNAL FILE %%%");
    }
}
