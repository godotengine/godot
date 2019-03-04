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
/** @file Default implementation of IOSystem using the standard C file functions */

#include <assimp/StringComparison.h>

#include <assimp/DefaultIOSystem.h>
#include <assimp/DefaultIOStream.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/ai_assert.h>
#include <stdlib.h>

#ifdef __unix__
#include <sys/param.h>
#include <stdlib.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

using namespace Assimp;

// maximum path length
// XXX http://insanecoding.blogspot.com/2007/11/pathmax-simply-isnt.html
#ifdef PATH_MAX
#   define PATHLIMIT PATH_MAX
#else
#   define PATHLIMIT 4096
#endif

// ------------------------------------------------------------------------------------------------
// Tests for the existence of a file at the given path.
bool DefaultIOSystem::Exists( const char* pFile) const
{
#ifdef _WIN32
    wchar_t fileName16[PATHLIMIT];

#ifndef WindowsStore
    bool isUnicode = IsTextUnicode(pFile, static_cast<int>(strlen(pFile)), NULL) != 0;
    if (isUnicode) {

        MultiByteToWideChar(CP_UTF8, MB_PRECOMPOSED, pFile, -1, fileName16, PATHLIMIT);
        struct __stat64 filestat;
        if (0 != _wstat64(fileName16, &filestat)) {
            return false;
        }
    } else {
#endif
        FILE* file = ::fopen(pFile, "rb");
        if (!file)
            return false;

        ::fclose(file);
#ifndef WindowsStore
    }
#endif
#else
    FILE* file = ::fopen( pFile, "rb");
    if( !file)
        return false;

    ::fclose( file);
#endif
    return true;
}

// ------------------------------------------------------------------------------------------------
// Open a new file with a given path.
IOStream* DefaultIOSystem::Open( const char* strFile, const char* strMode)
{
    ai_assert(NULL != strFile);
    ai_assert(NULL != strMode);
    FILE* file;
#ifdef _WIN32
    wchar_t fileName16[PATHLIMIT];
#ifndef WindowsStore
    bool isUnicode = IsTextUnicode(strFile, static_cast<int>(strlen(strFile)), NULL) != 0;
    if (isUnicode) {
        MultiByteToWideChar(CP_UTF8, MB_PRECOMPOSED, strFile, -1, fileName16, PATHLIMIT);
        std::string mode8(strMode);
        file = ::_wfopen(fileName16, std::wstring(mode8.begin(), mode8.end()).c_str());
    } else {
#endif
        file = ::fopen(strFile, strMode);
#ifndef WindowsStore
    }
#endif
#else
    file = ::fopen(strFile, strMode);
#endif
    if (nullptr == file)
        return nullptr;

    return new DefaultIOStream(file, (std::string) strFile);
}

// ------------------------------------------------------------------------------------------------
// Closes the given file and releases all resources associated with it.
void DefaultIOSystem::Close( IOStream* pFile)
{
    delete pFile;
}

// ------------------------------------------------------------------------------------------------
// Returns the operation specific directory separator
char DefaultIOSystem::getOsSeparator() const
{
#ifndef _WIN32
    return '/';
#else
    return '\\';
#endif
}

// ------------------------------------------------------------------------------------------------
// IOSystem default implementation (ComparePaths isn't a pure virtual function)
bool IOSystem::ComparePaths (const char* one, const char* second) const
{
    return !ASSIMP_stricmp(one,second);
}

// ------------------------------------------------------------------------------------------------
// Convert a relative path into an absolute path
inline static void MakeAbsolutePath (const char* in, char* _out)
{
    ai_assert(in && _out);
#if defined( _MSC_VER ) || defined( __MINGW32__ )
#ifndef WindowsStore
    bool isUnicode = IsTextUnicode(in, static_cast<int>(strlen(in)), NULL) != 0;
    if (isUnicode) {
        wchar_t out16[PATHLIMIT];
        wchar_t in16[PATHLIMIT];
        MultiByteToWideChar(CP_UTF8, MB_PRECOMPOSED, in, -1, out16, PATHLIMIT);
        wchar_t* ret = ::_wfullpath(out16, in16, PATHLIMIT);
        if (ret) {
            WideCharToMultiByte(CP_UTF8, MB_PRECOMPOSED, out16, -1, _out, PATHLIMIT, nullptr, nullptr);
        }
        if (!ret) {
            // preserve the input path, maybe someone else is able to fix
            // the path before it is accessed (e.g. our file system filter)
            ASSIMP_LOG_WARN_F("Invalid path: ", std::string(in));
            strcpy(_out, in);
        }

    } else {
#endif
        char* ret = :: _fullpath(_out, in, PATHLIMIT);
        if (!ret) {
            // preserve the input path, maybe someone else is able to fix
            // the path before it is accessed (e.g. our file system filter)
            ASSIMP_LOG_WARN_F("Invalid path: ", std::string(in));
            strcpy(_out, in);
        }
#ifndef WindowsStore
    }
#endif
#else
    // use realpath
    char* ret = realpath(in, _out);
    if(!ret) {
        // preserve the input path, maybe someone else is able to fix
        // the path before it is accessed (e.g. our file system filter)
        ASSIMP_LOG_WARN_F("Invalid path: ", std::string(in));
        strcpy(_out,in);
    }
#endif
}

// ------------------------------------------------------------------------------------------------
// DefaultIOSystem's more specialized implementation
bool DefaultIOSystem::ComparePaths (const char* one, const char* second) const
{
    // chances are quite good both paths are formatted identically,
    // so we can hopefully return here already
    if( !ASSIMP_stricmp(one,second) )
        return true;

    char temp1[PATHLIMIT];
    char temp2[PATHLIMIT];

    MakeAbsolutePath (one, temp1);
    MakeAbsolutePath (second, temp2);

    return !ASSIMP_stricmp(temp1,temp2);
}

// ------------------------------------------------------------------------------------------------
std::string DefaultIOSystem::fileName( const std::string &path )
{
    std::string ret = path;
    std::size_t last = ret.find_last_of("\\/");
    if (last != std::string::npos) ret = ret.substr(last + 1);
    return ret;
}

// ------------------------------------------------------------------------------------------------
std::string DefaultIOSystem::completeBaseName( const std::string &path )
{
    std::string ret = fileName(path);
    std::size_t pos = ret.find_last_of('.');
    if(pos != ret.npos) ret = ret.substr(0, pos);
    return ret;
}

// ------------------------------------------------------------------------------------------------
std::string DefaultIOSystem::absolutePath( const std::string &path )
{
    std::string ret = path;
    std::size_t last = ret.find_last_of("\\/");
    if (last != std::string::npos) ret = ret.substr(0, last);
    return ret;
}

// ------------------------------------------------------------------------------------------------

#undef PATHLIMIT
