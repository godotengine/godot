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

#ifdef _WIN32
static std::wstring Utf8ToWide(const char* in)
{
    int size = MultiByteToWideChar(CP_UTF8, 0, in, -1, nullptr, 0);
    // size includes terminating null; std::wstring adds null automatically
    std::wstring out(static_cast<size_t>(size) - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, in, -1, &out[0], size);
    return out;
}

static std::string WideToUtf8(const wchar_t* in)
{
    int size = WideCharToMultiByte(CP_UTF8, 0, in, -1, nullptr, 0, nullptr, nullptr);
    // size includes terminating null; std::string adds null automatically
    std::string out(static_cast<size_t>(size) - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, in, -1, &out[0], size, nullptr, nullptr);
    return out;
}
#endif

// ------------------------------------------------------------------------------------------------
// Tests for the existence of a file at the given path.
bool DefaultIOSystem::Exists(const char* pFile) const
{
#ifdef _WIN32
    struct __stat64 filestat;
    if (_wstat64(Utf8ToWide(pFile).c_str(), &filestat) != 0) {
        return false;
    }
#else
    FILE* file = ::fopen(pFile, "rb");
    if (!file)
        return false;

    ::fclose(file);
#endif
    return true;
}

// ------------------------------------------------------------------------------------------------
// Open a new file with a given path.
IOStream* DefaultIOSystem::Open(const char* strFile, const char* strMode)
{
    ai_assert(strFile != nullptr);
    ai_assert(strMode != nullptr);
    FILE* file;
#ifdef _WIN32
    file = ::_wfopen(Utf8ToWide(strFile).c_str(), Utf8ToWide(strMode).c_str());
#else
    file = ::fopen(strFile, strMode);
#endif
    if (!file)
        return nullptr;

    return new DefaultIOStream(file, strFile);
}

// ------------------------------------------------------------------------------------------------
// Closes the given file and releases all resources associated with it.
void DefaultIOSystem::Close(IOStream* pFile)
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
bool IOSystem::ComparePaths(const char* one, const char* second) const
{
    return !ASSIMP_stricmp(one, second);
}

// ------------------------------------------------------------------------------------------------
// Convert a relative path into an absolute path
inline static std::string MakeAbsolutePath(const char* in)
{
    ai_assert(in);
    std::string out;
#ifdef _WIN32
    wchar_t* ret = ::_wfullpath(nullptr, Utf8ToWide(in).c_str(), 0);
    if (ret) {
        out = WideToUtf8(ret);
        free(ret);
    }
#else
    char* ret = realpath(in, nullptr);
    if (ret) {
        out = ret;
        free(ret);
    }
#endif
    if (!ret) {
        // preserve the input path, maybe someone else is able to fix
        // the path before it is accessed (e.g. our file system filter)
        ASSIMP_LOG_WARN_F("Invalid path: ", std::string(in));
        out = in;
    }
    return out;
}

// ------------------------------------------------------------------------------------------------
// DefaultIOSystem's more specialized implementation
bool DefaultIOSystem::ComparePaths(const char* one, const char* second) const
{
    // chances are quite good both paths are formatted identically,
    // so we can hopefully return here already
    if (!ASSIMP_stricmp(one, second))
        return true;

    std::string temp1 = MakeAbsolutePath(one);
    std::string temp2 = MakeAbsolutePath(second);

    return !ASSIMP_stricmp(temp1, temp2);
}

// ------------------------------------------------------------------------------------------------
std::string DefaultIOSystem::fileName(const std::string& path)
{
    std::string ret = path;
    std::size_t last = ret.find_last_of("\\/");
    if (last != std::string::npos) ret = ret.substr(last + 1);
    return ret;
}

// ------------------------------------------------------------------------------------------------
std::string DefaultIOSystem::completeBaseName(const std::string& path)
{
    std::string ret = fileName(path);
    std::size_t pos = ret.find_last_of('.');
    if (pos != std::string::npos) ret = ret.substr(0, pos);
    return ret;
}

// ------------------------------------------------------------------------------------------------
std::string DefaultIOSystem::absolutePath(const std::string& path)
{
    std::string ret = path;
    std::size_t last = ret.find_last_of("\\/");
    if (last != std::string::npos) ret = ret.substr(0, last);
    return ret;
}

// ------------------------------------------------------------------------------------------------
