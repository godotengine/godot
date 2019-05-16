/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2008, assimp team
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

/** @file FileSystemFilter.h
 *  Implements a filter system to filter calls to Exists() and Open()
 *  in order to improve the success rate of file opening ...
 */
#pragma once
#ifndef AI_FILESYSTEMFILTER_H_INC
#define AI_FILESYSTEMFILTER_H_INC

#include <assimp/IOSystem.hpp>
#include <assimp/DefaultLogger.hpp>
#include <assimp/fast_atof.h>
#include <assimp/ParsingUtils.h>

namespace Assimp    {

inline bool IsHex(char s) {
    return (s>='0' && s<='9') || (s>='a' && s<='f') || (s>='A' && s<='F');
}

// ---------------------------------------------------------------------------
/** File system filter
 */
class FileSystemFilter : public IOSystem
{
public:
    /** Constructor. */
    FileSystemFilter(const std::string& file, IOSystem* old)
    : mWrapped  (old)
    , mSrc_file(file)
    , mSep(mWrapped->getOsSeparator()) {
        ai_assert(nullptr != mWrapped);

        // Determine base directory
        mBase = mSrc_file;
        std::string::size_type ss2;
        if (std::string::npos != (ss2 = mBase.find_last_of("\\/")))  {
            mBase.erase(ss2,mBase.length()-ss2);
        } else {
            mBase = "";
        }

        // make sure the directory is terminated properly
        char s;

        if ( mBase.empty() ) {
            mBase = ".";
            mBase += getOsSeparator();
        } else if ((s = *(mBase.end()-1)) != '\\' && s != '/') {
            mBase += getOsSeparator();
        }

        DefaultLogger::get()->info("Import root directory is \'" + mBase + "\'");
    }

    /** Destructor. */
    ~FileSystemFilter() {
        // empty
    }

    // -------------------------------------------------------------------
    /** Tests for the existence of a file at the given path. */
    bool Exists( const char* pFile) const {
        ai_assert( nullptr != mWrapped );
        
        std::string tmp = pFile;

        // Currently this IOSystem is also used to open THE ONE FILE.
        if (tmp != mSrc_file)    {
            BuildPath(tmp);
            Cleanup(tmp);
        }

        return mWrapped->Exists(tmp);
    }

    // -------------------------------------------------------------------
    /** Returns the directory separator. */
    char getOsSeparator() const {
        return mSep;
    }

    // -------------------------------------------------------------------
    /** Open a new file with a given path. */
    IOStream* Open( const char* pFile, const char* pMode = "rb") {
        ai_assert( nullptr != mWrapped );
        if ( nullptr == pFile || nullptr == pMode ) {
            return nullptr;
        }
        
        ai_assert( nullptr != pFile );
        ai_assert( nullptr != pMode );

        // First try the unchanged path
        IOStream* s = mWrapped->Open(pFile,pMode);

        if (nullptr == s) {
            std::string tmp = pFile;

            // Try to convert between absolute and relative paths
            BuildPath(tmp);
            s = mWrapped->Open(tmp,pMode);

            if (nullptr == s) {
                // Finally, look for typical issues with paths
                // and try to correct them. This is our last
                // resort.
                tmp = pFile;
                Cleanup(tmp);
                BuildPath(tmp);
                s = mWrapped->Open(tmp,pMode);
            }
        }

        return s;
    }

    // -------------------------------------------------------------------
    /** Closes the given file and releases all resources associated with it. */
    void Close( IOStream* pFile) {
        ai_assert( nullptr != mWrapped );
        return mWrapped->Close(pFile);
    }

    // -------------------------------------------------------------------
    /** Compare two paths */
    bool ComparePaths (const char* one, const char* second) const {
        ai_assert( nullptr != mWrapped );
        return mWrapped->ComparePaths (one,second);
    }

    // -------------------------------------------------------------------
    /** Pushes a new directory onto the directory stack. */
    bool PushDirectory(const std::string &path ) {
        ai_assert( nullptr != mWrapped );
        return mWrapped->PushDirectory(path);
    }

    // -------------------------------------------------------------------
    /** Returns the top directory from the stack. */
    const std::string &CurrentDirectory() const {
        ai_assert( nullptr != mWrapped );
        return mWrapped->CurrentDirectory();
    }

    // -------------------------------------------------------------------
    /** Returns the number of directories stored on the stack. */
    size_t StackSize() const {
        ai_assert( nullptr != mWrapped );
        return mWrapped->StackSize();
    }

    // -------------------------------------------------------------------
    /** Pops the top directory from the stack. */
    bool PopDirectory() {
        ai_assert( nullptr != mWrapped );
        return mWrapped->PopDirectory();
    }

    // -------------------------------------------------------------------
    /** Creates an new directory at the given path. */
    bool CreateDirectory(const std::string &path) {
        ai_assert( nullptr != mWrapped );
        return mWrapped->CreateDirectory(path);
    }

    // -------------------------------------------------------------------
    /** Will change the current directory to the given path. */
    bool ChangeDirectory(const std::string &path) {
        ai_assert( nullptr != mWrapped );
        return mWrapped->ChangeDirectory(path);
    }

    // -------------------------------------------------------------------
    /** Delete file. */
    bool DeleteFile(const std::string &file) {
        ai_assert( nullptr != mWrapped );
        return mWrapped->DeleteFile(file);
    }

private:
    // -------------------------------------------------------------------
    /** Build a valid path from a given relative or absolute path.
     */
    void BuildPath (std::string& in) const {
        ai_assert( nullptr != mWrapped );
        // if we can already access the file, great.
        if (in.length() < 3 || mWrapped->Exists(in)) {
            return;
        }

        // Determine whether this is a relative path (Windows-specific - most assets are packaged on Windows).
        if (in[1] != ':') {

            // append base path and try
            const std::string tmp = mBase + in;
            if (mWrapped->Exists(tmp)) {
                in = tmp;
                return;
            }
        }

        // Chop of the file name and look in the model directory, if
        // this fails try all sub paths of the given path, i.e.
        // if the given path is foo/bar/something.lwo, try
        // <base>/something.lwo
        // <base>/bar/something.lwo
        // <base>/foo/bar/something.lwo
        std::string::size_type pos = in.rfind('/');
        if (std::string::npos == pos) {
            pos = in.rfind('\\');
        }

        if (std::string::npos != pos)   {
            std::string tmp;
            std::string::size_type last_dirsep = std::string::npos;

            while(true) {
                tmp = mBase;
                tmp += mSep;

                std::string::size_type dirsep = in.rfind('/', last_dirsep);
                if (std::string::npos == dirsep) {
                    dirsep = in.rfind('\\', last_dirsep);
                }

                if (std::string::npos == dirsep || dirsep == 0) {
                    // we did try this already.
                    break;
                }

                last_dirsep = dirsep-1;

                tmp += in.substr(dirsep+1, in.length()-pos);
                if (mWrapped->Exists(tmp)) {
                    in = tmp;
                    return;
                }
            }
        }

        // hopefully the underlying file system has another few tricks to access this file ...
    }

    // -------------------------------------------------------------------
    /** Cleanup the given path
     */
    void Cleanup (std::string& in) const {
        if(in.empty()) {
            return;
        }

        // Remove a very common issue when we're parsing file names: spaces at the
        // beginning of the path.
        char last = 0;
        std::string::iterator it = in.begin();
        while (IsSpaceOrNewLine( *it ))++it;
        if (it != in.begin()) {
            in.erase(in.begin(),it+1);
        }

        const char separator = getOsSeparator();
        for (it = in.begin(); it != in.end(); ++it) {
            // Exclude :// and \\, which remain untouched.
            // https://sourceforge.net/tracker/?func=detail&aid=3031725&group_id=226462&atid=1067632
            if ( !strncmp(&*it, "://", 3 )) {
                it += 3;
                continue;
            }
            if (it == in.begin() && !strncmp(&*it, "\\\\", 2)) {
                it += 2;
                continue;
            }

            // Cleanup path delimiters
            if (*it == '/' || (*it) == '\\') {
                *it = separator;

                // And we're removing double delimiters, frequent issue with
                // incorrectly composited paths ...
                if (last == *it) {
                    it = in.erase(it);
                    --it;
                }
            } else if (*it == '%' && in.end() - it > 2) {
                // Hex sequence in URIs
                if( IsHex((&*it)[0]) && IsHex((&*it)[1]) ) {
                    *it = HexOctetToDecimal(&*it);
                    it = in.erase(it+1,it+2);
                    --it;
                }
            }

            last = *it;
        }
    }

private:
    IOSystem *mWrapped;
    std::string mSrc_file, mBase;
    char mSep;
};

} //!ns Assimp

#endif //AI_DEFAULTIOSYSTEM_H_INC
