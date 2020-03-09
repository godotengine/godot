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

/** @file importerdesc.h
 *  @brief #aiImporterFlags, aiImporterDesc implementation.
 */
#pragma once
#ifndef AI_IMPORTER_DESC_H_INC
#define AI_IMPORTER_DESC_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif


/** Mixed set of flags for #aiImporterDesc, indicating some features
  *  common to many importers*/
enum aiImporterFlags {
    /** Indicates that there is a textual encoding of the
     *  file format; and that it is supported.*/
    aiImporterFlags_SupportTextFlavour = 0x1,

    /** Indicates that there is a binary encoding of the
     *  file format; and that it is supported.*/
    aiImporterFlags_SupportBinaryFlavour = 0x2,

    /** Indicates that there is a compressed encoding of the
     *  file format; and that it is supported.*/
    aiImporterFlags_SupportCompressedFlavour = 0x4,

    /** Indicates that the importer reads only a very particular
      * subset of the file format. This happens commonly for
      * declarative or procedural formats which cannot easily
      * be mapped to #aiScene */
    aiImporterFlags_LimitedSupport = 0x8,

    /** Indicates that the importer is highly experimental and
      * should be used with care. This only happens for trunk
      * (i.e. SVN) versions, experimental code is not included
      * in releases. */
    aiImporterFlags_Experimental = 0x10
};


/** Meta information about a particular importer. Importers need to fill
 *  this structure, but they can freely decide how talkative they are.
 *  A common use case for loader meta info is a user interface
 *  in which the user can choose between various import/export file
 *  formats. Building such an UI by hand means a lot of maintenance
 *  as importers/exporters are added to Assimp, so it might be useful
 *  to have a common mechanism to query some rough importer
 *  characteristics. */
struct aiImporterDesc {
    /** Full name of the importer (i.e. Blender3D importer)*/
    const char* mName;

    /** Original author (left blank if unknown or whole assimp team) */
    const char* mAuthor;

    /** Current maintainer, left blank if the author maintains */
    const char* mMaintainer;

    /** Implementation comments, i.e. unimplemented features*/
    const char* mComments;

    /** These flags indicate some characteristics common to many
        importers. */
    unsigned int mFlags;

    /** Minimum format version that can be loaded im major.minor format,
        both are set to 0 if there is either no version scheme
        or if the loader doesn't care. */
    unsigned int mMinMajor;
    unsigned int mMinMinor;

    /** Maximum format version that can be loaded im major.minor format,
        both are set to 0 if there is either no version scheme
        or if the loader doesn't care. Loaders that expect to be
        forward-compatible to potential future format versions should
        indicate  zero, otherwise they should specify the current
        maximum version.*/
    unsigned int mMaxMajor;
    unsigned int mMaxMinor;

    /** List of file extensions this importer can handle.
        List entries are separated by space characters.
        All entries are lower case without a leading dot (i.e.
        "xml dae" would be a valid value. Note that multiple
        importers may respond to the same file extension -
        assimp calls all importers in the order in which they
        are registered and each importer gets the opportunity
        to load the file until one importer "claims" the file. Apart
        from file extension checks, importers typically use
        other methods to quickly reject files (i.e. magic
        words) so this does not mean that common or generic
        file extensions such as XML would be tediously slow. */
    const char* mFileExtensions;
};

/** \brief  Returns the Importer description for a given extension.

Will return a NULL-pointer if no assigned importer desc. was found for the given extension
    \param  extension   [in] The extension to look for
    \return A pointer showing to the ImporterDesc, \see aiImporterDesc.
*/
ASSIMP_API const C_STRUCT aiImporterDesc* aiGetImporterDesc( const char *extension );

#endif // AI_IMPORTER_DESC_H_INC
