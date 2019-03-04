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

/** @file Declares a helper class, "CommentRemover", which can be
 *  used to remove comments (single and multi line) from a text file.
 */
#ifndef AI_REMOVE_COMMENTS_H_INC
#define AI_REMOVE_COMMENTS_H_INC


#include <assimp/defs.h>

namespace Assimp    {

// ---------------------------------------------------------------------------
/** \brief Helper class to remove single and multi line comments from a file
 *
 *  Some mesh formats like MD5 have comments that are quite similar
 *  to those in C or C++ so this code has been moved to a separate
 *  module.
 */
class ASSIMP_API CommentRemover
{
    // class cannot be instanced
    CommentRemover() {}

public:

    //! Remove single-line comments. The end of a line is
    //! expected to be either NL or CR or NLCR.
    //! \param szComment The start sequence of the comment, e.g. "//"
    //! \param szBuffer Buffer to work with
    //! \param chReplacement Character to be used as replacement
    //! for commented lines. By default this is ' '
    static void RemoveLineComments(const char* szComment,
        char* szBuffer, char chReplacement = ' ');

    //! Remove multi-line comments. The end of a line is
    //! expected to be either NL or CR or NLCR. Multi-line comments
    //! may not be nested (as in C).
    //! \param szCommentStart The start sequence of the comment, e.g. "/*"
    //! \param szCommentEnd The end sequence of the comment, e.g. "*/"
    //! \param szBuffer Buffer to work with
    //! \param chReplacement Character to be used as replacement
    //! for commented lines. By default this is ' '
    static void RemoveMultiLineComments(const char* szCommentStart,
        const char* szCommentEnd,char* szBuffer,
        char chReplacement = ' ');
};
} // ! Assimp

#endif // !! AI_REMOVE_COMMENTS_H_INC
