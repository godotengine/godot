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

/** @file MaterialSystem.h
 *  Now that #MaterialHelper is gone, this file only contains some
 *  internal material utility functions.
 */
#ifndef AI_MATERIALSYSTEM_H_INC
#define AI_MATERIALSYSTEM_H_INC

#include <stdint.h>

struct aiMaterial;

namespace Assimp    {

// ------------------------------------------------------------------------------
/** Computes a hash (hopefully unique) from all material properties
 *  The hash value reflects the current property state, so if you add any
 *  property and call this method again, the resulting hash value will be
 *  different. The hash is not persistent across different builds and platforms.
 *
 *  @param  includeMatName Set to 'true' to take all properties with
 *    '?' as initial character in their name into account.
 *    Currently #AI_MATKEY_NAME is the only example.
 *  @return 32 Bit jash value for the material
 */
uint32_t ComputeMaterialHash(const aiMaterial* mat, bool includeMatName = false);


} // ! namespace Assimp

#endif //!! AI_MATERIALSYSTEM_H_INC
