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

/** @file pbrmaterial.h
 *  @brief Defines the material system of the library
 */
#ifndef AI_PBRMATERIAL_H_INC
#define AI_PBRMATERIAL_H_INC

#define AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR "$mat.gltf.pbrMetallicRoughness.baseColorFactor", 0, 0
#define AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR "$mat.gltf.pbrMetallicRoughness.metallicFactor", 0, 0
#define AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR "$mat.gltf.pbrMetallicRoughness.roughnessFactor", 0, 0
#define AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_TEXTURE aiTextureType_DIFFUSE, 1
#define AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE aiTextureType_UNKNOWN, 0
#define AI_MATKEY_GLTF_ALPHAMODE "$mat.gltf.alphaMode", 0, 0
#define AI_MATKEY_GLTF_ALPHACUTOFF "$mat.gltf.alphaCutoff", 0, 0
#define AI_MATKEY_GLTF_PBRSPECULARGLOSSINESS "$mat.gltf.pbrSpecularGlossiness", 0, 0
#define AI_MATKEY_GLTF_PBRSPECULARGLOSSINESS_GLOSSINESS_FACTOR "$mat.gltf.pbrMetallicRoughness.glossinessFactor", 0, 0
#define AI_MATKEY_GLTF_UNLIT "$mat.gltf.unlit", 0, 0

#define _AI_MATKEY_GLTF_TEXTURE_TEXCOORD_BASE "$tex.file.texCoord"
#define _AI_MATKEY_GLTF_MAPPINGNAME_BASE "$tex.mappingname"
#define _AI_MATKEY_GLTF_MAPPINGID_BASE "$tex.mappingid"
#define _AI_MATKEY_GLTF_MAPPINGFILTER_MAG_BASE "$tex.mappingfiltermag"
#define _AI_MATKEY_GLTF_MAPPINGFILTER_MIN_BASE "$tex.mappingfiltermin"
#define _AI_MATKEY_GLTF_SCALE_BASE "$tex.scale"
#define _AI_MATKEY_GLTF_STRENGTH_BASE "$tex.strength"

#define AI_MATKEY_GLTF_TEXTURE_TEXCOORD(type, N) _AI_MATKEY_GLTF_TEXTURE_TEXCOORD_BASE, type, N
#define AI_MATKEY_GLTF_MAPPINGNAME(type, N) _AI_MATKEY_GLTF_MAPPINGNAME_BASE, type, N
#define AI_MATKEY_GLTF_MAPPINGID(type, N) _AI_MATKEY_GLTF_MAPPINGID_BASE, type, N
#define AI_MATKEY_GLTF_MAPPINGFILTER_MAG(type, N) _AI_MATKEY_GLTF_MAPPINGFILTER_MAG_BASE, type, N
#define AI_MATKEY_GLTF_MAPPINGFILTER_MIN(type, N) _AI_MATKEY_GLTF_MAPPINGFILTER_MIN_BASE, type, N
#define AI_MATKEY_GLTF_TEXTURE_SCALE(type, N) _AI_MATKEY_GLTF_SCALE_BASE, type, N
#define AI_MATKEY_GLTF_TEXTURE_STRENGTH(type, N) _AI_MATKEY_GLTF_STRENGTH_BASE, type, N

#endif //!!AI_PBRMATERIAL_H_INC
