/*************************************************************************/
/*  FBXImportSettings.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

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

/** @file  FBXImportSettings.h
 *  @brief FBX importer runtime configuration
 */
#ifndef FBX_IMPORT_SETTINGS_H
#define FBX_IMPORT_SETTINGS_H

namespace FBXDocParser {

/** FBX import settings, parts of which are publicly accessible via their corresponding AI_CONFIG constants */
struct ImportSettings {
	/** enable strict mode:
	 *   - only accept fbx 2012, 2013 files
	 *   - on the slightest error, give up.
	 *
	 *  Basically, strict mode means that the fbx file will actually
	 *  be validated.*/
	bool strictMode = true;

	/** specifies whether all geometry layers are read and scanned for
	 * usable data channels. The FBX spec indicates that many readers
	 * will only read the first channel and that this is in some way
	 * the recommended way- in reality, however, it happens a lot that
	 * vertex data is spread among multiple layers.*/
	bool readAllLayers = true;

	/** specifies whether all materials are read, or only those that
	 *  are referenced by at least one mesh. Reading all materials
	 *  may make FBX reading a lot slower since all objects
	 *  need to be processed.
	 *  This bit is ignored unless readMaterials=true.*/
	bool readAllMaterials = true;

	/** import materials (true) or skip them and assign a default
	 *  material.*/
	bool readMaterials = true;

	/** import embedded textures?*/
	bool readTextures = true;

	/** import cameras?*/
	bool readCameras = true;

	/** import light sources?*/
	bool readLights = true;

	/** import animations (i.e. animation curves, the node
	 *  skeleton is always imported).*/
	bool readAnimations = true;

	/** read bones (vertex weights and deform info).*/
	bool readWeights = true;

	/** preserve transformation pivots and offsets. Since these can
	 *  not directly be represented in assimp, additional dummy
	 *  nodes will be generated. Note that settings this to false
	 *  can make animation import a lot slower.
	 *
	 *  The naming scheme for the generated nodes is:
	 *    <OriginalName>_$AssimpFbx$_<TransformName>
	 *
	 *  where <TransformName> is one of
	 *    RotationPivot
	 *    RotationOffset
	 *    PreRotation
	 *    PostRotation
	 *    ScalingPivot
	 *    ScalingOffset
	 *    Translation
	 *    Scaling
	 *    Rotation
	 **/
	bool preservePivots = true;

	/** do not import animation curves that specify a constant
	 *  values matching the corresponding node transformation.*/
	bool optimizeEmptyAnimationCurves = true;

	/** use legacy naming for embedded textures eg: (*0, *1, *2).*/
	bool useLegacyEmbeddedTextureNaming = false;

	/** Empty bones shall be removed.*/
	bool removeEmptyBones = true;

	/** Set to true to perform a conversion from cm to meter after
	 *  the import.*/
	bool convertToMeters = false;
};
} // namespace FBXDocParser

#endif // FBX_IMPORT_SETTINGS_H
