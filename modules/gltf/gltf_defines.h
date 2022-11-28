/*************************************************************************/
/*  gltf_defines.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GLTF_DEFINES_H
#define GLTF_DEFINES_H

// This file should only be included by other headers.

// Godot classes used by GLTF headers.
class AnimationPlayer;
class BoneAttachment;
class CSGShape;
class DirectionalLight;
class GridMap;
class Light;
class MultiMeshInstance;
class Skeleton;
class Skin;

// GLTF classes.
struct GLTFAccessor;
class GLTFAnimation;
class GLTFBufferView;
class GLTFCamera;
class GLTFDocument;
class GLTFLight;
class GLTFMesh;
class GLTFNode;
class GLTFSkeleton;
class GLTFSkin;
class GLTFSpecGloss;
class GLTFState;
class GLTFTexture;
class GLTFTextureSampler;
class PackedSceneGLTF;

// GLTF index aliases.
using GLTFAccessorIndex = int;
using GLTFAnimationIndex = int;
using GLTFBufferIndex = int;
using GLTFBufferViewIndex = int;
using GLTFCameraIndex = int;
using GLTFImageIndex = int;
using GLTFLightIndex = int;
using GLTFMaterialIndex = int;
using GLTFMeshIndex = int;
using GLTFNodeIndex = int;
using GLTFSkeletonIndex = int;
using GLTFSkinIndex = int;
using GLTFTextureIndex = int;
using GLTFTextureSamplerIndex = int;

enum GLTFType {
	TYPE_SCALAR,
	TYPE_VEC2,
	TYPE_VEC3,
	TYPE_VEC4,
	TYPE_MAT2,
	TYPE_MAT3,
	TYPE_MAT4,
};

#endif // GLTF_DEFINES_H
