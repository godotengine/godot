/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2018, assimp team


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

/** @file config.h
 *  @brief Defines constants for configurable properties for the library
 *
 *  Typically these properties are set via
 *  #Assimp::Importer::SetPropertyFloat,
 *  #Assimp::Importer::SetPropertyInteger or
 *  #Assimp::Importer::SetPropertyString,
 *  depending on the data type of a property. All properties have a
 *  default value. See the doc for the mentioned methods for more details.
 *
 *  <br><br>
 *  The corresponding functions for use with the plain-c API are:
 *  #aiSetImportPropertyInteger,
 *  #aiSetImportPropertyFloat,
 *  #aiSetImportPropertyString
 */
#pragma once
#ifndef AI_CONFIG_H_INC
#define AI_CONFIG_H_INC


// ###########################################################################
// LIBRARY SETTINGS
// General, global settings
// ###########################################################################

// ---------------------------------------------------------------------------
/** @brief Enables time measurements.
 *
 *  If enabled, measures the time needed for each part of the loading
 *  process (i.e. IO time, importing, postprocessing, ..) and dumps
 *  these timings to the DefaultLogger. See the @link perf Performance
 *  Page@endlink for more information on this topic.
 *
 * Property type: bool. Default value: false.
 */
#define AI_CONFIG_GLOB_MEASURE_TIME  \
    "GLOB_MEASURE_TIME"


// ---------------------------------------------------------------------------
/** @brief Global setting to disable generation of skeleton dummy meshes
 *
 * Skeleton dummy meshes are generated as a visualization aid in cases which
 * the input data contains no geometry, but only animation data.
 * Property data type: bool. Default value: false
 */
// ---------------------------------------------------------------------------
#define AI_CONFIG_IMPORT_NO_SKELETON_MESHES \
    "IMPORT_NO_SKELETON_MESHES"



# if 0 // not implemented yet
// ---------------------------------------------------------------------------
/** @brief Set Assimp's multithreading policy.
 *
 * This setting is ignored if Assimp was built without boost.thread
 * support (ASSIMP_BUILD_NO_THREADING, which is implied by ASSIMP_BUILD_BOOST_WORKAROUND).
 * Possible values are: -1 to let Assimp decide what to do, 0 to disable
 * multithreading entirely and any number larger than 0 to force a specific
 * number of threads. Assimp is always free to ignore this settings, which is
 * merely a hint. Usually, the default value (-1) will be fine. However, if
 * Assimp is used concurrently from multiple user threads, it might be useful
 * to limit each Importer instance to a specific number of cores.
 *
 * For more information, see the @link threading Threading page@endlink.
 * Property type: int, default value: -1.
 */
#define AI_CONFIG_GLOB_MULTITHREADING  \
    "GLOB_MULTITHREADING"
#endif

// ###########################################################################
// POST PROCESSING SETTINGS
// Various stuff to fine-tune the behavior of a specific post processing step.
// ###########################################################################


// ---------------------------------------------------------------------------
/** @brief Maximum bone count per mesh for the SplitbyBoneCount step.
 *
 * Meshes are split until the maximum number of bones is reached. The default
 * value is AI_SBBC_DEFAULT_MAX_BONES, which may be altered at
 * compile-time.
 * Property data type: integer.
 */
// ---------------------------------------------------------------------------
#define AI_CONFIG_PP_SBBC_MAX_BONES \
    "PP_SBBC_MAX_BONES"


// default limit for bone count
#if (!defined AI_SBBC_DEFAULT_MAX_BONES)
#   define AI_SBBC_DEFAULT_MAX_BONES        60
#endif


// ---------------------------------------------------------------------------
/** @brief  Specifies the maximum angle that may be between two vertex tangents
 *         that their tangents and bi-tangents are smoothed.
 *
 * This applies to the CalcTangentSpace-Step. The angle is specified
 * in degrees. The maximum value is 175.
 * Property type: float. Default value: 45 degrees
 */
#define AI_CONFIG_PP_CT_MAX_SMOOTHING_ANGLE \
    "PP_CT_MAX_SMOOTHING_ANGLE"

// ---------------------------------------------------------------------------
/** @brief Source UV channel for tangent space computation.
 *
 * The specified channel must exist or an error will be raised.
 * Property type: integer. Default value: 0
 */
// ---------------------------------------------------------------------------
#define AI_CONFIG_PP_CT_TEXTURE_CHANNEL_INDEX \
    "PP_CT_TEXTURE_CHANNEL_INDEX"

// ---------------------------------------------------------------------------
/** @brief  Specifies the maximum angle that may be between two face normals
 *          at the same vertex position that their are smoothed together.
 *
 * Sometimes referred to as 'crease angle'.
 * This applies to the GenSmoothNormals-Step. The angle is specified
 * in degrees, so 180 is PI. The default value is 175 degrees (all vertex
 * normals are smoothed). The maximum value is 175, too. Property type: float.
 * Warning: setting this option may cause a severe loss of performance. The
 * performance is unaffected if the #AI_CONFIG_FAVOUR_SPEED flag is set but
 * the output quality may be reduced.
 */
#define AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE \
    "PP_GSN_MAX_SMOOTHING_ANGLE"


// ---------------------------------------------------------------------------
/** @brief Sets the colormap (= palette) to be used to decode embedded
 *         textures in MDL (Quake or 3DGS) files.
 *
 * This must be a valid path to a file. The file is 768 (256*3) bytes
 * large and contains RGB triplets for each of the 256 palette entries.
 * The default value is colormap.lmp. If the file is not found,
 * a default palette (from Quake 1) is used.
 * Property type: string.
 */
#define AI_CONFIG_IMPORT_MDL_COLORMAP       \
    "IMPORT_MDL_COLORMAP"

// ---------------------------------------------------------------------------
/** @brief Configures the #aiProcess_RemoveRedundantMaterials step to
 *  keep materials matching a name in a given list.
 *
 * This is a list of 1 to n strings, ' ' serves as delimiter character.
 * Identifiers containing whitespaces must be enclosed in *single*
 * quotation marks. For example:<tt>
 * "keep-me and_me_to anotherMaterialToBeKept \'name with whitespace\'"</tt>.
 * If a material matches on of these names, it will not be modified or
 * removed by the postprocessing step nor will other materials be replaced
 * by a reference to it. <br>
 * This option might be useful if you are using some magic material names
 * to pass additional semantics through the content pipeline. This ensures
 * they won't be optimized away, but a general optimization is still
 * performed for materials not contained in the list.
 * Property type: String. Default value: n/a
 * @note Linefeeds, tabs or carriage returns are treated as whitespace.
 *   Material names are case sensitive.
 */
#define AI_CONFIG_PP_RRM_EXCLUDE_LIST   \
    "PP_RRM_EXCLUDE_LIST"

// ---------------------------------------------------------------------------
/** @brief Configures the #aiProcess_PreTransformVertices step to
 *  keep the scene hierarchy. Meshes are moved to worldspace, but
 *  no optimization is performed (read: meshes with equal materials are not
 *  joined. The total number of meshes won't change).
 *
 * This option could be of use for you if the scene hierarchy contains
 * important additional information which you intend to parse.
 * For rendering, you can still render all meshes in the scene without
 * any transformations.
 * Property type: bool. Default value: false.
 */
#define AI_CONFIG_PP_PTV_KEEP_HIERARCHY     \
    "PP_PTV_KEEP_HIERARCHY"

// ---------------------------------------------------------------------------
/** @brief Configures the #aiProcess_PreTransformVertices step to normalize
 *  all vertex components into the [-1,1] range. That is, a bounding box
 *  for the whole scene is computed, the maximum component is taken and all
 *  meshes are scaled appropriately (uniformly of course!).
 *  This might be useful if you don't know the spatial dimension of the input
 *  data*/
#define AI_CONFIG_PP_PTV_NORMALIZE  \
    "PP_PTV_NORMALIZE"

// ---------------------------------------------------------------------------
/** @brief Configures the #aiProcess_PreTransformVertices step to use
 *  a users defined matrix as the scene root node transformation before
 *  transforming vertices.
 *  Property type: bool. Default value: false.
 */
#define AI_CONFIG_PP_PTV_ADD_ROOT_TRANSFORMATION    \
    "PP_PTV_ADD_ROOT_TRANSFORMATION"

// ---------------------------------------------------------------------------
/** @brief Configures the #aiProcess_PreTransformVertices step to use
 *  a users defined matrix as the scene root node transformation before
 *  transforming vertices. This property correspond to the 'a1' component
 *  of the transformation matrix.
 *  Property type: aiMatrix4x4.
 */
#define AI_CONFIG_PP_PTV_ROOT_TRANSFORMATION    \
    "PP_PTV_ROOT_TRANSFORMATION"

// ---------------------------------------------------------------------------
/** @brief Configures the #aiProcess_FindDegenerates step to
 *  remove degenerated primitives from the import - immediately.
 *
 * The default behaviour converts degenerated triangles to lines and
 * degenerated lines to points. See the documentation to the
 * #aiProcess_FindDegenerates step for a detailed example of the various ways
 * to get rid of these lines and points if you don't want them.
 * Property type: bool. Default value: false.
 */
#define AI_CONFIG_PP_FD_REMOVE \
    "PP_FD_REMOVE"

// ---------------------------------------------------------------------------
/**
 *  @brief  Configures the #aiProcess_FindDegenerates to check the area of a
 *  trinagle to be greates than e-6. If this is not the case the triangle will
 *  be removed if #AI_CONFIG_PP_FD_REMOVE is set to true.
 */
#define AI_CONFIG_PP_FD_CHECKAREA \
    "PP_FD_CHECKAREA"

// ---------------------------------------------------------------------------
/** @brief Configures the #aiProcess_OptimizeGraph step to preserve nodes
 * matching a name in a given list.
 *
 * This is a list of 1 to n strings, ' ' serves as delimiter character.
 * Identifiers containing whitespaces must be enclosed in *single*
 * quotation marks. For example:<tt>
 * "keep-me and_me_to anotherNodeToBeKept \'name with whitespace\'"</tt>.
 * If a node matches on of these names, it will not be modified or
 * removed by the postprocessing step.<br>
 * This option might be useful if you are using some magic node names
 * to pass additional semantics through the content pipeline. This ensures
 * they won't be optimized away, but a general optimization is still
 * performed for nodes not contained in the list.
 * Property type: String. Default value: n/a
 * @note Linefeeds, tabs or carriage returns are treated as whitespace.
 *   Node names are case sensitive.
 */
#define AI_CONFIG_PP_OG_EXCLUDE_LIST    \
    "PP_OG_EXCLUDE_LIST"

// ---------------------------------------------------------------------------
/** @brief  Set the maximum number of triangles in a mesh.
 *
 * This is used by the "SplitLargeMeshes" PostProcess-Step to determine
 * whether a mesh must be split or not.
 * @note The default value is AI_SLM_DEFAULT_MAX_TRIANGLES
 * Property type: integer.
 */
#define AI_CONFIG_PP_SLM_TRIANGLE_LIMIT \
    "PP_SLM_TRIANGLE_LIMIT"

// default value for AI_CONFIG_PP_SLM_TRIANGLE_LIMIT
#if (!defined AI_SLM_DEFAULT_MAX_TRIANGLES)
#   define AI_SLM_DEFAULT_MAX_TRIANGLES     1000000
#endif

// ---------------------------------------------------------------------------
/** @brief  Set the maximum number of vertices in a mesh.
 *
 * This is used by the "SplitLargeMeshes" PostProcess-Step to determine
 * whether a mesh must be split or not.
 * @note The default value is AI_SLM_DEFAULT_MAX_VERTICES
 * Property type: integer.
 */
#define AI_CONFIG_PP_SLM_VERTEX_LIMIT \
    "PP_SLM_VERTEX_LIMIT"

// default value for AI_CONFIG_PP_SLM_VERTEX_LIMIT
#if (!defined AI_SLM_DEFAULT_MAX_VERTICES)
#   define AI_SLM_DEFAULT_MAX_VERTICES      1000000
#endif

// ---------------------------------------------------------------------------
/** @brief Set the maximum number of bones affecting a single vertex
 *
 * This is used by the #aiProcess_LimitBoneWeights PostProcess-Step.
 * @note The default value is AI_LMW_MAX_WEIGHTS
 * Property type: integer.*/
#define AI_CONFIG_PP_LBW_MAX_WEIGHTS    \
    "PP_LBW_MAX_WEIGHTS"

// default value for AI_CONFIG_PP_LBW_MAX_WEIGHTS
#if (!defined AI_LMW_MAX_WEIGHTS)
#   define AI_LMW_MAX_WEIGHTS   0x4
#endif // !! AI_LMW_MAX_WEIGHTS

// ---------------------------------------------------------------------------
/** @brief Lower the deboning threshold in order to remove more bones.
 *
 * This is used by the #aiProcess_Debone PostProcess-Step.
 * @note The default value is AI_DEBONE_THRESHOLD
 * Property type: float.*/
#define AI_CONFIG_PP_DB_THRESHOLD \
    "PP_DB_THRESHOLD"

// default value for AI_CONFIG_PP_LBW_MAX_WEIGHTS
#if (!defined AI_DEBONE_THRESHOLD)
#   define AI_DEBONE_THRESHOLD  1.0f
#endif // !! AI_DEBONE_THRESHOLD

// ---------------------------------------------------------------------------
/** @brief Require all bones qualify for deboning before removing any
 *
 * This is used by the #aiProcess_Debone PostProcess-Step.
 * @note The default value is 0
 * Property type: bool.*/
#define AI_CONFIG_PP_DB_ALL_OR_NONE \
    "PP_DB_ALL_OR_NONE"

/** @brief Default value for the #AI_CONFIG_PP_ICL_PTCACHE_SIZE property
 */
#ifndef PP_ICL_PTCACHE_SIZE
#   define PP_ICL_PTCACHE_SIZE 12
#endif

// ---------------------------------------------------------------------------
/** @brief Set the size of the post-transform vertex cache to optimize the
 *    vertices for. This configures the #aiProcess_ImproveCacheLocality step.
 *
 * The size is given in vertices. Of course you can't know how the vertex
 * format will exactly look like after the import returns, but you can still
 * guess what your meshes will probably have.
 * @note The default value is #PP_ICL_PTCACHE_SIZE. That results in slight
 * performance improvements for most nVidia/AMD cards since 2002.
 * Property type: integer.
 */
#define AI_CONFIG_PP_ICL_PTCACHE_SIZE   "PP_ICL_PTCACHE_SIZE"

// ---------------------------------------------------------------------------
/** @brief Enumerates components of the aiScene and aiMesh data structures
 *  that can be excluded from the import using the #aiProcess_RemoveComponent step.
 *
 *  See the documentation to #aiProcess_RemoveComponent for more details.
 */
enum aiComponent
{
    /** Normal vectors */
#ifdef SWIG
    aiComponent_NORMALS = 0x2,
#else
    aiComponent_NORMALS = 0x2u,
#endif

    /** Tangents and bitangents go always together ... */
#ifdef SWIG
    aiComponent_TANGENTS_AND_BITANGENTS = 0x4,
#else
    aiComponent_TANGENTS_AND_BITANGENTS = 0x4u,
#endif

    /** ALL color sets
     * Use aiComponent_COLORn(N) to specify the N'th set */
    aiComponent_COLORS = 0x8,

    /** ALL texture UV sets
     * aiComponent_TEXCOORDn(N) to specify the N'th set  */
    aiComponent_TEXCOORDS = 0x10,

    /** Removes all bone weights from all meshes.
     * The scenegraph nodes corresponding to the bones are NOT removed.
     * use the #aiProcess_OptimizeGraph step to do this */
    aiComponent_BONEWEIGHTS = 0x20,

    /** Removes all node animations (aiScene::mAnimations).
     * The corresponding scenegraph nodes are NOT removed.
     * use the #aiProcess_OptimizeGraph step to do this */
    aiComponent_ANIMATIONS = 0x40,

    /** Removes all embedded textures (aiScene::mTextures) */
    aiComponent_TEXTURES = 0x80,

    /** Removes all light sources (aiScene::mLights).
     * The corresponding scenegraph nodes are NOT removed.
     * use the #aiProcess_OptimizeGraph step to do this */
    aiComponent_LIGHTS = 0x100,

    /** Removes all cameras (aiScene::mCameras).
     * The corresponding scenegraph nodes are NOT removed.
     * use the #aiProcess_OptimizeGraph step to do this */
    aiComponent_CAMERAS = 0x200,

    /** Removes all meshes (aiScene::mMeshes). */
    aiComponent_MESHES = 0x400,

    /** Removes all materials. One default material will
     * be generated, so aiScene::mNumMaterials will be 1. */
    aiComponent_MATERIALS = 0x800,


    /** This value is not used. It is just there to force the
     *  compiler to map this enum to a 32 Bit integer. */
#ifndef SWIG
    _aiComponent_Force32Bit = 0x9fffffff
#endif
};

// Remove a specific color channel 'n'
#define aiComponent_COLORSn(n) (1u << (n+20u))

// Remove a specific UV channel 'n'
#define aiComponent_TEXCOORDSn(n) (1u << (n+25u))

// ---------------------------------------------------------------------------
/** @brief Input parameter to the #aiProcess_RemoveComponent step:
 *  Specifies the parts of the data structure to be removed.
 *
 * See the documentation to this step for further details. The property
 * is expected to be an integer, a bitwise combination of the
 * #aiComponent flags defined above in this header. The default
 * value is 0. Important: if no valid mesh is remaining after the
 * step has been executed (e.g you thought it was funny to specify ALL
 * of the flags defined above) the import FAILS. Mainly because there is
 * no data to work on anymore ...
 */
#define AI_CONFIG_PP_RVC_FLAGS              \
    "PP_RVC_FLAGS"

// ---------------------------------------------------------------------------
/** @brief Input parameter to the #aiProcess_SortByPType step:
 *  Specifies which primitive types are removed by the step.
 *
 *  This is a bitwise combination of the aiPrimitiveType flags.
 *  Specifying all of them is illegal, of course. A typical use would
 *  be to exclude all line and point meshes from the import. This
 *  is an integer property, its default value is 0.
 */
#define AI_CONFIG_PP_SBP_REMOVE             \
    "PP_SBP_REMOVE"

// ---------------------------------------------------------------------------
/** @brief Input parameter to the #aiProcess_FindInvalidData step:
 *  Specifies the floating-point accuracy for animation values. The step
 *  checks for animation tracks where all frame values are absolutely equal
 *  and removes them. This tweakable controls the epsilon for floating-point
 *  comparisons - two keys are considered equal if the invariant
 *  abs(n0-n1)>epsilon holds true for all vector respectively quaternion
 *  components. The default value is 0.f - comparisons are exact then.
 */
#define AI_CONFIG_PP_FID_ANIM_ACCURACY              \
    "PP_FID_ANIM_ACCURACY"

// ---------------------------------------------------------------------------
/** @brief Input parameter to the #aiProcess_FindInvalidData step:
 *  Set to true to ignore texture coordinates. This may be useful if you have
 *  to assign different kind of textures like one for the summer or one for the winter.
 */
#define AI_CONFIG_PP_FID_IGNORE_TEXTURECOORDS        \
    "PP_FID_IGNORE_TEXTURECOORDS"

// TransformUVCoords evaluates UV scalings
#define AI_UVTRAFO_SCALING 0x1

// TransformUVCoords evaluates UV rotations
#define AI_UVTRAFO_ROTATION 0x2

// TransformUVCoords evaluates UV translation
#define AI_UVTRAFO_TRANSLATION 0x4

// Everything baked together -> default value
#define AI_UVTRAFO_ALL (AI_UVTRAFO_SCALING | AI_UVTRAFO_ROTATION | AI_UVTRAFO_TRANSLATION)

// ---------------------------------------------------------------------------
/** @brief Input parameter to the #aiProcess_TransformUVCoords step:
 *  Specifies which UV transformations are evaluated.
 *
 *  This is a bitwise combination of the AI_UVTRAFO_XXX flags (integer
 *  property, of course). By default all transformations are enabled
 * (AI_UVTRAFO_ALL).
 */
#define AI_CONFIG_PP_TUV_EVALUATE               \
    "PP_TUV_EVALUATE"

// ---------------------------------------------------------------------------
/** @brief A hint to assimp to favour speed against import quality.
 *
 * Enabling this option may result in faster loading, but it needn't.
 * It represents just a hint to loaders and post-processing steps to use
 * faster code paths, if possible.
 * This property is expected to be an integer, != 0 stands for true.
 * The default value is 0.
 */
#define AI_CONFIG_FAVOUR_SPEED              \
 "FAVOUR_SPEED"


// ###########################################################################
// IMPORTER SETTINGS
// Various stuff to fine-tune the behaviour of specific importer plugins.
// ###########################################################################


// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will merge all geometry layers present
 *    in the source file or take only the first.
 *
 * The default value is true (1)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_READ_ALL_GEOMETRY_LAYERS \
    "IMPORT_FBX_READ_ALL_GEOMETRY_LAYERS"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will read all materials present in the
 *    source file or take only the referenced materials.
 *
 * This is void unless IMPORT_FBX_READ_MATERIALS=1.
 *
 * The default value is false (0)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_READ_ALL_MATERIALS \
    "IMPORT_FBX_READ_ALL_MATERIALS"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will read materials.
 *
 * The default value is true (1)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_READ_MATERIALS \
    "IMPORT_FBX_READ_MATERIALS"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will read embedded textures.
 *
 * The default value is true (1)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_READ_TEXTURES \
    "IMPORT_FBX_READ_TEXTURES"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will read cameras.
 *
 * The default value is true (1)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_READ_CAMERAS \
    "IMPORT_FBX_READ_CAMERAS"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will read light sources.
 *
 * The default value is true (1)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_READ_LIGHTS \
    "IMPORT_FBX_READ_LIGHTS"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will read animations.
 *
 * The default value is true (1)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_READ_ANIMATIONS \
    "IMPORT_FBX_READ_ANIMATIONS"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will act in strict mode in which only
 *    FBX 2013 is supported and any other sub formats are rejected. FBX 2013
 *    is the primary target for the importer, so this format is best
 *    supported and well-tested.
 *
 * The default value is false (0)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_STRICT_MODE \
    "IMPORT_FBX_STRICT_MODE"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will preserve pivot points for
 *    transformations (as extra nodes). If set to false, pivots and offsets
 *    will be evaluated whenever possible.
 *
 * The default value is true (1)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS \
    "IMPORT_FBX_PRESERVE_PIVOTS"

// ---------------------------------------------------------------------------
/** @brief Specifies whether the importer will drop empty animation curves or
 *    animation curves which match the bind pose transformation over their
 *    entire defined range.
 *
 * The default value is true (1)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_OPTIMIZE_EMPTY_ANIMATION_CURVES \
    "IMPORT_FBX_OPTIMIZE_EMPTY_ANIMATION_CURVES"

// ---------------------------------------------------------------------------
/** @brief Set whether the fbx importer will use the legacy embedded texture naming.
 *
 * The default value is false (0)
 * Property type: bool
 */
#define AI_CONFIG_IMPORT_FBX_EMBEDDED_TEXTURES_LEGACY_NAMING \
	"AI_CONFIG_IMPORT_FBX_EMBEDDED_TEXTURES_LEGACY_NAMING"

// ---------------------------------------------------------------------------
/** @brief  Set wether the importer shall not remove empty bones.
 *  
 *  Empty bone are often used to define connections for other models.
 */
#define AI_CONFIG_IMPORT_REMOVE_EMPTY_BONES \
    "AI_CONFIG_IMPORT_REMOVE_EMPTY_BONES"


// ---------------------------------------------------------------------------
/** @brief  Set wether the FBX importer shall convert the unit from cm to m.
 */
#define AI_CONFIG_FBX_CONVERT_TO_M \
    "AI_CONFIG_FBX_CONVERT_TO_M"

// ---------------------------------------------------------------------------
/** @brief  Set the vertex animation keyframe to be imported
 *
 * ASSIMP does not support vertex keyframes (only bone animation is supported).
 * The library reads only one frame of models with vertex animations.
 * By default this is the first frame.
 * \note The default value is 0. This option applies to all importers.
 *   However, it is also possible to override the global setting
 *   for a specific loader. You can use the AI_CONFIG_IMPORT_XXX_KEYFRAME
 *   options (where XXX is a placeholder for the file format for which you
 *   want to override the global setting).
 * Property type: integer.
 */
#define AI_CONFIG_IMPORT_GLOBAL_KEYFRAME    "IMPORT_GLOBAL_KEYFRAME"

#define AI_CONFIG_IMPORT_MD3_KEYFRAME       "IMPORT_MD3_KEYFRAME"
#define AI_CONFIG_IMPORT_MD2_KEYFRAME       "IMPORT_MD2_KEYFRAME"
#define AI_CONFIG_IMPORT_MDL_KEYFRAME       "IMPORT_MDL_KEYFRAME"
#define AI_CONFIG_IMPORT_MDC_KEYFRAME       "IMPORT_MDC_KEYFRAME"
#define AI_CONFIG_IMPORT_SMD_KEYFRAME       "IMPORT_SMD_KEYFRAME"
#define AI_CONFIG_IMPORT_UNREAL_KEYFRAME    "IMPORT_UNREAL_KEYFRAME"

// ---------------------------------------------------------------------------
/** Smd load multiple animations
 *
 *  Property type: bool. Default value: true.
 */
#define AI_CONFIG_IMPORT_SMD_LOAD_ANIMATION_LIST "IMPORT_SMD_LOAD_ANIMATION_LIST"

// ---------------------------------------------------------------------------
/** @brief  Configures the AC loader to collect all surfaces which have the
 *    "Backface cull" flag set in separate meshes.
 *
 *  Property type: bool. Default value: true.
 */
#define AI_CONFIG_IMPORT_AC_SEPARATE_BFCULL \
    "IMPORT_AC_SEPARATE_BFCULL"

// ---------------------------------------------------------------------------
/** @brief  Configures whether the AC loader evaluates subdivision surfaces (
 *  indicated by the presence of the 'subdiv' attribute in the file). By
 *  default, Assimp performs the subdivision using the standard
 *  Catmull-Clark algorithm
 *
 * * Property type: bool. Default value: true.
 */
#define AI_CONFIG_IMPORT_AC_EVAL_SUBDIVISION    \
    "IMPORT_AC_EVAL_SUBDIVISION"

// ---------------------------------------------------------------------------
/** @brief  Configures the UNREAL 3D loader to separate faces with different
 *    surface flags (e.g. two-sided vs. single-sided).
 *
 * * Property type: bool. Default value: true.
 */
#define AI_CONFIG_IMPORT_UNREAL_HANDLE_FLAGS \
    "UNREAL_HANDLE_FLAGS"

// ---------------------------------------------------------------------------
/** @brief Configures the terragen import plugin to compute uv's for
 *  terrains, if not given. Furthermore a default texture is assigned.
 *
 * UV coordinates for terrains are so simple to compute that you'll usually
 * want to compute them on your own, if you need them. This option is intended
 * for model viewers which want to offer an easy way to apply textures to
 * terrains.
 * * Property type: bool. Default value: false.
 */
#define AI_CONFIG_IMPORT_TER_MAKE_UVS \
    "IMPORT_TER_MAKE_UVS"

// ---------------------------------------------------------------------------
/** @brief  Configures the ASE loader to always reconstruct normal vectors
 *  basing on the smoothing groups loaded from the file.
 *
 * Some ASE files have carry invalid normals, other don't.
 * * Property type: bool. Default value: true.
 */
#define AI_CONFIG_IMPORT_ASE_RECONSTRUCT_NORMALS    \
    "IMPORT_ASE_RECONSTRUCT_NORMALS"

// ---------------------------------------------------------------------------
/** @brief  Configures the M3D loader to detect and process multi-part
 *    Quake player models.
 *
 * These models usually consist of 3 files, lower.md3, upper.md3 and
 * head.md3. If this property is set to true, Assimp will try to load and
 * combine all three files if one of them is loaded.
 * Property type: bool. Default value: true.
 */
#define AI_CONFIG_IMPORT_MD3_HANDLE_MULTIPART \
    "IMPORT_MD3_HANDLE_MULTIPART"

// ---------------------------------------------------------------------------
/** @brief  Tells the MD3 loader which skin files to load.
 *
 * When loading MD3 files, Assimp checks whether a file
 * [md3_file_name]_[skin_name].skin is existing. These files are used by
 * Quake III to be able to assign different skins (e.g. red and blue team)
 * to models. 'default', 'red', 'blue' are typical skin names.
 * Property type: String. Default value: "default".
 */
#define AI_CONFIG_IMPORT_MD3_SKIN_NAME \
    "IMPORT_MD3_SKIN_NAME"

// ---------------------------------------------------------------------------
/** @brief  Specify the Quake 3 shader file to be used for a particular
 *  MD3 file. This can also be a search path.
 *
 * By default Assimp's behaviour is as follows: If a MD3 file
 * <tt>any_path/models/any_q3_subdir/model_name/file_name.md3</tt> is
 * loaded, the library tries to locate the corresponding shader file in
 * <tt>any_path/scripts/model_name.shader</tt>. This property overrides this
 * behaviour. It can either specify a full path to the shader to be loaded
 * or alternatively the path (relative or absolute) to the directory where
 * the shaders for all MD3s to be loaded reside. Assimp attempts to open
 * <tt>IMPORT_MD3_SHADER_SRC/model_name.shader</tt> first, <tt>IMPORT_MD3_SHADER_SRC/file_name.shader</tt>
 * is the fallback file. Note that IMPORT_MD3_SHADER_SRC should have a terminal (back)slash.
 * Property type: String. Default value: n/a.
 */
#define AI_CONFIG_IMPORT_MD3_SHADER_SRC \
    "IMPORT_MD3_SHADER_SRC"

// ---------------------------------------------------------------------------
/** @brief  Configures the LWO loader to load just one layer from the model.
 *
 * LWO files consist of layers and in some cases it could be useful to load
 * only one of them. This property can be either a string - which specifies
 * the name of the layer - or an integer - the index of the layer. If the
 * property is not set the whole LWO model is loaded. Loading fails if the
 * requested layer is not available. The layer index is zero-based and the
 * layer name may not be empty.<br>
 * Property type: Integer. Default value: all layers are loaded.
 */
#define AI_CONFIG_IMPORT_LWO_ONE_LAYER_ONLY         \
    "IMPORT_LWO_ONE_LAYER_ONLY"

// ---------------------------------------------------------------------------
/** @brief  Configures the MD5 loader to not load the MD5ANIM file for
 *  a MD5MESH file automatically.
 *
 * The default strategy is to look for a file with the same name but the
 * MD5ANIM extension in the same directory. If it is found, it is loaded
 * and combined with the MD5MESH file. This configuration option can be
 * used to disable this behaviour.
 *
 * * Property type: bool. Default value: false.
 */
#define AI_CONFIG_IMPORT_MD5_NO_ANIM_AUTOLOAD           \
    "IMPORT_MD5_NO_ANIM_AUTOLOAD"

// ---------------------------------------------------------------------------
/** @brief Defines the begin of the time range for which the LWS loader
 *    evaluates animations and computes aiNodeAnim's.
 *
 * Assimp provides full conversion of LightWave's envelope system, including
 * pre and post conditions. The loader computes linearly subsampled animation
 * chanels with the frame rate given in the LWS file. This property defines
 * the start time. Note: animation channels are only generated if a node
 * has at least one envelope with more tan one key assigned. This property.
 * is given in frames, '0' is the first frame. By default, if this property
 * is not set, the importer takes the animation start from the input LWS
 * file ('FirstFrame' line)<br>
 * Property type: Integer. Default value: taken from file.
 *
 * @see AI_CONFIG_IMPORT_LWS_ANIM_END - end of the imported time range
 */
#define AI_CONFIG_IMPORT_LWS_ANIM_START         \
    "IMPORT_LWS_ANIM_START"
#define AI_CONFIG_IMPORT_LWS_ANIM_END           \
    "IMPORT_LWS_ANIM_END"

// ---------------------------------------------------------------------------
/** @brief Defines the output frame rate of the IRR loader.
 *
 * IRR animations are difficult to convert for Assimp and there will
 * always be a loss of quality. This setting defines how many keys per second
 * are returned by the converter.<br>
 * Property type: integer. Default value: 100
 */
#define AI_CONFIG_IMPORT_IRR_ANIM_FPS               \
    "IMPORT_IRR_ANIM_FPS"

// ---------------------------------------------------------------------------
/** @brief Ogre Importer will try to find referenced materials from this file.
 *
 * Ogre meshes reference with material names, this does not tell Assimp the file
 * where it is located in. Assimp will try to find the source file in the following
 * order: <material-name>.material, <mesh-filename-base>.material and
 * lastly the material name defined by this config property.
 * <br>
 * Property type: String. Default value: Scene.material.
 */
#define AI_CONFIG_IMPORT_OGRE_MATERIAL_FILE \
    "IMPORT_OGRE_MATERIAL_FILE"

// ---------------------------------------------------------------------------
/** @brief Ogre Importer detect the texture usage from its filename.
 *
 * Ogre material texture units do not define texture type, the textures usage
 * depends on the used shader or Ogre's fixed pipeline. If this config property
 * is true Assimp will try to detect the type from the textures filename postfix:
 * _n, _nrm, _nrml, _normal, _normals and _normalmap for normal map, _s, _spec,
 * _specular and _specularmap for specular map, _l, _light, _lightmap, _occ
 * and _occlusion for light map, _disp and _displacement for displacement map.
 * The matching is case insensitive. Post fix is taken between the last
 * underscore and the last period.
 * Default behavior is to detect type from lower cased texture unit name by
 * matching against: normalmap, specularmap, lightmap and displacementmap.
 * For both cases if no match is found aiTextureType_DIFFUSE is used.
 * <br>
 * Property type: Bool. Default value: false.
 */
#define AI_CONFIG_IMPORT_OGRE_TEXTURETYPE_FROM_FILENAME \
    "IMPORT_OGRE_TEXTURETYPE_FROM_FILENAME"

 /** @brief Specifies whether the Android JNI asset extraction is supported.
  *
  * Turn on this option if you want to manage assets in native
  * Android application without having to keep the internal directory and asset
  * manager pointer.
  */
 #define AI_CONFIG_ANDROID_JNI_ASSIMP_MANAGER_SUPPORT "AI_CONFIG_ANDROID_JNI_ASSIMP_MANAGER_SUPPORT"

// ---------------------------------------------------------------------------
/** @brief Specifies whether the IFC loader skips over IfcSpace elements.
 *
 * IfcSpace elements (and their geometric representations) are used to
 * represent, well, free space in a building storey.<br>
 * Property type: Bool. Default value: true.
 */
#define AI_CONFIG_IMPORT_IFC_SKIP_SPACE_REPRESENTATIONS "IMPORT_IFC_SKIP_SPACE_REPRESENTATIONS"

// ---------------------------------------------------------------------------
/** @brief Specifies whether the IFC loader will use its own, custom triangulation
 *   algorithm to triangulate wall and floor meshes.
 *
 * If this property is set to false, walls will be either triangulated by
 * #aiProcess_Triangulate or will be passed through as huge polygons with
 * faked holes (i.e. holes that are connected with the outer boundary using
 * a dummy edge). It is highly recommended to set this property to true
 * if you want triangulated data because #aiProcess_Triangulate is known to
 * have problems with the kind of polygons that the IFC loader spits out for
 * complicated meshes.
 * Property type: Bool. Default value: true.
 */
#define AI_CONFIG_IMPORT_IFC_CUSTOM_TRIANGULATION "IMPORT_IFC_CUSTOM_TRIANGULATION"

// ---------------------------------------------------------------------------
/** @brief  Set the tessellation conic angle for IFC smoothing curves.
 *
 * This is used by the IFC importer to determine the tessellation parameter
 * for smoothing curves.
 * @note The default value is AI_IMPORT_IFC_DEFAULT_SMOOTHING_ANGLE and the
 * accepted values are in range [5.0, 120.0].
 * Property type: Float.
 */
#define AI_CONFIG_IMPORT_IFC_SMOOTHING_ANGLE "IMPORT_IFC_SMOOTHING_ANGLE"

// default value for AI_CONFIG_IMPORT_IFC_SMOOTHING_ANGLE
#if (!defined AI_IMPORT_IFC_DEFAULT_SMOOTHING_ANGLE)
#   define AI_IMPORT_IFC_DEFAULT_SMOOTHING_ANGLE 10.0f
#endif

// ---------------------------------------------------------------------------
/** @brief  Set the tessellation for IFC cylindrical shapes.
 *
 * This is used by the IFC importer to determine the tessellation parameter
 * for cylindrical shapes, i.e. the number of segments used to approximate a circle.
 * @note The default value is AI_IMPORT_IFC_DEFAULT_CYLINDRICAL_TESSELLATION and the
 * accepted values are in range [3, 180].
 * Property type: Integer.
 */
#define AI_CONFIG_IMPORT_IFC_CYLINDRICAL_TESSELLATION "IMPORT_IFC_CYLINDRICAL_TESSELLATION"

// default value for AI_CONFIG_IMPORT_IFC_CYLINDRICAL_TESSELLATION
#if (!defined AI_IMPORT_IFC_DEFAULT_CYLINDRICAL_TESSELLATION)
#   define AI_IMPORT_IFC_DEFAULT_CYLINDRICAL_TESSELLATION 32
#endif

// ---------------------------------------------------------------------------
/** @brief Specifies whether the Collada loader will ignore the provided up direction.
 *
 * If this property is set to true, the up direction provided in the file header will
 * be ignored and the file will be loaded as is.
 * Property type: Bool. Default value: false.
 */
#define AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION "IMPORT_COLLADA_IGNORE_UP_DIRECTION"

// ---------------------------------------------------------------------------
/** @brief Specifies whether the Collada loader should use Collada names as node names.
 *
 * If this property is set to true, the Collada names will be used as the
 * node name. The default is to use the id tag (resp. sid tag, if no id tag is present)
 * instead.
 * Property type: Bool. Default value: false.
 */
#define AI_CONFIG_IMPORT_COLLADA_USE_COLLADA_NAMES "IMPORT_COLLADA_USE_COLLADA_NAMES"

// ---------- All the Export defines ------------

/** @brief Specifies the xfile use double for real values of float
 *
 * Property type: Bool. Default value: false.
 */

#define AI_CONFIG_EXPORT_XFILE_64BIT "EXPORT_XFILE_64BIT"

/** @brief Specifies whether the assimp export shall be able to export point clouds
 * 
 *  When this flag is not defined the render data has to contain valid faces.
 *  Point clouds are only a collection of vertices which have nor spatial organization
 *  by a face and the validation process will remove them. Enabling this feature will
 *  switch off the flag and enable the functionality to export pure point clouds.
 */
#define AI_CONFIG_EXPORT_POINT_CLOUDS "EXPORT_POINT_CLOUDS"

/**
 *  @brief  Specifies a gobal key factor for scale, float value
 */
#define AI_CONFIG_GLOBAL_SCALE_FACTOR_KEY "GLOBAL_SCALE_FACTOR"

#if (!defined AI_CONFIG_GLOBAL_SCALE_FACTOR_DEFAULT)
#   define AI_CONFIG_GLOBAL_SCALE_FACTOR_DEFAULT  1.0f
#endif // !! AI_DEBONE_THRESHOLD

#define AI_CONFIG_APP_SCALE_KEY "APP_SCALE_FACTOR"

#if (!defined AI_CONFIG_APP_SCALE_KEY)
#   define AI_CONFIG_APP_SCALE_KEY 1.0
#endif // AI_CONFIG_APP_SCALE_KEY


// ---------- All the Build/Compile-time defines ------------

/** @brief Specifies if double precision is supported inside assimp
 *
 * Property type: Bool. Default value: undefined.
 */

#cmakedefine ASSIMP_DOUBLE_PRECISION 1

#endif // !! AI_CONFIG_H_INC
