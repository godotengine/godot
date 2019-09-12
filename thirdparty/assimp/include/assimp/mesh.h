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

/** @file mesh.h
 *  @brief Declares the data structures in which the imported geometry is
    returned by ASSIMP: aiMesh, aiFace and aiBone data structures.
 */
#pragma once
#ifndef AI_MESH_H_INC
#define AI_MESH_H_INC

#include <assimp/types.h>
#include <assimp/aabb.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Limits. These values are required to match the settings Assimp was
// compiled against. Therefore, do not redefine them unless you build the
// library from source using the same definitions.
// ---------------------------------------------------------------------------

/** @def AI_MAX_FACE_INDICES
 *  Maximum number of indices per face (polygon). */

#ifndef AI_MAX_FACE_INDICES
#   define AI_MAX_FACE_INDICES 0x7fff
#endif

/** @def AI_MAX_BONE_WEIGHTS
 *  Maximum number of indices per face (polygon). */

#ifndef AI_MAX_BONE_WEIGHTS
#   define AI_MAX_BONE_WEIGHTS 0x7fffffff
#endif

/** @def AI_MAX_VERTICES
 *  Maximum number of vertices per mesh.  */

#ifndef AI_MAX_VERTICES
#   define AI_MAX_VERTICES 0x7fffffff
#endif

/** @def AI_MAX_FACES
 *  Maximum number of faces per mesh. */

#ifndef AI_MAX_FACES
#   define AI_MAX_FACES 0x7fffffff
#endif

/** @def AI_MAX_NUMBER_OF_COLOR_SETS
 *  Supported number of vertex color sets per mesh. */

#ifndef AI_MAX_NUMBER_OF_COLOR_SETS
#   define AI_MAX_NUMBER_OF_COLOR_SETS 0x8
#endif // !! AI_MAX_NUMBER_OF_COLOR_SETS

/** @def AI_MAX_NUMBER_OF_TEXTURECOORDS
 *  Supported number of texture coord sets (UV(W) channels) per mesh */

#ifndef AI_MAX_NUMBER_OF_TEXTURECOORDS
#   define AI_MAX_NUMBER_OF_TEXTURECOORDS 0x8
#endif // !! AI_MAX_NUMBER_OF_TEXTURECOORDS

// ---------------------------------------------------------------------------
/** @brief A single face in a mesh, referring to multiple vertices.
 *
 * If mNumIndices is 3, we call the face 'triangle', for mNumIndices > 3
 * it's called 'polygon' (hey, that's just a definition!).
 * <br>
 * aiMesh::mPrimitiveTypes can be queried to quickly examine which types of
 * primitive are actually present in a mesh. The #aiProcess_SortByPType flag
 * executes a special post-processing algorithm which splits meshes with
 * *different* primitive types mixed up (e.g. lines and triangles) in several
 * 'clean' submeshes. Furthermore there is a configuration option (
 * #AI_CONFIG_PP_SBP_REMOVE) to force #aiProcess_SortByPType to remove
 * specific kinds of primitives from the imported scene, completely and forever.
 * In many cases you'll probably want to set this setting to
 * @code
 * aiPrimitiveType_LINE|aiPrimitiveType_POINT
 * @endcode
 * Together with the #aiProcess_Triangulate flag you can then be sure that
 * #aiFace::mNumIndices is always 3.
 * @note Take a look at the @link data Data Structures page @endlink for
 * more information on the layout and winding order of a face.
 */
struct aiFace
{
    //! Number of indices defining this face.
    //! The maximum value for this member is #AI_MAX_FACE_INDICES.
    unsigned int mNumIndices;

    //! Pointer to the indices array. Size of the array is given in numIndices.
    unsigned int* mIndices;

#ifdef __cplusplus

    //! Default constructor
    aiFace() AI_NO_EXCEPT
    : mNumIndices( 0 )
    , mIndices( nullptr ) {
        // empty
    }

    //! Default destructor. Delete the index array
    ~aiFace()
    {
        delete [] mIndices;
    }

    //! Copy constructor. Copy the index array
    aiFace( const aiFace& o)
    : mNumIndices(0)
    , mIndices( nullptr ) {
        *this = o;
    }

    //! Assignment operator. Copy the index array
    aiFace& operator = ( const aiFace& o) {
        if (&o == this) {
            return *this;
        }

        delete[] mIndices;
        mNumIndices = o.mNumIndices;
        if (mNumIndices) {
            mIndices = new unsigned int[mNumIndices];
            ::memcpy( mIndices, o.mIndices, mNumIndices * sizeof( unsigned int));
        } else {
            mIndices = nullptr;
        }

        return *this;
    }

    //! Comparison operator. Checks whether the index array
    //! of two faces is identical
    bool operator== (const aiFace& o) const {
        if (mIndices == o.mIndices) {
            return true;
        }

        if (nullptr != mIndices && mNumIndices != o.mNumIndices) {
            return false;
        }

        if (nullptr == mIndices) {
            return false;
        }

        for (unsigned int i = 0; i < this->mNumIndices; ++i) {
            if (mIndices[i] != o.mIndices[i]) {
                return false;
            }
        }

        return true;
    }

    //! Inverse comparison operator. Checks whether the index
    //! array of two faces is NOT identical
    bool operator != (const aiFace& o) const {
        return !(*this == o);
    }
#endif // __cplusplus
}; // struct aiFace


// ---------------------------------------------------------------------------
/** @brief A single influence of a bone on a vertex.
 */
struct aiVertexWeight {
    //! Index of the vertex which is influenced by the bone.
    unsigned int mVertexId;

    //! The strength of the influence in the range (0...1).
    //! The influence from all bones at one vertex amounts to 1.
    float mWeight;

#ifdef __cplusplus

    //! Default constructor
    aiVertexWeight() AI_NO_EXCEPT
    : mVertexId(0)
    , mWeight(0.0f) {
        // empty
    }

    //! Initialization from a given index and vertex weight factor
    //! \param pID ID
    //! \param pWeight Vertex weight factor
    aiVertexWeight( unsigned int pID, float pWeight )
    : mVertexId( pID )
    , mWeight( pWeight ) {
        // empty
    }

    bool operator == ( const aiVertexWeight &rhs ) const {
        return ( mVertexId == rhs.mVertexId && mWeight == rhs.mWeight );
    }

    bool operator != ( const aiVertexWeight &rhs ) const {
        return ( *this == rhs );
    }

#endif // __cplusplus
};


// ---------------------------------------------------------------------------
/** @brief A single bone of a mesh.
 *
 *  A bone has a name by which it can be found in the frame hierarchy and by
 *  which it can be addressed by animations. In addition it has a number of
 *  influences on vertices, and a matrix relating the mesh position to the
 *  position of the bone at the time of binding.
 */
struct aiBone {
    //! The name of the bone.
    C_STRUCT aiString mName;

    //! The number of vertices affected by this bone.
    //! The maximum value for this member is #AI_MAX_BONE_WEIGHTS.
    unsigned int mNumWeights;

    //! The influence weights of this bone, by vertex index.
    C_STRUCT aiVertexWeight* mWeights;

    /** Matrix that transforms from bone space to mesh space in bind pose.
     *
     * This matrix describes the position of the mesh
     * in the local space of this bone when the skeleton was bound.
     * Thus it can be used directly to determine a desired vertex position,
     * given the world-space transform of the bone when animated,
     * and the position of the vertex in mesh space.
     *
     * It is sometimes called an inverse-bind matrix,
     * or inverse bind pose matrix.
     */
    C_STRUCT aiMatrix4x4 mOffsetMatrix;

#ifdef __cplusplus

    //! Default constructor
    aiBone() AI_NO_EXCEPT
    : mName()
    , mNumWeights( 0 )
    , mWeights( nullptr )
    , mOffsetMatrix() {
        // empty
    }

    //! Copy constructor
    aiBone(const aiBone& other)
    : mName( other.mName )
    , mNumWeights( other.mNumWeights )
    , mWeights(nullptr)
    , mOffsetMatrix( other.mOffsetMatrix ) {
        if (other.mWeights && other.mNumWeights) {
            mWeights = new aiVertexWeight[mNumWeights];
            ::memcpy(mWeights,other.mWeights,mNumWeights * sizeof(aiVertexWeight));
        }
    }


    //! Assignment operator
    aiBone &operator=(const aiBone& other) {
        if (this == &other) {
            return *this;
        }

        mName         = other.mName;
        mNumWeights   = other.mNumWeights;
        mOffsetMatrix = other.mOffsetMatrix;

        if (other.mWeights && other.mNumWeights)
        {
            if (mWeights) {
                delete[] mWeights;
            }

            mWeights = new aiVertexWeight[mNumWeights];
            ::memcpy(mWeights,other.mWeights,mNumWeights * sizeof(aiVertexWeight));
        }

        return *this;
    }

    bool operator == ( const aiBone &rhs ) const {
        if ( mName != rhs.mName || mNumWeights != rhs.mNumWeights ) {
            return false;
        }

        for ( size_t i = 0; i < mNumWeights; ++i ) {
            if ( mWeights[ i ] != rhs.mWeights[ i ] ) {
                return false;
            }
        }

        return true;
    }
    //! Destructor - deletes the array of vertex weights
    ~aiBone() {
        delete [] mWeights;
    }
#endif // __cplusplus
};


// ---------------------------------------------------------------------------
/** @brief Enumerates the types of geometric primitives supported by Assimp.
 *
 *  @see aiFace Face data structure
 *  @see aiProcess_SortByPType Per-primitive sorting of meshes
 *  @see aiProcess_Triangulate Automatic triangulation
 *  @see AI_CONFIG_PP_SBP_REMOVE Removal of specific primitive types.
 */
enum aiPrimitiveType
{
    /** A point primitive.
     *
     * This is just a single vertex in the virtual world,
     * #aiFace contains just one index for such a primitive.
     */
    aiPrimitiveType_POINT       = 0x1,

    /** A line primitive.
     *
     * This is a line defined through a start and an end position.
     * #aiFace contains exactly two indices for such a primitive.
     */
    aiPrimitiveType_LINE        = 0x2,

    /** A triangular primitive.
     *
     * A triangle consists of three indices.
     */
    aiPrimitiveType_TRIANGLE    = 0x4,

    /** A higher-level polygon with more than 3 edges.
     *
     * A triangle is a polygon, but polygon in this context means
     * "all polygons that are not triangles". The "Triangulate"-Step
     * is provided for your convenience, it splits all polygons in
     * triangles (which are much easier to handle).
     */
    aiPrimitiveType_POLYGON     = 0x8,


    /** This value is not used. It is just here to force the
     *  compiler to map this enum to a 32 Bit integer.
     */
#ifndef SWIG
    _aiPrimitiveType_Force32Bit = INT_MAX
#endif
}; //! enum aiPrimitiveType

// Get the #aiPrimitiveType flag for a specific number of face indices
#define AI_PRIMITIVE_TYPE_FOR_N_INDICES(n) \
    ((n) > 3 ? aiPrimitiveType_POLYGON : (aiPrimitiveType)(1u << ((n)-1)))



// ---------------------------------------------------------------------------
/** @brief An AnimMesh is an attachment to an #aiMesh stores per-vertex
 *  animations for a particular frame.
 *
 *  You may think of an #aiAnimMesh as a `patch` for the host mesh, which
 *  replaces only certain vertex data streams at a particular time.
 *  Each mesh stores n attached attached meshes (#aiMesh::mAnimMeshes).
 *  The actual relationship between the time line and anim meshes is
 *  established by #aiMeshAnim, which references singular mesh attachments
 *  by their ID and binds them to a time offset.
*/
struct aiAnimMesh
{
    /**Anim Mesh name */
    C_STRUCT aiString mName;

    /** Replacement for aiMesh::mVertices. If this array is non-NULL,
     *  it *must* contain mNumVertices entries. The corresponding
     *  array in the host mesh must be non-NULL as well - animation
     *  meshes may neither add or nor remove vertex components (if
     *  a replacement array is NULL and the corresponding source
     *  array is not, the source data is taken instead)*/
    C_STRUCT aiVector3D* mVertices;

    /** Replacement for aiMesh::mNormals.  */
    C_STRUCT aiVector3D* mNormals;

    /** Replacement for aiMesh::mTangents. */
    C_STRUCT aiVector3D* mTangents;

    /** Replacement for aiMesh::mBitangents. */
    C_STRUCT aiVector3D* mBitangents;

    /** Replacement for aiMesh::mColors */
    C_STRUCT aiColor4D* mColors[AI_MAX_NUMBER_OF_COLOR_SETS];

    /** Replacement for aiMesh::mTextureCoords */
    C_STRUCT aiVector3D* mTextureCoords[AI_MAX_NUMBER_OF_TEXTURECOORDS];

    /** The number of vertices in the aiAnimMesh, and thus the length of all
     * the member arrays.
     *
     * This has always the same value as the mNumVertices property in the
     * corresponding aiMesh. It is duplicated here merely to make the length
     * of the member arrays accessible even if the aiMesh is not known, e.g.
     * from language bindings.
     */
    unsigned int mNumVertices;
    
    /** 
     * Weight of the AnimMesh. 
     */
    float mWeight;

#ifdef __cplusplus

    aiAnimMesh() AI_NO_EXCEPT
        : mVertices( nullptr )
        , mNormals(nullptr)
        , mTangents(nullptr)
        , mBitangents(nullptr)
        , mColors()
        , mTextureCoords()
        , mNumVertices( 0 )
        , mWeight( 0.0f )
    {
        // fixme consider moving this to the ctor initializer list as well
        for( unsigned int a = 0; a < AI_MAX_NUMBER_OF_TEXTURECOORDS; a++){
            mTextureCoords[a] = nullptr;
        }
        for( unsigned int a = 0; a < AI_MAX_NUMBER_OF_COLOR_SETS; a++) {
            mColors[a] = nullptr;
        }
    }

    ~aiAnimMesh()
    {
        delete [] mVertices;
        delete [] mNormals;
        delete [] mTangents;
        delete [] mBitangents;
        for( unsigned int a = 0; a < AI_MAX_NUMBER_OF_TEXTURECOORDS; a++) {
            delete [] mTextureCoords[a];
        }
        for( unsigned int a = 0; a < AI_MAX_NUMBER_OF_COLOR_SETS; a++) {
            delete [] mColors[a];
        }
    }

    /** Check whether the anim mesh overrides the vertex positions
     *  of its host mesh*/
    bool HasPositions() const {
        return mVertices != nullptr;
    }

    /** Check whether the anim mesh overrides the vertex normals
     *  of its host mesh*/
    bool HasNormals() const {
        return mNormals != nullptr;
    }

    /** Check whether the anim mesh overrides the vertex tangents
     *  and bitangents of its host mesh. As for aiMesh,
     *  tangents and bitangents always go together. */
    bool HasTangentsAndBitangents() const {
        return mTangents != nullptr;
    }

    /** Check whether the anim mesh overrides a particular
     * set of vertex colors on his host mesh.
     *  @param pIndex 0<index<AI_MAX_NUMBER_OF_COLOR_SETS */
    bool HasVertexColors( unsigned int pIndex) const    {
        return pIndex >= AI_MAX_NUMBER_OF_COLOR_SETS ? false : mColors[pIndex] != nullptr;
    }

    /** Check whether the anim mesh overrides a particular
     * set of texture coordinates on his host mesh.
     *  @param pIndex 0<index<AI_MAX_NUMBER_OF_TEXTURECOORDS */
    bool HasTextureCoords( unsigned int pIndex) const   {
        return pIndex >= AI_MAX_NUMBER_OF_TEXTURECOORDS ? false : mTextureCoords[pIndex] != nullptr;
    }

#endif
};

// ---------------------------------------------------------------------------
/** @brief Enumerates the methods of mesh morphing supported by Assimp.
 */
enum aiMorphingMethod
{
    /** Interpolation between morph targets */
    aiMorphingMethod_VERTEX_BLEND       = 0x1,

    /** Normalized morphing between morph targets  */
    aiMorphingMethod_MORPH_NORMALIZED   = 0x2,

    /** Relative morphing between morph targets  */
    aiMorphingMethod_MORPH_RELATIVE     = 0x3,

    /** This value is not used. It is just here to force the
     *  compiler to map this enum to a 32 Bit integer.
     */
#ifndef SWIG
    _aiMorphingMethod_Force32Bit = INT_MAX
#endif
}; //! enum aiMorphingMethod

// ---------------------------------------------------------------------------
/** @brief A mesh represents a geometry or model with a single material.
*
* It usually consists of a number of vertices and a series of primitives/faces
* referencing the vertices. In addition there might be a series of bones, each
* of them addressing a number of vertices with a certain weight. Vertex data
* is presented in channels with each channel containing a single per-vertex
* information such as a set of texture coords or a normal vector.
* If a data pointer is non-null, the corresponding data stream is present.
* From C++-programs you can also use the comfort functions Has*() to
* test for the presence of various data streams.
*
* A Mesh uses only a single material which is referenced by a material ID.
* @note The mPositions member is usually not optional. However, vertex positions
* *could* be missing if the #AI_SCENE_FLAGS_INCOMPLETE flag is set in
* @code
* aiScene::mFlags
* @endcode
*/
struct aiMesh
{
    /** Bitwise combination of the members of the #aiPrimitiveType enum.
     * This specifies which types of primitives are present in the mesh.
     * The "SortByPrimitiveType"-Step can be used to make sure the
     * output meshes consist of one primitive type each.
     */
    unsigned int mPrimitiveTypes;

    /** The number of vertices in this mesh.
    * This is also the size of all of the per-vertex data arrays.
    * The maximum value for this member is #AI_MAX_VERTICES.
    */
    unsigned int mNumVertices;

    /** The number of primitives (triangles, polygons, lines) in this  mesh.
    * This is also the size of the mFaces array.
    * The maximum value for this member is #AI_MAX_FACES.
    */
    unsigned int mNumFaces;

    /** Vertex positions.
    * This array is always present in a mesh. The array is
    * mNumVertices in size.
    */
    C_STRUCT aiVector3D* mVertices;

    /** Vertex normals.
    * The array contains normalized vectors, NULL if not present.
    * The array is mNumVertices in size. Normals are undefined for
    * point and line primitives. A mesh consisting of points and
    * lines only may not have normal vectors. Meshes with mixed
    * primitive types (i.e. lines and triangles) may have normals,
    * but the normals for vertices that are only referenced by
    * point or line primitives are undefined and set to QNaN (WARN:
    * qNaN compares to inequal to *everything*, even to qNaN itself.
    * Using code like this to check whether a field is qnan is:
    * @code
    * #define IS_QNAN(f) (f != f)
    * @endcode
    * still dangerous because even 1.f == 1.f could evaluate to false! (
    * remember the subtleties of IEEE754 artithmetics). Use stuff like
    * @c fpclassify instead.
    * @note Normal vectors computed by Assimp are always unit-length.
    * However, this needn't apply for normals that have been taken
    *   directly from the model file.
    */
    C_STRUCT aiVector3D* mNormals;

    /** Vertex tangents.
    * The tangent of a vertex points in the direction of the positive
    * X texture axis. The array contains normalized vectors, NULL if
    * not present. The array is mNumVertices in size. A mesh consisting
    * of points and lines only may not have normal vectors. Meshes with
    * mixed primitive types (i.e. lines and triangles) may have
    * normals, but the normals for vertices that are only referenced by
    * point or line primitives are undefined and set to qNaN.  See
    * the #mNormals member for a detailed discussion of qNaNs.
    * @note If the mesh contains tangents, it automatically also
    * contains bitangents.
    */
    C_STRUCT aiVector3D* mTangents;

    /** Vertex bitangents.
    * The bitangent of a vertex points in the direction of the positive
    * Y texture axis. The array contains normalized vectors, NULL if not
    * present. The array is mNumVertices in size.
    * @note If the mesh contains tangents, it automatically also contains
    * bitangents.
    */
    C_STRUCT aiVector3D* mBitangents;

    /** Vertex color sets.
    * A mesh may contain 0 to #AI_MAX_NUMBER_OF_COLOR_SETS vertex
    * colors per vertex. NULL if not present. Each array is
    * mNumVertices in size if present.
    */
    C_STRUCT aiColor4D* mColors[AI_MAX_NUMBER_OF_COLOR_SETS];

    /** Vertex texture coords, also known as UV channels.
    * A mesh may contain 0 to AI_MAX_NUMBER_OF_TEXTURECOORDS per
    * vertex. NULL if not present. The array is mNumVertices in size.
    */
    C_STRUCT aiVector3D* mTextureCoords[AI_MAX_NUMBER_OF_TEXTURECOORDS];

    /** Specifies the number of components for a given UV channel.
    * Up to three channels are supported (UVW, for accessing volume
    * or cube maps). If the value is 2 for a given channel n, the
    * component p.z of mTextureCoords[n][p] is set to 0.0f.
    * If the value is 1 for a given channel, p.y is set to 0.0f, too.
    * @note 4D coords are not supported
    */
    unsigned int mNumUVComponents[AI_MAX_NUMBER_OF_TEXTURECOORDS];

    /** The faces the mesh is constructed from.
    * Each face refers to a number of vertices by their indices.
    * This array is always present in a mesh, its size is given
    * in mNumFaces. If the #AI_SCENE_FLAGS_NON_VERBOSE_FORMAT
    * is NOT set each face references an unique set of vertices.
    */
    C_STRUCT aiFace* mFaces;

    /** The number of bones this mesh contains.
    * Can be 0, in which case the mBones array is NULL.
    */
    unsigned int mNumBones;

    /** The bones of this mesh.
    * A bone consists of a name by which it can be found in the
    * frame hierarchy and a set of vertex weights.
    */
    C_STRUCT aiBone** mBones;

    /** The material used by this mesh.
     * A mesh uses only a single material. If an imported model uses
     * multiple materials, the import splits up the mesh. Use this value
     * as index into the scene's material list.
     */
    unsigned int mMaterialIndex;

    /** Name of the mesh. Meshes can be named, but this is not a
     *  requirement and leaving this field empty is totally fine.
     *  There are mainly three uses for mesh names:
     *   - some formats name nodes and meshes independently.
     *   - importers tend to split meshes up to meet the
     *      one-material-per-mesh requirement. Assigning
     *      the same (dummy) name to each of the result meshes
     *      aids the caller at recovering the original mesh
     *      partitioning.
     *   - Vertex animations refer to meshes by their names.
     **/
    C_STRUCT aiString mName;


    /** The number of attachment meshes. Note! Currently only works with Collada loader. */
    unsigned int mNumAnimMeshes;

    /** Attachment meshes for this mesh, for vertex-based animation.
     *  Attachment meshes carry replacement data for some of the
     *  mesh'es vertex components (usually positions, normals).
     *  Note! Currently only works with Collada loader.*/
    C_STRUCT aiAnimMesh** mAnimMeshes;

    /** 
     *  Method of morphing when animeshes are specified. 
     */
    unsigned int mMethod;

    /**
     *
     */
    C_STRUCT aiAABB mAABB;
	
#ifdef __cplusplus

    //! Default constructor. Initializes all members to 0
    aiMesh() AI_NO_EXCEPT
    : mPrimitiveTypes( 0 )
    , mNumVertices( 0 )
    , mNumFaces( 0 )
    , mVertices( nullptr )
    , mNormals(nullptr)
    , mTangents(nullptr)
    , mBitangents(nullptr)
    , mColors()
    , mTextureCoords()
    , mNumUVComponents()
    , mFaces(nullptr)
    , mNumBones( 0 )
    , mBones(nullptr)
    , mMaterialIndex( 0 )
    , mNumAnimMeshes( 0 )
    , mAnimMeshes(nullptr)
    , mMethod( 0 )
    , mAABB() {
        for( unsigned int a = 0; a < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++a ) {
            mNumUVComponents[a] = 0;
            mTextureCoords[a] = nullptr;
        }

        for (unsigned int a = 0; a < AI_MAX_NUMBER_OF_COLOR_SETS; ++a) {
            mColors[a] = nullptr;
        }
    }

    //! Deletes all storage allocated for the mesh
    ~aiMesh() {
        delete [] mVertices;
        delete [] mNormals;
        delete [] mTangents;
        delete [] mBitangents;
        for( unsigned int a = 0; a < AI_MAX_NUMBER_OF_TEXTURECOORDS; a++) {
            delete [] mTextureCoords[a];
        }
        for( unsigned int a = 0; a < AI_MAX_NUMBER_OF_COLOR_SETS; a++) {
            delete [] mColors[a];
        }

        // DO NOT REMOVE THIS ADDITIONAL CHECK
        if (mNumBones && mBones)    {
            for( unsigned int a = 0; a < mNumBones; a++) {
                delete mBones[a];
            }
            delete [] mBones;
        }

        if (mNumAnimMeshes && mAnimMeshes)  {
            for( unsigned int a = 0; a < mNumAnimMeshes; a++) {
                delete mAnimMeshes[a];
            }
            delete [] mAnimMeshes;
        }

        delete [] mFaces;
    }

    //! Check whether the mesh contains positions. Provided no special
    //! scene flags are set, this will always be true
    bool HasPositions() const
        { return mVertices != nullptr && mNumVertices > 0; }

    //! Check whether the mesh contains faces. If no special scene flags
    //! are set this should always return true
    bool HasFaces() const
        { return mFaces != nullptr && mNumFaces > 0; }

    //! Check whether the mesh contains normal vectors
    bool HasNormals() const
        { return mNormals != nullptr && mNumVertices > 0; }

    //! Check whether the mesh contains tangent and bitangent vectors
    //! It is not possible that it contains tangents and no bitangents
    //! (or the other way round). The existence of one of them
    //! implies that the second is there, too.
    bool HasTangentsAndBitangents() const
        { return mTangents != nullptr && mBitangents != nullptr && mNumVertices > 0; }

    //! Check whether the mesh contains a vertex color set
    //! \param pIndex Index of the vertex color set
    bool HasVertexColors( unsigned int pIndex) const {
        if (pIndex >= AI_MAX_NUMBER_OF_COLOR_SETS) {
            return false;
        } else {
            return mColors[pIndex] != nullptr && mNumVertices > 0;
        }
    }

    //! Check whether the mesh contains a texture coordinate set
    //! \param pIndex Index of the texture coordinates set
    bool HasTextureCoords( unsigned int pIndex) const {
        if (pIndex >= AI_MAX_NUMBER_OF_TEXTURECOORDS) {
            return false;
        } else {
            return mTextureCoords[pIndex] != nullptr && mNumVertices > 0;
        }
    }

    //! Get the number of UV channels the mesh contains
    unsigned int GetNumUVChannels() const {
        unsigned int n( 0 );
        while (n < AI_MAX_NUMBER_OF_TEXTURECOORDS && mTextureCoords[n]) {
            ++n;
        }

        return n;
    }

    //! Get the number of vertex color channels the mesh contains
    unsigned int GetNumColorChannels() const {
        unsigned int n(0);
        while (n < AI_MAX_NUMBER_OF_COLOR_SETS && mColors[n]) {
            ++n;
        }
        return n;
    }

    //! Check whether the mesh contains bones
    bool HasBones() const {
        return mBones != nullptr && mNumBones > 0;
    }

#endif // __cplusplus
};

#ifdef __cplusplus
}
#endif //! extern "C"
#endif // AI_MESH_H_INC

