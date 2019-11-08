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

/** @file glTFAsset.h
 * Declares a glTF class to handle gltf/glb files
 *
 * glTF Extensions Support:
 *   KHR_binary_glTF: full
 *   KHR_materials_common: full
 */
#ifndef GLTFASSET_H_INC
#define GLTFASSET_H_INC

#ifndef ASSIMP_BUILD_NO_GLTF_IMPORTER

#include <assimp/Exceptional.h>

#include <map>
#include <string>
#include <list>
#include <vector>
#include <algorithm>
#include <stdexcept>

#define RAPIDJSON_HAS_STDSTRING 1
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#ifdef ASSIMP_API
#   include <memory>
#   include <assimp/DefaultIOSystem.h>
#   include <assimp/ByteSwapper.h>
#else
#   include <memory>
#   define AI_SWAP4(p)
#   define ai_assert
#endif


#if _MSC_VER > 1500 || (defined __GNUC___)
#       define ASSIMP_GLTF_USE_UNORDERED_MULTIMAP
#   else
#       define gltf_unordered_map map
#endif

#ifdef ASSIMP_GLTF_USE_UNORDERED_MULTIMAP
#   include <unordered_map>
#   if _MSC_VER > 1600
#       define gltf_unordered_map unordered_map
#   else
#       define gltf_unordered_map tr1::unordered_map
#   endif
#endif

#include "glTF/glTFCommon.h"

namespace glTF
{
    using glTFCommon::shared_ptr;
    using glTFCommon::IOSystem;
    using glTFCommon::IOStream;

    using rapidjson::Value;
    using rapidjson::Document;

    class Asset;
    class AssetWriter;

    struct BufferView; // here due to cross-reference
    struct Texture;
    struct Light;
    struct Skin;

    using glTFCommon::vec3;
    using glTFCommon::vec4;
    using glTFCommon::mat4;

    //! Magic number for GLB files
    #define AI_GLB_MAGIC_NUMBER "glTF"

    #ifdef ASSIMP_API
        #include <assimp/Compiler/pushpack1.h>
    #endif

    //! For the KHR_binary_glTF extension (binary .glb file)
    //! 20-byte header (+ the JSON + a "body" data section)
    struct GLB_Header
    {
        uint8_t magic[4];     //!< Magic number: "glTF"
        uint32_t version;     //!< Version number (always 1 as of the last update)
        uint32_t length;      //!< Total length of the Binary glTF, including header, scene, and body, in bytes
        uint32_t sceneLength; //!< Length, in bytes, of the glTF scene
        uint32_t sceneFormat; //!< Specifies the format of the glTF scene (see the SceneFormat enum)
    } PACK_STRUCT;

    #ifdef ASSIMP_API
        #include <assimp/Compiler/poppack1.h>
    #endif


    //! Values for the GLB_Header::sceneFormat field
    enum SceneFormat
    {
        SceneFormat_JSON = 0
    };

    //! Values for the mesh primitive modes
    enum PrimitiveMode
    {
        PrimitiveMode_POINTS = 0,
        PrimitiveMode_LINES = 1,
        PrimitiveMode_LINE_LOOP = 2,
        PrimitiveMode_LINE_STRIP = 3,
        PrimitiveMode_TRIANGLES = 4,
        PrimitiveMode_TRIANGLE_STRIP = 5,
        PrimitiveMode_TRIANGLE_FAN = 6
    };

    //! Values for the Accessor::componentType field
    enum ComponentType
    {
        ComponentType_BYTE = 5120,
        ComponentType_UNSIGNED_BYTE = 5121,
        ComponentType_SHORT = 5122,
        ComponentType_UNSIGNED_SHORT = 5123,
        ComponentType_UNSIGNED_INT = 5125,
        ComponentType_FLOAT = 5126
    };

    inline unsigned int ComponentTypeSize(ComponentType t)
    {
        switch (t) {
            case ComponentType_SHORT:
            case ComponentType_UNSIGNED_SHORT:
                return 2;

            case ComponentType_UNSIGNED_INT:
            case ComponentType_FLOAT:
                return 4;

            case ComponentType_BYTE:
            case ComponentType_UNSIGNED_BYTE:
                return 1;
            default:
                std::string err = "GLTF: Unsupported Component Type ";
                err += t;
                throw DeadlyImportError(err);
        }
    }

    //! Values for the BufferView::target field
    enum BufferViewTarget
    {
        BufferViewTarget_ARRAY_BUFFER = 34962,
        BufferViewTarget_ELEMENT_ARRAY_BUFFER = 34963
    };

    //! Values for the Sampler::magFilter field
    enum SamplerMagFilter
    {
        SamplerMagFilter_Nearest = 9728,
        SamplerMagFilter_Linear = 9729
    };

    //! Values for the Sampler::minFilter field
    enum SamplerMinFilter
    {
        SamplerMinFilter_Nearest = 9728,
        SamplerMinFilter_Linear = 9729,
        SamplerMinFilter_Nearest_Mipmap_Nearest = 9984,
        SamplerMinFilter_Linear_Mipmap_Nearest = 9985,
        SamplerMinFilter_Nearest_Mipmap_Linear = 9986,
        SamplerMinFilter_Linear_Mipmap_Linear = 9987
    };

    //! Values for the Sampler::wrapS and Sampler::wrapT field
    enum SamplerWrap
    {
        SamplerWrap_Clamp_To_Edge = 33071,
        SamplerWrap_Mirrored_Repeat = 33648,
        SamplerWrap_Repeat = 10497
    };

    //! Values for the Texture::format and Texture::internalFormat fields
    enum TextureFormat
    {
        TextureFormat_ALPHA = 6406,
        TextureFormat_RGB = 6407,
        TextureFormat_RGBA = 6408,
        TextureFormat_LUMINANCE = 6409,
        TextureFormat_LUMINANCE_ALPHA = 6410
    };

    //! Values for the Texture::target field
    enum TextureTarget
    {
        TextureTarget_TEXTURE_2D = 3553
    };

    //! Values for the Texture::type field
    enum TextureType
    {
        TextureType_UNSIGNED_BYTE = 5121,
        TextureType_UNSIGNED_SHORT_5_6_5 = 33635,
        TextureType_UNSIGNED_SHORT_4_4_4_4 = 32819,
        TextureType_UNSIGNED_SHORT_5_5_5_1 = 32820
    };


    //! Values for the Accessor::type field (helper class)
    class AttribType
    {
    public:
        enum Value
            { SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4 };

    private:
        static const size_t NUM_VALUES = static_cast<size_t>(MAT4)+1;

        struct Info
            { const char* name; unsigned int numComponents; };

        template<int N> struct data
            { static const Info infos[NUM_VALUES]; };

    public:
        inline static Value FromString(const char* str)
        {
            for (size_t i = 0; i < NUM_VALUES; ++i) {
                if (strcmp(data<0>::infos[i].name, str) == 0) {
                    return static_cast<Value>(i);
                }
            }
            return SCALAR;
        }

        inline static const char* ToString(Value type)
        {
            return data<0>::infos[static_cast<size_t>(type)].name;
        }

        inline static unsigned int GetNumComponents(Value type)
        {
            return data<0>::infos[static_cast<size_t>(type)].numComponents;
        }
    };

    // must match the order of the AttribTypeTraits::Value enum!
    template<int N> const AttribType::Info
    AttribType::data<N>::infos[AttribType::NUM_VALUES] = {
        { "SCALAR", 1 }, { "VEC2", 2 }, { "VEC3", 3 }, { "VEC4", 4 }, { "MAT2", 4 }, { "MAT3", 9 }, { "MAT4", 16 }
    };



    //! A reference to one top-level object, which is valid
    //! until the Asset instance is destroyed
    template<class T>
    class Ref
    {
        std::vector<T*>* vector;
        unsigned int index;

    public:
        Ref() : vector(0), index(0) {}
        Ref(std::vector<T*>& vec, unsigned int idx) : vector(&vec), index(idx) {}

        inline unsigned int GetIndex() const
            { return index; }

        operator bool() const
            { return vector != 0; }

        T* operator->()
            { return (*vector)[index]; }

        T& operator*()
            { return *((*vector)[index]); }
    };

    //! Helper struct to represent values that might not be present
    template<class T>
    struct Nullable
    {
        T value;
        bool isPresent;

        Nullable() : isPresent(false) {}
        Nullable(T& val) : value(val), isPresent(true) {}
    };


    //! Base class for all glTF top-level objects
    struct Object
    {
        std::string id;   //!< The globally unique ID used to reference this object
        std::string name; //!< The user-defined name of this object

        //! Objects marked as special are not exported (used to emulate the binary body buffer)
        virtual bool IsSpecial() const
            { return false; }

        virtual ~Object() {}

        //! Maps special IDs to another ID, where needed. Subclasses may override it (statically)
        static const char* TranslateId(Asset& /*r*/, const char* id)
            { return id; }
    };

    //
    // Classes for each glTF top-level object type
    //

    //! A typed view into a BufferView. A BufferView contains raw binary data.
    //! An accessor provides a typed view into a BufferView or a subset of a BufferView
    //! similar to how WebGL's vertexAttribPointer() defines an attribute in a buffer.
    struct Accessor : public Object
    {
        Ref<BufferView> bufferView;  //!< The ID of the bufferView. (required)
        unsigned int byteOffset;     //!< The offset relative to the start of the bufferView in bytes. (required)
        unsigned int byteStride;     //!< The stride, in bytes, between attributes referenced by this accessor. (default: 0)
        ComponentType componentType; //!< The datatype of components in the attribute. (required)
        unsigned int count;          //!< The number of attributes referenced by this accessor. (required)
        AttribType::Value type;      //!< Specifies if the attribute is a scalar, vector, or matrix. (required)
        std::vector<float> max;      //!< Maximum value of each component in this attribute.
        std::vector<float> min;      //!< Minimum value of each component in this attribute.

        unsigned int GetNumComponents();
        unsigned int GetBytesPerComponent();
        unsigned int GetElementSize();

        inline uint8_t* GetPointer();

        template<class T>
        bool ExtractData(T*& outData);

        void WriteData(size_t count, const void* src_buffer, size_t src_stride);

        //! Helper class to iterate the data
        class Indexer
        {
            friend struct Accessor;

            Accessor& accessor;
            uint8_t* data;
            size_t elemSize, stride;

            Indexer(Accessor& acc);

        public:

            //! Accesses the i-th value as defined by the accessor
            template<class T>
            T GetValue(int i);

            //! Accesses the i-th value as defined by the accessor
            inline unsigned int GetUInt(int i)
            {
                return GetValue<unsigned int>(i);
            }

            inline bool IsValid() const
            {
                return data != 0;
            }
        };

        inline Indexer GetIndexer()
        {
            return Indexer(*this);
        }

        Accessor() {}
        void Read(Value& obj, Asset& r);
    };

    //! A buffer points to binary geometry, animation, or skins.
    struct Buffer : public Object
	{
		/********************* Types *********************/
	public:

		enum Type
		{
			Type_arraybuffer,
			Type_text
		};

		/// \struct SEncodedRegion
		/// Descriptor of encoded region in "bufferView".
		struct SEncodedRegion
		{
			const size_t Offset;///< Offset from begin of "bufferView" to encoded region, in bytes.
			const size_t EncodedData_Length;///< Size of encoded region, in bytes.
			uint8_t* const DecodedData;///< Cached encoded data.
			const size_t DecodedData_Length;///< Size of decoded region, in bytes.
			const std::string ID;///< ID of the region.

			/// \fn SEncodedRegion(const size_t pOffset, const size_t pEncodedData_Length, uint8_t* pDecodedData, const size_t pDecodedData_Length, const std::string pID)
			/// Constructor.
			/// \param [in] pOffset - offset from begin of "bufferView" to encoded region, in bytes.
			/// \param [in] pEncodedData_Length - size of encoded region, in bytes.
			/// \param [in] pDecodedData - pointer to decoded data array.
			/// \param [in] pDecodedData_Length - size of encoded region, in bytes.
			/// \param [in] pID - ID of the region.
			SEncodedRegion(const size_t pOffset, const size_t pEncodedData_Length, uint8_t* pDecodedData, const size_t pDecodedData_Length, const std::string pID)
				: Offset(pOffset), EncodedData_Length(pEncodedData_Length), DecodedData(pDecodedData), DecodedData_Length(pDecodedData_Length), ID(pID)
			{}

			/// \fn ~SEncodedRegion()
			/// Destructor.
			~SEncodedRegion() { delete [] DecodedData; }
		};

		/******************* Variables *******************/

		//std::string uri; //!< The uri of the buffer. Can be a filepath, a data uri, etc. (required)
		size_t byteLength; //!< The length of the buffer in bytes. (default: 0)
		//std::string type; //!< XMLHttpRequest responseType (default: "arraybuffer")

		Type type;

		/// \var EncodedRegion_Current
		/// Pointer to currently active encoded region.
		/// Why not decoding all regions at once and not to set one buffer with decoded data?
		/// Yes, why not? Even "accessor" point to decoded data. I mean that fields "byteOffset", "byteStride" and "count" has values which describes decoded
		/// data array. But only in range of mesh while is active parameters from "compressedData". For another mesh accessors point to decoded data too. But
		/// offset is counted for another regions is encoded.
		/// Example. You have two meshes. For every of it you have 4 bytes of data. That data compressed to 2 bytes. So, you have buffer with encoded data:
		/// M1_E0, M1_E1, M2_E0, M2_E1.
		/// After decoding you'll get:
		/// M1_D0, M1_D1, M1_D2, M1_D3, M2_D0, M2_D1, M2_D2, M2_D3.
		/// "accessors" must to use values that point to decoded data - obviously. So, you'll expect "accessors" like
		/// "accessor_0" : { byteOffset: 0, byteLength: 4}, "accessor_1" : { byteOffset: 4, byteLength: 4}
		/// but in real life you'll get:
		/// "accessor_0" : { byteOffset: 0, byteLength: 4}, "accessor_1" : { byteOffset: 2, byteLength: 4}
		/// Yes, accessor of next mesh has offset and length which mean: current mesh data is decoded, all other data is encoded.
		/// And when before you start to read data of current mesh (with encoded data ofcourse) you must decode region of "bufferView", after read finished
		/// delete encoding mark. And after that you can repeat process: decode data of mesh, read, delete decoded data.
		///
		/// Remark. Encoding all data at once is good in world with computers which do not has RAM limitation. So, you must use step by step encoding in
		/// exporter and importer. And, thanks to such way, there is no need to load whole file into memory.
		SEncodedRegion* EncodedRegion_Current;

	private:

		shared_ptr<uint8_t> mData; //!< Pointer to the data
		bool mIsSpecial; //!< Set to true for special cases (e.g. the body buffer)
        size_t capacity = 0; //!< The capacity of the buffer in bytes. (default: 0)
		/// \var EncodedRegion_List
		/// List of encoded regions.
		std::list<SEncodedRegion*> EncodedRegion_List;

		/******************* Functions *******************/

	public:

		Buffer();
		~Buffer();

		void Read(Value& obj, Asset& r);

        bool LoadFromStream(IOStream& stream, size_t length = 0, size_t baseOffset = 0);

		/// \fn void EncodedRegion_Mark(const size_t pOffset, const size_t pEncodedData_Length, uint8_t* pDecodedData, const size_t pDecodedData_Length, const std::string& pID)
		/// Mark region of "bufferView" as encoded. When data is request from such region then "bufferView" use decoded data.
		/// \param [in] pOffset - offset from begin of "bufferView" to encoded region, in bytes.
		/// \param [in] pEncodedData_Length - size of encoded region, in bytes.
		/// \param [in] pDecodedData - pointer to decoded data array.
		/// \param [in] pDecodedData_Length - size of encoded region, in bytes.
		/// \param [in] pID - ID of the region.
		void EncodedRegion_Mark(const size_t pOffset, const size_t pEncodedData_Length, uint8_t* pDecodedData, const size_t pDecodedData_Length, const std::string& pID);

		/// \fn void EncodedRegion_SetCurrent(const std::string& pID)
		/// Select current encoded region by ID. \sa EncodedRegion_Current.
		/// \param [in] pID - ID of the region.
		void EncodedRegion_SetCurrent(const std::string& pID);

		/// \fn bool ReplaceData(const size_t pBufferData_Offset, const size_t pBufferData_Count, const uint8_t* pReplace_Data, const size_t pReplace_Count)
		/// Replace part of buffer data. Pay attention that function work with original array of data (\ref mData) not with encoded regions.
		/// \param [in] pBufferData_Offset - index of first element in buffer from which new data will be placed.
		/// \param [in] pBufferData_Count - count of bytes in buffer which will be replaced.
		/// \param [in] pReplace_Data - pointer to array with new data for buffer.
		/// \param [in] pReplace_Count - count of bytes in new data.
		/// \return true - if successfully replaced, false if input arguments is out of range.
		bool ReplaceData(const size_t pBufferData_Offset, const size_t pBufferData_Count, const uint8_t* pReplace_Data, const size_t pReplace_Count);

        size_t AppendData(uint8_t* data, size_t length);
        void Grow(size_t amount);

        uint8_t* GetPointer()
            { return mData.get(); }

        void MarkAsSpecial()
            { mIsSpecial = true; }

        bool IsSpecial() const
            { return mIsSpecial; }

        std::string GetURI()
            { return std::string(this->id) + ".bin"; }

        static const char* TranslateId(Asset& r, const char* id);
    };

    //! A view into a buffer generally representing a subset of the buffer.
    struct BufferView : public Object
    {
        Ref<Buffer> buffer; //! The ID of the buffer. (required)
        size_t byteOffset; //! The offset into the buffer in bytes. (required)
        size_t byteLength; //! The length of the bufferView in bytes. (default: 0)

        BufferViewTarget target; //! The target that the WebGL buffer should be bound to.

        void Read(Value& obj, Asset& r);
    };

    struct Camera : public Object
    {
        enum Type
        {
            Perspective,
            Orthographic
        };

        Type type;

        union
        {
            struct {
                float aspectRatio; //!<The floating - point aspect ratio of the field of view. (0 = undefined = use the canvas one)
                float yfov;  //!<The floating - point vertical field of view in radians. (required)
                float zfar;  //!<The floating - point distance to the far clipping plane. (required)
                float znear; //!< The floating - point distance to the near clipping plane. (required)
            } perspective;

            struct {
                float xmag;  //! The floating-point horizontal magnification of the view. (required)
                float ymag;  //! The floating-point vertical magnification of the view. (required)
                float zfar;  //! The floating-point distance to the far clipping plane. (required)
                float znear; //! The floating-point distance to the near clipping plane. (required)
            } ortographic;
        };

        Camera() {}
        void Read(Value& obj, Asset& r);
    };


    //! Image data used to create a texture.
    struct Image : public Object
    {
        std::string uri; //! The uri of the image, that can be a file path, a data URI, etc.. (required)

        Ref<BufferView> bufferView;

        std::string mimeType;

        int width, height;

    private:
        std::unique_ptr<uint8_t[]> mData;
        size_t mDataLength;

    public:

        Image();
        void Read(Value& obj, Asset& r);

        inline bool HasData() const
            { return mDataLength > 0; }

        inline size_t GetDataLength() const
            { return mDataLength; }

        inline const uint8_t* GetData() const
            { return mData.get(); }

        inline uint8_t* StealData();

        inline void SetData(uint8_t* data, size_t length, Asset& r);
    };

    //! Holds a material property that can be a texture or a color
    struct TexProperty
    {
        Ref<Texture> texture;
        vec4 color;
    };

    //! The material appearance of a primitive.
    struct Material : public Object
    {
        //Ref<Sampler> source; //!< The ID of the technique.
        //std::gltf_unordered_map<std::string, std::string> values; //!< A dictionary object of parameter values.

        //! Techniques defined by KHR_materials_common
        enum Technique
        {
            Technique_undefined = 0,
            Technique_BLINN,
            Technique_PHONG,
            Technique_LAMBERT,
            Technique_CONSTANT
        };

        TexProperty ambient;
        TexProperty diffuse;
        TexProperty specular;
        TexProperty emission;

        bool doubleSided;
        bool transparent;
        float transparency;
        float shininess;

        Technique technique;

        Material() { SetDefaults(); }
        void Read(Value& obj, Asset& r);
        void SetDefaults();
    };

    //! A set of primitives to be rendered. A node can contain one or more meshes. A node's transform places the mesh in the scene.
    struct Mesh : public Object
    {
        typedef std::vector< Ref<Accessor> > AccessorList;

        struct Primitive
        {
            PrimitiveMode mode;

            struct Attributes {
                AccessorList position, normal, texcoord, color, joint, jointmatrix, weight;
            } attributes;

            Ref<Accessor> indices;

            Ref<Material> material;
        };

		/// \struct SExtension
		/// Extension used for mesh.
		struct SExtension
		{
			/// \enum EType
			/// Type of extension.
			enum EType
			{
				#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
					Compression_Open3DGC,///< Compression of mesh data using Open3DGC algorithm.
				#endif

				Unknown
			};

			EType Type;///< Type of extension.

			/// \fn SExtension
			/// Constructor.
			/// \param [in] pType - type of extension.
			SExtension(const EType pType)
				: Type(pType)
			{}

            virtual ~SExtension() {
                // empty
            }
		};

		#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
			/// \struct SCompression_Open3DGC
			/// Compression of mesh data using Open3DGC algorithm.
			struct SCompression_Open3DGC : public SExtension
			{
				using SExtension::Type;

				std::string Buffer;///< ID of "buffer" used for storing compressed data.
				size_t Offset;///< Offset in "bufferView" where compressed data are stored.
				size_t Count;///< Count of elements in compressed data. Is always equivalent to size in bytes: look comments for "Type" and "Component_Type".
				bool Binary;///< If true then "binary" mode is used for coding, if false - "ascii" mode.
				size_t IndicesCount;///< Count of indices in mesh.
				size_t VerticesCount;///< Count of vertices in mesh.
				// AttribType::Value Type;///< Is always "SCALAR".
				// ComponentType Component_Type;///< Is always "ComponentType_UNSIGNED_BYTE" (5121).

				/// \fn SCompression_Open3DGC
				/// Constructor.
				SCompression_Open3DGC()
				: SExtension(Compression_Open3DGC) {
                    // empty
                }

                virtual ~SCompression_Open3DGC() {
                    // empty
                }
			};
		#endif

        std::vector<Primitive> primitives;
		std::list<SExtension*> Extension;///< List of extensions used in mesh.

        Mesh() {}

		/// \fn ~Mesh()
		/// Destructor.
		~Mesh() { for(std::list<SExtension*>::iterator it = Extension.begin(), it_end = Extension.end(); it != it_end; it++) { delete *it; }; }

		/// \fn void Read(Value& pJSON_Object, Asset& pAsset_Root)
		/// Get mesh data from JSON-object and place them to root asset.
		/// \param [in] pJSON_Object - reference to pJSON-object from which data are read.
		/// \param [out] pAsset_Root - reference to root assed where data will be stored.
		void Read(Value& pJSON_Object, Asset& pAsset_Root);

		#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
			/// \fn void Decode_O3DGC(const SCompression_Open3DGC& pCompression_Open3DGC, Asset& pAsset_Root)
			/// Decode part of "buffer" which encoded with Open3DGC algorithm.
			/// \param [in] pCompression_Open3DGC - reference to structure which describe encoded region.
			/// \param [out] pAsset_Root - reference to root assed where data will be stored.
			void Decode_O3DGC(const SCompression_Open3DGC& pCompression_Open3DGC, Asset& pAsset_Root);
		#endif
    };

    struct Node : public Object
    {
        std::vector< Ref<Node> > children;
        std::vector< Ref<Mesh> > meshes;

        Nullable<mat4> matrix;
        Nullable<vec3> translation;
        Nullable<vec4> rotation;
        Nullable<vec3> scale;

        Ref<Camera> camera;
        Ref<Light>  light;

        std::vector< Ref<Node> > skeletons;       //!< The ID of skeleton nodes. Each of which is the root of a node hierarchy.
        Ref<Skin>  skin;                          //!< The ID of the skin referenced by this node.
        std::string jointName;                    //!< Name used when this node is a joint in a skin.

        Ref<Node> parent;                         //!< This is not part of the glTF specification. Used as a helper.

        Node() {}
        void Read(Value& obj, Asset& r);
    };

    struct Program : public Object
    {
        Program() {}
        void Read(Value& obj, Asset& r);
    };


    struct Sampler : public Object
    {
        SamplerMagFilter magFilter; //!< The texture magnification filter. (required)
        SamplerMinFilter minFilter; //!< The texture minification filter. (required)
        SamplerWrap wrapS;          //!< The texture wrapping in the S direction. (required)
        SamplerWrap wrapT;          //!< The texture wrapping in the T direction. (required)

        Sampler() {}
        void Read(Value& obj, Asset& r);
        void SetDefaults();
    };

    struct Scene : public Object
    {
        std::vector< Ref<Node> > nodes;

        Scene() {}
        void Read(Value& obj, Asset& r);
    };

    struct Shader : public Object
    {
        Shader() {}
        void Read(Value& obj, Asset& r);
    };

    struct Skin : public Object
    {
        Nullable<mat4> bindShapeMatrix;       //!< Floating-point 4x4 transformation matrix stored in column-major order.
        Ref<Accessor> inverseBindMatrices;    //!< The ID of the accessor containing the floating-point 4x4 inverse-bind matrices.
        std::vector<Ref<Node>> jointNames;    //!< Joint names of the joints (nodes with a jointName property) in this skin.
        std::string name;                     //!< The user-defined name of this object.

        Skin() {}
        void Read(Value& obj, Asset& r);
    };

    struct Technique : public Object
    {
        struct Parameters
        {

        };

        struct States
        {

        };

        struct Functions
        {

        };

        Technique() {}
        void Read(Value& obj, Asset& r);
    };

    //! A texture and its sampler.
    struct Texture : public Object
    {
        Ref<Sampler> sampler; //!< The ID of the sampler used by this texture. (required)
        Ref<Image> source;    //!< The ID of the image used by this texture. (required)

        //TextureFormat format; //!< The texture's format. (default: TextureFormat_RGBA)
        //TextureFormat internalFormat; //!< The texture's internal format. (default: TextureFormat_RGBA)

        //TextureTarget target; //!< The target that the WebGL texture should be bound to. (default: TextureTarget_TEXTURE_2D)
        //TextureType type; //!< Texel datatype. (default: TextureType_UNSIGNED_BYTE)

        Texture() {}
        void Read(Value& obj, Asset& r);
    };


    //! A light (from KHR_materials_common extension)
    struct Light : public Object
    {
        enum Type
        {
            Type_undefined,
            Type_ambient,
            Type_directional,
            Type_point,
            Type_spot
        };

        Type type;

        vec4 color;
        float distance;
        float constantAttenuation;
        float linearAttenuation;
        float quadraticAttenuation;
        float falloffAngle;
        float falloffExponent;

        Light() {}
        void Read(Value& obj, Asset& r);

        void SetDefaults();
    };

    struct Animation : public Object
    {
        struct AnimSampler {
            std::string id;               //!< The ID of this sampler.
            std::string input;            //!< The ID of a parameter in this animation to use as key-frame input.
            std::string interpolation;    //!< Type of interpolation algorithm to use between key-frames.
            std::string output;           //!< The ID of a parameter in this animation to use as key-frame output.
        };

        struct AnimChannel {
            std::string sampler;         //!< The ID of one sampler present in the containing animation's samplers property.

            struct AnimTarget {
                Ref<Node> id;            //!< The ID of the node to animate.
                std::string path;        //!< The name of property of the node to animate ("translation", "rotation", or "scale").
            } target;
        };

        struct AnimParameters {
            Ref<Accessor> TIME;           //!< Accessor reference to a buffer storing a array of floating point scalar values.
            Ref<Accessor> rotation;       //!< Accessor reference to a buffer storing a array of four-component floating-point vectors.
            Ref<Accessor> scale;          //!< Accessor reference to a buffer storing a array of three-component floating-point vectors.
            Ref<Accessor> translation;    //!< Accessor reference to a buffer storing a array of three-component floating-point vectors.
        };

        // AnimChannel Channels[3];            //!< Connect the output values of the key-frame animation to a specific node in the hierarchy.
        // AnimParameters Parameters;          //!< The samplers that interpolate between the key-frames.
        // AnimSampler Samplers[3];            //!< The parameterized inputs representing the key-frame data.

        std::vector<AnimChannel> Channels;            //!< Connect the output values of the key-frame animation to a specific node in the hierarchy.
        AnimParameters Parameters;                    //!< The samplers that interpolate between the key-frames.
        std::vector<AnimSampler> Samplers;         //!< The parameterized inputs representing the key-frame data.

        Animation() {}
        void Read(Value& obj, Asset& r);
    };


    //! Base class for LazyDict that acts as an interface
    class LazyDictBase
    {
    public:
        virtual ~LazyDictBase() {}

        virtual void AttachToDocument(Document& doc) = 0;
        virtual void DetachFromDocument() = 0;

        virtual void WriteObjects(AssetWriter& writer) = 0;
    };


    template<class T>
    class LazyDict;

    //! (Implemented in glTFAssetWriter.h)
    template<class T>
    void WriteLazyDict(LazyDict<T>& d, AssetWriter& w);


    //! Manages lazy loading of the glTF top-level objects, and keeps a reference to them by ID
    //! It is the owner the loaded objects, so when it is destroyed it also deletes them
    template<class T>
    class LazyDict : public LazyDictBase
    {
        friend class Asset;
        friend class AssetWriter;

        typedef typename std::gltf_unordered_map< std::string, unsigned int > Dict;

        std::vector<T*>  mObjs;      //! The read objects
        Dict             mObjsById;  //! The read objects accessible by id
        const char*      mDictId;    //! ID of the dictionary object
        const char*      mExtId;     //! ID of the extension defining the dictionary
        Value*           mDict;      //! JSON dictionary object
        Asset&           mAsset;     //! The asset instance

        void AttachToDocument(Document& doc);
        void DetachFromDocument();

        void WriteObjects(AssetWriter& writer)
            { WriteLazyDict<T>(*this, writer); }

        Ref<T> Add(T* obj);

    public:
        LazyDict(Asset& asset, const char* dictId, const char* extId = 0);
        ~LazyDict();

        Ref<T> Get(const char* id);
        Ref<T> Get(unsigned int i);
        Ref<T> Get(const std::string& pID) { return Get(pID.c_str()); }

        Ref<T> Create(const char* id);
        Ref<T> Create(const std::string& id)
            { return Create(id.c_str()); }

        inline unsigned int Size() const
            { return unsigned(mObjs.size()); }

        inline T& operator[](size_t i)
            { return *mObjs[i]; }

    };


    struct AssetMetadata
    {
        std::string copyright; //!< A copyright message suitable for display to credit the content creator.
        std::string generator; //!< Tool that generated this glTF model.Useful for debugging.
        bool premultipliedAlpha; //!< Specifies if the shaders were generated with premultiplied alpha. (default: false)

        struct {
            std::string api;     //!< Specifies the target rendering API (default: "WebGL")
            std::string version; //!< Specifies the target rendering API (default: "1.0.3")
        } profile; //!< Specifies the target rendering API and version, e.g., WebGL 1.0.3. (default: {})

        std::string version; //!< The glTF format version (should be 1.0)

        void Read(Document& doc);

        AssetMetadata()
            : premultipliedAlpha(false)
            , version("")
        {
        }
    };

    //
    // glTF Asset class
    //

    //! Root object for a glTF asset
    class Asset
    {
        typedef std::gltf_unordered_map<std::string, int> IdMap;

        template<class T>
        friend class LazyDict;

        friend struct Buffer; // To access OpenFile

        friend class AssetWriter;

    private:
        IOSystem* mIOSystem;

        std::string mCurrentAssetDir;

        size_t mSceneLength;
        size_t mBodyOffset, mBodyLength;

        std::vector<LazyDictBase*> mDicts;

        IdMap mUsedIds;

        Ref<Buffer> mBodyBuffer;

        Asset(Asset&);
        Asset& operator=(const Asset&);

    public:

        //! Keeps info about the enabled extensions
        struct Extensions
        {
            bool KHR_binary_glTF;
            bool KHR_materials_common;

        } extensionsUsed;

        AssetMetadata asset;


        // Dictionaries for each type of object

        LazyDict<Accessor>    accessors;
        LazyDict<Animation>   animations;
        LazyDict<Buffer>      buffers;
        LazyDict<BufferView>  bufferViews;
        LazyDict<Camera>      cameras;
        LazyDict<Image>       images;
        LazyDict<Material>    materials;
        LazyDict<Mesh>        meshes;
        LazyDict<Node>        nodes;
        //LazyDict<Program>   programs;
        LazyDict<Sampler>     samplers;
        LazyDict<Scene>       scenes;
        //LazyDict<Shader>    shaders;
        LazyDict<Skin>      skins;
        //LazyDict<Technique> techniques;
        LazyDict<Texture>     textures;

        LazyDict<Light>       lights; // KHR_materials_common ext

        Ref<Scene> scene;

    public:
        Asset(IOSystem* io = 0)
            : mIOSystem(io)
            , asset()
            , accessors     (*this, "accessors")
            , animations    (*this, "animations")
            , buffers       (*this, "buffers")
            , bufferViews   (*this, "bufferViews")
            , cameras       (*this, "cameras")
            , images        (*this, "images")
            , materials     (*this, "materials")
            , meshes        (*this, "meshes")
            , nodes         (*this, "nodes")
            //, programs    (*this, "programs")
            , samplers      (*this, "samplers")
            , scenes        (*this, "scenes")
            //, shaders     (*this, "shaders")
            , skins       (*this, "skins")
            //, techniques  (*this, "techniques")
            , textures      (*this, "textures")
            , lights        (*this, "lights", "KHR_materials_common")
        {
            memset(&extensionsUsed, 0, sizeof(extensionsUsed));
        }

        //! Main function
        void Load(const std::string& file, bool isBinary = false);

        //! Enables the "KHR_binary_glTF" extension on the asset
        void SetAsBinary();

        //! Search for an available name, starting from the given strings
        std::string FindUniqueID(const std::string& str, const char* suffix);

        Ref<Buffer> GetBodyBuffer()
            { return mBodyBuffer; }

    private:
        void ReadBinaryHeader(IOStream& stream);

        void ReadExtensionsUsed(Document& doc);


        IOStream* OpenFile(std::string path, const char* mode, bool absolute = false);
    };

}

// Include the implementation of the methods
#include "glTFAsset.inl"

#endif // ASSIMP_BUILD_NO_GLTF_IMPORTER

#endif // GLTFASSET_H_INC
