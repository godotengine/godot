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

#include <assimp/StringUtils.h>
#include <iomanip>

// Header files, Assimp
#include <assimp/DefaultLogger.hpp>

#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
	// Header files, Open3DGC.
#	include <Open3DGC/o3dgcSC3DMCDecoder.h>
#endif

using namespace Assimp;
using namespace glTFCommon;

namespace glTF {

namespace {

    //
    // JSON Value reading helpers
    //

    template<class T>
    struct ReadHelper { static bool Read(Value& val, T& out) {
        return val.IsInt() ? out = static_cast<T>(val.GetInt()), true : false;
    }};

    template<> struct ReadHelper<bool> { static bool Read(Value& val, bool& out) {
        return val.IsBool() ? out = val.GetBool(), true : false;
    }};

    template<> struct ReadHelper<float> { static bool Read(Value& val, float& out) {
        return val.IsNumber() ? out = static_cast<float>(val.GetDouble()), true : false;
    }};

    template<unsigned int N> struct ReadHelper<float[N]> { static bool Read(Value& val, float (&out)[N]) {
        if (!val.IsArray() || val.Size() != N) return false;
        for (unsigned int i = 0; i < N; ++i) {
            if (val[i].IsNumber())
                out[i] = static_cast<float>(val[i].GetDouble());
        }
        return true;
    }};

    template<> struct ReadHelper<const char*> { static bool Read(Value& val, const char*& out) {
        return val.IsString() ? (out = val.GetString(), true) : false;
    }};

    template<> struct ReadHelper<std::string> { static bool Read(Value& val, std::string& out) {
        return val.IsString() ? (out = std::string(val.GetString(), val.GetStringLength()), true) : false;
    }};

    template<class T> struct ReadHelper< Nullable<T> > { static bool Read(Value& val, Nullable<T>& out) {
        return out.isPresent = ReadHelper<T>::Read(val, out.value);
    }};

    template<> struct ReadHelper<uint64_t> { static bool Read(Value& val, uint64_t& out) {
        return val.IsUint64() ? out = val.GetUint64(), true : false;
    }};

    template<> struct ReadHelper<int64_t> { static bool Read(Value& val, int64_t& out) {
        return val.IsInt64() ? out = val.GetInt64(), true : false;
    }};

    template<class T>
    inline static bool ReadValue(Value& val, T& out)
    {
        return ReadHelper<T>::Read(val, out);
    }

    template<class T>
    inline static bool ReadMember(Value& obj, const char* id, T& out)
    {
        Value::MemberIterator it = obj.FindMember(id);
        if (it != obj.MemberEnd()) {
            return ReadHelper<T>::Read(it->value, out);
        }
        return false;
    }

    template<class T>
    inline static T MemberOrDefault(Value& obj, const char* id, T defaultValue)
    {
        T out;
        return ReadMember(obj, id, out) ? out : defaultValue;
    }

    inline Value* FindMember(Value& val, const char* id)
    {
        Value::MemberIterator it = val.FindMember(id);
        return (it != val.MemberEnd()) ? &it->value : 0;
    }

    inline Value* FindString(Value& val, const char* id)
    {
        Value::MemberIterator it = val.FindMember(id);
        return (it != val.MemberEnd() && it->value.IsString()) ? &it->value : 0;
    }

    inline Value* FindNumber(Value& val, const char* id)
    {
        Value::MemberIterator it = val.FindMember(id);
        return (it != val.MemberEnd() && it->value.IsNumber()) ? &it->value : 0;
    }

    inline Value* FindArray(Value& val, const char* id)
    {
        Value::MemberIterator it = val.FindMember(id);
        return (it != val.MemberEnd() && it->value.IsArray()) ? &it->value : 0;
    }

    inline Value* FindObject(Value& val, const char* id)
    {
        Value::MemberIterator it = val.FindMember(id);
        return (it != val.MemberEnd() && it->value.IsObject()) ? &it->value : 0;
    }
}

//
// LazyDict methods
//

template<class T>
inline LazyDict<T>::LazyDict(Asset& asset, const char* dictId, const char* extId)
    : mDictId(dictId), mExtId(extId), mDict(0), mAsset(asset)
{
    asset.mDicts.push_back(this); // register to the list of dictionaries
}

template<class T>
inline LazyDict<T>::~LazyDict()
{
    for (size_t i = 0; i < mObjs.size(); ++i) {
        delete mObjs[i];
    }
}


template<class T>
inline void LazyDict<T>::AttachToDocument(Document& doc)
{
    Value* container = 0;

    if (mExtId) {
        if (Value* exts = FindObject(doc, "extensions")) {
            container = FindObject(*exts, mExtId);
        }
    }
    else {
        container = &doc;
    }

    if (container) {
        mDict = FindObject(*container, mDictId);
    }
}

template<class T>
inline void LazyDict<T>::DetachFromDocument()
{
    mDict = 0;
}

template<class T>
Ref<T> LazyDict<T>::Get(unsigned int i)
{
    return Ref<T>(mObjs, i);
}

template<class T>
Ref<T> LazyDict<T>::Get(const char* id)
{
    id = T::TranslateId(mAsset, id);

    typename Dict::iterator it = mObjsById.find(id);
    if (it != mObjsById.end()) { // already created?
        return Ref<T>(mObjs, it->second);
    }

    // read it from the JSON object
    if (!mDict) {
        throw DeadlyImportError("GLTF: Missing section \"" + std::string(mDictId) + "\"");
    }

    Value::MemberIterator obj = mDict->FindMember(id);
    if (obj == mDict->MemberEnd()) {
        throw DeadlyImportError("GLTF: Missing object with id \"" + std::string(id) + "\" in \"" + mDictId + "\"");
    }
    if (!obj->value.IsObject()) {
        throw DeadlyImportError("GLTF: Object with id \"" + std::string(id) + "\" is not a JSON object");
    }

    // create an instance of the given type
    T* inst = new T();
    inst->id = id;
    ReadMember(obj->value, "name", inst->name);
    inst->Read(obj->value, mAsset);
    return Add(inst);
}

template<class T>
Ref<T> LazyDict<T>::Add(T* obj)
{
    unsigned int idx = unsigned(mObjs.size());
    mObjs.push_back(obj);
    mObjsById[obj->id] = idx;
    mAsset.mUsedIds[obj->id] = true;
    return Ref<T>(mObjs, idx);
}

template<class T>
Ref<T> LazyDict<T>::Create(const char* id)
{
    Asset::IdMap::iterator it = mAsset.mUsedIds.find(id);
    if (it != mAsset.mUsedIds.end()) {
        throw DeadlyImportError("GLTF: two objects with the same ID exist");
    }
    T* inst = new T();
    inst->id = id;
    return Add(inst);
}


//
// glTF dictionary objects methods
//


inline Buffer::Buffer()
	: byteLength(0), type(Type_arraybuffer), EncodedRegion_Current(nullptr), mIsSpecial(false)
{ }

inline Buffer::~Buffer()
{
	for(SEncodedRegion* reg : EncodedRegion_List) delete reg;
}

inline const char* Buffer::TranslateId(Asset& r, const char* id)
{
    // Compatibility with old spec
    if (r.extensionsUsed.KHR_binary_glTF && strcmp(id, "KHR_binary_glTF") == 0) {
        return "binary_glTF";
    }

    return id;
}

inline void Buffer::Read(Value& obj, Asset& r)
{
    size_t statedLength = MemberOrDefault<size_t>(obj, "byteLength", 0);
    byteLength = statedLength;

    Value* it = FindString(obj, "uri");
    if (!it) {
        if (statedLength > 0) {
            throw DeadlyImportError("GLTF: buffer with non-zero length missing the \"uri\" attribute");
        }
        return;
    }

    const char* uri = it->GetString();

    glTFCommon::Util::DataURI dataURI;
    if (ParseDataURI(uri, it->GetStringLength(), dataURI)) {
        if (dataURI.base64) {
            uint8_t* data = 0;
            this->byteLength = Util::DecodeBase64(dataURI.data, dataURI.dataLength, data);
            this->mData.reset(data, std::default_delete<uint8_t[]>());

            if (statedLength > 0 && this->byteLength != statedLength) {
                throw DeadlyImportError("GLTF: buffer \"" + id + "\", expected " + to_string(statedLength) +
                    " bytes, but found " + to_string(dataURI.dataLength));
            }
        }
        else { // assume raw data
            if (statedLength != dataURI.dataLength) {
                throw DeadlyImportError("GLTF: buffer \"" + id + "\", expected " + to_string(statedLength) +
                                        " bytes, but found " + to_string(dataURI.dataLength));
            }

            this->mData.reset(new uint8_t[dataURI.dataLength], std::default_delete<uint8_t[]>());
            memcpy( this->mData.get(), dataURI.data, dataURI.dataLength );
        }
    }
    else { // Local file
        if (byteLength > 0) {
            std::string dir = !r.mCurrentAssetDir.empty() ? (r.mCurrentAssetDir + "/") : "";

            IOStream* file = r.OpenFile(dir + uri, "rb");
            if (file) {
                bool ok = LoadFromStream(*file, byteLength);
                delete file;

                if (!ok)
                    throw DeadlyImportError("GLTF: error while reading referenced file \"" + std::string(uri) + "\"" );
            }
            else {
                throw DeadlyImportError("GLTF: could not open referenced file \"" + std::string(uri) + "\"");
            }
        }
    }
}

inline bool Buffer::LoadFromStream(IOStream& stream, size_t length, size_t baseOffset)
{
    byteLength = length ? length : stream.FileSize();

    if (baseOffset) {
        stream.Seek(baseOffset, aiOrigin_SET);
    }

    mData.reset(new uint8_t[byteLength], std::default_delete<uint8_t[]>());

    if (stream.Read(mData.get(), byteLength, 1) != 1) {
        return false;
    }
    return true;
}

inline void Buffer::EncodedRegion_Mark(const size_t pOffset, const size_t pEncodedData_Length, uint8_t* pDecodedData, const size_t pDecodedData_Length, const std::string& pID)
{
	// Check pointer to data
	if(pDecodedData == nullptr) throw DeadlyImportError("GLTF: for marking encoded region pointer to decoded data must be provided.");

	// Check offset
	if(pOffset > byteLength)
	{
		const uint8_t val_size = 32;

		char val[val_size];

		ai_snprintf(val, val_size, "%llu", (long long)pOffset);
		throw DeadlyImportError(std::string("GLTF: incorrect offset value (") + val + ") for marking encoded region.");
	}

	// Check length
	if((pOffset + pEncodedData_Length) > byteLength)
	{
		const uint8_t val_size = 64;

		char val[val_size];

		ai_snprintf(val, val_size, "%llu, %llu", (long long)pOffset, (long long)pEncodedData_Length);
		throw DeadlyImportError(std::string("GLTF: encoded region with offset/length (") + val + ") is out of range.");
	}

	// Add new region
	EncodedRegion_List.push_back(new SEncodedRegion(pOffset, pEncodedData_Length, pDecodedData, pDecodedData_Length, pID));
	// And set new value for "byteLength"
	byteLength += (pDecodedData_Length - pEncodedData_Length);
}

inline void Buffer::EncodedRegion_SetCurrent(const std::string& pID)
{
	if((EncodedRegion_Current != nullptr) && (EncodedRegion_Current->ID == pID)) return;

	for(SEncodedRegion* reg : EncodedRegion_List)
	{
		if(reg->ID == pID)
		{
			EncodedRegion_Current = reg;

			return;
		}

	}

	throw DeadlyImportError("GLTF: EncodedRegion with ID: \"" + pID + "\" not found.");
}

inline bool Buffer::ReplaceData(const size_t pBufferData_Offset, const size_t pBufferData_Count, const uint8_t* pReplace_Data, const size_t pReplace_Count)
{
const size_t new_data_size = byteLength + pReplace_Count - pBufferData_Count;

uint8_t* new_data;

	if((pBufferData_Count == 0) || (pReplace_Count == 0) || (pReplace_Data == nullptr)) return false;

	new_data = new uint8_t[new_data_size];
	// Copy data which place before replacing part.
	memcpy(new_data, mData.get(), pBufferData_Offset);
	// Copy new data.
	memcpy(&new_data[pBufferData_Offset], pReplace_Data, pReplace_Count);
	// Copy data which place after replacing part.
	memcpy(&new_data[pBufferData_Offset + pReplace_Count], &mData.get()[pBufferData_Offset + pBufferData_Count], pBufferData_Offset);
	// Apply new data
	mData.reset(new_data, std::default_delete<uint8_t[]>());
	byteLength = new_data_size;

	return true;
}

inline size_t Buffer::AppendData(uint8_t* data, size_t length)
{
    size_t offset = this->byteLength;
    Grow(length);
    memcpy(mData.get() + offset, data, length);
    return offset;
}

inline void Buffer::Grow(size_t amount)
{
    if (amount <= 0) return;
    if (capacity >= byteLength + amount)
    {
        byteLength += amount;
        return;
    }

    // Shift operation is standard way to divide integer by 2, it doesn't cast it to float back and forth, also works for odd numbers,
    // originally it would look like: static_cast<size_t>(capacity * 1.5f)
    capacity = std::max(capacity + (capacity >> 1), byteLength + amount);

    uint8_t* b = new uint8_t[capacity];
    if (mData) memcpy(b, mData.get(), byteLength);
    mData.reset(b, std::default_delete<uint8_t[]>());
    byteLength += amount;
}

//
// struct BufferView
//

inline void BufferView::Read(Value& obj, Asset& r)
{
    const char* bufferId = MemberOrDefault<const char*>(obj, "buffer", 0);
    if (bufferId) {
        buffer = r.buffers.Get(bufferId);
    }

    byteOffset = MemberOrDefault(obj, "byteOffset", 0u);
    byteLength = MemberOrDefault(obj, "byteLength", 0u);
}

//
// struct Accessor
//

inline void Accessor::Read(Value& obj, Asset& r)
{
    const char* bufferViewId = MemberOrDefault<const char*>(obj, "bufferView", 0);
    if (bufferViewId) {
        bufferView = r.bufferViews.Get(bufferViewId);
    }

    byteOffset = MemberOrDefault(obj, "byteOffset", 0u);
    byteStride = MemberOrDefault(obj, "byteStride", 0u);
    componentType = MemberOrDefault(obj, "componentType", ComponentType_BYTE);
    count = MemberOrDefault(obj, "count", 0u);

    const char* typestr;
    type = ReadMember(obj, "type", typestr) ? AttribType::FromString(typestr) : AttribType::SCALAR;
}

inline unsigned int Accessor::GetNumComponents()
{
    return AttribType::GetNumComponents(type);
}

inline unsigned int Accessor::GetBytesPerComponent()
{
    return int(ComponentTypeSize(componentType));
}

inline unsigned int Accessor::GetElementSize()
{
    return GetNumComponents() * GetBytesPerComponent();
}

inline uint8_t* Accessor::GetPointer()
{
    if (!bufferView || !bufferView->buffer) return 0;
    uint8_t* basePtr = bufferView->buffer->GetPointer();
    if (!basePtr) return 0;

    size_t offset = byteOffset + bufferView->byteOffset;

	// Check if region is encoded.
	if(bufferView->buffer->EncodedRegion_Current != nullptr)
	{
		const size_t begin = bufferView->buffer->EncodedRegion_Current->Offset;
		const size_t end = begin + bufferView->buffer->EncodedRegion_Current->DecodedData_Length;

		if((offset >= begin) && (offset < end))
			return &bufferView->buffer->EncodedRegion_Current->DecodedData[offset - begin];
	}

	return basePtr + offset;
}

namespace {
    inline void CopyData(size_t count,
            const uint8_t* src, size_t src_stride,
                  uint8_t* dst, size_t dst_stride)
    {
        if (src_stride == dst_stride) {
            memcpy(dst, src, count * src_stride);
        }
        else {
            size_t sz = std::min(src_stride, dst_stride);
            for (size_t i = 0; i < count; ++i) {
                memcpy(dst, src, sz);
                if (sz < dst_stride) {
                    memset(dst + sz, 0, dst_stride - sz);
                }
                src += src_stride;
                dst += dst_stride;
            }
        }
    }
}

template<class T>
bool Accessor::ExtractData(T*& outData)
{
    uint8_t* data = GetPointer();
    if (!data) return false;

    const size_t elemSize = GetElementSize();
    const size_t totalSize = elemSize * count;

    const size_t stride = byteStride ? byteStride : elemSize;

    const size_t targetElemSize = sizeof(T);
    ai_assert(elemSize <= targetElemSize);

    ai_assert(count*stride <= bufferView->byteLength);

    outData = new T[count];
    if (stride == elemSize && targetElemSize == elemSize) {
        memcpy(outData, data, totalSize);
    }
    else {
        for (size_t i = 0; i < count; ++i) {
            memcpy(outData + i, data + i*stride, elemSize);
        }
    }

    return true;
}

inline void Accessor::WriteData(size_t count, const void* src_buffer, size_t src_stride)
{
    uint8_t* buffer_ptr = bufferView->buffer->GetPointer();
    size_t offset = byteOffset + bufferView->byteOffset;

    size_t dst_stride = GetNumComponents() * GetBytesPerComponent();

    const uint8_t* src = reinterpret_cast<const uint8_t*>(src_buffer);
    uint8_t*       dst = reinterpret_cast<      uint8_t*>(buffer_ptr + offset);

    ai_assert(dst + count*dst_stride <= buffer_ptr + bufferView->buffer->byteLength);
    CopyData(count, src, src_stride, dst, dst_stride);
}



inline Accessor::Indexer::Indexer(Accessor& acc)
    : accessor(acc)
    , data(acc.GetPointer())
    , elemSize(acc.GetElementSize())
    , stride(acc.byteStride ? acc.byteStride : elemSize)
{

}

//! Accesses the i-th value as defined by the accessor
template<class T>
T Accessor::Indexer::GetValue(int i)
{
    ai_assert(data);
    ai_assert(i*stride < accessor.bufferView->byteLength);
    T value = T();
    memcpy(&value, data + i*stride, elemSize);
    //value >>= 8 * (sizeof(T) - elemSize);
    return value;
}

inline Image::Image()
    : width(0)
    , height(0)
    , mDataLength(0)
{

}

inline void Image::Read(Value& obj, Asset& r)
{
    // Check for extensions first (to detect binary embedded data)
    if (Value* extensions = FindObject(obj, "extensions")) {
        if (r.extensionsUsed.KHR_binary_glTF) {
            if (Value* ext = FindObject(*extensions, "KHR_binary_glTF")) {

                width  = MemberOrDefault(*ext, "width", 0);
                height = MemberOrDefault(*ext, "height", 0);

                ReadMember(*ext, "mimeType", mimeType);

                const char* bufferViewId;
                if (ReadMember(*ext, "bufferView", bufferViewId)) {
                    Ref<BufferView> bv = r.bufferViews.Get(bufferViewId);
                    if (bv) {
                        mDataLength = bv->byteLength;
                        mData.reset(new uint8_t[mDataLength]);
                        memcpy(mData.get(), bv->buffer->GetPointer() + bv->byteOffset, mDataLength);
                    }
                }
            }
        }
    }

    if (!mDataLength) {
        if (Value* uri = FindString(obj, "uri")) {
            const char* uristr = uri->GetString();

            glTFCommon::Util::DataURI dataURI;
            if (ParseDataURI(uristr, uri->GetStringLength(), dataURI)) {
                mimeType = dataURI.mediaType;
                if (dataURI.base64) {
                    uint8_t *ptr = nullptr;
                    mDataLength = glTFCommon::Util::DecodeBase64(dataURI.data, dataURI.dataLength, ptr);
                    mData.reset(ptr);
                }
            }
            else {
                this->uri = uristr;
            }
        }
    }
}

inline uint8_t* Image::StealData()
{
    mDataLength = 0;
    return mData.release();
}

inline void Image::SetData(uint8_t* data, size_t length, Asset& r)
{
    Ref<Buffer> b = r.GetBodyBuffer();
    if (b) { // binary file: append to body
        std::string bvId = r.FindUniqueID(this->id, "imgdata");
        bufferView = r.bufferViews.Create(bvId);

        bufferView->buffer = b;
        bufferView->byteLength = length;
        bufferView->byteOffset = b->AppendData(data, length);
    }
    else { // text file: will be stored as a data uri
        this->mData.reset(data);
        this->mDataLength = length;
    }
}

inline void Sampler::Read(Value& obj, Asset& /*r*/)
{
    SetDefaults();

    ReadMember(obj, "magFilter", magFilter);
    ReadMember(obj, "minFilter", minFilter);
    ReadMember(obj, "wrapS", wrapS);
    ReadMember(obj, "wrapT", wrapT);
}

inline void Sampler::SetDefaults()
{
    magFilter = SamplerMagFilter_Linear;
    minFilter = SamplerMinFilter_Linear;
    wrapS = SamplerWrap_Repeat;
    wrapT = SamplerWrap_Repeat;
}

inline void Texture::Read(Value& obj, Asset& r)
{
    const char* sourcestr;
    if (ReadMember(obj, "source", sourcestr)) {
        source = r.images.Get(sourcestr);
    }

    const char* samplerstr;
    if (ReadMember(obj, "sampler", samplerstr)) {
        sampler = r.samplers.Get(samplerstr);
    }
}

namespace {
    inline void ReadMaterialProperty(Asset& r, Value& vals, const char* propName, TexProperty& out)
    {
        if (Value* prop = FindMember(vals, propName)) {
            if (prop->IsString()) {
                out.texture = r.textures.Get(prop->GetString());
            }
            else {
                ReadValue(*prop, out.color);
            }
        }
    }
}

inline void Material::Read(Value& material, Asset& r)
{
    SetDefaults();

    if (Value* values = FindObject(material, "values")) {
        ReadMaterialProperty(r, *values, "ambient", this->ambient);
        ReadMaterialProperty(r, *values, "diffuse", this->diffuse);
        ReadMaterialProperty(r, *values, "specular", this->specular);

        ReadMember(*values, "transparency", transparency);
        ReadMember(*values, "shininess", shininess);
    }

    if (Value* extensions = FindObject(material, "extensions")) {
        if (r.extensionsUsed.KHR_materials_common) {
            if (Value* ext = FindObject(*extensions, "KHR_materials_common")) {
                if (Value* tnq = FindString(*ext, "technique")) {
                    const char* t = tnq->GetString();
                    if      (strcmp(t, "BLINN") == 0)    technique = Technique_BLINN;
                    else if (strcmp(t, "PHONG") == 0)    technique = Technique_PHONG;
                    else if (strcmp(t, "LAMBERT") == 0)  technique = Technique_LAMBERT;
                    else if (strcmp(t, "CONSTANT") == 0) technique = Technique_CONSTANT;
                }

                if (Value* values = FindObject(*ext, "values")) {
                    ReadMaterialProperty(r, *values, "ambient", this->ambient);
                    ReadMaterialProperty(r, *values, "diffuse", this->diffuse);
                    ReadMaterialProperty(r, *values, "specular", this->specular);

                    ReadMember(*values, "doubleSided", doubleSided);
                    ReadMember(*values, "transparent", transparent);
                    ReadMember(*values, "transparency", transparency);
                    ReadMember(*values, "shininess", shininess);
                }
            }
        }
    }
}

namespace {
    void SetVector(vec4& v, float x, float y, float z, float w)
        { v[0] = x; v[1] = y; v[2] = z; v[3] = w; }
}

inline void Material::SetDefaults()
{
    SetVector(ambient.color, 0, 0, 0, 1);
    SetVector(diffuse.color, 0, 0, 0, 1);
    SetVector(specular.color, 0, 0, 0, 1);
    SetVector(emission.color, 0, 0, 0, 1);

    doubleSided = false;
    transparent = false;
    transparency = 1.0;
    shininess = 0.0;

    technique = Technique_undefined;
}

namespace {

    template<int N>
    inline int Compare(const char* attr, const char (&str)[N]) {
        return (strncmp(attr, str, N - 1) == 0) ? N - 1 : 0;
    }

    inline bool GetAttribVector(Mesh::Primitive& p, const char* attr, Mesh::AccessorList*& v, int& pos)
    {
        if ((pos = Compare(attr, "POSITION"))) {
            v = &(p.attributes.position);
        }
        else if ((pos = Compare(attr, "NORMAL"))) {
            v = &(p.attributes.normal);
        }
        else if ((pos = Compare(attr, "TEXCOORD"))) {
            v = &(p.attributes.texcoord);
        }
        else if ((pos = Compare(attr, "COLOR"))) {
            v = &(p.attributes.color);
        }
        else if ((pos = Compare(attr, "JOINT"))) {
            v = &(p.attributes.joint);
        }
        else if ((pos = Compare(attr, "JOINTMATRIX"))) {
            v = &(p.attributes.jointmatrix);
        }
        else if ((pos = Compare(attr, "WEIGHT"))) {
            v = &(p.attributes.weight);
        }
        else return false;
        return true;
    }
}

inline void Mesh::Read(Value& pJSON_Object, Asset& pAsset_Root)
{
	/****************** Mesh primitives ******************/
	if (Value* primitives = FindArray(pJSON_Object, "primitives")) {
        this->primitives.resize(primitives->Size());
        for (unsigned int i = 0; i < primitives->Size(); ++i) {
            Value& primitive = (*primitives)[i];

            Primitive& prim = this->primitives[i];
            prim.mode = MemberOrDefault(primitive, "mode", PrimitiveMode_TRIANGLES);

            if (Value* attrs = FindObject(primitive, "attributes")) {
                for (Value::MemberIterator it = attrs->MemberBegin(); it != attrs->MemberEnd(); ++it) {
                    if (!it->value.IsString()) continue;
                    const char* attr = it->name.GetString();
                    // Valid attribute semantics include POSITION, NORMAL, TEXCOORD, COLOR, JOINT, JOINTMATRIX,
                    // and WEIGHT.Attribute semantics can be of the form[semantic]_[set_index], e.g., TEXCOORD_0, TEXCOORD_1, etc.

                    int undPos = 0;
                    Mesh::AccessorList* vec = 0;
                    if (GetAttribVector(prim, attr, vec, undPos)) {
                        size_t idx = (attr[undPos] == '_') ? atoi(attr + undPos + 1) : 0;
                        if ((*vec).size() <= idx) (*vec).resize(idx + 1);
						(*vec)[idx] = pAsset_Root.accessors.Get(it->value.GetString());
                    }
                }
            }

            if (Value* indices = FindString(primitive, "indices")) {
				prim.indices = pAsset_Root.accessors.Get(indices->GetString());
            }

            if (Value* material = FindString(primitive, "material")) {
				prim.material = pAsset_Root.materials.Get(material->GetString());
            }
        }
    }

	/****************** Mesh extensions ******************/
	Value* json_extensions = FindObject(pJSON_Object, "extensions");

	if(json_extensions == nullptr) goto mr_skip_extensions;

	for(Value::MemberIterator it_memb = json_extensions->MemberBegin(); it_memb != json_extensions->MemberEnd(); it_memb++)
	{
#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
        if(it_memb->name.GetString() == std::string("Open3DGC-compression"))
		{
			// Search for compressed data.
			// Compressed data contain description of part of "buffer" which is encoded. This part must be decoded and
			// new data will replace old encoded part by request. In fact \"compressedData\" is kind of "accessor" structure.
			Value* comp_data = FindObject(it_memb->value, "compressedData");

			if(comp_data == nullptr) throw DeadlyImportError("GLTF: \"Open3DGC-compression\" must has \"compressedData\".");

            ASSIMP_LOG_INFO("GLTF: Decompressing Open3DGC data.");

			/************** Read data from JSON-document **************/
			#define MESH_READ_COMPRESSEDDATA_MEMBER(pFieldName, pOut) \
				if(!ReadMember(*comp_data, pFieldName, pOut)) \
				{ \
					throw DeadlyImportError(std::string("GLTF: \"compressedData\" must has \"") + pFieldName + "\"."); \
				}

			const char* mode_str;
			const char* type_str;
			ComponentType component_type;
			SCompression_Open3DGC* ext_o3dgc = new SCompression_Open3DGC;

			MESH_READ_COMPRESSEDDATA_MEMBER("buffer", ext_o3dgc->Buffer);
			MESH_READ_COMPRESSEDDATA_MEMBER("byteOffset", ext_o3dgc->Offset);
			MESH_READ_COMPRESSEDDATA_MEMBER("componentType", component_type);
			MESH_READ_COMPRESSEDDATA_MEMBER("type", type_str);
			MESH_READ_COMPRESSEDDATA_MEMBER("count", ext_o3dgc->Count);
			MESH_READ_COMPRESSEDDATA_MEMBER("mode", mode_str);
			MESH_READ_COMPRESSEDDATA_MEMBER("indicesCount", ext_o3dgc->IndicesCount);
			MESH_READ_COMPRESSEDDATA_MEMBER("verticesCount", ext_o3dgc->VerticesCount);

			#undef MESH_READ_COMPRESSEDDATA_MEMBER

			// Check some values
			if(strcmp(type_str, "SCALAR")) throw DeadlyImportError("GLTF: only \"SCALAR\" type is supported for compressed data.");
			if(component_type != ComponentType_UNSIGNED_BYTE) throw DeadlyImportError("GLTF: only \"UNSIGNED_BYTE\" component type is supported for compressed data.");

			// Set read/write data mode.
			if(strcmp(mode_str, "binary") == 0)
				ext_o3dgc->Binary = true;
			else if(strcmp(mode_str, "ascii") == 0)
				ext_o3dgc->Binary = false;
			else
				throw DeadlyImportError(std::string("GLTF: for compressed data supported modes is: \"ascii\", \"binary\". Not the: \"") + mode_str + "\".");

			/************************ Decoding ************************/
			Decode_O3DGC(*ext_o3dgc, pAsset_Root);
			Extension.push_back(ext_o3dgc);// store info in mesh extensions list.
		}// if(it_memb->name.GetString() == "Open3DGC-compression")
		else
#endif
		{
			throw DeadlyImportError(std::string("GLTF: Unknown mesh extension: \"") + it_memb->name.GetString() + "\".");
		}
	}// for(Value::MemberIterator it_memb = json_extensions->MemberBegin(); it_memb != json_extensions->MemberEnd(); json_extensions++)

mr_skip_extensions:

	return;// After label some operators must be present.
}

#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
inline void Mesh::Decode_O3DGC(const SCompression_Open3DGC& pCompression_Open3DGC, Asset& pAsset_Root)
{
typedef unsigned short IndicesType;///< \sa glTFExporter::ExportMeshes.

o3dgc::SC3DMCDecoder<IndicesType> decoder;
o3dgc::IndexedFaceSet<IndicesType> ifs;
o3dgc::BinaryStream bstream;
uint8_t* decoded_data;
size_t decoded_data_size = 0;
Ref<Buffer> buf = pAsset_Root.buffers.Get(pCompression_Open3DGC.Buffer);

	// Read data from buffer and place it in BinaryStream for decoder.
	// Just "Count" because always is used type equivalent to uint8_t.
	bstream.LoadFromBuffer(&buf->GetPointer()[pCompression_Open3DGC.Offset], static_cast<unsigned long>(pCompression_Open3DGC.Count));

	// After decoding header we can get size of primitives.
	if(decoder.DecodeHeader(ifs, bstream) != o3dgc::O3DGC_OK) throw DeadlyImportError("GLTF: can not decode Open3DGC header.");

	/****************** Get sizes of arrays and check sizes ******************/
	// Note. See "Limitations for meshes when using Open3DGC-compression".

	// Indices
	size_t size_coordindex = ifs.GetNCoordIndex() * 3;// See float attributes note.

	if(primitives[0].indices->count != size_coordindex)
		throw DeadlyImportError("GLTF: Open3DGC. Compressed indices count (" + to_string(size_coordindex) +
								") not equal to uncompressed (" + to_string(primitives[0].indices->count) + ").");

	size_coordindex *= sizeof(IndicesType);
	// Coordinates
	size_t size_coord = ifs.GetNCoord();// See float attributes note.

	if(primitives[0].attributes.position[0]->count != size_coord)
		throw DeadlyImportError("GLTF: Open3DGC. Compressed positions count (" + to_string(size_coord) +
								") not equal to uncompressed (" + to_string(primitives[0].attributes.position[0]->count) + ").");

	size_coord *= 3 * sizeof(float);
	// Normals
	size_t size_normal = ifs.GetNNormal();// See float attributes note.

	if(primitives[0].attributes.normal[0]->count != size_normal)
		throw DeadlyImportError("GLTF: Open3DGC. Compressed normals count (" + to_string(size_normal) +
								") not equal to uncompressed (" + to_string(primitives[0].attributes.normal[0]->count) + ").");

	size_normal *= 3 * sizeof(float);
	// Additional attributes.
	std::vector<size_t> size_floatattr;
	std::vector<size_t> size_intattr;

	size_floatattr.resize(ifs.GetNumFloatAttributes());
	size_intattr.resize(ifs.GetNumIntAttributes());

	decoded_data_size = size_coordindex + size_coord + size_normal;
	for(size_t idx = 0, idx_end = size_floatattr.size(), idx_texcoord = 0; idx < idx_end; idx++)
	{
		// size = number_of_elements * components_per_element * size_of_component.
		// Note. But as you can see above, at first we are use this variable in meaning "count". After checking count of objects...
		size_t tval = ifs.GetNFloatAttribute(static_cast<unsigned long>(idx));

		switch(ifs.GetFloatAttributeType(static_cast<unsigned long>(idx)))
		{
			case o3dgc::O3DGC_IFS_FLOAT_ATTRIBUTE_TYPE_TEXCOORD:
				// Check situation when encoded data contain texture coordinates but primitive not.
				if(idx_texcoord < primitives[0].attributes.texcoord.size())
				{
					if(primitives[0].attributes.texcoord[idx]->count != tval)
						throw DeadlyImportError("GLTF: Open3DGC. Compressed texture coordinates count (" + to_string(tval) +
												") not equal to uncompressed (" + to_string(primitives[0].attributes.texcoord[idx]->count) + ").");

					idx_texcoord++;
				}
				else
				{
					ifs.SetNFloatAttribute(static_cast<unsigned long>(idx), 0ul);// Disable decoding this attribute.
				}

				break;
			default:
				throw DeadlyImportError("GLTF: Open3DGC. Unsupported type of float attribute: " + to_string(ifs.GetFloatAttributeType(static_cast<unsigned long>(idx))));
		}

		tval *=  ifs.GetFloatAttributeDim(static_cast<unsigned long>(idx)) * sizeof(o3dgc::Real);// After checking count of objects we can get size of array.
		size_floatattr[idx] = tval;
		decoded_data_size += tval;
	}

	for(size_t idx = 0, idx_end = size_intattr.size(); idx < idx_end; idx++)
	{
		// size = number_of_elements * components_per_element * size_of_component. See float attributes note.
		size_t tval = ifs.GetNIntAttribute(static_cast<unsigned long>(idx));
		switch( ifs.GetIntAttributeType(static_cast<unsigned long>(idx) ) )
		{
            case o3dgc::O3DGC_IFS_INT_ATTRIBUTE_TYPE_UNKOWN:
            case o3dgc::O3DGC_IFS_INT_ATTRIBUTE_TYPE_INDEX:
            case o3dgc::O3DGC_IFS_INT_ATTRIBUTE_TYPE_JOINT_ID:
            case o3dgc::O3DGC_IFS_INT_ATTRIBUTE_TYPE_INDEX_BUFFER_ID:
                break;

			default:
				throw DeadlyImportError("GLTF: Open3DGC. Unsupported type of int attribute: " + to_string(ifs.GetIntAttributeType(static_cast<unsigned long>(idx))));
		}

		tval *= ifs.GetIntAttributeDim(static_cast<unsigned long>(idx)) * sizeof(long);// See float attributes note.
		size_intattr[idx] = tval;
		decoded_data_size += tval;
	}

	// Create array for decoded data.
	decoded_data = new uint8_t[decoded_data_size];

	/****************** Set right array regions for decoder ******************/

	auto get_buf_offset = [](Ref<Accessor>& pAccessor) -> size_t { return pAccessor->byteOffset + pAccessor->bufferView->byteOffset; };

	// Indices
	ifs.SetCoordIndex((IndicesType* const)(decoded_data + get_buf_offset(primitives[0].indices)));
	// Coordinates
	ifs.SetCoord((o3dgc::Real* const)(decoded_data + get_buf_offset(primitives[0].attributes.position[0])));
	// Normals
	if(size_normal)
	{
		ifs.SetNormal((o3dgc::Real* const)(decoded_data + get_buf_offset(primitives[0].attributes.normal[0])));
	}

	for(size_t idx = 0, idx_end = size_floatattr.size(), idx_texcoord = 0; idx < idx_end; idx++)
	{
		switch(ifs.GetFloatAttributeType(static_cast<unsigned long>(idx)))
		{
			case o3dgc::O3DGC_IFS_FLOAT_ATTRIBUTE_TYPE_TEXCOORD:
				if(idx_texcoord < primitives[0].attributes.texcoord.size())
				{
					// See above about absent attributes.
					ifs.SetFloatAttribute(static_cast<unsigned long>(idx), (o3dgc::Real* const)(decoded_data + get_buf_offset(primitives[0].attributes.texcoord[idx])));
					idx_texcoord++;
				}

				break;
			default:
				throw DeadlyImportError("GLTF: Open3DGC. Unsupported type of float attribute: " + to_string(ifs.GetFloatAttributeType(static_cast<unsigned long>(idx))));
		}
	}

	for(size_t idx = 0, idx_end = size_intattr.size(); idx < idx_end; idx++) {
		switch(ifs.GetIntAttributeType(static_cast<unsigned int>(idx))) {
            case o3dgc::O3DGC_IFS_INT_ATTRIBUTE_TYPE_UNKOWN:
            case o3dgc::O3DGC_IFS_INT_ATTRIBUTE_TYPE_INDEX:
            case o3dgc::O3DGC_IFS_INT_ATTRIBUTE_TYPE_JOINT_ID:
            case o3dgc::O3DGC_IFS_INT_ATTRIBUTE_TYPE_INDEX_BUFFER_ID:
                break;

			// ifs.SetIntAttribute(idx, (long* const)(decoded_data + get_buf_offset(primitives[0].attributes.joint)));
			default:
				throw DeadlyImportError("GLTF: Open3DGC. Unsupported type of int attribute: " + to_string(ifs.GetIntAttributeType(static_cast<unsigned long>(idx))));
		}
	}

	//
	// Decode data
	//
    if ( decoder.DecodePayload( ifs, bstream ) != o3dgc::O3DGC_OK ) {
        throw DeadlyImportError( "GLTF: can not decode Open3DGC data." );
    }

	// Set encoded region for "buffer".
	buf->EncodedRegion_Mark(pCompression_Open3DGC.Offset, pCompression_Open3DGC.Count, decoded_data, decoded_data_size, id);
	// No. Do not delete "output_data". After calling "EncodedRegion_Mark" bufferView is owner of "output_data".
	// "delete [] output_data;"
}
#endif

inline void Camera::Read(Value& obj, Asset& /*r*/)
{
    type = MemberOrDefault(obj, "type", Camera::Perspective);

    const char* subobjId = (type == Camera::Orthographic) ? "ortographic" : "perspective";

    Value* it = FindObject(obj, subobjId);
    if (!it) throw DeadlyImportError("GLTF: Camera missing its parameters");

    if (type == Camera::Perspective) {
        perspective.aspectRatio = MemberOrDefault(*it, "aspectRatio", 0.f);
        perspective.yfov        = MemberOrDefault(*it, "yfov", 3.1415f/2.f);
        perspective.zfar        = MemberOrDefault(*it, "zfar", 100.f);
        perspective.znear       = MemberOrDefault(*it, "znear", 0.01f);
    }
    else {
        ortographic.xmag  = MemberOrDefault(obj, "xmag", 1.f);
        ortographic.ymag  = MemberOrDefault(obj, "ymag", 1.f);
        ortographic.zfar  = MemberOrDefault(obj, "zfar", 100.f);
        ortographic.znear = MemberOrDefault(obj, "znear", 0.01f);
    }
}

inline void Light::Read(Value& obj, Asset& /*r*/)
{
    SetDefaults();

    if (Value* type = FindString(obj, "type")) {
        const char* t = type->GetString();
        if      (strcmp(t, "ambient") == 0)     this->type = Type_ambient;
        else if (strcmp(t, "directional") == 0) this->type = Type_directional;
        else if (strcmp(t, "point") == 0)       this->type = Type_point;
        else if (strcmp(t, "spot") == 0)        this->type = Type_spot;

        if (this->type != Type_undefined) {
            if (Value* vals = FindString(obj, t)) {
                ReadMember(*vals, "color", color);

                ReadMember(*vals, "constantAttenuation", constantAttenuation);
                ReadMember(*vals, "linearAttenuation", linearAttenuation);
                ReadMember(*vals, "quadraticAttenuation", quadraticAttenuation);
                ReadMember(*vals, "distance", distance);

                ReadMember(*vals, "falloffAngle", falloffAngle);
                ReadMember(*vals, "falloffExponent", falloffExponent);
            }
        }
    }
}

inline void Light::SetDefaults()
{
    #ifndef M_PI
        const float M_PI = 3.14159265358979323846f;
    #endif

    type = Type_undefined;

    SetVector(color, 0.f, 0.f, 0.f, 1.f);

    constantAttenuation = 0.f;
    linearAttenuation = 1.f;
    quadraticAttenuation = 1.f;
    distance = 0.f;

    falloffAngle = static_cast<float>(M_PI / 2.f);
    falloffExponent = 0.f;
}

inline 
void Node::Read(Value& obj, Asset& r) {
    if (name.empty()) {
        name = id;
    }

    if (Value* children = FindArray(obj, "children")) {
        this->children.reserve(children->Size());
        for (unsigned int i = 0; i < children->Size(); ++i) {
            Value& child = (*children)[i];
            if (child.IsString()) {
                // get/create the child node
                Ref<Node> chn = r.nodes.Get(child.GetString());
                if (chn) this->children.push_back(chn);
            }
        }
    }


    if (Value* matrix = FindArray(obj, "matrix")) {
        ReadValue(*matrix, this->matrix);
    }
    else {
        ReadMember(obj, "translation", translation);
        ReadMember(obj, "scale", scale);
        ReadMember(obj, "rotation", rotation);
    }

    if (Value* meshes = FindArray(obj, "meshes")) {
        unsigned numMeshes = (unsigned)meshes->Size();

        std::vector<unsigned int> meshList;

        this->meshes.reserve(numMeshes);
        for (unsigned i = 0; i < numMeshes; ++i) {
            if ((*meshes)[i].IsString()) {
                Ref<Mesh> mesh = r.meshes.Get((*meshes)[i].GetString());
                if (mesh) this->meshes.push_back(mesh);
            }
        }
    }

    if (Value* camera = FindString(obj, "camera")) {
        this->camera = r.cameras.Get(camera->GetString());
        if (this->camera)
            this->camera->id = this->id;
    }

    // TODO load "skeletons", "skin", "jointName"

    if (Value* extensions = FindObject(obj, "extensions")) {
        if (r.extensionsUsed.KHR_materials_common) {

            if (Value* ext = FindObject(*extensions, "KHR_materials_common")) {
                if (Value* light = FindString(*ext, "light")) {
                    this->light = r.lights.Get(light->GetString());
                }
            }

        }
    }
}

inline void Scene::Read(Value& obj, Asset& r)
{
    if (Value* array = FindArray(obj, "nodes")) {
        for (unsigned int i = 0; i < array->Size(); ++i) {
            if (!(*array)[i].IsString()) continue;
            Ref<Node> node = r.nodes.Get((*array)[i].GetString());
            if (node)
                this->nodes.push_back(node);
        }
    }
}


inline void AssetMetadata::Read(Document& doc)
{
    // read the version, etc.
    if (Value* obj = FindObject(doc, "asset")) {
        ReadMember(*obj, "copyright", copyright);
        ReadMember(*obj, "generator", generator);

        premultipliedAlpha = MemberOrDefault(*obj, "premultipliedAlpha", false);

        if (Value* versionString = FindString(*obj, "version")) {
            version = versionString->GetString();
        } else if (Value* versionNumber = FindNumber (*obj, "version")) {
            char buf[4];

            ai_snprintf(buf, 4, "%.1f", versionNumber->GetDouble());

            version = buf;
        }

        if (Value* profile = FindObject(*obj, "profile")) {
            ReadMember(*profile, "api",     this->profile.api);
            ReadMember(*profile, "version", this->profile.version);
        }
    }

    if (version.empty() || version[0] != '1') {
        throw DeadlyImportError("GLTF: Unsupported glTF version: " + version);
    }
}



//
// Asset methods implementation
//

inline void Asset::ReadBinaryHeader(IOStream& stream)
{
    GLB_Header header;
    if (stream.Read(&header, sizeof(header), 1) != 1) {
        throw DeadlyImportError("GLTF: Unable to read the file header");
    }

    if (strncmp((char*)header.magic, AI_GLB_MAGIC_NUMBER, sizeof(header.magic)) != 0) {
        throw DeadlyImportError("GLTF: Invalid binary glTF file");
    }

    AI_SWAP4(header.version);
    asset.version = to_string(header.version);
    if (header.version != 1) {
        throw DeadlyImportError("GLTF: Unsupported binary glTF version");
    }

    AI_SWAP4(header.sceneFormat);
    if (header.sceneFormat != SceneFormat_JSON) {
        throw DeadlyImportError("GLTF: Unsupported binary glTF scene format");
    }

    AI_SWAP4(header.length);
    AI_SWAP4(header.sceneLength);

    mSceneLength = static_cast<size_t>(header.sceneLength);

    mBodyOffset = sizeof(header)+mSceneLength;
    mBodyOffset = (mBodyOffset + 3) & ~3; // Round up to next multiple of 4

    mBodyLength = header.length - mBodyOffset;
}

inline void Asset::Load(const std::string& pFile, bool isBinary)
{
    mCurrentAssetDir.clear();
    int pos = std::max(int(pFile.rfind('/')), int(pFile.rfind('\\')));
    if (pos != int(std::string::npos)) mCurrentAssetDir = pFile.substr(0, pos + 1);

    shared_ptr<IOStream> stream(OpenFile(pFile.c_str(), "rb", true));
    if (!stream) {
        throw DeadlyImportError("GLTF: Could not open file for reading");
    }

    // is binary? then read the header
    if (isBinary) {
        SetAsBinary(); // also creates the body buffer
        ReadBinaryHeader(*stream);
    }
    else {
        mSceneLength = stream->FileSize();
        mBodyLength = 0;
    }


    // read the scene data

    std::vector<char> sceneData(mSceneLength + 1);
    sceneData[mSceneLength] = '\0';

    if (stream->Read(&sceneData[0], 1, mSceneLength) != mSceneLength) {
        throw DeadlyImportError("GLTF: Could not read the file contents");
    }


    // parse the JSON document

    Document doc;
    doc.ParseInsitu(&sceneData[0]);

    if (doc.HasParseError()) {
        char buffer[32];
        ai_snprintf(buffer, 32, "%d", static_cast<int>(doc.GetErrorOffset()));
        throw DeadlyImportError(std::string("GLTF: JSON parse error, offset ") + buffer + ": "
            + GetParseError_En(doc.GetParseError()));
    }

    if (!doc.IsObject()) {
        throw DeadlyImportError("GLTF: JSON document root must be a JSON object");
    }

    // Fill the buffer instance for the current file embedded contents
    if (mBodyLength > 0) {
        if (!mBodyBuffer->LoadFromStream(*stream, mBodyLength, mBodyOffset)) {
            throw DeadlyImportError("GLTF: Unable to read gltf file");
        }
    }


    // Load the metadata
    asset.Read(doc);
    ReadExtensionsUsed(doc);

    // Prepare the dictionaries
    for (size_t i = 0; i < mDicts.size(); ++i) {
        mDicts[i]->AttachToDocument(doc);
    }



    // Read the "scene" property, which specifies which scene to load
    // and recursively load everything referenced by it
    if (Value* scene = FindString(doc, "scene")) {
        this->scene = scenes.Get(scene->GetString());
    }

    // Clean up
    for (size_t i = 0; i < mDicts.size(); ++i) {
        mDicts[i]->DetachFromDocument();
    }
}

inline void Asset::SetAsBinary()
{
    if (!extensionsUsed.KHR_binary_glTF) {
        extensionsUsed.KHR_binary_glTF = true;
        mBodyBuffer = buffers.Create("binary_glTF");
        mBodyBuffer->MarkAsSpecial();
    }
}


inline void Asset::ReadExtensionsUsed(Document& doc)
{
    Value* extsUsed = FindArray(doc, "extensionsUsed");
    if (!extsUsed) return;

    std::gltf_unordered_map<std::string, bool> exts;

    for (unsigned int i = 0; i < extsUsed->Size(); ++i) {
        if ((*extsUsed)[i].IsString()) {
            exts[(*extsUsed)[i].GetString()] = true;
        }
    }

    #define CHECK_EXT(EXT) \
        if (exts.find(#EXT) != exts.end()) extensionsUsed.EXT = true;

    CHECK_EXT(KHR_binary_glTF);
    CHECK_EXT(KHR_materials_common);

    #undef CHECK_EXT
}

inline IOStream* Asset::OpenFile(std::string path, const char* mode, bool /*absolute*/)
{
    #ifdef ASSIMP_API
        return mIOSystem->Open(path, mode);
    #else
        if (path.size() < 2) return 0;
        if (!absolute && path[1] != ':' && path[0] != '/') { // relative?
            path = mCurrentAssetDir + path;
        }
        FILE* f = fopen(path.c_str(), mode);
        return f ? new IOStream(f) : 0;
    #endif
}

inline std::string Asset::FindUniqueID(const std::string& str, const char* suffix)
{
    std::string id = str;

    if (!id.empty()) {
        if (mUsedIds.find(id) == mUsedIds.end())
            return id;

        id += "_";
    }

    id += suffix;

    Asset::IdMap::iterator it = mUsedIds.find(id);
    if (it == mUsedIds.end())
        return id;

    char buffer[1024];
    int offset = ai_snprintf(buffer, sizeof(buffer), "%s_", id.c_str());
    for (int i = 0; it != mUsedIds.end(); ++i) {
        ai_snprintf(buffer + offset, sizeof(buffer) - offset, "%d", i);
        id = buffer;
        it = mUsedIds.find(id);
    }

    return id;
}

} // ns glTF
