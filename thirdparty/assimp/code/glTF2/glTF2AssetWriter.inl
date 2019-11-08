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

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>

namespace glTF2 {

    using rapidjson::StringBuffer;
    using rapidjson::PrettyWriter;
    using rapidjson::Writer;
    using rapidjson::StringRef;
    using rapidjson::StringRef;

    namespace {

        template<size_t N>
        inline Value& MakeValue(Value& val, float(&r)[N], MemoryPoolAllocator<>& al) {
            val.SetArray();
            val.Reserve(N, al);
            for (decltype(N) i = 0; i < N; ++i) {
                val.PushBack(r[i], al);
            }
            return val;
        }

        inline Value& MakeValue(Value& val, const std::vector<float> & r, MemoryPoolAllocator<>& al) {
            val.SetArray();
            val.Reserve(static_cast<rapidjson::SizeType>(r.size()), al);
            for (unsigned int i = 0; i < r.size(); ++i) {
                val.PushBack(r[i], al);
            }
            return val;
        }

        inline Value& MakeValue(Value& val, float r, MemoryPoolAllocator<>& /*al*/) {
            val.SetDouble(r);

            return val;
        }

        template<class T>
        inline void AddRefsVector(Value& obj, const char* fieldId, std::vector< Ref<T> >& v, MemoryPoolAllocator<>& al) {
            if (v.empty()) return;
            Value lst;
            lst.SetArray();
            lst.Reserve(unsigned(v.size()), al);
            for (size_t i = 0; i < v.size(); ++i) {
                lst.PushBack(v[i]->index, al);
            }
            obj.AddMember(StringRef(fieldId), lst, al);
        }


    }

    inline void Write(Value& obj, Accessor& a, AssetWriter& w)
    {
        obj.AddMember("bufferView", a.bufferView->index, w.mAl);
        obj.AddMember("byteOffset", (unsigned int)a.byteOffset, w.mAl);

        obj.AddMember("componentType", int(a.componentType), w.mAl);
        obj.AddMember("count", (unsigned int)a.count, w.mAl);
        obj.AddMember("type", StringRef(AttribType::ToString(a.type)), w.mAl);

        Value vTmpMax, vTmpMin;
        obj.AddMember("max", MakeValue(vTmpMax, a.max, w.mAl), w.mAl);
        obj.AddMember("min", MakeValue(vTmpMin, a.min, w.mAl), w.mAl);
    }

    inline void Write(Value& obj, Animation& a, AssetWriter& w)
    {
        /****************** Channels *******************/
        Value channels;
        channels.SetArray();
        channels.Reserve(unsigned(a.channels.size()), w.mAl);

        for (size_t i = 0; i < unsigned(a.channels.size()); ++i) {
            Animation::Channel& c = a.channels[i];
            Value valChannel;
            valChannel.SetObject();
            {
                valChannel.AddMember("sampler", c.sampler, w.mAl);

                Value valTarget;
                valTarget.SetObject();
                {
                    valTarget.AddMember("node", c.target.node->index, w.mAl);
                    switch (c.target.path) {
                        case AnimationPath_TRANSLATION:
                            valTarget.AddMember("path", "translation", w.mAl);
                            break;
                        case AnimationPath_ROTATION:
                            valTarget.AddMember("path", "rotation", w.mAl);
                            break;
                        case AnimationPath_SCALE:
                            valTarget.AddMember("path", "scale", w.mAl);
                            break;
                        case AnimationPath_WEIGHTS:
                            valTarget.AddMember("path", "weights", w.mAl);
                            break;
                    }
                }
                valChannel.AddMember("target", valTarget, w.mAl);
            }
            channels.PushBack(valChannel, w.mAl);
        }
        obj.AddMember("channels", channels, w.mAl);

        /****************** Samplers *******************/
        Value valSamplers;
        valSamplers.SetArray();

        for (size_t i = 0; i < unsigned(a.samplers.size()); ++i) {
            Animation::Sampler& s = a.samplers[i];
            Value valSampler;
            valSampler.SetObject();
            {
                valSampler.AddMember("input", s.input->index, w.mAl);
                switch (s.interpolation) {
                    case Interpolation_LINEAR:
                        valSampler.AddMember("path", "LINEAR", w.mAl);
                        break;
                    case Interpolation_STEP:
                        valSampler.AddMember("path", "STEP", w.mAl);
                        break;
                    case Interpolation_CUBICSPLINE:
                        valSampler.AddMember("path", "CUBICSPLINE", w.mAl);
                        break;
                }
                valSampler.AddMember("output", s.output->index, w.mAl);
            }
            valSamplers.PushBack(valSampler, w.mAl);
        }
        obj.AddMember("samplers", valSamplers, w.mAl);
    }

    inline void Write(Value& obj, Buffer& b, AssetWriter& w)
    {
        obj.AddMember("byteLength", static_cast<uint64_t>(b.byteLength), w.mAl);

        const auto uri = b.GetURI();
        const auto relativeUri = uri.substr(uri.find_last_of("/\\") + 1u);
        obj.AddMember("uri", Value(relativeUri, w.mAl).Move(), w.mAl);
    }

    inline void Write(Value& obj, BufferView& bv, AssetWriter& w)
    {
        obj.AddMember("buffer", bv.buffer->index, w.mAl);
        obj.AddMember("byteOffset", static_cast<uint64_t>(bv.byteOffset), w.mAl);
        obj.AddMember("byteLength", static_cast<uint64_t>(bv.byteLength), w.mAl);
        if (bv.byteStride != 0) {
            obj.AddMember("byteStride", bv.byteStride, w.mAl);
        }
        if (bv.target != 0) {
            obj.AddMember("target", int(bv.target), w.mAl);
        }
    }

    inline void Write(Value& /*obj*/, Camera& /*c*/, AssetWriter& /*w*/)
    {

    }

    inline void Write(Value& /*obj*/, Light& /*c*/, AssetWriter& /*w*/)
    {

    }

    inline void Write(Value& obj, Image& img, AssetWriter& w)
    {
        if (img.bufferView) {
            obj.AddMember("bufferView", img.bufferView->index, w.mAl);
            obj.AddMember("mimeType", Value(img.mimeType, w.mAl).Move(), w.mAl);
        }
        else {
            std::string uri;
            if (img.HasData()) {
                uri = "data:" + (img.mimeType.empty() ? "application/octet-stream" : img.mimeType);
                uri += ";base64,";
                glTFCommon::Util::EncodeBase64(img.GetData(), img.GetDataLength(), uri);
            }
            else {
                uri = img.uri;
            }

            obj.AddMember("uri", Value(uri, w.mAl).Move(), w.mAl);
        }
    }

    namespace {
        inline void SetTexBasic(TextureInfo t, Value& tex, MemoryPoolAllocator<>& al)
        {
            tex.SetObject();
            tex.AddMember("index", t.texture->index, al);

            if (t.texCoord != 0) {
                tex.AddMember("texCoord", t.texCoord, al);
            }
        }

        inline void WriteTex(Value& obj, TextureInfo t, const char* propName, MemoryPoolAllocator<>& al)
        {

            if (t.texture) {
                Value tex;

                SetTexBasic(t, tex, al);

                obj.AddMember(StringRef(propName), tex, al);
            }
        }

        inline void WriteTex(Value& obj, NormalTextureInfo t, const char* propName, MemoryPoolAllocator<>& al)
        {

            if (t.texture) {
                Value tex;

                SetTexBasic(t, tex, al);

                if (t.scale != 1) {
                    tex.AddMember("scale", t.scale, al);
                }

                obj.AddMember(StringRef(propName), tex, al);
            }
        }

        inline void WriteTex(Value& obj, OcclusionTextureInfo t, const char* propName, MemoryPoolAllocator<>& al)
        {

            if (t.texture) {
                Value tex;

                SetTexBasic(t, tex, al);

                if (t.strength != 1) {
                    tex.AddMember("strength", t.strength, al);
                }

                obj.AddMember(StringRef(propName), tex, al);
            }
        }

        template<size_t N>
        inline void WriteVec(Value& obj, float(&prop)[N], const char* propName, MemoryPoolAllocator<>& al)
        {
            Value arr;
            obj.AddMember(StringRef(propName), MakeValue(arr, prop, al), al);
        }

        template<size_t N>
        inline void WriteVec(Value& obj, float(&prop)[N], const char* propName, const float(&defaultVal)[N], MemoryPoolAllocator<>& al)
        {
            if (!std::equal(std::begin(prop), std::end(prop), std::begin(defaultVal))) {
                WriteVec(obj, prop, propName, al);
            }
        }

        inline void WriteFloat(Value& obj, float prop, const char* propName, MemoryPoolAllocator<>& al)
        {
            Value num;
            obj.AddMember(StringRef(propName), MakeValue(num, prop, al), al);
        }
    }

    inline void Write(Value& obj, Material& m, AssetWriter& w)
    {
        Value pbrMetallicRoughness;
        pbrMetallicRoughness.SetObject();
        {
            WriteTex(pbrMetallicRoughness, m.pbrMetallicRoughness.baseColorTexture, "baseColorTexture", w.mAl);
            WriteTex(pbrMetallicRoughness, m.pbrMetallicRoughness.metallicRoughnessTexture, "metallicRoughnessTexture", w.mAl);
            WriteVec(pbrMetallicRoughness, m.pbrMetallicRoughness.baseColorFactor, "baseColorFactor", defaultBaseColor, w.mAl);

            if (m.pbrMetallicRoughness.metallicFactor != 1) {
                WriteFloat(pbrMetallicRoughness, m.pbrMetallicRoughness.metallicFactor, "metallicFactor", w.mAl);
            }

            if (m.pbrMetallicRoughness.roughnessFactor != 1) {
                WriteFloat(pbrMetallicRoughness, m.pbrMetallicRoughness.roughnessFactor, "roughnessFactor", w.mAl);
            }
        }

        if (!pbrMetallicRoughness.ObjectEmpty()) {
            obj.AddMember("pbrMetallicRoughness", pbrMetallicRoughness, w.mAl);
        }

        WriteTex(obj, m.normalTexture, "normalTexture", w.mAl);
        WriteTex(obj, m.emissiveTexture, "emissiveTexture", w.mAl);
        WriteTex(obj, m.occlusionTexture, "occlusionTexture", w.mAl);
        WriteVec(obj, m.emissiveFactor, "emissiveFactor", defaultEmissiveFactor, w.mAl);

        if (m.alphaCutoff != 0.5) {
            WriteFloat(obj, m.alphaCutoff, "alphaCutoff", w.mAl);
        }

        if (m.alphaMode != "OPAQUE") {
            obj.AddMember("alphaMode", Value(m.alphaMode, w.mAl).Move(), w.mAl);
        }

        if (m.doubleSided) {
            obj.AddMember("doubleSided", m.doubleSided, w.mAl);
        }

        Value exts;
        exts.SetObject();

        if (m.pbrSpecularGlossiness.isPresent) {
            Value pbrSpecularGlossiness;
            pbrSpecularGlossiness.SetObject();

            PbrSpecularGlossiness &pbrSG = m.pbrSpecularGlossiness.value;

            //pbrSpecularGlossiness
            WriteVec(pbrSpecularGlossiness, pbrSG.diffuseFactor, "diffuseFactor", defaultDiffuseFactor, w.mAl);
            WriteVec(pbrSpecularGlossiness, pbrSG.specularFactor, "specularFactor", defaultSpecularFactor, w.mAl);

            if (pbrSG.glossinessFactor != 1) {
                WriteFloat(pbrSpecularGlossiness, pbrSG.glossinessFactor, "glossinessFactor", w.mAl);
            }

            WriteTex(pbrSpecularGlossiness, pbrSG.diffuseTexture, "diffuseTexture", w.mAl);
            WriteTex(pbrSpecularGlossiness, pbrSG.specularGlossinessTexture, "specularGlossinessTexture", w.mAl);

            if (!pbrSpecularGlossiness.ObjectEmpty()) {
                exts.AddMember("KHR_materials_pbrSpecularGlossiness", pbrSpecularGlossiness, w.mAl);
            }
        }

        if (m.unlit) {
          Value unlit;
          unlit.SetObject();
          exts.AddMember("KHR_materials_unlit", unlit, w.mAl);
        }

        if (!exts.ObjectEmpty()) {
            obj.AddMember("extensions", exts, w.mAl);
        }
    }

    namespace {
        inline void WriteAttrs(AssetWriter& w, Value& attrs, Mesh::AccessorList& lst,
            const char* semantic, bool forceNumber = false)
        {
            if (lst.empty()) return;
            if (lst.size() == 1 && !forceNumber) {
                attrs.AddMember(StringRef(semantic), lst[0]->index, w.mAl);
            }
            else {
                for (size_t i = 0; i < lst.size(); ++i) {
                    char buffer[32];
                    ai_snprintf(buffer, 32, "%s_%d", semantic, int(i));
                    attrs.AddMember(Value(buffer, w.mAl).Move(), lst[i]->index, w.mAl);
                }
            }
        }
    }

    inline void Write(Value& obj, Mesh& m, AssetWriter& w)
    {
		/****************** Primitives *******************/
        Value primitives;
        primitives.SetArray();
        primitives.Reserve(unsigned(m.primitives.size()), w.mAl);

        for (size_t i = 0; i < m.primitives.size(); ++i) {
            Mesh::Primitive& p = m.primitives[i];
            Value prim;
            prim.SetObject();
            {
                prim.AddMember("mode", Value(int(p.mode)).Move(), w.mAl);

                if (p.material)
                    prim.AddMember("material", p.material->index, w.mAl);

                if (p.indices)
                    prim.AddMember("indices", p.indices->index, w.mAl);

                Value attrs;
                attrs.SetObject();
                {
                    WriteAttrs(w, attrs, p.attributes.position, "POSITION");
                    WriteAttrs(w, attrs, p.attributes.normal, "NORMAL");
                    WriteAttrs(w, attrs, p.attributes.texcoord, "TEXCOORD", true);
                    WriteAttrs(w, attrs, p.attributes.color, "COLOR", true);
                    WriteAttrs(w, attrs, p.attributes.joint, "JOINTS", true);
                    WriteAttrs(w, attrs, p.attributes.weight, "WEIGHTS", true);
                }
                prim.AddMember("attributes", attrs, w.mAl);
            }
            primitives.PushBack(prim, w.mAl);
        }

        obj.AddMember("primitives", primitives, w.mAl);
    }

    inline void Write(Value& obj, Node& n, AssetWriter& w)
    {

        if (n.matrix.isPresent) {
            Value val;
            obj.AddMember("matrix", MakeValue(val, n.matrix.value, w.mAl).Move(), w.mAl);
        }

        if (n.translation.isPresent) {
            Value val;
            obj.AddMember("translation", MakeValue(val, n.translation.value, w.mAl).Move(), w.mAl);
        }

        if (n.scale.isPresent) {
            Value val;
            obj.AddMember("scale", MakeValue(val, n.scale.value, w.mAl).Move(), w.mAl);
        }
        if (n.rotation.isPresent) {
            Value val;
            obj.AddMember("rotation", MakeValue(val, n.rotation.value, w.mAl).Move(), w.mAl);
        }

        AddRefsVector(obj, "children", n.children, w.mAl);

        if (!n.meshes.empty()) {
            obj.AddMember("mesh", n.meshes[0]->index, w.mAl);
        }

        AddRefsVector(obj, "skeletons", n.skeletons, w.mAl);

        if (n.skin) {
            obj.AddMember("skin", n.skin->index, w.mAl);
        }

        if (!n.jointName.empty()) {
          obj.AddMember("jointName", n.jointName, w.mAl);
        }
    }

    inline void Write(Value& /*obj*/, Program& /*b*/, AssetWriter& /*w*/)
    {

    }

    inline void Write(Value& obj, Sampler& b, AssetWriter& w)
    {
        if (!b.name.empty()) {
            obj.AddMember("name", b.name, w.mAl);
        }

        if (b.wrapS != SamplerWrap::UNSET && b.wrapS != SamplerWrap::Repeat) {
            obj.AddMember("wrapS", static_cast<unsigned int>(b.wrapS), w.mAl);
        }

        if (b.wrapT != SamplerWrap::UNSET && b.wrapT != SamplerWrap::Repeat) {
            obj.AddMember("wrapT", static_cast<unsigned int>(b.wrapT), w.mAl);
        }

        if (b.magFilter != SamplerMagFilter::UNSET) {
            obj.AddMember("magFilter", static_cast<unsigned int>(b.magFilter), w.mAl);
        }

        if (b.minFilter != SamplerMinFilter::UNSET) {
            obj.AddMember("minFilter", static_cast<unsigned int>(b.minFilter), w.mAl);
        }
    }

    inline void Write(Value& scene, Scene& s, AssetWriter& w)
    {
        AddRefsVector(scene, "nodes", s.nodes, w.mAl);
    }

    inline void Write(Value& /*obj*/, Shader& /*b*/, AssetWriter& /*w*/)
    {

    }

    inline void Write(Value& obj, Skin& b, AssetWriter& w)
    {
        /****************** jointNames *******************/
        Value vJointNames;
        vJointNames.SetArray();
        vJointNames.Reserve(unsigned(b.jointNames.size()), w.mAl);

        for (size_t i = 0; i < unsigned(b.jointNames.size()); ++i) {
            vJointNames.PushBack(b.jointNames[i]->index, w.mAl);
        }
        obj.AddMember("joints", vJointNames, w.mAl);

        if (b.bindShapeMatrix.isPresent) {
            Value val;
            obj.AddMember("bindShapeMatrix", MakeValue(val, b.bindShapeMatrix.value, w.mAl).Move(), w.mAl);
        }

        if (b.inverseBindMatrices) {
            obj.AddMember("inverseBindMatrices", b.inverseBindMatrices->index, w.mAl);
        }

    }

    inline void Write(Value& obj, Texture& tex, AssetWriter& w)
    {
        if (tex.source) {
            obj.AddMember("source", tex.source->index, w.mAl);
        }
        if (tex.sampler) {
            obj.AddMember("sampler", tex.sampler->index, w.mAl);
        }
    }


    inline AssetWriter::AssetWriter(Asset& a)
        : mDoc()
        , mAsset(a)
        , mAl(mDoc.GetAllocator())
    {
        mDoc.SetObject();

        WriteMetadata();
        WriteExtensionsUsed();

        // Dump the contents of the dictionaries
        for (size_t i = 0; i < a.mDicts.size(); ++i) {
            a.mDicts[i]->WriteObjects(*this);
        }

        // Add the target scene field
        if (mAsset.scene) {
            mDoc.AddMember("scene", mAsset.scene->index, mAl);
        }
    }

    inline void AssetWriter::WriteFile(const char* path)
    {
        std::unique_ptr<IOStream> jsonOutFile(mAsset.OpenFile(path, "wt", true));

        if (jsonOutFile == 0) {
            throw DeadlyExportError("Could not open output file: " + std::string(path));
        }

        StringBuffer docBuffer;

        PrettyWriter<StringBuffer> writer(docBuffer);
        mDoc.Accept(writer);

        if (jsonOutFile->Write(docBuffer.GetString(), docBuffer.GetSize(), 1) != 1) {
            throw DeadlyExportError("Failed to write scene data!");
        }

        // Write buffer data to separate .bin files
        for (unsigned int i = 0; i < mAsset.buffers.Size(); ++i) {
            Ref<Buffer> b = mAsset.buffers.Get(i);

            std::string binPath = b->GetURI();

            std::unique_ptr<IOStream> binOutFile(mAsset.OpenFile(binPath, "wb", true));

            if (binOutFile == 0) {
                throw DeadlyExportError("Could not open output file: " + binPath);
            }

            if (b->byteLength > 0) {
                if (binOutFile->Write(b->GetPointer(), b->byteLength, 1) != 1) {
                    throw DeadlyExportError("Failed to write binary file: " + binPath);
                }
            }
        }
    }

    inline void AssetWriter::WriteGLBFile(const char* path)
    {
        std::unique_ptr<IOStream> outfile(mAsset.OpenFile(path, "wb", true));

        if (outfile == 0) {
            throw DeadlyExportError("Could not open output file: " + std::string(path));
        }

        Ref<Buffer> bodyBuffer = mAsset.GetBodyBuffer();
        if (bodyBuffer->byteLength > 0) {
            rapidjson::Value glbBodyBuffer;
            glbBodyBuffer.SetObject();
            glbBodyBuffer.AddMember("byteLength", static_cast<uint64_t>(bodyBuffer->byteLength), mAl);
            mDoc["buffers"].PushBack(glbBodyBuffer, mAl);
        }

        // Padding with spaces as required by the spec
        uint32_t padding = 0x20202020;

        //
        // JSON chunk
        //

        StringBuffer docBuffer;
        Writer<StringBuffer> writer(docBuffer);
        mDoc.Accept(writer);

        uint32_t jsonChunkLength = (docBuffer.GetSize() + 3) & ~3; // Round up to next multiple of 4
        auto paddingLength = jsonChunkLength - docBuffer.GetSize();

        GLB_Chunk jsonChunk;
        jsonChunk.chunkLength = jsonChunkLength;
        jsonChunk.chunkType = ChunkType_JSON;
        AI_SWAP4(jsonChunk.chunkLength);

        outfile->Seek(sizeof(GLB_Header), aiOrigin_SET);
        if (outfile->Write(&jsonChunk, 1, sizeof(GLB_Chunk)) != sizeof(GLB_Chunk)) {
            throw DeadlyExportError("Failed to write scene data header!");
        }
        if (outfile->Write(docBuffer.GetString(), 1, docBuffer.GetSize()) != docBuffer.GetSize()) {
            throw DeadlyExportError("Failed to write scene data!");
        }
        if (paddingLength && outfile->Write(&padding, 1, paddingLength) != paddingLength) {
            throw DeadlyExportError("Failed to write scene data padding!");
        }

        //
        // Binary chunk
        //

        uint32_t binaryChunkLength = 0;
        if (bodyBuffer->byteLength > 0) {
            binaryChunkLength = (bodyBuffer->byteLength + 3) & ~3; // Round up to next multiple of 4
            auto paddingLength = binaryChunkLength - bodyBuffer->byteLength;

            GLB_Chunk binaryChunk;
            binaryChunk.chunkLength = binaryChunkLength;
            binaryChunk.chunkType = ChunkType_BIN;
            AI_SWAP4(binaryChunk.chunkLength);

            size_t bodyOffset = sizeof(GLB_Header) + sizeof(GLB_Chunk) + jsonChunk.chunkLength;
            outfile->Seek(bodyOffset, aiOrigin_SET);
            if (outfile->Write(&binaryChunk, 1, sizeof(GLB_Chunk)) != sizeof(GLB_Chunk)) {
                throw DeadlyExportError("Failed to write body data header!");
            }
            if (outfile->Write(bodyBuffer->GetPointer(), 1, bodyBuffer->byteLength) != bodyBuffer->byteLength) {
                throw DeadlyExportError("Failed to write body data!");
            }
            if (paddingLength && outfile->Write(&padding, 1, paddingLength) != paddingLength) {
                throw DeadlyExportError("Failed to write body data padding!");
            }
        }

        //
        // Header
        //

        GLB_Header header;
        memcpy(header.magic, AI_GLB_MAGIC_NUMBER, sizeof(header.magic));

        header.version = 2;
        AI_SWAP4(header.version);

        header.length = uint32_t(sizeof(GLB_Header) + 2 * sizeof(GLB_Chunk) + jsonChunkLength + binaryChunkLength);
        AI_SWAP4(header.length);

        outfile->Seek(0, aiOrigin_SET);
        if (outfile->Write(&header, 1, sizeof(GLB_Header)) != sizeof(GLB_Header)) {
            throw DeadlyExportError("Failed to write the header!");
        }
    }

    inline void AssetWriter::WriteMetadata()
    {
        Value asset;
        asset.SetObject();
        asset.AddMember("version", Value(mAsset.asset.version, mAl).Move(), mAl);
        asset.AddMember("generator", Value(mAsset.asset.generator, mAl).Move(), mAl);
        mDoc.AddMember("asset", asset, mAl);
    }

    inline void AssetWriter::WriteExtensionsUsed()
    {
        Value exts;
        exts.SetArray();
        {
            // This is used to export pbrSpecularGlossiness materials with GLTF 2.
            if (this->mAsset.extensionsUsed.KHR_materials_pbrSpecularGlossiness) {
                exts.PushBack(StringRef("KHR_materials_pbrSpecularGlossiness"), mAl);
            }

            if (this->mAsset.extensionsUsed.KHR_materials_unlit) {
              exts.PushBack(StringRef("KHR_materials_unlit"), mAl);
            }
        }

        if (!exts.Empty())
            mDoc.AddMember("extensionsUsed", exts, mAl);
    }

    template<class T>
    void AssetWriter::WriteObjects(LazyDict<T>& d)
    {
        if (d.mObjs.empty()) return;

        Value* container = &mDoc;

        if (d.mExtId) {
            Value* exts = FindObject(mDoc, "extensions");
            if (!exts) {
                mDoc.AddMember("extensions", Value().SetObject().Move(), mDoc.GetAllocator());
                exts = FindObject(mDoc, "extensions");
            }

            if (!(container = FindObject(*exts, d.mExtId))) {
                exts->AddMember(StringRef(d.mExtId), Value().SetObject().Move(), mDoc.GetAllocator());
                container = FindObject(*exts, d.mExtId);
            }
        }

        Value* dict;
        if (!(dict = FindArray(*container, d.mDictId))) {
            container->AddMember(StringRef(d.mDictId), Value().SetArray().Move(), mDoc.GetAllocator());
            dict = FindArray(*container, d.mDictId);
            if (nullptr == dict) {
                return;
            }
        }

        for (size_t i = 0; i < d.mObjs.size(); ++i) {
            if (d.mObjs[i]->IsSpecial()) continue;

            Value obj;
            obj.SetObject();

            if (!d.mObjs[i]->name.empty()) {
                obj.AddMember("name", StringRef(d.mObjs[i]->name.c_str()), mAl);
            }

            Write(obj, *d.mObjs[i], *this);

            dict->PushBack(obj, mAl);
        }
    }

    template<class T>
    void WriteLazyDict(LazyDict<T>& d, AssetWriter& w)
    {
        w.WriteObjects(d);
    }

}


