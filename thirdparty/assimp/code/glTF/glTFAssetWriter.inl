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

namespace glTF {

    using rapidjson::StringBuffer;
    using rapidjson::PrettyWriter;
    using rapidjson::Writer;
    using rapidjson::StringRef;
    using rapidjson::StringRef;

    namespace {

        template<size_t N>
        inline 
        Value& MakeValue(Value& val, float(&r)[N], MemoryPoolAllocator<>& al) {
            val.SetArray();
            val.Reserve(N, al);
            for (decltype(N) i = 0; i < N; ++i) {
                val.PushBack(r[i], al);
            }
            return val;
        }

        inline 
        Value& MakeValue(Value& val, const std::vector<float> & r, MemoryPoolAllocator<>& al) {
            val.SetArray();
            val.Reserve(static_cast<rapidjson::SizeType>(r.size()), al);
            for (unsigned int i = 0; i < r.size(); ++i) {
                val.PushBack(r[i], al);
            }
            return val;
        }

        template<class T>
        inline void AddRefsVector(Value& obj, const char* fieldId, std::vector< Ref<T> >& v, MemoryPoolAllocator<>& al) {
            if (v.empty()) return;
            Value lst;
            lst.SetArray();
            lst.Reserve(unsigned(v.size()), al);
            for (size_t i = 0; i < v.size(); ++i) {
                lst.PushBack(StringRef(v[i]->id), al);
            }
            obj.AddMember(StringRef(fieldId), lst, al);
        }


    }

    inline void Write(Value& obj, Accessor& a, AssetWriter& w)
    {
        obj.AddMember("bufferView", Value(a.bufferView->id, w.mAl).Move(), w.mAl);
        obj.AddMember("byteOffset", a.byteOffset, w.mAl);
        obj.AddMember("byteStride", a.byteStride, w.mAl);
        obj.AddMember("componentType", int(a.componentType), w.mAl);
        obj.AddMember("count", a.count, w.mAl);
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
        channels.Reserve(unsigned(a.Channels.size()), w.mAl);

        for (size_t i = 0; i < unsigned(a.Channels.size()); ++i) {
            Animation::AnimChannel& c = a.Channels[i];
            Value valChannel;
            valChannel.SetObject();
            {
                valChannel.AddMember("sampler", c.sampler, w.mAl);

                Value valTarget;
                valTarget.SetObject();
                {
                    valTarget.AddMember("id", StringRef(c.target.id->id), w.mAl);
                    valTarget.AddMember("path", c.target.path, w.mAl);
                }
                valChannel.AddMember("target", valTarget, w.mAl);
            }
            channels.PushBack(valChannel, w.mAl);
        }
        obj.AddMember("channels", channels, w.mAl);

        /****************** Parameters *******************/
        Value valParameters;
        valParameters.SetObject();
        {
            if (a.Parameters.TIME) {
                valParameters.AddMember("TIME", StringRef(a.Parameters.TIME->id), w.mAl);
            }
            if (a.Parameters.rotation) {
                valParameters.AddMember("rotation", StringRef(a.Parameters.rotation->id), w.mAl);
            }
            if (a.Parameters.scale) {
                valParameters.AddMember("scale", StringRef(a.Parameters.scale->id), w.mAl);
            }
            if (a.Parameters.translation) {
                valParameters.AddMember("translation", StringRef(a.Parameters.translation->id), w.mAl);
            }
        }
        obj.AddMember("parameters", valParameters, w.mAl);

        /****************** Samplers *******************/
        Value valSamplers;
        valSamplers.SetObject();

        for (size_t i = 0; i < unsigned(a.Samplers.size()); ++i) {
            Animation::AnimSampler& s = a.Samplers[i];
            Value valSampler;
            valSampler.SetObject();
            {
                valSampler.AddMember("input", s.input, w.mAl);
                valSampler.AddMember("interpolation", s.interpolation, w.mAl);
                valSampler.AddMember("output", s.output, w.mAl);
            }
            valSamplers.AddMember(StringRef(s.id), valSampler, w.mAl);
        }
        obj.AddMember("samplers", valSamplers, w.mAl);
    }

    inline void Write(Value& obj, Buffer& b, AssetWriter& w)
    {
        const char* type;
        switch (b.type) {
            case Buffer::Type_text:
                type = "text"; break;
            default:
                type = "arraybuffer";
        }

        obj.AddMember("byteLength", static_cast<uint64_t>(b.byteLength), w.mAl);
        obj.AddMember("type", StringRef(type), w.mAl);
        obj.AddMember("uri", Value(b.GetURI(), w.mAl).Move(), w.mAl);
    }

    inline void Write(Value& obj, BufferView& bv, AssetWriter& w)
    {
        obj.AddMember("buffer", Value(bv.buffer->id, w.mAl).Move(), w.mAl);
        obj.AddMember("byteOffset", static_cast<uint64_t>(bv.byteOffset), w.mAl);
        obj.AddMember("byteLength", static_cast<uint64_t>(bv.byteLength), w.mAl);
        obj.AddMember("target", int(bv.target), w.mAl);
    }

    inline void Write(Value& /*obj*/, Camera& /*c*/, AssetWriter& /*w*/)
    {

    }

    inline void Write(Value& obj, Image& img, AssetWriter& w)
    {
        std::string uri;
        if (w.mAsset.extensionsUsed.KHR_binary_glTF && img.bufferView) {
            Value exts, ext;
            exts.SetObject();
            ext.SetObject();

            ext.AddMember("bufferView", StringRef(img.bufferView->id), w.mAl);

            if (!img.mimeType.empty())
                ext.AddMember("mimeType", StringRef(img.mimeType), w.mAl);

            exts.AddMember("KHR_binary_glTF", ext, w.mAl);
            obj.AddMember("extensions", exts, w.mAl);
            return;
        }
        else if (img.HasData()) {
            uri = "data:" + (img.mimeType.empty() ? "application/octet-stream" : img.mimeType);
            uri += ";base64,";
            glTFCommon::Util::EncodeBase64(img.GetData(), img.GetDataLength(), uri);
        }
        else {
            uri = img.uri;
        }

        obj.AddMember("uri", Value(uri, w.mAl).Move(), w.mAl);
    }

    namespace {
        inline void WriteColorOrTex(Value& obj, TexProperty& prop, const char* propName, MemoryPoolAllocator<>& al)
        {
            if (prop.texture)
                obj.AddMember(StringRef(propName), Value(prop.texture->id, al).Move(), al);
            else {
                Value col;
                obj.AddMember(StringRef(propName), MakeValue(col, prop.color, al), al);
            }
        }
    }

    inline void Write(Value& obj, Material& m, AssetWriter& w)
    {
        Value v;
        v.SetObject();
        {
            WriteColorOrTex(v, m.ambient, "ambient", w.mAl);
            WriteColorOrTex(v, m.diffuse, "diffuse", w.mAl);
            WriteColorOrTex(v, m.specular, "specular", w.mAl);
            WriteColorOrTex(v, m.emission, "emission", w.mAl);

            if (m.transparent)
                v.AddMember("transparency", m.transparency, w.mAl);

            v.AddMember("shininess", m.shininess, w.mAl);
        }
        obj.AddMember("values", v, w.mAl);
    }

    namespace {
        inline void WriteAttrs(AssetWriter& w, Value& attrs, Mesh::AccessorList& lst,
            const char* semantic, bool forceNumber = false)
        {
            if (lst.empty()) return;
            if (lst.size() == 1 && !forceNumber) {
                attrs.AddMember(StringRef(semantic), Value(lst[0]->id, w.mAl).Move(), w.mAl);
            }
            else {
                for (size_t i = 0; i < lst.size(); ++i) {
                    char buffer[32];
                    ai_snprintf(buffer, 32, "%s_%d", semantic, int(i));
                    attrs.AddMember(Value(buffer, w.mAl).Move(), Value(lst[i]->id, w.mAl).Move(), w.mAl);
                }
            }
        }
    }

    inline void Write(Value& obj, Mesh& m, AssetWriter& w)
    {
		/********************* Name **********************/
		obj.AddMember("name", m.name, w.mAl);

		/**************** Mesh extensions ****************/
		if(m.Extension.size() > 0)
		{
			Value json_extensions;

			json_extensions.SetObject();
			for(Mesh::SExtension* ptr_ext : m.Extension)
			{
				switch(ptr_ext->Type)
				{
#ifdef ASSIMP_IMPORTER_GLTF_USE_OPEN3DGC
					case Mesh::SExtension::EType::Compression_Open3DGC:
						{
							Value json_comp_data;
							Mesh::SCompression_Open3DGC* ptr_ext_comp = (Mesh::SCompression_Open3DGC*)ptr_ext;

							// filling object "compressedData"
							json_comp_data.SetObject();
							json_comp_data.AddMember("buffer", ptr_ext_comp->Buffer, w.mAl);
							json_comp_data.AddMember("byteOffset", static_cast<uint64_t>(ptr_ext_comp->Offset), w.mAl);
							json_comp_data.AddMember("componentType", 5121, w.mAl);
							json_comp_data.AddMember("type", "SCALAR", w.mAl);
							json_comp_data.AddMember("count", static_cast<uint64_t>(ptr_ext_comp->Count), w.mAl);
							if(ptr_ext_comp->Binary)
								json_comp_data.AddMember("mode", "binary", w.mAl);
							else
								json_comp_data.AddMember("mode", "ascii", w.mAl);

							json_comp_data.AddMember("indicesCount", static_cast<uint64_t>(ptr_ext_comp->IndicesCount), w.mAl);
							json_comp_data.AddMember("verticesCount", static_cast<uint64_t>(ptr_ext_comp->VerticesCount), w.mAl);
							// filling object "Open3DGC-compression"
							Value json_o3dgc;

							json_o3dgc.SetObject();
							json_o3dgc.AddMember("compressedData", json_comp_data, w.mAl);
							// add member to object "extensions"
							json_extensions.AddMember("Open3DGC-compression", json_o3dgc, w.mAl);
						}

						break;
#endif
					default:
						throw DeadlyImportError("GLTF: Can not write mesh: unknown mesh extension, only Open3DGC is supported.");
				}// switch(ptr_ext->Type)
			}// for(Mesh::SExtension* ptr_ext : m.Extension)

			// Add extensions to mesh
			obj.AddMember("extensions", json_extensions, w.mAl);
		}// if(m.Extension.size() > 0)

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
                    prim.AddMember("material", p.material->id, w.mAl);

                if (p.indices)
                    prim.AddMember("indices", Value(p.indices->id, w.mAl).Move(), w.mAl);

                Value attrs;
                attrs.SetObject();
                {
                    WriteAttrs(w, attrs, p.attributes.position, "POSITION");
                    WriteAttrs(w, attrs, p.attributes.normal, "NORMAL");
                    WriteAttrs(w, attrs, p.attributes.texcoord, "TEXCOORD", true);
                    WriteAttrs(w, attrs, p.attributes.color, "COLOR");
                    WriteAttrs(w, attrs, p.attributes.joint, "JOINT");
                    WriteAttrs(w, attrs, p.attributes.jointmatrix, "JOINTMATRIX");
                    WriteAttrs(w, attrs, p.attributes.weight, "WEIGHT");
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

        AddRefsVector(obj, "meshes", n.meshes, w.mAl);

        AddRefsVector(obj, "skeletons", n.skeletons, w.mAl);

        if (n.skin) {
            obj.AddMember("skin", Value(n.skin->id, w.mAl).Move(), w.mAl);
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
        if (b.wrapS) {
            obj.AddMember("wrapS", b.wrapS, w.mAl);
        }
        if (b.wrapT) {
            obj.AddMember("wrapT", b.wrapT, w.mAl);
        }
        if (b.magFilter) {
            obj.AddMember("magFilter", b.magFilter, w.mAl);
        }
        if (b.minFilter) {
            obj.AddMember("minFilter", b.minFilter, w.mAl);
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
            vJointNames.PushBack(StringRef(b.jointNames[i]->jointName), w.mAl);
        }
        obj.AddMember("jointNames", vJointNames, w.mAl);

        if (b.bindShapeMatrix.isPresent) {
            Value val;
            obj.AddMember("bindShapeMatrix", MakeValue(val, b.bindShapeMatrix.value, w.mAl).Move(), w.mAl);
        }

        if (b.inverseBindMatrices) {
            obj.AddMember("inverseBindMatrices", Value(b.inverseBindMatrices->id, w.mAl).Move(), w.mAl);
        }

    }

    inline void Write(Value& /*obj*/, Technique& /*b*/, AssetWriter& /*w*/)
    {

    }

    inline void Write(Value& obj, Texture& tex, AssetWriter& w)
    {
        if (tex.source) {
            obj.AddMember("source", Value(tex.source->id, w.mAl).Move(), w.mAl);
        }
        if (tex.sampler) {
            obj.AddMember("sampler", Value(tex.sampler->id, w.mAl).Move(), w.mAl);
        }
    }

    inline void Write(Value& /*obj*/, Light& /*b*/, AssetWriter& /*w*/)
    {

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
            mDoc.AddMember("scene", StringRef(mAsset.scene->id), mAl);
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

        // we will write the header later, skip its size
        outfile->Seek(sizeof(GLB_Header), aiOrigin_SET);

        StringBuffer docBuffer;
        Writer<StringBuffer> writer(docBuffer);
        mDoc.Accept(writer);

        if (outfile->Write(docBuffer.GetString(), docBuffer.GetSize(), 1) != 1) {
            throw DeadlyExportError("Failed to write scene data!");
        }

        WriteBinaryData(outfile.get(), docBuffer.GetSize());
    }

    inline void AssetWriter::WriteBinaryData(IOStream* outfile, size_t sceneLength)
    {
        //
        // write the body data
        //

        size_t bodyLength = 0;
        if (Ref<Buffer> b = mAsset.GetBodyBuffer()) {
            bodyLength = b->byteLength;

            if (bodyLength > 0) {
                size_t bodyOffset = sizeof(GLB_Header) + sceneLength;
                bodyOffset = (bodyOffset + 3) & ~3; // Round up to next multiple of 4

                outfile->Seek(bodyOffset, aiOrigin_SET);

                if (outfile->Write(b->GetPointer(), b->byteLength, 1) != 1) {
                    throw DeadlyExportError("Failed to write body data!");
                }
            }
        }

        //
        // write the header
        //

        GLB_Header header;
        memcpy(header.magic, AI_GLB_MAGIC_NUMBER, sizeof(header.magic));

        header.version = 1;
        AI_SWAP4(header.version);

        header.length = uint32_t(sizeof(header) + sceneLength + bodyLength);
        AI_SWAP4(header.length);

        header.sceneLength = uint32_t(sceneLength);
        AI_SWAP4(header.sceneLength);

        header.sceneFormat = SceneFormat_JSON;
        AI_SWAP4(header.sceneFormat);

        outfile->Seek(0, aiOrigin_SET);

        if (outfile->Write(&header, 1, sizeof(header)) != sizeof(header)) {
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
            if (false)
                exts.PushBack(StringRef("KHR_binary_glTF"), mAl);

            if (false)
                exts.PushBack(StringRef("KHR_materials_common"), mAl);
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
        if (!(dict = FindObject(*container, d.mDictId))) {
            container->AddMember(StringRef(d.mDictId), Value().SetObject().Move(), mDoc.GetAllocator());
            dict = FindObject(*container, d.mDictId);
        }

        for (size_t i = 0; i < d.mObjs.size(); ++i) {
            if (d.mObjs[i]->IsSpecial()) continue;

            Value obj;
            obj.SetObject();

            if (!d.mObjs[i]->name.empty()) {
                obj.AddMember("name", StringRef(d.mObjs[i]->name.c_str()), mAl);
            }

            Write(obj, *d.mObjs[i], *this);

            dict->AddMember(StringRef(d.mObjs[i]->id), obj, mAl);
        }
    }

    template<class T>
    void WriteLazyDict(LazyDict<T>& d, AssetWriter& w)
    {
        w.WriteObjects(d);
    }

}


