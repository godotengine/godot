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
#ifndef ASSIMP_BUILD_NO_EXPORT
#ifndef ASSIMP_BUILD_NO_FBX_EXPORTER

#include "FBXExporter.h"
#include "FBXExportNode.h"
#include "FBXExportProperty.h"
#include "FBXCommon.h"

#include <assimp/version.h> // aiGetVersion
#include <assimp/IOSystem.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/DefaultLogger.hpp>
#include <assimp/StreamWriter.h> // StreamWriterLE
#include <assimp/Exceptional.h> // DeadlyExportError
#include <assimp/material.h> // aiTextureType
#include <assimp/scene.h>
#include <assimp/mesh.h>

// Header files, standard library.
#include <memory> // shared_ptr
#include <string>
#include <sstream> // stringstream
#include <ctime> // localtime, tm_*
#include <map>
#include <set>
#include <vector>
#include <array>
#include <unordered_set>

// RESOURCES:
// https://code.blender.org/2013/08/fbx-binary-file-format-specification/
// https://wiki.blender.org/index.php/User:Mont29/Foundation/FBX_File_Structure

const ai_real DEG = ai_real( 57.29577951308232087679815481 ); // degrees per radian

// some constants that we'll use for writing metadata
namespace FBX {
    const std::string EXPORT_VERSION_STR = "7.4.0";
    const uint32_t EXPORT_VERSION_INT = 7400; // 7.4 == 2014/2015
    // FBX files have some hashed values that depend on the creation time field,
    // but for now we don't actually know how to generate these.
    // what we can do is set them to a known-working version.
    // this is the data that Blender uses in their FBX export process.
    const std::string GENERIC_CTIME = "1970-01-01 10:00:00:000";
    const std::string GENERIC_FILEID =
        "\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1";
    const std::string GENERIC_FOOTID =
        "\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e";
    const std::string FOOT_MAGIC =
        "\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b";
    const std::string COMMENT_UNDERLINE =
        ";------------------------------------------------------------------";
}

using namespace Assimp;
using namespace FBX;

namespace Assimp {

    // ---------------------------------------------------------------------
    // Worker function for exporting a scene to binary FBX.
    // Prototyped and registered in Exporter.cpp
    void ExportSceneFBX (
        const char* pFile,
        IOSystem* pIOSystem,
        const aiScene* pScene,
        const ExportProperties* pProperties
    ){
        // initialize the exporter
        FBXExporter exporter(pScene, pProperties);

        // perform binary export
        exporter.ExportBinary(pFile, pIOSystem);
    }

    // ---------------------------------------------------------------------
    // Worker function for exporting a scene to ASCII FBX.
    // Prototyped and registered in Exporter.cpp
    void ExportSceneFBXA (
        const char* pFile,
        IOSystem* pIOSystem,
        const aiScene* pScene,
        const ExportProperties* pProperties
    ){
        // initialize the exporter
        FBXExporter exporter(pScene, pProperties);

        // perform ascii export
        exporter.ExportAscii(pFile, pIOSystem);
    }

} // end of namespace Assimp

FBXExporter::FBXExporter ( const aiScene* pScene, const ExportProperties* pProperties )
: binary(false)
, mScene(pScene)
, mProperties(pProperties)
, outfile()
, connections()
, mesh_uids()
, material_uids()
, node_uids() {
    // will probably need to determine UIDs, connections, etc here.
    // basically anything that needs to be known
    // before we start writing sections to the stream.
}

void FBXExporter::ExportBinary (
    const char* pFile,
    IOSystem* pIOSystem
){
    // remember that we're exporting in binary mode
    binary = true;

    // we're not currently using these preferences,
    // but clang will cry about it if we never touch it.
    // TODO: some of these might be relevant to export
    (void)mProperties;

    // open the indicated file for writing (in binary mode)
    outfile.reset(pIOSystem->Open(pFile,"wb"));
    if (!outfile) {
        throw DeadlyExportError(
            "could not open output .fbx file: " + std::string(pFile)
        );
    }

    // first a binary-specific file header
    WriteBinaryHeader();

    // the rest of the file is in node entries.
    // we have to serialize each entry before we write to the output,
    // as the first thing we write is the byte offset of the _next_ entry.
    // Either that or we can skip back to write the offset when we finish.
    WriteAllNodes();

    // finally we have a binary footer to the file
    WriteBinaryFooter();

    // explicitly release file pointer,
    // so we don't have to rely on class destruction.
    outfile.reset();
}

void FBXExporter::ExportAscii (
    const char* pFile,
    IOSystem* pIOSystem
){
    // remember that we're exporting in ascii mode
    binary = false;

    // open the indicated file for writing in text mode
    outfile.reset(pIOSystem->Open(pFile,"wt"));
    if (!outfile) {
        throw DeadlyExportError(
            "could not open output .fbx file: " + std::string(pFile)
        );
    }

    // write the ascii header
    WriteAsciiHeader();

    // write all the sections
    WriteAllNodes();

    // make sure the file ends with a newline.
    // note: if the file is opened in text mode,
    // this should do the right cross-platform thing.
    outfile->Write("\n", 1, 1);

    // explicitly release file pointer,
    // so we don't have to rely on class destruction.
    outfile.reset();
}

void FBXExporter::WriteAsciiHeader()
{
    // basically just a comment at the top of the file
    std::stringstream head;
    head << "; FBX " << EXPORT_VERSION_STR << " project file\n";
    head << "; Created by the Open Asset Import Library (Assimp)\n";
    head << "; http://assimp.org\n";
    head << "; -------------------------------------------------\n";
    const std::string ascii_header = head.str();
    outfile->Write(ascii_header.c_str(), ascii_header.size(), 1);
}

void FBXExporter::WriteAsciiSectionHeader(const std::string& title)
{
    StreamWriterLE outstream(outfile);
    std::stringstream s;
    s << "\n\n; " << title << '\n';
    s << FBX::COMMENT_UNDERLINE << "\n";
    outstream.PutString(s.str());
}

void FBXExporter::WriteBinaryHeader()
{
    // first a specific sequence of 23 bytes, always the same
    const char binary_header[24] = "Kaydara FBX Binary\x20\x20\x00\x1a\x00";
    outfile->Write(binary_header, 1, 23);

    // then FBX version number, "multiplied" by 1000, as little-endian uint32.
    // so 7.3 becomes 7300 == 0x841C0000, 7.4 becomes 7400 == 0xE81C0000, etc
    {
        StreamWriterLE outstream(outfile);
        outstream.PutU4(EXPORT_VERSION_INT);
    } // StreamWriter destructor writes the data to the file

    // after this the node data starts immediately
    // (probably with the FBXHEaderExtension node)
}

void FBXExporter::WriteBinaryFooter()
{
    outfile->Write(NULL_RECORD.c_str(), NULL_RECORD.size(), 1);

    outfile->Write(GENERIC_FOOTID.c_str(), GENERIC_FOOTID.size(), 1);

    // here some padding is added for alignment to 16 bytes.
    // if already aligned, the full 16 bytes is added.
    size_t pos = outfile->Tell();
    size_t pad = 16 - (pos % 16);
    for (size_t i = 0; i < pad; ++i) {
        outfile->Write("\x00", 1, 1);
    }

    // not sure what this is, but it seems to always be 0 in modern files
    for (size_t i = 0; i < 4; ++i) {
        outfile->Write("\x00", 1, 1);
    }

    // now the file version again
    {
        StreamWriterLE outstream(outfile);
        outstream.PutU4(EXPORT_VERSION_INT);
    } // StreamWriter destructor writes the data to the file

    // and finally some binary footer added to all files
    for (size_t i = 0; i < 120; ++i) {
        outfile->Write("\x00", 1, 1);
    }
    outfile->Write(FOOT_MAGIC.c_str(), FOOT_MAGIC.size(), 1);
}

void FBXExporter::WriteAllNodes ()
{
    // header
    // (and fileid, creation time, creator, if binary)
    WriteHeaderExtension();

    // global settings
    WriteGlobalSettings();

    // documents
    WriteDocuments();

    // references
    WriteReferences();

    // definitions
    WriteDefinitions();

    // objects
    WriteObjects();

    // connections
    WriteConnections();

    // WriteTakes? (deprecated since at least 2015 (fbx 7.4))
}

//FBXHeaderExtension top-level node
void FBXExporter::WriteHeaderExtension ()
{
    if (!binary) {
        // no title, follows directly from the top comment
    }
    FBX::Node n("FBXHeaderExtension");
    StreamWriterLE outstream(outfile);
    int indent = 0;

    // begin node
    n.Begin(outstream, binary, indent);

    // write properties
    // (none)

    // finish properties
    n.EndProperties(outstream, binary, indent, 0);

    // begin children
    n.BeginChildren(outstream, binary, indent);

    indent = 1;

    // write child nodes
    FBX::Node::WritePropertyNode(
        "FBXHeaderVersion", int32_t(1003), outstream, binary, indent
    );
    FBX::Node::WritePropertyNode(
        "FBXVersion", int32_t(EXPORT_VERSION_INT), outstream, binary, indent
    );
    if (binary) {
        FBX::Node::WritePropertyNode(
            "EncryptionType", int32_t(0), outstream, binary, indent
        );
    }

    FBX::Node CreationTimeStamp("CreationTimeStamp");
    time_t rawtime;
    time(&rawtime);
    struct tm * now = localtime(&rawtime);
    CreationTimeStamp.AddChild("Version", int32_t(1000));
    CreationTimeStamp.AddChild("Year", int32_t(now->tm_year + 1900));
    CreationTimeStamp.AddChild("Month", int32_t(now->tm_mon + 1));
    CreationTimeStamp.AddChild("Day", int32_t(now->tm_mday));
    CreationTimeStamp.AddChild("Hour", int32_t(now->tm_hour));
    CreationTimeStamp.AddChild("Minute", int32_t(now->tm_min));
    CreationTimeStamp.AddChild("Second", int32_t(now->tm_sec));
    CreationTimeStamp.AddChild("Millisecond", int32_t(0));
    CreationTimeStamp.Dump(outstream, binary, indent);

    std::stringstream creator;
    creator << "Open Asset Import Library (Assimp) " << aiGetVersionMajor()
            << "." << aiGetVersionMinor() << "." << aiGetVersionRevision();
    FBX::Node::WritePropertyNode(
        "Creator", creator.str(), outstream, binary, indent
    );

    //FBX::Node sceneinfo("SceneInfo");
    //sceneinfo.AddProperty("GlobalInfo" + FBX::SEPARATOR + "SceneInfo");
    // not sure if any of this is actually needed,
    // so just write an empty node for now.
    //sceneinfo.Dump(outstream, binary, indent);

    indent = 0;

    // finish node
    n.End(outstream, binary, indent, true);

    // that's it for FBXHeaderExtension...
    if (!binary) { return; }

    // but binary files also need top-level FileID, CreationTime, Creator:
    std::vector<uint8_t> raw(GENERIC_FILEID.size());
    for (size_t i = 0; i < GENERIC_FILEID.size(); ++i) {
        raw[i] = uint8_t(GENERIC_FILEID[i]);
    }
    FBX::Node::WritePropertyNode(
        "FileId", raw, outstream, binary, indent
    );
    FBX::Node::WritePropertyNode(
        "CreationTime", GENERIC_CTIME, outstream, binary, indent
    );
    FBX::Node::WritePropertyNode(
        "Creator", creator.str(), outstream, binary, indent
    );
}

void FBXExporter::WriteGlobalSettings ()
{
    if (!binary) {
        // no title, follows directly from the header extension
    }
    FBX::Node gs("GlobalSettings");
    gs.AddChild("Version", int32_t(1000));

    FBX::Node p("Properties70");
    p.AddP70int("UpAxis", 1);
    p.AddP70int("UpAxisSign", 1);
    p.AddP70int("FrontAxis", 2);
    p.AddP70int("FrontAxisSign", 1);
    p.AddP70int("CoordAxis", 0);
    p.AddP70int("CoordAxisSign", 1);
    p.AddP70int("OriginalUpAxis", 1);
    p.AddP70int("OriginalUpAxisSign", 1);
    p.AddP70double("UnitScaleFactor", 1.0);
    p.AddP70double("OriginalUnitScaleFactor", 1.0);
    p.AddP70color("AmbientColor", 0.0, 0.0, 0.0);
    p.AddP70string("DefaultCamera", "Producer Perspective");
    p.AddP70enum("TimeMode", 11);
    p.AddP70enum("TimeProtocol", 2);
    p.AddP70enum("SnapOnFrameMode", 0);
    p.AddP70time("TimeSpanStart", 0); // TODO: animation support
    p.AddP70time("TimeSpanStop", FBX::SECOND); // TODO: animation support
    p.AddP70double("CustomFrameRate", -1.0);
    p.AddP70("TimeMarker", "Compound", "", ""); // not sure what this is
    p.AddP70int("CurrentTimeMarker", -1);
    gs.AddChild(p);

    gs.Dump(outfile, binary, 0);
}

void FBXExporter::WriteDocuments ()
{
    if (!binary) {
        WriteAsciiSectionHeader("Documents Description");
    }
    
    // not sure what the use of multiple documents would be,
    // or whether any end-application supports it
    FBX::Node docs("Documents");
    docs.AddChild("Count", int32_t(1));
    FBX::Node doc("Document");

    // generate uid
    int64_t uid = generate_uid();
    doc.AddProperties(uid, "", "Scene");
    FBX::Node p("Properties70");
    p.AddP70("SourceObject", "object", "", ""); // what is this even for?
    p.AddP70string("ActiveAnimStackName", ""); // should do this properly?
    doc.AddChild(p);

    // UID for root node in scene hierarchy.
    // always set to 0 in the case of a single document.
    // not sure what happens if more than one document exists,
    // but that won't matter to us as we're exporting a single scene.
    doc.AddChild("RootNode", int64_t(0));

    docs.AddChild(doc);
    docs.Dump(outfile, binary, 0);
}

void FBXExporter::WriteReferences ()
{
    if (!binary) {
        WriteAsciiSectionHeader("Document References");
    }
    // always empty for now.
    // not really sure what this is for.
    FBX::Node n("References");
    n.force_has_children = true;
    n.Dump(outfile, binary, 0);
}


// ---------------------------------------------------------------
// some internal helper functions used for writing the definitions
// (before any actual data is written)
// ---------------------------------------------------------------

size_t count_nodes(const aiNode* n) {
    size_t count = 1;
    for (size_t i = 0; i < n->mNumChildren; ++i) {
        count += count_nodes(n->mChildren[i]);
    }
    return count;
}

bool has_phong_mat(const aiScene* scene)
{
    // just search for any material with a shininess exponent
    for (size_t i = 0; i < scene->mNumMaterials; ++i) {
        aiMaterial* mat = scene->mMaterials[i];
        float shininess = 0;
        mat->Get(AI_MATKEY_SHININESS, shininess);
        if (shininess > 0) {
            return true;
        }
    }
    return false;
}

size_t count_images(const aiScene* scene) {
    std::unordered_set<std::string> images;
    aiString texpath;
    for (size_t i = 0; i < scene->mNumMaterials; ++i) {
        aiMaterial* mat = scene->mMaterials[i];
        for (
            size_t tt = aiTextureType_DIFFUSE;
            tt < aiTextureType_UNKNOWN;
            ++tt
        ){
            const aiTextureType textype = static_cast<aiTextureType>(tt);
            const size_t texcount = mat->GetTextureCount(textype);
            for (unsigned int j = 0; j < texcount; ++j) {
                mat->GetTexture(textype, j, &texpath);
                images.insert(std::string(texpath.C_Str()));
            }
        }
    }
    return images.size();
}

size_t count_textures(const aiScene* scene) {
    size_t count = 0;
    for (size_t i = 0; i < scene->mNumMaterials; ++i) {
        aiMaterial* mat = scene->mMaterials[i];
        for (
            size_t tt = aiTextureType_DIFFUSE;
            tt < aiTextureType_UNKNOWN;
            ++tt
        ){
            // TODO: handle layered textures
            if (mat->GetTextureCount(static_cast<aiTextureType>(tt)) > 0) {
                count += 1;
            }
        }
    }
    return count;
}

size_t count_deformers(const aiScene* scene) {
    size_t count = 0;
    for (size_t i = 0; i < scene->mNumMeshes; ++i) {
        const size_t n = scene->mMeshes[i]->mNumBones;
        if (n) {
            // 1 main deformer, 1 subdeformer per bone
            count += n + 1;
        }
    }
    return count;
}

void FBXExporter::WriteDefinitions ()
{
    // basically this is just bookkeeping:
    // determining how many of each type of object there are
    // and specifying the base properties to use when otherwise unspecified.

    // ascii section header
    if (!binary) {
        WriteAsciiSectionHeader("Object definitions");
    }

    // we need to count the objects
    int32_t count;
    int32_t total_count = 0;

    // and store them
    std::vector<FBX::Node> object_nodes;
    FBX::Node n, pt, p;

    // GlobalSettings
    // this seems to always be here in Maya exports
    n = FBX::Node("ObjectType", "GlobalSettings");
    count = 1;
    n.AddChild("Count", count);
    object_nodes.push_back(n);
    total_count += count;

    // AnimationStack / FbxAnimStack
    // this seems to always be here in Maya exports,
    // but no harm seems to come of leaving it out.
    count = mScene->mNumAnimations;
    if (count) {
        n = FBX::Node("ObjectType", "AnimationStack");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "FbxAnimStack");
        p = FBX::Node("Properties70");
        p.AddP70string("Description", "");
        p.AddP70time("LocalStart", 0);
        p.AddP70time("LocalStop", 0);
        p.AddP70time("ReferenceStart", 0);
        p.AddP70time("ReferenceStop", 0);
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // AnimationLayer / FbxAnimLayer
    // this seems to always be here in Maya exports,
    // but no harm seems to come of leaving it out.
    // Assimp doesn't support animation layers,
    // so there will be one per aiAnimation
    count = mScene->mNumAnimations;
    if (count) {
        n = FBX::Node("ObjectType", "AnimationLayer");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "FBXAnimLayer");
        p = FBX::Node("Properties70");
        p.AddP70("Weight", "Number", "", "A", double(100));
        p.AddP70bool("Mute", 0);
        p.AddP70bool("Solo", 0);
        p.AddP70bool("Lock", 0);
        p.AddP70color("Color", 0.8, 0.8, 0.8);
        p.AddP70("BlendMode", "enum", "", "", int32_t(0));
        p.AddP70("RotationAccumulationMode", "enum", "", "", int32_t(0));
        p.AddP70("ScaleAccumulationMode", "enum", "", "", int32_t(0));
        p.AddP70("BlendModeBypass", "ULongLong", "", "", int64_t(0));
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // NodeAttribute
    // this is completely absurd.
    // there can only be one "NodeAttribute" template,
    // but FbxSkeleton, FbxCamera, FbxLight all are "NodeAttributes".
    // so if only one exists we should set the template for that,
    // otherwise... we just pick one :/.
    // the others have to set all their properties every instance,
    // because there's no template.
    count = 1; // TODO: select properly
    if (count) {
        // FbxSkeleton
        n = FBX::Node("ObjectType", "NodeAttribute");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "FbxSkeleton");
        p = FBX::Node("Properties70");
        p.AddP70color("Color", 0.8, 0.8, 0.8);
        p.AddP70double("Size", 33.333333333333);
        p.AddP70("LimbLength", "double", "Number", "H", double(1));
        // note: not sure what the "H" flag is for - hidden?
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // Model / FbxNode
    // <~~ node hierarchy
    count = int32_t(count_nodes(mScene->mRootNode)) - 1; // (not counting root node)
    if (count) {
        n = FBX::Node("ObjectType", "Model");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "FbxNode");
        p = FBX::Node("Properties70");
        p.AddP70enum("QuaternionInterpolate", 0);
        p.AddP70vector("RotationOffset", 0.0, 0.0, 0.0);
        p.AddP70vector("RotationPivot", 0.0, 0.0, 0.0);
        p.AddP70vector("ScalingOffset", 0.0, 0.0, 0.0);
        p.AddP70vector("ScalingPivot", 0.0, 0.0, 0.0);
        p.AddP70bool("TranslationActive", 0);
        p.AddP70vector("TranslationMin", 0.0, 0.0, 0.0);
        p.AddP70vector("TranslationMax", 0.0, 0.0, 0.0);
        p.AddP70bool("TranslationMinX", 0);
        p.AddP70bool("TranslationMinY", 0);
        p.AddP70bool("TranslationMinZ", 0);
        p.AddP70bool("TranslationMaxX", 0);
        p.AddP70bool("TranslationMaxY", 0);
        p.AddP70bool("TranslationMaxZ", 0);
        p.AddP70enum("RotationOrder", 0);
        p.AddP70bool("RotationSpaceForLimitOnly", 0);
        p.AddP70double("RotationStiffnessX", 0.0);
        p.AddP70double("RotationStiffnessY", 0.0);
        p.AddP70double("RotationStiffnessZ", 0.0);
        p.AddP70double("AxisLen", 10.0);
        p.AddP70vector("PreRotation", 0.0, 0.0, 0.0);
        p.AddP70vector("PostRotation", 0.0, 0.0, 0.0);
        p.AddP70bool("RotationActive", 0);
        p.AddP70vector("RotationMin", 0.0, 0.0, 0.0);
        p.AddP70vector("RotationMax", 0.0, 0.0, 0.0);
        p.AddP70bool("RotationMinX", 0);
        p.AddP70bool("RotationMinY", 0);
        p.AddP70bool("RotationMinZ", 0);
        p.AddP70bool("RotationMaxX", 0);
        p.AddP70bool("RotationMaxY", 0);
        p.AddP70bool("RotationMaxZ", 0);
        p.AddP70enum("InheritType", 0);
        p.AddP70bool("ScalingActive", 0);
        p.AddP70vector("ScalingMin", 0.0, 0.0, 0.0);
        p.AddP70vector("ScalingMax", 1.0, 1.0, 1.0);
        p.AddP70bool("ScalingMinX", 0);
        p.AddP70bool("ScalingMinY", 0);
        p.AddP70bool("ScalingMinZ", 0);
        p.AddP70bool("ScalingMaxX", 0);
        p.AddP70bool("ScalingMaxY", 0);
        p.AddP70bool("ScalingMaxZ", 0);
        p.AddP70vector("GeometricTranslation", 0.0, 0.0, 0.0);
        p.AddP70vector("GeometricRotation", 0.0, 0.0, 0.0);
        p.AddP70vector("GeometricScaling", 1.0, 1.0, 1.0);
        p.AddP70double("MinDampRangeX", 0.0);
        p.AddP70double("MinDampRangeY", 0.0);
        p.AddP70double("MinDampRangeZ", 0.0);
        p.AddP70double("MaxDampRangeX", 0.0);
        p.AddP70double("MaxDampRangeY", 0.0);
        p.AddP70double("MaxDampRangeZ", 0.0);
        p.AddP70double("MinDampStrengthX", 0.0);
        p.AddP70double("MinDampStrengthY", 0.0);
        p.AddP70double("MinDampStrengthZ", 0.0);
        p.AddP70double("MaxDampStrengthX", 0.0);
        p.AddP70double("MaxDampStrengthY", 0.0);
        p.AddP70double("MaxDampStrengthZ", 0.0);
        p.AddP70double("PreferedAngleX", 0.0);
        p.AddP70double("PreferedAngleY", 0.0);
        p.AddP70double("PreferedAngleZ", 0.0);
        p.AddP70("LookAtProperty", "object", "", "");
        p.AddP70("UpVectorProperty", "object", "", "");
        p.AddP70bool("Show", 1);
        p.AddP70bool("NegativePercentShapeSupport", 1);
        p.AddP70int("DefaultAttributeIndex", -1);
        p.AddP70bool("Freeze", 0);
        p.AddP70bool("LODBox", 0);
        p.AddP70(
            "Lcl Translation", "Lcl Translation", "", "A",
            double(0), double(0), double(0)
        );
        p.AddP70(
            "Lcl Rotation", "Lcl Rotation", "", "A",
            double(0), double(0), double(0)
        );
        p.AddP70(
            "Lcl Scaling", "Lcl Scaling", "", "A",
            double(1), double(1), double(1)
        );
        p.AddP70("Visibility", "Visibility", "", "A", double(1));
        p.AddP70(
            "Visibility Inheritance", "Visibility Inheritance", "", "",
            int32_t(1)
        );
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // Geometry / FbxMesh
    // <~~ aiMesh
    count = mScene->mNumMeshes;
    if (count) {
        n = FBX::Node("ObjectType", "Geometry");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "FbxMesh");
        p = FBX::Node("Properties70");
        p.AddP70color("Color", 0, 0, 0);
        p.AddP70vector("BBoxMin", 0, 0, 0);
        p.AddP70vector("BBoxMax", 0, 0, 0);
        p.AddP70bool("Primary Visibility", 1);
        p.AddP70bool("Casts Shadows", 1);
        p.AddP70bool("Receive Shadows", 1);
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // Material / FbxSurfacePhong, FbxSurfaceLambert, FbxSurfaceMaterial
    // <~~ aiMaterial
    // basically if there's any phong material this is defined as phong,
    // and otherwise lambert.
    // More complex materials cause a bare-bones FbxSurfaceMaterial definition
    // and are treated specially, as they're not really supported by FBX.
    // TODO: support Maya's Stingray PBS material
    count = mScene->mNumMaterials;
    if (count) {
        bool has_phong = has_phong_mat(mScene);
        n = FBX::Node("ObjectType", "Material");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate");
        if (has_phong) {
            pt.AddProperty("FbxSurfacePhong");
        } else {
            pt.AddProperty("FbxSurfaceLambert");
        }
        p = FBX::Node("Properties70");
        if (has_phong) {
            p.AddP70string("ShadingModel", "Phong");
        } else {
            p.AddP70string("ShadingModel", "Lambert");
        }
        p.AddP70bool("MultiLayer", 0);
        p.AddP70colorA("EmissiveColor", 0.0, 0.0, 0.0);
        p.AddP70numberA("EmissiveFactor", 1.0);
        p.AddP70colorA("AmbientColor", 0.2, 0.2, 0.2);
        p.AddP70numberA("AmbientFactor", 1.0);
        p.AddP70colorA("DiffuseColor", 0.8, 0.8, 0.8);
        p.AddP70numberA("DiffuseFactor", 1.0);
        p.AddP70vector("Bump", 0.0, 0.0, 0.0);
        p.AddP70vector("NormalMap", 0.0, 0.0, 0.0);
        p.AddP70double("BumpFactor", 1.0);
        p.AddP70colorA("TransparentColor", 0.0, 0.0, 0.0);
        p.AddP70numberA("TransparencyFactor", 0.0);
        p.AddP70color("DisplacementColor", 0.0, 0.0, 0.0);
        p.AddP70double("DisplacementFactor", 1.0);
        p.AddP70color("VectorDisplacementColor", 0.0, 0.0, 0.0);
        p.AddP70double("VectorDisplacementFactor", 1.0);
        if (has_phong) {
            p.AddP70colorA("SpecularColor", 0.2, 0.2, 0.2);
            p.AddP70numberA("SpecularFactor", 1.0);
            p.AddP70numberA("ShininessExponent", 20.0);
            p.AddP70colorA("ReflectionColor", 0.0, 0.0, 0.0);
            p.AddP70numberA("ReflectionFactor", 1.0);
        }
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // Video / FbxVideo
    // one for each image file.
    count = int32_t(count_images(mScene));
    if (count) {
        n = FBX::Node("ObjectType", "Video");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "FbxVideo");
        p = FBX::Node("Properties70");
        p.AddP70bool("ImageSequence", 0);
        p.AddP70int("ImageSequenceOffset", 0);
        p.AddP70double("FrameRate", 0.0);
        p.AddP70int("LastFrame", 0);
        p.AddP70int("Width", 0);
        p.AddP70int("Height", 0);
        p.AddP70("Path", "KString", "XRefUrl", "", "");
        p.AddP70int("StartFrame", 0);
        p.AddP70int("StopFrame", 0);
        p.AddP70double("PlaySpeed", 0.0);
        p.AddP70time("Offset", 0);
        p.AddP70enum("InterlaceMode", 0);
        p.AddP70bool("FreeRunning", 0);
        p.AddP70bool("Loop", 0);
        p.AddP70enum("AccessMode", 0);
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // Texture / FbxFileTexture
    // <~~ aiTexture
    count = int32_t(count_textures(mScene));
    if (count) {
        n = FBX::Node("ObjectType", "Texture");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "FbxFileTexture");
        p = FBX::Node("Properties70");
        p.AddP70enum("TextureTypeUse", 0);
        p.AddP70numberA("Texture alpha", 1.0);
        p.AddP70enum("CurrentMappingType", 0);
        p.AddP70enum("WrapModeU", 0);
        p.AddP70enum("WrapModeV", 0);
        p.AddP70bool("UVSwap", 0);
        p.AddP70bool("PremultiplyAlpha", 1);
        p.AddP70vectorA("Translation", 0.0, 0.0, 0.0);
        p.AddP70vectorA("Rotation", 0.0, 0.0, 0.0);
        p.AddP70vectorA("Scaling", 1.0, 1.0, 1.0);
        p.AddP70vector("TextureRotationPivot", 0.0, 0.0, 0.0);
        p.AddP70vector("TextureScalingPivot", 0.0, 0.0, 0.0);
        p.AddP70enum("CurrentTextureBlendMode", 1);
        p.AddP70string("UVSet", "default");
        p.AddP70bool("UseMaterial", 0);
        p.AddP70bool("UseMipMap", 0);
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // AnimationCurveNode / FbxAnimCurveNode
    count = mScene->mNumAnimations * 3;
    if (count) {
        n = FBX::Node("ObjectType", "AnimationCurveNode");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "FbxAnimCurveNode");
        p = FBX::Node("Properties70");
        p.AddP70("d", "Compound", "", "");
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // AnimationCurve / FbxAnimCurve
    count = mScene->mNumAnimations * 9;
    if (count) {
        n = FBX::Node("ObjectType", "AnimationCurve");
        n.AddChild("Count", count);
        object_nodes.push_back(n);
        total_count += count;
    }

    // Pose
    count = 0;
    for (size_t i = 0; i < mScene->mNumMeshes; ++i) {
        aiMesh* mesh = mScene->mMeshes[i];
        if (mesh->HasBones()) { ++count; }
    }
    if (count) {
        n = FBX::Node("ObjectType", "Pose");
        n.AddChild("Count", count);
        object_nodes.push_back(n);
        total_count += count;
    }

    // Deformer
    count = int32_t(count_deformers(mScene));
    if (count) {
        n = FBX::Node("ObjectType", "Deformer");
        n.AddChild("Count", count);
        object_nodes.push_back(n);
        total_count += count;
    }

    // (template)
    count = 0;
    if (count) {
        n = FBX::Node("ObjectType", "");
        n.AddChild("Count", count);
        pt = FBX::Node("PropertyTemplate", "");
        p = FBX::Node("Properties70");
        pt.AddChild(p);
        n.AddChild(pt);
        object_nodes.push_back(n);
        total_count += count;
    }

    // now write it all
    FBX::Node defs("Definitions");
    defs.AddChild("Version", int32_t(100));
    defs.AddChild("Count", int32_t(total_count));
    for (auto &n : object_nodes) { defs.AddChild(n); }
    defs.Dump(outfile, binary, 0);
}


// -------------------------------------------------------------------
// some internal helper functions used for writing the objects section
// (which holds the actual data)
// -------------------------------------------------------------------

aiNode* get_node_for_mesh(unsigned int meshIndex, aiNode* node)
{
    for (size_t i = 0; i < node->mNumMeshes; ++i) {
        if (node->mMeshes[i] == meshIndex) {
            return node;
        }
    }
    for (size_t i = 0; i < node->mNumChildren; ++i) {
        aiNode* ret = get_node_for_mesh(meshIndex, node->mChildren[i]);
        if (ret) { return ret; }
    }
    return nullptr;
}

aiMatrix4x4 get_world_transform(const aiNode* node, const aiScene* scene)
{
    std::vector<const aiNode*> node_chain;
    while (node != scene->mRootNode) {
        node_chain.push_back(node);
        node = node->mParent;
    }
    aiMatrix4x4 transform;
    for (auto n = node_chain.rbegin(); n != node_chain.rend(); ++n) {
        transform *= (*n)->mTransformation;
    }
    return transform;
}

int64_t to_ktime(double ticks, const aiAnimation* anim) {
    if (anim->mTicksPerSecond <= 0) {
        return static_cast<int64_t>(ticks) * FBX::SECOND;
    }
    return (static_cast<int64_t>(ticks) / static_cast<int64_t>(anim->mTicksPerSecond)) * FBX::SECOND;
}

int64_t to_ktime(double time) {
    return (static_cast<int64_t>(time * FBX::SECOND));
}

void FBXExporter::WriteObjects ()
{
    if (!binary) {
        WriteAsciiSectionHeader("Object properties");
    }
    // numbers should match those given in definitions! make sure to check
    StreamWriterLE outstream(outfile);
    FBX::Node object_node("Objects");
    int indent = 0;
    object_node.Begin(outstream, binary, indent);
    object_node.EndProperties(outstream, binary, indent);
    object_node.BeginChildren(outstream, binary, indent);

    // geometry (aiMesh)
    mesh_uids.clear();
    indent = 1;
    for (size_t mi = 0; mi < mScene->mNumMeshes; ++mi) {
        // it's all about this mesh
        aiMesh* m = mScene->mMeshes[mi];

        // start the node record
        FBX::Node n("Geometry");
        int64_t uid = generate_uid();
        mesh_uids.push_back(uid);
        n.AddProperty(uid);
        n.AddProperty(FBX::SEPARATOR + "Geometry");
        n.AddProperty("Mesh");
        n.Begin(outstream, binary, indent);
        n.DumpProperties(outstream, binary, indent);
        n.EndProperties(outstream, binary, indent);
        n.BeginChildren(outstream, binary, indent);
        indent = 2;

        // output vertex data - each vertex should be unique (probably)
        std::vector<double> flattened_vertices;
        // index of original vertex in vertex data vector
        std::vector<int32_t> vertex_indices;
        // map of vertex value to its index in the data vector
        std::map<aiVector3D,size_t> index_by_vertex_value;
        int32_t index = 0;
        for (size_t vi = 0; vi < m->mNumVertices; ++vi) {
            aiVector3D vtx = m->mVertices[vi];
            auto elem = index_by_vertex_value.find(vtx);
            if (elem == index_by_vertex_value.end()) {
                vertex_indices.push_back(index);
                index_by_vertex_value[vtx] = index;
                flattened_vertices.push_back(vtx[0]);
                flattened_vertices.push_back(vtx[1]);
                flattened_vertices.push_back(vtx[2]);
                ++index;
            } else {
                vertex_indices.push_back(int32_t(elem->second));
            }
        }
        FBX::Node::WritePropertyNode(
            "Vertices", flattened_vertices, outstream, binary, indent
        );

        // output polygon data as a flattened array of vertex indices.
        // the last vertex index of each polygon is negated and - 1
        std::vector<int32_t> polygon_data;
        for (size_t fi = 0; fi < m->mNumFaces; ++fi) {
            const aiFace &f = m->mFaces[fi];
            for (size_t pvi = 0; pvi < f.mNumIndices - 1; ++pvi) {
                polygon_data.push_back(vertex_indices[f.mIndices[pvi]]);
            }
            polygon_data.push_back(
                -1 - vertex_indices[f.mIndices[f.mNumIndices-1]]
            );
        }
        FBX::Node::WritePropertyNode(
            "PolygonVertexIndex", polygon_data, outstream, binary, indent
        );

        // here could be edges but they're insane.
        // it's optional anyway, so let's ignore it.

        FBX::Node::WritePropertyNode(
            "GeometryVersion", int32_t(124), outstream, binary, indent
        );

        // normals, if any
        if (m->HasNormals()) {
            FBX::Node normals("LayerElementNormal", int32_t(0));
            normals.Begin(outstream, binary, indent);
            normals.DumpProperties(outstream, binary, indent);
            normals.EndProperties(outstream, binary, indent);
            normals.BeginChildren(outstream, binary, indent);
            indent = 3;
            FBX::Node::WritePropertyNode(
                "Version", int32_t(101), outstream, binary, indent
            );
            FBX::Node::WritePropertyNode(
                "Name", "", outstream, binary, indent
            );
            FBX::Node::WritePropertyNode(
                "MappingInformationType", "ByPolygonVertex",
                outstream, binary, indent
            );
            // TODO: vertex-normals or indexed normals when appropriate
            FBX::Node::WritePropertyNode(
                "ReferenceInformationType", "Direct",
                outstream, binary, indent
            );
            std::vector<double> normal_data;
            normal_data.reserve(3 * polygon_data.size());
            for (size_t fi = 0; fi < m->mNumFaces; ++fi) {
                const aiFace &f = m->mFaces[fi];
                for (size_t pvi = 0; pvi < f.mNumIndices; ++pvi) {
                    const aiVector3D &n = m->mNormals[f.mIndices[pvi]];
                    normal_data.push_back(n.x);
                    normal_data.push_back(n.y);
                    normal_data.push_back(n.z);
                }
            }
            FBX::Node::WritePropertyNode(
                "Normals", normal_data, outstream, binary, indent
            );
            // note: version 102 has a NormalsW also... not sure what it is,
            // so we can stick with version 101 for now.
            indent = 2;
            normals.End(outstream, binary, indent, true);
        }

        // uvs, if any
        for (size_t uvi = 0; uvi < m->GetNumUVChannels(); ++uvi) {
            if (m->mNumUVComponents[uvi] > 2) {
                // FBX only supports 2-channel UV maps...
                // or at least i'm not sure how to indicate a different number
                std::stringstream err;
                err << "Only 2-channel UV maps supported by FBX,";
                err << " but mesh " << mi;
                if (m->mName.length) {
                    err << " (" << m->mName.C_Str() << ")";
                }
                err << " UV map " << uvi;
                err << " has " << m->mNumUVComponents[uvi];
                err << " components! Data will be preserved,";
                err << " but may be incorrectly interpreted on load.";
                ASSIMP_LOG_WARN(err.str());
            }
            FBX::Node uv("LayerElementUV", int32_t(uvi));
            uv.Begin(outstream, binary, indent);
            uv.DumpProperties(outstream, binary, indent);
            uv.EndProperties(outstream, binary, indent);
            uv.BeginChildren(outstream, binary, indent);
            indent = 3;
            FBX::Node::WritePropertyNode(
                "Version", int32_t(101), outstream, binary, indent
            );
            // it doesn't seem like assimp keeps the uv map name,
            // so just leave it blank.
            FBX::Node::WritePropertyNode(
                "Name", "", outstream, binary, indent
            );
            FBX::Node::WritePropertyNode(
                "MappingInformationType", "ByPolygonVertex",
                outstream, binary, indent
            );
            FBX::Node::WritePropertyNode(
                "ReferenceInformationType", "IndexToDirect",
                outstream, binary, indent
            );

            std::vector<double> uv_data;
            std::vector<int32_t> uv_indices;
            std::map<aiVector3D,int32_t> index_by_uv;
            int32_t index = 0;
            for (size_t fi = 0; fi < m->mNumFaces; ++fi) {
                const aiFace &f = m->mFaces[fi];
                for (size_t pvi = 0; pvi < f.mNumIndices; ++pvi) {
                    const aiVector3D &uv =
                        m->mTextureCoords[uvi][f.mIndices[pvi]];
                    auto elem = index_by_uv.find(uv);
                    if (elem == index_by_uv.end()) {
                        index_by_uv[uv] = index;
                        uv_indices.push_back(index);
                        for (unsigned int x = 0; x < m->mNumUVComponents[uvi]; ++x) {
                            uv_data.push_back(uv[x]);
                        }
                        ++index;
                    } else {
                        uv_indices.push_back(elem->second);
                    }
                }
            }
            FBX::Node::WritePropertyNode(
                "UV", uv_data, outstream, binary, indent
            );
            FBX::Node::WritePropertyNode(
                "UVIndex", uv_indices, outstream, binary, indent
            );
            indent = 2;
            uv.End(outstream, binary, indent, true);
        }

        // i'm not really sure why this material section exists,
        // as the material is linked via "Connections".
        // it seems to always have the same "0" value.
        FBX::Node mat("LayerElementMaterial", int32_t(0));
        mat.AddChild("Version", int32_t(101));
        mat.AddChild("Name", "");
        mat.AddChild("MappingInformationType", "AllSame");
        mat.AddChild("ReferenceInformationType", "IndexToDirect");
        std::vector<int32_t> mat_indices = {0};
        mat.AddChild("Materials", mat_indices);
        mat.Dump(outstream, binary, indent);

        // finally we have the layer specifications,
        // which select the normals / UV set / etc to use.
        // TODO: handle multiple uv sets correctly?
        FBX::Node layer("Layer", int32_t(0));
        layer.AddChild("Version", int32_t(100));
        FBX::Node le("LayerElement");
        le.AddChild("Type", "LayerElementNormal");
        le.AddChild("TypedIndex", int32_t(0));
        layer.AddChild(le);
        le = FBX::Node("LayerElement");
        le.AddChild("Type", "LayerElementMaterial");
        le.AddChild("TypedIndex", int32_t(0));
        layer.AddChild(le);
        le = FBX::Node("LayerElement");
        le.AddChild("Type", "LayerElementUV");
        le.AddChild("TypedIndex", int32_t(0));
        layer.AddChild(le);
        layer.Dump(outstream, binary, indent);

        // finish the node record
        indent = 1;
        n.End(outstream, binary, indent, true);
    }

    // aiMaterial
    material_uids.clear();
    for (size_t i = 0; i < mScene->mNumMaterials; ++i) {
        // it's all about this material
        aiMaterial* m = mScene->mMaterials[i];

        // these are used to receive material data
        float f; aiColor3D c;

        // start the node record
        FBX::Node n("Material");

        int64_t uid = generate_uid();
        material_uids.push_back(uid);
        n.AddProperty(uid);

        aiString name;
        m->Get(AI_MATKEY_NAME, name);
        n.AddProperty(name.C_Str() + FBX::SEPARATOR + "Material");

        n.AddProperty("");

        n.AddChild("Version", int32_t(102));
        f = 0;
        m->Get(AI_MATKEY_SHININESS, f);
        bool phong = (f > 0);
        if (phong) {
            n.AddChild("ShadingModel", "phong");
        } else {
            n.AddChild("ShadingModel", "lambert");
        }
        n.AddChild("MultiLayer", int32_t(0));

        FBX::Node p("Properties70");

        // materials exported using the FBX SDK have two sets of fields.
        // there are the properties specified in the PropertyTemplate,
        // which are those supported by the modernFBX SDK,
        // and an extra set of properties with simpler names.
        // The extra properties are a legacy material system from pre-2009.
        //
        // In the modern system, each property has "color" and "factor".
        // Generally the interpretation of these seems to be
        // that the colour is multiplied by the factor before use,
        // but this is not always clear-cut.
        //
        // Usually assimp only stores the colour,
        // so we can just leave the factors at the default "1.0".

        // first we can export the "standard" properties
        if (m->Get(AI_MATKEY_COLOR_AMBIENT, c) == aiReturn_SUCCESS) {
            p.AddP70colorA("AmbientColor", c.r, c.g, c.b);
            //p.AddP70numberA("AmbientFactor", 1.0);
        }
        if (m->Get(AI_MATKEY_COLOR_DIFFUSE, c) == aiReturn_SUCCESS) {
            p.AddP70colorA("DiffuseColor", c.r, c.g, c.b);
            //p.AddP70numberA("DiffuseFactor", 1.0);
        }
        if (m->Get(AI_MATKEY_COLOR_TRANSPARENT, c) == aiReturn_SUCCESS) {
            // "TransparentColor" / "TransparencyFactor"...
            // thanks FBX, for your insightful interpretation of consistency
            p.AddP70colorA("TransparentColor", c.r, c.g, c.b);
            // TransparencyFactor defaults to 0.0, so set it to 1.0.
            // note: Maya always sets this to 1.0,
            // so we can't use it sensibly as "Opacity".
            // In stead we rely on the legacy "Opacity" value, below.
            // Blender also relies on "Opacity" not "TransparencyFactor",
            // probably for a similar reason.
            p.AddP70numberA("TransparencyFactor", 1.0);
        }
        if (m->Get(AI_MATKEY_COLOR_REFLECTIVE, c) == aiReturn_SUCCESS) {
            p.AddP70colorA("ReflectionColor", c.r, c.g, c.b);
        }
        if (m->Get(AI_MATKEY_REFLECTIVITY, f) == aiReturn_SUCCESS) {
            p.AddP70numberA("ReflectionFactor", f);
        }
        if (phong) {
            if (m->Get(AI_MATKEY_COLOR_SPECULAR, c) == aiReturn_SUCCESS) {
                p.AddP70colorA("SpecularColor", c.r, c.g, c.b);
            }
            if (m->Get(AI_MATKEY_SHININESS_STRENGTH, f) == aiReturn_SUCCESS) {
                p.AddP70numberA("ShininessFactor", f);
            }
            if (m->Get(AI_MATKEY_SHININESS, f) == aiReturn_SUCCESS) {
                p.AddP70numberA("ShininessExponent", f);
            }
            if (m->Get(AI_MATKEY_REFLECTIVITY, f) == aiReturn_SUCCESS) {
                p.AddP70numberA("ReflectionFactor", f);
            }
        }

        // Now the legacy system.
        // For safety let's include it.
        // thrse values don't exist in the property template,
        // and usually are completely ignored when loading.
        // One notable exception is the "Opacity" property,
        // which Blender uses as (1.0 - alpha).
        c.r = 0.0f; c.g = 0.0f; c.b = 0.0f;
        m->Get(AI_MATKEY_COLOR_EMISSIVE, c);
        p.AddP70vector("Emissive", c.r, c.g, c.b);
        c.r = 0.2f; c.g = 0.2f; c.b = 0.2f;
        m->Get(AI_MATKEY_COLOR_AMBIENT, c);
        p.AddP70vector("Ambient", c.r, c.g, c.b);
        c.r = 0.8f; c.g = 0.8f; c.b = 0.8f;
        m->Get(AI_MATKEY_COLOR_DIFFUSE, c);
        p.AddP70vector("Diffuse", c.r, c.g, c.b);
        // The FBX SDK determines "Opacity" from transparency colour (RGB)
        // and factor (F) as: O = (1.0 - F * ((R + G + B) / 3)).
        // However we actually have an opacity value,
        // so we should take it from AI_MATKEY_OPACITY if possible.
        // It might make more sense to use TransparencyFactor,
        // but Blender actually loads "Opacity" correctly, so let's use it.
        f = 1.0f;
        if (m->Get(AI_MATKEY_COLOR_TRANSPARENT, c) == aiReturn_SUCCESS) {
            f = 1.0f - ((c.r + c.g + c.b) / 3.0f);
        }
        m->Get(AI_MATKEY_OPACITY, f);
        p.AddP70double("Opacity", f);
        if (phong) {
            // specular color is multiplied by shininess_strength
            c.r = 0.2f; c.g = 0.2f; c.b = 0.2f;
            m->Get(AI_MATKEY_COLOR_SPECULAR, c);
            f = 1.0f;
            m->Get(AI_MATKEY_SHININESS_STRENGTH, f);
            p.AddP70vector("Specular", f*c.r, f*c.g, f*c.b);
            f = 20.0f;
            m->Get(AI_MATKEY_SHININESS, f);
            p.AddP70double("Shininess", f);
            // Legacy "Reflectivity" is F*F*((R+G+B)/3),
            // where F is the proportion of light reflected (AKA reflectivity),
            // and RGB is the reflective colour of the material.
            // No idea why, but we might as well set it the same way.
            f = 0.0f;
            m->Get(AI_MATKEY_REFLECTIVITY, f);
            c.r = 1.0f, c.g = 1.0f, c.b = 1.0f;
            m->Get(AI_MATKEY_COLOR_REFLECTIVE, c);
            p.AddP70double("Reflectivity", f*f*((c.r+c.g+c.b)/3.0));
        }

        n.AddChild(p);

        n.Dump(outstream, binary, indent);
    }

    // we need to look up all the images we're using,
    // so we can generate uids, and eliminate duplicates.
    std::map<std::string, int64_t> uid_by_image;
    for (size_t i = 0; i < mScene->mNumMaterials; ++i) {
        aiString texpath;
        aiMaterial* mat = mScene->mMaterials[i];
        for (
            size_t tt = aiTextureType_DIFFUSE;
            tt < aiTextureType_UNKNOWN;
            ++tt
        ){
            const aiTextureType textype = static_cast<aiTextureType>(tt);
            const size_t texcount = mat->GetTextureCount(textype);
            for (size_t j = 0; j < texcount; ++j) {
                mat->GetTexture(textype, (unsigned int)j, &texpath);
                const std::string texstring = texpath.C_Str();
                auto elem = uid_by_image.find(texstring);
                if (elem == uid_by_image.end()) {
                    uid_by_image[texstring] = generate_uid();
                }
            }
        }
    }

    // FbxVideo - stores images used by textures.
    for (const auto &it : uid_by_image) {
        if (it.first.compare(0, 1, "*") == 0) {
            // TODO: embedded textures
            continue;
        }
        FBX::Node n("Video");
        const int64_t& uid = it.second;
        const std::string name = ""; // TODO: ... name???
        n.AddProperties(uid, name + FBX::SEPARATOR + "Video", "Clip");
        n.AddChild("Type", "Clip");
        FBX::Node p("Properties70");
        // TODO: get full path... relative path... etc... ugh...
        // for now just use the same path for everything,
        // and hopefully one of them will work out.
        const std::string& path = it.first;
        p.AddP70("Path", "KString", "XRefUrl", "", path);
        n.AddChild(p);
        n.AddChild("UseMipMap", int32_t(0));
        n.AddChild("Filename", path);
        n.AddChild("RelativeFilename", path);
        n.Dump(outstream, binary, indent);
    }

    // Textures
    // referenced by material_index/texture_type pairs.
    std::map<std::pair<size_t,size_t>,int64_t> texture_uids;
    const std::map<aiTextureType,std::string> prop_name_by_tt = {
        {aiTextureType_DIFFUSE, "DiffuseColor"},
        {aiTextureType_SPECULAR, "SpecularColor"},
        {aiTextureType_AMBIENT, "AmbientColor"},
        {aiTextureType_EMISSIVE, "EmissiveColor"},
        {aiTextureType_HEIGHT, "Bump"},
        {aiTextureType_NORMALS, "NormalMap"},
        {aiTextureType_SHININESS, "ShininessExponent"},
        {aiTextureType_OPACITY, "TransparentColor"},
        {aiTextureType_DISPLACEMENT, "DisplacementColor"},
        //{aiTextureType_LIGHTMAP, "???"},
        {aiTextureType_REFLECTION, "ReflectionColor"}
        //{aiTextureType_UNKNOWN, ""}
    };
    for (size_t i = 0; i < mScene->mNumMaterials; ++i) {
        // textures are attached to materials
        aiMaterial* mat = mScene->mMaterials[i];
        int64_t material_uid = material_uids[i];

        for (
            size_t j = aiTextureType_DIFFUSE;
            j < aiTextureType_UNKNOWN;
            ++j
        ) {
            const aiTextureType tt = static_cast<aiTextureType>(j);
            size_t n = mat->GetTextureCount(tt);

            if (n < 1) { // no texture of this type
                continue;
            }

            if (n > 1) {
                // TODO: multilayer textures
                std::stringstream err;
                err << "Multilayer textures not supported (for now),";
                err << " skipping texture type " << j;
                err << " of material " << i;
                ASSIMP_LOG_WARN(err.str());
            }

            // get image path for this (single-image) texture
            aiString tpath;
            if (mat->GetTexture(tt, 0, &tpath) != aiReturn_SUCCESS) {
                std::stringstream err;
                err << "Failed to get texture 0 for texture of type " << tt;
                err << " on material " << i;
                err << ", however GetTextureCount returned 1.";
                throw DeadlyExportError(err.str());
            }
            const std::string texture_path(tpath.C_Str());

            // get connected image uid
            auto elem = uid_by_image.find(texture_path);
            if (elem == uid_by_image.end()) {
                // this should never happen
                std::stringstream err;
                err << "Failed to find video element for texture with path";
                err << " \"" << texture_path << "\"";
                err << ", type " << j << ", material " << i;
                throw DeadlyExportError(err.str());
            }
            const int64_t image_uid = elem->second;

            // get the name of the material property to connect to
            auto elem2 = prop_name_by_tt.find(tt);
            if (elem2 == prop_name_by_tt.end()) {
                // don't know how to handle this type of texture,
                // so skip it.
                std::stringstream err;
                err << "Not sure how to handle texture of type " << j;
                err << " on material " << i;
                err << ", skipping...";
                ASSIMP_LOG_WARN(err.str());
                continue;
            }
            const std::string& prop_name = elem2->second;

            // generate a uid for this texture
            const int64_t texture_uid = generate_uid();

            // link the texture to the material
            connections.emplace_back(
                "C", "OP", texture_uid, material_uid, prop_name
            );

            // link the image data to the texture
            connections.emplace_back("C", "OO", image_uid, texture_uid);

            // now write the actual texture node
            FBX::Node tnode("Texture");
            // TODO: some way to determine texture name?
            const std::string texture_name = "" + FBX::SEPARATOR + "Texture";
            tnode.AddProperties(texture_uid, texture_name, "");
            // there really doesn't seem to be a better type than this:
            tnode.AddChild("Type", "TextureVideoClip");
            tnode.AddChild("Version", int32_t(202));
            tnode.AddChild("TextureName", texture_name);
            FBX::Node p("Properties70");
            p.AddP70enum("CurrentTextureBlendMode", 0); // TODO: verify
            //p.AddP70string("UVSet", ""); // TODO: how should this work?
            p.AddP70bool("UseMaterial", 1);
            tnode.AddChild(p);
            // can't easily detrmine which texture path will be correct,
            // so just store what we have in every field.
            // these being incorrect is a common problem with FBX anyway.
            tnode.AddChild("FileName", texture_path);
            tnode.AddChild("RelativeFilename", texture_path);
            tnode.AddChild("ModelUVTranslation", double(0.0), double(0.0));
            tnode.AddChild("ModelUVScaling", double(1.0), double(1.0));
            tnode.AddChild("Texture_Alpha_Source", "None");
            tnode.AddChild(
                "Cropping", int32_t(0), int32_t(0), int32_t(0), int32_t(0)
            );
            tnode.Dump(outstream, binary, indent);
        }
    }

    // bones.
    //
    // output structure:
    // subset of node hierarchy that are "skeleton",
    // i.e. do not have meshes but only bones.
    // but.. i'm not sure how anyone could guarantee that...
    //
    // input...
    // well, for each mesh it has "bones",
    // and the bone names correspond to nodes.
    // of course we also need the parent nodes,
    // as they give some of the transform........
    //
    // well. we can assume a sane input, i suppose.
    //
    // so input is the bone node hierarchy,
    // with an extra thing for the transformation of the MESH in BONE space.
    //
    // output is a set of bone nodes,
    // a "bindpose" which indicates the default local transform of all bones,
    // and a set of "deformers".
    // each deformer is parented to a mesh geometry,
    // and has one or more "subdeformer"s as children.
    // each subdeformer has one bone node as a child,
    // and represents the influence of that bone on the grandparent mesh.
    // the subdeformer has a list of indices, and weights,
    // with indices specifying vertex indices,
    // and weights specifying the corresponding influence of this bone.
    // it also has Transform and TransformLink elements,
    // specifying the transform of the MESH in BONE space,
    // and the transformation of the BONE in WORLD space,
    // likely in the bindpose.
    //
    // the input bone structure is different but similar,
    // storing the number of weights for this bone,
    // and an array of (vertex index, weight) pairs.
    //
    // one sticky point is that the number of vertices may not match,
    // because assimp splits vertices by normal, uv, etc.

    // first we should mark the skeleton for each mesh.
    // the skeleton must include not only the aiBones,
    // but also all their parent nodes.
    // anything that affects the position of any bone node must be included.
    std::vector<std::set<const aiNode*>> skeleton_by_mesh(mScene->mNumMeshes);
    // at the same time we can build a list of all the skeleton nodes,
    // which will be used later to mark them as type "limbNode".
    std::unordered_set<const aiNode*> limbnodes;
    // and a map of nodes by bone name, as finding them is annoying.
    std::map<std::string,aiNode*> node_by_bone;
    for (size_t mi = 0; mi < mScene->mNumMeshes; ++mi) {
        const aiMesh* m = mScene->mMeshes[mi];
        std::set<const aiNode*> skeleton;
        for (size_t bi =0; bi < m->mNumBones; ++bi) {
            const aiBone* b = m->mBones[bi];
            const std::string name(b->mName.C_Str());
            auto elem = node_by_bone.find(name);
            aiNode* n;
            if (elem != node_by_bone.end()) {
                n = elem->second;
            } else {
                n = mScene->mRootNode->FindNode(b->mName);
                if (!n) {
                    // this should never happen
                    std::stringstream err;
                    err << "Failed to find node for bone: \"" << name << "\"";
                    throw DeadlyExportError(err.str());
                }
                node_by_bone[name] = n;
                limbnodes.insert(n);
            }
            skeleton.insert(n);
            // mark all parent nodes as skeleton as well,
            // up until we find the root node,
            // or else the node containing the mesh,
            // or else the parent of a node containig the mesh.
            for (
                const aiNode* parent = n->mParent;
                parent && parent != mScene->mRootNode;
                parent = parent->mParent
            ) {
                // if we've already done this node we can skip it all
                if (skeleton.count(parent)) {
                    break;
                }
                // ignore fbx transform nodes as these will be collapsed later
                // TODO: cache this by aiNode*
                const std::string node_name(parent->mName.C_Str());
                if (node_name.find(MAGIC_NODE_TAG) != std::string::npos) {
                    continue;
                }
                // otherwise check if this is the root of the skeleton
                bool end = false;
                // is the mesh part of this node?
                for (size_t i = 0; i < parent->mNumMeshes; ++i) {
                    if (parent->mMeshes[i] == mi) {
                        end = true;
                        break;
                    }
                }
                // is the mesh in one of the children of this node?
                for (size_t j = 0; j < parent->mNumChildren; ++j) {
                    aiNode* child = parent->mChildren[j];
                    for (size_t i = 0; i < child->mNumMeshes; ++i) {
                        if (child->mMeshes[i] == mi) {
                            end = true;
                            break;
                        }
                    }
                    if (end) { break; }
                }
                limbnodes.insert(parent);
                skeleton.insert(parent);
                // if it was the skeleton root we can finish here
                if (end) { break; }
            }
        }
        skeleton_by_mesh[mi] = skeleton;
    }

    // we'll need the uids for the bone nodes, so generate them now
    for (size_t i = 0; i < mScene->mNumMeshes; ++i) {
        auto &s = skeleton_by_mesh[i];
        for (const aiNode* n : s) {
            auto elem = node_uids.find(n);
            if (elem == node_uids.end()) {
                node_uids[n] = generate_uid();
            }
        }
    }

    // now, for each aiMesh, we need to export a deformer,
    // and for each aiBone a subdeformer,
    // which should have all the skinning info.
    // these will need to be connected properly to the mesh,
    // and we can do that all now.
    for (size_t mi = 0; mi < mScene->mNumMeshes; ++mi) {
        const aiMesh* m = mScene->mMeshes[mi];
        if (!m->HasBones()) {
            continue;
        }
        // make a deformer for this mesh
        int64_t deformer_uid = generate_uid();
        FBX::Node dnode("Deformer");
        dnode.AddProperties(deformer_uid, FBX::SEPARATOR + "Deformer", "Skin");
        dnode.AddChild("Version", int32_t(101));
        // "acuracy"... this is not a typo....
        dnode.AddChild("Link_DeformAcuracy", double(50));
        dnode.AddChild("SkinningType", "Linear"); // TODO: other modes?
        dnode.Dump(outstream, binary, indent);

        // connect it
        connections.emplace_back("C", "OO", deformer_uid, mesh_uids[mi]);

        // we will be indexing by vertex...
        // but there might be a different number of "vertices"
        // between assimp and our output FBX.
        // this code is cut-and-pasted from the geometry section above...
        // ideally this should not be so.
        // ---
        // index of original vertex in vertex data vector
        std::vector<int32_t> vertex_indices;
        // map of vertex value to its index in the data vector
        std::map<aiVector3D,size_t> index_by_vertex_value;
        int32_t index = 0;
        for (size_t vi = 0; vi < m->mNumVertices; ++vi) {
            aiVector3D vtx = m->mVertices[vi];
            auto elem = index_by_vertex_value.find(vtx);
            if (elem == index_by_vertex_value.end()) {
                vertex_indices.push_back(index);
                index_by_vertex_value[vtx] = index;
                ++index;
            } else {
                vertex_indices.push_back(int32_t(elem->second));
            }
        }

        // TODO, FIXME: this won't work if anything is not in the bind pose.
        // for now if such a situation is detected, we throw an exception.
        std::set<const aiBone*> not_in_bind_pose;
        std::set<const aiNode*> no_offset_matrix;

        // first get this mesh's position in world space,
        // as we'll need it for each subdeformer.
        //
        // ...of course taking the position of the MESH doesn't make sense,
        // as it can be instanced to many nodes.
        // All we can do is assume no instancing,
        // and take the first node we find that contains the mesh.
        aiNode* mesh_node = get_node_for_mesh((unsigned int)mi, mScene->mRootNode);
        aiMatrix4x4 mesh_xform = get_world_transform(mesh_node, mScene);

        // now make a subdeformer for each bone in the skeleton
        const std::set<const aiNode*> &skeleton = skeleton_by_mesh[mi];
        for (const aiNode* bone_node : skeleton) {
            // if there's a bone for this node, find it
            const aiBone* b = nullptr;
            for (size_t bi = 0; bi < m->mNumBones; ++bi) {
                // TODO: this probably should index by something else
                const std::string name(m->mBones[bi]->mName.C_Str());
                if (node_by_bone[name] == bone_node) {
                    b = m->mBones[bi];
                    break;
                }
            }
            if (!b) {
                no_offset_matrix.insert(bone_node);
            }

            // start the subdeformer node
            const int64_t subdeformer_uid = generate_uid();
            FBX::Node sdnode("Deformer");
            sdnode.AddProperties(
                subdeformer_uid, FBX::SEPARATOR + "SubDeformer", "Cluster"
            );
            sdnode.AddChild("Version", int32_t(100));
            sdnode.AddChild("UserData", "", "");

            // add indices and weights, if any
            if (b) {
                std::vector<int32_t> subdef_indices;
                std::vector<double> subdef_weights;
                int32_t last_index = -1;
                for (size_t wi = 0; wi < b->mNumWeights; ++wi) {
                    int32_t vi = vertex_indices[b->mWeights[wi].mVertexId];
                    if (vi == last_index) {
                        // only for vertices we exported to fbx
                        // TODO, FIXME: this assumes identically-located vertices
                        // will always deform in the same way.
                        // as assimp doesn't store a separate list of "positions",
                        // there's not much that can be done about this
                        // other than assuming that identical position means
                        // identical vertex.
                        continue;
                    }
                    subdef_indices.push_back(vi);
                    subdef_weights.push_back(b->mWeights[wi].mWeight);
                    last_index = vi;
                }
                // yes, "indexes"
                sdnode.AddChild("Indexes", subdef_indices);
                sdnode.AddChild("Weights", subdef_weights);
            }

            // transform is the transform of the mesh, but in bone space.
            // if the skeleton is in the bind pose,
            // we can take the inverse of the world-space bone transform
            // and multiply by the world-space transform of the mesh.
            aiMatrix4x4 bone_xform = get_world_transform(bone_node, mScene);
            aiMatrix4x4 inverse_bone_xform = bone_xform;
            inverse_bone_xform.Inverse();
            aiMatrix4x4 tr = inverse_bone_xform * mesh_xform;

            // this should be the same as the bone's mOffsetMatrix.
            // if it's not the same, the skeleton isn't in the bind pose.
            const float epsilon = 1e-4f; // some error is to be expected
            bool bone_xform_okay = true;
            if (b && ! tr.Equal(b->mOffsetMatrix, epsilon)) {
                not_in_bind_pose.insert(b);
                bone_xform_okay = false;
            }

            // if we have a bone we should use the mOffsetMatrix,
            // otherwise try to just use the calculated transform.
            if (b) {
                sdnode.AddChild("Transform", b->mOffsetMatrix);
            } else {
                sdnode.AddChild("Transform", tr);
            }
            // note: it doesn't matter if we mix these,
            // because if they disagree we'll throw an exception later.
            // it could be that the skeleton is not in the bone pose
            // but all bones are still defined,
            // in which case this would use the mOffsetMatrix for everything
            // and a correct skeleton would still be output.

            // transformlink should be the position of the bone in world space.
            // if the bone is in the bind pose (or nonexistent),
            // we can just use the matrix we already calculated
            if (bone_xform_okay) {
                sdnode.AddChild("TransformLink", bone_xform);
            // otherwise we can only work it out using the mesh position.
            } else {
                aiMatrix4x4 trl = b->mOffsetMatrix;
                trl.Inverse();
                trl *= mesh_xform;
                sdnode.AddChild("TransformLink", trl);
            }
            // note: this means we ALWAYS rely on the mesh node transform
            // being unchanged from the time the skeleton was bound.
            // there's not really any way around this at the moment.

            // done
            sdnode.Dump(outstream, binary, indent);

            // lastly, connect to the parent deformer
            connections.emplace_back(
                "C", "OO", subdeformer_uid, deformer_uid
            );

            // we also need to connect the limb node to the subdeformer.
            connections.emplace_back(
                "C", "OO", node_uids[bone_node], subdeformer_uid
            );
        }

        // if we cannot create a valid FBX file, simply die.
        // this will both prevent unnecessary bug reports,
        // and tell the user what they can do to fix the situation
        // (i.e. export their model in the bind pose).
        if (no_offset_matrix.size() && not_in_bind_pose.size()) {
            std::stringstream err;
            err << "Not enough information to construct bind pose";
            err << " for mesh " << mi << "!";
            err << " Transform matrix for bone \"";
            err << (*not_in_bind_pose.begin())->mName.C_Str() << "\"";
            if (not_in_bind_pose.size() > 1) {
                err << " (and " << not_in_bind_pose.size() - 1 << " more)";
            }
            err << " does not match mOffsetMatrix,";
            err << " and node \"";
            err << (*no_offset_matrix.begin())->mName.C_Str() << "\"";
            if (no_offset_matrix.size() > 1) {
                err << " (and " << no_offset_matrix.size() - 1 << " more)";
            }
            err << " has no offset matrix to rely on.";
            err << " Please ensure bones are in the bind pose to export.";
            throw DeadlyExportError(err.str());
        }

    }

    // BindPose
    //
    // This is a legacy system, which should be unnecessary.
    //
    // Somehow including it slows file loading by the official FBX SDK,
    // and as it can reconstruct it from the deformers anyway,
    // this is not currently included.
    //
    // The code is kept here in case it's useful in the future,
    // but it's pretty much a hack anyway,
    // as assimp doesn't store bindpose information for full skeletons.
    //
    /*for (size_t mi = 0; mi < mScene->mNumMeshes; ++mi) {
        aiMesh* mesh = mScene->mMeshes[mi];
        if (! mesh->HasBones()) { continue; }
        int64_t bindpose_uid = generate_uid();
        FBX::Node bpnode("Pose");
        bpnode.AddProperty(bindpose_uid);
        // note: this uid is never linked or connected to anything.
        bpnode.AddProperty(FBX::SEPARATOR + "Pose"); // blank name
        bpnode.AddProperty("BindPose");

        bpnode.AddChild("Type", "BindPose");
        bpnode.AddChild("Version", int32_t(100));

        aiNode* mesh_node = get_node_for_mesh(mi, mScene->mRootNode);

        // next get the whole skeleton for this mesh.
        // we need it all to define the bindpose section.
        // the FBX SDK will complain if it's missing,
        // and also if parents of used bones don't have a subdeformer.
        // order shouldn't matter.
        std::set<aiNode*> skeleton;
        for (size_t bi = 0; bi < mesh->mNumBones; ++bi) {
            // bone node should have already been indexed
            const aiBone* b = mesh->mBones[bi];
            const std::string bone_name(b->mName.C_Str());
            aiNode* parent = node_by_bone[bone_name];
            // insert all nodes down to the root or mesh node
            while (
                parent
                && parent != mScene->mRootNode
                && parent != mesh_node
            ) {
                skeleton.insert(parent);
                parent = parent->mParent;
            }
        }

        // number of pose nodes. includes one for the mesh itself.
        bpnode.AddChild("NbPoseNodes", int32_t(1 + skeleton.size()));

        // the first pose node is always the mesh itself
        FBX::Node pose("PoseNode");
        pose.AddChild("Node", mesh_uids[mi]);
        aiMatrix4x4 mesh_node_xform = get_world_transform(mesh_node, mScene);
        pose.AddChild("Matrix", mesh_node_xform);
        bpnode.AddChild(pose);

        for (aiNode* bonenode : skeleton) {
            // does this node have a uid yet?
            int64_t node_uid;
            auto node_uid_iter = node_uids.find(bonenode);
            if (node_uid_iter != node_uids.end()) {
                node_uid = node_uid_iter->second;
            } else {
                node_uid = generate_uid();
                node_uids[bonenode] = node_uid;
            }

            // make a pose thingy
            pose = FBX::Node("PoseNode");
            pose.AddChild("Node", node_uid);
            aiMatrix4x4 node_xform = get_world_transform(bonenode, mScene);
            pose.AddChild("Matrix", node_xform);
            bpnode.AddChild(pose);
        }

        // now write it
        bpnode.Dump(outstream, binary, indent);
    }*/

    // TODO: cameras, lights

    // write nodes (i.e. model hierarchy)
    // start at root node
    WriteModelNodes(
        outstream, mScene->mRootNode, 0, limbnodes
    );

    // animations
    //
    // in FBX there are:
    // * AnimationStack - corresponds to an aiAnimation
    // * AnimationLayer - a combinable animation component
    // * AnimationCurveNode - links the property to be animated
    // * AnimationCurve - defines animation data for a single property value
    //
    // the CurveNode also provides the default value for a property,
    // such as the X, Y, Z coordinates for animatable translation.
    //
    // the Curve only specifies values for one component of the property,
    // so there will be a separate AnimationCurve for X, Y, and Z.
    //
    // Assimp has:
    // * aiAnimation - basically corresponds to an AnimationStack
    // * aiNodeAnim - defines all animation for one aiNode
    // * aiVectorKey/aiQuatKey - define the keyframe data for T/R/S
    //
    // assimp has no equivalent for AnimationLayer,
    // and these are flattened on FBX import.
    // we can assume there will be one per AnimationStack.
    //
    // the aiNodeAnim contains all animation data for a single aiNode,
    // which will correspond to three AnimationCurveNode's:
    // one each for translation, rotation and scale.
    // The data for each of these will be put in 9 AnimationCurve's,
    // T.X, T.Y, T.Z, R.X, R.Y, R.Z, etc.

    // AnimationStack / aiAnimation
    std::vector<int64_t> animation_stack_uids(mScene->mNumAnimations);
    for (size_t ai = 0; ai < mScene->mNumAnimations; ++ai) {
        int64_t animstack_uid = generate_uid();
        animation_stack_uids[ai] = animstack_uid;
        const aiAnimation* anim = mScene->mAnimations[ai];

        FBX::Node asnode("AnimationStack");
        std::string name = anim->mName.C_Str() + FBX::SEPARATOR + "AnimStack";
        asnode.AddProperties(animstack_uid, name, "");
        FBX::Node p("Properties70");
        p.AddP70time("LocalStart", 0); // assimp doesn't store this
        p.AddP70time("LocalStop", to_ktime(anim->mDuration, anim));
        p.AddP70time("ReferenceStart", 0);
        p.AddP70time("ReferenceStop", to_ktime(anim->mDuration, anim));
        asnode.AddChild(p);

        // this node absurdly always pretends it has children
        // (in this case it does, but just in case...)
        asnode.force_has_children = true;
        asnode.Dump(outstream, binary, indent);

        // note: animation stacks are not connected to anything
    }

    // AnimationLayer - one per aiAnimation
    std::vector<int64_t> animation_layer_uids(mScene->mNumAnimations);
    for (size_t ai = 0; ai < mScene->mNumAnimations; ++ai) {
        int64_t animlayer_uid = generate_uid();
        animation_layer_uids[ai] = animlayer_uid;
        FBX::Node alnode("AnimationLayer");
        alnode.AddProperties(animlayer_uid, FBX::SEPARATOR + "AnimLayer", "");

        // this node absurdly always pretends it has children
        alnode.force_has_children = true;
        alnode.Dump(outstream, binary, indent);

        // connect to the relevant animstack
        connections.emplace_back(
            "C", "OO", animlayer_uid, animation_stack_uids[ai]
        );
    }

    // AnimCurveNode - three per aiNodeAnim
    std::vector<std::vector<std::array<int64_t,3>>> curve_node_uids;
    for (size_t ai = 0; ai < mScene->mNumAnimations; ++ai) {
        const aiAnimation* anim = mScene->mAnimations[ai];
        const int64_t layer_uid = animation_layer_uids[ai];
        std::vector<std::array<int64_t,3>> nodeanim_uids;
        for (size_t nai = 0; nai < anim->mNumChannels; ++nai) {
            const aiNodeAnim* na = anim->mChannels[nai];
            // get the corresponding aiNode
            const aiNode* node = mScene->mRootNode->FindNode(na->mNodeName);
            // and its transform
            const aiMatrix4x4 node_xfm = get_world_transform(node, mScene);
            aiVector3D T, R, S;
            node_xfm.Decompose(S, R, T);

            // AnimationCurveNode uids
            std::array<int64_t,3> ids;
            ids[0] = generate_uid(); // T
            ids[1] = generate_uid(); // R
            ids[2] = generate_uid(); // S

            // translation
            WriteAnimationCurveNode(outstream,
                ids[0], "T", T, "Lcl Translation",
                layer_uid, node_uids[node]
            );

            // rotation
            WriteAnimationCurveNode(outstream,
                ids[1], "R", R, "Lcl Rotation",
                layer_uid, node_uids[node]
            );

            // scale
            WriteAnimationCurveNode(outstream,
                ids[2], "S", S, "Lcl Scale",
                layer_uid, node_uids[node]
            );

            // store the uids for later use
            nodeanim_uids.push_back(ids);
        }
        curve_node_uids.push_back(nodeanim_uids);
    }

    // AnimCurve - defines actual keyframe data.
    // there's a separate curve for every component of every vector,
    // for example a transform curvenode will have separate X/Y/Z AnimCurve's
    for (size_t ai = 0; ai < mScene->mNumAnimations; ++ai) {
        const aiAnimation* anim = mScene->mAnimations[ai];
        for (size_t nai = 0; nai < anim->mNumChannels; ++nai) {
            const aiNodeAnim* na = anim->mChannels[nai];
            // get the corresponding aiNode
            const aiNode* node = mScene->mRootNode->FindNode(na->mNodeName);
            // and its transform
            const aiMatrix4x4 node_xfm = get_world_transform(node, mScene);
            aiVector3D T, R, S;
            node_xfm.Decompose(S, R, T);
            const std::array<int64_t,3>& ids = curve_node_uids[ai][nai];

            std::vector<int64_t> times;
            std::vector<float> xval, yval, zval;

            // position/translation
            for (size_t ki = 0; ki < na->mNumPositionKeys; ++ki) {
                const aiVectorKey& k = na->mPositionKeys[ki];
                times.push_back(to_ktime(k.mTime));
                xval.push_back(k.mValue.x);
                yval.push_back(k.mValue.y);
                zval.push_back(k.mValue.z);
            }
            // one curve each for X, Y, Z
            WriteAnimationCurve(outstream, T.x, times, xval, ids[0], "d|X");
            WriteAnimationCurve(outstream, T.y, times, yval, ids[0], "d|Y");
            WriteAnimationCurve(outstream, T.z, times, zval, ids[0], "d|Z");

            // rotation
            times.clear(); xval.clear(); yval.clear(); zval.clear();
            for (size_t ki = 0; ki < na->mNumRotationKeys; ++ki) {
                const aiQuatKey& k = na->mRotationKeys[ki];
                times.push_back(to_ktime(k.mTime));
                // TODO: aiQuaternion method to convert to Euler...
                aiMatrix4x4 m(k.mValue.GetMatrix());
                aiVector3D qs, qr, qt;
                m.Decompose(qs, qr, qt);
                qr *= DEG;
                xval.push_back(qr.x);
                yval.push_back(qr.y);
                zval.push_back(qr.z);
            }
            WriteAnimationCurve(outstream, R.x, times, xval, ids[1], "d|X");
            WriteAnimationCurve(outstream, R.y, times, yval, ids[1], "d|Y");
            WriteAnimationCurve(outstream, R.z, times, zval, ids[1], "d|Z");

            // scaling/scale
            times.clear(); xval.clear(); yval.clear(); zval.clear();
            for (size_t ki = 0; ki < na->mNumScalingKeys; ++ki) {
                const aiVectorKey& k = na->mScalingKeys[ki];
                times.push_back(to_ktime(k.mTime));
                xval.push_back(k.mValue.x);
                yval.push_back(k.mValue.y);
                zval.push_back(k.mValue.z);
            }
            WriteAnimationCurve(outstream, S.x, times, xval, ids[2], "d|X");
            WriteAnimationCurve(outstream, S.y, times, yval, ids[2], "d|Y");
            WriteAnimationCurve(outstream, S.z, times, zval, ids[2], "d|Z");
        }
    }

    indent = 0;
    object_node.End(outstream, binary, indent, true);
}

// convenience map of magic node name strings to FBX properties,
// including the expected type of transform.
const std::map<std::string,std::pair<std::string,char>> transform_types = {
    {"Translation", {"Lcl Translation", 't'}},
    {"RotationOffset", {"RotationOffset", 't'}},
    {"RotationPivot", {"RotationPivot", 't'}},
    {"PreRotation", {"PreRotation", 'r'}},
    {"Rotation", {"Lcl Rotation", 'r'}},
    {"PostRotation", {"PostRotation", 'r'}},
    {"RotationPivotInverse", {"RotationPivotInverse", 'i'}},
    {"ScalingOffset", {"ScalingOffset", 't'}},
    {"ScalingPivot", {"ScalingPivot", 't'}},
    {"Scaling", {"Lcl Scaling", 's'}},
    {"ScalingPivotInverse", {"ScalingPivotInverse", 'i'}},
    {"GeometricScaling", {"GeometricScaling", 's'}},
    {"GeometricRotation", {"GeometricRotation", 'r'}},
    {"GeometricTranslation", {"GeometricTranslation", 't'}},
    {"GeometricTranslationInverse", {"GeometricTranslationInverse", 'i'}},
    {"GeometricRotationInverse", {"GeometricRotationInverse", 'i'}},
    {"GeometricScalingInverse", {"GeometricScalingInverse", 'i'}}
};

// write a single model node to the stream
void FBXExporter::WriteModelNode(
    StreamWriterLE& outstream,
    bool binary,
    const aiNode* node,
    int64_t node_uid,
    const std::string& type,
    const std::vector<std::pair<std::string,aiVector3D>>& transform_chain,
    TransformInheritance inherit_type
){
    const aiVector3D zero = {0, 0, 0};
    const aiVector3D one = {1, 1, 1};
    FBX::Node m("Model");
    std::string name = node->mName.C_Str() + FBX::SEPARATOR + "Model";
    m.AddProperties(node_uid, name, type);
    m.AddChild("Version", int32_t(232));
    FBX::Node p("Properties70");
    p.AddP70bool("RotationActive", 1);
    p.AddP70int("DefaultAttributeIndex", 0);
    p.AddP70enum("InheritType", inherit_type);
    if (transform_chain.empty()) {
        // decompose 4x4 transform matrix into TRS
        aiVector3D t, r, s;
        node->mTransformation.Decompose(s, r, t);
        if (t != zero) {
            p.AddP70(
                "Lcl Translation", "Lcl Translation", "", "A",
                double(t.x), double(t.y), double(t.z)
            );
        }
        if (r != zero) {
            p.AddP70(
                "Lcl Rotation", "Lcl Rotation", "", "A",
                double(DEG*r.x), double(DEG*r.y), double(DEG*r.z)
            );
        }
        if (s != one) {
            p.AddP70(
                "Lcl Scaling", "Lcl Scaling", "", "A",
                double(s.x), double(s.y), double(s.z)
            );
        }
    } else {
        // apply the transformation chain.
        // these transformation elements are created when importing FBX,
        // which has a complex transformation hierarchy for each node.
        // as such we can bake the hierarchy back into the node on export.
        for (auto &item : transform_chain) {
            auto elem = transform_types.find(item.first);
            if (elem == transform_types.end()) {
                // then this is a bug
                std::stringstream err;
                err << "unrecognized FBX transformation type: ";
                err << item.first;
                throw DeadlyExportError(err.str());
            }
            const std::string &name = elem->second.first;
            const aiVector3D &v = item.second;
            if (name.compare(0, 4, "Lcl ") == 0) {
                // special handling for animatable properties
                p.AddP70(
                    name, name, "", "A",
                    double(v.x), double(v.y), double(v.z)
                );
            } else {
                p.AddP70vector(name, v.x, v.y, v.z);
            }
        }
    }
    m.AddChild(p);

    // not sure what these are for,
    // but they seem to be omnipresent
    m.AddChild("Shading", Property(true));
    m.AddChild("Culling", Property("CullingOff"));

    m.Dump(outstream, binary, 1);
}

// wrapper for WriteModelNodes to create and pass a blank transform chain
void FBXExporter::WriteModelNodes(
    StreamWriterLE& s,
    const aiNode* node,
    int64_t parent_uid,
    const std::unordered_set<const aiNode*>& limbnodes
) {
    std::vector<std::pair<std::string,aiVector3D>> chain;
    WriteModelNodes(s, node, parent_uid, limbnodes, chain);
}

void FBXExporter::WriteModelNodes(
    StreamWriterLE& outstream,
    const aiNode* node,
    int64_t parent_uid,
    const std::unordered_set<const aiNode*>& limbnodes,
    std::vector<std::pair<std::string,aiVector3D>>& transform_chain
) {
    // first collapse any expanded transformation chains created by FBX import.
    std::string node_name(node->mName.C_Str());
    if (node_name.find(MAGIC_NODE_TAG) != std::string::npos) {
        auto pos = node_name.find(MAGIC_NODE_TAG) + MAGIC_NODE_TAG.size() + 1;
        std::string type_name = node_name.substr(pos);
        auto elem = transform_types.find(type_name);
        if (elem == transform_types.end()) {
            // then this is a bug and should be fixed
            std::stringstream err;
            err << "unrecognized FBX transformation node";
            err << " of type " << type_name << " in node " << node_name;
            throw DeadlyExportError(err.str());
        }
        aiVector3D t, r, s;
        node->mTransformation.Decompose(s, r, t);
        switch (elem->second.second) {
        case 'i': // inverse
            // we don't need to worry about the inverse matrices
            break;
        case 't': // translation
            transform_chain.emplace_back(elem->first, t);
            break;
        case 'r': // rotation
            r *= float(DEG);
            transform_chain.emplace_back(elem->first, r);
            break;
        case 's': // scale
            transform_chain.emplace_back(elem->first, s);
            break;
        default:
            // this should never happen
            std::stringstream err;
            err << "unrecognized FBX transformation type code: ";
            err << elem->second.second;
            throw DeadlyExportError(err.str());
        }
        // now continue on to any child nodes
        for (unsigned i = 0; i < node->mNumChildren; ++i) {
            WriteModelNodes(
                outstream,
                node->mChildren[i],
                parent_uid,
                limbnodes,
                transform_chain
            );
        }
        return;
    }

    int64_t node_uid = 0;
    // generate uid and connect to parent, if not the root node,
    if (node != mScene->mRootNode) {
        auto elem = node_uids.find(node);
        if (elem != node_uids.end()) {
            node_uid = elem->second;
        } else {
            node_uid = generate_uid();
            node_uids[node] = node_uid;
        }
        connections.emplace_back("C", "OO", node_uid, parent_uid);
    }

    // what type of node is this?
    if (node == mScene->mRootNode) {
        // handled later
    } else if (node->mNumMeshes == 1) {
        // connect to child mesh, which should have been written previously
        connections.emplace_back(
            "C", "OO", mesh_uids[node->mMeshes[0]], node_uid
        );
        // also connect to the material for the child mesh
        connections.emplace_back(
            "C", "OO",
            material_uids[mScene->mMeshes[node->mMeshes[0]]->mMaterialIndex],
            node_uid
        );
        // write model node
        WriteModelNode(
            outstream, binary, node, node_uid, "Mesh", transform_chain
        );
    } else if (limbnodes.count(node)) {
        WriteModelNode(
            outstream, binary, node, node_uid, "LimbNode", transform_chain
        );
        // we also need to write a nodeattribute to mark it as a skeleton
        int64_t node_attribute_uid = generate_uid();
        FBX::Node na("NodeAttribute");
        na.AddProperties(
            node_attribute_uid, FBX::SEPARATOR + "NodeAttribute", "LimbNode"
        );
        na.AddChild("TypeFlags", Property("Skeleton"));
        na.Dump(outstream, binary, 1);
        // and connect them
        connections.emplace_back("C", "OO", node_attribute_uid, node_uid);
    } else {
        // generate a null node so we can add children to it
        WriteModelNode(
            outstream, binary, node, node_uid, "Null", transform_chain
        );
    }

    // if more than one child mesh, make nodes for each mesh
    if (node->mNumMeshes > 1 || node == mScene->mRootNode) {
        for (size_t i = 0; i < node->mNumMeshes; ++i) {
            // make a new model node
            int64_t new_node_uid = generate_uid();
            // connect to parent node
            connections.emplace_back("C", "OO", new_node_uid, node_uid);
            // connect to child mesh, which should have been written previously
            connections.emplace_back(
                "C", "OO", mesh_uids[node->mMeshes[i]], new_node_uid
            );
            // also connect to the material for the child mesh
            connections.emplace_back(
                "C", "OO",
                material_uids[
                    mScene->mMeshes[node->mMeshes[i]]->mMaterialIndex
                ],
                new_node_uid
            );
            // write model node
            FBX::Node m("Model");
            // take name from mesh name, if it exists
            std::string name = mScene->mMeshes[node->mMeshes[i]]->mName.C_Str();
            name += FBX::SEPARATOR + "Model";
            m.AddProperties(new_node_uid, name, "Mesh");
            m.AddChild("Version", int32_t(232));
            FBX::Node p("Properties70");
            p.AddP70enum("InheritType", 1);
            m.AddChild(p);
            m.Dump(outstream, binary, 1);
        }
    }

    // now recurse into children
    for (size_t i = 0; i < node->mNumChildren; ++i) {
        WriteModelNodes(
            outstream, node->mChildren[i], node_uid, limbnodes
        );
    }
}


void FBXExporter::WriteAnimationCurveNode(
    StreamWriterLE& outstream,
    int64_t uid,
    std::string name, // "T", "R", or "S"
    aiVector3D default_value,
    std::string property_name, // "Lcl Translation" etc
    int64_t layer_uid,
    int64_t node_uid
) {
    FBX::Node n("AnimationCurveNode");
    n.AddProperties(uid, name + FBX::SEPARATOR + "AnimCurveNode", "");
    FBX::Node p("Properties70");
    p.AddP70numberA("d|X", default_value.x);
    p.AddP70numberA("d|Y", default_value.y);
    p.AddP70numberA("d|Z", default_value.z);
    n.AddChild(p);
    n.Dump(outstream, binary, 1);
    // connect to layer
    this->connections.emplace_back("C", "OO", uid, layer_uid);
    // connect to bone
    this->connections.emplace_back("C", "OP", uid, node_uid, property_name);
}


void FBXExporter::WriteAnimationCurve(
    StreamWriterLE& outstream,
    double default_value,
    const std::vector<int64_t>& times,
    const std::vector<float>& values,
    int64_t curvenode_uid,
    const std::string& property_link // "d|X", "d|Y", etc
) {
    FBX::Node n("AnimationCurve");
    int64_t curve_uid = generate_uid();
    n.AddProperties(curve_uid, FBX::SEPARATOR + "AnimCurve", "");
    n.AddChild("Default", default_value);
    n.AddChild("KeyVer", int32_t(4009));
    n.AddChild("KeyTime", times);
    n.AddChild("KeyValueFloat", values);
    // TODO: keyattr flags and data (STUB for now)
    n.AddChild("KeyAttrFlags", std::vector<int32_t>{0});
    n.AddChild("KeyAttrDataFloat", std::vector<float>{0,0,0,0});
    n.AddChild(
        "KeyAttrRefCount",
        std::vector<int32_t>{static_cast<int32_t>(times.size())}
    );
    n.Dump(outstream, binary, 1);
    this->connections.emplace_back(
        "C", "OP", curve_uid, curvenode_uid, property_link
    );
}


void FBXExporter::WriteConnections ()
{
    // we should have completed the connection graph already,
    // so basically just dump it here
    if (!binary) {
        WriteAsciiSectionHeader("Object connections");
    }
    // TODO: comments with names in the ascii version
    FBX::Node conn("Connections");
    StreamWriterLE outstream(outfile);
    conn.Begin(outstream, binary, 0);
    conn.BeginChildren(outstream, binary, 0);
    for (auto &n : connections) {
        n.Dump(outstream, binary, 1);
    }
    conn.End(outstream, binary, 0, !connections.empty());
    connections.clear();
}

#endif // ASSIMP_BUILD_NO_FBX_EXPORTER
#endif // ASSIMP_BUILD_NO_EXPORT
