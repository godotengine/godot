/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team

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

/** @file FBXExporter.h
* Declares the exporter class to write a scene to an fbx file
*/
#ifndef AI_FBXEXPORTER_H_INC
#define AI_FBXEXPORTER_H_INC

#ifndef ASSIMP_BUILD_NO_FBX_EXPORTER

#include "FBXExportNode.h" // FBX::Node
#include "FBXCommon.h" // FBX::TransformInheritance

#include <assimp/types.h>
//#include <assimp/material.h>
#include <assimp/StreamWriter.h> // StreamWriterLE
#include <assimp/Exceptional.h> // DeadlyExportError

#include <vector>
#include <map>
#include <unordered_set>
#include <memory> // shared_ptr
#include <sstream> // stringstream

struct aiScene;
struct aiNode;
//struct aiMaterial;

namespace Assimp
{
    class IOSystem;
    class IOStream;
    class ExportProperties;

    // ---------------------------------------------------------------------
    /** Helper class to export a given scene to an FBX file. */
    // ---------------------------------------------------------------------
    class FBXExporter
    {
    public:
        /// Constructor for a specific scene to export
        FBXExporter(const aiScene* pScene, const ExportProperties* pProperties);

        // call one of these methods to export
        void ExportBinary(const char* pFile, IOSystem* pIOSystem);
        void ExportAscii(const char* pFile, IOSystem* pIOSystem);

    private:
        bool binary; // whether current export is in binary or ascii format
        const aiScene* mScene; // the scene to export
        const ExportProperties* mProperties; // currently unused
        std::shared_ptr<IOStream> outfile; // file to write to

        std::vector<FBX::Node> connections; // connection storage

        std::vector<int64_t> mesh_uids;
        std::vector<int64_t> material_uids;
        std::map<const aiNode*,int64_t> node_uids;

        // this crude unique-ID system is actually fine
        int64_t last_uid = 999999;
        int64_t generate_uid() { return ++last_uid; }

        // binary files have a specific header and footer,
        // in addition to the actual data
        void WriteBinaryHeader();
        void WriteBinaryFooter();

        // ascii files have a comment at the top
        void WriteAsciiHeader();

        // WriteAllNodes does the actual export.
        // It just calls all the Write<Section> methods below in order.
        void WriteAllNodes();

        // Methods to write individual sections.
        // The order here matches the order inside an FBX file.
        // Each method corresponds to a top-level FBX section,
        // except WriteHeader which also includes some binary-only sections
        // and WriteFooter which is binary data only.
        void WriteHeaderExtension();
        // WriteFileId(); // binary-only, included in WriteHeader
        // WriteCreationTime(); // binary-only, included in WriteHeader
        // WriteCreator(); // binary-only, included in WriteHeader
        void WriteGlobalSettings();
        void WriteDocuments();
        void WriteReferences();
        void WriteDefinitions();
        void WriteObjects();
        void WriteConnections();
        // WriteTakes(); // deprecated since at least 2015 (fbx 7.4)

        // helpers
        void WriteAsciiSectionHeader(const std::string& title);
        void WriteModelNodes(
            Assimp::StreamWriterLE& s,
            const aiNode* node,
            int64_t parent_uid,
            const std::unordered_set<const aiNode*>& limbnodes
        );
        void WriteModelNodes( // usually don't call this directly
            StreamWriterLE& s,
            const aiNode* node,
            int64_t parent_uid,
            const std::unordered_set<const aiNode*>& limbnodes,
            std::vector<std::pair<std::string,aiVector3D>>& transform_chain
        );
        void WriteModelNode( // nor this
            StreamWriterLE& s,
            bool binary,
            const aiNode* node,
            int64_t node_uid,
            const std::string& type,
            const std::vector<std::pair<std::string,aiVector3D>>& xfm_chain,
            FBX::TransformInheritance ti_type=FBX::TransformInheritance_RSrs
        );
        void WriteAnimationCurveNode(
            StreamWriterLE& outstream,
            int64_t uid,
            const std::string& name, // "T", "R", or "S"
            aiVector3D default_value,
            std::string property_name, // "Lcl Translation" etc
            int64_t animation_layer_uid,
            int64_t node_uid
        );
        void WriteAnimationCurve(
            StreamWriterLE& outstream,
            double default_value,
            const std::vector<int64_t>& times,
            const std::vector<float>& values,
            int64_t curvenode_id,
            const std::string& property_link // "d|X", "d|Y", etc
        );
    };
}

#endif // ASSIMP_BUILD_NO_FBX_EXPORTER

#endif // AI_FBXEXPORTER_H_INC
