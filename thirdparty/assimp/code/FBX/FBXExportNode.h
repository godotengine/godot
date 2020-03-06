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

/** @file FBXExportNode.h
* Declares the FBX::Node helper class for fbx export.
*/
#ifndef AI_FBXEXPORTNODE_H_INC
#define AI_FBXEXPORTNODE_H_INC

#ifndef ASSIMP_BUILD_NO_FBX_EXPORTER

#include "FBXExportProperty.h"

#include <assimp/StreamWriter.h> // StreamWriterLE

#include <string>
#include <vector>

namespace Assimp {
namespace FBX {
    class Node;
}

class FBX::Node {
public: 
    // TODO: accessors
    std::string name; // node name
    std::vector<FBX::FBXExportProperty> properties; // node properties
    std::vector<FBX::Node> children; // child nodes

    // some nodes always pretend they have children...
    bool force_has_children = false;

public: // constructors
    /// The default class constructor.
    Node() = default;

    /// The class constructor with the name.
    Node(const std::string& n)
    : name(n)
    , properties()
    , children()
    , force_has_children( false ) {
        // empty
    }

    // convenience template to construct with properties directly
    template <typename... More>
    Node(const std::string& n, const More... more)
    : name(n)
    , properties()
    , children()
    , force_has_children(false) {
        AddProperties(more...);
    }

public: // functions to add properties or children
    // add a single property to the node
    template <typename T>
    void AddProperty(T value) {
        properties.emplace_back(value);
    }

    // convenience function to add multiple properties at once
    template <typename T, typename... More>
    void AddProperties(T value, More... more) {
        properties.emplace_back(value);
        AddProperties(more...);
    }
    void AddProperties() {}

    // add a child node directly
    void AddChild(const Node& node) { children.push_back(node); }

    // convenience function to add a child node with a single property
    template <typename... More>
    void AddChild(
        const std::string& name,
        More... more
    ) {
        FBX::Node c(name);
        c.AddProperties(more...);
        children.push_back(c);
    }

public: // support specifically for dealing with Properties70 nodes

    // it really is simpler to make these all separate functions.
    // the versions with 'A' suffixes are for animatable properties.
    // those often follow a completely different format internally in FBX.
    void AddP70int(const std::string& name, int32_t value);
    void AddP70bool(const std::string& name, bool value);
    void AddP70double(const std::string& name, double value);
    void AddP70numberA(const std::string& name, double value);
    void AddP70color(const std::string& name, double r, double g, double b);
    void AddP70colorA(const std::string& name, double r, double g, double b);
    void AddP70vector(const std::string& name, double x, double y, double z);
    void AddP70vectorA(const std::string& name, double x, double y, double z);
    void AddP70string(const std::string& name, const std::string& value);
    void AddP70enum(const std::string& name, int32_t value);
    void AddP70time(const std::string& name, int64_t value);

    // template for custom P70 nodes.
    // anything that doesn't fit in the above can be created manually.
    template <typename... More>
    void AddP70(
        const std::string& name,
        const std::string& type,
        const std::string& type2,
        const std::string& flags,
        More... more
    ) {
        Node n("P");
        n.AddProperties(name, type, type2, flags, more...);
        AddChild(n);
    }

public: // member functions for writing data to a file or stream

    // write the full node to the given file or stream
    void Dump(
        std::shared_ptr<Assimp::IOStream> outfile,
        bool binary, int indent
    );
    void Dump(Assimp::StreamWriterLE &s, bool binary, int indent);

    // these other functions are for writing data piece by piece.
    // they must be used carefully.
    // for usage examples see FBXExporter.cpp.
    void Begin(Assimp::StreamWriterLE &s, bool binary, int indent);
    void DumpProperties(Assimp::StreamWriterLE& s, bool binary, int indent);
    void EndProperties(Assimp::StreamWriterLE &s, bool binary, int indent);
    void EndProperties(
        Assimp::StreamWriterLE &s, bool binary, int indent,
        size_t num_properties
    );
    void BeginChildren(Assimp::StreamWriterLE &s, bool binary, int indent);
    void DumpChildren(Assimp::StreamWriterLE& s, bool binary, int indent);
    void End(
        Assimp::StreamWriterLE &s, bool binary, int indent,
        bool has_children
    );

private: // internal functions used for writing

    void DumpBinary(Assimp::StreamWriterLE &s);
    void DumpAscii(Assimp::StreamWriterLE &s, int indent);
    void DumpAscii(std::ostream &s, int indent);

    void BeginBinary(Assimp::StreamWriterLE &s);
    void DumpPropertiesBinary(Assimp::StreamWriterLE& s);
    void EndPropertiesBinary(Assimp::StreamWriterLE &s);
    void EndPropertiesBinary(Assimp::StreamWriterLE &s, size_t num_properties);
    void DumpChildrenBinary(Assimp::StreamWriterLE& s);
    void EndBinary(Assimp::StreamWriterLE &s, bool has_children);

    void BeginAscii(std::ostream &s, int indent);
    void DumpPropertiesAscii(std::ostream &s, int indent);
    void BeginChildrenAscii(std::ostream &s, int indent);
    void DumpChildrenAscii(std::ostream &s, int indent);
    void EndAscii(std::ostream &s, int indent, bool has_children);

private: // data used for binary dumps
    size_t start_pos; // starting position in stream
    size_t end_pos; // ending position in stream
    size_t property_start; // starting position of property section

public: // static member functions

    // convenience function to create a node with a single property,
    // and write it to the stream.
    template <typename T>
    static void WritePropertyNode(
        const std::string& name,
        const T value,
        Assimp::StreamWriterLE& s,
        bool binary, int indent
    ) {
        FBX::FBXExportProperty p(value);
        FBX::Node node(name, p);
        node.Dump(s, binary, indent);
    }

    // convenience function to create and write a property node,
    // holding a single property which is an array of values.
    // does not copy the data, so is efficient for large arrays.
    static void WritePropertyNode(
        const std::string& name,
        const std::vector<double>& v,
        Assimp::StreamWriterLE& s,
        bool binary, int indent
    );

    // convenience function to create and write a property node,
    // holding a single property which is an array of values.
    // does not copy the data, so is efficient for large arrays.
    static void WritePropertyNode(
        const std::string& name,
        const std::vector<int32_t>& v,
        Assimp::StreamWriterLE& s,
        bool binary, int indent
    );

private: // static helper functions
    static void WritePropertyNodeAscii(
        const std::string& name,
        const std::vector<double>& v,
        Assimp::StreamWriterLE& s,
        int indent
    );
    static void WritePropertyNodeAscii(
        const std::string& name,
        const std::vector<int32_t>& v,
        Assimp::StreamWriterLE& s,
        int indent
    );
    static void WritePropertyNodeBinary(
        const std::string& name,
        const std::vector<double>& v,
        Assimp::StreamWriterLE& s
    );
    static void WritePropertyNodeBinary(
        const std::string& name,
        const std::vector<int32_t>& v,
        Assimp::StreamWriterLE& s
    );

};
}

#endif // ASSIMP_BUILD_NO_FBX_EXPORTER

#endif // AI_FBXEXPORTNODE_H_INC
