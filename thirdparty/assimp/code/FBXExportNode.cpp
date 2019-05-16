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

#include "FBXExportNode.h"
#include "FBXCommon.h"

#include <assimp/StreamWriter.h> // StreamWriterLE
#include <assimp/Exceptional.h> // DeadlyExportError
#include <assimp/ai_assert.h>
#include <assimp/StringUtils.h> // ai_snprintf

#include <string>
#include <ostream>
#include <sstream> // ostringstream
#include <memory> // shared_ptr

// AddP70<type> helpers... there's no usable pattern here,
// so all are defined as separate functions.
// Even "animatable" properties are often completely different
// from the standard (nonanimated) property definition,
// so they are specified with an 'A' suffix.

void FBX::Node::AddP70int(
    const std::string& name, int32_t value
) {
    FBX::Node n("P");
    n.AddProperties(name, "int", "Integer", "", value);
    AddChild(n);
}

void FBX::Node::AddP70bool(
    const std::string& name, bool value
) {
    FBX::Node n("P");
    n.AddProperties(name, "bool", "", "", int32_t(value));
    AddChild(n);
}

void FBX::Node::AddP70double(
    const std::string& name, double value
) {
    FBX::Node n("P");
    n.AddProperties(name, "double", "Number", "", value);
    AddChild(n);
}

void FBX::Node::AddP70numberA(
    const std::string& name, double value
) {
    FBX::Node n("P");
    n.AddProperties(name, "Number", "", "A", value);
    AddChild(n);
}

void FBX::Node::AddP70color(
    const std::string& name, double r, double g, double b
) {
    FBX::Node n("P");
    n.AddProperties(name, "ColorRGB", "Color", "", r, g, b);
    AddChild(n);
}

void FBX::Node::AddP70colorA(
    const std::string& name, double r, double g, double b
) {
    FBX::Node n("P");
    n.AddProperties(name, "Color", "", "A", r, g, b);
    AddChild(n);
}

void FBX::Node::AddP70vector(
    const std::string& name, double x, double y, double z
) {
    FBX::Node n("P");
    n.AddProperties(name, "Vector3D", "Vector", "", x, y, z);
    AddChild(n);
}

void FBX::Node::AddP70vectorA(
    const std::string& name, double x, double y, double z
) {
    FBX::Node n("P");
    n.AddProperties(name, "Vector", "", "A", x, y, z);
    AddChild(n);
}

void FBX::Node::AddP70string(
    const std::string& name, const std::string& value
) {
    FBX::Node n("P");
    n.AddProperties(name, "KString", "", "", value);
    AddChild(n);
}

void FBX::Node::AddP70enum(
    const std::string& name, int32_t value
) {
    FBX::Node n("P");
    n.AddProperties(name, "enum", "", "", value);
    AddChild(n);
}

void FBX::Node::AddP70time(
    const std::string& name, int64_t value
) {
    FBX::Node n("P");
    n.AddProperties(name, "KTime", "Time", "", value);
    AddChild(n);
}


// public member functions for writing nodes to stream

void FBX::Node::Dump(
    std::shared_ptr<Assimp::IOStream> outfile,
    bool binary, int indent
) {
    if (binary) {
        Assimp::StreamWriterLE outstream(outfile);
        DumpBinary(outstream);
    } else {
        std::ostringstream ss;
        DumpAscii(ss, indent);
        std::string s = ss.str();
        outfile->Write(s.c_str(), s.size(), 1);
    }
}

void FBX::Node::Dump(
    Assimp::StreamWriterLE &outstream,
    bool binary, int indent
) {
    if (binary) {
        DumpBinary(outstream);
    } else {
        std::ostringstream ss;
        DumpAscii(ss, indent);
        outstream.PutString(ss.str());
    }
}


// public member functions for low-level writing

void FBX::Node::Begin(
    Assimp::StreamWriterLE &s,
    bool binary, int indent
) {
    if (binary) {
        BeginBinary(s);
    } else {
        // assume we're at the correct place to start already
        (void)indent;
        std::ostringstream ss;
        BeginAscii(ss, indent);
        s.PutString(ss.str());
    }
}

void FBX::Node::DumpProperties(
    Assimp::StreamWriterLE& s,
    bool binary, int indent
) {
    if (binary) {
        DumpPropertiesBinary(s);
    } else {
        std::ostringstream ss;
        DumpPropertiesAscii(ss, indent);
        s.PutString(ss.str());
    }
}

void FBX::Node::EndProperties(
    Assimp::StreamWriterLE &s,
    bool binary, int indent
) {
    EndProperties(s, binary, indent, properties.size());
}

void FBX::Node::EndProperties(
    Assimp::StreamWriterLE &s,
    bool binary, int indent,
    size_t num_properties
) {
    if (binary) {
        EndPropertiesBinary(s, num_properties);
    } else {
        // nothing to do
        (void)indent;
    }
}

void FBX::Node::BeginChildren(
    Assimp::StreamWriterLE &s,
    bool binary, int indent
) {
    if (binary) {
        // nothing to do
    } else {
        std::ostringstream ss;
        BeginChildrenAscii(ss, indent);
        s.PutString(ss.str());
    }
}

void FBX::Node::DumpChildren(
    Assimp::StreamWriterLE& s,
    bool binary, int indent
) {
    if (binary) {
        DumpChildrenBinary(s);
    } else {
        std::ostringstream ss;
        DumpChildrenAscii(ss, indent);
        s.PutString(ss.str());
    }
}

void FBX::Node::End(
    Assimp::StreamWriterLE &s,
    bool binary, int indent,
    bool has_children
) {
    if (binary) {
        EndBinary(s, has_children);
    } else {
        std::ostringstream ss;
        EndAscii(ss, indent, has_children);
        s.PutString(ss.str());
    }
}


// public member functions for writing to binary fbx

void FBX::Node::DumpBinary(Assimp::StreamWriterLE &s)
{
    // write header section (with placeholders for some things)
    BeginBinary(s);

    // write properties
    DumpPropertiesBinary(s);

    // go back and fill in property related placeholders
    EndPropertiesBinary(s, properties.size());

    // write children
    DumpChildrenBinary(s);

    // finish, filling in end offset placeholder
    EndBinary(s, force_has_children || !children.empty());
}


// public member functions for writing to ascii fbx

void FBX::Node::DumpAscii(std::ostream &s, int indent)
{
    // write name
    BeginAscii(s, indent);

    // write properties
    DumpPropertiesAscii(s, indent);

    if (force_has_children || !children.empty()) {
        // begin children (with a '{')
        BeginChildrenAscii(s, indent + 1);
        // write children
        DumpChildrenAscii(s, indent + 1);
    }

    // finish (also closing the children bracket '}')
    EndAscii(s, indent, force_has_children || !children.empty());
}


// private member functions for low-level writing to fbx

void FBX::Node::BeginBinary(Assimp::StreamWriterLE &s)
{
    // remember start pos so we can come back and write the end pos
    this->start_pos = s.Tell();

    // placeholders for end pos and property section info
    s.PutU4(0); // end pos
    s.PutU4(0); // number of properties
    s.PutU4(0); // total property section length

    // node name
    s.PutU1(uint8_t(name.size())); // length of node name
    s.PutString(name); // node name as raw bytes

    // property data comes after here
    this->property_start = s.Tell();
}

void FBX::Node::DumpPropertiesBinary(Assimp::StreamWriterLE& s)
{
    for (auto &p : properties) {
        p.DumpBinary(s);
    }
}

void FBX::Node::EndPropertiesBinary(
    Assimp::StreamWriterLE &s,
    size_t num_properties
) {
    if (num_properties == 0) { return; }
    size_t pos = s.Tell();
    ai_assert(pos > property_start);
    size_t property_section_size = pos - property_start;
    s.Seek(start_pos + 4);
    s.PutU4(uint32_t(num_properties));
    s.PutU4(uint32_t(property_section_size));
    s.Seek(pos);
}

void FBX::Node::DumpChildrenBinary(Assimp::StreamWriterLE& s)
{
    for (FBX::Node& child : children) {
        child.DumpBinary(s);
    }
}

void FBX::Node::EndBinary(
    Assimp::StreamWriterLE &s,
    bool has_children
) {
    // if there were children, add a null record
    if (has_children) { s.PutString(FBX::NULL_RECORD); }

    // now go back and write initial pos
    this->end_pos = s.Tell();
    s.Seek(start_pos);
    s.PutU4(uint32_t(end_pos));
    s.Seek(end_pos);
}


void FBX::Node::BeginAscii(std::ostream& s, int indent)
{
    s << '\n';
    for (int i = 0; i < indent; ++i) { s << '\t'; }
    s << name << ": ";
}

void FBX::Node::DumpPropertiesAscii(std::ostream &s, int indent)
{
    for (size_t i = 0; i < properties.size(); ++i) {
        if (i > 0) { s << ", "; }
        properties[i].DumpAscii(s, indent);
    }
}

void FBX::Node::BeginChildrenAscii(std::ostream& s, int indent)
{
    // only call this if there are actually children
    s << " {";
    (void)indent;
}

void FBX::Node::DumpChildrenAscii(std::ostream& s, int indent)
{
    // children will need a lot of padding and corralling
    if (children.size() || force_has_children) {
        for (size_t i = 0; i < children.size(); ++i) {
            // no compression in ascii files, so skip this node if it exists
            if (children[i].name == "EncryptionType") { continue; }
            // the child can dump itself
            children[i].DumpAscii(s, indent);
        }
    }
}

void FBX::Node::EndAscii(std::ostream& s, int indent, bool has_children)
{
    if (!has_children) { return; } // nothing to do
    s << '\n';
    for (int i = 0; i < indent; ++i) { s << '\t'; }
    s << "}";
}

// private helpers for static member functions

// ascii property node from vector of doubles
void FBX::Node::WritePropertyNodeAscii(
    const std::string& name,
    const std::vector<double>& v,
    Assimp::StreamWriterLE& s,
    int indent
){
    char buffer[32];
    FBX::Node node(name);
    node.Begin(s, false, indent);
    std::string vsize = to_string(v.size());
    // *<size> {
    s.PutChar('*'); s.PutString(vsize); s.PutString(" {\n");
    // indent + 1
    for (int i = 0; i < indent + 1; ++i) { s.PutChar('\t'); }
    // a: value,value,value,...
    s.PutString("a: ");
    int count = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) { s.PutChar(','); }
        int len = ai_snprintf(buffer, sizeof(buffer), "%f", v[i]);
        count += len;
        if (count > 2048) { s.PutChar('\n'); count = 0; }
        if (len < 0 || len > 31) {
            // this should never happen
            throw DeadlyExportError("failed to convert double to string");
        }
        for (int j = 0; j < len; ++j) { s.PutChar(buffer[j]); }
    }
    // }
    s.PutChar('\n');
    for (int i = 0; i < indent; ++i) { s.PutChar('\t'); }
    s.PutChar('}'); s.PutChar(' ');
    node.End(s, false, indent, false);
}

// ascii property node from vector of int32_t
void FBX::Node::WritePropertyNodeAscii(
    const std::string& name,
    const std::vector<int32_t>& v,
    Assimp::StreamWriterLE& s,
    int indent
){
    char buffer[32];
    FBX::Node node(name);
    node.Begin(s, false, indent);
    std::string vsize = to_string(v.size());
    // *<size> {
    s.PutChar('*'); s.PutString(vsize); s.PutString(" {\n");
    // indent + 1
    for (int i = 0; i < indent + 1; ++i) { s.PutChar('\t'); }
    // a: value,value,value,...
    s.PutString("a: ");
    int count = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) { s.PutChar(','); }
        int len = ai_snprintf(buffer, sizeof(buffer), "%d", v[i]);
        count += len;
        if (count > 2048) { s.PutChar('\n'); count = 0; }
        if (len < 0 || len > 31) {
            // this should never happen
            throw DeadlyExportError("failed to convert double to string");
        }
        for (int j = 0; j < len; ++j) { s.PutChar(buffer[j]); }
    }
    // }
    s.PutChar('\n');
    for (int i = 0; i < indent; ++i) { s.PutChar('\t'); }
    s.PutChar('}'); s.PutChar(' ');
    node.End(s, false, indent, false);
}

// binary property node from vector of doubles
// TODO: optional zip compression!
void FBX::Node::WritePropertyNodeBinary(
    const std::string& name,
    const std::vector<double>& v,
    Assimp::StreamWriterLE& s
){
    FBX::Node node(name);
    node.BeginBinary(s);
    s.PutU1('d');
    s.PutU4(uint32_t(v.size())); // number of elements
    s.PutU4(0); // no encoding (1 would be zip-compressed)
    s.PutU4(uint32_t(v.size()) * 8); // data size
    for (auto it = v.begin(); it != v.end(); ++it) { s.PutF8(*it); }
    node.EndPropertiesBinary(s, 1);
    node.EndBinary(s, false);
}

// binary property node from vector of int32_t
// TODO: optional zip compression!
void FBX::Node::WritePropertyNodeBinary(
    const std::string& name,
    const std::vector<int32_t>& v,
    Assimp::StreamWriterLE& s
){
    FBX::Node node(name);
    node.BeginBinary(s);
    s.PutU1('i');
    s.PutU4(uint32_t(v.size())); // number of elements
    s.PutU4(0); // no encoding (1 would be zip-compressed)
    s.PutU4(uint32_t(v.size()) * 4); // data size
    for (auto it = v.begin(); it != v.end(); ++it) { s.PutI4(*it); }
    node.EndPropertiesBinary(s, 1);
    node.EndBinary(s, false);
}

// public static member functions

// convenience function to create and write a property node,
// holding a single property which is an array of values.
// does not copy the data, so is efficient for large arrays.
void FBX::Node::WritePropertyNode(
    const std::string& name,
    const std::vector<double>& v,
    Assimp::StreamWriterLE& s,
    bool binary, int indent
){
    if (binary) {
        FBX::Node::WritePropertyNodeBinary(name, v, s);
    } else {
        FBX::Node::WritePropertyNodeAscii(name, v, s, indent);
    }
}

// convenience function to create and write a property node,
// holding a single property which is an array of values.
// does not copy the data, so is efficient for large arrays.
void FBX::Node::WritePropertyNode(
    const std::string& name,
    const std::vector<int32_t>& v,
    Assimp::StreamWriterLE& s,
    bool binary, int indent
){
    if (binary) {
        FBX::Node::WritePropertyNodeBinary(name, v, s);
    } else {
        FBX::Node::WritePropertyNodeAscii(name, v, s, indent);
    }
}

#endif // ASSIMP_BUILD_NO_FBX_EXPORTER
#endif // ASSIMP_BUILD_NO_EXPORT
