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
  copyright notice, this list of conditions and the*
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

/** @file  FBXDocument.cpp
 *  @brief Implementation of the FBX DOM classes
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

#include "FBXDocument.h"
#include "FBXMeshGeometry.h"
#include "FBXParser.h"
#include "FBXUtil.h"
#include "FBXImporter.h"
#include "FBXImportSettings.h"
#include "FBXDocumentUtil.h"
#include "FBXProperties.h"

#include <memory>
#include <functional>
#include <map>

namespace Assimp {
namespace FBX {

using namespace Util;

// ------------------------------------------------------------------------------------------------
LazyObject::LazyObject(uint64_t id, const Element& element, const Document& doc)
: doc(doc)
, element(element)
, id(id)
, flags() {
    // empty
}

// ------------------------------------------------------------------------------------------------
LazyObject::~LazyObject()
{
    // empty
}

// ------------------------------------------------------------------------------------------------
const Object* LazyObject::Get(bool dieOnError)
{
    if(IsBeingConstructed() || FailedToConstruct()) {
        return nullptr;
    }

    if (object.get()) {
        return object.get();
    }

    // if this is the root object, we return a dummy since there
    // is no root object int he fbx file - it is just referenced
    // with id 0.
    if(id == 0L) {
        object.reset(new Object(id, element, "Model::RootNode"));
        return object.get();
    }

    const Token& key = element.KeyToken();
    const TokenList& tokens = element.Tokens();

    if(tokens.size() < 3) {
        DOMError("expected at least 3 tokens: id, name and class tag",&element);
    }

    const char* err;
    std::string name = ParseTokenAsString(*tokens[1],err);
    if (err) {
        DOMError(err,&element);
    }

    // small fix for binary reading: binary fbx files don't use
    // prefixes such as Model:: in front of their names. The
    // loading code expects this at many places, though!
    // so convert the binary representation (a 0x0001) to the
    // double colon notation.
    if(tokens[1]->IsBinary()) {
        for (size_t i = 0; i < name.length(); ++i) {
            if (name[i] == 0x0 && name[i+1] == 0x1) {
                name = name.substr(i+2) + "::" + name.substr(0,i);
            }
        }
    }

    const std::string classtag = ParseTokenAsString(*tokens[2],err);
    if (err) {
        DOMError(err,&element);
    }

    // prevent recursive calls
    flags |= BEING_CONSTRUCTED;

    try {
        // this needs to be relatively fast since it happens a lot,
        // so avoid constructing strings all the time.
        const char* obtype = key.begin();
        const size_t length = static_cast<size_t>(key.end()-key.begin());

        // For debugging
        //dumpObjectClassInfo( objtype, classtag );

        if (!strncmp(obtype,"Geometry",length)) {
            if (!strcmp(classtag.c_str(),"Mesh")) {
                object.reset(new MeshGeometry(id,element,name,doc));
            }
            if (!strcmp(classtag.c_str(), "Shape")) {
                object.reset(new ShapeGeometry(id, element, name, doc));
            }
            if (!strcmp(classtag.c_str(), "Line")) {
                object.reset(new LineGeometry(id, element, name, doc));
            }
        }
        else if (!strncmp(obtype,"NodeAttribute",length)) {
            if (!strcmp(classtag.c_str(),"Camera")) {
                object.reset(new Camera(id,element,doc,name));
            }
            else if (!strcmp(classtag.c_str(),"CameraSwitcher")) {
                object.reset(new CameraSwitcher(id,element,doc,name));
            }
            else if (!strcmp(classtag.c_str(),"Light")) {
                object.reset(new Light(id,element,doc,name));
            }
            else if (!strcmp(classtag.c_str(),"Null")) {
                object.reset(new Null(id,element,doc,name));
            }
            else if (!strcmp(classtag.c_str(),"LimbNode")) {
                object.reset(new LimbNode(id,element,doc,name));
            }
        }
        else if (!strncmp(obtype,"Deformer",length)) {
            if (!strcmp(classtag.c_str(),"Cluster")) {
                object.reset(new Cluster(id,element,doc,name));
            }
            else if (!strcmp(classtag.c_str(),"Skin")) {
                object.reset(new Skin(id,element,doc,name));
            }
            else if (!strcmp(classtag.c_str(), "BlendShape")) {
                object.reset(new BlendShape(id, element, doc, name));
            }
            else if (!strcmp(classtag.c_str(), "BlendShapeChannel")) {
                object.reset(new BlendShapeChannel(id, element, doc, name));
            }
        }
        else if ( !strncmp( obtype, "Model", length ) ) {
            // FK and IK effectors are not supported
            if ( strcmp( classtag.c_str(), "IKEffector" ) && strcmp( classtag.c_str(), "FKEffector" ) ) {
                object.reset( new Model( id, element, doc, name ) );
            }
        }
        else if (!strncmp(obtype,"Material",length)) {
            object.reset(new Material(id,element,doc,name));
        }
        else if (!strncmp(obtype,"Texture",length)) {
            object.reset(new Texture(id,element,doc,name));
        }
        else if (!strncmp(obtype,"LayeredTexture",length)) {
            object.reset(new LayeredTexture(id,element,doc,name));
        }
        else if (!strncmp(obtype,"Video",length)) {
            object.reset(new Video(id,element,doc,name));
        }
        else if (!strncmp(obtype,"AnimationStack",length)) {
            object.reset(new AnimationStack(id,element,name,doc));
        }
        else if (!strncmp(obtype,"AnimationLayer",length)) {
            object.reset(new AnimationLayer(id,element,name,doc));
        }
        // note: order matters for these two
        else if (!strncmp(obtype,"AnimationCurve",length)) {
            object.reset(new AnimationCurve(id,element,name,doc));
        }
        else if (!strncmp(obtype,"AnimationCurveNode",length)) {
            object.reset(new AnimationCurveNode(id,element,name,doc));
        }
    }
    catch(std::exception& ex) {
        flags &= ~BEING_CONSTRUCTED;
        flags |= FAILED_TO_CONSTRUCT;

        if(dieOnError || doc.Settings().strictMode) {
            throw;
        }

        // note: the error message is already formatted, so raw logging is ok
        if(!DefaultLogger::isNullLogger()) {
            ASSIMP_LOG_ERROR(ex.what());
        }
        return NULL;
    }

    if (!object.get()) {
        //DOMError("failed to convert element to DOM object, class: " + classtag + ", name: " + name,&element);
    }

    flags &= ~BEING_CONSTRUCTED;
    return object.get();
}

// ------------------------------------------------------------------------------------------------
Object::Object(uint64_t id, const Element& element, const std::string& name)
: element(element)
, name(name)
, id(id)
{
    // empty
}

// ------------------------------------------------------------------------------------------------
Object::~Object()
{
    // empty
}

// ------------------------------------------------------------------------------------------------
FileGlobalSettings::FileGlobalSettings(const Document& doc, std::shared_ptr<const PropertyTable> props)
: props(props)
, doc(doc)
{
    // empty
}

// ------------------------------------------------------------------------------------------------
FileGlobalSettings::~FileGlobalSettings()
{
    // empty
}

// ------------------------------------------------------------------------------------------------
Document::Document(const Parser& parser, const ImportSettings& settings)
: settings(settings)
, parser(parser)
{
    // Cannot use array default initialization syntax because vc8 fails on it
    for (auto &timeStamp : creationTimeStamp) {
        timeStamp = 0;
    }

    ReadHeader();
    ReadPropertyTemplates();

    ReadGlobalSettings();

    // This order is important, connections need parsed objects to check
    // whether connections are ok or not. Objects may not be evaluated yet,
    // though, since this may require valid connections.
    ReadObjects();
    ReadConnections();
}

// ------------------------------------------------------------------------------------------------
Document::~Document()
{
    for(ObjectMap::value_type& v : objects) {
        delete v.second;
    }

    for(ConnectionMap::value_type& v : src_connections) {
        delete v.second;
    }
    // |dest_connections| contain the same Connection objects as the |src_connections|
}

// ------------------------------------------------------------------------------------------------
static const unsigned int LowerSupportedVersion = 7100;
static const unsigned int UpperSupportedVersion = 7400;

void Document::ReadHeader() {
    // Read ID objects from "Objects" section
    const Scope& sc = parser.GetRootScope();
    const Element* const ehead = sc["FBXHeaderExtension"];
    if(!ehead || !ehead->Compound()) {
        DOMError("no FBXHeaderExtension dictionary found");
    }

    const Scope& shead = *ehead->Compound();
    fbxVersion = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(shead,"FBXVersion",ehead),0));

    // While we may have some success with newer files, we don't support
    // the older 6.n fbx format
    if(fbxVersion < LowerSupportedVersion ) {
        DOMError("unsupported, old format version, supported are only FBX 2011, FBX 2012 and FBX 2013");
    }
    if(fbxVersion > UpperSupportedVersion ) {
        if(Settings().strictMode) {
            DOMError("unsupported, newer format version, supported are only FBX 2011, FBX 2012 and FBX 2013"
                " (turn off strict mode to try anyhow) ");
        }
        else {
            DOMWarning("unsupported, newer format version, supported are only FBX 2011, FBX 2012 and FBX 2013,"
                " trying to read it nevertheless");
        }
    }

    const Element* const ecreator = shead["Creator"];
    if(ecreator) {
        creator = ParseTokenAsString(GetRequiredToken(*ecreator,0));
    }

    const Element* const etimestamp = shead["CreationTimeStamp"];
    if(etimestamp && etimestamp->Compound()) {
        const Scope& stimestamp = *etimestamp->Compound();
        creationTimeStamp[0] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp,"Year"),0));
        creationTimeStamp[1] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp,"Month"),0));
        creationTimeStamp[2] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp,"Day"),0));
        creationTimeStamp[3] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp,"Hour"),0));
        creationTimeStamp[4] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp,"Minute"),0));
        creationTimeStamp[5] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp,"Second"),0));
        creationTimeStamp[6] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp,"Millisecond"),0));
    }
}

// ------------------------------------------------------------------------------------------------
void Document::ReadGlobalSettings()
{
    const Scope& sc = parser.GetRootScope();
    const Element* const ehead = sc["GlobalSettings"];
    if ( nullptr == ehead || !ehead->Compound() ) {
        DOMWarning( "no GlobalSettings dictionary found" );
        globals.reset(new FileGlobalSettings(*this, std::make_shared<const PropertyTable>()));
        return;
    }

    std::shared_ptr<const PropertyTable> props = GetPropertyTable( *this, "", *ehead, *ehead->Compound(), true );

    //double v = PropertyGet<float>( *props.get(), std::string("UnitScaleFactor"), 1.0 );

    if(!props) {
        DOMError("GlobalSettings dictionary contains no property table");
    }

    globals.reset(new FileGlobalSettings(*this, props));
}

// ------------------------------------------------------------------------------------------------
void Document::ReadObjects()
{
    // read ID objects from "Objects" section
    const Scope& sc = parser.GetRootScope();
    const Element* const eobjects = sc["Objects"];
    if(!eobjects || !eobjects->Compound()) {
        DOMError("no Objects dictionary found");
    }

    // add a dummy entry to represent the Model::RootNode object (id 0),
    // which is only indirectly defined in the input file
    objects[0] = new LazyObject(0L, *eobjects, *this);

    const Scope& sobjects = *eobjects->Compound();
    for(const ElementMap::value_type& el : sobjects.Elements()) {

        // extract ID
        const TokenList& tok = el.second->Tokens();

        if (tok.empty()) {
            DOMError("expected ID after object key",el.second);
        }

        const char* err;
        const uint64_t id = ParseTokenAsID(*tok[0], err);
        if(err) {
            DOMError(err,el.second);
        }

        // id=0 is normally implicit
        if(id == 0L) {
            DOMError("encountered object with implicitly defined id 0",el.second);
        }

        if(objects.find(id) != objects.end()) {
            DOMWarning("encountered duplicate object id, ignoring first occurrence",el.second);
        }

        objects[id] = new LazyObject(id, *el.second, *this);

        // grab all animation stacks upfront since there is no listing of them
        if(!strcmp(el.first.c_str(),"AnimationStack")) {
            animationStacks.push_back(id);
        }
    }
}

// ------------------------------------------------------------------------------------------------
void Document::ReadPropertyTemplates()
{
    const Scope& sc = parser.GetRootScope();
    // read property templates from "Definitions" section
    const Element* const edefs = sc["Definitions"];
    if(!edefs || !edefs->Compound()) {
        DOMWarning("no Definitions dictionary found");
        return;
    }

    const Scope& sdefs = *edefs->Compound();
    const ElementCollection otypes = sdefs.GetCollection("ObjectType");
    for(ElementMap::const_iterator it = otypes.first; it != otypes.second; ++it) {
        const Element& el = *(*it).second;
        const Scope* sc = el.Compound();
        if(!sc) {
            DOMWarning("expected nested scope in ObjectType, ignoring",&el);
            continue;
        }

        const TokenList& tok = el.Tokens();
        if(tok.empty()) {
            DOMWarning("expected name for ObjectType element, ignoring",&el);
            continue;
        }

        const std::string& oname = ParseTokenAsString(*tok[0]);

        const ElementCollection templs = sc->GetCollection("PropertyTemplate");
        for(ElementMap::const_iterator it = templs.first; it != templs.second; ++it) {
            const Element& el = *(*it).second;
            const Scope* sc = el.Compound();
            if(!sc) {
                DOMWarning("expected nested scope in PropertyTemplate, ignoring",&el);
                continue;
            }

            const TokenList& tok = el.Tokens();
            if(tok.empty()) {
                DOMWarning("expected name for PropertyTemplate element, ignoring",&el);
                continue;
            }

            const std::string& pname = ParseTokenAsString(*tok[0]);

            const Element* Properties70 = (*sc)["Properties70"];
            if(Properties70) {
                std::shared_ptr<const PropertyTable> props = std::make_shared<const PropertyTable>(
                    *Properties70,std::shared_ptr<const PropertyTable>(static_cast<const PropertyTable*>(NULL))
                );

                templates[oname+"."+pname] = props;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
void Document::ReadConnections()
{
    const Scope& sc = parser.GetRootScope();
    // read property templates from "Definitions" section
    const Element* const econns = sc["Connections"];
    if(!econns || !econns->Compound()) {
        DOMError("no Connections dictionary found");
    }

    uint64_t insertionOrder = 0l;
    const Scope& sconns = *econns->Compound();
    const ElementCollection conns = sconns.GetCollection("C");
    for(ElementMap::const_iterator it = conns.first; it != conns.second; ++it) {
        const Element& el = *(*it).second;
        const std::string& type = ParseTokenAsString(GetRequiredToken(el,0));

        // PP = property-property connection, ignored for now
        // (tokens: "PP", ID1, "Property1", ID2, "Property2")
        if ( type == "PP" ) {
            continue;
        }

        const uint64_t src = ParseTokenAsID(GetRequiredToken(el,1));
        const uint64_t dest = ParseTokenAsID(GetRequiredToken(el,2));

        // OO = object-object connection
        // OP = object-property connection, in which case the destination property follows the object ID
        const std::string& prop = (type == "OP" ? ParseTokenAsString(GetRequiredToken(el,3)) : "");

        if(objects.find(src) == objects.end()) {
            DOMWarning("source object for connection does not exist",&el);
            continue;
        }

        // dest may be 0 (root node) but we added a dummy object before
        if(objects.find(dest) == objects.end()) {
            DOMWarning("destination object for connection does not exist",&el);
            continue;
        }

        // add new connection
        const Connection* const c = new Connection(insertionOrder++,src,dest,prop,*this);
        src_connections.insert(ConnectionMap::value_type(src,c));
        dest_connections.insert(ConnectionMap::value_type(dest,c));
    }
}

// ------------------------------------------------------------------------------------------------
const std::vector<const AnimationStack*>& Document::AnimationStacks() const
{
    if (!animationStacksResolved.empty() || animationStacks.empty()) {
        return animationStacksResolved;
    }

    animationStacksResolved.reserve(animationStacks.size());
    for(uint64_t id : animationStacks) {
        LazyObject* const lazy = GetObject(id);
        const AnimationStack* stack;
        if(!lazy || !(stack = lazy->Get<AnimationStack>())) {
            DOMWarning("failed to read AnimationStack object");
            continue;
        }
        animationStacksResolved.push_back(stack);
    }

    return animationStacksResolved;
}

// ------------------------------------------------------------------------------------------------
LazyObject* Document::GetObject(uint64_t id) const
{
    ObjectMap::const_iterator it = objects.find(id);
    return it == objects.end() ? nullptr : (*it).second;
}

#define MAX_CLASSNAMES 6

// ------------------------------------------------------------------------------------------------
std::vector<const Connection*> Document::GetConnectionsSequenced(uint64_t id, const ConnectionMap& conns) const
{
    std::vector<const Connection*> temp;

    const std::pair<ConnectionMap::const_iterator,ConnectionMap::const_iterator> range =
        conns.equal_range(id);

    temp.reserve(std::distance(range.first,range.second));
    for (ConnectionMap::const_iterator it = range.first; it != range.second; ++it) {
        temp.push_back((*it).second);
    }

    std::sort(temp.begin(), temp.end(), std::mem_fn(&Connection::Compare));

    return temp; // NRVO should handle this
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection*> Document::GetConnectionsSequenced(uint64_t id, bool is_src,
    const ConnectionMap& conns,
    const char* const* classnames,
    size_t count) const

{
    ai_assert(classnames);
    ai_assert( count != 0 );
    ai_assert( count <= MAX_CLASSNAMES);

    size_t lengths[MAX_CLASSNAMES];

    const size_t c = count;
    for (size_t i = 0; i < c; ++i) {
        lengths[ i ] = strlen(classnames[i]);
    }

    std::vector<const Connection*> temp;
    const std::pair<ConnectionMap::const_iterator,ConnectionMap::const_iterator> range =
        conns.equal_range(id);

    temp.reserve(std::distance(range.first,range.second));
    for (ConnectionMap::const_iterator it = range.first; it != range.second; ++it) {
        const Token& key = (is_src
            ? (*it).second->LazyDestinationObject()
            : (*it).second->LazySourceObject()
        ).GetElement().KeyToken();

        const char* obtype = key.begin();

        for (size_t i = 0; i < c; ++i) {
            ai_assert(classnames[i]);
            if(static_cast<size_t>(std::distance(key.begin(),key.end())) == lengths[i] && !strncmp(classnames[i],obtype,lengths[i])) {
                obtype = nullptr;
                break;
            }
        }

        if(obtype) {
            continue;
        }

        temp.push_back((*it).second);
    }

    std::sort(temp.begin(), temp.end(), std::mem_fn(&Connection::Compare));
    return temp; // NRVO should handle this
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection*> Document::GetConnectionsBySourceSequenced(uint64_t source) const
{
    return GetConnectionsSequenced(source, ConnectionsBySource());
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection*> Document::GetConnectionsBySourceSequenced(uint64_t src, const char* classname) const
{
    const char* arr[] = {classname};
    return GetConnectionsBySourceSequenced(src, arr,1);
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection*> Document::GetConnectionsBySourceSequenced(uint64_t source, 
        const char* const* classnames, size_t count) const
{
    return GetConnectionsSequenced(source, true, ConnectionsBySource(),classnames, count);
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection*> Document::GetConnectionsByDestinationSequenced(uint64_t dest,
        const char* classname) const
{
    const char* arr[] = {classname};
    return GetConnectionsByDestinationSequenced(dest, arr,1);
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection*> Document::GetConnectionsByDestinationSequenced(uint64_t dest) const
{
    return GetConnectionsSequenced(dest, ConnectionsByDestination());
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection*> Document::GetConnectionsByDestinationSequenced(uint64_t dest,
    const char* const* classnames, size_t count) const

{
    return GetConnectionsSequenced(dest, false, ConnectionsByDestination(),classnames, count);
}

// ------------------------------------------------------------------------------------------------
Connection::Connection(uint64_t insertionOrder,  uint64_t src, uint64_t dest, const std::string& prop,
        const Document& doc)

: insertionOrder(insertionOrder)
, prop(prop)
, src(src)
, dest(dest)
, doc(doc)
{
    ai_assert(doc.Objects().find(src) != doc.Objects().end());
    // dest may be 0 (root node)
    ai_assert(!dest || doc.Objects().find(dest) != doc.Objects().end());
}

// ------------------------------------------------------------------------------------------------
Connection::~Connection()
{
    // empty
}

// ------------------------------------------------------------------------------------------------
LazyObject& Connection::LazySourceObject() const
{
    LazyObject* const lazy = doc.GetObject(src);
    ai_assert(lazy);
    return *lazy;
}

// ------------------------------------------------------------------------------------------------
LazyObject& Connection::LazyDestinationObject() const
{
    LazyObject* const lazy = doc.GetObject(dest);
    ai_assert(lazy);
    return *lazy;
}

// ------------------------------------------------------------------------------------------------
const Object* Connection::SourceObject() const
{
    LazyObject* const lazy = doc.GetObject(src);
    ai_assert(lazy);
    return lazy->Get();
}

// ------------------------------------------------------------------------------------------------
const Object* Connection::DestinationObject() const
{
    LazyObject* const lazy = doc.GetObject(dest);
    ai_assert(lazy);
    return lazy->Get();
}

} // !FBX
} // !Assimp

#endif
