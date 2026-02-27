//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLFunctionStitching.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL
{
class FunctionStitchingAttributeAlwaysInline;
class FunctionStitchingFunctionNode;
class FunctionStitchingGraph;
class FunctionStitchingInputNode;
class StitchedLibraryDescriptor;

_MTL_OPTIONS(NS::UInteger, StitchedLibraryOptions) {
    StitchedLibraryOptionNone = 0,
    StitchedLibraryOptionFailOnBinaryArchiveMiss = 1,
    StitchedLibraryOptionStoreLibraryInMetalPipelinesScript = 1 << 1,
};

class FunctionStitchingAttribute : public NS::Referencing<FunctionStitchingAttribute>
{
};
class FunctionStitchingAttributeAlwaysInline : public NS::Referencing<FunctionStitchingAttributeAlwaysInline, FunctionStitchingAttribute>
{
public:
    static FunctionStitchingAttributeAlwaysInline* alloc();

    FunctionStitchingAttributeAlwaysInline*        init();
};
class FunctionStitchingNode : public NS::Copying<FunctionStitchingNode>
{
};
class FunctionStitchingInputNode : public NS::Referencing<FunctionStitchingInputNode, FunctionStitchingNode>
{
public:
    static FunctionStitchingInputNode* alloc();

    NS::UInteger                       argumentIndex() const;

    FunctionStitchingInputNode*        init();
    FunctionStitchingInputNode*        init(NS::UInteger argument);

    void                               setArgumentIndex(NS::UInteger argumentIndex);
};
class FunctionStitchingFunctionNode : public NS::Referencing<FunctionStitchingFunctionNode, FunctionStitchingNode>
{
public:
    static FunctionStitchingFunctionNode* alloc();

    NS::Array*                            arguments() const;

    NS::Array*                            controlDependencies() const;

    FunctionStitchingFunctionNode*        init();
    FunctionStitchingFunctionNode*        init(const NS::String* name, const NS::Array* arguments, const NS::Array* controlDependencies);

    NS::String*                           name() const;

    void                                  setArguments(const NS::Array* arguments);

    void                                  setControlDependencies(const NS::Array* controlDependencies);

    void                                  setName(const NS::String* name);
};
class FunctionStitchingGraph : public NS::Copying<FunctionStitchingGraph>
{
public:
    static FunctionStitchingGraph* alloc();

    NS::Array*                     attributes() const;

    NS::String*                    functionName() const;

    FunctionStitchingGraph*        init();
    FunctionStitchingGraph*        init(const NS::String* functionName, const NS::Array* nodes, const MTL::FunctionStitchingFunctionNode* outputNode, const NS::Array* attributes);

    NS::Array*                     nodes() const;

    FunctionStitchingFunctionNode* outputNode() const;

    void                           setAttributes(const NS::Array* attributes);

    void                           setFunctionName(const NS::String* functionName);

    void                           setNodes(const NS::Array* nodes);

    void                           setOutputNode(const MTL::FunctionStitchingFunctionNode* outputNode);
};
class StitchedLibraryDescriptor : public NS::Copying<StitchedLibraryDescriptor>
{
public:
    static StitchedLibraryDescriptor* alloc();

    NS::Array*                        binaryArchives() const;

    NS::Array*                        functionGraphs() const;

    NS::Array*                        functions() const;

    StitchedLibraryDescriptor*        init();

    StitchedLibraryOptions            options() const;

    void                              setBinaryArchives(const NS::Array* binaryArchives);

    void                              setFunctionGraphs(const NS::Array* functionGraphs);

    void                              setFunctions(const NS::Array* functions);

    void                              setOptions(MTL::StitchedLibraryOptions options);
};

}
_MTL_INLINE MTL::FunctionStitchingAttributeAlwaysInline* MTL::FunctionStitchingAttributeAlwaysInline::alloc()
{
    return NS::Object::alloc<MTL::FunctionStitchingAttributeAlwaysInline>(_MTL_PRIVATE_CLS(MTLFunctionStitchingAttributeAlwaysInline));
}

_MTL_INLINE MTL::FunctionStitchingAttributeAlwaysInline* MTL::FunctionStitchingAttributeAlwaysInline::init()
{
    return NS::Object::init<MTL::FunctionStitchingAttributeAlwaysInline>();
}

_MTL_INLINE MTL::FunctionStitchingInputNode* MTL::FunctionStitchingInputNode::alloc()
{
    return NS::Object::alloc<MTL::FunctionStitchingInputNode>(_MTL_PRIVATE_CLS(MTLFunctionStitchingInputNode));
}

_MTL_INLINE NS::UInteger MTL::FunctionStitchingInputNode::argumentIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(argumentIndex));
}

_MTL_INLINE MTL::FunctionStitchingInputNode* MTL::FunctionStitchingInputNode::init()
{
    return NS::Object::init<MTL::FunctionStitchingInputNode>();
}

_MTL_INLINE MTL::FunctionStitchingInputNode* MTL::FunctionStitchingInputNode::init(NS::UInteger argument)
{
    return Object::sendMessage<MTL::FunctionStitchingInputNode*>(this, _MTL_PRIVATE_SEL(initWithArgumentIndex_), argument);
}

_MTL_INLINE void MTL::FunctionStitchingInputNode::setArgumentIndex(NS::UInteger argumentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setArgumentIndex_), argumentIndex);
}

_MTL_INLINE MTL::FunctionStitchingFunctionNode* MTL::FunctionStitchingFunctionNode::alloc()
{
    return NS::Object::alloc<MTL::FunctionStitchingFunctionNode>(_MTL_PRIVATE_CLS(MTLFunctionStitchingFunctionNode));
}

_MTL_INLINE NS::Array* MTL::FunctionStitchingFunctionNode::arguments() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(arguments));
}

_MTL_INLINE NS::Array* MTL::FunctionStitchingFunctionNode::controlDependencies() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(controlDependencies));
}

_MTL_INLINE MTL::FunctionStitchingFunctionNode* MTL::FunctionStitchingFunctionNode::init()
{
    return NS::Object::init<MTL::FunctionStitchingFunctionNode>();
}

_MTL_INLINE MTL::FunctionStitchingFunctionNode* MTL::FunctionStitchingFunctionNode::init(const NS::String* name, const NS::Array* arguments, const NS::Array* controlDependencies)
{
    return Object::sendMessage<MTL::FunctionStitchingFunctionNode*>(this, _MTL_PRIVATE_SEL(initWithName_arguments_controlDependencies_), name, arguments, controlDependencies);
}

_MTL_INLINE NS::String* MTL::FunctionStitchingFunctionNode::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE void MTL::FunctionStitchingFunctionNode::setArguments(const NS::Array* arguments)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setArguments_), arguments);
}

_MTL_INLINE void MTL::FunctionStitchingFunctionNode::setControlDependencies(const NS::Array* controlDependencies)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlDependencies_), controlDependencies);
}

_MTL_INLINE void MTL::FunctionStitchingFunctionNode::setName(const NS::String* name)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setName_), name);
}

_MTL_INLINE MTL::FunctionStitchingGraph* MTL::FunctionStitchingGraph::alloc()
{
    return NS::Object::alloc<MTL::FunctionStitchingGraph>(_MTL_PRIVATE_CLS(MTLFunctionStitchingGraph));
}

_MTL_INLINE NS::Array* MTL::FunctionStitchingGraph::attributes() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(attributes));
}

_MTL_INLINE NS::String* MTL::FunctionStitchingGraph::functionName() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(functionName));
}

_MTL_INLINE MTL::FunctionStitchingGraph* MTL::FunctionStitchingGraph::init()
{
    return NS::Object::init<MTL::FunctionStitchingGraph>();
}

_MTL_INLINE MTL::FunctionStitchingGraph* MTL::FunctionStitchingGraph::init(const NS::String* functionName, const NS::Array* nodes, const MTL::FunctionStitchingFunctionNode* outputNode, const NS::Array* attributes)
{
    return Object::sendMessage<MTL::FunctionStitchingGraph*>(this, _MTL_PRIVATE_SEL(initWithFunctionName_nodes_outputNode_attributes_), functionName, nodes, outputNode, attributes);
}

_MTL_INLINE NS::Array* MTL::FunctionStitchingGraph::nodes() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(nodes));
}

_MTL_INLINE MTL::FunctionStitchingFunctionNode* MTL::FunctionStitchingGraph::outputNode() const
{
    return Object::sendMessage<MTL::FunctionStitchingFunctionNode*>(this, _MTL_PRIVATE_SEL(outputNode));
}

_MTL_INLINE void MTL::FunctionStitchingGraph::setAttributes(const NS::Array* attributes)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAttributes_), attributes);
}

_MTL_INLINE void MTL::FunctionStitchingGraph::setFunctionName(const NS::String* functionName)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctionName_), functionName);
}

_MTL_INLINE void MTL::FunctionStitchingGraph::setNodes(const NS::Array* nodes)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setNodes_), nodes);
}

_MTL_INLINE void MTL::FunctionStitchingGraph::setOutputNode(const MTL::FunctionStitchingFunctionNode* outputNode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOutputNode_), outputNode);
}

_MTL_INLINE MTL::StitchedLibraryDescriptor* MTL::StitchedLibraryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::StitchedLibraryDescriptor>(_MTL_PRIVATE_CLS(MTLStitchedLibraryDescriptor));
}

_MTL_INLINE NS::Array* MTL::StitchedLibraryDescriptor::binaryArchives() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(binaryArchives));
}

_MTL_INLINE NS::Array* MTL::StitchedLibraryDescriptor::functionGraphs() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(functionGraphs));
}

_MTL_INLINE NS::Array* MTL::StitchedLibraryDescriptor::functions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(functions));
}

_MTL_INLINE MTL::StitchedLibraryDescriptor* MTL::StitchedLibraryDescriptor::init()
{
    return NS::Object::init<MTL::StitchedLibraryDescriptor>();
}

_MTL_INLINE MTL::StitchedLibraryOptions MTL::StitchedLibraryDescriptor::options() const
{
    return Object::sendMessage<MTL::StitchedLibraryOptions>(this, _MTL_PRIVATE_SEL(options));
}

_MTL_INLINE void MTL::StitchedLibraryDescriptor::setBinaryArchives(const NS::Array* binaryArchives)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBinaryArchives_), binaryArchives);
}

_MTL_INLINE void MTL::StitchedLibraryDescriptor::setFunctionGraphs(const NS::Array* functionGraphs)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctionGraphs_), functionGraphs);
}

_MTL_INLINE void MTL::StitchedLibraryDescriptor::setFunctions(const NS::Array* functions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctions_), functions);
}

_MTL_INLINE void MTL::StitchedLibraryDescriptor::setOptions(MTL::StitchedLibraryOptions options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOptions_), options);
}
