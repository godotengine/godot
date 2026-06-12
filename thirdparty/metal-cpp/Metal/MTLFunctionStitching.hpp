#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace NS {
    class Array;
    class String;
}

namespace MTL
{

_MTL_OPTIONS(NS::UInteger, StitchedLibraryOptions) {
    StitchedLibraryOptionNone = 0,
    StitchedLibraryOptionFailOnBinaryArchiveMiss = 1 << 0,
    StitchedLibraryOptionStoreLibraryInMetalPipelinesScript = 1 << 1,
};


class FunctionStitchingAttribute;
class FunctionStitchingAttributeAlwaysInline;
class FunctionStitchingNode;
class FunctionStitchingInputNode;
class FunctionStitchingFunctionNode;
class FunctionStitchingGraph;
class StitchedLibraryDescriptor;

class FunctionStitchingAttribute : public NS::Referencing<FunctionStitchingAttribute>
{
public:
};

class FunctionStitchingAttributeAlwaysInline : public NS::Referencing<FunctionStitchingAttributeAlwaysInline>
{
public:
    static FunctionStitchingAttributeAlwaysInline* alloc();
    FunctionStitchingAttributeAlwaysInline*        init() const;

};

class FunctionStitchingNode : public NS::Copying<FunctionStitchingNode>
{
public:
};

class FunctionStitchingInputNode : public NS::Referencing<FunctionStitchingInputNode>
{
public:
    static FunctionStitchingInputNode* alloc();
    FunctionStitchingInputNode*        init() const;

    NS::UInteger                     argumentIndex() const;
    MTL::FunctionStitchingInputNode* init(NS::UInteger argument);
    void                             setArgumentIndex(NS::UInteger argumentIndex);

};

class FunctionStitchingFunctionNode : public NS::Referencing<FunctionStitchingFunctionNode>
{
public:
    static FunctionStitchingFunctionNode* alloc();
    FunctionStitchingFunctionNode*        init() const;

    NS::Array*                          arguments() const;
    NS::Array*                          controlDependencies() const;
    MTL::FunctionStitchingFunctionNode* init(NS::String* name, NS::Array* arguments, NS::Array* controlDependencies);
    NS::String*                         name() const;
    void                                setArguments(NS::Array* arguments);
    void                                setControlDependencies(NS::Array* controlDependencies);
    void                                setName(NS::String* name);

};

class FunctionStitchingGraph : public NS::Copying<FunctionStitchingGraph>
{
public:
    static FunctionStitchingGraph* alloc();
    FunctionStitchingGraph*        init() const;

    NS::Array*                          attributes() const;
    NS::String*                         functionName() const;
    MTL::FunctionStitchingGraph*        init(NS::String* functionName, NS::Array* nodes, MTL::FunctionStitchingFunctionNode* outputNode, NS::Array* attributes);
    NS::Array*                          nodes() const;
    MTL::FunctionStitchingFunctionNode* outputNode() const;
    void                                setAttributes(NS::Array* attributes);
    void                                setFunctionName(NS::String* functionName);
    void                                setNodes(NS::Array* nodes);
    void                                setOutputNode(MTL::FunctionStitchingFunctionNode* outputNode);

};

class StitchedLibraryDescriptor : public NS::Copying<StitchedLibraryDescriptor>
{
public:
    static StitchedLibraryDescriptor* alloc();
    StitchedLibraryDescriptor*        init() const;

    NS::Array*                  binaryArchives() const;
    NS::Array*                  functionGraphs() const;
    NS::Array*                  functions() const;
    MTL::StitchedLibraryOptions options() const;
    void                        setBinaryArchives(NS::Array* binaryArchives);
    void                        setFunctionGraphs(NS::Array* functionGraphs);
    void                        setFunctions(NS::Array* functions);
    void                        setOptions(MTL::StitchedLibraryOptions options);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLFunctionStitchingAttribute;
extern "C" void *OBJC_CLASS_$_MTLFunctionStitchingAttributeAlwaysInline;
extern "C" void *OBJC_CLASS_$_MTLFunctionStitchingNode;
extern "C" void *OBJC_CLASS_$_MTLFunctionStitchingInputNode;
extern "C" void *OBJC_CLASS_$_MTLFunctionStitchingFunctionNode;
extern "C" void *OBJC_CLASS_$_MTLFunctionStitchingGraph;
extern "C" void *OBJC_CLASS_$_MTLStitchedLibraryDescriptor;

_MTL_INLINE MTL::FunctionStitchingAttributeAlwaysInline* MTL::FunctionStitchingAttributeAlwaysInline::alloc()
{
    return _MTL_msg_MTL__FunctionStitchingAttributeAlwaysInlinep_alloc((const void*)&OBJC_CLASS_$_MTLFunctionStitchingAttributeAlwaysInline, nullptr);
}

_MTL_INLINE MTL::FunctionStitchingAttributeAlwaysInline* MTL::FunctionStitchingAttributeAlwaysInline::init() const
{
    return _MTL_msg_MTL__FunctionStitchingAttributeAlwaysInlinep_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionStitchingInputNode* MTL::FunctionStitchingInputNode::alloc()
{
    return _MTL_msg_MTL__FunctionStitchingInputNodep_alloc((const void*)&OBJC_CLASS_$_MTLFunctionStitchingInputNode, nullptr);
}

_MTL_INLINE MTL::FunctionStitchingInputNode* MTL::FunctionStitchingInputNode::init() const
{
    return _MTL_msg_MTL__FunctionStitchingInputNodep_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::FunctionStitchingInputNode::argumentIndex() const
{
    return _MTL_msg_NS__UInteger_argumentIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionStitchingInputNode::setArgumentIndex(NS::UInteger argumentIndex)
{
    _MTL_msg_v_setArgumentIndex__NS__UInteger((const void*)this, nullptr, argumentIndex);
}

_MTL_INLINE MTL::FunctionStitchingInputNode* MTL::FunctionStitchingInputNode::init(NS::UInteger argument)
{
    return _MTL_msg_MTL__FunctionStitchingInputNodep_initWithArgumentIndex__NS__UInteger((const void*)this, nullptr, argument);
}

_MTL_INLINE MTL::FunctionStitchingFunctionNode* MTL::FunctionStitchingFunctionNode::alloc()
{
    return _MTL_msg_MTL__FunctionStitchingFunctionNodep_alloc((const void*)&OBJC_CLASS_$_MTLFunctionStitchingFunctionNode, nullptr);
}

_MTL_INLINE MTL::FunctionStitchingFunctionNode* MTL::FunctionStitchingFunctionNode::init() const
{
    return _MTL_msg_MTL__FunctionStitchingFunctionNodep_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::FunctionStitchingFunctionNode::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionStitchingFunctionNode::setName(NS::String* name)
{
    _MTL_msg_v_setName__NS__Stringp((const void*)this, nullptr, name);
}

_MTL_INLINE NS::Array* MTL::FunctionStitchingFunctionNode::arguments() const
{
    return _MTL_msg_NS__Arrayp_arguments((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionStitchingFunctionNode::setArguments(NS::Array* arguments)
{
    _MTL_msg_v_setArguments__NS__Arrayp((const void*)this, nullptr, arguments);
}

_MTL_INLINE NS::Array* MTL::FunctionStitchingFunctionNode::controlDependencies() const
{
    return _MTL_msg_NS__Arrayp_controlDependencies((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionStitchingFunctionNode::setControlDependencies(NS::Array* controlDependencies)
{
    _MTL_msg_v_setControlDependencies__NS__Arrayp((const void*)this, nullptr, controlDependencies);
}

_MTL_INLINE MTL::FunctionStitchingFunctionNode* MTL::FunctionStitchingFunctionNode::init(NS::String* name, NS::Array* arguments, NS::Array* controlDependencies)
{
    return _MTL_msg_MTL__FunctionStitchingFunctionNodep_initWithName_arguments_controlDependencies__NS__Stringp_NS__Arrayp_NS__Arrayp((const void*)this, nullptr, name, arguments, controlDependencies);
}

_MTL_INLINE MTL::FunctionStitchingGraph* MTL::FunctionStitchingGraph::alloc()
{
    return _MTL_msg_MTL__FunctionStitchingGraphp_alloc((const void*)&OBJC_CLASS_$_MTLFunctionStitchingGraph, nullptr);
}

_MTL_INLINE MTL::FunctionStitchingGraph* MTL::FunctionStitchingGraph::init() const
{
    return _MTL_msg_MTL__FunctionStitchingGraphp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::FunctionStitchingGraph::functionName() const
{
    return _MTL_msg_NS__Stringp_functionName((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionStitchingGraph::setFunctionName(NS::String* functionName)
{
    _MTL_msg_v_setFunctionName__NS__Stringp((const void*)this, nullptr, functionName);
}

_MTL_INLINE NS::Array* MTL::FunctionStitchingGraph::nodes() const
{
    return _MTL_msg_NS__Arrayp_nodes((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionStitchingGraph::setNodes(NS::Array* nodes)
{
    _MTL_msg_v_setNodes__NS__Arrayp((const void*)this, nullptr, nodes);
}

_MTL_INLINE MTL::FunctionStitchingFunctionNode* MTL::FunctionStitchingGraph::outputNode() const
{
    return _MTL_msg_MTL__FunctionStitchingFunctionNodep_outputNode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionStitchingGraph::setOutputNode(MTL::FunctionStitchingFunctionNode* outputNode)
{
    _MTL_msg_v_setOutputNode__MTL__FunctionStitchingFunctionNodep((const void*)this, nullptr, outputNode);
}

_MTL_INLINE NS::Array* MTL::FunctionStitchingGraph::attributes() const
{
    return _MTL_msg_NS__Arrayp_attributes((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionStitchingGraph::setAttributes(NS::Array* attributes)
{
    _MTL_msg_v_setAttributes__NS__Arrayp((const void*)this, nullptr, attributes);
}

_MTL_INLINE MTL::FunctionStitchingGraph* MTL::FunctionStitchingGraph::init(NS::String* functionName, NS::Array* nodes, MTL::FunctionStitchingFunctionNode* outputNode, NS::Array* attributes)
{
    return _MTL_msg_MTL__FunctionStitchingGraphp_initWithFunctionName_nodes_outputNode_attributes__NS__Stringp_NS__Arrayp_MTL__FunctionStitchingFunctionNodep_NS__Arrayp((const void*)this, nullptr, functionName, nodes, outputNode, attributes);
}

_MTL_INLINE MTL::StitchedLibraryDescriptor* MTL::StitchedLibraryDescriptor::alloc()
{
    return _MTL_msg_MTL__StitchedLibraryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLStitchedLibraryDescriptor, nullptr);
}

_MTL_INLINE MTL::StitchedLibraryDescriptor* MTL::StitchedLibraryDescriptor::init() const
{
    return _MTL_msg_MTL__StitchedLibraryDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::StitchedLibraryDescriptor::functionGraphs() const
{
    return _MTL_msg_NS__Arrayp_functionGraphs((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StitchedLibraryDescriptor::setFunctionGraphs(NS::Array* functionGraphs)
{
    _MTL_msg_v_setFunctionGraphs__NS__Arrayp((const void*)this, nullptr, functionGraphs);
}

_MTL_INLINE NS::Array* MTL::StitchedLibraryDescriptor::functions() const
{
    return _MTL_msg_NS__Arrayp_functions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StitchedLibraryDescriptor::setFunctions(NS::Array* functions)
{
    _MTL_msg_v_setFunctions__NS__Arrayp((const void*)this, nullptr, functions);
}

_MTL_INLINE NS::Array* MTL::StitchedLibraryDescriptor::binaryArchives() const
{
    return _MTL_msg_NS__Arrayp_binaryArchives((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StitchedLibraryDescriptor::setBinaryArchives(NS::Array* binaryArchives)
{
    _MTL_msg_v_setBinaryArchives__NS__Arrayp((const void*)this, nullptr, binaryArchives);
}

_MTL_INLINE MTL::StitchedLibraryOptions MTL::StitchedLibraryDescriptor::options() const
{
    return _MTL_msg_MTL__StitchedLibraryOptions_options((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StitchedLibraryDescriptor::setOptions(MTL::StitchedLibraryOptions options)
{
    _MTL_msg_v_setOptions__MTL__StitchedLibraryOptions((const void*)this, nullptr, options);
}
