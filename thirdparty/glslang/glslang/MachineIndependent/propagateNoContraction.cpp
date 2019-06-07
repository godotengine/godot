//
// Copyright (C) 2015-2016 Google, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of Google Inc. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Visit the nodes in the glslang intermediate tree representation to
// propagate the 'noContraction' qualifier.
//

#include "propagateNoContraction.h"

#include <cstdlib>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "localintermediate.h"
namespace {

// Use a string to hold the access chain information, as in most cases the
// access chain is short and may contain only one element, which is the symbol
// ID.
// Example: struct {float a; float b;} s;
//  Object s.a will be represented with: <symbol ID of s>/0
//  Object s.b will be represented with: <symbol ID of s>/1
//  Object s will be represented with: <symbol ID of s>
// For members of vector, matrix and arrays, they will be represented with the
// same symbol ID of their container symbol objects. This is because their
// preciseness is always the same as their container symbol objects.
typedef std::string ObjectAccessChain;

// The delimiter used in the ObjectAccessChain string to separate symbol ID and
// different level of struct indices.
const char ObjectAccesschainDelimiter = '/';

// Mapping from Symbol IDs of symbol nodes, to their defining operation
// nodes.
typedef std::unordered_multimap<ObjectAccessChain, glslang::TIntermOperator*> NodeMapping;
// Mapping from object nodes to their access chain info string.
typedef std::unordered_map<glslang::TIntermTyped*, ObjectAccessChain> AccessChainMapping;

// Set of object IDs.
typedef std::unordered_set<ObjectAccessChain> ObjectAccesschainSet;
// Set of return branch nodes.
typedef std::unordered_set<glslang::TIntermBranch*> ReturnBranchNodeSet;

// A helper function to tell whether a node is 'noContraction'. Returns true if
// the node has 'noContraction' qualifier, otherwise false.
bool isPreciseObjectNode(glslang::TIntermTyped* node)
{
    return node->getType().getQualifier().noContraction;
}

// Returns true if the opcode is a dereferencing one.
bool isDereferenceOperation(glslang::TOperator op)
{
    switch (op) {
    case glslang::EOpIndexDirect:
    case glslang::EOpIndexDirectStruct:
    case glslang::EOpIndexIndirect:
    case glslang::EOpVectorSwizzle:
    case glslang::EOpMatrixSwizzle:
        return true;
    default:
        return false;
    }
}

// Returns true if the opcode leads to an assignment operation.
bool isAssignOperation(glslang::TOperator op)
{
    switch (op) {
    case glslang::EOpAssign:
    case glslang::EOpAddAssign:
    case glslang::EOpSubAssign:
    case glslang::EOpMulAssign:
    case glslang::EOpVectorTimesMatrixAssign:
    case glslang::EOpVectorTimesScalarAssign:
    case glslang::EOpMatrixTimesScalarAssign:
    case glslang::EOpMatrixTimesMatrixAssign:
    case glslang::EOpDivAssign:
    case glslang::EOpModAssign:
    case glslang::EOpAndAssign:
    case glslang::EOpLeftShiftAssign:
    case glslang::EOpRightShiftAssign:
    case glslang::EOpInclusiveOrAssign:
    case glslang::EOpExclusiveOrAssign:

    case glslang::EOpPostIncrement:
    case glslang::EOpPostDecrement:
    case glslang::EOpPreIncrement:
    case glslang::EOpPreDecrement:
        return true;
    default:
        return false;
    }
}

// A helper function to get the unsigned int from a given constant union node.
// Note the node should only hold a uint scalar.
unsigned getStructIndexFromConstantUnion(glslang::TIntermTyped* node)
{
    assert(node->getAsConstantUnion() && node->getAsConstantUnion()->isScalar());
    unsigned struct_dereference_index = node->getAsConstantUnion()->getConstArray()[0].getUConst();
    return struct_dereference_index;
}

// A helper function to generate symbol_label.
ObjectAccessChain generateSymbolLabel(glslang::TIntermSymbol* node)
{
    ObjectAccessChain symbol_id =
        std::to_string(node->getId()) + "(" + node->getName().c_str() + ")";
    return symbol_id;
}

// Returns true if the operation is an arithmetic operation and valid for
// the 'NoContraction' decoration.
bool isArithmeticOperation(glslang::TOperator op)
{
    switch (op) {
    case glslang::EOpAddAssign:
    case glslang::EOpSubAssign:
    case glslang::EOpMulAssign:
    case glslang::EOpVectorTimesMatrixAssign:
    case glslang::EOpVectorTimesScalarAssign:
    case glslang::EOpMatrixTimesScalarAssign:
    case glslang::EOpMatrixTimesMatrixAssign:
    case glslang::EOpDivAssign:
    case glslang::EOpModAssign:

    case glslang::EOpNegative:

    case glslang::EOpAdd:
    case glslang::EOpSub:
    case glslang::EOpMul:
    case glslang::EOpDiv:
    case glslang::EOpMod:

    case glslang::EOpVectorTimesScalar:
    case glslang::EOpVectorTimesMatrix:
    case glslang::EOpMatrixTimesVector:
    case glslang::EOpMatrixTimesScalar:
    case glslang::EOpMatrixTimesMatrix:

    case glslang::EOpDot:

    case glslang::EOpPostIncrement:
    case glslang::EOpPostDecrement:
    case glslang::EOpPreIncrement:
    case glslang::EOpPreDecrement:
        return true;
    default:
        return false;
    }
}

// A helper class to help manage the populating_initial_no_contraction_ flag.
template <typename T> class StateSettingGuard {
public:
    StateSettingGuard(T* state_ptr, T new_state_value)
        : state_ptr_(state_ptr), previous_state_(*state_ptr)
    {
        *state_ptr = new_state_value;
    }
    StateSettingGuard(T* state_ptr) : state_ptr_(state_ptr), previous_state_(*state_ptr) {}
    void setState(T new_state_value) { *state_ptr_ = new_state_value; }
    ~StateSettingGuard() { *state_ptr_ = previous_state_; }

private:
    T* state_ptr_;
    T previous_state_;
};

// A helper function to get the front element from a given ObjectAccessChain
ObjectAccessChain getFrontElement(const ObjectAccessChain& chain)
{
    size_t pos_delimiter = chain.find(ObjectAccesschainDelimiter);
    return pos_delimiter == std::string::npos ? chain : chain.substr(0, pos_delimiter);
}

// A helper function to get the access chain starting from the second element.
ObjectAccessChain subAccessChainFromSecondElement(const ObjectAccessChain& chain)
{
    size_t pos_delimiter = chain.find(ObjectAccesschainDelimiter);
    return pos_delimiter == std::string::npos ? "" : chain.substr(pos_delimiter + 1);
}

// A helper function to get the access chain after removing a given prefix.
ObjectAccessChain getSubAccessChainAfterPrefix(const ObjectAccessChain& chain,
                                               const ObjectAccessChain& prefix)
{
    size_t pos = chain.find(prefix);
    if (pos != 0)
        return chain;
    return chain.substr(prefix.length() + sizeof(ObjectAccesschainDelimiter));
}

//
// A traverser which traverses the whole AST and populates:
//  1) A mapping from symbol nodes' IDs to their defining operation nodes.
//  2) A set of access chains of the initial precise object nodes.
//
class TSymbolDefinitionCollectingTraverser : public glslang::TIntermTraverser {
public:
    TSymbolDefinitionCollectingTraverser(NodeMapping* symbol_definition_mapping,
                                         AccessChainMapping* accesschain_mapping,
                                         ObjectAccesschainSet* precise_objects,
                                         ReturnBranchNodeSet* precise_return_nodes);

    bool visitUnary(glslang::TVisit, glslang::TIntermUnary*) override;
    bool visitBinary(glslang::TVisit, glslang::TIntermBinary*) override;
    void visitSymbol(glslang::TIntermSymbol*) override;
    bool visitAggregate(glslang::TVisit, glslang::TIntermAggregate*) override;
    bool visitBranch(glslang::TVisit, glslang::TIntermBranch*) override;

protected:
    TSymbolDefinitionCollectingTraverser& operator=(const TSymbolDefinitionCollectingTraverser&);

    // The mapping from symbol node IDs to their defining nodes. This should be
    // populated along traversing the AST.
    NodeMapping& symbol_definition_mapping_;
    // The set of symbol node IDs for precise symbol nodes, the ones marked as
    // 'noContraction'.
    ObjectAccesschainSet& precise_objects_;
    // The set of precise return nodes.
    ReturnBranchNodeSet& precise_return_nodes_;
    // A temporary cache of the symbol node whose defining node is to be found
    // currently along traversing the AST.
    ObjectAccessChain current_object_;
    // A map from object node to its access chain. This traverser stores
    // the built access chains into this map for each object node it has
    // visited.
    AccessChainMapping& accesschain_mapping_;
    // The pointer to the Function Definition node, so we can get the
    // preciseness of the return expression from it when we traverse the
    // return branch node.
    glslang::TIntermAggregate* current_function_definition_node_;
};

TSymbolDefinitionCollectingTraverser::TSymbolDefinitionCollectingTraverser(
    NodeMapping* symbol_definition_mapping, AccessChainMapping* accesschain_mapping,
    ObjectAccesschainSet* precise_objects,
    std::unordered_set<glslang::TIntermBranch*>* precise_return_nodes)
    : TIntermTraverser(true, false, false), symbol_definition_mapping_(*symbol_definition_mapping),
      precise_objects_(*precise_objects), precise_return_nodes_(*precise_return_nodes),
      current_object_(), accesschain_mapping_(*accesschain_mapping),
      current_function_definition_node_(nullptr) {}

// Visits a symbol node, set the current_object_ to the
// current node symbol ID, and record a mapping from this node to the current
// current_object_, which is the just obtained symbol
// ID.
void TSymbolDefinitionCollectingTraverser::visitSymbol(glslang::TIntermSymbol* node)
{
    current_object_ = generateSymbolLabel(node);
    accesschain_mapping_[node] = current_object_;
}

// Visits an aggregate node, traverses all of its children.
bool TSymbolDefinitionCollectingTraverser::visitAggregate(glslang::TVisit,
                                                          glslang::TIntermAggregate* node)
{
    // This aggregate node might be a function definition node, in which case we need to
    // cache this node, so we can get the preciseness information of the return value
    // of this function later.
    StateSettingGuard<glslang::TIntermAggregate*> current_function_definition_node_setting_guard(
        &current_function_definition_node_);
    if (node->getOp() == glslang::EOpFunction) {
        // This is function definition node, we need to cache this node so that we can
        // get the preciseness of the return value later.
        current_function_definition_node_setting_guard.setState(node);
    }
    // Traverse the items in the sequence.
    glslang::TIntermSequence& seq = node->getSequence();
    for (int i = 0; i < (int)seq.size(); ++i) {
        current_object_.clear();
        seq[i]->traverse(this);
    }
    return false;
}

bool TSymbolDefinitionCollectingTraverser::visitBranch(glslang::TVisit,
                                                       glslang::TIntermBranch* node)
{
    if (node->getFlowOp() == glslang::EOpReturn && node->getExpression() &&
        current_function_definition_node_ &&
        current_function_definition_node_->getType().getQualifier().noContraction) {
        // This node is a return node with an expression, and its function has a
        // precise return value. We need to find the involved objects in its
        // expression and add them to the set of initial precise objects.
        precise_return_nodes_.insert(node);
        node->getExpression()->traverse(this);
    }
    return false;
}

// Visits a unary node. This might be an implicit assignment like i++, i--. etc.
bool TSymbolDefinitionCollectingTraverser::visitUnary(glslang::TVisit /* visit */,
                                                      glslang::TIntermUnary* node)
{
    current_object_.clear();
    node->getOperand()->traverse(this);
    if (isAssignOperation(node->getOp())) {
        // We should always be able to get an access chain of the operand node.
        assert(!current_object_.empty());

        // If the operand node object is 'precise', we collect its access chain
        // for the initial set of 'precise' objects.
        if (isPreciseObjectNode(node->getOperand())) {
            // The operand node is an 'precise' object node, add its
            // access chain to the set of 'precise' objects. This is to collect
            // the initial set of 'precise' objects.
            precise_objects_.insert(current_object_);
        }
        // Gets the symbol ID from the object's access chain.
        ObjectAccessChain id_symbol = getFrontElement(current_object_);
        // Add a mapping from the symbol ID to this assignment operation node.
        symbol_definition_mapping_.insert(std::make_pair(id_symbol, node));
    }
    // A unary node is not a dereference node, so we clear the access chain which
    // is under construction.
    current_object_.clear();
    return false;
}

// Visits a binary node and updates the mapping from symbol IDs to the definition
// nodes. Also collects the access chains for the initial precise objects.
bool TSymbolDefinitionCollectingTraverser::visitBinary(glslang::TVisit /* visit */,
                                                       glslang::TIntermBinary* node)
{
    // Traverses the left node to build the access chain info for the object.
    current_object_.clear();
    node->getLeft()->traverse(this);

    if (isAssignOperation(node->getOp())) {
        // We should always be able to get an access chain for the left node.
        assert(!current_object_.empty());

        // If the left node object is 'precise', it is an initial precise object
        // specified in the shader source. Adds it to the initial work list to
        // process later.
        if (isPreciseObjectNode(node->getLeft())) {
            // The left node is an 'precise' object node, add its access chain to
            // the set of 'precise' objects. This is to collect the initial set
            // of 'precise' objects.
            precise_objects_.insert(current_object_);
        }
        // Gets the symbol ID from the object access chain, which should be the
        // first element recorded in the access chain.
        ObjectAccessChain id_symbol = getFrontElement(current_object_);
        // Adds a mapping from the symbol ID to this assignment operation node.
        symbol_definition_mapping_.insert(std::make_pair(id_symbol, node));

        // Traverses the right node, there may be other 'assignment'
        // operations in the right.
        current_object_.clear();
        node->getRight()->traverse(this);

    } else if (isDereferenceOperation(node->getOp())) {
        // The left node (parent node) is a struct type object. We need to
        // record the access chain information of the current node into its
        // object id.
        if (node->getOp() == glslang::EOpIndexDirectStruct) {
            unsigned struct_dereference_index = getStructIndexFromConstantUnion(node->getRight());
            current_object_.push_back(ObjectAccesschainDelimiter);
            current_object_.append(std::to_string(struct_dereference_index));
        }
        accesschain_mapping_[node] = current_object_;

        // For a dereference node, there is no need to traverse the right child
        // node as the right node should always be an integer type object.

    } else {
        // For other binary nodes, still traverse the right node.
        current_object_.clear();
        node->getRight()->traverse(this);
    }
    return false;
}

// Traverses the AST and returns a tuple of four members:
// 1) a mapping from symbol IDs to the definition nodes (aka. assignment nodes) of these symbols.
// 2) a mapping from object nodes in the AST to the access chains of these objects.
// 3) a set of access chains of precise objects.
// 4) a set of return nodes with precise expressions.
std::tuple<NodeMapping, AccessChainMapping, ObjectAccesschainSet, ReturnBranchNodeSet>
getSymbolToDefinitionMappingAndPreciseSymbolIDs(const glslang::TIntermediate& intermediate)
{
    auto result_tuple = std::make_tuple(NodeMapping(), AccessChainMapping(), ObjectAccesschainSet(),
                                        ReturnBranchNodeSet());

    TIntermNode* root = intermediate.getTreeRoot();
    if (root == 0)
        return result_tuple;

    NodeMapping& symbol_definition_mapping = std::get<0>(result_tuple);
    AccessChainMapping& accesschain_mapping = std::get<1>(result_tuple);
    ObjectAccesschainSet& precise_objects = std::get<2>(result_tuple);
    ReturnBranchNodeSet& precise_return_nodes = std::get<3>(result_tuple);

    // Traverses the AST and populate the results.
    TSymbolDefinitionCollectingTraverser collector(&symbol_definition_mapping, &accesschain_mapping,
                                                   &precise_objects, &precise_return_nodes);
    root->traverse(&collector);

    return result_tuple;
}

//
// A traverser that determine whether the left node (or operand node for unary
// node) of an assignment node is 'precise', containing 'precise' or not,
// according to the access chain a given precise object which share the same
// symbol as the left node.
//
// Post-orderly traverses the left node subtree of an binary assignment node and:
//
//  1) Propagates the 'precise' from the left object nodes to this object node.
//
//  2) Builds object access chain along the traversal, and also compares with
//  the access chain of the given 'precise' object along with the traversal to
//  tell if the node to be defined is 'precise' or not.
//
class TNoContractionAssigneeCheckingTraverser : public glslang::TIntermTraverser {

    enum DecisionStatus {
        // The object node to be assigned to may contain 'precise' objects and also not 'precise' objects.
        Mixed = 0,
        // The object node to be assigned to is either a 'precise' object or a struct objects whose members are all 'precise'.
        Precise = 1,
        // The object node to be assigned to is not a 'precise' object.
        NotPreicse = 2,
    };

public:
    TNoContractionAssigneeCheckingTraverser(const AccessChainMapping& accesschain_mapping)
        : TIntermTraverser(true, false, false), accesschain_mapping_(accesschain_mapping),
          precise_object_(nullptr) {}

    // Checks the preciseness of a given assignment node with a precise object
    // represented as access chain. The precise object shares the same symbol
    // with the assignee of the given assignment node. Return a tuple of two:
    //
    //  1) The preciseness of the assignee node of this assignment node. True
    //  if the assignee contains 'precise' objects or is 'precise', false if
    //  the assignee is not 'precise' according to the access chain of the given
    //  precise object.
    //
    //  2) The incremental access chain from the assignee node to its nested
    //  'precise' object, according to the access chain of the given precise
    //  object. This incremental access chain can be empty, which means the
    //  assignee is 'precise'. Otherwise it shows the path to the nested
    //  precise object.
    std::tuple<bool, ObjectAccessChain>
    getPrecisenessAndRemainedAccessChain(glslang::TIntermOperator* node,
                                         const ObjectAccessChain& precise_object)
    {
        assert(isAssignOperation(node->getOp()));
        precise_object_ = &precise_object;
        ObjectAccessChain assignee_object;
        if (glslang::TIntermBinary* BN = node->getAsBinaryNode()) {
            // This is a binary assignment node, we need to check the
            // preciseness of the left node.
            assert(accesschain_mapping_.count(BN->getLeft()));
            // The left node (assignee node) is an object node, traverse the
            // node to let the 'precise' of nesting objects being transfered to
            // nested objects.
            BN->getLeft()->traverse(this);
            // After traversing the left node, if the left node is 'precise',
            // we can conclude this assignment should propagate 'precise'.
            if (isPreciseObjectNode(BN->getLeft())) {
                return make_tuple(true, ObjectAccessChain());
            }
            // If the preciseness of the left node (assignee node) can not
            // be determined by now, we need to compare the access chain string
            // of the assignee object with the given precise object.
            assignee_object = accesschain_mapping_.at(BN->getLeft());

        } else if (glslang::TIntermUnary* UN = node->getAsUnaryNode()) {
            // This is a unary assignment node, we need to check the
            // preciseness of the operand node. For unary assignment node, the
            // operand node should always be an object node.
            assert(accesschain_mapping_.count(UN->getOperand()));
            // Traverse the operand node to let the 'precise' being propagated
            // from lower nodes to upper nodes.
            UN->getOperand()->traverse(this);
            // After traversing the operand node, if the operand node is
            // 'precise', this assignment should propagate 'precise'.
            if (isPreciseObjectNode(UN->getOperand())) {
                return make_tuple(true, ObjectAccessChain());
            }
            // If the preciseness of the operand node (assignee node) can not
            // be determined by now, we need to compare the access chain string
            // of the assignee object with the given precise object.
            assignee_object = accesschain_mapping_.at(UN->getOperand());
        } else {
            // Not a binary or unary node, should not happen.
            assert(false);
        }

        // Compare the access chain string of the assignee node with the given
        // precise object to determine if this assignment should propagate
        // 'precise'.
        if (assignee_object.find(precise_object) == 0) {
            // The access chain string of the given precise object is a prefix
            // of assignee's access chain string. The assignee should be
            // 'precise'.
            return make_tuple(true, ObjectAccessChain());
        } else if (precise_object.find(assignee_object) == 0) {
            // The assignee's access chain string is a prefix of the given
            // precise object, the assignee object contains 'precise' object,
            // and we need to pass the remained access chain to the object nodes
            // in the right.
            return make_tuple(true, getSubAccessChainAfterPrefix(precise_object, assignee_object));
        } else {
            // The access chain strings do not match, the assignee object can
            // not be labeled as 'precise' according to the given precise
            // object.
            return make_tuple(false, ObjectAccessChain());
        }
    }

protected:
    TNoContractionAssigneeCheckingTraverser& operator=(const TNoContractionAssigneeCheckingTraverser&);

    bool visitBinary(glslang::TVisit, glslang::TIntermBinary* node) override;
    void visitSymbol(glslang::TIntermSymbol* node) override;

    // A map from object nodes to their access chain string (used as object ID).
    const AccessChainMapping& accesschain_mapping_;
    // A given precise object, represented in it access chain string. This
    // precise object is used to be compared with the assignee node to tell if
    // the assignee node is 'precise', contains 'precise' object or not
    // 'precise'.
    const ObjectAccessChain* precise_object_;
};

// Visits a binary node. If the node is an object node, it must be a dereference
// node. In such cases, if the left node is 'precise', this node should also be
// 'precise'.
bool TNoContractionAssigneeCheckingTraverser::visitBinary(glslang::TVisit,
                                                          glslang::TIntermBinary* node)
{
    // Traverses the left so that we transfer the 'precise' from nesting object
    // to its nested object.
    node->getLeft()->traverse(this);
    // If this binary node is an object node, we should have it in the
    // accesschain_mapping_.
    if (accesschain_mapping_.count(node)) {
        // A binary object node must be a dereference node.
        assert(isDereferenceOperation(node->getOp()));
        // If the left node is 'precise', this node should also be precise,
        // otherwise, compare with the given precise_object_. If the
        // access chain of this node matches with the given precise_object_,
        // this node should be marked as 'precise'.
        if (isPreciseObjectNode(node->getLeft())) {
            node->getWritableType().getQualifier().noContraction = true;
        } else if (accesschain_mapping_.at(node) == *precise_object_) {
            node->getWritableType().getQualifier().noContraction = true;
        }
    }
    return false;
}

// Visits a symbol node, if the symbol node ID (its access chain string) matches
// with the given precise object, this node should be 'precise'.
void TNoContractionAssigneeCheckingTraverser::visitSymbol(glslang::TIntermSymbol* node)
{
    // A symbol node should always be an object node, and should have been added
    // to the map from object nodes to their access chain strings.
    assert(accesschain_mapping_.count(node));
    if (accesschain_mapping_.at(node) == *precise_object_) {
        node->getWritableType().getQualifier().noContraction = true;
    }
}

//
// A traverser that only traverses the right side of binary assignment nodes
// and the operand node of unary assignment nodes.
//
// 1) Marks arithmetic operations as 'NoContraction'.
//
// 2) Find the object which should be marked as 'precise' in the right and
//    update the 'precise' object work list.
//
class TNoContractionPropagator : public glslang::TIntermTraverser {
public:
    TNoContractionPropagator(ObjectAccesschainSet* precise_objects,
                             const AccessChainMapping& accesschain_mapping)
        : TIntermTraverser(true, false, false),
          precise_objects_(*precise_objects), added_precise_object_ids_(),
          remained_accesschain_(), accesschain_mapping_(accesschain_mapping) {}

    // Propagates 'precise' in the right nodes of a given assignment node with
    // access chain record from the assignee node to a 'precise' object it
    // contains.
    void
    propagateNoContractionInOneExpression(glslang::TIntermTyped* defining_node,
                                          const ObjectAccessChain& assignee_remained_accesschain)
    {
        remained_accesschain_ = assignee_remained_accesschain;
        if (glslang::TIntermBinary* BN = defining_node->getAsBinaryNode()) {
            assert(isAssignOperation(BN->getOp()));
            BN->getRight()->traverse(this);
            if (isArithmeticOperation(BN->getOp())) {
                BN->getWritableType().getQualifier().noContraction = true;
            }
        } else if (glslang::TIntermUnary* UN = defining_node->getAsUnaryNode()) {
            assert(isAssignOperation(UN->getOp()));
            UN->getOperand()->traverse(this);
            if (isArithmeticOperation(UN->getOp())) {
                UN->getWritableType().getQualifier().noContraction = true;
            }
        }
    }

    // Propagates 'precise' in a given precise return node.
    void propagateNoContractionInReturnNode(glslang::TIntermBranch* return_node)
    {
        remained_accesschain_ = "";
        assert(return_node->getFlowOp() == glslang::EOpReturn && return_node->getExpression());
        return_node->getExpression()->traverse(this);
    }

protected:
    TNoContractionPropagator& operator=(const TNoContractionPropagator&);

    // Visits an aggregate node. The node can be a initializer list, in which
    // case we need to find the 'precise' or 'precise' containing object node
    // with the access chain record. In other cases, just need to traverse all
    // the children nodes.
    bool visitAggregate(glslang::TVisit, glslang::TIntermAggregate* node) override
    {
        if (!remained_accesschain_.empty() && node->getOp() == glslang::EOpConstructStruct) {
            // This is a struct initializer node, and the remained
            // access chain is not empty, we need to refer to the
            // assignee_remained_access_chain_ to find the nested
            // 'precise' object. And we don't need to visit other nodes in this
            // aggregate node.

            // Gets the struct dereference index that leads to 'precise' object.
            ObjectAccessChain precise_accesschain_index_str =
                getFrontElement(remained_accesschain_);
            unsigned precise_accesschain_index = (unsigned)strtoul(precise_accesschain_index_str.c_str(), nullptr, 10);
            // Gets the node pointed by the access chain index extracted before.
            glslang::TIntermTyped* potential_precise_node =
                node->getSequence()[precise_accesschain_index]->getAsTyped();
            assert(potential_precise_node);
            // Pop the front access chain index from the path, and visit the nested node.
            {
                ObjectAccessChain next_level_accesschain =
                    subAccessChainFromSecondElement(remained_accesschain_);
                StateSettingGuard<ObjectAccessChain> setup_remained_accesschain_for_next_level(
                    &remained_accesschain_, next_level_accesschain);
                potential_precise_node->traverse(this);
            }
            return false;
        }
        return true;
    }

    // Visits a binary node. A binary node can be an object node, e.g. a dereference node.
    // As only the top object nodes in the right side of an assignment needs to be visited
    // and added to 'precise' work list, this traverser won't visit the children nodes of
    // an object node. If the binary node does not represent an object node, it should
    // go on to traverse its children nodes and if it is an arithmetic operation node, this
    // operation should be marked as 'noContraction'.
    bool visitBinary(glslang::TVisit, glslang::TIntermBinary* node) override
    {
        if (isDereferenceOperation(node->getOp())) {
            // This binary node is an object node. Need to update the precise
            // object set with the access chain of this node + remained
            // access chain .
            ObjectAccessChain new_precise_accesschain = accesschain_mapping_.at(node);
            if (remained_accesschain_.empty()) {
                node->getWritableType().getQualifier().noContraction = true;
            } else {
                new_precise_accesschain += ObjectAccesschainDelimiter + remained_accesschain_;
            }
            // Cache the access chain as added precise object, so we won't add the
            // same object to the work list again.
            if (!added_precise_object_ids_.count(new_precise_accesschain)) {
                precise_objects_.insert(new_precise_accesschain);
                added_precise_object_ids_.insert(new_precise_accesschain);
            }
            // Only the upper-most object nodes should be visited, so do not
            // visit children of this object node.
            return false;
        }
        // If this is an arithmetic operation, marks this node as 'noContraction'.
        if (isArithmeticOperation(node->getOp()) && node->getBasicType() != glslang::EbtInt) {
            node->getWritableType().getQualifier().noContraction = true;
        }
        // As this node is not an object node, need to traverse the children nodes.
        return true;
    }

    // Visits a unary node. A unary node can not be an object node. If the operation
    // is an arithmetic operation, need to mark this node as 'noContraction'.
    bool visitUnary(glslang::TVisit /* visit */, glslang::TIntermUnary* node) override
    {
        // If this is an arithmetic operation, marks this with 'noContraction'
        if (isArithmeticOperation(node->getOp())) {
            node->getWritableType().getQualifier().noContraction = true;
        }
        return true;
    }

    // Visits a symbol node. A symbol node is always an object node. So we
    // should always be able to find its in our collected mapping from object
    // nodes to access chains.  As an object node, a symbol node can be either
    // 'precise' or containing 'precise' objects according to unused
    // access chain information we have when we visit this node.
    void visitSymbol(glslang::TIntermSymbol* node) override
    {
        // Symbol nodes are object nodes and should always have an
        // access chain collected before matches with it.
        assert(accesschain_mapping_.count(node));
        ObjectAccessChain new_precise_accesschain = accesschain_mapping_.at(node);
        // If the unused access chain is empty, this symbol node should be
        // marked as 'precise'.  Otherwise, the unused access chain should be
        // appended to the symbol ID to build a new access chain which points to
        // the nested 'precise' object in this symbol object.
        if (remained_accesschain_.empty()) {
            node->getWritableType().getQualifier().noContraction = true;
        } else {
            new_precise_accesschain += ObjectAccesschainDelimiter + remained_accesschain_;
        }
        // Add the new 'precise' access chain to the work list and make sure we
        // don't visit it again.
        if (!added_precise_object_ids_.count(new_precise_accesschain)) {
            precise_objects_.insert(new_precise_accesschain);
            added_precise_object_ids_.insert(new_precise_accesschain);
        }
    }

    // A set of precise objects, represented as access chains.
    ObjectAccesschainSet& precise_objects_;
    // Visited symbol nodes, should not revisit these nodes.
    ObjectAccesschainSet added_precise_object_ids_;
    // The left node of an assignment operation might be an parent of 'precise' objects.
    // This means the left node might not be an 'precise' object node, but it may contains
    // 'precise' qualifier which should be propagated to the corresponding child node in
    // the right. So we need the path from the left node to its nested 'precise' node to
    // tell us how to find the corresponding 'precise' node in the right.
    ObjectAccessChain remained_accesschain_;
    // A map from node pointers to their access chains.
    const AccessChainMapping& accesschain_mapping_;
};
}

namespace glslang {

void PropagateNoContraction(const glslang::TIntermediate& intermediate)
{
    // First, traverses the AST, records symbols with their defining operations
    // and collects the initial set of precise symbols (symbol nodes that marked
    // as 'noContraction') and precise return nodes.
    auto mappings_and_precise_objects =
        getSymbolToDefinitionMappingAndPreciseSymbolIDs(intermediate);

    // The mapping of symbol node IDs to their defining nodes. This enables us
    // to get the defining node directly from a given symbol ID without
    // traversing the tree again.
    NodeMapping& symbol_definition_mapping = std::get<0>(mappings_and_precise_objects);

    // The mapping of object nodes to their access chains recorded.
    AccessChainMapping& accesschain_mapping = std::get<1>(mappings_and_precise_objects);

    // The initial set of 'precise' objects which are represented as the
    // access chain toward them.
    ObjectAccesschainSet& precise_object_accesschains = std::get<2>(mappings_and_precise_objects);

    // The set of 'precise' return nodes.
    ReturnBranchNodeSet& precise_return_nodes = std::get<3>(mappings_and_precise_objects);

    // Second, uses the initial set of precise objects as a work list, pops an
    // access chain, extract the symbol ID from it. Then:
    //  1) Check the assignee object, see if it is 'precise' object node or
    //  contains 'precise' object. Obtain the incremental access chain from the
    //  assignee node to its nested 'precise' node (if any).
    //  2) If the assignee object node is 'precise' or it contains 'precise'
    //  objects, traverses the right side of the assignment operation
    //  expression to mark arithmetic operations as 'noContration' and update
    //  'precise' access chain work list with new found object nodes.
    // Repeat above steps until the work list is empty.
    TNoContractionAssigneeCheckingTraverser checker(accesschain_mapping);
    TNoContractionPropagator propagator(&precise_object_accesschains, accesschain_mapping);

    // We have two initial precise work lists to handle:
    //  1) precise return nodes
    //  2) precise object access chains
    // We should process the precise return nodes first and the involved
    // objects in the return expression should be added to the precise object
    // access chain set.
    while (!precise_return_nodes.empty()) {
        glslang::TIntermBranch* precise_return_node = *precise_return_nodes.begin();
        propagator.propagateNoContractionInReturnNode(precise_return_node);
        precise_return_nodes.erase(precise_return_node);
    }

    while (!precise_object_accesschains.empty()) {
        // Get the access chain of a precise object from the work list.
        ObjectAccessChain precise_object_accesschain = *precise_object_accesschains.begin();
        // Get the symbol id from the access chain.
        ObjectAccessChain symbol_id = getFrontElement(precise_object_accesschain);
        // Get all the defining nodes of that symbol ID.
        std::pair<NodeMapping::iterator, NodeMapping::iterator> range =
            symbol_definition_mapping.equal_range(symbol_id);
        // Visits all the assignment nodes of that symbol ID and
        //  1) Check if the assignee node is 'precise' or contains 'precise'
        //  objects.
        //  2) Propagate the 'precise' to the top layer object nodes
        //  in the right side of the assignment operation, update the 'precise'
        //  work list with new access chains representing the new 'precise'
        //  objects, and mark arithmetic operations as 'noContraction'.
        for (NodeMapping::iterator defining_node_iter = range.first;
             defining_node_iter != range.second; defining_node_iter++) {
            TIntermOperator* defining_node = defining_node_iter->second;
            // Check the assignee node.
            auto checker_result = checker.getPrecisenessAndRemainedAccessChain(
                defining_node, precise_object_accesschain);
            bool& contain_precise = std::get<0>(checker_result);
            ObjectAccessChain& remained_accesschain = std::get<1>(checker_result);
            // If the assignee node is 'precise' or contains 'precise', propagate the
            // 'precise' to the right. Otherwise just skip this assignment node.
            if (contain_precise) {
                propagator.propagateNoContractionInOneExpression(defining_node,
                                                                 remained_accesschain);
            }
        }
        // Remove the last processed 'precise' object from the work list.
        precise_object_accesschains.erase(precise_object_accesschain);
    }
}
};
