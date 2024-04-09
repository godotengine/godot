//
// Copyright (C) 2017 LunarG, Inc.
// Copyright (C) 2018 Google, Inc.
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
//    Neither the name of Google, Inc., nor the names of its
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

#include "attribute.h"
#include "../Include/intermediate.h"
#include "ParseHelper.h"

namespace glslang {

// extract integers out of attribute arguments stored in attribute aggregate
bool TAttributeArgs::getInt(int& value, int argNum) const 
{
    const TConstUnion* intConst = getConstUnion(EbtInt, argNum);

    if (intConst == nullptr)
        return false;

    value = intConst->getIConst();
    return true;
}


// extract strings out of attribute arguments stored in attribute aggregate.
// convert to lower case if converToLower is true (for case-insensitive compare convenience)
bool TAttributeArgs::getString(TString& value, int argNum, bool convertToLower) const 
{
    const TConstUnion* stringConst = getConstUnion(EbtString, argNum);

    if (stringConst == nullptr)
        return false;

    value = *stringConst->getSConst();

    // Convenience.
    if (convertToLower)
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);

    return true;
}

// How many arguments were supplied?
int TAttributeArgs::size() const
{
    return args == nullptr ? 0 : (int)args->getSequence().size();
}

// Helper to get attribute const union.  Returns nullptr on failure.
const TConstUnion* TAttributeArgs::getConstUnion(TBasicType basicType, int argNum) const
{
    if (args == nullptr)
        return nullptr;

    if (argNum >= (int)args->getSequence().size())
        return nullptr;

    if (args->getSequence()[argNum]->getAsConstantUnion() == nullptr)
        return nullptr;

    const TConstUnion* constVal = &args->getSequence()[argNum]->getAsConstantUnion()->getConstArray()[0];
    if (constVal == nullptr || constVal->getType() != basicType)
        return nullptr;

    return constVal;
}

// Implementation of TParseContext parts of attributes
TAttributeType TParseContext::attributeFromName(const TString& name) const
{
    if (name == "branch" || name == "dont_flatten")
        return EatBranch;
    else if (name == "flatten")
        return EatFlatten;
    else if (name == "unroll")
        return EatUnroll;
    else if (name == "loop" || name == "dont_unroll")
        return EatLoop;
    else if (name == "dependency_infinite")
        return EatDependencyInfinite;
    else if (name == "dependency_length")
        return EatDependencyLength;
    else if (name == "min_iterations")
        return EatMinIterations;
    else if (name == "max_iterations")
        return EatMaxIterations;
    else if (name == "iteration_multiple")
        return EatIterationMultiple;
    else if (name == "peel_count")
        return EatPeelCount;
    else if (name == "partial_count")
        return EatPartialCount;
    else if (name == "subgroup_uniform_control_flow")
        return EatSubgroupUniformControlFlow;
    else if (name == "export")
        return EatExport;
    else
        return EatNone;
}

// Make an initial leaf for the grammar from a no-argument attribute
TAttributes* TParseContext::makeAttributes(const TString& identifier) const
{
    TAttributes *attributes = nullptr;
    attributes = NewPoolObject(attributes);
    TAttributeArgs args = { attributeFromName(identifier), nullptr };
    attributes->push_back(args);
    return attributes;
}

// Make an initial leaf for the grammar from a one-argument attribute
TAttributes* TParseContext::makeAttributes(const TString& identifier, TIntermNode* node) const
{
    TAttributes *attributes = nullptr;
    attributes = NewPoolObject(attributes);

    // for now, node is always a simple single expression, but other code expects
    // a list, so make it so
    TIntermAggregate* agg = intermediate.makeAggregate(node);
    TAttributeArgs args = { attributeFromName(identifier), agg };
    attributes->push_back(args);
    return attributes;
}

// Merge two sets of attributes into a single set.
// The second argument is destructively consumed.
TAttributes* TParseContext::mergeAttributes(TAttributes* attr1, TAttributes* attr2) const
{
    attr1->splice(attr1->end(), *attr2);
    return attr1;
}

//
// Selection attributes
//
void TParseContext::handleSelectionAttributes(const TAttributes& attributes, TIntermNode* node)
{
    TIntermSelection* selection = node->getAsSelectionNode();
    if (selection == nullptr)
        return;

    for (auto it = attributes.begin(); it != attributes.end(); ++it) {
        if (it->size() > 0) {
            warn(node->getLoc(), "attribute with arguments not recognized, skipping", "", "");
            continue;
        }

        switch (it->name) {
        case EatFlatten:
            selection->setFlatten();
            break;
        case EatBranch:
            selection->setDontFlatten();
            break;
        default:
            warn(node->getLoc(), "attribute does not apply to a selection", "", "");
            break;
        }
    }
}

//
// Switch attributes
//
void TParseContext::handleSwitchAttributes(const TAttributes& attributes, TIntermNode* node)
{
    TIntermSwitch* selection = node->getAsSwitchNode();
    if (selection == nullptr)
        return;

    for (auto it = attributes.begin(); it != attributes.end(); ++it) {
        if (it->size() > 0) {
            warn(node->getLoc(), "attribute with arguments not recognized, skipping", "", "");
            continue;
        }

        switch (it->name) {
        case EatFlatten:
            selection->setFlatten();
            break;
        case EatBranch:
            selection->setDontFlatten();
            break;
        default:
            warn(node->getLoc(), "attribute does not apply to a switch", "", "");
            break;
        }
    }
}

//
// Loop attributes
//
void TParseContext::handleLoopAttributes(const TAttributes& attributes, TIntermNode* node)
{
    TIntermLoop* loop = node->getAsLoopNode();
    if (loop == nullptr) {
        // the actual loop might be part of a sequence
        TIntermAggregate* agg = node->getAsAggregate();
        if (agg == nullptr)
            return;
        for (auto it = agg->getSequence().begin(); it != agg->getSequence().end(); ++it) {
            loop = (*it)->getAsLoopNode();
            if (loop != nullptr)
                break;
        }
        if (loop == nullptr)
            return;
    }

    for (auto it = attributes.begin(); it != attributes.end(); ++it) {

        const auto noArgument = [&](const char* feature) {
            if (it->size() > 0) {
                warn(node->getLoc(), "expected no arguments", feature, "");
                return false;
            }
            return true;
        };

        const auto positiveSignedArgument = [&](const char* feature, int& value) {
            if (it->size() == 1 && it->getInt(value)) {
                if (value <= 0) {
                    error(node->getLoc(), "must be positive", feature, "");
                    return false;
                }
            } else {
                warn(node->getLoc(), "expected a single integer argument", feature, "");
                return false;
            }
            return true;
        };

        const auto unsignedArgument = [&](const char* feature, unsigned int& uiValue) {
            int value;
            if (!(it->size() == 1 && it->getInt(value))) {
                warn(node->getLoc(), "expected a single integer argument", feature, "");
                return false;
            }
            uiValue = (unsigned int)value;
            return true;
        };

        const auto positiveUnsignedArgument = [&](const char* feature, unsigned int& uiValue) {
            int value;
            if (it->size() == 1 && it->getInt(value)) {
                if (value == 0) {
                    error(node->getLoc(), "must be greater than or equal to 1", feature, "");
                    return false;
                }
            } else {
                warn(node->getLoc(), "expected a single integer argument", feature, "");
                return false;
            }
            uiValue = (unsigned int)value;
            return true;
        };

        const auto spirv14 = [&](const char* feature) {
            if (spvVersion.spv > 0 && spvVersion.spv < EShTargetSpv_1_4)
                warn(node->getLoc(), "attribute requires a SPIR-V 1.4 target-env", feature, "");
        };

        int value = 0;
        unsigned uiValue = 0;
        switch (it->name) {
        case EatUnroll:
            if (noArgument("unroll"))
                loop->setUnroll();
            break;
        case EatLoop:
            if (noArgument("dont_unroll"))
                loop->setDontUnroll();
            break;
        case EatDependencyInfinite:
            if (noArgument("dependency_infinite"))
                loop->setLoopDependency(TIntermLoop::dependencyInfinite);
            break;
        case EatDependencyLength:
            if (positiveSignedArgument("dependency_length", value))
                loop->setLoopDependency(value);
            break;
        case EatMinIterations:
            spirv14("min_iterations");
            if (unsignedArgument("min_iterations", uiValue))
                loop->setMinIterations(uiValue);
            break;
        case EatMaxIterations:
            spirv14("max_iterations");
            if (unsignedArgument("max_iterations", uiValue))
                loop->setMaxIterations(uiValue);
            break;
        case EatIterationMultiple:
            spirv14("iteration_multiple");
            if (positiveUnsignedArgument("iteration_multiple", uiValue))
                loop->setIterationMultiple(uiValue);
            break;
        case EatPeelCount:
            spirv14("peel_count");
            if (unsignedArgument("peel_count", uiValue))
                loop->setPeelCount(uiValue);
            break;
        case EatPartialCount:
            spirv14("partial_count");
            if (unsignedArgument("partial_count", uiValue))
                loop->setPartialCount(uiValue);
            break;
        default:
            warn(node->getLoc(), "attribute does not apply to a loop", "", "");
            break;
        }
    }
}


//
// Function attributes
//
void TParseContext::handleFunctionAttributes(const TSourceLoc& loc, const TAttributes& attributes)
{
    for (auto it = attributes.begin(); it != attributes.end(); ++it) {
        if (it->size() > 0) {
            warn(loc, "attribute with arguments not recognized, skipping", "", "");
            continue;
        }

        switch (it->name) {
        case EatSubgroupUniformControlFlow:
            requireExtensions(loc, 1, &E_GL_EXT_subgroup_uniform_control_flow, "attribute");
            intermediate.setSubgroupUniformControlFlow();
            break;
        default:
            warn(loc, "attribute does not apply to a function", "", "");
            break;
        }
    }
}

} // end namespace glslang
