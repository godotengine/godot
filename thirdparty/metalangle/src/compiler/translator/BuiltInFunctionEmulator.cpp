//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/BuiltInFunctionEmulator.h"
#include "angle_gl.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

class BuiltInFunctionEmulator::BuiltInFunctionEmulationMarker : public TIntermTraverser
{
  public:
    BuiltInFunctionEmulationMarker(BuiltInFunctionEmulator &emulator)
        : TIntermTraverser(true, false, false), mEmulator(emulator)
    {}

    bool visitUnary(Visit visit, TIntermUnary *node) override
    {
        if (node->getFunction())
        {
            bool needToEmulate = mEmulator.setFunctionCalled(node->getFunction());
            if (needToEmulate)
                node->setUseEmulatedFunction();
        }
        return true;
    }

    bool visitAggregate(Visit visit, TIntermAggregate *node) override
    {
        // Here we handle all the built-in functions mapped to ops, not just the ones that are
        // currently identified as problematic.
        if (node->isConstructor() || node->isFunctionCall())
        {
            return true;
        }
        bool needToEmulate = mEmulator.setFunctionCalled(node->getFunction());
        if (needToEmulate)
            node->setUseEmulatedFunction();
        return true;
    }

  private:
    BuiltInFunctionEmulator &mEmulator;
};

BuiltInFunctionEmulator::BuiltInFunctionEmulator() {}

void BuiltInFunctionEmulator::addEmulatedFunction(const TSymbolUniqueId &uniqueId,
                                                  const char *emulatedFunctionDefinition)
{
    mEmulatedFunctions[uniqueId.get()] = std::string(emulatedFunctionDefinition);
}

void BuiltInFunctionEmulator::addEmulatedFunctionWithDependency(
    const TSymbolUniqueId &dependency,
    const TSymbolUniqueId &uniqueId,
    const char *emulatedFunctionDefinition)
{
    mEmulatedFunctions[uniqueId.get()]    = std::string(emulatedFunctionDefinition);
    mFunctionDependencies[uniqueId.get()] = dependency.get();
}

bool BuiltInFunctionEmulator::isOutputEmpty() const
{
    return (mFunctions.size() == 0);
}

void BuiltInFunctionEmulator::outputEmulatedFunctions(TInfoSinkBase &out) const
{
    for (const auto &function : mFunctions)
    {
        const char *body = findEmulatedFunction(function);
        ASSERT(body);
        out << body;
        out << "\n\n";
    }
}

const char *BuiltInFunctionEmulator::findEmulatedFunction(int uniqueId) const
{
    for (const auto &queryFunction : mQueryFunctions)
    {
        const char *result = queryFunction(uniqueId);
        if (result)
        {
            return result;
        }
    }

    const auto &result = mEmulatedFunctions.find(uniqueId);
    if (result != mEmulatedFunctions.end())
    {
        return result->second.c_str();
    }

    return nullptr;
}

bool BuiltInFunctionEmulator::setFunctionCalled(const TFunction *function)
{
    ASSERT(function != nullptr);
    return setFunctionCalled(function->uniqueId().get());
}

bool BuiltInFunctionEmulator::setFunctionCalled(int uniqueId)
{
    if (!findEmulatedFunction(uniqueId))
    {
        return false;
    }

    for (size_t i = 0; i < mFunctions.size(); ++i)
    {
        if (mFunctions[i] == uniqueId)
            return true;
    }
    // If the function depends on another, mark the dependency as called.
    auto dependency = mFunctionDependencies.find(uniqueId);
    if (dependency != mFunctionDependencies.end())
    {
        setFunctionCalled((*dependency).second);
    }
    mFunctions.push_back(uniqueId);
    return true;
}

void BuiltInFunctionEmulator::markBuiltInFunctionsForEmulation(TIntermNode *root)
{
    ASSERT(root);

    if (mEmulatedFunctions.empty() && mQueryFunctions.empty())
        return;

    BuiltInFunctionEmulationMarker marker(*this);
    root->traverse(&marker);
}

void BuiltInFunctionEmulator::cleanup()
{
    mFunctions.clear();
    mFunctionDependencies.clear();
}

void BuiltInFunctionEmulator::addFunctionMap(BuiltinQueryFunc queryFunc)
{
    mQueryFunctions.push_back(queryFunc);
}

// static
void BuiltInFunctionEmulator::WriteEmulatedFunctionName(TInfoSinkBase &out, const char *name)
{
    ASSERT(name[strlen(name) - 1] != '(');
    out << name << "_emu";
}

}  // namespace sh
