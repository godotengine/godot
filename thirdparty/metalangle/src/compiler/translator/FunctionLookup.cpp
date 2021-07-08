//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FunctionLookup.cpp: Used for storing function calls that have not yet been resolved during
// parsing.
//

#include "compiler/translator/FunctionLookup.h"
#include "compiler/translator/ImmutableStringBuilder.h"

namespace sh
{

namespace
{

const char kFunctionMangledNameSeparator = '(';

constexpr const ImmutableString kEmptyName("");

// Helper function for GetMangledNames
// Gets all ordered combinations of elements in list[currentIndex, end]
std::vector<std::vector<int>> GetImplicitConversionCombinations(const std::vector<int> &list)
{
    std::vector<std::vector<int>> target;
    target.push_back(std::vector<int>());

    for (size_t currentIndex = 0; currentIndex < list.size(); currentIndex++)
    {
        size_t prevIterSize = target.size();
        for (size_t copyIndex = 0; copyIndex < prevIterSize; copyIndex++)
        {
            std::vector<int> combination = target[copyIndex];
            combination.push_back(list[currentIndex]);
            target.push_back(combination);
        }
    }

    return target;
}

}  // anonymous namespace

TFunctionLookup::TFunctionLookup(const ImmutableString &name,
                                 const TType *constructorType,
                                 const TSymbol *symbol)
    : mName(name), mConstructorType(constructorType), mThisNode(nullptr), mSymbol(symbol)
{}

// static
TFunctionLookup *TFunctionLookup::CreateConstructor(const TType *type)
{
    ASSERT(type != nullptr);
    return new TFunctionLookup(kEmptyName, type, nullptr);
}

// static
TFunctionLookup *TFunctionLookup::CreateFunctionCall(const ImmutableString &name,
                                                     const TSymbol *symbol)
{
    ASSERT(name != "");
    return new TFunctionLookup(name, nullptr, symbol);
}

const ImmutableString &TFunctionLookup::name() const
{
    return mName;
}

ImmutableString TFunctionLookup::getMangledName() const
{
    return GetMangledName(mName.data(), mArguments);
}

ImmutableString TFunctionLookup::GetMangledName(const char *functionName,
                                                const TIntermSequence &arguments)
{
    std::string newName(functionName);
    newName += kFunctionMangledNameSeparator;

    for (TIntermNode *argument : arguments)
    {
        newName += argument->getAsTyped()->getType().getMangledName();
    }
    return ImmutableString(newName);
}

std::vector<ImmutableString> GetMangledNames(const char *functionName,
                                             const TIntermSequence &arguments)
{
    std::vector<ImmutableString> target;

    std::vector<int> indexes;
    for (int i = 0; i < static_cast<int>(arguments.size()); i++)
    {
        TIntermNode *argument = arguments[i];
        TBasicType argType    = argument->getAsTyped()->getType().getBasicType();
        if (argType == EbtInt || argType == EbtUInt)
        {
            indexes.push_back(i);
        }
    }

    std::vector<std::vector<int>> combinations = GetImplicitConversionCombinations(indexes);
    for (const std::vector<int> &combination : combinations)
    {
        // combination: ordered list of indexes for arguments that should be converted to float
        std::string newName(functionName);
        newName += kFunctionMangledNameSeparator;
        // combination[currentIndex] represents index of next argument to be converted
        int currentIndex = 0;
        for (int i = 0; i < (int)arguments.size(); i++)
        {
            TIntermNode *argument = arguments[i];

            if (currentIndex != static_cast<int>(combination.size()) &&
                combination[currentIndex] == i)
            {
                // Convert
                TType type = argument->getAsTyped()->getType();
                type.setBasicType(EbtFloat);
                newName += type.getMangledName();
                currentIndex++;
            }
            else
            {
                // Don't convert
                newName += argument->getAsTyped()->getType().getMangledName();
            }
        }
        target.push_back(ImmutableString(newName));
    }

    return target;
}

std::vector<ImmutableString> TFunctionLookup::getMangledNamesForImplicitConversions() const
{
    return GetMangledNames(mName.data(), mArguments);
}

bool TFunctionLookup::isConstructor() const
{
    return mConstructorType != nullptr;
}

const TType &TFunctionLookup::constructorType() const
{
    return *mConstructorType;
}

void TFunctionLookup::setThisNode(TIntermTyped *thisNode)
{
    mThisNode = thisNode;
}

TIntermTyped *TFunctionLookup::thisNode() const
{
    return mThisNode;
}

void TFunctionLookup::addArgument(TIntermTyped *argument)
{
    mArguments.push_back(argument);
}

TIntermSequence &TFunctionLookup::arguments()
{
    return mArguments;
}

const TSymbol *TFunctionLookup::symbol() const
{
    return mSymbol;
}

}  // namespace sh
