//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_BUILTINFUNCTIONEMULATOR_H_
#define COMPILER_TRANSLATOR_BUILTINFUNCTIONEMULATOR_H_

#include "compiler/translator/InfoSink.h"

namespace sh
{

class TIntermNode;
class TFunction;
class TSymbolUniqueId;

using BuiltinQueryFunc = const char *(int);

//
// This class decides which built-in functions need to be replaced with the emulated ones. It can be
// used to work around driver bugs or implement functions that are not natively implemented on a
// specific platform.
//
class BuiltInFunctionEmulator
{
  public:
    BuiltInFunctionEmulator();

    void markBuiltInFunctionsForEmulation(TIntermNode *root);

    void cleanup();

    // "name" gets written as "name_emu".
    static void WriteEmulatedFunctionName(TInfoSinkBase &out, const char *name);

    bool isOutputEmpty() const;

    // Output function emulation definition. This should be before any other shader source.
    void outputEmulatedFunctions(TInfoSinkBase &out) const;

    // Add functions that need to be emulated.
    void addEmulatedFunction(const TSymbolUniqueId &uniqueId,
                             const char *emulatedFunctionDefinition);

    void addEmulatedFunctionWithDependency(const TSymbolUniqueId &dependency,
                                           const TSymbolUniqueId &uniqueId,
                                           const char *emulatedFunctionDefinition);

    void addFunctionMap(BuiltinQueryFunc queryFunc);

  private:
    class BuiltInFunctionEmulationMarker;

    // Records that a function is called by the shader and might need to be emulated. If the
    // function is not in mEmulatedFunctions, this becomes a no-op. Returns true if the function
    // call needs to be replaced with an emulated one.
    bool setFunctionCalled(const TFunction *function);
    bool setFunctionCalled(int uniqueId);

    const char *findEmulatedFunction(int uniqueId) const;

    // Map from function unique id to emulated function definition
    std::map<int, std::string> mEmulatedFunctions;

    // Map from dependent functions to their dependencies. This structure allows each function to
    // have at most one dependency.
    std::map<int, int> mFunctionDependencies;

    // Called function ids
    std::vector<int> mFunctions;

    // Constexpr function tables.
    std::vector<BuiltinQueryFunc *> mQueryFunctions;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_BUILTINFUNCTIONEMULATOR_H_
