//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_PREPROCESSOR_H_
#define COMPILER_PREPROCESSOR_PREPROCESSOR_H_

#include <cstddef>

#include "GLSLANG/ShaderLang.h"
#include "common/angleutils.h"

namespace angle
{

namespace pp
{

class Diagnostics;
class DirectiveHandler;
struct PreprocessorImpl;
struct Token;

struct PreprocessorSettings final
{
    PreprocessorSettings(ShShaderSpec shaderSpec)
        : maxMacroExpansionDepth(1000), shaderSpec(shaderSpec)
    {}

    PreprocessorSettings(const PreprocessorSettings &other) = default;

    int maxMacroExpansionDepth;
    ShShaderSpec shaderSpec;
};

class Preprocessor : angle::NonCopyable
{
  public:
    Preprocessor(Diagnostics *diagnostics,
                 DirectiveHandler *directiveHandler,
                 const PreprocessorSettings &settings);
    ~Preprocessor();

    // count: specifies the number of elements in the string and length arrays.
    // string: specifies an array of pointers to strings.
    // length: specifies an array of string lengths.
    // If length is NULL, each string is assumed to be null terminated.
    // If length is a value other than NULL, it points to an array containing
    // a string length for each of the corresponding elements of string.
    // Each element in the length array may contain the length of the
    // corresponding string or a value less than 0 to indicate that the string
    // is null terminated.
    bool init(size_t count, const char *const string[], const int length[]);
    // Adds a pre-defined macro.
    void predefineMacro(const char *name, int value);

    void lex(Token *token);

    // Set maximum preprocessor token size
    void setMaxTokenSize(size_t maxTokenSize);

  private:
    PreprocessorImpl *mImpl;
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_PREPROCESSOR_H_
