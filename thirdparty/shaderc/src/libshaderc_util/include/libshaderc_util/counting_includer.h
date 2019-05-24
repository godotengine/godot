// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#ifndef LIBSHADERC_UTIL_COUNTING_INCLUDER_H
#define LIBSHADERC_UTIL_COUNTING_INCLUDER_H

#include <atomic>

#include "glslang/Public/ShaderLang.h"

#include "libshaderc_util/mutex.h"

namespace shaderc_util {

// An Includer that counts how many #include directives it saw.
// Inclusions are internally serialized, but releasing a previous result
// can occur concurrently.
class CountingIncluder : public glslang::TShader::Includer {
 public:
  // Done as .store(0) instead of in the initializer list for the following
  // reasons:
  // Clang > 3.6 will complain about it if it is written as ({0}).
  // VS2013 fails if it is written as {0}.
  // G++-4.8 does not correctly support std::atomic_init.
  CountingIncluder() {
    num_include_directives_.store(0);
  }

  enum class IncludeType {
    System,  // Only do < > include search
    Local,   // Only do " " include search
  };

  // Resolves an include request for a source by name, type, and name of the
  // requesting source.  For the semantics of the result, see the base class.
  // Also increments num_include_directives and returns the results of
  // include_delegate(filename).  Subclasses should override include_delegate()
  // instead of this method.  Inclusions are serialized.
  glslang::TShader::Includer::IncludeResult* includeSystem(
      const char* requested_source, const char* requesting_source,
      size_t include_depth) final {
    ++num_include_directives_;
    include_mutex_.lock();
    auto result = include_delegate(requested_source, requesting_source,
                                   IncludeType::System, include_depth);
    include_mutex_.unlock();
    return result;
  }

  // Like includeSystem, but for "local" include search.
  glslang::TShader::Includer::IncludeResult* includeLocal(
      const char* requested_source, const char* requesting_source,
      size_t include_depth) final {
    ++num_include_directives_;
    include_mutex_.lock();
    auto result = include_delegate(requested_source, requesting_source,
                                   IncludeType::Local, include_depth);
    include_mutex_.unlock();
    return result;
  }

  // Releases the given IncludeResult.
  void releaseInclude(glslang::TShader::Includer::IncludeResult* result) final {
    release_delegate(result);
  }

  int num_include_directives() const { return num_include_directives_.load(); }

 private:

  // Invoked by this class to provide results to
  // glslang::TShader::Includer::include.
  virtual glslang::TShader::Includer::IncludeResult* include_delegate(
      const char* requested_source, const char* requesting_source,
      IncludeType type, size_t include_depth) = 0;

  // Release the given IncludeResult.
  virtual void release_delegate(
      glslang::TShader::Includer::IncludeResult* result) = 0;

  // The number of #include directive encountered.
  std::atomic_int num_include_directives_;

  // A mutex to protect against concurrent inclusions.  We can't trust
  // our delegates to be safe for concurrent inclusions.
  shaderc_util::mutex include_mutex_;
};
}

#endif  // LIBSHADERC_UTIL_COUNTING_INCLUDER_H
