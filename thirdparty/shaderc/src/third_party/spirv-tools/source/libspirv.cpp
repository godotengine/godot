// Copyright (c) 2016 Google Inc.
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

#include "spirv-tools/libspirv.hpp"

#include <iostream>

#include <string>
#include <utility>
#include <vector>

#include "source/table.h"

namespace spvtools {

Context::Context(spv_target_env env) : context_(spvContextCreate(env)) {}

Context::Context(Context&& other) : context_(other.context_) {
  other.context_ = nullptr;
}

Context& Context::operator=(Context&& other) {
  spvContextDestroy(context_);
  context_ = other.context_;
  other.context_ = nullptr;

  return *this;
}

Context::~Context() { spvContextDestroy(context_); }

void Context::SetMessageConsumer(MessageConsumer consumer) {
  SetContextMessageConsumer(context_, std::move(consumer));
}

spv_context& Context::CContext() { return context_; }

const spv_context& Context::CContext() const { return context_; }

// Structs for holding the data members for SpvTools.
struct SpirvTools::Impl {
  explicit Impl(spv_target_env env) : context(spvContextCreate(env)) {
    // The default consumer in spv_context_t is a null consumer, which provides
    // equivalent functionality (from the user's perspective) as a real consumer
    // does nothing.
  }
  ~Impl() { spvContextDestroy(context); }

  spv_context context;  // C interface context object.
};

SpirvTools::SpirvTools(spv_target_env env) : impl_(new Impl(env)) {}

SpirvTools::~SpirvTools() {}

void SpirvTools::SetMessageConsumer(MessageConsumer consumer) {
  SetContextMessageConsumer(impl_->context, std::move(consumer));
}

bool SpirvTools::Assemble(const std::string& text,
                          std::vector<uint32_t>* binary,
                          uint32_t options) const {
  return Assemble(text.data(), text.size(), binary, options);
}

bool SpirvTools::Assemble(const char* text, const size_t text_size,
                          std::vector<uint32_t>* binary,
                          uint32_t options) const {
  spv_binary spvbinary = nullptr;
  spv_result_t status = spvTextToBinaryWithOptions(
      impl_->context, text, text_size, options, &spvbinary, nullptr);
  if (status == SPV_SUCCESS) {
    binary->assign(spvbinary->code, spvbinary->code + spvbinary->wordCount);
  }
  spvBinaryDestroy(spvbinary);
  return status == SPV_SUCCESS;
}

bool SpirvTools::Disassemble(const std::vector<uint32_t>& binary,
                             std::string* text, uint32_t options) const {
  return Disassemble(binary.data(), binary.size(), text, options);
}

bool SpirvTools::Disassemble(const uint32_t* binary, const size_t binary_size,
                             std::string* text, uint32_t options) const {
  spv_text spvtext = nullptr;
  spv_result_t status = spvBinaryToText(impl_->context, binary, binary_size,
                                        options, &spvtext, nullptr);
  if (status == SPV_SUCCESS) {
    text->assign(spvtext->str, spvtext->str + spvtext->length);
  }
  spvTextDestroy(spvtext);
  return status == SPV_SUCCESS;
}

bool SpirvTools::Validate(const std::vector<uint32_t>& binary) const {
  return Validate(binary.data(), binary.size());
}

bool SpirvTools::Validate(const uint32_t* binary,
                          const size_t binary_size) const {
  return spvValidateBinary(impl_->context, binary, binary_size, nullptr) ==
         SPV_SUCCESS;
}

bool SpirvTools::Validate(const uint32_t* binary, const size_t binary_size,
                          spv_validator_options options) const {
  spv_const_binary_t the_binary{binary, binary_size};
  spv_diagnostic diagnostic = nullptr;
  bool valid = spvValidateWithOptions(impl_->context, options, &the_binary,
                                      &diagnostic) == SPV_SUCCESS;
  if (!valid && impl_->context->consumer) {
    impl_->context->consumer.operator()(
        SPV_MSG_ERROR, nullptr, diagnostic->position, diagnostic->error);
  }
  spvDiagnosticDestroy(diagnostic);
  return valid;
}

bool SpirvTools::IsValid() const { return impl_->context != nullptr; }

}  // namespace spvtools
