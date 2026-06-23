// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_COMPRESSION_ENTROPY_SYMBOL_DECODING_H_
#define DRACO_COMPRESSION_ENTROPY_SYMBOL_DECODING_H_

#include "draco/core/decoder_buffer.h"

namespace draco {

// Decodes an array of symbols that was previously encoded with an entropy code.
// Returns false on error.
bool DecodeSymbols(uint32_t num_values, int num_components,
                   DecoderBuffer *src_buffer, uint32_t *out_values);

}  // namespace draco

#endif  // DRACO_COMPRESSION_ENTROPY_SYMBOL_DECODING_H_
