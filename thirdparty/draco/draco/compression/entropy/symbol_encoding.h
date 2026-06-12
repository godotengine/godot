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
#ifndef DRACO_COMPRESSION_ENTROPY_SYMBOL_ENCODING_H_
#define DRACO_COMPRESSION_ENTROPY_SYMBOL_ENCODING_H_

#include "draco/compression/config/compression_shared.h"
#include "draco/core/encoder_buffer.h"
#include "draco/core/options.h"

namespace draco {

// Encodes an array of symbols using an entropy coding. This function
// automatically decides whether to encode the symbol values using bit
// length tags (see EncodeTaggedSymbols), or whether to encode them directly
// (see EncodeRawSymbols). The symbols can be grouped into separate components
// that can be used for better compression. |options| is an optional parameter
// that allows more direct control over various stages of the symbol encoding
// (see below for functions that are used to set valid options).
// Returns false on error.
bool EncodeSymbols(const uint32_t *symbols, int num_values, int num_components,
                   const Options *options, EncoderBuffer *target_buffer);

// Sets an option that forces symbol encoder to use the specified encoding
// method.
void SetSymbolEncodingMethod(Options *options, SymbolCodingMethod method);

// Sets the desired compression level for symbol encoding in range <0, 10> where
// 0 is the worst but fastest compression and 10 is the best but slowest
// compression. If the option is not set, default value of 7 is used.
// Returns false if an invalid level has been set.
bool SetSymbolEncodingCompressionLevel(Options *options, int compression_level);

}  // namespace draco

#endif  // DRACO_COMPRESSION_ENTROPY_SYMBOL_ENCODING_H_
