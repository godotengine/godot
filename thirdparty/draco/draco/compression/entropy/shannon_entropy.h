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
#ifndef DRACO_COMPRESSION_ENTROPY_SHANNON_ENTROPY_H_
#define DRACO_COMPRESSION_ENTROPY_SHANNON_ENTROPY_H_

#include <stdint.h>

#include <vector>

namespace draco {

// Computes an approximate Shannon entropy of symbols stored in the provided
// input array |symbols|. The entropy corresponds to the number of bits that is
// required to represent/store all the symbols using an optimal entropy coding
// algorithm. See for example "A mathematical theory of communication" by
// Shannon'48 (http://ieeexplore.ieee.org/document/6773024/).
//
// |max_value| is a required input that define the maximum value in the input
// |symbols| array.
//
// |out_num_unique_symbols| is an optional output argument that stores the
// number of unique symbols contained within the |symbols| array.
// TODO(ostava): This should be renamed or the return value should be changed to
// return the actual entropy and not the number of bits needed to represent the
// input symbols.
int64_t ComputeShannonEntropy(const uint32_t *symbols, int num_symbols,
                              int max_value, int *out_num_unique_symbols);

// Computes the Shannon entropy of |num_values| Boolean entries, where
// |num_true_values| are set to true.
// Returns entropy between 0-1.
double ComputeBinaryShannonEntropy(uint32_t num_values,
                                   uint32_t num_true_values);

// Class that can be used to keep track of the Shannon entropy on streamed data.
// As new symbols are pushed to the tracker, the entropy is automatically
// recomputed. The class also support recomputing the entropy without actually
// pushing the symbols to the tracker through the Peek() method.
class ShannonEntropyTracker {
 public:
  ShannonEntropyTracker();

  // Struct for holding entropy data about the symbols added to the tracker.
  // It can be used to compute the number of bits needed to store the data using
  // the method:
  //   ShannonEntropyTracker::GetNumberOfDataBits(entropy_data);
  // or to compute the approximate size of the frequency table needed by the
  // rans coding using method:
  //   ShannonEntropyTracker::GetNumberOfRAnsTableBits(entropy_data);
  struct EntropyData {
    double entropy_norm;
    int num_values;
    int max_symbol;
    int num_unique_symbols;
    EntropyData()
        : entropy_norm(0.0),
          num_values(0),
          max_symbol(0),
          num_unique_symbols(0) {}
  };

  // Adds new symbols to the tracker and recomputes the entropy accordingly.
  EntropyData Push(const uint32_t *symbols, int num_symbols);

  // Returns new entropy data for the tracker as if |symbols| were added to the
  // tracker without actually changing the status of the tracker.
  EntropyData Peek(const uint32_t *symbols, int num_symbols);

  // Gets the number of bits needed for encoding symbols added to the tracker.
  int64_t GetNumberOfDataBits() const {
    return GetNumberOfDataBits(entropy_data_);
  }

  // Gets the number of bits needed for encoding frequency table using the rans
  // encoder.
  int64_t GetNumberOfRAnsTableBits() const {
    return GetNumberOfRAnsTableBits(entropy_data_);
  }

  // Gets the number of bits needed for encoding given |entropy_data|.
  static int64_t GetNumberOfDataBits(const EntropyData &entropy_data);

  // Gets the number of bits needed for encoding frequency table using the rans
  // encoder for the given |entropy_data|.
  static int64_t GetNumberOfRAnsTableBits(const EntropyData &entropy_data);

 private:
  EntropyData UpdateSymbols(const uint32_t *symbols, int num_symbols,
                            bool push_changes);

  std::vector<int32_t> frequencies_;

  EntropyData entropy_data_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ENTROPY_SHANNON_ENTROPY_H_
