#include "draco/compression/entropy/shannon_entropy.h"

#include <cmath>
#include <vector>

#include "draco/compression/entropy/rans_symbol_coding.h"

namespace draco {

int64_t ComputeShannonEntropy(const uint32_t *symbols, int num_symbols,
                              int max_value, int *out_num_unique_symbols) {
  // First find frequency of all unique symbols in the input array.
  int num_unique_symbols = 0;
  std::vector<int> symbol_frequencies(max_value + 1, 0);
  for (int i = 0; i < num_symbols; ++i) {
    ++symbol_frequencies[symbols[i]];
  }
  double total_bits = 0;
  double num_symbols_d = num_symbols;
  for (int i = 0; i < max_value + 1; ++i) {
    if (symbol_frequencies[i] > 0) {
      ++num_unique_symbols;
      // Compute Shannon entropy for the symbol.
      // We don't want to use std::log2 here for Android build.
      total_bits +=
          symbol_frequencies[i] *
          log2(static_cast<double>(symbol_frequencies[i]) / num_symbols_d);
    }
  }
  if (out_num_unique_symbols) {
    *out_num_unique_symbols = num_unique_symbols;
  }
  // Entropy is always negative.
  return static_cast<int64_t>(-total_bits);
}

double ComputeBinaryShannonEntropy(uint32_t num_values,
                                   uint32_t num_true_values) {
  if (num_values == 0) {
    return 0;
  }

  // We can exit early if the data set has 0 entropy.
  if (num_true_values == 0 || num_values == num_true_values) {
    return 0;
  }
  const double true_freq =
      static_cast<double>(num_true_values) / static_cast<double>(num_values);
  const double false_freq = 1.0 - true_freq;
  return -(true_freq * std::log2(true_freq) +
           false_freq * std::log2(false_freq));
}

ShannonEntropyTracker::ShannonEntropyTracker() {}

ShannonEntropyTracker::EntropyData ShannonEntropyTracker::Peek(
    const uint32_t *symbols, int num_symbols) {
  return UpdateSymbols(symbols, num_symbols, false);
}

ShannonEntropyTracker::EntropyData ShannonEntropyTracker::Push(
    const uint32_t *symbols, int num_symbols) {
  return UpdateSymbols(symbols, num_symbols, true);
}

ShannonEntropyTracker::EntropyData ShannonEntropyTracker::UpdateSymbols(
    const uint32_t *symbols, int num_symbols, bool push_changes) {
  EntropyData ret_data = entropy_data_;
  ret_data.num_values += num_symbols;
  for (int i = 0; i < num_symbols; ++i) {
    const uint32_t symbol = symbols[i];
    if (frequencies_.size() <= symbol) {
      frequencies_.resize(symbol + 1, 0);
    }

    // Update the entropy of the stream. Note that entropy of |N| values
    // represented by |S| unique symbols is defined as:
    //
    //  entropy = -sum_over_S(symbol_frequency / N * log2(symbol_frequency / N))
    //
    // To avoid the need to recompute the entire sum when new values are added,
    // we can instead update a so called entropy norm that is defined as:
    //
    //  entropy_norm = sum_over_S(symbol_frequency * log2(symbol_frequency))
    //
    // In this case, all we need to do is update entries on the symbols where
    // the frequency actually changed.
    //
    // Note that entropy_norm and entropy can be easily transformed to the
    // actual entropy as:
    //
    //  entropy = log2(N) - entropy_norm / N
    //
    double old_symbol_entropy_norm = 0;
    int &frequency = frequencies_[symbol];
    if (frequency > 1) {
      old_symbol_entropy_norm = frequency * std::log2(frequency);
    } else if (frequency == 0) {
      ret_data.num_unique_symbols++;
      if (symbol > static_cast<uint32_t>(ret_data.max_symbol)) {
        ret_data.max_symbol = symbol;
      }
    }
    frequency++;
    const double new_symbol_entropy_norm = frequency * std::log2(frequency);

    // Update the final entropy.
    ret_data.entropy_norm += new_symbol_entropy_norm - old_symbol_entropy_norm;
  }
  if (push_changes) {
    // Update entropy data of the stream.
    entropy_data_ = ret_data;
  } else {
    // We are only peeking so do not update the stream.
    // Revert changes in the frequency table.
    for (int i = 0; i < num_symbols; ++i) {
      const uint32_t symbol = symbols[i];
      frequencies_[symbol]--;
    }
  }
  return ret_data;
}

int64_t ShannonEntropyTracker::GetNumberOfDataBits(
    const EntropyData &entropy_data) {
  if (entropy_data.num_values < 2) {
    return 0;
  }
  // We need to compute the number of bits required to represent the stream
  // using the entropy norm. Note that:
  //
  //   entropy = log2(num_values) - entropy_norm / num_values
  //
  // and number of bits required for the entropy is: num_values * entropy
  //
  return static_cast<int64_t>(
      ceil(entropy_data.num_values * std::log2(entropy_data.num_values) -
           entropy_data.entropy_norm));
}

int64_t ShannonEntropyTracker::GetNumberOfRAnsTableBits(
    const EntropyData &entropy_data) {
  return ApproximateRAnsFrequencyTableBits(entropy_data.max_symbol + 1,
                                           entropy_data.num_unique_symbols);
}

}  // namespace draco
