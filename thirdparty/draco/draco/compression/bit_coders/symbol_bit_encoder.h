#ifndef DRACO_COMPRESSION_BIT_CODERS_SYMBOL_BIT_ENCODER_H_
#define DRACO_COMPRESSION_BIT_CODERS_SYMBOL_BIT_ENCODER_H_

#include <algorithm>
#include <vector>

#include "draco/core/encoder_buffer.h"

namespace draco {

// Class for encoding bits using the symbol entropy encoding. Wraps
// |EncodeSymbols|. Note that this uses a symbol-based encoding scheme for
// encoding bits.
class SymbolBitEncoder {
 public:
  // Must be called before any Encode* function is called.
  void StartEncoding() { Clear(); }

  // Encode one bit. If |bit| is true encode a 1, otherwise encode a 0.
  void EncodeBit(bool bit) { EncodeLeastSignificantBits32(1, bit ? 1 : 0); }

  // Encode |nbits| LSBs of |value| as a symbol. |nbits| must be > 0 and <= 32.
  void EncodeLeastSignificantBits32(int nbits, uint32_t value);

  // Ends the bit encoding and stores the result into the target_buffer.
  void EndEncoding(EncoderBuffer *target_buffer);

 private:
  void Clear();

  std::vector<uint32_t> symbols_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_BIT_CODERS_SYMBOL_BIT_ENCODER_H_
