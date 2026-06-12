#ifndef DRACO_COMPRESSION_BIT_CODERS_SYMBOL_BIT_DECODER_H_
#define DRACO_COMPRESSION_BIT_CODERS_SYMBOL_BIT_DECODER_H_

#include <algorithm>
#include <vector>

#include "draco/core/decoder_buffer.h"

namespace draco {

// Class for decoding bits using the symbol entropy encoding. Wraps
// |DecodeSymbols|. Note that this uses a symbol-based encoding scheme for
// encoding bits.
class SymbolBitDecoder {
 public:
  // Sets |source_buffer| as the buffer to decode bits from.
  bool StartDecoding(DecoderBuffer *source_buffer);

  // Decode one bit. Returns true if the bit is a 1, otherwise false.
  bool DecodeNextBit();

  // Decode the next |nbits| and return the sequence in |value|. |nbits| must be
  // > 0 and <= 32.
  void DecodeLeastSignificantBits32(int nbits, uint32_t *value);

  void EndDecoding() { Clear(); }

 private:
  void Clear();

  std::vector<uint32_t> symbols_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_BIT_CODERS_SYMBOL_BIT_DECODER_H_
