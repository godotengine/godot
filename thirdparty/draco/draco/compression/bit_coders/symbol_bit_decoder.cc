#include "draco/compression/bit_coders/symbol_bit_decoder.h"

#include "draco/compression/entropy/symbol_decoding.h"

namespace draco {

bool SymbolBitDecoder::StartDecoding(DecoderBuffer *source_buffer) {
  uint32_t size;
  if (!source_buffer->Decode(&size)) {
    return false;
  }

  symbols_.resize(size);
  if (!DecodeSymbols(size, 1, source_buffer, symbols_.data())) {
    return false;
  }
  std::reverse(symbols_.begin(), symbols_.end());
  return true;
}

bool SymbolBitDecoder::DecodeNextBit() {
  uint32_t symbol;
  DecodeLeastSignificantBits32(1, &symbol);
  DRACO_DCHECK(symbol == 0 || symbol == 1);
  return symbol == 1;
}

void SymbolBitDecoder::DecodeLeastSignificantBits32(int nbits,
                                                    uint32_t *value) {
  DRACO_DCHECK_LE(1, nbits);
  DRACO_DCHECK_LE(nbits, 32);
  DRACO_DCHECK_NE(value, nullptr);
  // Testing: check to make sure there is something to decode.
  DRACO_DCHECK_GT(symbols_.size(), 0);

  (*value) = symbols_.back();
  symbols_.pop_back();

  const int discarded_bits = 32 - nbits;
  (*value) <<= discarded_bits;
  (*value) >>= discarded_bits;
}

void SymbolBitDecoder::Clear() {
  symbols_.clear();
  symbols_.shrink_to_fit();
}

}  // namespace draco
