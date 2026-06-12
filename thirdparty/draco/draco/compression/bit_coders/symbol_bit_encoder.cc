#include "draco/compression/bit_coders/symbol_bit_encoder.h"

#include "draco/compression/entropy/symbol_encoding.h"

namespace draco {

void SymbolBitEncoder::EncodeLeastSignificantBits32(int nbits, uint32_t value) {
  DRACO_DCHECK_LE(1, nbits);
  DRACO_DCHECK_LE(nbits, 32);

  const int discarded_bits = 32 - nbits;
  value <<= discarded_bits;
  value >>= discarded_bits;

  symbols_.push_back(value);
}

void SymbolBitEncoder::EndEncoding(EncoderBuffer *target_buffer) {
  target_buffer->Encode(static_cast<uint32_t>(symbols_.size()));
  EncodeSymbols(symbols_.data(), static_cast<int>(symbols_.size()), 1, nullptr,
                target_buffer);
  Clear();
}

void SymbolBitEncoder::Clear() {
  symbols_.clear();
  symbols_.shrink_to_fit();
}

}  // namespace draco
