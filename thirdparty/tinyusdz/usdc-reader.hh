// SPDX-License-Identifier: Apache 2.0
// Copyright 2020-2022 Syoyo Fujita.
// Copyright 2023-Present Light Transport Entertainment Inc.
//
#pragma once

#include "stream-reader.hh"
#include "tinyusdz.hh"

namespace tinyusdz {
namespace usdc {

///
/// USDC(Crate) reader
///

struct USDCReaderConfig {
  int32_t numThreads = -1; // -1 = use system's # of threads
  uint32_t kMaxPrimNestLevel = 256;
  uint32_t kMaxFieldValuePairs = 4096;
  uint32_t kMaxTokenLength = 4096; // Max length of `token`
  uint32_t kMaxStringLength = 1024*1024*64; // Max length of `string` data
  uint32_t kMaxElementSize = 8192; // Max allowed value for `elementSize`
  size_t kMaxAllowedMemoryInMB = 1024*16; //Max allowed memory usage in [mb]

  bool allow_unknown_prims = true;
  bool allow_unknown_apiSchemas = true;

  bool strict_allowedToken_check = false;
};

class USDCReader {
 public:
  USDCReader(StreamReader *sr,
             const USDCReaderConfig &config = USDCReaderConfig());
  ~USDCReader();

  void set_reader_config(const USDCReaderConfig &config);
  const USDCReaderConfig get_reader_config() const;

  bool ReadUSDC();

  bool ReconstructStage(Stage *stage);

  // For composition.
  bool get_as_layer(Layer *layer);

  // Approximated memory usage in [mb]
  size_t GetMemoryUsage() const;

  std::string GetError();
  std::string GetWarning();

 private:
  class Impl;
  Impl *impl_{};
};

}  // namespace usdc
}  // namespace tinyusdz
