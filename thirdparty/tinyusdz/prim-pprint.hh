// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
 
#pragma once

#include <string>
#include <cstdint>

#include "prim-types.hh"

namespace tinyusdz {
namespace prim {

//
// Impelemnted in pprinter.cc at the moment.
//
std::string print_references(const ReferenceList &references, const uint32_t indent);
std::string print_payload(const PayloadList &payload, const uint32_t indent);
std::string print_layeroffset(const LayerOffset &layeroffset, const uint32_t indent);

std::string print_prim(const Prim &prim, const uint32_t indent=0);
std::string print_primspec(const PrimSpec &primspec, const uint32_t indent=0);

} // namespace prim

inline std::string to_string(const Prim &prim) {
  return prim::print_prim(prim);
}

inline std::string to_string(const PrimSpec &primspec) {
  return prim::print_primspec(primspec);
}

} // namespace tinyusdz
