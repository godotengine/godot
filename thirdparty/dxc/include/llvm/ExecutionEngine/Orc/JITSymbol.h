//===----------- JITSymbol.h - JIT symbol abstraction -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Abstraction for target process addresses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_JITSYMBOL_H
#define LLVM_EXECUTIONENGINE_ORC_JITSYMBOL_H

#include "llvm/ExecutionEngine/JITSymbolFlags.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <functional>

namespace llvm {
namespace orc {

/// @brief Represents an address in the target process's address space.
typedef uint64_t TargetAddress;

/// @brief Represents a symbol in the JIT.
class JITSymbol : public JITSymbolBase {
public:

  typedef std::function<TargetAddress()> GetAddressFtor;

  /// @brief Create a 'null' symbol that represents failure to find a symbol
  ///        definition.
  JITSymbol(std::nullptr_t)
      : JITSymbolBase(JITSymbolFlags::None), CachedAddr(0) {}

  /// @brief Create a symbol for a definition with a known address.
  JITSymbol(TargetAddress Addr, JITSymbolFlags Flags)
    : JITSymbolBase(Flags), CachedAddr(Addr) {}

  /// @brief Create a symbol for a definition that doesn't have a known address
  ///        yet.
  /// @param GetAddress A functor to materialize a definition (fixing the
  ///        address) on demand.
  ///
  ///   This constructor allows a JIT layer to provide a reference to a symbol
  /// definition without actually materializing the definition up front. The
  /// user can materialize the definition at any time by calling the getAddress
  /// method.
  JITSymbol(GetAddressFtor GetAddress, JITSymbolFlags Flags)
      : JITSymbolBase(Flags), GetAddress(std::move(GetAddress)), CachedAddr(0) {}

  /// @brief Returns true if the symbol exists, false otherwise.
  explicit operator bool() const { return CachedAddr || GetAddress; }

  /// @brief Get the address of the symbol in the target address space. Returns
  ///        '0' if the symbol does not exist.
  TargetAddress getAddress() {
    if (GetAddress) {
      CachedAddr = GetAddress();
      assert(CachedAddr && "Symbol could not be materialized.");
      GetAddress = nullptr;
    }
    return CachedAddr;
  }

private:
  GetAddressFtor GetAddress;
  TargetAddress CachedAddr;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_JITSYMBOL_H
