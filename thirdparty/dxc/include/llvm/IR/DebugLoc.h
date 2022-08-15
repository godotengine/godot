//===- DebugLoc.h - Debug Location Information ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of light weight data structures used
// to describe and track debug location information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGLOC_H
#define LLVM_IR_DEBUGLOC_H

#include "llvm/IR/TrackingMDRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

  class LLVMContext;
  class raw_ostream;
  class DILocation;

  /// \brief A debug info location.
  ///
  /// This class is a wrapper around a tracking reference to an \a DILocation
  /// pointer.
  ///
  /// To avoid extra includes, \a DebugLoc doubles the \a DILocation API with a
  /// one based on relatively opaque \a MDNode pointers.
  class DebugLoc {
    TrackingMDNodeRef Loc;

  public:
    DebugLoc() {}
    DebugLoc(DebugLoc &&X) : Loc(std::move(X.Loc)) {}
    DebugLoc(const DebugLoc &X) : Loc(X.Loc) {}
    DebugLoc &operator=(DebugLoc &&X) {
      Loc = std::move(X.Loc);
      return *this;
    }
    DebugLoc &operator=(const DebugLoc &X) {
      Loc = X.Loc;
      return *this;
    }

    /// \brief Construct from an \a DILocation.
    DebugLoc(const DILocation *L);

    /// \brief Construct from an \a MDNode.
    ///
    /// Note: if \c N is not an \a DILocation, a verifier check will fail, and
    /// accessors will crash.  However, construction from other nodes is
    /// supported in order to handle forward references when reading textual
    /// IR.
    explicit DebugLoc(const MDNode *N);

    /// \brief Get the underlying \a DILocation.
    ///
    /// \pre !*this or \c isa<DILocation>(getAsMDNode()).
    /// @{
    DILocation *get() const;
    operator DILocation *() const { return get(); }
    DILocation *operator->() const { return get(); }
    DILocation &operator*() const { return *get(); }
    /// @}

    /// \brief Check for null.
    ///
    /// Check for null in a way that is safe with broken debug info.  Unlike
    /// the conversion to \c DILocation, this doesn't require that \c Loc is of
    /// the right type.  Important for cases like \a llvm::StripDebugInfo() and
    /// \a Instruction::hasMetadata().
    explicit operator bool() const { return Loc; }

    /// \brief Check whether this has a trivial destructor.
    bool hasTrivialDestructor() const { return Loc.hasTrivialDestructor(); }

    /// \brief Create a new DebugLoc.
    ///
    /// Create a new DebugLoc at the specified line/col and scope/inline.  This
    /// forwards to \a DILocation::get().
    ///
    /// If \c !Scope, returns a default-constructed \a DebugLoc.
    ///
    /// FIXME: Remove this.  Users should use DILocation::get().
    static DebugLoc get(unsigned Line, unsigned Col, const MDNode *Scope,
                        const MDNode *InlinedAt = nullptr);

    unsigned getLine() const;
    unsigned getCol() const;
    MDNode *getScope() const;
    DILocation *getInlinedAt() const;

    /// \brief Get the fully inlined-at scope for a DebugLoc.
    ///
    /// Gets the inlined-at scope for a DebugLoc.
    MDNode *getInlinedAtScope() const;

    /// \brief Find the debug info location for the start of the function.
    ///
    /// Walk up the scope chain of given debug loc and find line number info
    /// for the function.
    ///
    /// FIXME: Remove this.  Users should use DILocation/DILocalScope API to
    /// find the subprogram, and then DILocation::get().
    DebugLoc getFnDebugLoc() const;

    /// \brief Return \c this as a bar \a MDNode.
    MDNode *getAsMDNode() const { return Loc; }

    bool operator==(const DebugLoc &DL) const { return Loc == DL.Loc; }
    bool operator!=(const DebugLoc &DL) const { return Loc != DL.Loc; }

    void dump() const;

    /// \brief prints source location /path/to/file.exe:line:col @[inlined at]
    void print(raw_ostream &OS) const;
  };

} // end namespace llvm

#endif /* LLVM_SUPPORT_DEBUGLOC_H */
