//===- llvm/IR/MetadataTracking.h - Metadata tracking ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Low-level functions to enable tracking of metadata that could RAUW.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_METADATATRACKING_H
#define LLVM_IR_METADATATRACKING_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Casting.h"
#include <type_traits>

namespace llvm {

class Metadata;
class MetadataAsValue;

/// \brief API for tracking metadata references through RAUW and deletion.
///
/// Shared API for updating \a Metadata pointers in subclasses that support
/// RAUW.
///
/// This API is not meant to be used directly.  See \a TrackingMDRef for a
/// user-friendly tracking reference.
class MetadataTracking {
public:
  /// \brief Track the reference to metadata.
  ///
  /// Register \c MD with \c *MD, if the subclass supports tracking.  If \c *MD
  /// gets RAUW'ed, \c MD will be updated to the new address.  If \c *MD gets
  /// deleted, \c MD will be set to \c nullptr.
  ///
  /// If tracking isn't supported, \c *MD will not change.
  ///
  /// \return true iff tracking is supported by \c MD.
  static bool track(Metadata *&MD) {
    return track(&MD, *MD, static_cast<Metadata *>(nullptr));
  }

  /// \brief Track the reference to metadata for \a Metadata.
  ///
  /// As \a track(Metadata*&), but with support for calling back to \c Owner to
  /// tell it that its operand changed.  This could trigger \c Owner being
  /// re-uniqued.
  static bool track(void *Ref, Metadata &MD, Metadata &Owner) {
    return track(Ref, MD, &Owner);
  }

  /// \brief Track the reference to metadata for \a MetadataAsValue.
  ///
  /// As \a track(Metadata*&), but with support for calling back to \c Owner to
  /// tell it that its operand changed.  This could trigger \c Owner being
  /// re-uniqued.
  static bool track(void *Ref, Metadata &MD, MetadataAsValue &Owner) {
    return track(Ref, MD, &Owner);
  }

  /// \brief Stop tracking a reference to metadata.
  ///
  /// Stops \c *MD from tracking \c MD.
  static void untrack(Metadata *&MD) { untrack(&MD, *MD); }
  static void untrack(void *Ref, Metadata &MD);

  /// \brief Move tracking from one reference to another.
  ///
  /// Semantically equivalent to \c untrack(MD) followed by \c track(New),
  /// except that ownership callbacks are maintained.
  ///
  /// Note: it is an error if \c *MD does not equal \c New.
  ///
  /// \return true iff tracking is supported by \c MD.
  static bool retrack(Metadata *&MD, Metadata *&New) {
    return retrack(&MD, *MD, &New);
  }
  static bool retrack(void *Ref, Metadata &MD, void *New);

  /// \brief Check whether metadata is replaceable.
  static bool isReplaceable(const Metadata &MD);

  typedef PointerUnion<MetadataAsValue *, Metadata *> OwnerTy;

private:
  /// \brief Track a reference to metadata for an owner.
  ///
  /// Generalized version of tracking.
  static bool track(void *Ref, Metadata &MD, OwnerTy Owner);
};

} // end namespace llvm

#endif
