//===- MetadataTracking.cpp - Implement metadata tracking -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Metadata tracking.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/MetadataTracking.h"
#include "llvm/IR/Metadata.h"

using namespace llvm;

ReplaceableMetadataImpl *ReplaceableMetadataImpl::get(Metadata &MD) {
  if (auto *N = dyn_cast<MDNode>(&MD))
    return N->Context.getReplaceableUses();
  return dyn_cast<ValueAsMetadata>(&MD);
}

bool MetadataTracking::track(void *Ref, Metadata &MD, OwnerTy Owner) {
  assert(Ref && "Expected live reference");
  assert((Owner || *static_cast<Metadata **>(Ref) == &MD) &&
         "Reference without owner must be direct");
  if (auto *R = ReplaceableMetadataImpl::get(MD)) {
    R->addRef(Ref, Owner);
    return true;
  }
  return false;
}

void MetadataTracking::untrack(void *Ref, Metadata &MD) {
  assert(Ref && "Expected live reference");
  if (auto *R = ReplaceableMetadataImpl::get(MD))
    R->dropRef(Ref);
}

bool MetadataTracking::retrack(void *Ref, Metadata &MD, void *New) {
  assert(Ref && "Expected live reference");
  assert(New && "Expected live reference");
  assert(Ref != New && "Expected change");
  if (auto *R = ReplaceableMetadataImpl::get(MD)) {
    R->moveRef(Ref, New, MD);
    return true;
  }
  return false;
}

bool MetadataTracking::isReplaceable(const Metadata &MD) {
  return ReplaceableMetadataImpl::get(const_cast<Metadata &>(MD));
}
