// Copyright 2010 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Original author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// dwarf2diehandler.cc: Implement the dwarf2reader::DieDispatcher class.
// See dwarf2diehandler.h for details.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <assert.h>
#include <stdint.h>

#include <string>

#include "common/dwarf/dwarf2diehandler.h"
#include "common/using_std_string.h"

namespace google_breakpad {

DIEDispatcher::~DIEDispatcher() {
  while (!die_handlers_.empty()) {
    HandlerStack& entry = die_handlers_.top();
    if (entry.handler_ != root_handler_)
      delete entry.handler_;
    die_handlers_.pop();
  }
}

bool DIEDispatcher::StartCompilationUnit(uint64_t offset, uint8_t address_size,
                                         uint8_t offset_size, uint64_t cu_length,
                                         uint8_t dwarf_version) {
  return root_handler_->StartCompilationUnit(offset, address_size,
                                             offset_size, cu_length,
                                             dwarf_version);
}

bool DIEDispatcher::StartDIE(uint64_t offset, enum DwarfTag tag) {
  // The stack entry for the parent of this DIE, if there is one.
  HandlerStack* parent = die_handlers_.empty() ? NULL : &die_handlers_.top();

  // Does this call indicate that we're done receiving the parent's
  // attributes' values?  If so, call its EndAttributes member function.
  if (parent && parent->handler_ && !parent->reported_attributes_end_) {
    parent->reported_attributes_end_ = true;
    if (!parent->handler_->EndAttributes()) {
      // Finish off this handler now. and edit *PARENT to indicate that
      // we don't want to visit any of the children.
      parent->handler_->Finish();
      if (parent->handler_ != root_handler_)
        delete parent->handler_;
      parent->handler_ = NULL;
      return false;
    }
  }

  // Find a handler for this DIE.
  DIEHandler* handler;
  if (parent) {
    if (parent->handler_)
      // Ask the parent to find a handler.
      handler = parent->handler_->FindChildHandler(offset, tag);
    else
      // No parent handler means we're not interested in any of our
      // children.
      handler = NULL;
  } else {
    // This is the root DIE.  For a non-root DIE, the parent's handler
    // decides whether to visit it, but the root DIE has no parent
    // handler, so we have a special method on the root DIE handler
    // itself to decide.
    if (root_handler_->StartRootDIE(offset, tag))
      handler = root_handler_;
    else
      handler = NULL;
  }

  // Push a handler stack entry for this new handler. As an
  // optimization, we don't push NULL-handler entries on top of other
  // NULL-handler entries; we just let the oldest such entry stand for
  // the whole subtree.
  if (handler || !parent || parent->handler_) {
    HandlerStack entry;
    entry.offset_ = offset;
    entry.handler_ = handler;
    entry.reported_attributes_end_ = false;
    die_handlers_.push(entry);
  }

  return handler != NULL;
}

void DIEDispatcher::EndDIE(uint64_t offset) {
  assert(!die_handlers_.empty());
  HandlerStack* entry = &die_handlers_.top();
  if (entry->handler_) {
    // This entry had better be the handler for this DIE.
    assert(entry->offset_ == offset);
    // If a DIE has no children, this EndDIE call indicates that we're
    // done receiving its attributes' values.
    if (!entry->reported_attributes_end_)
      entry->handler_->EndAttributes(); // Ignore return value: no children.
    entry->handler_->Finish();
    if (entry->handler_ != root_handler_)
      delete entry->handler_;
  } else {
    // If this DIE is within a tree we're ignoring, then don't pop the
    // handler stack: that entry stands for the whole tree.
    if (entry->offset_ != offset)
      return;
  }
  die_handlers_.pop();
}

void DIEDispatcher::ProcessAttributeUnsigned(uint64_t offset,
                                             enum DwarfAttribute attr,
                                             enum DwarfForm form,
                                             uint64_t data) {
  HandlerStack& current = die_handlers_.top();
  // This had better be an attribute of the DIE we were meant to handle.
  assert(offset == current.offset_);
  current.handler_->ProcessAttributeUnsigned(attr, form, data);
}

void DIEDispatcher::ProcessAttributeSigned(uint64_t offset,
                                           enum DwarfAttribute attr,
                                           enum DwarfForm form,
                                           int64_t data) {
  HandlerStack& current = die_handlers_.top();
  // This had better be an attribute of the DIE we were meant to handle.
  assert(offset == current.offset_);
  current.handler_->ProcessAttributeSigned(attr, form, data);
}

void DIEDispatcher::ProcessAttributeReference(uint64_t offset,
                                              enum DwarfAttribute attr,
                                              enum DwarfForm form,
                                              uint64_t data) {
  HandlerStack& current = die_handlers_.top();
  // This had better be an attribute of the DIE we were meant to handle.
  assert(offset == current.offset_);
  current.handler_->ProcessAttributeReference(attr, form, data);
}

void DIEDispatcher::ProcessAttributeBuffer(uint64_t offset,
                                           enum DwarfAttribute attr,
                                           enum DwarfForm form,
                                           const uint8_t* data,
                                           uint64_t len) {
  HandlerStack& current = die_handlers_.top();
  // This had better be an attribute of the DIE we were meant to handle.
  assert(offset == current.offset_);
  current.handler_->ProcessAttributeBuffer(attr, form, data, len);
}

void DIEDispatcher::ProcessAttributeString(uint64_t offset,
                                           enum DwarfAttribute attr,
                                           enum DwarfForm form,
                                           const string& data) {
  HandlerStack& current = die_handlers_.top();
  // This had better be an attribute of the DIE we were meant to handle.
  assert(offset == current.offset_);
  current.handler_->ProcessAttributeString(attr, form, data);
}

void DIEDispatcher::ProcessAttributeSignature(uint64_t offset,
                                              enum DwarfAttribute attr,
                                              enum DwarfForm form,
                                              uint64_t signature) {
  HandlerStack& current = die_handlers_.top();
  // This had better be an attribute of the DIE we were meant to handle.
  assert(offset == current.offset_);
  current.handler_->ProcessAttributeSignature(attr, form, signature);
}

} // namespace google_breakpad
