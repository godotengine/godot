// -*- mode: c++ -*-

// Copyright (c) 2010 Google Inc. All Rights Reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

// dwarf2reader::CompilationUnit is a simple and direct parser for
// DWARF data, but its handler interface is not convenient to use.  In
// particular:
//
// - CompilationUnit calls Dwarf2Handler's member functions to report
//   every attribute's value, regardless of what sort of DIE it is.
//   As a result, the ProcessAttributeX functions end up looking like
//   this:
//
//     switch (parent_die_tag) {
//       case DW_TAG_x:
//         switch (attribute_name) {
//           case DW_AT_y:
//             handle attribute y of DIE type x
//           ...
//         } break;
//       ...
//     } 
//
//   In C++ it's much nicer to use virtual function dispatch to find
//   the right code for a given case than to switch on the DIE tag
//   like this.
//
// - Processing different kinds of DIEs requires different sets of
//   data: lexical block DIEs have start and end addresses, but struct
//   type DIEs don't.  It would be nice to be able to have separate
//   handler classes for separate kinds of DIEs, each with the members
//   appropriate to its role, instead of having one handler class that
//   needs to hold data for every DIE type.
//
// - There should be a separate instance of the appropriate handler
//   class for each DIE, instead of a single object with tables
//   tracking all the dies in the compilation unit.
//
// - It's not convenient to take some action after all a DIE's
//   attributes have been seen, but before visiting any of its
//   children.  The only indication you have that a DIE's attribute
//   list is complete is that you get either a StartDIE or an EndDIE
//   call.
//
// - It's not convenient to make use of the tree structure of the
//   DIEs.  Skipping all the children of a given die requires
//   maintaining state and returning false from StartDIE until we get
//   an EndDIE call with the appropriate offset.
//
// This interface tries to take care of all that.  (You're shocked, I'm sure.)
//
// Using the classes here, you provide an initial handler for the root
// DIE of the compilation unit.  Each handler receives its DIE's
// attributes, and provides fresh handler objects for children of
// interest, if any.  The three classes are:
//
// - DIEHandler: the base class for your DIE-type-specific handler
//   classes.
//
// - RootDIEHandler: derived from DIEHandler, the base class for your
//   root DIE handler class.
//
// - DIEDispatcher: derived from Dwarf2Handler, an instance of this
//   invokes your DIE-type-specific handler objects.
//
// In detail:
//
// - Define handler classes specialized for the DIE types you're
//   interested in.  These handler classes must inherit from
//   DIEHandler.  Thus:
//
//     class My_DW_TAG_X_Handler: public DIEHandler { ... };
//     class My_DW_TAG_Y_Handler: public DIEHandler { ... };
//
//   DIEHandler subclasses needn't correspond exactly to single DIE
//   types, as shown here; the point is that you can have several
//   different classes appropriate to different kinds of DIEs.
//
// - In particular, define a handler class for the compilation
//   unit's root DIE, that inherits from RootDIEHandler:
//
//     class My_DW_TAG_compile_unit_Handler: public RootDIEHandler { ... };
//
//   RootDIEHandler inherits from DIEHandler, adding a few additional
//   member functions for examining the compilation unit as a whole,
//   and other quirks of rootness.
//
// - Then, create a DIEDispatcher instance, passing it an instance of
//   your root DIE handler class, and use that DIEDispatcher as the
//   dwarf2reader::CompilationUnit's handler:
//
//     My_DW_TAG_compile_unit_Handler root_die_handler(...);
//     DIEDispatcher die_dispatcher(&root_die_handler);
//     CompilationUnit reader(sections, offset, bytereader, &die_dispatcher);
//
//   Here, 'die_dispatcher' acts as a shim between 'reader' and the
//   various DIE-specific handlers you have defined.
//
// - When you call reader.Start(), die_dispatcher behaves as follows,
//   starting with your root die handler and the compilation unit's
//   root DIE:
//
//   - It calls the handler's ProcessAttributeX member functions for
//     each of the DIE's attributes.
//
//   - It calls the handler's EndAttributes member function.  This
//     should return true if any of the DIE's children should be
//     visited, in which case:
//
//     - For each of the DIE's children, die_dispatcher calls the
//       DIE's handler's FindChildHandler member function.  If that
//       returns a pointer to a DIEHandler instance, then
//       die_dispatcher uses that handler to process the child, using
//       this procedure recursively.  Alternatively, if
//       FindChildHandler returns NULL, die_dispatcher ignores that
//       child and its descendants.
// 
//   - When die_dispatcher has finished processing all the DIE's
//     children, it invokes the handler's Finish() member function,
//     and destroys the handler.  (As a special case, it doesn't
//     destroy the root DIE handler.)
// 
// This allows the code for handling a particular kind of DIE to be
// gathered together in a single class, makes it easy to skip all the
// children or individual children of a particular DIE, and provides
// appropriate parental context for each die.

#ifndef COMMON_DWARF_DWARF2DIEHANDLER_H__
#define COMMON_DWARF_DWARF2DIEHANDLER_H__

#include <stdint.h>

#include <stack>
#include <string>

#include "common/dwarf/types.h"
#include "common/dwarf/dwarf2enums.h"
#include "common/dwarf/dwarf2reader.h"
#include "common/using_std_string.h"

namespace google_breakpad {

// A base class for handlers for specific DIE types.  The series of
// calls made on a DIE handler is as follows:
//
// - for each attribute of the DIE:
//   - ProcessAttributeX()
// - EndAttributes()
// - if that returned true, then for each child:
//   - FindChildHandler()
//   - if that returns a non-NULL pointer to a new handler:
//     - recurse, with the new handler and the child die
// - Finish()
// - destruction
class DIEHandler {
 public:
  DIEHandler() { }
  virtual ~DIEHandler() { }

  // When we visit a DIE, we first use these member functions to
  // report the DIE's attributes and their values.  These have the
  // same restrictions as the corresponding member functions of
  // dwarf2reader::Dwarf2Handler.
  //
  // Since DWARF does not specify in what order attributes must
  // appear, avoid making decisions in these functions that would be
  // affected by the presence of other attributes. The EndAttributes
  // function is a more appropriate place for such work, as all the
  // DIE's attributes have been seen at that point.
  //
  // The default definitions ignore the values they are passed.
  virtual void ProcessAttributeUnsigned(enum DwarfAttribute attr,
                                        enum DwarfForm form,
                                        uint64_t data) { }
  virtual void ProcessAttributeSigned(enum DwarfAttribute attr,
                                      enum DwarfForm form,
                                      int64_t data) { }
  virtual void ProcessAttributeReference(enum DwarfAttribute attr,
                                         enum DwarfForm form,
                                         uint64_t data) { }
  virtual void ProcessAttributeBuffer(enum DwarfAttribute attr,
                                      enum DwarfForm form,
                                      const uint8_t* data,
                                      uint64_t len) { }
  virtual void ProcessAttributeString(enum DwarfAttribute attr,
                                      enum DwarfForm form,
                                      const string& data) { }
  virtual void ProcessAttributeSignature(enum DwarfAttribute attr,
                                         enum DwarfForm form,
                                         uint64_t signture) { }

  // Once we have reported all the DIE's attributes' values, we call
  // this member function.  If it returns false, we skip all the DIE's
  // children.  If it returns true, we call FindChildHandler on each
  // child.  If that returns a handler object, we use that to visit
  // the child; otherwise, we skip the child.
  //
  // This is a good place to make decisions that depend on more than
  // one attribute. DWARF does not specify in what order attributes
  // must appear, so only when the EndAttributes function is called
  // does the handler have a complete picture of the DIE's attributes.
  //
  // The default definition elects to ignore the DIE's children.
  // You'll need to override this if you override FindChildHandler,
  // but at least the default behavior isn't to pass the children to
  // FindChildHandler, which then ignores them all.
  virtual bool EndAttributes() { return false; }

  // If EndAttributes returns true to indicate that some of the DIE's
  // children might be of interest, then we apply this function to
  // each of the DIE's children.  If it returns a handler object, then
  // we use that to visit the child DIE.  If it returns NULL, we skip
  // that child DIE (and all its descendants).
  //
  // OFFSET is the offset of the child; TAG indicates what kind of DIE
  // it is.
  //
  // The default definition skips all children.
  virtual DIEHandler* FindChildHandler(uint64_t offset, enum DwarfTag tag) {
    return NULL;
  }

  // When we are done processing a DIE, we call this member function.
  // This happens after the EndAttributes call, all FindChildHandler
  // calls (if any), and all operations on the children themselves (if
  // any). We call Finish on every handler --- even if EndAttributes
  // returns false.
  virtual void Finish() { };
};

// A subclass of DIEHandler, with additional kludges for handling the
// compilation unit's root die.
class RootDIEHandler : public DIEHandler {
 public:
  bool handle_inline;

  explicit RootDIEHandler(bool handle_inline = false)
      : handle_inline(handle_inline) {}
  virtual ~RootDIEHandler() {}

  // We pass the values reported via Dwarf2Handler::StartCompilationUnit
  // to this member function, and skip the entire compilation unit if it
  // returns false.  So the root DIE handler is actually also
  // responsible for handling the compilation unit metadata.
  // The default definition always visits the compilation unit.
  virtual bool StartCompilationUnit(uint64_t offset, uint8_t address_size,
                                    uint8_t offset_size, uint64_t cu_length,
                                    uint8_t dwarf_version) { return true; }

  // For the root DIE handler only, we pass the offset, tag and
  // attributes of the compilation unit's root DIE.  This is the only
  // way the root DIE handler can find the root DIE's tag.  If this
  // function returns true, we will visit the root DIE using the usual
  // DIEHandler methods; otherwise, we skip the entire compilation
  // unit.
  //
  // The default definition elects to visit the root DIE.
  virtual bool StartRootDIE(uint64_t offset, enum DwarfTag tag) { return true; }
};

class DIEDispatcher: public Dwarf2Handler {
 public:
  // Create a Dwarf2Handler which uses ROOT_HANDLER as the handler for
  // the compilation unit's root die, as described for the DIEHandler
  // class.
  DIEDispatcher(RootDIEHandler* root_handler) : root_handler_(root_handler) { }
  // Destroying a DIEDispatcher destroys all active handler objects
  // except the root handler.
  ~DIEDispatcher();
  bool StartCompilationUnit(uint64_t offset, uint8_t address_size,
                            uint8_t offset_size, uint64_t cu_length,
                            uint8_t dwarf_version);
  bool StartDIE(uint64_t offset, enum DwarfTag tag);
  void ProcessAttributeUnsigned(uint64_t offset,
                                enum DwarfAttribute attr,
                                enum DwarfForm form,
                                uint64_t data);
  void ProcessAttributeSigned(uint64_t offset,
                              enum DwarfAttribute attr,
                              enum DwarfForm form,
                              int64_t data);
  void ProcessAttributeReference(uint64_t offset,
                                 enum DwarfAttribute attr,
                                 enum DwarfForm form,
                                 uint64_t data);
  void ProcessAttributeBuffer(uint64_t offset,
                              enum DwarfAttribute attr,
                              enum DwarfForm form,
                              const uint8_t* data,
                              uint64_t len);
  void ProcessAttributeString(uint64_t offset,
                              enum DwarfAttribute attr,
                              enum DwarfForm form,
                              const string& data);
  void ProcessAttributeSignature(uint64_t offset,
                                 enum DwarfAttribute attr,
                                 enum DwarfForm form,
                                 uint64_t signature);
  void EndDIE(uint64_t offset);

 private:

  // The type of a handler stack entry.  This includes some fields
  // which don't really need to be on the stack --- they could just be
  // single data members of DIEDispatcher --- but putting them here
  // makes it easier to see that the code is correct.
  struct HandlerStack {
    // The offset of the DIE for this handler stack entry.
    uint64_t offset_;

    // The handler object interested in this DIE's attributes and
    // children.  If NULL, we're not interested in either.
    DIEHandler* handler_;

    // Have we reported the end of this DIE's attributes to the handler?
    bool reported_attributes_end_;
  };

  // Stack of DIE attribute handlers.  At StartDIE(D), the top of the
  // stack is the handler of D's parent, whom we may ask for a handler
  // for D itself.  At EndDIE(D), the top of the stack is D's handler.
  // Special cases:
  //
  // - Before we've seen the compilation unit's root DIE, the stack is
  //   empty; we'll call root_handler_'s special member functions, and
  //   perhaps push root_handler_ on the stack to look at the root's
  //   immediate children.
  //
  // - When we decide to ignore a subtree, we only push an entry on
  //   the stack for the root of the tree being ignored, rather than
  //   pushing lots of stack entries with handler_ set to NULL.
  std::stack<HandlerStack> die_handlers_;

  // The root handler.  We don't push it on die_handlers_ until we
  // actually get the StartDIE call for the root.
  RootDIEHandler* root_handler_;
};

} // namespace google_breakpad
#endif  // COMMON_DWARF_DWARF2DIEHANDLER_H__
