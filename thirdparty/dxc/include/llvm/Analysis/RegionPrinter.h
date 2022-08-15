//===-- RegionPrinter.h - Region printer external interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines external functions that can be called to explicitly
// instantiate the region printer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_REGIONPRINTER_H
#define LLVM_ANALYSIS_REGIONPRINTER_H

namespace llvm {
  class FunctionPass;
  FunctionPass *createRegionViewerPass();
  FunctionPass *createRegionOnlyViewerPass();
  FunctionPass *createRegionPrinterPass();
  FunctionPass *createRegionOnlyPrinterPass();
} // End llvm namespace

#endif
