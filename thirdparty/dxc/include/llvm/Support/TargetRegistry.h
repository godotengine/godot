//===-- Support/TargetRegistry.h - Target Registration ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the TargetRegistry interface, which tools can use to access
// the appropriate target specific classes (TargetMachine, AsmPrinter, etc.)
// which have been registered.
//
// Target specific class implementations should register themselves using the
// appropriate TargetRegistry interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TARGETREGISTRY_H
#define LLVM_SUPPORT_TARGETREGISTRY_H

#include "llvm-c/Disassembler.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FormattedStream.h"
#include <cassert>
#include <memory>
#include <string>

namespace llvm {
class AsmPrinter;
class MCAsmBackend;
class MCAsmInfo;
class MCAsmParser;
class MCCodeEmitter;
class MCCodeGenInfo;
class MCContext;
class MCDisassembler;
class MCInstrAnalysis;
class MCInstPrinter;
class MCInstrInfo;
class MCRegisterInfo;
class MCStreamer;
class MCSubtargetInfo;
class MCSymbolizer;
class MCRelocationInfo;
class MCTargetAsmParser;
class MCTargetOptions;
class MCTargetStreamer;
class TargetMachine;
class TargetOptions;
class raw_ostream;
class raw_pwrite_stream;
class formatted_raw_ostream;

MCStreamer *createNullStreamer(MCContext &Ctx);
MCStreamer *createAsmStreamer(MCContext &Ctx,
                              std::unique_ptr<formatted_raw_ostream> OS,
                              bool isVerboseAsm, bool useDwarfDirectory,
                              MCInstPrinter *InstPrint, MCCodeEmitter *CE,
                              MCAsmBackend *TAB, bool ShowInst);

/// Takes ownership of \p TAB and \p CE.
MCStreamer *createELFStreamer(MCContext &Ctx, MCAsmBackend &TAB,
                              raw_pwrite_stream &OS, MCCodeEmitter *CE,
                              bool RelaxAll);
MCStreamer *createMachOStreamer(MCContext &Ctx, MCAsmBackend &TAB,
                                raw_pwrite_stream &OS, MCCodeEmitter *CE,
                                bool RelaxAll, bool DWARFMustBeAtTheEnd,
                                bool LabelSections = false);

MCRelocationInfo *createMCRelocationInfo(const Triple &TT, MCContext &Ctx);

MCSymbolizer *createMCSymbolizer(const Triple &TT, LLVMOpInfoCallback GetOpInfo,
                                 LLVMSymbolLookupCallback SymbolLookUp,
                                 void *DisInfo, MCContext *Ctx,
                                 std::unique_ptr<MCRelocationInfo> &&RelInfo);

/// Target - Wrapper for Target specific information.
///
/// For registration purposes, this is a POD type so that targets can be
/// registered without the use of static constructors.
///
/// Targets should implement a single global instance of this class (which
/// will be zero initialized), and pass that instance to the TargetRegistry as
/// part of their initialization.
class Target {
public:
  friend struct TargetRegistry;

  typedef bool (*ArchMatchFnTy)(Triple::ArchType Arch);

  typedef MCAsmInfo *(*MCAsmInfoCtorFnTy)(const MCRegisterInfo &MRI,
                                          const Triple &TT);
  typedef MCCodeGenInfo *(*MCCodeGenInfoCtorFnTy)(const Triple &TT,
                                                  Reloc::Model RM,
                                                  CodeModel::Model CM,
                                                  CodeGenOpt::Level OL);
  typedef MCInstrInfo *(*MCInstrInfoCtorFnTy)(void);
  typedef MCInstrAnalysis *(*MCInstrAnalysisCtorFnTy)(const MCInstrInfo *Info);
  typedef MCRegisterInfo *(*MCRegInfoCtorFnTy)(const Triple &TT);
  typedef MCSubtargetInfo *(*MCSubtargetInfoCtorFnTy)(const Triple &TT,
                                                      StringRef CPU,
                                                      StringRef Features);
  typedef TargetMachine *(*TargetMachineCtorTy)(
      const Target &T, const Triple &TT, StringRef CPU, StringRef Features,
      const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
      CodeGenOpt::Level OL);
  // If it weren't for layering issues (this header is in llvm/Support, but
  // depends on MC?) this should take the Streamer by value rather than rvalue
  // reference.
  typedef AsmPrinter *(*AsmPrinterCtorTy)(
      TargetMachine &TM, std::unique_ptr<MCStreamer> &&Streamer);
  typedef MCAsmBackend *(*MCAsmBackendCtorTy)(const Target &T,
                                              const MCRegisterInfo &MRI,
                                              const Triple &TT, StringRef CPU);
  typedef MCTargetAsmParser *(*MCAsmParserCtorTy)(
      MCSubtargetInfo &STI, MCAsmParser &P, const MCInstrInfo &MII,
      const MCTargetOptions &Options);
  typedef MCDisassembler *(*MCDisassemblerCtorTy)(const Target &T,
                                                  const MCSubtargetInfo &STI,
                                                  MCContext &Ctx);
  typedef MCInstPrinter *(*MCInstPrinterCtorTy)(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI);
  typedef MCCodeEmitter *(*MCCodeEmitterCtorTy)(const MCInstrInfo &II,
                                                const MCRegisterInfo &MRI,
                                                MCContext &Ctx);
  typedef MCStreamer *(*ELFStreamerCtorTy)(const Triple &T, MCContext &Ctx,
                                           MCAsmBackend &TAB,
                                           raw_pwrite_stream &OS,
                                           MCCodeEmitter *Emitter,
                                           bool RelaxAll);
  typedef MCStreamer *(*MachOStreamerCtorTy)(MCContext &Ctx, MCAsmBackend &TAB,
                                             raw_pwrite_stream &OS,
                                             MCCodeEmitter *Emitter,
                                             bool RelaxAll,
                                             bool DWARFMustBeAtTheEnd);
  typedef MCStreamer *(*COFFStreamerCtorTy)(MCContext &Ctx, MCAsmBackend &TAB,
                                            raw_pwrite_stream &OS,
                                            MCCodeEmitter *Emitter,
                                            bool RelaxAll);
  typedef MCTargetStreamer *(*NullTargetStreamerCtorTy)(MCStreamer &S);
  typedef MCTargetStreamer *(*AsmTargetStreamerCtorTy)(
      MCStreamer &S, formatted_raw_ostream &OS, MCInstPrinter *InstPrint,
      bool IsVerboseAsm);
  typedef MCTargetStreamer *(*ObjectTargetStreamerCtorTy)(
      MCStreamer &S, const MCSubtargetInfo &STI);
  typedef MCRelocationInfo *(*MCRelocationInfoCtorTy)(const Triple &TT,
                                                      MCContext &Ctx);
  typedef MCSymbolizer *(*MCSymbolizerCtorTy)(
      const Triple &TT, LLVMOpInfoCallback GetOpInfo,
      LLVMSymbolLookupCallback SymbolLookUp, void *DisInfo, MCContext *Ctx,
      std::unique_ptr<MCRelocationInfo> &&RelInfo);

private:
  /// Next - The next registered target in the linked list, maintained by the
  /// TargetRegistry.
  Target *Next;

  /// The target function for checking if an architecture is supported.
  ArchMatchFnTy ArchMatchFn;

  /// Name - The target name.
  const char *Name;

  /// ShortDesc - A short description of the target.
  const char *ShortDesc;

  /// HasJIT - Whether this target supports the JIT.
  bool HasJIT;

  /// MCAsmInfoCtorFn - Constructor function for this target's MCAsmInfo, if
  /// registered.
  MCAsmInfoCtorFnTy MCAsmInfoCtorFn;

  /// MCCodeGenInfoCtorFn - Constructor function for this target's
  /// MCCodeGenInfo, if registered.
  MCCodeGenInfoCtorFnTy MCCodeGenInfoCtorFn;

  /// MCInstrInfoCtorFn - Constructor function for this target's MCInstrInfo,
  /// if registered.
  MCInstrInfoCtorFnTy MCInstrInfoCtorFn;

  /// MCInstrAnalysisCtorFn - Constructor function for this target's
  /// MCInstrAnalysis, if registered.
  MCInstrAnalysisCtorFnTy MCInstrAnalysisCtorFn;

  /// MCRegInfoCtorFn - Constructor function for this target's MCRegisterInfo,
  /// if registered.
  MCRegInfoCtorFnTy MCRegInfoCtorFn;

  /// MCSubtargetInfoCtorFn - Constructor function for this target's
  /// MCSubtargetInfo, if registered.
  MCSubtargetInfoCtorFnTy MCSubtargetInfoCtorFn;

  /// TargetMachineCtorFn - Construction function for this target's
  /// TargetMachine, if registered.
  TargetMachineCtorTy TargetMachineCtorFn;

  /// MCAsmBackendCtorFn - Construction function for this target's
  /// MCAsmBackend, if registered.
  MCAsmBackendCtorTy MCAsmBackendCtorFn;

  /// MCAsmParserCtorFn - Construction function for this target's
  /// MCTargetAsmParser, if registered.
  MCAsmParserCtorTy MCAsmParserCtorFn;

  /// AsmPrinterCtorFn - Construction function for this target's AsmPrinter,
  /// if registered.
  AsmPrinterCtorTy AsmPrinterCtorFn;

  /// MCDisassemblerCtorFn - Construction function for this target's
  /// MCDisassembler, if registered.
  MCDisassemblerCtorTy MCDisassemblerCtorFn;

  /// MCInstPrinterCtorFn - Construction function for this target's
  /// MCInstPrinter, if registered.
  MCInstPrinterCtorTy MCInstPrinterCtorFn;

  /// MCCodeEmitterCtorFn - Construction function for this target's
  /// CodeEmitter, if registered.
  MCCodeEmitterCtorTy MCCodeEmitterCtorFn;

  // Construction functions for the various object formats, if registered.
  COFFStreamerCtorTy COFFStreamerCtorFn;
  MachOStreamerCtorTy MachOStreamerCtorFn;
  ELFStreamerCtorTy ELFStreamerCtorFn;

  /// Construction function for this target's null TargetStreamer, if
  /// registered (default = nullptr).
  NullTargetStreamerCtorTy NullTargetStreamerCtorFn;

  /// Construction function for this target's asm TargetStreamer, if
  /// registered (default = nullptr).
  AsmTargetStreamerCtorTy AsmTargetStreamerCtorFn;

  /// Construction function for this target's obj TargetStreamer, if
  /// registered (default = nullptr).
  ObjectTargetStreamerCtorTy ObjectTargetStreamerCtorFn;

  /// MCRelocationInfoCtorFn - Construction function for this target's
  /// MCRelocationInfo, if registered (default = llvm::createMCRelocationInfo)
  MCRelocationInfoCtorTy MCRelocationInfoCtorFn;

  /// MCSymbolizerCtorFn - Construction function for this target's
  /// MCSymbolizer, if registered (default = llvm::createMCSymbolizer)
  MCSymbolizerCtorTy MCSymbolizerCtorFn;

public:
  Target()
      : COFFStreamerCtorFn(nullptr), MachOStreamerCtorFn(nullptr),
        ELFStreamerCtorFn(nullptr), NullTargetStreamerCtorFn(nullptr),
        AsmTargetStreamerCtorFn(nullptr), ObjectTargetStreamerCtorFn(nullptr),
        MCRelocationInfoCtorFn(nullptr), MCSymbolizerCtorFn(nullptr) {}

  /// @name Target Information
  /// @{

  // getNext - Return the next registered target.
  const Target *getNext() const { return Next; }

  /// getName - Get the target name.
  const char *getName() const { return Name; }

  /// getShortDescription - Get a short description of the target.
  const char *getShortDescription() const { return ShortDesc; }

  /// @}
  /// @name Feature Predicates
  /// @{

  /// hasJIT - Check if this targets supports the just-in-time compilation.
  bool hasJIT() const { return HasJIT; }

  /// hasTargetMachine - Check if this target supports code generation.
  bool hasTargetMachine() const { return TargetMachineCtorFn != nullptr; }

  /// hasMCAsmBackend - Check if this target supports .o generation.
  bool hasMCAsmBackend() const { return MCAsmBackendCtorFn != nullptr; }

  /// @}
  /// @name Feature Constructors
  /// @{

  /// createMCAsmInfo - Create a MCAsmInfo implementation for the specified
  /// target triple.
  ///
  /// \param TheTriple This argument is used to determine the target machine
  /// feature set; it should always be provided. Generally this should be
  /// either the target triple from the module, or the target triple of the
  /// host if that does not exist.
  MCAsmInfo *createMCAsmInfo(const MCRegisterInfo &MRI,
                             StringRef TheTriple) const {
    if (!MCAsmInfoCtorFn)
      return nullptr;
    return MCAsmInfoCtorFn(MRI, Triple(TheTriple));
  }

  /// createMCCodeGenInfo - Create a MCCodeGenInfo implementation.
  ///
  MCCodeGenInfo *createMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                     CodeModel::Model CM,
                                     CodeGenOpt::Level OL) const {
    if (!MCCodeGenInfoCtorFn)
      return nullptr;
    return MCCodeGenInfoCtorFn(Triple(TT), RM, CM, OL);
  }

  /// createMCInstrInfo - Create a MCInstrInfo implementation.
  ///
  MCInstrInfo *createMCInstrInfo() const {
    if (!MCInstrInfoCtorFn)
      return nullptr;
    return MCInstrInfoCtorFn();
  }

  /// createMCInstrAnalysis - Create a MCInstrAnalysis implementation.
  ///
  MCInstrAnalysis *createMCInstrAnalysis(const MCInstrInfo *Info) const {
    if (!MCInstrAnalysisCtorFn)
      return nullptr;
    return MCInstrAnalysisCtorFn(Info);
  }

  /// createMCRegInfo - Create a MCRegisterInfo implementation.
  ///
  MCRegisterInfo *createMCRegInfo(StringRef TT) const {
    if (!MCRegInfoCtorFn)
      return nullptr;
    return MCRegInfoCtorFn(Triple(TT));
  }

  /// createMCSubtargetInfo - Create a MCSubtargetInfo implementation.
  ///
  /// \param TheTriple This argument is used to determine the target machine
  /// feature set; it should always be provided. Generally this should be
  /// either the target triple from the module, or the target triple of the
  /// host if that does not exist.
  /// \param CPU This specifies the name of the target CPU.
  /// \param Features This specifies the string representation of the
  /// additional target features.
  MCSubtargetInfo *createMCSubtargetInfo(StringRef TheTriple, StringRef CPU,
                                         StringRef Features) const {
    if (!MCSubtargetInfoCtorFn)
      return nullptr;
    return MCSubtargetInfoCtorFn(Triple(TheTriple), CPU, Features);
  }

  /// createTargetMachine - Create a target specific machine implementation
  /// for the specified \p Triple.
  ///
  /// \param TT This argument is used to determine the target machine
  /// feature set; it should always be provided. Generally this should be
  /// either the target triple from the module, or the target triple of the
  /// host if that does not exist.
  TargetMachine *
  createTargetMachine(StringRef TT, StringRef CPU, StringRef Features,
                      const TargetOptions &Options,
                      Reloc::Model RM = Reloc::Default,
                      CodeModel::Model CM = CodeModel::Default,
                      CodeGenOpt::Level OL = CodeGenOpt::Default) const {
    if (!TargetMachineCtorFn)
      return nullptr;
    return TargetMachineCtorFn(*this, Triple(TT), CPU, Features, Options, RM,
                               CM, OL);
  }

  /// createMCAsmBackend - Create a target specific assembly parser.
  ///
  /// \param TheTriple The target triple string.
  MCAsmBackend *createMCAsmBackend(const MCRegisterInfo &MRI,
                                   StringRef TheTriple, StringRef CPU) const {
    if (!MCAsmBackendCtorFn)
      return nullptr;
    return MCAsmBackendCtorFn(*this, MRI, Triple(TheTriple), CPU);
  }

  /// createMCAsmParser - Create a target specific assembly parser.
  ///
  /// \param Parser The target independent parser implementation to use for
  /// parsing and lexing.
  MCTargetAsmParser *createMCAsmParser(MCSubtargetInfo &STI,
                                       MCAsmParser &Parser,
                                       const MCInstrInfo &MII,
                                       const MCTargetOptions &Options) const {
    if (!MCAsmParserCtorFn)
      return nullptr;
    return MCAsmParserCtorFn(STI, Parser, MII, Options);
  }

  /// createAsmPrinter - Create a target specific assembly printer pass.  This
  /// takes ownership of the MCStreamer object.
  AsmPrinter *createAsmPrinter(TargetMachine &TM,
                               std::unique_ptr<MCStreamer> &&Streamer) const {
    if (!AsmPrinterCtorFn)
      return nullptr;
    return AsmPrinterCtorFn(TM, std::move(Streamer));
  }

  MCDisassembler *createMCDisassembler(const MCSubtargetInfo &STI,
                                       MCContext &Ctx) const {
    if (!MCDisassemblerCtorFn)
      return nullptr;
    return MCDisassemblerCtorFn(*this, STI, Ctx);
  }

  MCInstPrinter *createMCInstPrinter(const Triple &T, unsigned SyntaxVariant,
                                     const MCAsmInfo &MAI,
                                     const MCInstrInfo &MII,
                                     const MCRegisterInfo &MRI) const {
    if (!MCInstPrinterCtorFn)
      return nullptr;
    return MCInstPrinterCtorFn(T, SyntaxVariant, MAI, MII, MRI);
  }

  /// createMCCodeEmitter - Create a target specific code emitter.
  MCCodeEmitter *createMCCodeEmitter(const MCInstrInfo &II,
                                     const MCRegisterInfo &MRI,
                                     MCContext &Ctx) const {
    if (!MCCodeEmitterCtorFn)
      return nullptr;
    return MCCodeEmitterCtorFn(II, MRI, Ctx);
  }

  /// Create a target specific MCStreamer.
  ///
  /// \param T The target triple.
  /// \param Ctx The target context.
  /// \param TAB The target assembler backend object. Takes ownership.
  /// \param OS The stream object.
  /// \param Emitter The target independent assembler object.Takes ownership.
  /// \param RelaxAll Relax all fixups?
  MCStreamer *createMCObjectStreamer(const Triple &T, MCContext &Ctx,
                                     MCAsmBackend &TAB, raw_pwrite_stream &OS,
                                     MCCodeEmitter *Emitter,
                                     const MCSubtargetInfo &STI, bool RelaxAll,
                                     bool DWARFMustBeAtTheEnd) const {
    MCStreamer *S;
    switch (T.getObjectFormat()) {
    default:
      llvm_unreachable("Unknown object format");
    case Triple::COFF:
      assert(T.isOSWindows() && "only Windows COFF is supported");
      S = COFFStreamerCtorFn(Ctx, TAB, OS, Emitter, RelaxAll);
      break;
    case Triple::MachO:
      if (MachOStreamerCtorFn)
        S = MachOStreamerCtorFn(Ctx, TAB, OS, Emitter, RelaxAll,
                                DWARFMustBeAtTheEnd);
      else
        S = createMachOStreamer(Ctx, TAB, OS, Emitter, RelaxAll,
                                DWARFMustBeAtTheEnd);
      break;
    case Triple::ELF:
      if (ELFStreamerCtorFn)
        S = ELFStreamerCtorFn(T, Ctx, TAB, OS, Emitter, RelaxAll);
      else
        S = createELFStreamer(Ctx, TAB, OS, Emitter, RelaxAll);
      break;
    }
    if (ObjectTargetStreamerCtorFn)
      ObjectTargetStreamerCtorFn(*S, STI);
    return S;
  }

  MCStreamer *createAsmStreamer(MCContext &Ctx,
                                std::unique_ptr<formatted_raw_ostream> OS,
                                bool IsVerboseAsm, bool UseDwarfDirectory,
                                MCInstPrinter *InstPrint, MCCodeEmitter *CE,
                                MCAsmBackend *TAB, bool ShowInst) const {
    formatted_raw_ostream &OSRef = *OS;
    MCStreamer *S = llvm::createAsmStreamer(Ctx, std::move(OS), IsVerboseAsm,
                                            UseDwarfDirectory, InstPrint, CE,
                                            TAB, ShowInst);
    createAsmTargetStreamer(*S, OSRef, InstPrint, IsVerboseAsm);
    return S;
  }

  MCTargetStreamer *createAsmTargetStreamer(MCStreamer &S,
                                            formatted_raw_ostream &OS,
                                            MCInstPrinter *InstPrint,
                                            bool IsVerboseAsm) const {
    if (AsmTargetStreamerCtorFn)
      return AsmTargetStreamerCtorFn(S, OS, InstPrint, IsVerboseAsm);
    return nullptr;
  }

  MCStreamer *createNullStreamer(MCContext &Ctx) const {
    MCStreamer *S = llvm::createNullStreamer(Ctx);
    createNullTargetStreamer(*S);
    return S;
  }

  MCTargetStreamer *createNullTargetStreamer(MCStreamer &S) const {
    if (NullTargetStreamerCtorFn)
      return NullTargetStreamerCtorFn(S);
    return nullptr;
  }

  /// createMCRelocationInfo - Create a target specific MCRelocationInfo.
  ///
  /// \param TT The target triple.
  /// \param Ctx The target context.
  MCRelocationInfo *createMCRelocationInfo(StringRef TT, MCContext &Ctx) const {
    MCRelocationInfoCtorTy Fn = MCRelocationInfoCtorFn
                                    ? MCRelocationInfoCtorFn
                                    : llvm::createMCRelocationInfo;
    return Fn(Triple(TT), Ctx);
  }

  /// createMCSymbolizer - Create a target specific MCSymbolizer.
  ///
  /// \param TT The target triple.
  /// \param GetOpInfo The function to get the symbolic information for
  /// operands.
  /// \param SymbolLookUp The function to lookup a symbol name.
  /// \param DisInfo The pointer to the block of symbolic information for above
  /// call
  /// back.
  /// \param Ctx The target context.
  /// \param RelInfo The relocation information for this target. Takes
  /// ownership.
  MCSymbolizer *
  createMCSymbolizer(StringRef TT, LLVMOpInfoCallback GetOpInfo,
                     LLVMSymbolLookupCallback SymbolLookUp, void *DisInfo,
                     MCContext *Ctx,
                     std::unique_ptr<MCRelocationInfo> &&RelInfo) const {
    MCSymbolizerCtorTy Fn =
        MCSymbolizerCtorFn ? MCSymbolizerCtorFn : llvm::createMCSymbolizer;
    return Fn(Triple(TT), GetOpInfo, SymbolLookUp, DisInfo, Ctx,
              std::move(RelInfo));
  }

  /// @}
};

/// TargetRegistry - Generic interface to target specific features.
struct TargetRegistry {
  // FIXME: Make this a namespace, probably just move all the Register*
  // functions into Target (currently they all just set members on the Target
  // anyway, and Target friends this class so those functions can...
  // function).
  TargetRegistry() = delete;

  class iterator
      : public std::iterator<std::forward_iterator_tag, Target, ptrdiff_t> {
    const Target *Current;
    explicit iterator(Target *T) : Current(T) {}
    friend struct TargetRegistry;

  public:
    iterator() : Current(nullptr) {}

    bool operator==(const iterator &x) const { return Current == x.Current; }
    bool operator!=(const iterator &x) const { return !operator==(x); }

    // Iterator traversal: forward iteration only
    iterator &operator++() { // Preincrement
      assert(Current && "Cannot increment end iterator!");
      Current = Current->getNext();
      return *this;
    }
    iterator operator++(int) { // Postincrement
      iterator tmp = *this;
      ++*this;
      return tmp;
    }

    const Target &operator*() const {
      assert(Current && "Cannot dereference end iterator!");
      return *Current;
    }

    const Target *operator->() const { return &operator*(); }
  };

  /// printRegisteredTargetsForVersion - Print the registered targets
  /// appropriately for inclusion in a tool's version output.
  static void printRegisteredTargetsForVersion();

  /// @name Registry Access
  /// @{

  static iterator_range<iterator> targets();

  /// lookupTarget - Lookup a target based on a target triple.
  ///
  /// \param Triple - The triple to use for finding a target.
  /// \param Error - On failure, an error string describing why no target was
  /// found.
  static const Target *lookupTarget(const std::string &Triple,
                                    std::string &Error);

  /// lookupTarget - Lookup a target based on an architecture name
  /// and a target triple.  If the architecture name is non-empty,
  /// then the lookup is done by architecture.  Otherwise, the target
  /// triple is used.
  ///
  /// \param ArchName - The architecture to use for finding a target.
  /// \param TheTriple - The triple to use for finding a target.  The
  /// triple is updated with canonical architecture name if a lookup
  /// by architecture is done.
  /// \param Error - On failure, an error string describing why no target was
  /// found.
  static const Target *lookupTarget(const std::string &ArchName,
                                    Triple &TheTriple, std::string &Error);

  /// @}
  /// @name Target Registration
  /// @{

  /// RegisterTarget - Register the given target. Attempts to register a
  /// target which has already been registered will be ignored.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Name - The target name. This should be a static string.
  /// @param ShortDesc - A short target description. This should be a static
  /// string.
  /// @param ArchMatchFn - The arch match checking function for this target.
  /// @param HasJIT - Whether the target supports JIT code
  /// generation.
  static void RegisterTarget(Target &T, const char *Name, const char *ShortDesc,
                             Target::ArchMatchFnTy ArchMatchFn,
                             bool HasJIT = false);

  /// RegisterMCAsmInfo - Register a MCAsmInfo implementation for the
  /// given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct a MCAsmInfo for the target.
  static void RegisterMCAsmInfo(Target &T, Target::MCAsmInfoCtorFnTy Fn) {
    T.MCAsmInfoCtorFn = Fn;
  }

  /// RegisterMCCodeGenInfo - Register a MCCodeGenInfo implementation for the
  /// given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct a MCCodeGenInfo for the target.
  static void RegisterMCCodeGenInfo(Target &T,
                                    Target::MCCodeGenInfoCtorFnTy Fn) {
    T.MCCodeGenInfoCtorFn = Fn;
  }

  /// RegisterMCInstrInfo - Register a MCInstrInfo implementation for the
  /// given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct a MCInstrInfo for the target.
  static void RegisterMCInstrInfo(Target &T, Target::MCInstrInfoCtorFnTy Fn) {
    T.MCInstrInfoCtorFn = Fn;
  }

  /// RegisterMCInstrAnalysis - Register a MCInstrAnalysis implementation for
  /// the given target.
  static void RegisterMCInstrAnalysis(Target &T,
                                      Target::MCInstrAnalysisCtorFnTy Fn) {
    T.MCInstrAnalysisCtorFn = Fn;
  }

  /// RegisterMCRegInfo - Register a MCRegisterInfo implementation for the
  /// given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct a MCRegisterInfo for the target.
  static void RegisterMCRegInfo(Target &T, Target::MCRegInfoCtorFnTy Fn) {
    T.MCRegInfoCtorFn = Fn;
  }

  /// RegisterMCSubtargetInfo - Register a MCSubtargetInfo implementation for
  /// the given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct a MCSubtargetInfo for the target.
  static void RegisterMCSubtargetInfo(Target &T,
                                      Target::MCSubtargetInfoCtorFnTy Fn) {
    T.MCSubtargetInfoCtorFn = Fn;
  }

  /// RegisterTargetMachine - Register a TargetMachine implementation for the
  /// given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct a TargetMachine for the target.
  static void RegisterTargetMachine(Target &T, Target::TargetMachineCtorTy Fn) {
    T.TargetMachineCtorFn = Fn;
  }

  /// RegisterMCAsmBackend - Register a MCAsmBackend implementation for the
  /// given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct an AsmBackend for the target.
  static void RegisterMCAsmBackend(Target &T, Target::MCAsmBackendCtorTy Fn) {
    T.MCAsmBackendCtorFn = Fn;
  }

  /// RegisterMCAsmParser - Register a MCTargetAsmParser implementation for
  /// the given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct an MCTargetAsmParser for the target.
  static void RegisterMCAsmParser(Target &T, Target::MCAsmParserCtorTy Fn) {
    T.MCAsmParserCtorFn = Fn;
  }

  /// RegisterAsmPrinter - Register an AsmPrinter implementation for the given
  /// target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct an AsmPrinter for the target.
  static void RegisterAsmPrinter(Target &T, Target::AsmPrinterCtorTy Fn) {
    T.AsmPrinterCtorFn = Fn;
  }

  /// RegisterMCDisassembler - Register a MCDisassembler implementation for
  /// the given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct an MCDisassembler for the target.
  static void RegisterMCDisassembler(Target &T,
                                     Target::MCDisassemblerCtorTy Fn) {
    T.MCDisassemblerCtorFn = Fn;
  }

  /// RegisterMCInstPrinter - Register a MCInstPrinter implementation for the
  /// given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct an MCInstPrinter for the target.
  static void RegisterMCInstPrinter(Target &T, Target::MCInstPrinterCtorTy Fn) {
    T.MCInstPrinterCtorFn = Fn;
  }

  /// RegisterMCCodeEmitter - Register a MCCodeEmitter implementation for the
  /// given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct an MCCodeEmitter for the target.
  static void RegisterMCCodeEmitter(Target &T, Target::MCCodeEmitterCtorTy Fn) {
    T.MCCodeEmitterCtorFn = Fn;
  }

  static void RegisterCOFFStreamer(Target &T, Target::COFFStreamerCtorTy Fn) {
    T.COFFStreamerCtorFn = Fn;
  }

  static void RegisterMachOStreamer(Target &T, Target::MachOStreamerCtorTy Fn) {
    T.MachOStreamerCtorFn = Fn;
  }

  static void RegisterELFStreamer(Target &T, Target::ELFStreamerCtorTy Fn) {
    T.ELFStreamerCtorFn = Fn;
  }

  static void RegisterNullTargetStreamer(Target &T,
                                         Target::NullTargetStreamerCtorTy Fn) {
    T.NullTargetStreamerCtorFn = Fn;
  }

  static void RegisterAsmTargetStreamer(Target &T,
                                        Target::AsmTargetStreamerCtorTy Fn) {
    T.AsmTargetStreamerCtorFn = Fn;
  }

  static void
  RegisterObjectTargetStreamer(Target &T,
                               Target::ObjectTargetStreamerCtorTy Fn) {
    T.ObjectTargetStreamerCtorFn = Fn;
  }

  /// RegisterMCRelocationInfo - Register an MCRelocationInfo
  /// implementation for the given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct an MCRelocationInfo for the target.
  static void RegisterMCRelocationInfo(Target &T,
                                       Target::MCRelocationInfoCtorTy Fn) {
    T.MCRelocationInfoCtorFn = Fn;
  }

  /// RegisterMCSymbolizer - Register an MCSymbolizer
  /// implementation for the given target.
  ///
  /// Clients are responsible for ensuring that registration doesn't occur
  /// while another thread is attempting to access the registry. Typically
  /// this is done by initializing all targets at program startup.
  ///
  /// @param T - The target being registered.
  /// @param Fn - A function to construct an MCSymbolizer for the target.
  static void RegisterMCSymbolizer(Target &T, Target::MCSymbolizerCtorTy Fn) {
    T.MCSymbolizerCtorFn = Fn;
  }

  /// @}
};

//===--------------------------------------------------------------------===//

/// RegisterTarget - Helper template for registering a target, for use in the
/// target's initialization function. Usage:
///
///
/// Target TheFooTarget; // The global target instance.
///
/// extern "C" void LLVMInitializeFooTargetInfo() {
///   RegisterTarget<Triple::foo> X(TheFooTarget, "foo", "Foo description");
/// }
template <Triple::ArchType TargetArchType = Triple::UnknownArch,
          bool HasJIT = false>
struct RegisterTarget {
  RegisterTarget(Target &T, const char *Name, const char *Desc) {
    TargetRegistry::RegisterTarget(T, Name, Desc, &getArchMatch, HasJIT);
  }

  static bool getArchMatch(Triple::ArchType Arch) {
    return Arch == TargetArchType;
  }
};

/// RegisterMCAsmInfo - Helper template for registering a target assembly info
/// implementation.  This invokes the static "Create" method on the class to
/// actually do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCAsmInfo<FooMCAsmInfo> X(TheFooTarget);
/// }
template <class MCAsmInfoImpl> struct RegisterMCAsmInfo {
  RegisterMCAsmInfo(Target &T) {
    TargetRegistry::RegisterMCAsmInfo(T, &Allocator);
  }

private:
  static MCAsmInfo *Allocator(const MCRegisterInfo & /*MRI*/,
                              const Triple &TT) {
    return new MCAsmInfoImpl(TT);
  }
};

/// RegisterMCAsmInfoFn - Helper template for registering a target assembly info
/// implementation.  This invokes the specified function to do the
/// construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCAsmInfoFn X(TheFooTarget, TheFunction);
/// }
struct RegisterMCAsmInfoFn {
  RegisterMCAsmInfoFn(Target &T, Target::MCAsmInfoCtorFnTy Fn) {
    TargetRegistry::RegisterMCAsmInfo(T, Fn);
  }
};

/// RegisterMCCodeGenInfo - Helper template for registering a target codegen
/// info
/// implementation.  This invokes the static "Create" method on the class
/// to actually do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCCodeGenInfo<FooMCCodeGenInfo> X(TheFooTarget);
/// }
template <class MCCodeGenInfoImpl> struct RegisterMCCodeGenInfo {
  RegisterMCCodeGenInfo(Target &T) {
    TargetRegistry::RegisterMCCodeGenInfo(T, &Allocator);
  }

private:
  static MCCodeGenInfo *Allocator(const Triple & /*TT*/, Reloc::Model /*RM*/,
                                  CodeModel::Model /*CM*/,
                                  CodeGenOpt::Level /*OL*/) {
    return new MCCodeGenInfoImpl();
  }
};

/// RegisterMCCodeGenInfoFn - Helper template for registering a target codegen
/// info implementation.  This invokes the specified function to do the
/// construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCCodeGenInfoFn X(TheFooTarget, TheFunction);
/// }
struct RegisterMCCodeGenInfoFn {
  RegisterMCCodeGenInfoFn(Target &T, Target::MCCodeGenInfoCtorFnTy Fn) {
    TargetRegistry::RegisterMCCodeGenInfo(T, Fn);
  }
};

/// RegisterMCInstrInfo - Helper template for registering a target instruction
/// info implementation.  This invokes the static "Create" method on the class
/// to actually do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCInstrInfo<FooMCInstrInfo> X(TheFooTarget);
/// }
template <class MCInstrInfoImpl> struct RegisterMCInstrInfo {
  RegisterMCInstrInfo(Target &T) {
    TargetRegistry::RegisterMCInstrInfo(T, &Allocator);
  }

private:
  static MCInstrInfo *Allocator() { return new MCInstrInfoImpl(); }
};

/// RegisterMCInstrInfoFn - Helper template for registering a target
/// instruction info implementation.  This invokes the specified function to
/// do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCInstrInfoFn X(TheFooTarget, TheFunction);
/// }
struct RegisterMCInstrInfoFn {
  RegisterMCInstrInfoFn(Target &T, Target::MCInstrInfoCtorFnTy Fn) {
    TargetRegistry::RegisterMCInstrInfo(T, Fn);
  }
};

/// RegisterMCInstrAnalysis - Helper template for registering a target
/// instruction analyzer implementation.  This invokes the static "Create"
/// method on the class to actually do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCInstrAnalysis<FooMCInstrAnalysis> X(TheFooTarget);
/// }
template <class MCInstrAnalysisImpl> struct RegisterMCInstrAnalysis {
  RegisterMCInstrAnalysis(Target &T) {
    TargetRegistry::RegisterMCInstrAnalysis(T, &Allocator);
  }

private:
  static MCInstrAnalysis *Allocator(const MCInstrInfo *Info) {
    return new MCInstrAnalysisImpl(Info);
  }
};

/// RegisterMCInstrAnalysisFn - Helper template for registering a target
/// instruction analyzer implementation.  This invokes the specified function
/// to do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCInstrAnalysisFn X(TheFooTarget, TheFunction);
/// }
struct RegisterMCInstrAnalysisFn {
  RegisterMCInstrAnalysisFn(Target &T, Target::MCInstrAnalysisCtorFnTy Fn) {
    TargetRegistry::RegisterMCInstrAnalysis(T, Fn);
  }
};

/// RegisterMCRegInfo - Helper template for registering a target register info
/// implementation.  This invokes the static "Create" method on the class to
/// actually do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCRegInfo<FooMCRegInfo> X(TheFooTarget);
/// }
template <class MCRegisterInfoImpl> struct RegisterMCRegInfo {
  RegisterMCRegInfo(Target &T) {
    TargetRegistry::RegisterMCRegInfo(T, &Allocator);
  }

private:
  static MCRegisterInfo *Allocator(const Triple & /*TT*/) {
    return new MCRegisterInfoImpl();
  }
};

/// RegisterMCRegInfoFn - Helper template for registering a target register
/// info implementation.  This invokes the specified function to do the
/// construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCRegInfoFn X(TheFooTarget, TheFunction);
/// }
struct RegisterMCRegInfoFn {
  RegisterMCRegInfoFn(Target &T, Target::MCRegInfoCtorFnTy Fn) {
    TargetRegistry::RegisterMCRegInfo(T, Fn);
  }
};

/// RegisterMCSubtargetInfo - Helper template for registering a target
/// subtarget info implementation.  This invokes the static "Create" method
/// on the class to actually do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCSubtargetInfo<FooMCSubtargetInfo> X(TheFooTarget);
/// }
template <class MCSubtargetInfoImpl> struct RegisterMCSubtargetInfo {
  RegisterMCSubtargetInfo(Target &T) {
    TargetRegistry::RegisterMCSubtargetInfo(T, &Allocator);
  }

private:
  static MCSubtargetInfo *Allocator(const Triple & /*TT*/, StringRef /*CPU*/,
                                    StringRef /*FS*/) {
    return new MCSubtargetInfoImpl();
  }
};

/// RegisterMCSubtargetInfoFn - Helper template for registering a target
/// subtarget info implementation.  This invokes the specified function to
/// do the construction.  Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterMCSubtargetInfoFn X(TheFooTarget, TheFunction);
/// }
struct RegisterMCSubtargetInfoFn {
  RegisterMCSubtargetInfoFn(Target &T, Target::MCSubtargetInfoCtorFnTy Fn) {
    TargetRegistry::RegisterMCSubtargetInfo(T, Fn);
  }
};

/// RegisterTargetMachine - Helper template for registering a target machine
/// implementation, for use in the target machine initialization
/// function. Usage:
///
/// extern "C" void LLVMInitializeFooTarget() {
///   extern Target TheFooTarget;
///   RegisterTargetMachine<FooTargetMachine> X(TheFooTarget);
/// }
template <class TargetMachineImpl> struct RegisterTargetMachine {
  RegisterTargetMachine(Target &T) {
    TargetRegistry::RegisterTargetMachine(T, &Allocator);
  }

private:
  static TargetMachine *Allocator(const Target &T, const Triple &TT,
                                  StringRef CPU, StringRef FS,
                                  const TargetOptions &Options, Reloc::Model RM,
                                  CodeModel::Model CM, CodeGenOpt::Level OL) {
    return new TargetMachineImpl(T, TT, CPU, FS, Options, RM, CM, OL);
  }
};

/// RegisterMCAsmBackend - Helper template for registering a target specific
/// assembler backend. Usage:
///
/// extern "C" void LLVMInitializeFooMCAsmBackend() {
///   extern Target TheFooTarget;
///   RegisterMCAsmBackend<FooAsmLexer> X(TheFooTarget);
/// }
template <class MCAsmBackendImpl> struct RegisterMCAsmBackend {
  RegisterMCAsmBackend(Target &T) {
    TargetRegistry::RegisterMCAsmBackend(T, &Allocator);
  }

private:
  static MCAsmBackend *Allocator(const Target &T, const MCRegisterInfo &MRI,
                                 const Triple &TheTriple, StringRef CPU) {
    return new MCAsmBackendImpl(T, MRI, TheTriple, CPU);
  }
};

/// RegisterMCAsmParser - Helper template for registering a target specific
/// assembly parser, for use in the target machine initialization
/// function. Usage:
///
/// extern "C" void LLVMInitializeFooMCAsmParser() {
///   extern Target TheFooTarget;
///   RegisterMCAsmParser<FooAsmParser> X(TheFooTarget);
/// }
template <class MCAsmParserImpl> struct RegisterMCAsmParser {
  RegisterMCAsmParser(Target &T) {
    TargetRegistry::RegisterMCAsmParser(T, &Allocator);
  }

private:
  static MCTargetAsmParser *Allocator(MCSubtargetInfo &STI, MCAsmParser &P,
                                      const MCInstrInfo &MII,
                                      const MCTargetOptions &Options) {
    return new MCAsmParserImpl(STI, P, MII, Options);
  }
};

/// RegisterAsmPrinter - Helper template for registering a target specific
/// assembly printer, for use in the target machine initialization
/// function. Usage:
///
/// extern "C" void LLVMInitializeFooAsmPrinter() {
///   extern Target TheFooTarget;
///   RegisterAsmPrinter<FooAsmPrinter> X(TheFooTarget);
/// }
template <class AsmPrinterImpl> struct RegisterAsmPrinter {
  RegisterAsmPrinter(Target &T) {
    TargetRegistry::RegisterAsmPrinter(T, &Allocator);
  }

private:
  static AsmPrinter *Allocator(TargetMachine &TM,
                               std::unique_ptr<MCStreamer> &&Streamer) {
    return new AsmPrinterImpl(TM, std::move(Streamer));
  }
};

/// RegisterMCCodeEmitter - Helper template for registering a target specific
/// machine code emitter, for use in the target initialization
/// function. Usage:
///
/// extern "C" void LLVMInitializeFooMCCodeEmitter() {
///   extern Target TheFooTarget;
///   RegisterMCCodeEmitter<FooCodeEmitter> X(TheFooTarget);
/// }
template <class MCCodeEmitterImpl> struct RegisterMCCodeEmitter {
  RegisterMCCodeEmitter(Target &T) {
    TargetRegistry::RegisterMCCodeEmitter(T, &Allocator);
  }

private:
  static MCCodeEmitter *Allocator(const MCInstrInfo & /*II*/,
                                  const MCRegisterInfo & /*MRI*/,
                                  MCContext & /*Ctx*/) {
    return new MCCodeEmitterImpl();
  }
};
}

#endif
