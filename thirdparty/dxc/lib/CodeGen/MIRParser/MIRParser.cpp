//===- MIRParser.cpp - MIR serialization format parser implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that parses the optional LLVM IR and machine
// functions that are stored in MIR files.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "MIParser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include <memory>

using namespace llvm;

namespace llvm {

/// This class implements the parsing of LLVM IR that's embedded inside a MIR
/// file.
class MIRParserImpl {
  SourceMgr SM;
  StringRef Filename;
  LLVMContext &Context;
  StringMap<std::unique_ptr<yaml::MachineFunction>> Functions;
  SlotMapping IRSlots;
  /// Maps from register class names to register classes.
  StringMap<const TargetRegisterClass *> Names2RegClasses;

public:
  MIRParserImpl(std::unique_ptr<MemoryBuffer> Contents, StringRef Filename,
                LLVMContext &Context);

  void reportDiagnostic(const SMDiagnostic &Diag);

  /// Report an error with the given message at unknown location.
  ///
  /// Always returns true.
  bool error(const Twine &Message);

  /// Report an error with the given message at the given location.
  ///
  /// Always returns true.
  bool error(SMLoc Loc, const Twine &Message);

  /// Report a given error with the location translated from the location in an
  /// embedded string literal to a location in the MIR file.
  ///
  /// Always returns true.
  bool error(const SMDiagnostic &Error, SMRange SourceRange);

  /// Try to parse the optional LLVM module and the machine functions in the MIR
  /// file.
  ///
  /// Return null if an error occurred.
  std::unique_ptr<Module> parse();

  /// Parse the machine function in the current YAML document.
  ///
  /// \param NoLLVMIR - set to true when the MIR file doesn't have LLVM IR.
  /// A dummy IR function is created and inserted into the given module when
  /// this parameter is true.
  ///
  /// Return true if an error occurred.
  bool parseMachineFunction(yaml::Input &In, Module &M, bool NoLLVMIR);

  /// Initialize the machine function to the state that's described in the MIR
  /// file.
  ///
  /// Return true if error occurred.
  bool initializeMachineFunction(MachineFunction &MF);

  /// Initialize the machine basic block using it's YAML representation.
  ///
  /// Return true if an error occurred.
  bool initializeMachineBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB,
                                   const yaml::MachineBasicBlock &YamlMBB,
                                   const PerFunctionMIParsingState &PFS);

  bool
  initializeRegisterInfo(const MachineFunction &MF,
                         MachineRegisterInfo &RegInfo,
                         const yaml::MachineFunction &YamlMF,
                         DenseMap<unsigned, unsigned> &VirtualRegisterSlots);

  bool initializeFrameInfo(MachineFrameInfo &MFI,
                           const yaml::MachineFunction &YamlMF);

private:
  /// Return a MIR diagnostic converted from an MI string diagnostic.
  SMDiagnostic diagFromMIStringDiag(const SMDiagnostic &Error,
                                    SMRange SourceRange);

  /// Return a MIR diagnostic converted from an LLVM assembly diagnostic.
  SMDiagnostic diagFromLLVMAssemblyDiag(const SMDiagnostic &Error,
                                        SMRange SourceRange);

  /// Create an empty function with the given name.
  void createDummyFunction(StringRef Name, Module &M);

  void initNames2RegClasses(const MachineFunction &MF);

  /// Check if the given identifier is a name of a register class.
  ///
  /// Return null if the name isn't a register class.
  const TargetRegisterClass *getRegClass(const MachineFunction &MF,
                                         StringRef Name);
};

} // end namespace llvm

MIRParserImpl::MIRParserImpl(std::unique_ptr<MemoryBuffer> Contents,
                             StringRef Filename, LLVMContext &Context)
    : SM(), Filename(Filename), Context(Context) {
  SM.AddNewSourceBuffer(std::move(Contents), SMLoc());
}

bool MIRParserImpl::error(const Twine &Message) {
  Context.diagnose(DiagnosticInfoMIRParser(
      DS_Error, SMDiagnostic(Filename, SourceMgr::DK_Error, Message.str())));
  return true;
}

bool MIRParserImpl::error(SMLoc Loc, const Twine &Message) {
  Context.diagnose(DiagnosticInfoMIRParser(
      DS_Error, SM.GetMessage(Loc, SourceMgr::DK_Error, Message)));
  return true;
}

bool MIRParserImpl::error(const SMDiagnostic &Error, SMRange SourceRange) {
  assert(Error.getKind() == SourceMgr::DK_Error && "Expected an error");
  reportDiagnostic(diagFromMIStringDiag(Error, SourceRange));
  return true;
}

void MIRParserImpl::reportDiagnostic(const SMDiagnostic &Diag) {
  DiagnosticSeverity Kind;
  switch (Diag.getKind()) {
  case SourceMgr::DK_Error:
    Kind = DS_Error;
    break;
  case SourceMgr::DK_Warning:
    Kind = DS_Warning;
    break;
  case SourceMgr::DK_Note:
    Kind = DS_Note;
    break;
  }
  Context.diagnose(DiagnosticInfoMIRParser(Kind, Diag));
}

static void handleYAMLDiag(const SMDiagnostic &Diag, void *Context) {
  reinterpret_cast<MIRParserImpl *>(Context)->reportDiagnostic(Diag);
}

std::unique_ptr<Module> MIRParserImpl::parse() {
  yaml::Input In(SM.getMemoryBuffer(SM.getMainFileID())->getBuffer(),
                 /*Ctxt=*/nullptr, handleYAMLDiag, this);
  In.setContext(&In);

  if (!In.setCurrentDocument()) {
    if (In.error())
      return nullptr;
    // Create an empty module when the MIR file is empty.
    return llvm::make_unique<Module>(Filename, Context);
  }

  std::unique_ptr<Module> M;
  bool NoLLVMIR = false;
  // Parse the block scalar manually so that we can return unique pointer
  // without having to go trough YAML traits.
  if (const auto *BSN =
          dyn_cast_or_null<yaml::BlockScalarNode>(In.getCurrentNode())) {
    SMDiagnostic Error;
    M = parseAssembly(MemoryBufferRef(BSN->getValue(), Filename), Error,
                      Context, &IRSlots);
    if (!M) {
      reportDiagnostic(diagFromLLVMAssemblyDiag(Error, BSN->getSourceRange()));
      return M;
    }
    In.nextDocument();
    if (!In.setCurrentDocument())
      return M;
  } else {
    // Create an new, empty module.
    M = llvm::make_unique<Module>(Filename, Context);
    NoLLVMIR = true;
  }

  // Parse the machine functions.
  do {
    if (parseMachineFunction(In, *M, NoLLVMIR))
      return nullptr;
    In.nextDocument();
  } while (In.setCurrentDocument());

  return M;
}

bool MIRParserImpl::parseMachineFunction(yaml::Input &In, Module &M,
                                         bool NoLLVMIR) {
  auto MF = llvm::make_unique<yaml::MachineFunction>();
  yaml::yamlize(In, *MF, false);
  if (In.error())
    return true;
  auto FunctionName = MF->Name;
  if (Functions.find(FunctionName) != Functions.end())
    return error(Twine("redefinition of machine function '") + FunctionName +
                 "'");
  Functions.insert(std::make_pair(FunctionName, std::move(MF)));
  if (NoLLVMIR)
    createDummyFunction(FunctionName, M);
  else if (!M.getFunction(FunctionName))
    return error(Twine("function '") + FunctionName +
                 "' isn't defined in the provided LLVM IR");
  return false;
}

void MIRParserImpl::createDummyFunction(StringRef Name, Module &M) {
  auto &Context = M.getContext();
  Function *F = cast<Function>(M.getOrInsertFunction(
      Name, FunctionType::get(Type::getVoidTy(Context), false)));
  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  new UnreachableInst(Context, BB);
}

bool MIRParserImpl::initializeMachineFunction(MachineFunction &MF) {
  auto It = Functions.find(MF.getName());
  if (It == Functions.end())
    return error(Twine("no machine function information for function '") +
                 MF.getName() + "' in the MIR file");
  // TODO: Recreate the machine function.
  const yaml::MachineFunction &YamlMF = *It->getValue();
  if (YamlMF.Alignment)
    MF.setAlignment(YamlMF.Alignment);
  MF.setExposesReturnsTwice(YamlMF.ExposesReturnsTwice);
  MF.setHasInlineAsm(YamlMF.HasInlineAsm);
  PerFunctionMIParsingState PFS;
  if (initializeRegisterInfo(MF, MF.getRegInfo(), YamlMF,
                             PFS.VirtualRegisterSlots))
    return true;
  if (initializeFrameInfo(*MF.getFrameInfo(), YamlMF))
    return true;

  const auto &F = *MF.getFunction();
  for (const auto &YamlMBB : YamlMF.BasicBlocks) {
    const BasicBlock *BB = nullptr;
    const yaml::StringValue &Name = YamlMBB.Name;
    if (!Name.Value.empty()) {
      BB = dyn_cast_or_null<BasicBlock>(
          F.getValueSymbolTable().lookup(Name.Value));
      if (!BB)
        return error(Name.SourceRange.Start,
                     Twine("basic block '") + Name.Value +
                         "' is not defined in the function '" + MF.getName() +
                         "'");
    }
    auto *MBB = MF.CreateMachineBasicBlock(BB);
    MF.insert(MF.end(), MBB);
    bool WasInserted =
        PFS.MBBSlots.insert(std::make_pair(YamlMBB.ID, MBB)).second;
    if (!WasInserted)
      return error(Twine("redefinition of machine basic block with id #") +
                   Twine(YamlMBB.ID));
  }

  if (YamlMF.BasicBlocks.empty())
    return error(Twine("machine function '") + Twine(MF.getName()) +
                 "' requires at least one machine basic block in its body");
  // Initialize the machine basic blocks after creating them all so that the
  // machine instructions parser can resolve the MBB references.
  unsigned I = 0;
  for (const auto &YamlMBB : YamlMF.BasicBlocks) {
    if (initializeMachineBasicBlock(MF, *MF.getBlockNumbered(I++), YamlMBB,
                                    PFS))
      return true;
  }
  return false;
}

bool MIRParserImpl::initializeMachineBasicBlock(
    MachineFunction &MF, MachineBasicBlock &MBB,
    const yaml::MachineBasicBlock &YamlMBB,
    const PerFunctionMIParsingState &PFS) {
  MBB.setAlignment(YamlMBB.Alignment);
  if (YamlMBB.AddressTaken)
    MBB.setHasAddressTaken();
  MBB.setIsLandingPad(YamlMBB.IsLandingPad);
  SMDiagnostic Error;
  // Parse the successors.
  for (const auto &MBBSource : YamlMBB.Successors) {
    MachineBasicBlock *SuccMBB = nullptr;
    if (parseMBBReference(SuccMBB, SM, MF, MBBSource.Value, PFS, IRSlots,
                          Error))
      return error(Error, MBBSource.SourceRange);
    // TODO: Report an error when adding the same successor more than once.
    MBB.addSuccessor(SuccMBB);
  }
  // Parse the liveins.
  for (const auto &LiveInSource : YamlMBB.LiveIns) {
    unsigned Reg = 0;
    if (parseNamedRegisterReference(Reg, SM, MF, LiveInSource.Value, PFS,
                                    IRSlots, Error))
      return error(Error, LiveInSource.SourceRange);
    MBB.addLiveIn(Reg);
  }
  // Parse the instructions.
  for (const auto &MISource : YamlMBB.Instructions) {
    MachineInstr *MI = nullptr;
    if (parseMachineInstr(MI, SM, MF, MISource.Value, PFS, IRSlots, Error))
      return error(Error, MISource.SourceRange);
    MBB.insert(MBB.end(), MI);
  }
  return false;
}

bool MIRParserImpl::initializeRegisterInfo(
    const MachineFunction &MF, MachineRegisterInfo &RegInfo,
    const yaml::MachineFunction &YamlMF,
    DenseMap<unsigned, unsigned> &VirtualRegisterSlots) {
  assert(RegInfo.isSSA());
  if (!YamlMF.IsSSA)
    RegInfo.leaveSSA();
  assert(RegInfo.tracksLiveness());
  if (!YamlMF.TracksRegLiveness)
    RegInfo.invalidateLiveness();
  RegInfo.enableSubRegLiveness(YamlMF.TracksSubRegLiveness);

  // Parse the virtual register information.
  for (const auto &VReg : YamlMF.VirtualRegisters) {
    const auto *RC = getRegClass(MF, VReg.Class.Value);
    if (!RC)
      return error(VReg.Class.SourceRange.Start,
                   Twine("use of undefined register class '") +
                       VReg.Class.Value + "'");
    unsigned Reg = RegInfo.createVirtualRegister(RC);
    // TODO: Report an error when the same virtual register with the same ID is
    // redefined.
    VirtualRegisterSlots.insert(std::make_pair(VReg.ID, Reg));
  }
  return false;
}

bool MIRParserImpl::initializeFrameInfo(MachineFrameInfo &MFI,
                                        const yaml::MachineFunction &YamlMF) {
  const yaml::MachineFrameInfo &YamlMFI = YamlMF.FrameInfo;
  MFI.setFrameAddressIsTaken(YamlMFI.IsFrameAddressTaken);
  MFI.setReturnAddressIsTaken(YamlMFI.IsReturnAddressTaken);
  MFI.setHasStackMap(YamlMFI.HasStackMap);
  MFI.setHasPatchPoint(YamlMFI.HasPatchPoint);
  MFI.setStackSize(YamlMFI.StackSize);
  MFI.setOffsetAdjustment(YamlMFI.OffsetAdjustment);
  if (YamlMFI.MaxAlignment)
    MFI.ensureMaxAlignment(YamlMFI.MaxAlignment);
  MFI.setAdjustsStack(YamlMFI.AdjustsStack);
  MFI.setHasCalls(YamlMFI.HasCalls);
  MFI.setMaxCallFrameSize(YamlMFI.MaxCallFrameSize);
  MFI.setHasOpaqueSPAdjustment(YamlMFI.HasOpaqueSPAdjustment);
  MFI.setHasVAStart(YamlMFI.HasVAStart);
  MFI.setHasMustTailInVarArgFunc(YamlMFI.HasMustTailInVarArgFunc);

  // Initialize the fixed frame objects.
  for (const auto &Object : YamlMF.FixedStackObjects) {
    int ObjectIdx;
    if (Object.Type != yaml::FixedMachineStackObject::SpillSlot)
      ObjectIdx = MFI.CreateFixedObject(Object.Size, Object.Offset,
                                        Object.IsImmutable, Object.IsAliased);
    else
      ObjectIdx = MFI.CreateFixedSpillStackObject(Object.Size, Object.Offset);
    MFI.setObjectAlignment(ObjectIdx, Object.Alignment);
    // TODO: Store the mapping between fixed object IDs and object indices to
    // parse fixed stack object references correctly.
  }

  // Initialize the ordinary frame objects.
  for (const auto &Object : YamlMF.StackObjects) {
    int ObjectIdx;
    if (Object.Type == yaml::MachineStackObject::VariableSized)
      ObjectIdx =
          MFI.CreateVariableSizedObject(Object.Alignment, /*Alloca=*/nullptr);
    else
      ObjectIdx = MFI.CreateStackObject(
          Object.Size, Object.Alignment,
          Object.Type == yaml::MachineStackObject::SpillSlot);
    MFI.setObjectOffset(ObjectIdx, Object.Offset);
    // TODO: Store the mapping between object IDs and object indices to parse
    // stack object references correctly.
  }
  return false;
}

SMDiagnostic MIRParserImpl::diagFromMIStringDiag(const SMDiagnostic &Error,
                                                 SMRange SourceRange) {
  assert(SourceRange.isValid() && "Invalid source range");
  SMLoc Loc = SourceRange.Start;
  bool HasQuote = Loc.getPointer() < SourceRange.End.getPointer() &&
                  *Loc.getPointer() == '\'';
  // Translate the location of the error from the location in the MI string to
  // the corresponding location in the MIR file.
  Loc = Loc.getFromPointer(Loc.getPointer() + Error.getColumnNo() +
                           (HasQuote ? 1 : 0));

  // TODO: Translate any source ranges as well.
  return SM.GetMessage(Loc, Error.getKind(), Error.getMessage(), None,
                       Error.getFixIts());
}

SMDiagnostic MIRParserImpl::diagFromLLVMAssemblyDiag(const SMDiagnostic &Error,
                                                     SMRange SourceRange) {
  assert(SourceRange.isValid());

  // Translate the location of the error from the location in the llvm IR string
  // to the corresponding location in the MIR file.
  auto LineAndColumn = SM.getLineAndColumn(SourceRange.Start);
  unsigned Line = LineAndColumn.first + Error.getLineNo() - 1;
  unsigned Column = Error.getColumnNo();
  StringRef LineStr = Error.getLineContents();
  SMLoc Loc = Error.getLoc();

  // Get the full line and adjust the column number by taking the indentation of
  // LLVM IR into account.
  for (line_iterator L(*SM.getMemoryBuffer(SM.getMainFileID()), false), E;
       L != E; ++L) {
    if (L.line_number() == Line) {
      LineStr = *L;
      Loc = SMLoc::getFromPointer(LineStr.data());
      auto Indent = LineStr.find(Error.getLineContents());
      if (Indent != StringRef::npos)
        Column += Indent;
      break;
    }
  }

  return SMDiagnostic(SM, Loc, Filename, Line, Column, Error.getKind(),
                      Error.getMessage(), LineStr, Error.getRanges(),
                      Error.getFixIts());
}

void MIRParserImpl::initNames2RegClasses(const MachineFunction &MF) {
  if (!Names2RegClasses.empty())
    return;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (unsigned I = 0, E = TRI->getNumRegClasses(); I < E; ++I) {
    const auto *RC = TRI->getRegClass(I);
    Names2RegClasses.insert(
        std::make_pair(StringRef(TRI->getRegClassName(RC)).lower(), RC));
  }
}

const TargetRegisterClass *MIRParserImpl::getRegClass(const MachineFunction &MF,
                                                      StringRef Name) {
  initNames2RegClasses(MF);
  auto RegClassInfo = Names2RegClasses.find(Name);
  if (RegClassInfo == Names2RegClasses.end())
    return nullptr;
  return RegClassInfo->getValue();
}

MIRParser::MIRParser(std::unique_ptr<MIRParserImpl> Impl)
    : Impl(std::move(Impl)) {}

MIRParser::~MIRParser() {}

std::unique_ptr<Module> MIRParser::parseLLVMModule() { return Impl->parse(); }

bool MIRParser::initializeMachineFunction(MachineFunction &MF) {
  return Impl->initializeMachineFunction(MF);
}

std::unique_ptr<MIRParser> llvm::createMIRParserFromFile(StringRef Filename,
                                                         SMDiagnostic &Error,
                                                         LLVMContext &Context) {
  auto FileOrErr = MemoryBuffer::getFile(Filename);
  if (std::error_code EC = FileOrErr.getError()) {
    Error = SMDiagnostic(Filename, SourceMgr::DK_Error,
                         "Could not open input file: " + EC.message());
    return nullptr;
  }
  return createMIRParser(std::move(FileOrErr.get()), Context);
}

std::unique_ptr<MIRParser>
llvm::createMIRParser(std::unique_ptr<MemoryBuffer> Contents,
                      LLVMContext &Context) {
  auto Filename = Contents->getBufferIdentifier();
  return llvm::make_unique<MIRParser>(
      llvm::make_unique<MIRParserImpl>(std::move(Contents), Filename, Context));
}
