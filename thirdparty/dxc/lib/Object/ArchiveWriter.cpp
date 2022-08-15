//===- ArchiveWriter.cpp - ar File Format implementation --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the writeArchive function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ArchiveWriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace llvm;

NewArchiveIterator::NewArchiveIterator(object::Archive::child_iterator I,
                                       StringRef Name)
    : IsNewMember(false), Name(Name), OldI(I) {}

NewArchiveIterator::NewArchiveIterator(StringRef NewFilename, StringRef Name)
    : IsNewMember(true), Name(Name), NewFilename(NewFilename) {}

StringRef NewArchiveIterator::getName() const { return Name; }

bool NewArchiveIterator::isNewMember() const { return IsNewMember; }

object::Archive::child_iterator NewArchiveIterator::getOld() const {
  assert(!IsNewMember);
  return OldI;
}

StringRef NewArchiveIterator::getNew() const {
  assert(IsNewMember);
  return NewFilename;
}

llvm::ErrorOr<int>
NewArchiveIterator::getFD(sys::fs::file_status &NewStatus) const {
  assert(IsNewMember);
  int NewFD;
  if (auto EC = sys::fs::openFileForRead(NewFilename, NewFD))
    return EC;
  assert(NewFD != -1);

  if (auto EC = sys::fs::status(NewFD, NewStatus))
    return EC;

  // Opening a directory doesn't make sense. Let it fail.
  // Linux cannot open directories with open(2), although
  // cygwin and *bsd can.
  if (NewStatus.type() == sys::fs::file_type::directory_file)
    return make_error_code(errc::is_a_directory);

  return NewFD;
}

template <typename T>
static void printWithSpacePadding(raw_fd_ostream &OS, T Data, unsigned Size,
				  bool MayTruncate = false) {
  uint64_t OldPos = OS.tell();
  OS << Data;
  unsigned SizeSoFar = OS.tell() - OldPos;
  if (Size > SizeSoFar) {
    OS.indent(Size - SizeSoFar);
  } else if (Size < SizeSoFar) {
    assert(MayTruncate && "Data doesn't fit in Size");
    // Some of the data this is used for (like UID) can be larger than the
    // space available in the archive format. Truncate in that case.
    OS.seek(OldPos + Size);
  }
}

static void print32(raw_ostream &Out, object::Archive::Kind Kind,
                    uint32_t Val) {
  if (Kind == object::Archive::K_GNU)
    support::endian::Writer<support::big>(Out).write(Val);
  else
    support::endian::Writer<support::little>(Out).write(Val);
}

static void printRestOfMemberHeader(raw_fd_ostream &Out,
                                    const sys::TimeValue &ModTime, unsigned UID,
                                    unsigned GID, unsigned Perms,
                                    unsigned Size) {
  printWithSpacePadding(Out, ModTime.toEpochTime(), 12);
  printWithSpacePadding(Out, UID, 6, true);
  printWithSpacePadding(Out, GID, 6, true);
  printWithSpacePadding(Out, format("%o", Perms), 8);
  printWithSpacePadding(Out, Size, 10);
  Out << "`\n";
}

static void printGNUSmallMemberHeader(raw_fd_ostream &Out, StringRef Name,
                                      const sys::TimeValue &ModTime,
                                      unsigned UID, unsigned GID,
                                      unsigned Perms, unsigned Size) {
  printWithSpacePadding(Out, Twine(Name) + "/", 16);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

static void printBSDMemberHeader(raw_fd_ostream &Out, StringRef Name,
                                 const sys::TimeValue &ModTime, unsigned UID,
                                 unsigned GID, unsigned Perms, unsigned Size) {
  uint64_t PosAfterHeader = Out.tell() + 60 + Name.size();
  // Pad so that even 64 bit object files are aligned.
  unsigned Pad = OffsetToAlignment(PosAfterHeader, 8);
  unsigned NameWithPadding = Name.size() + Pad;
  printWithSpacePadding(Out, Twine("#1/") + Twine(NameWithPadding), 16);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms,
                          NameWithPadding + Size);
  Out << Name;
  assert(PosAfterHeader == Out.tell());
  while (Pad--)
    Out.write(uint8_t(0));
}

static void
printMemberHeader(raw_fd_ostream &Out, object::Archive::Kind Kind,
                  StringRef Name,
                  std::vector<unsigned>::iterator &StringMapIndexIter,
                  const sys::TimeValue &ModTime, unsigned UID, unsigned GID,
                  unsigned Perms, unsigned Size) {
  if (Kind == object::Archive::K_BSD)
    return printBSDMemberHeader(Out, Name, ModTime, UID, GID, Perms, Size);
  if (Name.size() < 16)
    return printGNUSmallMemberHeader(Out, Name, ModTime, UID, GID, Perms, Size);
  Out << '/';
  printWithSpacePadding(Out, *StringMapIndexIter++, 15);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

static void writeStringTable(raw_fd_ostream &Out,
                             ArrayRef<NewArchiveIterator> Members,
                             std::vector<unsigned> &StringMapIndexes) {
  unsigned StartOffset = 0;
  for (ArrayRef<NewArchiveIterator>::iterator I = Members.begin(),
                                              E = Members.end();
       I != E; ++I) {
    StringRef Name = I->getName();
    if (Name.size() < 16)
      continue;
    if (StartOffset == 0) {
      printWithSpacePadding(Out, "//", 58);
      Out << "`\n";
      StartOffset = Out.tell();
    }
    StringMapIndexes.push_back(Out.tell() - StartOffset);
    Out << Name << "/\n";
  }
  if (StartOffset == 0)
    return;
  if (Out.tell() % 2)
    Out << '\n';
  int Pos = Out.tell();
  Out.seek(StartOffset - 12);
  printWithSpacePadding(Out, Pos - StartOffset, 10);
  Out.seek(Pos);
}

static sys::TimeValue now(bool Deterministic) {
  if (!Deterministic)
    return sys::TimeValue::now();
  sys::TimeValue TV;
  TV.fromEpochTime(0);
  return TV;
}

// Returns the offset of the first reference to a member offset.
static ErrorOr<unsigned>
writeSymbolTable(raw_fd_ostream &Out, object::Archive::Kind Kind,
                 ArrayRef<NewArchiveIterator> Members,
                 ArrayRef<MemoryBufferRef> Buffers,
                 std::vector<unsigned> &MemberOffsetRefs, bool Deterministic) {
  unsigned HeaderStartOffset = 0;
  unsigned BodyStartOffset = 0;
  SmallString<128> NameBuf;
  raw_svector_ostream NameOS(NameBuf);
  LLVMContext Context;
  for (unsigned MemberNum = 0, N = Members.size(); MemberNum < N; ++MemberNum) {
    MemoryBufferRef MemberBuffer = Buffers[MemberNum];
    ErrorOr<std::unique_ptr<object::SymbolicFile>> ObjOrErr =
        object::SymbolicFile::createSymbolicFile(
            MemberBuffer, sys::fs::file_magic::unknown, &Context);
    if (!ObjOrErr)
      continue;  // FIXME: check only for "not an object file" errors.
    object::SymbolicFile &Obj = *ObjOrErr.get();

    if (!HeaderStartOffset) {
      HeaderStartOffset = Out.tell();
      if (Kind == object::Archive::K_GNU)
        printGNUSmallMemberHeader(Out, "", now(Deterministic), 0, 0, 0, 0);
      else
        printBSDMemberHeader(Out, "__.SYMDEF", now(Deterministic), 0, 0, 0, 0);
      BodyStartOffset = Out.tell();
      print32(Out, Kind, 0); // number of entries or bytes
    }

    for (const object::BasicSymbolRef &S : Obj.symbols()) {
      uint32_t Symflags = S.getFlags();
      if (Symflags & object::SymbolRef::SF_FormatSpecific)
        continue;
      if (!(Symflags & object::SymbolRef::SF_Global))
        continue;
      if (Symflags & object::SymbolRef::SF_Undefined)
        continue;

      unsigned NameOffset = NameOS.tell();
      if (auto EC = S.printName(NameOS))
        return EC;
      NameOS << '\0';
      MemberOffsetRefs.push_back(MemberNum);
      if (Kind == object::Archive::K_BSD)
        print32(Out, Kind, NameOffset);
      print32(Out, Kind, 0); // member offset
    }
  }

  if (HeaderStartOffset == 0)
    return 0;

  StringRef StringTable = NameOS.str();
  if (Kind == object::Archive::K_BSD)
    print32(Out, Kind, StringTable.size()); // byte count of the string table
  Out << StringTable;

  // ld64 requires the next member header to start at an offset that is
  // 4 bytes aligned.
  unsigned Pad = OffsetToAlignment(Out.tell(), 4);
  while (Pad--)
    Out.write(uint8_t(0));

  // Patch up the size of the symbol table now that we know how big it is.
  unsigned Pos = Out.tell();
  const unsigned MemberHeaderSize = 60;
  Out.seek(HeaderStartOffset + 48); // offset of the size field.
  printWithSpacePadding(Out, Pos - MemberHeaderSize - HeaderStartOffset, 10);

  // Patch up the number of symbols.
  Out.seek(BodyStartOffset);
  unsigned NumSyms = MemberOffsetRefs.size();
  if (Kind == object::Archive::K_GNU)
    print32(Out, Kind, NumSyms);
  else
    print32(Out, Kind, NumSyms * 8);

  Out.seek(Pos);
  return BodyStartOffset + 4;
}

std::pair<StringRef, std::error_code> llvm::writeArchive(
    StringRef ArcName, std::vector<NewArchiveIterator> &NewMembers,
    bool WriteSymtab, object::Archive::Kind Kind, bool Deterministic) {
  SmallString<128> TmpArchive;
  int TmpArchiveFD;
  if (auto EC = sys::fs::createUniqueFile(ArcName + ".temp-archive-%%%%%%%.a",
                                          TmpArchiveFD, TmpArchive))
    return std::make_pair(ArcName, EC);

  tool_output_file Output(TmpArchive, TmpArchiveFD);
  raw_fd_ostream &Out = Output.os();
  Out << "!<arch>\n";

  std::vector<unsigned> MemberOffsetRefs;

  std::vector<std::unique_ptr<MemoryBuffer>> Buffers;
  std::vector<MemoryBufferRef> Members;
  std::vector<sys::fs::file_status> NewMemberStatus;

  for (unsigned I = 0, N = NewMembers.size(); I < N; ++I) {
    NewArchiveIterator &Member = NewMembers[I];
    MemoryBufferRef MemberRef;

    if (Member.isNewMember()) {
      StringRef Filename = Member.getNew();
      NewMemberStatus.resize(NewMemberStatus.size() + 1);
      sys::fs::file_status &Status = NewMemberStatus.back();
      ErrorOr<int> FD = Member.getFD(Status);
      if (auto EC = FD.getError())
        return std::make_pair(Filename, EC);
      ErrorOr<std::unique_ptr<MemoryBuffer>> MemberBufferOrErr =
          MemoryBuffer::getOpenFile(FD.get(), Filename, Status.getSize(),
                                    false);
      if (auto EC = MemberBufferOrErr.getError())
        return std::make_pair(Filename, EC);
      if (close(FD.get()) != 0)
        return std::make_pair(Filename,
                              std::error_code(errno, std::generic_category()));
      Buffers.push_back(std::move(MemberBufferOrErr.get()));
      MemberRef = Buffers.back()->getMemBufferRef();
    } else {
      object::Archive::child_iterator OldMember = Member.getOld();
      ErrorOr<MemoryBufferRef> MemberBufferOrErr =
          OldMember->getMemoryBufferRef();
      if (auto EC = MemberBufferOrErr.getError())
        return std::make_pair("", EC);
      MemberRef = MemberBufferOrErr.get();
    }
    Members.push_back(MemberRef);
  }

  unsigned MemberReferenceOffset = 0;
  if (WriteSymtab) {
    ErrorOr<unsigned> MemberReferenceOffsetOrErr = writeSymbolTable(
        Out, Kind, NewMembers, Members, MemberOffsetRefs, Deterministic);
    if (auto EC = MemberReferenceOffsetOrErr.getError())
      return std::make_pair(ArcName, EC);
    MemberReferenceOffset = MemberReferenceOffsetOrErr.get();
  }

  std::vector<unsigned> StringMapIndexes;
  if (Kind != object::Archive::K_BSD)
    writeStringTable(Out, NewMembers, StringMapIndexes);

  unsigned MemberNum = 0;
  unsigned NewMemberNum = 0;
  std::vector<unsigned>::iterator StringMapIndexIter = StringMapIndexes.begin();
  std::vector<unsigned> MemberOffset;
  for (const NewArchiveIterator &I : NewMembers) {
    MemoryBufferRef File = Members[MemberNum++];

    unsigned Pos = Out.tell();
    MemberOffset.push_back(Pos);

    sys::TimeValue ModTime;
    unsigned UID;
    unsigned GID;
    unsigned Perms;
    if (Deterministic) {
      ModTime.fromEpochTime(0);
      UID = 0;
      GID = 0;
      Perms = 0644;
    } else if (I.isNewMember()) {
      const sys::fs::file_status &Status = NewMemberStatus[NewMemberNum];
      ModTime = Status.getLastModificationTime();
      UID = Status.getUser();
      GID = Status.getGroup();
      Perms = Status.permissions();
    } else {
      object::Archive::child_iterator OldMember = I.getOld();
      ModTime = OldMember->getLastModified();
      UID = OldMember->getUID();
      GID = OldMember->getGID();
      Perms = OldMember->getAccessMode();
    }

    if (I.isNewMember()) {
      StringRef FileName = I.getNew();
      const sys::fs::file_status &Status = NewMemberStatus[NewMemberNum++];
      printMemberHeader(Out, Kind, sys::path::filename(FileName),
                        StringMapIndexIter, ModTime, UID, GID, Perms,
                        Status.getSize());
    } else {
      object::Archive::child_iterator OldMember = I.getOld();
      printMemberHeader(Out, Kind, I.getName(), StringMapIndexIter, ModTime,
                        UID, GID, Perms, OldMember->getSize());
    }

    Out << File.getBuffer();

    if (Out.tell() % 2)
      Out << '\n';
  }

  if (MemberReferenceOffset) {
    Out.seek(MemberReferenceOffset);
    for (unsigned MemberNum : MemberOffsetRefs) {
      if (Kind == object::Archive::K_BSD)
        Out.seek(Out.tell() + 4); // skip over the string offset
      print32(Out, Kind, MemberOffset[MemberNum]);
    }
  }

  Output.keep();
  Out.close();
  sys::fs::rename(TmpArchive, ArcName);
  return std::make_pair("", std::error_code());
}
