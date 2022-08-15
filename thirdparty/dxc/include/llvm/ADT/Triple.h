//===-- llvm/ADT/Triple.h - Target triple helper class ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_TRIPLE_H
#define LLVM_ADT_TRIPLE_H

#include "llvm/ADT/Twine.h"

// Some system headers or GCC predefined macros conflict with identifiers in
// this file.  Undefine them here.
#undef NetBSD
#undef mips
#undef sparc

namespace llvm {

/// Triple - Helper class for working with autoconf configuration names. For
/// historical reasons, we also call these 'triples' (they used to contain
/// exactly three fields).
///
/// Configuration names are strings in the canonical form:
///   ARCHITECTURE-VENDOR-OPERATING_SYSTEM
/// or
///   ARCHITECTURE-VENDOR-OPERATING_SYSTEM-ENVIRONMENT
///
/// This class is used for clients which want to support arbitrary
/// configuration names, but also want to implement certain special
/// behavior for particular configurations. This class isolates the mapping
/// from the components of the configuration name to well known IDs.
///
/// At its core the Triple class is designed to be a wrapper for a triple
/// string; the constructor does not change or normalize the triple string.
/// Clients that need to handle the non-canonical triples that users often
/// specify should use the normalize method.
///
/// See autoconf/config.guess for a glimpse into what configuration names
/// look like in practice.
class Triple {
public:
  enum ArchType {
    UnknownArch,

    arm,        // ARM (little endian): arm, armv.*, xscale
    armeb,      // ARM (big endian): armeb
    aarch64,    // AArch64 (little endian): aarch64
    aarch64_be, // AArch64 (big endian): aarch64_be
    bpfel,      // eBPF or extended BPF or 64-bit BPF (little endian)
    bpfeb,      // eBPF or extended BPF or 64-bit BPF (big endian)
    hexagon,    // Hexagon: hexagon
    mips,       // MIPS: mips, mipsallegrex
    mipsel,     // MIPSEL: mipsel, mipsallegrexel
    mips64,     // MIPS64: mips64
    mips64el,   // MIPS64EL: mips64el
    msp430,     // MSP430: msp430
    ppc,        // PPC: powerpc
    ppc64,      // PPC64: powerpc64, ppu
    ppc64le,    // PPC64LE: powerpc64le
    r600,       // R600: AMD GPUs HD2XXX - HD6XXX
    amdgcn,     // AMDGCN: AMD GCN GPUs
    sparc,      // Sparc: sparc
    sparcv9,    // Sparcv9: Sparcv9
    sparcel,    // Sparc: (endianness = little). NB: 'Sparcle' is a CPU variant
    systemz,    // SystemZ: s390x
    tce,        // TCE (http://tce.cs.tut.fi/): tce
    thumb,      // Thumb (little endian): thumb, thumbv.*
    thumbeb,    // Thumb (big endian): thumbeb
    x86,        // X86: i[3-9]86
    x86_64,     // X86-64: amd64, x86_64
    xcore,      // XCore: xcore
    nvptx,      // NVPTX: 32-bit
    nvptx64,    // NVPTX: 64-bit
    le32,       // le32: generic little-endian 32-bit CPU (PNaCl / Emscripten)
    le64,       // le64: generic little-endian 64-bit CPU (PNaCl / Emscripten)
    amdil,      // AMDIL
    amdil64,    // AMDIL with 64-bit pointers
    hsail,      // AMD HSAIL
    hsail64,    // AMD HSAIL with 64-bit pointers
    spir,       // SPIR: standard portable IR for OpenCL 32-bit version
    spir64,     // SPIR: standard portable IR for OpenCL 64-bit version
    // HLSL Change Begins
    dxil,       // DXIL: DirectX Intermediate Language 32-bit
    dxil64,     // DXIL: DirectX Intermediate Language 64-bit
    // HLSL Change Ends
    kalimba,    // Kalimba: generic kalimba
    shave,      // SHAVE: Movidius vector VLIW processors
    wasm32,     // WebAssembly with 32-bit pointers
    wasm64,     // WebAssembly with 64-bit pointers
    LastArchType = wasm64
  };
  enum SubArchType {
    NoSubArch,

    ARMSubArch_v8_1a,
    ARMSubArch_v8,
    ARMSubArch_v7,
    ARMSubArch_v7em,
    ARMSubArch_v7m,
    ARMSubArch_v7s,
    ARMSubArch_v6,
    ARMSubArch_v6m,
    ARMSubArch_v6k,
    ARMSubArch_v6t2,
    ARMSubArch_v5,
    ARMSubArch_v5te,
    ARMSubArch_v4t,

    KalimbaSubArch_v3,
    KalimbaSubArch_v4,
    KalimbaSubArch_v5
  };
  enum VendorType {
    UnknownVendor,

    Apple,
    PC,
    SCEI,
    BGP,
    BGQ,
    Freescale,
    IBM,
    ImaginationTechnologies,
    MipsTechnologies,
    Microsoft,    // HLSL Change
    NVIDIA,
    CSR,
    LastVendorType = CSR
  };
  enum OSType {
    UnknownOS,

    CloudABI,
    Darwin,
    DragonFly,
    FreeBSD,
    IOS,
    KFreeBSD,
    Linux,
    Lv2,        // PS3
    MacOSX,
    NetBSD,
    OpenBSD,
    Solaris,
    Win32,
    Haiku,
    Minix,
    RTEMS,
    NaCl,       // Native Client
    CNK,        // BG/P Compute-Node Kernel
    Bitrig,
    AIX,
    CUDA,       // NVIDIA CUDA
    NVCL,       // NVIDIA OpenCL
    AMDHSA,     // AMD HSA Runtime
    DirectX,    // HLSL Change
    PS4,
    LastOSType = PS4
  };
  enum EnvironmentType {
    UnknownEnvironment,

    GNU,
    GNUEABI,
    GNUEABIHF,
    GNUX32,
    CODE16,
    EABI,
    EABIHF,
    Android,

    MSVC,
    Itanium,
    Cygnus,
    LastEnvironmentType = Cygnus
  };
  enum ObjectFormatType {
    UnknownObjectFormat,

    COFF,
    ELF,
    MachO,
  };

private:
  std::string Data;

  /// The parsed arch type.
  ArchType Arch;

  /// The parsed subarchitecture type.
  SubArchType SubArch;

  /// The parsed vendor type.
  VendorType Vendor;

  /// The parsed OS type.
  OSType OS;

  /// The parsed Environment type.
  EnvironmentType Environment;

  /// The object format type.
  ObjectFormatType ObjectFormat;

public:
  /// @name Constructors
  /// @{

  /// \brief Default constructor is the same as an empty string and leaves all
  /// triple fields unknown.
  Triple() : Data(), Arch(), Vendor(), OS(), Environment(), ObjectFormat() {}

  explicit Triple(const Twine &Str);
  Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr);
  Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr,
         const Twine &EnvironmentStr);

  bool operator==(const Triple &Other) const {
    return Arch == Other.Arch && SubArch == Other.SubArch &&
           Vendor == Other.Vendor && OS == Other.OS &&
           Environment == Other.Environment &&
           ObjectFormat == Other.ObjectFormat;
  }

  /// @}
  /// @name Normalization
  /// @{

  /// normalize - Turn an arbitrary machine specification into the canonical
  /// triple form (or something sensible that the Triple class understands if
  /// nothing better can reasonably be done).  In particular, it handles the
  /// common case in which otherwise valid components are in the wrong order.
  static std::string normalize(StringRef Str);

  /// \brief Return the normalized form of this triple's string.
  std::string normalize() const { return normalize(Data); }

  /// @}
  /// @name Typed Component Access
  /// @{

  /// getArch - Get the parsed architecture type of this triple.
  ArchType getArch() const { return Arch; }

  /// getSubArch - get the parsed subarchitecture type for this triple.
  SubArchType getSubArch() const { return SubArch; }

  /// getVendor - Get the parsed vendor type of this triple.
  VendorType getVendor() const { return Vendor; }

  /// getOS - Get the parsed operating system type of this triple.
  OSType getOS() const { return OS; }

  /// hasEnvironment - Does this triple have the optional environment
  /// (fourth) component?
  bool hasEnvironment() const {
    return getEnvironmentName() != "";
  }

  /// getEnvironment - Get the parsed environment type of this triple.
  EnvironmentType getEnvironment() const { return Environment; }

  /// \brief Parse the version number from the OS name component of the
  /// triple, if present.
  ///
  /// For example, "fooos1.2.3" would return (1, 2, 3).
  ///
  /// If an entry is not defined, it will be returned as 0.
  void getEnvironmentVersion(unsigned &Major, unsigned &Minor,
                             unsigned &Micro) const;

  /// getFormat - Get the object format for this triple.
  ObjectFormatType getObjectFormat() const { return ObjectFormat; }

  /// getOSVersion - Parse the version number from the OS name component of the
  /// triple, if present.
  ///
  /// For example, "fooos1.2.3" would return (1, 2, 3).
  ///
  /// If an entry is not defined, it will be returned as 0.
  void getOSVersion(unsigned &Major, unsigned &Minor, unsigned &Micro) const;

  /// getOSMajorVersion - Return just the major version number, this is
  /// specialized because it is a common query.
  unsigned getOSMajorVersion() const {
    unsigned Maj, Min, Micro;
    getOSVersion(Maj, Min, Micro);
    return Maj;
  }

  /// getMacOSXVersion - Parse the version number as with getOSVersion and then
  /// translate generic "darwin" versions to the corresponding OS X versions.
  /// This may also be called with IOS triples but the OS X version number is
  /// just set to a constant 10.4.0 in that case.  Returns true if successful.
  bool getMacOSXVersion(unsigned &Major, unsigned &Minor,
                        unsigned &Micro) const;

  /// getiOSVersion - Parse the version number as with getOSVersion.  This should
  /// only be called with IOS triples.
  void getiOSVersion(unsigned &Major, unsigned &Minor,
                     unsigned &Micro) const;

  /// @}
  /// @name Direct Component Access
  /// @{

  const std::string &str() const { return Data; }

  const std::string &getTriple() const { return Data; }

  /// getArchName - Get the architecture (first) component of the
  /// triple.
  StringRef getArchName() const;

  /// getVendorName - Get the vendor (second) component of the triple.
  StringRef getVendorName() const;

  /// getOSName - Get the operating system (third) component of the
  /// triple.
  StringRef getOSName() const;

  /// getEnvironmentName - Get the optional environment (fourth)
  /// component of the triple, or "" if empty.
  StringRef getEnvironmentName() const;

  /// getOSAndEnvironmentName - Get the operating system and optional
  /// environment components as a single string (separated by a '-'
  /// if the environment component is present).
  StringRef getOSAndEnvironmentName() const;

  /// @}
  /// @name Convenience Predicates
  /// @{

  /// \brief Test whether the architecture is 64-bit
  ///
  /// Note that this tests for 64-bit pointer width, and nothing else. Note
  /// that we intentionally expose only three predicates, 64-bit, 32-bit, and
  /// 16-bit. The inner details of pointer width for particular architectures
  /// is not summed up in the triple, and so only a coarse grained predicate
  /// system is provided.
  bool isArch64Bit() const;

  /// \brief Test whether the architecture is 32-bit
  ///
  /// Note that this tests for 32-bit pointer width, and nothing else.
  bool isArch32Bit() const;

  /// \brief Test whether the architecture is 16-bit
  ///
  /// Note that this tests for 16-bit pointer width, and nothing else.
  bool isArch16Bit() const;

  /// isOSVersionLT - Helper function for doing comparisons against version
  /// numbers included in the target triple.
  bool isOSVersionLT(unsigned Major, unsigned Minor = 0,
                     unsigned Micro = 0) const {
    unsigned LHS[3];
    getOSVersion(LHS[0], LHS[1], LHS[2]);

    if (LHS[0] != Major)
      return LHS[0] < Major;
    if (LHS[1] != Minor)
      return LHS[1] < Minor;
    if (LHS[2] != Micro)
      return LHS[1] < Micro;

    return false;
  }

  bool isOSVersionLT(const Triple &Other) const {
    unsigned RHS[3];
    Other.getOSVersion(RHS[0], RHS[1], RHS[2]);
    return isOSVersionLT(RHS[0], RHS[1], RHS[2]);
  }

  /// isMacOSXVersionLT - Comparison function for checking OS X version
  /// compatibility, which handles supporting skewed version numbering schemes
  /// used by the "darwin" triples.
  unsigned isMacOSXVersionLT(unsigned Major, unsigned Minor = 0,
                             unsigned Micro = 0) const {
    assert(isMacOSX() && "Not an OS X triple!");

    // If this is OS X, expect a sane version number.
    if (getOS() == Triple::MacOSX)
      return isOSVersionLT(Major, Minor, Micro);

    // Otherwise, compare to the "Darwin" number.
    assert(Major == 10 && "Unexpected major version");
    return isOSVersionLT(Minor + 4, Micro, 0);
  }

  /// isMacOSX - Is this a Mac OS X triple. For legacy reasons, we support both
  /// "darwin" and "osx" as OS X triples.
  bool isMacOSX() const {
    return getOS() == Triple::Darwin || getOS() == Triple::MacOSX;
  }

  /// Is this an iOS triple.
  bool isiOS() const {
    return getOS() == Triple::IOS;
  }

  /// isOSDarwin - Is this a "Darwin" OS (OS X or iOS).
  bool isOSDarwin() const {
    return isMacOSX() || isiOS();
  }

  bool isOSNetBSD() const {
    return getOS() == Triple::NetBSD;
  }

  bool isOSOpenBSD() const {
    return getOS() == Triple::OpenBSD;
  }

  bool isOSFreeBSD() const {
    return getOS() == Triple::FreeBSD;
  }

  bool isOSDragonFly() const { return getOS() == Triple::DragonFly; }

  bool isOSSolaris() const {
    return getOS() == Triple::Solaris;
  }

  bool isOSBitrig() const {
    return getOS() == Triple::Bitrig;
  }

  bool isWindowsMSVCEnvironment() const {
    return getOS() == Triple::Win32 &&
           (getEnvironment() == Triple::UnknownEnvironment ||
            getEnvironment() == Triple::MSVC);
  }

  bool isKnownWindowsMSVCEnvironment() const {
    return getOS() == Triple::Win32 && getEnvironment() == Triple::MSVC;
  }

  bool isWindowsItaniumEnvironment() const {
    return getOS() == Triple::Win32 && getEnvironment() == Triple::Itanium;
  }

  bool isWindowsCygwinEnvironment() const {
    return getOS() == Triple::Win32 && getEnvironment() == Triple::Cygnus;
  }

  bool isWindowsGNUEnvironment() const {
    return getOS() == Triple::Win32 && getEnvironment() == Triple::GNU;
  }

  /// \brief Tests for either Cygwin or MinGW OS
  bool isOSCygMing() const {
    return isWindowsCygwinEnvironment() || isWindowsGNUEnvironment();
  }

  /// \brief Is this a "Windows" OS targeting a "MSVCRT.dll" environment.
  bool isOSMSVCRT() const {
    return isWindowsMSVCEnvironment() || isWindowsGNUEnvironment() ||
           isWindowsItaniumEnvironment();
  }

  /// \brief Tests whether the OS is Windows.
  bool isOSWindows() const {
    return getOS() == Triple::Win32;
  }

  /// \brief Tests whether the OS is NaCl (Native Client)
  bool isOSNaCl() const {
    return getOS() == Triple::NaCl;
  }

  /// \brief Tests whether the OS is Linux.
  bool isOSLinux() const {
    return getOS() == Triple::Linux;
  }

  /// \brief Tests whether the OS uses the ELF binary format.
  bool isOSBinFormatELF() const {
    return getObjectFormat() == Triple::ELF;
  }

  /// \brief Tests whether the OS uses the COFF binary format.
  bool isOSBinFormatCOFF() const {
    return getObjectFormat() == Triple::COFF;
  }

  /// \brief Tests whether the environment is MachO.
  bool isOSBinFormatMachO() const {
    return getObjectFormat() == Triple::MachO;
  }

  /// \brief Tests whether the target is the PS4 CPU
  bool isPS4CPU() const {
    return getArch() == Triple::x86_64 &&
           getVendor() == Triple::SCEI &&
           getOS() == Triple::PS4;
  }

  /// \brief Tests whether the target is the PS4 platform
  bool isPS4() const {
    return getVendor() == Triple::SCEI &&
           getOS() == Triple::PS4;
  }

  /// @}
  /// @name Mutators
  /// @{

  /// setArch - Set the architecture (first) component of the triple
  /// to a known type.
  void setArch(ArchType Kind);

  /// setVendor - Set the vendor (second) component of the triple to a
  /// known type.
  void setVendor(VendorType Kind);

  /// setOS - Set the operating system (third) component of the triple
  /// to a known type.
  void setOS(OSType Kind);

  /// setEnvironment - Set the environment (fourth) component of the triple
  /// to a known type.
  void setEnvironment(EnvironmentType Kind);

  /// setObjectFormat - Set the object file format
  void setObjectFormat(ObjectFormatType Kind);

  /// setTriple - Set all components to the new triple \p Str.
  void setTriple(const Twine &Str);

  /// setArchName - Set the architecture (first) component of the
  /// triple by name.
  void setArchName(StringRef Str);

  /// setVendorName - Set the vendor (second) component of the triple
  /// by name.
  void setVendorName(StringRef Str);

  /// setOSName - Set the operating system (third) component of the
  /// triple by name.
  void setOSName(StringRef Str);

  /// setEnvironmentName - Set the optional environment (fourth)
  /// component of the triple by name.
  void setEnvironmentName(StringRef Str);

  /// setOSAndEnvironmentName - Set the operating system and optional
  /// environment components with a single string.
  void setOSAndEnvironmentName(StringRef Str);

  /// @}
  /// @name Helpers to build variants of a particular triple.
  /// @{

  /// \brief Form a triple with a 32-bit variant of the current architecture.
  ///
  /// This can be used to move across "families" of architectures where useful.
  ///
  /// \returns A new triple with a 32-bit architecture or an unknown
  ///          architecture if no such variant can be found.
  llvm::Triple get32BitArchVariant() const;

  /// \brief Form a triple with a 64-bit variant of the current architecture.
  ///
  /// This can be used to move across "families" of architectures where useful.
  ///
  /// \returns A new triple with a 64-bit architecture or an unknown
  ///          architecture if no such variant can be found.
  llvm::Triple get64BitArchVariant() const;

  /// Form a triple with a big endian variant of the current architecture.
  ///
  /// This can be used to move across "families" of architectures where useful.
  ///
  /// \returns A new triple with a big endian architecture or an unknown
  ///          architecture if no such variant can be found.
  llvm::Triple getBigEndianArchVariant() const;

  /// Form a triple with a little endian variant of the current architecture.
  ///
  /// This can be used to move across "families" of architectures where useful.
  ///
  /// \returns A new triple with a little endian architecture or an unknown
  ///          architecture if no such variant can be found.
  llvm::Triple getLittleEndianArchVariant() const;

  /// Get the (LLVM) name of the minimum ARM CPU for the arch we are targeting.
  ///
  /// \param Arch the architecture name (e.g., "armv7s"). If it is an empty
  /// string then the triple's arch name is used.
  const char* getARMCPUForArch(StringRef Arch = StringRef()) const;

  /// @}
  /// @name Static helpers for IDs.
  /// @{

  /// getArchTypeName - Get the canonical name for the \p Kind architecture.
  static const char *getArchTypeName(ArchType Kind);

  /// getArchTypePrefix - Get the "prefix" canonical name for the \p Kind
  /// architecture. This is the prefix used by the architecture specific
  /// builtins, and is suitable for passing to \see
  /// Intrinsic::getIntrinsicForGCCBuiltin().
  ///
  /// \return - The architecture prefix, or 0 if none is defined.
  static const char *getArchTypePrefix(ArchType Kind);

  /// getVendorTypeName - Get the canonical name for the \p Kind vendor.
  static const char *getVendorTypeName(VendorType Kind);

  /// getOSTypeName - Get the canonical name for the \p Kind operating system.
  static const char *getOSTypeName(OSType Kind);

  /// getEnvironmentTypeName - Get the canonical name for the \p Kind
  /// environment.
  static const char *getEnvironmentTypeName(EnvironmentType Kind);

  /// @}
  /// @name Static helpers for converting alternate architecture names.
  /// @{

  /// getArchTypeForLLVMName - The canonical type for the given LLVM
  /// architecture name (e.g., "x86").
  static ArchType getArchTypeForLLVMName(StringRef Str);

  /// @}
};

} // End llvm namespace


#endif
