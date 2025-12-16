/*
Copyright (C) 2001-present by Serge Lamikhov-Center

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef ELFIO_DUMP_HPP
#define ELFIO_DUMP_HPP

#include <algorithm>
#include <string>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <elfio/elfio.hpp>

namespace ELFIO {

static const struct class_table_t
{
    const char  key;
    const char* str;
} class_table[] = {
    { ELFCLASS32, "ELF32" },
    { ELFCLASS64, "ELF64" },
};

static const struct endian_table_t
{
    const char  key;
    const char* str;
} endian_table[] = {
    { ELFDATANONE, "None" },
    { ELFDATA2LSB, "Little endian" },
    { ELFDATA2MSB, "Big endian" },
};

static const struct os_abi_table_t
{
    const unsigned char key;
    const char*         str;
} os_abi_table[] = {
    { ELFOSABI_NONE, "UNIX System V" },
    { ELFOSABI_HPUX, "Hewlett-Packard HP-UX" },
    { ELFOSABI_NETBSD, "NetBSD" },
    { ELFOSABI_LINUX, "Linux" },
    { ELFOSABI_HURD, "GNU Hurd" },
    { ELFOSABI_SOLARIS, "Sun Solaris" },
    { ELFOSABI_AIX, "AIX" },
    { ELFOSABI_IRIX, "IRIX" },
    { ELFOSABI_FREEBSD, "FreeBSD" },
    { ELFOSABI_TRU64, "Compaq TRU64 UNIX" },
    { ELFOSABI_MODESTO, "Novell Modesto" },
    { ELFOSABI_OPENBSD, "Open BSD" },
    { ELFOSABI_OPENVMS, "Open VMS" },
    { ELFOSABI_NSK, "Hewlett-Packard Non-Stop Kernel" },
    { ELFOSABI_AROS, "Amiga Research OS" },
    { ELFOSABI_FENIXOS, "FenixOS" },
    { ELFOSABI_NUXI, "Nuxi CloudABI" },
    { ELFOSABI_OPENVOS, "OpenVOS" },
    { ELFOSABI_ARM, "ARM" },
    { ELFOSABI_STANDALONE, "Standalone (embedded)" },
};

static const struct version_table_t
{
    const Elf64_Word key;
    const char*      str;
} version_table[] = {
    { EV_NONE, "None" },
    { EV_CURRENT, "Current" },
};

static const struct type_table_t
{
    const Elf32_Half key;
    const char*      str;
} type_table[] = {
    { ET_NONE, "No file type" },    { ET_REL, "Relocatable file" },
    { ET_EXEC, "Executable file" }, { ET_DYN, "Shared object file" },
    { ET_CORE, "Core file" },
};

static const struct machine_table_t
{
    const Elf64_Half key;
    const char*      str;
} machine_table[] = {
    { EM_NONE, "No machine" },
    { EM_M32, "AT&T WE 32100" },
    { EM_SPARC, "SUN SPARC" },
    { EM_386, "Intel 80386" },
    { EM_68K, "Motorola m68k family" },
    { EM_88K, "Motorola m88k family" },
    { EM_486, "Intel 80486// Reserved for future use" },
    { EM_860, "Intel 80860" },
    { EM_MIPS, "MIPS R3000 (officially, big-endian only)" },
    { EM_S370, "IBM System/370" },
    { EM_MIPS_RS3_LE,
      "MIPS R3000 little-endian (Oct 4 1999 Draft) Deprecated" },
    { EM_res011, "Reserved" },
    { EM_res012, "Reserved" },
    { EM_res013, "Reserved" },
    { EM_res014, "Reserved" },
    { EM_PARISC, "HPPA" },
    { EM_res016, "Reserved" },
    { EM_VPP550, "Fujitsu VPP500" },
    { EM_SPARC32PLUS, "Sun's v8plus" },
    { EM_960, "Intel 80960" },
    { EM_PPC, "PowerPC" },
    { EM_PPC64, "64-bit PowerPC" },
    { EM_S390, "IBM S/390" },
    { EM_SPU, "Sony/Toshiba/IBM SPU" },
    { EM_res024, "Reserved" },
    { EM_res025, "Reserved" },
    { EM_res026, "Reserved" },
    { EM_res027, "Reserved" },
    { EM_res028, "Reserved" },
    { EM_res029, "Reserved" },
    { EM_res030, "Reserved" },
    { EM_res031, "Reserved" },
    { EM_res032, "Reserved" },
    { EM_res033, "Reserved" },
    { EM_res034, "Reserved" },
    { EM_res035, "Reserved" },
    { EM_V800, "NEC V800 series" },
    { EM_FR20, "Fujitsu FR20" },
    { EM_RH32, "TRW RH32" },
    { EM_MCORE, "Motorola M*Core // May also be taken by Fujitsu MMA" },
    { EM_RCE, "Old name for MCore" },
    { EM_ARM, "ARM" },
    { EM_OLD_ALPHA, "Digital Alpha" },
    { EM_SH, "Renesas (formerly Hitachi) / SuperH SH" },
    { EM_SPARCV9, "SPARC v9 64-bit" },
    { EM_TRICORE, "Siemens Tricore embedded processor" },
    { EM_ARC, "ARC Cores" },
    { EM_H8_300, "Renesas (formerly Hitachi) H8/300" },
    { EM_H8_300H, "Renesas (formerly Hitachi) H8/300H" },
    { EM_H8S, "Renesas (formerly Hitachi) H8S" },
    { EM_H8_500, "Renesas (formerly Hitachi) H8/500" },
    { EM_IA_64, "Intel IA-64 Processor" },
    { EM_MIPS_X, "Stanford MIPS-X" },
    { EM_COLDFIRE, "Motorola Coldfire" },
    { EM_68HC12, "Motorola M68HC12" },
    { EM_MMA, "Fujitsu Multimedia Accelerator" },
    { EM_PCP, "Siemens PCP" },
    { EM_NCPU, "Sony nCPU embedded RISC processor" },
    { EM_NDR1, "Denso NDR1 microprocesspr" },
    { EM_STARCORE, "Motorola Star*Core processor" },
    { EM_ME16, "Toyota ME16 processor" },
    { EM_ST100, "STMicroelectronics ST100 processor" },
    { EM_TINYJ, "Advanced Logic Corp. TinyJ embedded processor" },
    { EM_X86_64, "Advanced Micro Devices X86-64 processor" },
    { EM_PDSP, "Sony DSP Processor" },
    { EM_PDP10, "Digital Equipment Corp. PDP-10" },
    { EM_PDP11, "Digital Equipment Corp. PDP-11" },
    { EM_FX66, "Siemens FX66 microcontroller" },
    { EM_ST9PLUS, "STMicroelectronics ST9+ 8/16 bit microcontroller" },
    { EM_ST7, "STMicroelectronics ST7 8-bit microcontroller" },
    { EM_68HC16, "Motorola MC68HC16 Microcontroller" },
    { EM_68HC11, "Motorola MC68HC11 Microcontroller" },
    { EM_68HC08, "Motorola MC68HC08 Microcontroller" },
    { EM_68HC05, "Motorola MC68HC05 Microcontroller" },
    { EM_SVX, "Silicon Graphics SVx" },
    { EM_ST19, "STMicroelectronics ST19 8-bit cpu" },
    { EM_VAX, "Digital VAX" },
    { EM_CRIS, "Axis Communications 32-bit embedded processor" },
    { EM_JAVELIN, "Infineon Technologies 32-bit embedded cpu" },
    { EM_FIREPATH, "Element 14 64-bit DSP processor" },
    { EM_ZSP, "LSI Logic's 16-bit DSP processor" },
    { EM_MMIX, "Donald Knuth's educational 64-bit processor" },
    { EM_HUANY, "Harvard's machine-independent format" },
    { EM_PRISM, "SiTera Prism" },
    { EM_AVR, "Atmel AVR 8-bit microcontroller" },
    { EM_FR30, "Fujitsu FR30" },
    { EM_D10V, "Mitsubishi D10V" },
    { EM_D30V, "Mitsubishi D30V" },
    { EM_V850, "NEC v850" },
    { EM_M32R, "Renesas M32R (formerly Mitsubishi M32R)" },
    { EM_MN10300, "Matsushita MN10300" },
    { EM_MN10200, "Matsushita MN10200" },
    { EM_PJ, "picoJava" },
    { EM_OPENRISC, "OpenRISC 32-bit embedded processor" },
    { EM_ARC_A5, "ARC Cores Tangent-A5" },
    { EM_XTENSA, "Tensilica Xtensa Architecture" },
    { EM_VIDEOCORE, "Alphamosaic VideoCore processor" },
    { EM_TMM_GPP, "Thompson Multimedia General Purpose Processor" },
    { EM_NS32K, "National Semiconductor 32000 series" },
    { EM_TPC, "Tenor Network TPC processor" },
    { EM_SNP1K, "Trebia SNP 1000 processor" },
    { EM_ST200, "STMicroelectronics ST200 microcontroller" },
    { EM_IP2K, "Ubicom IP2022 micro controller" },
    { EM_MAX, "MAX Processor" },
    { EM_CR, "National Semiconductor CompactRISC" },
    { EM_F2MC16, "Fujitsu F2MC16" },
    { EM_MSP430, "TI msp430 micro controller" },
    { EM_BLACKFIN, "ADI Blackfin" },
    { EM_SE_C33, "S1C33 Family of Seiko Epson processors" },
    { EM_SEP, "Sharp embedded microprocessor" },
    { EM_ARCA, "Arca RISC Microprocessor" },
    { EM_UNICORE, "Microprocessor series from PKU-Unity Ltd. and MPRC of "
                  "Peking University" },
    { EM_EXCESS, "eXcess: 16/32/64-bit configurable embedded CPU" },
    { EM_DXP, "Icera Semiconductor Inc. Deep Execution Processor" },
    { EM_ALTERA_NIOS2, "Altera Nios II soft-core processor" },
    { EM_CRX, "National Semiconductor CRX" },
    { EM_XGATE, "Motorola XGATE embedded processor" },
    { EM_C166, "Infineon C16x/XC16x processor" },
    { EM_M16C, "Renesas M16C series microprocessors" },
    { EM_DSPIC30F, "Microchip Technology dsPIC30F Digital Signal Controller" },
    { EM_CE, "Freescale Communication Engine RISC core" },
    { EM_M32C, "Renesas M32C series microprocessors" },
    { EM_res121, "Reserved" },
    { EM_res122, "Reserved" },
    { EM_res123, "Reserved" },
    { EM_res124, "Reserved" },
    { EM_res125, "Reserved" },
    { EM_res126, "Reserved" },
    { EM_res127, "Reserved" },
    { EM_res128, "Reserved" },
    { EM_res129, "Reserved" },
    { EM_res130, "Reserved" },
    { EM_TSK3000, "Altium TSK3000 core" },
    { EM_RS08, "Freescale RS08 embedded processor" },
    { EM_res133, "Reserved" },
    { EM_ECOG2, "Cyan Technology eCOG2 microprocessor" },
    { EM_SCORE, "Sunplus Score" },
    { EM_SCORE7, "Sunplus S+core7 RISC processor" },
    { EM_DSP24, "New Japan Radio (NJR) 24-bit DSP Processor" },
    { EM_VIDEOCORE3, "Broadcom VideoCore III processor" },
    { EM_LATTICEMICO32, "RISC processor for Lattice FPGA architecture" },
    { EM_SE_C17, "Seiko Epson C17 family" },
    { EM_TI_C6000, "Texas Instruments TMS320C6000 DSP family" },
    { EM_TI_C2000, "Texas Instruments TMS320C2000 DSP family" },
    { EM_TI_C5500, "Texas Instruments TMS320C55x DSP family" },
    { EM_res143, "Reserved" },
    { EM_res144, "Reserved" },
    { EM_res145, "Reserved" },
    { EM_res146, "Reserved" },
    { EM_res147, "Reserved" },
    { EM_res148, "Reserved" },
    { EM_res149, "Reserved" },
    { EM_res150, "Reserved" },
    { EM_res151, "Reserved" },
    { EM_res152, "Reserved" },
    { EM_res153, "Reserved" },
    { EM_res154, "Reserved" },
    { EM_res155, "Reserved" },
    { EM_res156, "Reserved" },
    { EM_res157, "Reserved" },
    { EM_res158, "Reserved" },
    { EM_res159, "Reserved" },
    { EM_MMDSP_PLUS, "STMicroelectronics 64bit VLIW Data Signal Processor" },
    { EM_CYPRESS_M8C, "Cypress M8C microprocessor" },
    { EM_R32C, "Renesas R32C series microprocessors" },
    { EM_TRIMEDIA, "NXP Semiconductors TriMedia architecture family" },
    { EM_QDSP6, "QUALCOMM DSP6 Processor" },
    { EM_8051, "Intel 8051 and variants" },
    { EM_STXP7X, "STMicroelectronics STxP7x family" },
    { EM_NDS32,
      "Andes Technology compact code size embedded RISC processor family" },
    { EM_ECOG1, "Cyan Technology eCOG1X family" },
    { EM_ECOG1X, "Cyan Technology eCOG1X family" },
    { EM_MAXQ30, "Dallas Semiconductor MAXQ30 Core Micro-controllers" },
    { EM_XIMO16, "New Japan Radio (NJR) 16-bit DSP Processor" },
    { EM_MANIK, "M2000 Reconfigurable RISC Microprocessor" },
    { EM_CRAYNV2, "Cray Inc. NV2 vector architecture" },
    { EM_RX, "Renesas RX family" },
    { EM_METAG, "Imagination Technologies META processor architecture" },
    { EM_MCST_ELBRUS, "MCST Elbrus general purpose hardware architecture" },
    { EM_ECOG16, "Cyan Technology eCOG16 family" },
    { EM_CR16, "National Semiconductor CompactRISC 16-bit processor" },
    { EM_ETPU, "Freescale Extended Time Processing Unit" },
    { EM_SLE9X, "Infineon Technologies SLE9X core" },
    { EM_L1OM, "Intel L1OM" },
    { EM_INTEL181, "Reserved by Intel" },
    { EM_INTEL182, "Reserved by Intel" },
    { EM_AARCH64, "ARM AArch64" },
    { EM_res184, "Reserved by ARM" },
    { EM_AVR32, "Atmel Corporation 32-bit microprocessor family" },
    { EM_STM8, "STMicroeletronics STM8 8-bit microcontroller" },
    { EM_TILE64, "Tilera TILE64 multicore architecture family" },
    { EM_TILEPRO, "Tilera TILEPro multicore architecture family" },
    { EM_MICROBLAZE, "Xilinx MicroBlaze 32-bit RISC soft processor core" },
    { EM_CUDA, "NVIDIA CUDA architecture " },
    { EM_TILEGX, "Tilera TILE-Gx multicore architecture family" },
    { EM_CLOUDSHIELD, "CloudShield architecture family" },
    { EM_COREA_1ST, "KIPO-KAIST Core-A 1st generation processor family" },
    { EM_COREA_2ND, "KIPO-KAIST Core-A 2nd generation processor family" },
    { EM_ARC_COMPACT2, "Synopsys ARCompact V2" },
    { EM_OPEN8, "Open8 8-bit RISC soft processor core" },
    { EM_RL78, "Renesas RL78 family" },
    { EM_VIDEOCORE5, "Broadcom VideoCore V processor" },
    { EM_78KOR, "Renesas 78KOR family" },
    { EM_56800EX, "Freescale 56800EX Digital Signal Controller (DSC)" },
    { EM_BA1, "Beyond BA1 CPU architecture" },
    { EM_BA2, "Beyond BA2 CPU architecture" },
    { EM_XCORE, "XMOS xCORE processor family" },
    { EM_MCHP_PIC, "Microchip 8-bit PIC(r) family" },
    { EM_INTEL205, "Reserved by Intel" },
    { EM_INTEL206, "Reserved by Intel" },
    { EM_INTEL207, "Reserved by Intel" },
    { EM_INTEL208, "Reserved by Intel" },
    { EM_INTEL209, "Reserved by Intel" },
    { EM_KM32, "KM211 KM32 32-bit processor" },
    { EM_KMX32, "KM211 KMX32 32-bit processor" },
    { EM_KMX16, "KM211 KMX16 16-bit processor" },
    { EM_KMX8, "KM211 KMX8 8-bit processor" },
    { EM_KVARC, "KM211 KVARC processor" },
    { EM_CDP, "Paneve CDP architecture family" },
    { EM_COGE, "Cognitive Smart Memory Processor" },
    { EM_COOL, "iCelero CoolEngine" },
    { EM_NORC, "Nanoradio Optimized RISC" },
    { EM_CSR_KALIMBA, "CSR Kalimba architecture family" },
    { EM_Z80, "Zilog Z80" },
    { EM_VISIUM, "Controls and Data Services VISIUMcore processor" },
    { EM_FT32, "FTDI Chip FT32 high performance 32-bit RISC architecture" },
    { EM_MOXIE, "Moxie processor family" },
    { EM_AMDGPU, "AMD GPU architecture" },
    { EM_RISCV, "RISC-V" },
    { EM_LANAI, "Lanai processor" },
    { EM_CEVA, "CEVA Processor Architecture Family" },
    { EM_CEVA_X2, "CEVA X2 Processor Family" },
    { EM_BPF, "Linux BPF – in-kernel virtual machine" },
    { EM_GRAPHCORE_IPU, "Graphcore Intelligent Processing Unit" },
    { EM_IMG1, "Imagination Technologies" },
    { EM_NFP, "Netronome Flow Processor (P)" },
    { EM_CSKY, "C-SKY processor family" },
    { EM_ARC_COMPACT3_64, "Synopsys ARCv2.3 64-bit" },
    { EM_MCS6502, "MOS Technology MCS 6502 processor" },
    { EM_ARC_COMPACT3, "Synopsys ARCv2.3 32-bit" },
    { EM_KVX, "Kalray VLIW core of the MPPA processor family" },
    { EM_65816, "WDC 65816/65C816" },
    { EM_LOONGARCH, "Loongson Loongarch" },
    { EM_KF32, "ChipON KungFu32" },
    { EM_MT, "Morpho Techologies MT processor" },
    { EM_ALPHA, "Alpha" },
    { EM_WEBASSEMBLY, "Web Assembly" },
    { EM_DLX, "OpenDLX" },
    { EM_XSTORMY16, "Sanyo XStormy16 CPU core" },
    { EM_IQ2000, "Vitesse IQ2000" },
    { EM_M32C_OLD, "M32C_OLD" },
    { EM_NIOS32, "Altera Nios" },
    { EM_CYGNUS_MEP, "Toshiba MeP Media Engine" },
    { EM_ADAPTEVA_EPIPHANY, "Adapteva EPIPHANY" },
    { EM_CYGNUS_FRV, "Fujitsu FR-V" },
    { EM_S12Z, "Freescale S12Z" },
};

static const struct section_type_table_t
{
    const Elf64_Word key;
    const char*      str;
} section_type_table[] = {
    { SHT_NULL, "NULL" },
    { SHT_PROGBITS, "PROGBITS" },
    { SHT_SYMTAB, "SYMTAB" },
    { SHT_STRTAB, "STRTAB" },
    { SHT_RELA, "RELA" },
    { SHT_HASH, "HASH" },
    { SHT_DYNAMIC, "DYNAMIC" },
    { SHT_NOTE, "NOTE" },
    { SHT_NOBITS, "NOBITS" },
    { SHT_REL, "REL" },
    { SHT_SHLIB, "SHLIB" },
    { SHT_DYNSYM, "DYNSYM" },
    { SHT_INIT_ARRAY, "INIT_ARRAY" },
    { SHT_FINI_ARRAY, "FINI_ARRAY" },
    { SHT_PREINIT_ARRAY, "PREINIT_ARRAY" },
    { SHT_GROUP, "GROUP" },
    { SHT_SYMTAB_SHNDX, "SYMTAB_SHNDX" },
    { SHT_GNU_ATTRIBUTES, "GNU_ATTRIBUTES" },
    { SHT_GNU_HASH, "GNU_HASH" },
    { SHT_GNU_LIBLIST, "GNU_LIBLIST" },
    { SHT_CHECKSUM, "CHECKSUM" },
    { SHT_LOSUNW, "LOSUNW" },
    { SHT_SUNW_move, "SUNW_move" },
    { SHT_SUNW_COMDAT, "SUNW_COMDAT" },
    { SHT_SUNW_syminfo, "SUNW_syminfo" },
    { SHT_GNU_verdef, "GNU_verdef" },
    { SHT_GNU_verneed, "GNU_verneed" },
    { SHT_GNU_versym, "GNU_versym" },
    { SHT_ARM_EXIDX, "ARM_EXIDX" },
    { SHT_ARM_PREEMPTMAP, "ARM_PREEMPTMAP" },
    { SHT_ARM_ATTRIBUTES, "ARM_ATTRIBUTES" },
    { SHT_ARM_DEBUGOVERLAY, "ARM_DEBUGOVERLAY" },
    { SHT_ARM_OVERLAYSECTION, "ARM_OVERLAYSECTION" },

};

static const struct segment_type_table_t
{
    const Elf_Word key;
    const char*    str;
} segment_type_table[] = {
    { PT_NULL, "NULL" },
    { PT_LOAD, "LOAD" },
    { PT_DYNAMIC, "DYNAMIC" },
    { PT_INTERP, "INTERP" },
    { PT_NOTE, "NOTE" },
    { PT_SHLIB, "SHLIB" },
    { PT_PHDR, "PHDR" },
    { PT_TLS, "TLS" },
    { PT_GNU_EH_FRAME, "GNU_EH_FRAME" },
    { PT_GNU_STACK, "GNU_STACK" },
    { PT_GNU_RELRO, "GNU_RELRO" },
    { PT_GNU_PROPERTY, "GNU_PROPERTY" },
    { PT_GNU_MBIND_LO, "GNU_MBIND_LO" },
    { PT_GNU_MBIND_HI, "GNU_MBIND_HI" },
    { PT_PAX_FLAGS, "PAX_FLAGS" },
    { PT_OPENBSD_RANDOMIZE, "OPENBSD_RANDOMIZE" },
    { PT_OPENBSD_WXNEEDED, "OPENBSD_WXNEEDED " },
    { PT_OPENBSD_BOOTDATA, "OPENBSD_BOOTDATA " },
    { PT_SUNWBSS, "PT_SUNWBSS" },
    { PT_SUNWSTACK, "SUNWSTACK" },
};

static const struct segment_flag_table_t
{
    const Elf_Word key;
    const char*    str;
} segment_flag_table[] = {
    { 0, "   " }, { 1, "  E" }, { 2, " W " }, { 3, " WE" },
    { 4, "R  " }, { 5, "R E" }, { 6, "RW " }, { 7, "RWE" },
};

static const struct symbol_bind_t
{
    const Elf_Word key;
    const char*    str;
} symbol_bind_table[] = {
    { STB_LOCAL, "LOCAL" },   { STB_GLOBAL, "GLOBAL" },
    { STB_WEAK, "WEAK" },     { STB_LOOS, "LOOS" },
    { STB_HIOS, "HIOS" },     { STB_MULTIDEF, "MULTIDEF" },
    { STB_LOPROC, "LOPROC" }, { STB_HIPROC, "HIPROC" },
};

static const struct symbol_type_t
{
    const Elf_Word key;
    const char*    str;
} symbol_type_table[] = {
    { STT_NOTYPE, "NOTYPE" }, { STT_OBJECT, "OBJECT" },
    { STT_FUNC, "FUNC" },     { STT_SECTION, "SECTION" },
    { STT_FILE, "FILE" },     { STT_COMMON, "COMMON" },
    { STT_TLS, "TLS" },       { STT_LOOS, "LOOS" },
    { STT_HIOS, "HIOS" },     { STT_LOPROC, "LOPROC" },
    { STT_HIPROC, "HIPROC" },
};

static const struct dynamic_tag_t
{
    const Elf_Word key;
    const char*    str;
} dynamic_tag_table[] = {
    { DT_NULL, "NULL" },
    { DT_NEEDED, "NEEDED" },
    { DT_PLTRELSZ, "PLTRELSZ" },
    { DT_PLTGOT, "PLTGOT" },
    { DT_HASH, "HASH" },
    { DT_STRTAB, "STRTAB" },
    { DT_SYMTAB, "SYMTAB" },
    { DT_RELA, "RELA" },
    { DT_RELASZ, "RELASZ" },
    { DT_RELAENT, "RELAENT" },
    { DT_STRSZ, "STRSZ" },
    { DT_SYMENT, "SYMENT" },
    { DT_INIT, "INIT" },
    { DT_FINI, "FINI" },
    { DT_SONAME, "SONAME" },
    { DT_RPATH, "RPATH" },
    { DT_SYMBOLIC, "SYMBOLIC" },
    { DT_REL, "REL" },
    { DT_RELSZ, "RELSZ" },
    { DT_RELENT, "RELENT" },
    { DT_PLTREL, "PLTREL" },
    { DT_DEBUG, "DEBUG" },
    { DT_TEXTREL, "TEXTREL" },
    { DT_JMPREL, "JMPREL" },
    { DT_BIND_NOW, "BIND_NOW" },
    { DT_INIT_ARRAY, "INIT_ARRAY" },
    { DT_FINI_ARRAY, "FINI_ARRAY" },
    { DT_INIT_ARRAYSZ, "INIT_ARRAYSZ" },
    { DT_FINI_ARRAYSZ, "FINI_ARRAYSZ" },
    { DT_RUNPATH, "RUNPATH" },
    { DT_FLAGS, "FLAGS" },
    { DT_ENCODING, "ENCODING" },
    { DT_PREINIT_ARRAY, "PREINIT_ARRAY" },
    { DT_PREINIT_ARRAYSZ, "PREINIT_ARRAYSZ" },
    { DT_MAXPOSTAGS, "MAXPOSTAGS" },
    { DT_GNU_HASH, "GNU_HASH" },
    { DT_TLSDESC_PLT, "TLSDESC_PLT" },
    { DT_TLSDESC_GOT, "TLSDESC_GOT" },
    { DT_GNU_CONFLICT, "GNU_CONFLICT" },
    { DT_GNU_LIBLIST, "GNU_LIBLIST" },
    { DT_CONFIG, "CONFIG" },
    { DT_DEPAUDIT, "DEPAUDIT" },
    { DT_AUDIT, "AUDIT" },
    { DT_PLTPAD, "PLTPAD" },
    { DT_MOVETAB, "MOVETAB" },
    { DT_SYMINFO, "SYMINFO" },
    { DT_ADDRRNGHI, "ADDRRNGHI" },
    { DT_VERSYM, "VERSYM" },
    { DT_RELACOUNT, "RELACOUNT" },
    { DT_RELCOUNT, "RELCOUNT" },
    { DT_FLAGS_1, "FLAGS_1" },
    { DT_VERDEF, "VERDEF" },
    { DT_VERDEFNUM, "VERDEFNUM" },
    { DT_VERNEED, "VERNEED" },
    { DT_VERNEEDNUM, "VERNEEDNUM" },
};

// clang-format off
static const struct note_tag_t
{
    struct note_values_t
    {
        Elf64_Word  type;
        std::string type_str;
        std::string description;
    };
    std::string                name;
    std::vector<note_values_t> values;
} note_tag_table[] = {
    { "",
      { { NT_PRSTATUS,     "NT_PRSTATUS",     "prstatus struct" },
        { NT_FPREGSET,     "NT_FPREGSET",     "fpregset struct" },
        { NT_PRPSINFO,     "NT_PRPSINFO",     "prpsinfo struct" },
        { NT_TASKSTRUCT,   "NT_TASKSTRUCT",   "task struct" },
        { NT_AUXV,         "NT_AUXV",         "Elfxx_auxv_t" },
        { NT_PSTATUS,      "NT_PSTATUS",      "pstatus struct" },
        { NT_FPREGS,       "NT_FPREGS",       "fpregset struct" },
        { NT_PSINFO,       "NT_PSINFO",       "psinfo struct" },
        { NT_LWPSTATUS,    "NT_LWPSTATUS",    "lwpstatus_t struct" },
        { NT_LWPSINFO,     "NT_LWPSINFO",     "lwpsinfo_t struct" },
        { NT_WIN32PSTATUS, "NT_WIN32PSTATUS", "win32_pstatus struct" },
      } },
    { "LINUX",
      { { NT_PRXFPREG,             "NT_PRXFPREG", "Contains a user_xfpregs_struct;" },
        { NT_PPC_VMX,              "NT_PPC_VMX", "PowerPC Altivec/VMX registers" },
        { NT_PPC_VSX,              "NT_PPC_VSX", "PowerPC VSX registers" },
        { NT_PPC_TAR,              "NT_PPC_TAR", "PowerPC Target Address Register" },
        { NT_PPC_PPR,              "NT_PPC_PPR", "PowerPC Program Priority Register" },
        { NT_PPC_DSCR,             "NT_PPC_DSCR", "PowerPC Data Stream Control Register" },
        { NT_PPC_EBB,              "NT_PPC_EBB", "PowerPC Event Based Branch Registers" },
        { NT_PPC_PMU,              "NT_PPC_PMU", "PowerPC Performance Monitor Registers" },
        { NT_PPC_TM_CGPR,          "NT_PPC_TM_CGPR", "PowerPC TM checkpointed GPR Registers" },
        { NT_PPC_TM_CFPR,          "NT_PPC_TM_CFPR", "PowerPC TM checkpointed FPR Registers" },
        { NT_PPC_TM_CVMX,          "NT_PPC_TM_CVMX", "PowerPC TM checkpointed VMX Registers" },
        { NT_PPC_TM_CVSX,          "NT_PPC_TM_CVSX", "PowerPC TM checkpointed VSX Registers" },
        { NT_PPC_TM_SPR,           "NT_PPC_TM_SPR",   "PowerPC TM Special Purpose Registers" },
        { NT_PPC_TM_CTAR,          "NT_PPC_TM_CTAR", "PowerPC TM checkpointed TAR" },
        { NT_PPC_TM_CPPR,          "NT_PPC_TM_CPPR", "PowerPC TM checkpointed PPR" },
        { NT_PPC_TM_CDSCR,         "NT_PPC_TM_CDSCR", "PowerPC TM checkpointed Data SCR" },
        { NT_386_TLS,              "NT_386_TLS", "x86 TLS information" },
        { NT_386_IOPERM,           "NT_386_IOPERM", "x86 io permissions" },
        { NT_X86_XSTATE,           "NT_X86_XSTATE", "x86 XSAVE extended state" },
        { NT_X86_CET,              "NT_X86_CET", "x86 CET state" },
        { NT_S390_HIGH_GPRS,       "NT_S390_HIGH_GPRS", "S/390 upper halves of GPRs " },
        { NT_S390_TIMER,           "NT_S390_TIMER", "S390 timer" },
        { NT_S390_TODCMP,          "NT_S390_TODCMP", "S390 TOD clock comparator" },
        { NT_S390_TODPREG,         "NT_S390_TODPREG", "S390 TOD programmable register" },
        { NT_S390_CTRS,            "NT_S390_CTRS", "S390 control registers" },
        { NT_S390_PREFIX,          "NT_S390_PREFIX", "S390 prefix register" },
        { NT_S390_LAST_BREAK,      "NT_S390_LAST_BREAK", "S390 breaking event address" },
        { NT_S390_SYSTEM_CALL,     "NT_S390_SYSTEM_CALL", "S390 system call restart data" },
        { NT_S390_TDB,             "NT_S390_TDB", "S390 transaction diagnostic block" },
        { NT_S390_VXRS_LOW,        "NT_S390_VXRS_LOW", "S390 vector registers 0-15 upper half" },
        { NT_S390_VXRS_HIGH,       "NT_S390_VXRS_HIGH", "S390 vector registers 16-31" },
        { NT_S390_GS_CB,           "NT_S390_GS_CB", "s390 guarded storage registers" },
        { NT_S390_GS_BC,           "NT_S390_GS_BC", "s390 guarded storage broadcast control block" },
        { NT_ARM_VFP,              "NT_ARM_VFP", "ARM VFP registers" },
        { NT_ARM_TLS,              "NT_ARM_TLS", "AArch TLS registers" },
        { NT_ARM_HW_BREAK,         "NT_ARM_HW_BREAK", "AArch hardware breakpoint registers" },
        { NT_ARM_HW_WATCH,         "NT_ARM_HW_WATCH", "AArch hardware watchpoint registers" },
        { NT_ARM_SVE,              "NT_ARM_SVE", "AArch SVE registers. " },
        { NT_ARM_PAC_MASK,         "NT_ARM_PAC_MASK", "AArch pointer authentication code masks" },
        { NT_ARM_PACA_KEYS,        "NT_ARM_PACA_KEYS", "ARM pointer authentication address keys" },
        { NT_ARM_PACG_KEYS,        "NT_ARM_PACG_KEYS", "ARM pointer authentication generic keys" },
        { NT_ARM_TAGGED_ADDR_CTRL, "NT_ARM_TAGGED_ADDR_CTRL", "AArch64 tagged address control (prctl())" },
        { NT_ARM_PAC_ENABLED_KEYS, "NT_ARM_PAC_ENABLED_KEYS", "AArch64 pointer authentication enabled keys (prctl())" },
        { NT_ARC_V2,               "NT_ARC_V2", "ARC HS accumulator/extra registers. " },
        { NT_LARCH_CPUCFG,         "NT_LARCH_CPUCFG", "LoongArch CPU config registers" },
        { NT_LARCH_CSR,            "NT_LARCH_CSR", "LoongArch Control State Registers" },
        { NT_LARCH_LSX,            "NT_LARCH_LSX", "LoongArch SIMD eXtension registers" },
        { NT_LARCH_LASX,           "NT_LARCH_LASX", "LoongArch Advanced SIMD eXtension registers" },
        { NT_RISCV_CSR,            "NT_RISCV_CSR", "RISC-V Control and Status Registers" },
      } },
    { "CORE",
      { { NT_LARCH_LBT, "NT_LARCH_LBT", "LoongArch Binary Translation registers" }
      } },
    { "FreeBSD",
      { { NT_FREEBSD_THRMISC,            "NT_FREEBSD_THRMISC",            "Thread miscellaneous info." },
        { NT_FREEBSD_PROCSTAT_PROC,      "NT_FREEBSD_PROCSTAT_PROC",      "Procstat proc data." },
        { NT_FREEBSD_PROCSTAT_FILES,     "NT_FREEBSD_PROCSTAT_FILES",     "Procstat files data." },
        { NT_FREEBSD_PROCSTAT_VMMAP,     "NT_FREEBSD_PROCSTAT_VMMAP",     "Procstat vmmap data." },
        { NT_FREEBSD_PROCSTAT_GROUPS,    "NT_FREEBSD_PROCSTAT_GROUPS",    "Procstat groups data." },
        { NT_FREEBSD_PROCSTAT_UMASK,     "NT_FREEBSD_PROCSTAT_UMASK",     "Procstat umask data." },
        { NT_FREEBSD_PROCSTAT_RLIMIT,    "NT_FREEBSD_PROCSTAT_RLIMIT",    "Procstat rlimit data." },
        { NT_FREEBSD_PROCSTAT_OSREL,     "NT_FREEBSD_PROCSTAT_OSREL",     "Procstat osreldate data." },
        { NT_FREEBSD_PROCSTAT_PSSTRINGS, "NT_FREEBSD_PROCSTAT_PSSTRINGS", "Procstat ps_strings data." },
        { NT_FREEBSD_PROCSTAT_AUXV,      "NT_FREEBSD_PROCSTAT_AUXV",      "Procstat auxv data." },
        { NT_FREEBSD_PTLWPINFO,          "NT_FREEBSD_PTLWPINFO",          "Thread ptrace miscellaneous info." },
      } },
    { "NetBSD-CORE",
      { { NT_NETBSDCORE_PROCINFO,  "NT_NETBSDCORE_PROCINFO",  "Has a struct procinfo" },
        { NT_NETBSDCORE_AUXV,      "NT_NETBSDCORE_AUXV",      "Has auxv data" },
        { NT_NETBSDCORE_LWPSTATUS, "NT_NETBSDCORE_LWPSTATUS", "Has LWPSTATUS data" },
        { NT_NETBSDCORE_FIRSTMACH, "NT_NETBSDCORE_FIRSTMACH", "start of machdep note types" },
      } },
    { "OpenBSD",
      { { NT_OPENBSD_PROCINFO, "NT_OPENBSD_PROCINFO", "" },
        { NT_OPENBSD_AUXV,     "NT_OPENBSD_AUXV",     "" },
        { NT_OPENBSD_REGS,     "NT_OPENBSD_REGS",     "" },
        { NT_OPENBSD_FPREGS,   "NT_OPENBSD_FPREGS",   "" },
        { NT_OPENBSD_XFPREGS,  "NT_OPENBSD_XFPREGS",  "" },
        { NT_OPENBSD_WCOOKIE,  "NT_OPENBSD_WCOOKIE",  "" },
      } },
    { "SPU",
      { { NT_SPU, "NT_SPU", "" }
      } },
    { "GNU",
      {
        { NT_GNU_ABI_TAG,          "NT_GNU_ABI_TAG",         "GNU ABI tag" },
        { NT_GNU_HWCAP,            "NT_GNU_HWCAP",           "Used by ld.so and kernel vDSO" },
        { NT_GNU_BUILD_ID,         "NT_GNU_BUILD_ID",        "Build ID of the binary" },
        { NT_GNU_GOLD_VERSION,     "NT_GNU_GOLD_VERSION",    "Version of GNU gold used to link the binary" },
        { NT_GNU_PROPERTY_TYPE_0,  "NT_GNU_PROPERTY_TYPE_0", "Property type 0" },
        // { NT_GNU_PROPERTY_TYPE_1,  "NT_GNU_PROPERTY_TYPE_1",  "Property type 1" },
        // { NT_GNU_PROPERTY_TYPE_2,  "NT_GNU_PROPERTY_TYPE_2",  "Property type 2" },
        // { NT_GNU_PROPERTY_TYPE_3,  "NT_GNU_PROPERTY_TYPE_3",  "Property type 3" },
        // { NT_GNU_PROPERTY_TYPE_4,  "NT_GNU_PROPERTY_TYPE_4",  "Property type 4" },
        // { NT_GNU_PROPERTY_TYPE_5,  "NT_GNU_PROPERTY_TYPE_5",  "Property type 5" },
        // { NT_GNU_PROPERTY_TYPE_6,  "NT_GNU_PROPERTY_TYPE_6",  "Property type 6" },
        // { NT_GNU_PROPERTY_TYPE_7,  "NT_GNU_PROPERTY_TYPE_7",  "Property type 7" },
        // { NT_GNU_PROPERTY_TYPE_8,  "NT_GNU_PROPERTY_TYPE_8",  "Property type 8" },
        // { NT_GNU_PROPERTY_TYPE_9,  "NT_GNU_PROPERTY_TYPE_9",  "Property type 9" },
        // { NT_GNU_PROPERTY_TYPE_10, "NT_GNU_PROPERTY_TYPE_10", "Property type 10" },
        // { NT_GNU_PROPERTY_TYPE_11, "NT_GNU_PROPERTY_TYPE_11", "Property type 11" },
        // { NT_GNU_PROPERTY_TYPE_12, "NT_GNU_PROPERTY_TYPE_12", "Property type 12" },
        // { NT_GNU_PROPERTY_TYPE_13, "NT_GNU_PROPERTY_TYPE_13", "Property type 13" },
        // { NT_GNU_PROPERTY_TYPE_14, "NT_GNU_PROPERTY_TYPE_14", "Property type 14" },
      } },
    // { "SOLARIS",
    //   { { NT_SOLARIS_AUXV, "NT_SOLARIS_AUXV", "" }
    //   } },
    // { "AIX",
    //   { { NT_AIX_AUXV, "NT_AIX_AUXV", "" }
    //   } },
    // { "IRIX",
    //   { { NT_IRIX_FPREGS, "NT_IRIX_FPREGS", "" }
    //   } },
};
// clang-format on

static const ELFIO::Elf_Xword MAX_DATA_ENTRIES = 64;

//------------------------------------------------------------------------------
// Class representing the ELF dump functionality
class dump
{
#define DUMP_DEC_FORMAT( width ) \
    std::setw( width ) << std::setfill( ' ' ) << std::dec << std::right
#define DUMP_HEX0x_FORMAT( width ) \
    "0x" << std::setw( width ) << std::setfill( '0' ) << std::hex << std::right
#define DUMP_HEX_FORMAT( width ) \
    std::setw( width ) << std::setfill( '0' ) << std::hex << std::right
#define DUMP_STR_FORMAT( width ) \
    std::setw( width ) << std::setfill( ' ' ) << std::hex << std::left

  public:
    //------------------------------------------------------------------------------
    // Dumps the ELF header information
    static void header( std::ostream& out, const elfio& reader )
    {
        if ( !reader.get_header_size() ) {
            return;
        }
        out << "ELF Header" << std::endl
            << std::endl
            << "  Class:      " << str_class( reader.get_class() ) << std::endl
            << "  Encoding:   " << str_endian( reader.get_encoding() )
            << std::endl
            << "  ELFVersion: " << str_version( reader.get_elf_version() )
            << std::endl
            << "  OS/ABI:     " << str_os_abi( reader.get_os_abi() )
            << std::endl
            << "  ABI Version:" << (int)reader.get_abi_version() << std::endl
            << "  Type:       " << str_type( reader.get_type() ) << std::endl
            << "  Machine:    " << str_machine( reader.get_machine() )
            << std::endl
            << "  Version:    " << str_version( reader.get_version() )
            << std::endl
            << "  Entry:      " << "0x" << std::hex << reader.get_entry()
            << std::endl
            << "  Flags:      " << "0x" << std::hex << reader.get_flags()
            << std::endl
            << std::endl;
    }

    //------------------------------------------------------------------------------
    // Dumps the section headers information
    static void section_headers( std::ostream& out, const elfio& reader )
    {
        Elf_Half n = reader.sections.size();

        if ( n == 0 ) {
            return;
        }

        out << "Section Headers:" << std::endl;
        if ( reader.get_class() == ELFCLASS32 ) { // Output for 32-bit
            out << "[  Nr ] Type              Addr       Size       ES   Flg "
                   "Lk   Inf   Al   Name"
                << std::endl;
        }
        else { // Output for 64-bit
            out << "[  Nr ] Type              Addr               Size          "
                   "     Offset     Flg"
                << std::endl
                << "        ES     Lk     Inf    Al       Name" << std::endl;
        }

        for ( Elf_Half i = 0; i < n; ++i ) { // For all sections
            const section* sec = reader.sections[i];
            section_header( out, i, sec, reader.get_class() );
        }

        out << "Key to Flags: W (write), A (alloc), X (execute), " << std::endl;
        out << "              M (merge), S (strings), I (info)," << std::endl;
        out << "              L (link order), O (extra OS processing required),"
            << std::endl;
        out << "              G (group), T (TLS), C (compressed), E (exclude)"
            << std::endl;
    }

    //------------------------------------------------------------------------------
    // Dumps a single section header information
    static void section_header( std::ostream&  out,
                                Elf_Half       no,
                                const section* sec,
                                unsigned char  elf_class )
    {
        std::ios_base::fmtflags original_flags = out.flags();

        // clang-format off
        if ( elf_class == ELFCLASS32 ) { // Output for 32-bit
            out << "[" << DUMP_DEC_FORMAT( 5 ) << no << "] "
                << DUMP_STR_FORMAT( 17 ) << str_section_type( sec->get_type() )
                << " " << DUMP_HEX0x_FORMAT( 8 ) << sec->get_address() << " "
                << DUMP_HEX0x_FORMAT( 8 ) << sec->get_size() << " "
                << DUMP_HEX0x_FORMAT( 2 ) << sec->get_entry_size() << " "
                << DUMP_STR_FORMAT( 3 ) << section_flags( sec->get_flags() )
                << " " << DUMP_HEX0x_FORMAT( 2 ) << sec->get_link() << " "
                << DUMP_HEX0x_FORMAT( 3 ) << sec->get_info() << " "
                << DUMP_HEX0x_FORMAT( 2 ) << sec->get_addr_align() << " "
                << DUMP_STR_FORMAT( 17 ) << sec->get_name() << " " << std::endl;
        }
        else { // Output for 64-bit
            out << "[" << DUMP_DEC_FORMAT( 5 ) << no << "] "
                << DUMP_STR_FORMAT( 17 ) << str_section_type( sec->get_type() ) << " "
                << DUMP_HEX0x_FORMAT( 16 ) << sec->get_address()                  << " "
                << DUMP_HEX0x_FORMAT( 16 ) << sec->get_size()                     << " "
                << DUMP_HEX0x_FORMAT(  8 ) << sec->get_offset()                   << " "
                << DUMP_STR_FORMAT(  3) << section_flags( sec->get_flags() )
                << std::endl
                << DUMP_STR_FORMAT( 8 ) << " "
                << DUMP_HEX0x_FORMAT( 4 ) << sec->get_entry_size() << " "
                << DUMP_HEX0x_FORMAT( 4 ) << sec->get_link()       << " "
                << DUMP_HEX0x_FORMAT( 4 ) << sec->get_info()       << " "
                << DUMP_HEX0x_FORMAT( 4 ) << sec->get_addr_align() << "   "
                << DUMP_STR_FORMAT( 17 ) << sec->get_name()
                << std::endl;
        }
        // clang-format on

        out.flags( original_flags );

        return;
    }

    //------------------------------------------------------------------------------
    // Dumps the segment headers information
    static void segment_headers( std::ostream& out, const elfio& reader )
    {
        Elf_Half n = reader.segments.size();
        if ( n == 0 ) {
            return;
        }

        out << "Program Headers:" << std::endl;
        if ( reader.get_class() == ELFCLASS32 ) { // Output for 32-bit
            out << "[  Nr ] Type           VirtAddr   PhysAddr   FileSize   "
                   "Mem.Size   Flags    Align"
                << std::endl;
        }
        else { // Output for 64-bit
            out << "[  Nr ] Type           Offset             VirtAddr         "
                   "  PhysAddr"

                << std::endl
                << "                       FileSize           MemSize          "
                   "   Flags  Align"
                << std::endl;
        }

        for ( Elf_Half i = 0; i < n; ++i ) {
            const segment* seg = reader.segments[i];
            segment_header( out, i, seg, reader.get_class() );
        }

        out << std::endl;
        for ( Elf_Half i = 0; i < n; ++i ) {
            out << "[" << i << "]" << " ";
            const segment* seg = reader.segments[i];
            for ( Elf_Half j = 0; j < seg->get_sections_num(); j++ ) {
                const section* sec =
                    reader.sections[seg->get_section_index_at( j )];
                out << sec->get_name() << " ";
            }
            out << std::endl;
        }

        out << std::endl;
    }

    //------------------------------------------------------------------------------
    // Dumps a single segment header information
    static void segment_header( std::ostream&  out,
                                Elf_Half       no,
                                const segment* seg,
                                unsigned int   elf_class )
    {
        std::ios_base::fmtflags original_flags = out.flags();
        // clang-format off
        if ( elf_class == ELFCLASS32 ) { // Output for 32-bit
            out << "[" << DUMP_DEC_FORMAT( 5 ) << no << "] "
                << DUMP_STR_FORMAT( 14 ) << str_segment_type( seg->get_type() )
                << " " << DUMP_HEX0x_FORMAT( 8 ) << seg->get_virtual_address()
                << " " << DUMP_HEX0x_FORMAT( 8 ) << seg->get_physical_address()
                << " " << DUMP_HEX0x_FORMAT( 8 ) << seg->get_file_size() << " "
                << DUMP_HEX0x_FORMAT( 8 ) << seg->get_memory_size() << " "
                << DUMP_STR_FORMAT( 8 ) << str_segment_flag( seg->get_flags() )
                << " " << DUMP_HEX0x_FORMAT( 8 ) << seg->get_align() << " "
                << std::endl;
        }
        else { // Output for 64-bit
            out << "[" << DUMP_DEC_FORMAT( 5 ) << no << "] "
                << DUMP_STR_FORMAT( 14 ) << str_segment_type( seg->get_type() ) << " "
                << DUMP_HEX0x_FORMAT( 16 ) << seg->get_offset()           << " "
                << DUMP_HEX0x_FORMAT( 16 ) << seg->get_virtual_address()  << " "
                << DUMP_HEX0x_FORMAT( 16 ) << seg->get_physical_address()
                << std::endl
                << DUMP_STR_FORMAT( 23 ) << " "
                << DUMP_HEX0x_FORMAT( 16 ) << seg->get_file_size()         << " "
                << DUMP_HEX0x_FORMAT( 16 ) << seg->get_memory_size()       << "  "
                << DUMP_STR_FORMAT(  3 ) << str_segment_flag( seg->get_flags() ) << "    "
                << DUMP_HEX0x_FORMAT(  1 ) << seg->get_align()
                << std::endl;
        }
        // clang-format on

        out.flags( original_flags );
    }

    //------------------------------------------------------------------------------
    // Dumps the symbol tables information
    static void symbol_tables( std::ostream& out, const elfio& reader )
    {
        for ( const auto& sec : reader.sections ) { // For all sections
            if ( SHT_SYMTAB == sec->get_type() ||
                 SHT_DYNSYM == sec->get_type() ) {
                const_symbol_section_accessor symbols( reader, sec.get() );

                Elf_Xword sym_no = symbols.get_symbols_num();
                if ( sym_no == 0 ) {
                    continue;
                }

                out << "Symbol table (" << sec->get_name() << ")" << std::endl;
                if ( reader.get_class() == ELFCLASS32 ) { // Output for 32-bit
                    out << "[  Nr ] Value      Size       Type    Bind     "
                           " Sect Name"
                        << std::endl;
                }
                else { // Output for 64-bit
                    out << "[  Nr ] Value              Size               "
                           "Type    Bind      Sect"
                        << std::endl
                        << "        Name" << std::endl;
                }
                for ( Elf_Xword i = 0; i < sym_no; ++i ) {
                    std::string   name;
                    Elf64_Addr    value   = 0;
                    Elf_Xword     size    = 0;
                    unsigned char bind    = 0;
                    unsigned char type    = 0;
                    Elf_Half      section = 0;
                    unsigned char other   = 0;
                    symbols.get_symbol( i, name, value, size, bind, type,
                                        section, other );
                    symbol_table( out, i, name, value, size, bind, type,
                                  section, reader.get_class() );
                }

                out << std::endl;
            }
        }
    }

    //------------------------------------------------------------------------------
    // Dumps a single symbol table entry information
    static void symbol_table( std::ostream&      out,
                              Elf_Xword          no,
                              const std::string& name,
                              Elf64_Addr         value,
                              Elf_Xword          size,
                              unsigned char      bind,
                              unsigned char      type,
                              Elf_Half           section,
                              unsigned int       elf_class )
    {
        std::ios_base::fmtflags original_flags = out.flags();

        if ( elf_class == ELFCLASS32 ) { // Output for 32-bit
            out << "[" << DUMP_DEC_FORMAT( 5 ) << no << "] "
                << DUMP_HEX0x_FORMAT( 8 ) << value << " "
                << DUMP_HEX0x_FORMAT( 8 ) << size << " " << DUMP_STR_FORMAT( 7 )
                << str_symbol_type( type ) << " " << DUMP_STR_FORMAT( 8 )
                << str_symbol_bind( bind ) << " " << DUMP_DEC_FORMAT( 5 )
                << section << " " << DUMP_STR_FORMAT( 1 ) << name << " "
                << std::endl;
        }
        else { // Output for 64-bit
            out << "[" << DUMP_DEC_FORMAT( 5 ) << no << "] "
                << DUMP_HEX0x_FORMAT( 16 ) << value << " "
                << DUMP_HEX0x_FORMAT( 16 ) << size << " "
                << DUMP_STR_FORMAT( 7 ) << str_symbol_type( type ) << " "
                << DUMP_STR_FORMAT( 8 ) << str_symbol_bind( bind ) << " "
                << DUMP_DEC_FORMAT( 5 ) << section << " " << std::endl
                << "        " << DUMP_STR_FORMAT( 1 ) << name << " "
                << std::endl;
        }

        out.flags( original_flags );
    }

    //------------------------------------------------------------------------------
    // Dumps the notes information
    static void notes( std::ostream& out, const elfio& reader )
    {
        for ( const auto& sec : reader.sections ) { // For all sections
            if ( SHT_NOTE == sec->get_type() ) {    // Look at notes
                note_section_accessor notes( reader, sec.get() );
                Elf_Word              no_notes = notes.get_notes_num();

                if ( no_notes == 0 )
                    continue;

                out << "Note section (" << sec->get_name() << ")" << std::endl
                    << "    No Name         Data size  Description"
                    << std::endl;
                for ( Elf_Word j = 0; j < no_notes; ++j ) { // For all notes
                    Elf_Word    type;
                    std::string name;
                    char*       desc;
                    Elf_Word    descsz;

                    if ( notes.get_note( j, type, name, desc, descsz ) ) {
                        // 'name' usually contains \0 at the end. Remove it
                        name = name.c_str();
                        note( out, j, type, name, desc, descsz );
                        out << std::endl;
                    }
                }

                out << std::endl;
            }
        }

        Elf_Half no = reader.segments.size();
        for ( Elf_Half i = 0; i < no; ++i ) { // For all segments
            segment* seg = reader.segments[i];
            if ( PT_NOTE == seg->get_type() ) { // Look at notes
                note_segment_accessor notes( reader, seg );
                Elf_Word              no_notes = notes.get_notes_num();

                if ( no_notes == 0 )
                    continue;

                out << "Note segment (" << i << ")" << std::endl
                    << "    No Name         Data size  Description"
                    << std::endl;
                for ( Elf_Word j = 0; j < no_notes; ++j ) { // For all notes
                    Elf_Word    type;
                    std::string name;
                    char*       desc;
                    Elf_Word    descsz;

                    if ( notes.get_note( j, type, name, desc, descsz ) ) {
                        // 'name' usually contains \0 at the end. Remove it
                        name = name.c_str();
                        note( out, j, type, name, desc, descsz );
                        out << std::endl;
                    }
                }

                out << std::endl;
            }
        }
    }

    //------------------------------------------------------------------------------
    // Dumps a single note information
    static void note( std::ostream&      out,
                      int                no,
                      Elf_Word           type,
                      const std::string& name,
                      void*              desc,
                      Elf_Word           descsz )
    {
        out << "  [" << DUMP_DEC_FORMAT( 2 ) << no << "] ";

        const auto name_group = std::find_if(
            std::begin( note_tag_table ), std::end( note_tag_table ),
            [&name]( const note_tag_t& entry ) { return entry.name == name; } );

        std::vector<note_tag_t::note_values_t>::const_iterator type_value;
        if ( name_group != std::end( note_tag_table ) ) {
            type_value = std::find_if(
                name_group->values.begin(), name_group->values.end(),
                [&type]( const note_tag_t::note_values_t& e ) {
                    return e.type == type;
                } );
        }

        if ( name_group != std::end( note_tag_table ) &&
             type_value != name_group->values.end() ) {
            out << DUMP_STR_FORMAT( 12 ) << name_group->name << " "
                << DUMP_HEX0x_FORMAT( 8 ) << descsz << " "
                << type_value->type_str << " (" << type_value->description
                << ")";
        }
        else {
            out << DUMP_STR_FORMAT( 12 ) << name << " "
                << DUMP_HEX0x_FORMAT( 8 ) << descsz << " "
                << DUMP_HEX0x_FORMAT( 8 ) << type;
        }

        if ( descsz != 0 ) {
            for ( Elf_Word i = 0; i < descsz; ++i ) {
                if ( i % 16 == 0 ) {
                    out << std::endl << "         ";
                }
                out << DUMP_HEX_FORMAT( 2 )
                    << ( std::uint32_t )( (std::uint8_t*)( desc ) )[i];
            }
        }
    }

    //------------------------------------------------------------------------------
    // Dumps the module information
    static void modinfo( std::ostream& out, const elfio& reader )
    {
        for ( const auto& sec : reader.sections ) { // For all sections
            if ( ".modinfo" == sec->get_name() ) {  // Look for the section
                out << "Section .modinfo" << std::endl;

                const_modinfo_section_accessor modinfo( sec.get() );
                for ( Elf_Word i = 0; i < modinfo.get_attribute_num(); i++ ) {
                    std::string field;
                    std::string value;
                    if ( modinfo.get_attribute( i, field, value ) ) {
                        out << "  " << std::setw( 20 ) << field
                            << std::setw( 0 ) << " = " << value << std::endl;
                    }
                }

                out << std::endl;
                break;
            }
        }
    }

    //------------------------------------------------------------------------------
    // Dumps the dynamic tags information
    static void dynamic_tags( std::ostream& out,
                              const elfio&  reader,
                              bool          name_only = false )
    {
        for ( const auto& sec : reader.sections ) { // For all sections
            if ( SHT_DYNAMIC == sec->get_type() ) {
                dynamic_section_accessor dynamic( reader, sec.get() );

                Elf_Xword dyn_no = dynamic.get_entries_num();
                if ( dyn_no == 0 )
                    continue;
                if ( !name_only ) {
                    out << "Dynamic section (" << sec->get_name() << ")"
                        << std::endl;
                    out << "[  Nr ] Tag              Name/Value" << std::endl;
                }
                for ( Elf_Xword i = 0; i < dyn_no; ++i ) {
                    Elf_Xword   tag   = 0;
                    Elf_Xword   value = 0;
                    std::string str;
                    dynamic.get_entry( i, tag, value, str );
                    dynamic_tag( out, i, tag, value, str, reader.get_class(),
                                 name_only );
                    if ( DT_NULL == tag ) {
                        break;
                    }
                }
                if ( !name_only ) {
                    out << std::endl;
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    // Dumps a single dynamic tag information
    static void dynamic_tag( std::ostream&      out,
                             Elf_Xword          no,
                             Elf_Xword          tag,
                             Elf_Xword          value,
                             const std::string& str,
                             unsigned int /*elf_class*/,
                             bool name_only = false )
    {
        if ( name_only ) {
            if ( str.empty() || tag != DT_NEEDED ) {
                return;
            }
            out << str << std::endl;
            return;
        }
        out << "[" << DUMP_DEC_FORMAT( 5 ) << no << "] "
            << DUMP_STR_FORMAT( 16 ) << str_dynamic_tag( tag ) << " ";
        if ( str.empty() ) {
            out << DUMP_HEX0x_FORMAT( 16 ) << value << " ";
        }
        else {
            out << DUMP_STR_FORMAT( 32 ) << str << " ";
        }
        out << std::endl;
    }

    //------------------------------------------------------------------------------
    // Dumps the section data
    static void section_data( std::ostream& out, const section* sec )
    {
        std::ios_base::fmtflags original_flags = out.flags();

        out << sec->get_name() << std::endl;
        const char* pdata = sec->get_data();
        if ( pdata ) {
            ELFIO::Elf_Xword i;
            for ( i = 0; i < std::min( sec->get_size(), MAX_DATA_ENTRIES );
                  ++i ) {
                if ( i % 16 == 0 ) {
                    out << "[" << DUMP_HEX0x_FORMAT( 8 ) << i << "]";
                }

                out << " " << DUMP_HEX0x_FORMAT( 2 )
                    << ( pdata[i] & 0x000000FF );

                if ( i % 16 == 15 ) {
                    out << std::endl;
                }
            }
            if ( i % 16 != 0 ) {
                out << std::endl;
            }

            out.flags( original_flags );
        }

        return;
    }

    //------------------------------------------------------------------------------
    // Dumps all sections data
    static void section_datas( std::ostream& out, const elfio& reader )
    {
        Elf_Half n = reader.sections.size();

        if ( n == 0 ) {
            return;
        }

        out << "Section Data:" << std::endl;

        for ( Elf_Half i = 1; i < n; ++i ) { // For all sections
            const section* sec = reader.sections[i];
            if ( sec->get_type() == SHT_NOBITS ) {
                continue;
            }
            section_data( out, sec );
        }

        out << std::endl;
    }

    //------------------------------------------------------------------------------
    // Dumps the segment data
    static void
    segment_data( std::ostream& out, Elf_Half no, const segment* seg )
    {
        std::ios_base::fmtflags original_flags = out.flags();

        out << "Segment # " << no << std::endl;
        const char* pdata = seg->get_data();
        if ( pdata ) {
            ELFIO::Elf_Xword i;
            for ( i = 0; i < std::min( seg->get_file_size(), MAX_DATA_ENTRIES );
                  ++i ) {
                if ( i % 16 == 0 ) {
                    out << "[" << DUMP_HEX0x_FORMAT( 8 ) << i << "]";
                }

                out << " " << DUMP_HEX0x_FORMAT( 2 )
                    << ( pdata[i] & 0x000000FF );

                if ( i % 16 == 15 ) {
                    out << std::endl;
                }
            }
            if ( i % 16 != 0 ) {
                out << std::endl;
            }

            out.flags( original_flags );
        }

        return;
    }

    //------------------------------------------------------------------------------
    // Dumps all segments data
    static void segment_datas( std::ostream& out, const elfio& reader )
    {
        Elf_Half n = reader.segments.size();

        if ( n == 0 ) {
            return;
        }

        out << "Segment Data:" << std::endl;

        for ( Elf_Half i = 0; i < n; ++i ) { // For all sections
            const segment* seg = reader.segments[i];
            segment_data( out, i, seg );
        }

        out << std::endl;
    }

//------------------------------------------------------------------------------
#define STR_FUNC_TABLE( name )                                         \
    template <typename T> static std::string str_##name( const T key ) \
    {                                                                  \
        return format_assoc( name##_table, key );                      \
    }

    STR_FUNC_TABLE( class )
    STR_FUNC_TABLE( endian )
    STR_FUNC_TABLE( version )
    STR_FUNC_TABLE( os_abi )
    STR_FUNC_TABLE( type )
    STR_FUNC_TABLE( machine )
    STR_FUNC_TABLE( section_type )
    STR_FUNC_TABLE( segment_type )
    STR_FUNC_TABLE( segment_flag )
    STR_FUNC_TABLE( symbol_bind )
    STR_FUNC_TABLE( symbol_type )
    STR_FUNC_TABLE( dynamic_tag )

#undef STR_FUNC_TABLE

  private:
    //------------------------------------------------------------------------------
    template <typename T, typename K>
    std::string static find_value_in_table( const T& table, const K& key )
    {
        std::string res = "?";
        for ( unsigned int i = 0; i < sizeof( table ) / sizeof( table[0] );
              ++i ) {
            if ( table[i].key == key ) {
                res = table[i].str;
                break;
            }
        }

        return res;
    }

    //------------------------------------------------------------------------------
    template <typename T, typename K>
    static std::string format_assoc( const T& table, const K& key )
    {
        std::string str = find_value_in_table( table, key );
        if ( str == "?" ) {
            std::ostringstream oss;
            oss << str << " (0x" << std::hex << key << ")";
            str = oss.str();
        }

        return str;
    }

    //------------------------------------------------------------------------------
    template <typename T>
    static std::string format_assoc( const T& table, const char key )
    {
        return format_assoc( table, (const int)key );
    }

    //------------------------------------------------------------------------------
    static std::string section_flags( Elf_Xword flags )
    {
        std::string ret = "";
        if ( flags & SHF_WRITE ) {
            ret += "W";
        }
        if ( flags & SHF_ALLOC ) {
            ret += "A";
        }
        if ( flags & SHF_EXECINSTR ) {
            ret += "X";
        }
        if ( flags & SHF_MERGE ) {
            ret += "M";
        }
        if ( flags & SHF_STRINGS ) {
            ret += "S";
        }
        if ( flags & SHF_INFO_LINK ) {
            ret += "I";
        }
        if ( flags & SHF_LINK_ORDER ) {
            ret += "L";
        }
        if ( flags & SHF_OS_NONCONFORMING ) {
            ret += "O";
        }
        if ( flags & SHF_GROUP ) {
            ret += "G";
        }
        if ( flags & SHF_TLS ) {
            ret += "T";
        }
        if ( flags & SHF_COMPRESSED ) {
            ret += "C";
        }
        if ( flags & SHF_EXCLUDE ) {
            ret += "E";
        }
        if ( flags & SHF_GNU_MBIND ) {
            ret += "D";
        }

        return ret;
    }

#undef DUMP_DEC_FORMAT
#undef DUMP_HEX0x_FORMAT
#undef DUMP_STR_FORMAT
}; // class dump
} // namespace ELFIO

#endif // ELFIO_DUMP_HPP
