// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#define CACHELINE_SIZE 64

#if !defined(PAGE_SIZE)
  #define PAGE_SIZE 4096
#endif

#define PAGE_SIZE_2M (2*1024*1024)
#define PAGE_SIZE_4K (4*1024)

#include "platform.h"

/* define isa namespace and ISA bitvector */
#if defined (__AVX512VL__)
#  define isa avx512skx
#  define ISA AVX512SKX
#  define ISA_STR "AVX512SKX"
#elif defined (__AVX512F__)
#  define isa avx512knl
#  define ISA AVX512KNL
#  define ISA_STR "AVX512KNL"
#elif defined (__AVX2__)
#  define isa avx2
#  define ISA AVX2
#  define ISA_STR "AVX2"
#elif defined(__AVXI__)
#  define isa avxi
#  define ISA AVXI
#  define ISA_STR "AVXI"
#elif defined(__AVX__)
#  define isa avx
#  define ISA AVX
#  define ISA_STR "AVX"
#elif defined (__SSE4_2__)
#  define isa sse42
#  define ISA SSE42
#  define ISA_STR "SSE4.2"
#elif defined (__SSE4_1__)
#  define isa sse41
#  define ISA SSE41
#  define ISA_STR "SSE4.1"
//#elif defined(__SSSE3__) // we demote this to SSE2, MacOSX code compiles with SSSE3 by default with ICC
//#  define isa ssse3
//#  define ISA SSSE3
//#  define ISA_STR "SSSE3"
//#elif defined(__SSE3__) // we demote this to SSE2, MacOSX code compiles with SSE3 by default with clang
//#  define isa sse3
//#  define ISA SSE3
//#  define ISA_STR "SSE3"
#elif defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__)
#  define isa sse2
#  define ISA SSE2
#  define ISA_STR "SSE2"
#elif defined(__SSE__)
#  define isa sse
#  define ISA SSE
#  define ISA_STR "SSE"
#else 
#error Unknown ISA
#endif

namespace embree
{
  enum CPUModel {
    CPU_UNKNOWN,
    CPU_CORE1,
    CPU_CORE2,
    CPU_CORE_NEHALEM,
    CPU_CORE_SANDYBRIDGE,
    CPU_HASWELL,
    CPU_KNIGHTS_LANDING,
    CPU_SKYLAKE_SERVER
  };

  /*! get the full path to the running executable */
  std::string getExecutableFileName();

  /*! return platform name */
  std::string getPlatformName();

  /*! get the full name of the compiler */
  std::string getCompilerName();

  /*! return the name of the CPU */
  std::string getCPUVendor();

  /*! get microprocessor model */
  CPUModel getCPUModel(); 

  /*! converts CPU model into string */
  std::string stringOfCPUModel(CPUModel model);

  /*! CPU features */
  static const int CPU_FEATURE_SSE    = 1 << 0;
  static const int CPU_FEATURE_SSE2   = 1 << 1;
  static const int CPU_FEATURE_SSE3   = 1 << 2;
  static const int CPU_FEATURE_SSSE3  = 1 << 3;
  static const int CPU_FEATURE_SSE41  = 1 << 4;
  static const int CPU_FEATURE_SSE42  = 1 << 5; 
  static const int CPU_FEATURE_POPCNT = 1 << 6;
  static const int CPU_FEATURE_AVX    = 1 << 7;
  static const int CPU_FEATURE_F16C   = 1 << 8;
  static const int CPU_FEATURE_RDRAND = 1 << 9;
  static const int CPU_FEATURE_AVX2   = 1 << 10;
  static const int CPU_FEATURE_FMA3   = 1 << 11;
  static const int CPU_FEATURE_LZCNT  = 1 << 12;
  static const int CPU_FEATURE_BMI1   = 1 << 13;
  static const int CPU_FEATURE_BMI2   = 1 << 14;
  static const int CPU_FEATURE_AVX512F = 1 << 16;
  static const int CPU_FEATURE_AVX512DQ = 1 << 17;    
  static const int CPU_FEATURE_AVX512PF = 1 << 18;
  static const int CPU_FEATURE_AVX512ER = 1 << 19;
  static const int CPU_FEATURE_AVX512CD = 1 << 20;
  static const int CPU_FEATURE_AVX512BW = 1 << 21;
  static const int CPU_FEATURE_AVX512VL = 1 << 22;
  static const int CPU_FEATURE_AVX512IFMA = 1 << 23;
  static const int CPU_FEATURE_AVX512VBMI = 1 << 24;
  static const int CPU_FEATURE_XMM_ENABLED = 1 << 25;
  static const int CPU_FEATURE_YMM_ENABLED = 1 << 26;
  static const int CPU_FEATURE_ZMM_ENABLED = 1 << 27;
 
  static const int CPU_FEATURE_PSEUDO_HIFREQ256BIT = 1 << 30;
 
  /*! get CPU features */
  int getCPUFeatures();

  /*! convert CPU features into a string */
  std::string stringOfCPUFeatures(int features);

  /*! creates a string of all supported targets that are supported */
  std::string supportedTargetList (int isa);

  /*! ISAs */
  static const int SSE    = CPU_FEATURE_SSE | CPU_FEATURE_XMM_ENABLED; 
  static const int SSE2   = SSE | CPU_FEATURE_SSE2;
  static const int SSE3   = SSE2 | CPU_FEATURE_SSE3;
  static const int SSSE3  = SSE3 | CPU_FEATURE_SSSE3;
  static const int SSE41  = SSSE3 | CPU_FEATURE_SSE41;
  static const int SSE42  = SSE41 | CPU_FEATURE_SSE42 | CPU_FEATURE_POPCNT;
  static const int AVX    = SSE42 | CPU_FEATURE_AVX | CPU_FEATURE_YMM_ENABLED;
  static const int AVXI   = AVX | CPU_FEATURE_F16C | CPU_FEATURE_RDRAND;
  static const int AVX2   = AVXI | CPU_FEATURE_AVX2 | CPU_FEATURE_FMA3 | CPU_FEATURE_BMI1 | CPU_FEATURE_BMI2 | CPU_FEATURE_LZCNT;
  static const int AVX512KNL = AVX2 | CPU_FEATURE_AVX512F | CPU_FEATURE_AVX512PF | CPU_FEATURE_AVX512ER | CPU_FEATURE_AVX512CD | CPU_FEATURE_ZMM_ENABLED;
  static const int AVX512SKX = AVX2 | CPU_FEATURE_AVX512F | CPU_FEATURE_AVX512DQ | CPU_FEATURE_AVX512CD | CPU_FEATURE_AVX512BW | CPU_FEATURE_AVX512VL | CPU_FEATURE_ZMM_ENABLED;

  static const int AVX_FAST = AVX | CPU_FEATURE_PSEUDO_HIFREQ256BIT;
  static const int AVX2_FAST = AVX2 | CPU_FEATURE_PSEUDO_HIFREQ256BIT;

  /*! converts ISA bitvector into a string */
  std::string stringOfISA(int features);

  /*! return the number of logical threads of the system */
  unsigned int getNumberOfLogicalThreads();
  
  /*! returns the size of the terminal window in characters */
  int getTerminalWidth();

  /*! returns performance counter in seconds */
  double getSeconds();

  /*! sleeps the specified number of seconds */
  void sleepSeconds(double t);

  /*! returns virtual address space occupied by process */
  size_t getVirtualMemoryBytes();

  /*! returns resident memory required by process */
  size_t getResidentMemoryBytes();
}
